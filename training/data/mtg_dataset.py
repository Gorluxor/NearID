# data_mtg.py
"""MTG dataset wrapper + MultiDataset for joint EncodeID + MTG training."""
from __future__ import annotations

import ctypes
import logging
import multiprocessing as _mp
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from PIL import Image

from datasets import load_dataset
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as TF

from .nearid_dataset import (
    EncodeIDDataConfig,
    apply_mask_to_image_pil,
    binarize_mask,
    dilate_mask,
    hf_image_to_pil,
)

logger = logging.getLogger(__name__)

# Column names in abdo-eldesokey/mtg-dataset
_MTG_POS_COLS = ["image_1_original", "image_2_original"]
_MTG_NEG_COLS = ["image_1_inpainted", "image_2_inpainted"]
_MTG_MASK_COLS = ["image_1_object_mask", "image_2_object_mask"]
_MTG_PART_MASK_COLS = ["image_1_part_mask", "image_2_part_mask"]


def _calculate_ratio(
    obj_mask_pil: Optional[Image.Image],
    part_mask_pil: Optional[Image.Image],
    mtg_min: float = 0.5, # 1.0 is disabled
    mtg_factor: float = 1.0, # 1.0 is no scaling, <1.0 reduces the margin to be more conservative
) -> Optional[float]:
    """Compute ratio = part_mask_area / object_mask_area.
    
    Returns None when masks are missing or object area is zero.
    Higher ratio means more of the object was edited -> larger margin is achievable.
    """
    if obj_mask_pil is None or part_mask_pil is None:
        return None
    obj_arr = np.array(obj_mask_pil.convert("L")) > 0
    part_arr = np.array(part_mask_pil.convert("L")) > 0
    sum_obj = obj_arr.sum()
    if sum_obj == 0:
        return None
    ratio = float(part_arr.sum() / sum_obj)
    if mtg_factor != 1.0:
        ratio *= mtg_factor
    return max(min(mtg_min, ratio), 0.0)  # Cap minimum ratio to avoid extreme margins when part_mask is very small
# ---------------------------------------------------------------------------
# MTGTrainDataset
# ---------------------------------------------------------------------------

class MTGTrainDataset(Dataset):
    """
    Wraps the HuggingFace ``abdo-eldesokey/mtg-dataset`` and emits items in the
    **exact same dict format** as ``EncodeIDDataset``.

    Slot mapping (pad_to=3):
        pos0 (anchor) : image_1_original
        pos1           : image_2_original   (same identity, different view)
        pos2           : None               (padded)
        neg0           : image_1_inpainted  (soft negative — partial edit)
        neg1           : image_2_inpainted  (soft negative — partial edit)
        neg2           : None               (padded)

    Mask columns (for foreground extraction via mask_prob / mask_prob_apn):
        mask slot 1 : image_1_object_mask
        mask slot 2 : image_2_object_mask
        mask slot 3 : None

    Dynamic margin:
        Per-slot margin = ratio = part_mask_area / object_mask_area.
        Higher ratio -> more editing -> easier negative -> larger achievable margin.
        Falls back to ``mtg_margin`` when masks are missing.

    This is fully compatible with ``collate_encodeid`` and ``pack_for_losses_dist``.
    """

    PAD_TO = 3  # must match EncodeIDDataConfig.pad_to

    def __init__(
        self,
        processor=None,
        config: EncodeIDDataConfig = EncodeIDDataConfig(),
        split: str = "train",
        hf_path: str = "abdo-eldesokey/mtg-dataset",
        indices: Optional[Sequence[int]] = None,
        return_pil: bool = False,
        mtg_margin: float = 0.1,
        mtg_min: float = 1.0, # disabled by default
        mtg_factor: float = 1.0,
    ):
        super().__init__()
        assert isinstance(hf_path, str), "Expected hf_path to be a string (HF dataset identifier or local path)"
        self.hf_ds = load_dataset(hf_path, split=split)
        assert hasattr(self.hf_ds, "__getitem__"), "Expected a mapped HF Dataset, not IterableDataset"
        self.processor = processor
        self.cfg = config
        self.mtg_factor = mtg_factor
        self.return_pil = return_pil or (processor is None)
        self.mtg_margin = mtg_margin
        self.mtg_min = mtg_min
        # Worker-safe epoch (mirrors EncodeIDDataset)
        self._epoch_shared = _mp.Value(ctypes.c_long, 0)

        # Subset mapping
        if indices is None:
            self._indices = None
            self._len = len(self.hf_ds)  # type: ignore[arg-type]
        else:
            idxs = list(map(int, indices))
            n = len(self.hf_ds)  # type: ignore[arg-type]
            for x in idxs:
                if x < 0 or x >= n:
                    raise IndexError(f"Subset index out of range: {x} (MTG dataset len={n})")
            self._indices = idxs
            self._len = len(idxs)

    # -- epoch management (for deterministic RNG) --------------------------

    def set_epoch(self, epoch: int) -> None:
        self._epoch_shared.value = int(epoch)

    @property
    def _epoch(self) -> int:
        return int(self._epoch_shared.value)

    def __len__(self) -> int:
        return self._len

    def _map_index(self, i: int) -> int:
        return int(self._indices[i]) if self._indices is not None else int(i)

    # -- deterministic RNG (same scheme as EncodeIDDataset) ----------------

    def _rng(self, *keys: int) -> np.random.Generator:
        h = (self.cfg.base_seed * 1_000_003) ^ (self._epoch * 10_007)
        for k in keys:
            kk = int(k) & 0xFFFFFFFFFFFFFFFF
            h ^= (kk + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            h = (h * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        return np.random.default_rng(h & 0xFFFFFFFF)

    # -- anchor shuffle (identical logic to EncodeIDDataset) ---------------

    def _compute_perm(self, pos_mask: torch.Tensor, dataset_idx: int) -> torch.Tensor:
        pad_to = self.PAD_TO
        perm = torch.arange(pad_to, dtype=torch.long)
        if not self.cfg.shuffle_anchor:
            return perm
        valid = torch.nonzero(pos_mask, as_tuple=False).flatten().tolist()
        invalid = [i for i in range(pad_to) if i not in valid]
        if len(valid) <= 1:
            return perm
        rng = self._rng(dataset_idx, 424242)
        rng.shuffle(valid)
        return torch.tensor(valid + invalid, dtype=torch.long)

    # -- safe image loading ------------------------------------------------

    @staticmethod
    def _load_image(row: Dict[str, Any], col: str) -> Optional[Image.Image]:
        raw = row.get(col, None)
        img = hf_image_to_pil(raw)
        if img is None:
            return None
        try:
            img = img.convert("RGB")
            img.load()
            return img
        except Exception as exc:
            logger.warning("MTG PIL decode failure for col=%s: %s", col, exc)
            return None

    @staticmethod
    def _load_mask(row: Dict[str, Any], col: str) -> Optional[Image.Image]:
        raw = row.get(col, None)
        mk = hf_image_to_pil(raw)
        if mk is None:
            return None
        try:
            mk = mk.convert("L")
            mk.load()
            return binarize_mask(mk)
        except Exception as exc:
            logger.warning("MTG mask decode failure for col=%s: %s", col, exc)
            return None

    # -- masking helper (best-effort, never drops slot) --------------------

    def _try_apply_mask(
        self,
        img: Image.Image,
        mask: Optional[Image.Image],
        *,
        log_prefix: str,
        dataset_idx: int,
        slot: int,
    ) -> Image.Image:
        if mask is None:
            logger.warning(
                "%s mask missing: dataset_idx=%d slot=%d",
                log_prefix, dataset_idx, slot,
            )
            return img
        try:
            mk = binarize_mask(mask)
            if self.cfg.mask_dilate:
                mk = dilate_mask(mk, self.cfg.mask_dilate_k)
            return apply_mask_to_image_pil(
                img, mk, keep=self.cfg.mask_keep, fill=self.cfg.mask_fill
            )
        except Exception as exc:
            logger.warning(
                "%s mask apply failure: dataset_idx=%d slot=%d: %s",
                log_prefix, dataset_idx, slot, exc,
            )
            return img

    # -- processor wrapper -------------------------------------------------

    def _apply_augmentations(
        self,
        img: Image.Image,
        dataset_idx: int,
        slot: int,
        is_pos: bool,
    ) -> Image.Image:
        """Apply deterministic augmentations (flip / jitter / translate+scale).

        Same logic as EncodeIDDataset._apply_augmentations.
        """
        role_seed = 1 if is_pos else 2

        # 1. Random Horizontal Flip
        if self.cfg.flip_prob > 0.0:
            rng_flip = self._rng(dataset_idx, 111111, role_seed, slot)
            if float(rng_flip.random()) < self.cfg.flip_prob:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # 2. Random Color Jitter
        if self.cfg.color_jitter_prob > 0.0:
            rng_jitter = self._rng(dataset_idx, 222222, role_seed, slot)
            if float(rng_jitter.random()) < self.cfg.color_jitter_prob:
                jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
                img = jitter(img)

        # 3. Random Translation + Scale (disalignment)
        if self.cfg.translate_prob > 0.0:
            rng_trans = self._rng(dataset_idx, 333333, role_seed, slot)
            if float(rng_trans.random()) < self.cfg.translate_prob:
                max_dx = self.cfg.translate_fraction * img.size[0]
                max_dy = self.cfg.translate_fraction * img.size[1]
                dx = float(rng_trans.uniform(-max_dx, max_dx))
                dy = float(rng_trans.uniform(-max_dy, max_dy))
                scale = float(rng_trans.uniform(
                    self.cfg.scale_range_min, self.cfg.scale_range_max
                ))
                img = TF.affine(  
                    img, # type: ignore
                    angle=0.0,
                    translate=[int(dx), int(dy)],
                    scale=scale,
                    shear=[0.0],
                    fill=[0, 0, 0],
                )

        return img

    def _process_to_fixed_pixels(
        self,
        pos_imgs: List[Optional[Image.Image]],
        neg_imgs: List[Optional[Image.Image]],
    ) -> torch.Tensor:
        assert self.processor is not None
        all_imgs = pos_imgs + neg_imgs  # len = 6
        valid: List[Image.Image] = []
        valid_slots: List[int] = []
        for s, im in enumerate(all_imgs):
            if im is None:
                continue
            valid.append(im)
            valid_slots.append(s)
        if not valid:
            return torch.empty((0,))
        proc = self.processor(images=valid, return_tensors="pt")
        pv = proc.pixel_values  # (Nvalid, 3, H, W)
        out = torch.zeros((len(all_imgs),) + tuple(pv.shape[1:]), dtype=pv.dtype)
        for k, slot in enumerate(valid_slots):
            out[slot] = pv[k]
        return out

    # -- __getitem__ -------------------------------------------------------

    def __getitem__(self, i: int) -> Optional[Dict[str, Any]]:
        try:
            dataset_idx = self._map_index(i)
            row: Dict[str, Any] = self.hf_ds[dataset_idx]  # type: ignore[index]

            # Load raw images (no masking yet)
            pos_imgs_orig: List[Optional[Image.Image]] = [None] * self.PAD_TO
            neg_imgs_orig: List[Optional[Image.Image]] = [None] * self.PAD_TO
            masks: List[Optional[Image.Image]] = [None] * self.PAD_TO

            for slot, (pcol, ncol, mcol) in enumerate(
                zip(_MTG_POS_COLS, _MTG_NEG_COLS, _MTG_MASK_COLS)
            ):
                pos_imgs_orig[slot] = self._load_image(row, pcol)
                neg_imgs_orig[slot] = self._load_image(row, ncol)
                masks[slot] = self._load_mask(row, mcol)

            # Load part masks for dynamic margin computation
            part_masks: List[Optional[Image.Image]] = [None] * self.PAD_TO
            for slot, pmcol in enumerate(_MTG_PART_MASK_COLS):
                part_masks[slot] = self._load_mask(row, pmcol)

            # Build valid masks
            pos_mask_orig = torch.tensor(
                [im is not None for im in pos_imgs_orig], dtype=torch.bool
            )
            neg_mask_orig = torch.tensor(
                [im is not None for im in neg_imgs_orig], dtype=torch.bool
            )

            # We require at least the two positives (img1_orig, img2_orig)
            if int(pos_mask_orig.sum()) < self.cfg.min_pos_required:
                raise ValueError(
                    f"MTG sample {dataset_idx} has only {int(pos_mask_orig.sum())} valid "
                    f"positives, requires at least {self.cfg.min_pos_required}."
                )

            # -- Per-sample masking decision (probability-based) ---------------
            apply_masks = False
            if self.cfg.mask_prob > 0.0:
                if self.cfg.mask_prob >= 1.0:
                    apply_masks = True
                else:
                    mask_rng = self._rng(dataset_idx, 777777)
                    apply_masks = float(mask_rng.random()) < self.cfg.mask_prob

            apn = getattr(self.cfg, "mask_prob_apn", None)

            # -- Anchor permutation --------------------------------------------
            perm = self._compute_perm(pos_mask_orig, dataset_idx)

            # -- mask_prob_apn (within-sample per-role masking) ----------------
            pos_do_mask = torch.zeros(self.PAD_TO, dtype=torch.bool)
            neg_do_mask = torch.zeros(self.PAD_TO, dtype=torch.bool)

            if apply_masks and apn is not None:
                if not (isinstance(apn, (list, tuple)) and len(apn) == 3):
                    raise ValueError(f"mask_prob_apn must be None or [a,p,n], got: {apn!r}")
                a_prob, p_prob, n_prob = float(apn[0]), float(apn[1]), float(apn[2])

                for new_pos in range(self.PAD_TO):
                    orig_slot = int(perm[new_pos].item())
                    if not bool(pos_mask_orig[orig_slot]):
                        continue
                    prob_pos = a_prob if new_pos == 0 else p_prob
                    rng_pos = self._rng(dataset_idx, 888888, new_pos, 1)
                    if float(rng_pos.random()) < prob_pos:
                        pos_do_mask[orig_slot] = True
                    rng_neg = self._rng(dataset_idx, 999999, new_pos, 2)
                    if float(rng_neg.random()) < n_prob:
                        neg_do_mask[orig_slot] = True

            # -- Apply masking -------------------------------------------------
            # Legacy mode: apply_masks and apn is None => mask all slots
            legacy_mask_all = apply_masks and apn is None

            for slot in range(self.PAD_TO):
                mk = masks[slot]

                # Positives
                if pos_imgs_orig[slot] is not None and (legacy_mask_all or bool(pos_do_mask[slot])):
                    pos_imgs_orig[slot] = self._try_apply_mask(
                        pos_imgs_orig[slot],  # type: ignore[arg-type]
                        mk,
                        log_prefix="MTG-Pos", dataset_idx=dataset_idx, slot=slot,
                    )
                # Negatives
                if neg_imgs_orig[slot] is not None and (legacy_mask_all or bool(neg_do_mask[slot])):
                    neg_imgs_orig[slot] = self._try_apply_mask(
                        neg_imgs_orig[slot],  # type: ignore[arg-type]
                        mk,
                        log_prefix="MTG-Neg", dataset_idx=dataset_idx, slot=slot,
                    )

            # -- Apply permutation ---------------------------------------------
            pos_imgs = [pos_imgs_orig[int(o)] for o in perm.tolist()]
            pos_mask = pos_mask_orig[perm]

            neg_imgs = [neg_imgs_orig[int(o)] for o in perm.tolist()]
            neg_mask = neg_mask_orig[perm]

            # -- Dynamic margin from oracle ratio --------------------------
            margin_per_orig = torch.full((self.PAD_TO,), self.mtg_margin, dtype=torch.float32)
            for slot in range(self.PAD_TO):
                ratio = _calculate_ratio(masks[slot], part_masks[slot], mtg_min=self.mtg_min, mtg_factor=self.mtg_factor)
                if ratio is not None:
                    margin_per_orig[slot] = ratio
            margin = margin_per_orig[perm]

            # -- Apply Augmentations (Flip / Jitter / Translate) ---------------
            _any_aug = (
                self.cfg.flip_prob > 0.0
                or self.cfg.color_jitter_prob > 0.0
                or self.cfg.translate_prob > 0.0
            )
            if _any_aug:
                for slot in range(len(pos_imgs)):
                    if pos_imgs[slot] is not None:
                        pos_imgs[slot] = self._apply_augmentations(
                            pos_imgs[slot], dataset_idx, slot, is_pos=True  # type: ignore[arg-type]
                        )
                for slot in range(len(neg_imgs)):
                    if neg_imgs[slot] is not None:
                        neg_imgs[slot] = self._apply_augmentations(
                            neg_imgs[slot], dataset_idx, slot, is_pos=False  # type: ignore[arg-type]
                        )

            # -- Build output dict (same schema as EncodeIDDataset) ------------
            neg_source = torch.zeros(self.PAD_TO, dtype=torch.long)  # single source

            out: Dict[str, Any] = {
                "sample_id": row.get("id", f"mtg_{dataset_idx}"),
                "category": row.get("category", "mtg"),
                "dataset_idx": int(dataset_idx),
                "subset_idx": int(i),
                "margin": margin,
                "perm": perm,
                "pos_mask": pos_mask,
                "neg_mask": neg_mask,
                "n_pos": int(pos_mask.sum().item()),
                "n_neg": int(neg_mask.sum().item()),
                "neg_source_per_orig_slot": neg_source,
            }

            if self.return_pil:
                out["pos_images"] = pos_imgs
                out["neg_images"] = neg_imgs
                return out

            pv = self._process_to_fixed_pixels(pos_imgs, neg_imgs)
            if pv.numel() == 0:
                return None
            out["pixel_values"] = pv
            return out

        except Exception as exc:
            if self.cfg.fail_hard:
                raise
            logger.debug("MTGTrainDataset.__getitem__(%d) failed: %s", i, exc)
            return None


# ---------------------------------------------------------------------------
# MultiDataset — thin ConcatDataset wrapper with set_epoch propagation
# ---------------------------------------------------------------------------

class MultiDataset(ConcatDataset):
    """
    ``torch.utils.data.ConcatDataset`` that propagates ``set_epoch`` to every
    child dataset that supports it (duck-typed).  Drop-in replacement — works
    with ``collate_encodeid`` because every child emits the same dict schema.
    """

    def __init__(self, datasets: List[Dataset]):
        super().__init__(datasets)
        logger.info(
            "MultiDataset: %d child datasets, total %d samples (%s)",
            len(datasets),
            len(self),
            " + ".join(str(len(d)) for d in datasets),  # type: ignore[arg-type]
        )

    def set_epoch(self, epoch: int) -> None:
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)  # type: ignore[union-attr]
