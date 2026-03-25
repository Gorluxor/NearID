# dataset_hf.py
from __future__ import annotations
import ctypes
import io
import json
import logging
import multiprocessing as _mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
# import resampling from PIL
from PIL.Image import Resampling
from pathlib import Path
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as TF
from datasets import load_dataset, load_from_disk, IterableDataset, DatasetDict
from datasets import Dataset as HFDataset
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def load_indices_json(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "indices" in obj:
        obj = obj["indices"]
    if not isinstance(obj, list):
        raise ValueError(f"indices JSON must be a list[int] or {{'indices': [...]}}; got {type(obj)}")
    out: List[int] = []
    for x in obj:
        try:
            out.append(int(x))
        except Exception:
            raise ValueError(f"Non-integer index in JSON: {x!r}")
    if len(out) != len(set(out)):
        raise ValueError("Duplicate indices found in indices JSON.")
    return out


def hf_image_to_pil(x: Any) -> Optional[Image.Image]:
    """Convert common HF dataset image representations into PIL.Image."""
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, dict):
        b = x.get("bytes", None)
        p = x.get("path", None)
        if b is not None:
            return Image.open(io.BytesIO(b))
        if p is not None:
            return Image.open(p)
    if hasattr(x, "convert"):  # duck typing
        return x.convert("RGB") # type: ignore
    return None


# -----------------------------------------------------------------------------
# Optional mask utilities (minimal)
# -----------------------------------------------------------------------------

def binarize_mask(mask_pil: Image.Image) -> Image.Image:
    m = mask_pil.convert("L")
    arr = (np.array(m) > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def dilate_mask(mask_pil: Image.Image, k: int = 3) -> Image.Image:
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be odd and >= 1")
    m = binarize_mask(mask_pil)
    m = m.filter(ImageFilter.MaxFilter(k))
    return binarize_mask(m)


def apply_mask_to_image_pil(
    img_pil: Image.Image,
    mask_pil: Image.Image,
    keep: str = "foreground",
    fill: str = "black",
) -> Image.Image:
    img = img_pil.convert("RGB")
    m = mask_pil.convert("L")
    if m.size != img.size:
        m = m.resize(img.size, resample=Resampling.NEAREST)
    m = binarize_mask(m)
    m01 = (np.array(m) > 0)
    arr = np.array(img).astype(np.uint8)

    if fill == "black":
        fill_rgb = np.array([0, 0, 0], dtype=np.uint8)
    elif fill == "white":
        fill_rgb = np.array([255, 255, 255], dtype=np.uint8)
    else:
        raise ValueError("fill must be 'black' or 'white'")

    if keep == "foreground":
        keep01 = m01
    elif keep == "background":
        keep01 = ~m01
    else:
        raise ValueError("keep must be 'foreground' or 'background'")

    out = arr.copy()
    out[~keep01] = fill_rgb
    return Image.fromarray(out, mode="RGB")


def _load_hf_any(path_or_repo: str, split: str = "train") -> HFDataset:
    p = Path(path_or_repo)
    ds = load_from_disk(str(p)) if p.exists() else load_dataset(path_or_repo)
    # DatasetDict -> pick split
    if hasattr(ds, "keys"):
        ds = ds[split]
    assert isinstance(ds, HFDataset), f"Expected a HuggingFace Dataset or DatasetDict with split='{split}', got {type(ds)}"
    return ds

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class EncodeIDDataConfig:
    # Column prefixes
    pos_prefix: str = "images"    # images1..3
    neg_prefix: str = "nimg"      # nimg1..3
    mask_prefix: str = "masks"    # masks1..3

    pad_to: int = 3
    min_pos_required: int = 2

    # Masks — probability-based (0.0 = never apply, 1.0 = always apply)
    mask_prob: float = 0.0
    # Optional within-sample masking probabilities when a sample is selected by mask_prob.
    # If set: [anchor_prob, positive_prob, negative_prob]
    # Anchor is pos slot 0 AFTER permutation (shuffle_anchor).
    mask_prob_apn: Optional[List[float]] = None
    mask_keep: str = "foreground"
    mask_fill: str = "black"
    mask_dilate: bool = False
    mask_dilate_k: int = 3

    # Determinism
    base_seed: int = 0

    # Slot/anchor shuffle:
    # If True: permute only valid positive slots deterministically per (epoch, dataset_idx)
    # Anchor is always slot 0 AFTER permutation.
    shuffle_anchor: bool = False

    # Negative sampling:
    # If multiple neg datasets provided, pick neg dataset independently per *original* slot.
    per_slot_neg_dataset: bool = True

    # Safety:
    # fail_hard: raise on bad samples instead of returning None (catches data bugs early).
    # verify_ids: assert pos/neg rows share the same 'id' field (catches misaligned datasets).
    fail_hard: bool = False
    verify_ids: bool = False

    margin_map: Optional[Dict[str, float]] = None
    margin_field: str = "method_tag"

    # --- Augmentations ---
    # Horizontal flip: single most important augmentation for identity learning.
    # Faces/bodies are broadly symmetric; independent flips per slot force the model
    # to learn true identity features instead of overfitting to pose/gaze direction.
    # Default 0.0 here (safe for eval/tests); set to 0.5 in DataConfig for training.
    flip_prob: float = 0.0

    # Color jitter: prevents the model from using lighting/sensor-colour as a
    # shortcut to match real-world anchors to positives, closing the domain gap
    # with generated negatives.  Kept conservative (brightness/contrast/sat=0.2,
    # hue=0.05) so identity-critical colour cues (skin-tone, hair colour) are
    # preserved. Default 0.0 (disabled); 0.2–0.3 recommended for most runs.
    color_jitter_prob: float = 0.0

    # Translation & scale ("disalignment"): ViTs assign fixed positional
    # embeddings per patch; if subjects are always centred, the network can
    # memorise identity at specific patch positions.  Small random shifts
    # (±10 % of image, ±10 % zoom) break this prior and force true spatial
    # invariance.  Default 0.0 (disabled); 0.3–0.5 recommended once the
    # baseline converges.
    translate_prob: float = 0.0
    translate_fraction: float = 0.1   # max shift as a fraction of image size
    scale_range_min: float = 0.9      # min zoom (< 1 = zoom out)
    scale_range_max: float = 1.1      # max zoom (> 1 = zoom in)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class EncodeIDDataset(Dataset):
    """
    Returns fixed 3 positive slots + fixed 3 negative slots (padded) with optional masking.

    Masking behavior (proposed):
      - mask_prob: per-sample gate (same as before).
      - mask_prob_apn (optional): if provided as [a, p, n], then within a masked sample:
          * anchor (pos slot 0 AFTER permutation): masked with prob a
          * other positives: masked with prob p
          * negatives: masked with prob n (per positive slot / aligned by orig_slot)
        If mask_prob_apn is None: legacy behavior (if sample masked => mask all slots).

    Output (processor provided):
        pixel_values: (6,3,H,W) slots [pos0,pos1,pos2,neg0,neg1,neg2]
        pos_mask: (3,) bool after perm
        neg_mask: (3,) bool after perm
        perm: (3,) long where perm[new_pos] = orig_pos
    """

    def __init__(
        self,
        hf_pos,
        hf_negs: Optional[Sequence[Any]] = None,
        processor=None,
        config: EncodeIDDataConfig = EncodeIDDataConfig(),
        indices: Optional[Sequence[int]] = None,
        return_pil: bool = False,
        neg_names: Optional[Sequence[str]] = None,
    ):
        self.hf_pos = hf_pos
        self.hf_negs = list(hf_negs) if hf_negs is not None else []
        self.processor = processor
        self.cfg = config
        self.return_pil = return_pil or (processor is None)
        self.neg_names: List[str] = (
            list(neg_names) if neg_names is not None
            else [f"neg[{i}]" for i in range(len(self.hf_negs))]
        )
        self.warn_once = True
        self.neg_names = [a.lower().replace("encodeid-", "") for a in self.neg_names]

        logger.info(f"Initialized negative dataset with names:{self.neg_names=}")

        # Worker-safe epoch: shared int so set_epoch() in the main process
        self._epoch_shared = _mp.Value(ctypes.c_long, 0)

        # Subset mapping (applies to pos + all negs)
        if indices is None:
            self._indices = None
            self._len = len(self.hf_pos)
            logger.info("Using full dataset with %d samples", self._len)
        else:
            idxs = list(map(int, indices))
            n = len(self.hf_pos)
            for x in idxs:
                if x < 0 or x >= n:
                    raise IndexError(f"Subset index out of range: {x} (dataset len={n})")
            self._indices = idxs
            self._len = len(idxs)
            logger.info("Using subset of dataset with %d samples (from indices file)", self._len)

        # Alignment checks for neg datasets (index-aligned)
        for j, nds in enumerate(self.hf_negs):
            if isinstance(nds, str):
                _local_ds = _load_hf_any(nds)
                assert _local_ds is not None, f"Failed to load neg dataset from {nds}"
                self.hf_negs[j] = _local_ds

            if len(self.hf_negs[j]) != len(self.hf_pos):
                raise ValueError(
                    f"Negative dataset[{j}] length {len(self.hf_negs[j])} != positives length {len(self.hf_pos)}. "
                    "Index alignment is required."
                )

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch. Worker-safe: propagates to persistent DataLoader workers."""
        self._epoch_shared.value = int(epoch)

    @property
    def _epoch(self) -> int:
        return int(self._epoch_shared.value)

    def __len__(self) -> int:
        return self._len

    def _map_index(self, i: int) -> int:
        if self._indices is None:
            return int(i)
        return int(self._indices[i])

    def _rng(self, *keys: int) -> np.random.Generator:
        # deterministic RNG keyed by (base_seed, epoch, keys...)
        h = (self.cfg.base_seed * 1_000_003) ^ (self._epoch * 10_007)
        for k in keys:
            kk = int(k) & 0xFFFFFFFFFFFFFFFF
            h ^= (kk + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            h = (h * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        return np.random.default_rng(h & 0xFFFFFFFF)

    def _load_slot_list(
        self,
        row: Dict[str, Any],
        prefix: str,
        apply_masks: bool,
    ) -> Tuple[List[Optional[Image.Image]], torch.Tensor]:
        pad_to = self.cfg.pad_to
        imgs: List[Optional[Image.Image]] = [None] * pad_to
        mask = torch.zeros((pad_to,), dtype=torch.bool)

        for j in range(1, pad_to + 1):
            raw = row.get(f"{prefix}{j}", None)
            img = hf_image_to_pil(raw)
            if img is None:
                continue
            try:
                img = img.convert("RGB")
                img.load()
            except Exception as exc:
                logger.warning("PIL decode failure for %s%d: %s", prefix, j, exc)
                continue

            if apply_masks:
                mk = hf_image_to_pil(row.get(f"{self.cfg.mask_prefix}{j}", None))
                if mk is None:
                    logger.warning(
                        "Mask missing for %s%d while mask_prob=%.2f — skipping slot",
                        prefix,
                        j,
                        self.cfg.mask_prob,
                    )
                    continue
                try:
                    mk = mk.convert("L")
                    mk.load()
                except Exception as exc:
                    logger.warning("Mask PIL decode failure for %s%d: %s", self.cfg.mask_prefix, j, exc)
                    continue
                mk = binarize_mask(mk)
                if self.cfg.mask_dilate:
                    mk = dilate_mask(mk, self.cfg.mask_dilate_k)
                img = apply_mask_to_image_pil(img, mk, keep=self.cfg.mask_keep, fill=self.cfg.mask_fill)

            imgs[j - 1] = img
            mask[j - 1] = True

        return imgs, mask

    def _sample_neg_dataset_for_slot(self, orig_slot: int, dataset_idx: int) -> int:
        """Choose which neg dataset to use for a given original slot (aligned mode)."""
        if not self.hf_negs:
            return -1
        if len(self.hf_negs) == 1:
            return 0
        rng = self._rng(dataset_idx, 200003, orig_slot)
        return int(rng.integers(0, len(self.hf_negs)))

    def _compute_perm(self, pos_mask: torch.Tensor, dataset_idx: int) -> torch.Tensor:
        """
        perm[new_pos] = orig_pos
        If shuffle_anchor: permute only valid positive slots, keep invalid padded slots at end.
        """
        pad_to = self.cfg.pad_to
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

    def _apply_augmentations(
        self,
        img: Image.Image,
        dataset_idx: int,
        slot: int,
        is_pos: bool,
    ) -> Image.Image:
        """Apply deterministic augmentations (flip / jitter / translate+scale).

        Called *after* masking so flipped/shifted pixels include the baked mask.
        Each augmentation uses an independent deterministic RNG seeded by
        (dataset_idx, role, slot) to guarantee reproducibility across DDP
        workers and epoch restarts.
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
                
                img = TF.affine(  # type: ignore[assignment]
                    img, # type:ignore
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
        """Process valid images only; scatter into fixed (6,3,H,W)."""
        assert self.processor is not None

        all_imgs = pos_imgs + neg_imgs  # len=6
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

    def _try_apply_mask_from_posrow(
        self,
        img: Image.Image,
        pos_row: Dict[str, Any],
        mask_col_1based: int,
        *,
        log_prefix: str,
        dataset_idx: int,
    ) -> Image.Image:
        """
        Best-effort mask application; on failure returns the original image.
        (Used by within-sample masking mode to avoid dropping slots.)
        """
        mk = hf_image_to_pil(pos_row.get(f"{self.cfg.mask_prefix}{mask_col_1based}", None))
        if mk is None:
            logger.warning(
                "%s mask missing: dataset_idx=%d col=%d while mask_prob=%.2f",
                log_prefix,
                dataset_idx,
                mask_col_1based,
                self.cfg.mask_prob,
            )
            return img
        try:
            mk = mk.convert("L")
            mk.load()
            mk = binarize_mask(mk)
            if self.cfg.mask_dilate:
                mk = dilate_mask(mk, self.cfg.mask_dilate_k)
            return apply_mask_to_image_pil(img, mk, keep=self.cfg.mask_keep, fill=self.cfg.mask_fill)
        except Exception as exc:
            logger.warning(
                "%s mask apply failure: dataset_idx=%d col=%d: %s",
                log_prefix,
                dataset_idx,
                mask_col_1based,
                exc,
            )
            return img

    def __getitem__(self, i: int) -> Optional[Dict[str, Any]]:
        try:
            dataset_idx = self._map_index(i)
            pos_row = self.hf_pos[dataset_idx]

            # Decide per-sample whether to enable masking (probability-based; deterministic)
            apply_masks = False
            if self.cfg.mask_prob > 0.0:
                if self.cfg.mask_prob >= 1.0:
                    apply_masks = True
                else:
                    mask_rng = self._rng(dataset_idx, 777777)
                    apply_masks = float(mask_rng.random()) < self.cfg.mask_prob

            # Optional within-sample role probabilities [anchor, pos, neg]
            apn = getattr(self.cfg, "mask_prob_apn", None)
            if apn is not None:
                if not (isinstance(apn, (list, tuple)) and len(apn) == 3):
                    raise ValueError(f"mask_prob_apn must be None or a 3-list [a,p,n], got: {apn!r}")
                a_prob, p_prob, n_prob = (float(apn[0]), float(apn[1]), float(apn[2]))
                for v in (a_prob, p_prob, n_prob):
                    if v < 0.0 or v > 1.0:
                        raise ValueError(f"mask_prob_apn values must be in [0,1], got: {apn!r}")

            # Load positives:
            # - legacy: if apply_masks and apn is None => apply masks during load (old behavior)
            # - new: if apn is not None => load unmasked first; apply per-slot masks after perm
            pos_imgs_orig, pos_mask_orig = self._load_slot_list(
                pos_row,
                prefix=self.cfg.pos_prefix,
                apply_masks=(apply_masks and apn is None),
            )
            if int(pos_mask_orig.sum()) < self.cfg.min_pos_required:
                raise ValueError(
                    f"Sample {dataset_idx} has only {int(pos_mask_orig.sum())} valid positives, "
                    f"requires at least {self.cfg.min_pos_required}."
                )

            # Permute positives (anchor shuffle)
            perm = self._compute_perm(pos_mask_orig, dataset_idx)  # perm[new]=orig

            # Within-sample masking decisions live in ORIGINAL slot space (orig_slot 0..pad_to-1)
            pos_do_mask_orig = torch.zeros((self.cfg.pad_to,), dtype=torch.bool)
            neg_do_mask_orig = torch.zeros((self.cfg.pad_to,), dtype=torch.bool)

            if apply_masks and apn is not None:
                a_prob, p_prob, n_prob = float(apn[0]), float(apn[1]), float(apn[2])

                # Decide per permuted role: anchor == new_pos 0
                for new_pos in range(self.cfg.pad_to):
                    orig_slot = int(perm[new_pos].item())
                    if not bool(pos_mask_orig[orig_slot]):
                        continue

                    prob_pos = a_prob if new_pos == 0 else p_prob
                    rng_pos = self._rng(dataset_idx, 888888, new_pos, 1)
                    if float(rng_pos.random()) < prob_pos:
                        pos_do_mask_orig[orig_slot] = True

                    rng_neg = self._rng(dataset_idx, 999999, new_pos, 2)
                    if float(rng_neg.random()) < n_prob:
                        neg_do_mask_orig[orig_slot] = True

                # Apply masks to positives (best-effort; no dropping)
                for orig_slot in range(self.cfg.pad_to):
                    im = pos_imgs_orig[orig_slot]
                    if im is None or not bool(pos_do_mask_orig[orig_slot]):
                        continue
                    pos_imgs_orig[orig_slot] = self._try_apply_mask_from_posrow(
                        im,
                        pos_row,
                        orig_slot + 1,
                        log_prefix="Pos",
                        dataset_idx=dataset_idx,
                    )

            # Materialize permuted positives
            pos_imgs = [pos_imgs_orig[int(o)] for o in perm.tolist()]
            pos_mask = pos_mask_orig[perm]

            # Prepare negatives
            neg_imgs_orig: List[Optional[Image.Image]] = [None] * self.cfg.pad_to
            neg_mask_orig = torch.zeros((self.cfg.pad_to,), dtype=torch.bool)
            neg_source_per_orig = torch.full((self.cfg.pad_to,), -1, dtype=torch.long)
            _missing_neg_slots: List[Tuple[int, int]] = []

            if self.hf_negs:
                if self.cfg.per_slot_neg_dataset:
                    # ── ALIGNED MODE ──
                    for orig_slot in range(self.cfg.pad_to):
                        if not bool(pos_mask_orig[orig_slot]):
                            continue

                        ds_id = self._sample_neg_dataset_for_slot(orig_slot, dataset_idx)
                        neg_source_per_orig[orig_slot] = ds_id
                        neg_row = self.hf_negs[ds_id][dataset_idx]

                        # Optional ID alignment check
                        if self.cfg.verify_ids:
                            pos_id = pos_row.get("id", None)
                            neg_id = neg_row.get("id", None)
                            if pos_id is not None and neg_id is not None and pos_id != neg_id:
                                raise ValueError(
                                    f"ID mismatch at dataset_idx={dataset_idx} slot={orig_slot}: "
                                    f"pos id={pos_id!r}, neg[{ds_id}] id={neg_id!r}"
                                )

                        raw = neg_row.get(f"{self.cfg.neg_prefix}{orig_slot + 1}", None)
                        nim = hf_image_to_pil(raw)
                        if nim is None:
                            _missing_neg_slots.append((orig_slot, ds_id))
                            continue
                        try:
                            nim = nim.convert("RGB")
                            nim.load()
                        except Exception as exc:
                            logger.warning(
                                "Neg PIL decode failure: dataset_idx=%d slot=%d ds=%d: %s",
                                dataset_idx,
                                orig_slot,
                                ds_id,
                                exc,
                            )
                            continue

                        # Masking:
                        # - legacy: apply_masks and apn is None => mask all negs
                        # - new: apply_masks and apn provided => mask per neg_do_mask_orig[orig_slot]
                        do_mask_neg = apply_masks and (apn is None or bool(neg_do_mask_orig[orig_slot]))
                        if do_mask_neg:
                            nim = self._try_apply_mask_from_posrow(
                                nim,
                                pos_row,
                                orig_slot + 1,
                                log_prefix="Neg(aligned)",
                                dataset_idx=dataset_idx,
                            )

                        neg_imgs_orig[orig_slot] = nim
                        neg_mask_orig[orig_slot] = True

                    if _missing_neg_slots:
                        slots_str = ", ".join(f"slot={s} ds={d}({self.neg_names[d]})" for s, d in _missing_neg_slots)
                        logger.warning(
                            "Neg image missing (pos exists): dataset_idx=%d — %s (aligned mode, %d/%d slots)",
                            dataset_idx,
                            slots_str,
                            len(_missing_neg_slots),
                            int(pos_mask_orig.sum()),
                        )

                else:
                    # ── RANDOM MODE ──
                    available: List[Tuple[int, int, Image.Image]] = []
                    for ds_id_c in range(len(self.hf_negs)):
                        neg_row_c = self.hf_negs[ds_id_c][dataset_idx]

                        if self.cfg.verify_ids:
                            pos_id = pos_row.get("id", None)
                            neg_id = neg_row_c.get("id", None)
                            if pos_id is not None and neg_id is not None and pos_id != neg_id:
                                raise ValueError(
                                    f"ID mismatch at dataset_idx={dataset_idx}: "
                                    f"pos id={pos_id!r}, neg[{ds_id_c}] id={neg_id!r}"
                                )

                        for col in range(1, self.cfg.pad_to + 1):
                            raw = neg_row_c.get(f"{self.cfg.neg_prefix}{col}", None)
                            nim = hf_image_to_pil(raw)
                            if nim is None:
                                continue
                            try:
                                nim = nim.convert("RGB")
                                nim.load()
                            except Exception as exc:
                                logger.warning(
                                    "Neg PIL decode failure: dataset_idx=%d ds=%d col=%d (random mode): %s",
                                    dataset_idx,
                                    ds_id_c,
                                    col,
                                    exc,
                                )
                                continue
                            available.append((ds_id_c, col, nim))

                    if available:
                        rng = self._rng(dataset_idx, 555555)
                        order = list(range(len(available)))
                        rng.shuffle(order)
                        available = [available[k] for k in order]

                    avail_idx = 0
                    for orig_slot in range(self.cfg.pad_to):
                        if not bool(pos_mask_orig[orig_slot]):
                            continue
                        if avail_idx >= len(available):
                            break
                        ds_id_r, col_r, nim = available[avail_idx]
                        avail_idx += 1
                        neg_source_per_orig[orig_slot] = ds_id_r

                        do_mask_neg = apply_masks and (apn is None or bool(neg_do_mask_orig[orig_slot]))
                        if do_mask_neg:
                            # In random mode, mask must match the SOURCE column col_r
                            nim = self._try_apply_mask_from_posrow(
                                nim,
                                pos_row,
                                col_r,
                                log_prefix="Neg(random)",
                                dataset_idx=dataset_idx,
                            )

                        neg_imgs_orig[orig_slot] = nim
                        neg_mask_orig[orig_slot] = True

                # Permute negatives to match positive ordering
                neg_imgs = [neg_imgs_orig[int(o)] for o in perm.tolist()]
                neg_mask = neg_mask_orig[perm]
            else:
                neg_imgs = [None, None, None]
                neg_mask = torch.zeros_like(pos_mask)

            # -------------------------------------------------------------------------
            # Dynamic Per-Slot Margin Logic
            # -------------------------------------------------------------------------
            default_margin = 0.1
            if self.cfg.margin_map is not None:
                default_margin = self.cfg.margin_map.get("default", 0.1)

            margin_per_orig = torch.full((self.cfg.pad_to,), default_margin, dtype=torch.float32)

            if self.cfg.margin_map is not None and self.hf_negs:
                for j in range(self.cfg.pad_to):
                    ds_id = int(neg_source_per_orig[j])
                    if ds_id >= 0:
                        neg_tag = self.neg_names[ds_id]
                        margin_per_orig[j] = self.cfg.margin_map.get(neg_tag, default_margin)
            elif self.warn_once:
                self.warn_once = False
                logger.info("Warning self.cfg margin_map is None")

            margin_permuted = margin_per_orig[perm]

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
                        neg_imgs[slot] = self._apply_augmentations( # type: ignore
                            neg_imgs[slot], dataset_idx, slot, is_pos=False  # type: ignore[arg-type]
                        ) 

            out: Dict[str, Any] = {
                "sample_id": pos_row.get("id", dataset_idx),
                "category": pos_row.get("category", ""),
                "dataset_idx": int(dataset_idx),
                "subset_idx": int(i),
                "margin": margin_permuted,  # (pad_to,)
                "perm": perm,
                "pos_mask": pos_mask,
                "neg_mask": neg_mask,
                "n_pos": int(pos_mask.sum().item()),
                "n_neg": int(neg_mask.sum().item()),
                "neg_source_per_orig_slot": neg_source_per_orig,
            }

            if self.return_pil:
                out["pos_images"] = pos_imgs
                out["neg_images"] = neg_imgs
                return out

            pv = self._process_to_fixed_pixels(pos_imgs, neg_imgs)  # type: ignore
            if pv.numel() == 0:
                return None
            out["pixel_values"] = pv
            return out

        except Exception as exc:
            if self.cfg.fail_hard:
                raise
            logger.debug("EncodeIDDataset.__getitem__(%d) failed: %s", i, exc)
            return None




# -----------------------------------------------------------------------------
# Collate
# -----------------------------------------------------------------------------

def collate_encodeid(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    assert batch is not None, "All samples in the batch are None after filtering. This may indicate a data issue or that mask_prob is too high."
    assert all(b is not None for b in batch), "Batch contains None samples after filtering"
    out = {
        "sample_id": [b["sample_id"] for b in batch if b],
        "category": [b["category"] for b in batch if b],
        "dataset_idx": torch.tensor([b["dataset_idx"] for b in batch], dtype=torch.long), # type: ignore
        "subset_idx": torch.tensor([b["subset_idx"] for b in batch], dtype=torch.long), # type: ignore
        "perm": torch.stack([b["perm"] for b in batch], dim=0),  # (B,3) # type: ignore
        "pos_mask": torch.stack([b["pos_mask"] for b in batch], dim=0),  # (B,3) # type: ignore
        "neg_mask": torch.stack([b["neg_mask"] for b in batch], dim=0),  # (B,3) # type: ignore
        "n_pos": torch.tensor([b["n_pos"] for b in batch], dtype=torch.long), # type: ignore
        "n_neg": torch.tensor([b["n_neg"] for b in batch], dtype=torch.long), # type: ignore
        "neg_source_per_orig_slot": torch.stack([b["neg_source_per_orig_slot"] for b in batch], dim=0),  # (B,3) # type: ignore
        "margin": torch.stack([b["margin"] for b in batch], dim=0)  # (B,3) per-slot margin # type: ignore
    } # type: ignore
    assert batch is not None and isinstance(batch[0], dict), "Expected batch to be a list of dicts after filtering None samples"
    if "pixel_values" in batch[0]: 
        out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch], dim=0)  # (B,6,3,H,W) # type: ignore
    else:
        out["pos_images"] = [b["pos_images"] for b in batch] # type: ignore
        out["neg_images"] = [b["neg_images"] for b in batch] # type: ignore
    # print(f'[DEBUG] Margin: {out["margin"]}')
    return out


# -----------------------------------------------------------------------------
# Convenience: pack embeddings for your StandardizedLoss interface
# -----------------------------------------------------------------------------

def pack_for_losses_dist(
    emb: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    emb: (B,6,D) in slot order [pos0,pos1,pos2,neg0,neg1,neg2]
    pos_mask, neg_mask: (B,3)

        Returns:
            anchor:   (B,D)
            positive: (B,2,D)  from pos1,pos2
            negative: (B,3,D)  from neg0..2
            pos_mask: (B,2)    mask for pos1,pos2
            neg_mask: (B,3)    mask for neg0..2
    """
    if emb.dim() != 3 or emb.shape[1] != 6:
        raise ValueError(f"Expected emb (B,6,D), got {tuple(emb.shape)}")

    anchor = emb[:, 0, :]
    positive = emb[:, 1:3, :]
    negative = emb[:, 3:6, :]
    pos_mask_2 = pos_mask[:, 1:3].contiguous()
    neg_mask_3 = neg_mask.contiguous()
    return anchor, positive, negative, pos_mask_2, neg_mask_3
