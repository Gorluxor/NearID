"""Score DreamBench++ with the FROZEN SigLIP2 encoder, both readouts.

The frozen trunk can be read out two ways, and the paper's tables use both:

  mean_pool  last_hidden_state.mean(dim=1)  — the MAP head is skipped entirely.
             This is what `ratings/data_backbone_rating/` contains, i.e. the
             "SigLIP2 (backbone)" number in the DB++ column of Table 1.

  map_head   SigLIP2's own pretrained Multi-head Attention Pooling head, i.e.
             `get_image_features()`. This is what evaluation/sim_test.py uses for
             the NearID and MTG columns, and it is the like-for-like baseline
             against NearID, which retrains exactly this head.
             Stored in `ratings/data_backbone_maphead_rating/`.

Both are computed from a single forward pass per image, so the comparison is exact.
Regenerating them requires the DreamBench++ generated images — see README.md.

    python -m dreambench_plus.score_backbone --sample_root samples
    python -m dreambench_plus.score_backbone --sample_root samples --out_dir /tmp/check

Then compare against the shipped files, or re-run the correlations:

    python -m dreambench_plus.pearson

Note the shipped `data_backbone_rating` was computed in fp16 and is quantised to
0.25/0.5 steps; this script runs fp32, so expect agreement at r ~ 0.9996 rather
than bit-equality.
"""

import json
import os
from typing import List, Optional

import fire
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

from .score_nearid import infer_method, list_images

BACKBONE = "google/siglip2-so400m-patch14-384"


class FrozenSigLIP2:
    """Frozen SigLIP2 exposing both readouts from one forward pass."""

    def __init__(self, model_id: str = BACKBONE, device: Optional[str] = None,
                 batch_size: int = 32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
        self.vision = model.vision_model.to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
        self.batch_size = batch_size

    @torch.no_grad()
    def embed(self, paths: List[str], desc: str = ""):
        maps, means = [], []
        batches = range(0, len(paths), self.batch_size)
        for i in tqdm(batches, desc=desc, leave=False, disable=not desc):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + self.batch_size]]
            pv = self.processor(images=imgs, return_tensors="pt")["pixel_values"]
            pv = pv.to(self.device, torch.float32)
            out = self.vision(pixel_values=pv, return_dict=True)
            maps.append(out.pooler_output.cpu())               # native MAP head
            means.append(out.last_hidden_state.mean(1).cpu())   # MAP head skipped
        return torch.cat(maps), torch.cat(means)


def cosine_x100(a: torch.Tensor, b: torch.Tensor):
    a, b = F.normalize(a, p=2, dim=-1), F.normalize(b, p=2, dim=-1)
    return (100.0 * (a * b).sum(-1)).tolist()


def main(
    sample_root: str,
    out_dir: str = "dreambench_plus/ratings",
    model: str = BACKBONE,
    batch_size: int = 32,
):
    """Score every DreamBench++ run directory under ``sample_root``, both readouts."""
    sample_root = os.path.abspath(sample_root)
    run_dirs = [
        os.path.join(sample_root, n)
        for n in sorted(os.listdir(sample_root))
        if os.path.isdir(os.path.join(sample_root, n, "src_image"))
    ]
    if not run_dirs:
        raise ValueError(f"No run directories with src_image/ found under {sample_root}")

    targets = {
        "mean_pool": os.path.join(out_dir, "data_backbone_rating"),
        "map_head": os.path.join(out_dir, "data_backbone_maphead_rating"),
    }
    for d in targets.values():
        os.makedirs(d, exist_ok=True)

    print(f"Found {len(run_dirs)} run dirs; frozen backbone = {model}")
    enc = FrozenSigLIP2(model, batch_size=batch_size)

    for rd in run_dirs:
        name = os.path.basename(os.path.normpath(rd))
        try:
            method = infer_method(name)
        except ValueError as e:
            print(f"  skipping {name}: {e}")
            continue

        src_dir, tgt_dir = os.path.join(rd, "src_image"), os.path.join(rd, "tgt_image")
        src, tgt = list_images(src_dir), list_images(tgt_dir)
        if len(src) != len(tgt):
            raise ValueError(f"{name}: {len(src)} src != {len(tgt)} tgt")

        keys = []
        for f_src, f_tgt in zip(src, tgt):
            rel_s = os.path.relpath(f_src, src_dir).rsplit(".", 1)[0]
            rel_t = os.path.relpath(f_tgt, tgt_dir).rsplit(".", 1)[0]
            if rel_s != rel_t:
                raise ValueError(f"pairing mismatch:\n  {f_src}\n  {f_tgt}")
            keys.append(rel_s.replace(os.sep, "-"))

        s_map, s_mean = enc.embed(src, desc=f"{method} src")
        t_map, t_mean = enc.embed(tgt, desc=f"{method} tgt")

        for variant, (a, b) in {
            "map_head": (s_map, t_map),
            "mean_pool": (s_mean, t_mean),
        }.items():
            scores = dict(zip(keys, [round(v, 4) for v in cosine_x100(a, b)]))
            path = os.path.join(targets[variant], f"{method}.json")
            with open(path, "w") as f:
                json.dump(scores, f, indent=4)
            mean = sum(scores.values()) / len(scores)
            print(f"  {method:28s} {variant:10s} n={len(scores)} mean={mean:.4f} -> {path}")


if __name__ == "__main__":
    fire.Fire(main)
