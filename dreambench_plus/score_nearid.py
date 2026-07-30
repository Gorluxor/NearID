"""Score DreamBench++ generated images with NearID.

Regenerates ``ratings/data_nearid_rating/<method>.json`` from the DreamBench++
generated images. The published JSONs already contain these scores, so this is
only needed to score a different checkpoint or a new method.

Obtain the generated images from the official DreamBench++ release
(https://github.com/yuangpeng/dreambench_plus) — each run directory holds a
``src_image/`` (reference) and ``tgt_image/`` (generated) subdirectory with
matching relative paths.

    python -m dreambench_plus.score_nearid --sample_root /path/to/dreambench_plus/samples

Score convention matches the DreamBench++ CLIP-I / DINO-I scripts:
``100 * cosine(f_reference, f_generated)``, keyed by the relative image path
with ``/`` replaced by ``-``.
"""

import json
import os
from typing import Dict, List, Optional

import fire
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

MODEL = "Aleksandar/nearid-siglip2"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# DreamBench++ run-directory prefix -> method name used by pearson.py.
# The generated folders carry generation settings and spell out vit_huge /
# vit_giant, while the rating files use the short vit_h / vit_g form.
PREFIX_TO_METHOD = {
    "blip_diffusion": "blip_diffusion",
    "dreambooth_lora_sdxl": "dreambooth_lora_sdxl",
    "dreambooth_sd": "dreambooth_sd",
    "emu2": "emu2",
    "ip_adapter_plus_vit_huge_sdxl": "ip_adapter_plus_vit_h_sdxl",
    "ip_adapter_vit_giant_sdxl": "ip_adapter_vit_g_sdxl",
    "textual_inversion_sd": "textual_inversion_sd",
}


def infer_method(run_name: str) -> str:
    for prefix, method in PREFIX_TO_METHOD.items():
        if run_name.startswith(prefix):
            return method
    raise ValueError(
        f"Cannot infer method for run dir {run_name!r}. Known prefixes: {list(PREFIX_TO_METHOD)}"
    )


def list_images(root: str) -> List[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                out.append(os.path.join(dirpath, f))
    return sorted(set(out))


class NearIDScorer:
    def __init__(self, model_id: str = MODEL, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.dtype = next(self.model.parameters()).dtype

    @torch.no_grad()
    def cosine_x100(self, img_a: Image.Image, img_b: Image.Image) -> float:
        inputs = self.processor(images=[img_a, img_b], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        emb = self.model.get_image_features(pixel_values=pixel_values)  # L2-normalised
        return float(100.0 * (emb[0] * emb[1]).sum())


def score_run(scorer: NearIDScorer, run_dir: str, out_dir: str) -> str:
    src_dir = os.path.join(run_dir, "src_image")
    tgt_dir = os.path.join(run_dir, "tgt_image")
    if not (os.path.isdir(src_dir) and os.path.isdir(tgt_dir)):
        raise ValueError(f"{run_dir} must contain src_image/ and tgt_image/")

    src_files, tgt_files = list_images(src_dir), list_images(tgt_dir)
    if len(src_files) != len(tgt_files):
        raise ValueError(f"{run_dir}: {len(src_files)} src images != {len(tgt_files)} tgt images")

    method = infer_method(os.path.basename(os.path.normpath(run_dir)))
    scores: Dict[str, float] = {}

    for f_src, f_tgt in tqdm(list(zip(src_files, tgt_files)), desc=method):
        rel_src = os.path.relpath(f_src, src_dir).rsplit(".", 1)[0]
        rel_tgt = os.path.relpath(f_tgt, tgt_dir).rsplit(".", 1)[0]
        if rel_src != rel_tgt:
            raise ValueError(f"pairing mismatch:\n  src: {f_src}\n  tgt: {f_tgt}")
        scores[rel_src.replace(os.sep, "-")] = scorer.cosine_x100(
            Image.open(f_src).convert("RGB"), Image.open(f_tgt).convert("RGB")
        )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{method}.json")
    with open(out_path, "w") as f:
        json.dump(dict(sorted(scores.items())), f, indent=4)
    print(f"{method}: {len(scores)} pairs, mean={sum(scores.values()) / len(scores):.4f} -> {out_path}")
    return out_path


def main(
    sample_root: str,
    model: str = MODEL,
    out_dir: str = "dreambench_plus/ratings/data_nearid_rating",
):
    """Score every DreamBench++ run directory under ``sample_root``."""
    sample_root = os.path.abspath(sample_root)
    run_dirs = [
        os.path.join(sample_root, n)
        for n in sorted(os.listdir(sample_root))
        if os.path.isdir(os.path.join(sample_root, n, "src_image"))
    ]
    if not run_dirs:
        raise ValueError(f"No run directories with src_image/ found under {sample_root}")

    print(f"Found {len(run_dirs)} run dirs; scoring with {model}")
    scorer = NearIDScorer(model)
    for rd in run_dirs:
        score_run(scorer, rd, out_dir)


if __name__ == "__main__":
    fire.Fire(main)
