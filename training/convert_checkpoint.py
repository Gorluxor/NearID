#!/usr/bin/env python
"""Convert an EncodeID checkpoint (checkpoint-3300) to the NearID HF format.

Usage::

    python nearid/convert_checkpoint.py \
        --checkpoint runs/trains/runs/SigLIP2_MAPInfoNCEExt/CLIPID-...-260301-070712/checkpoint-3300 \
        --output nearid/weights \
        --verify
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def convert(checkpoint_dir: str, output_dir: str) -> None:
    """Remap EncodeID state dict keys to NearID format."""
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load source weights -----------------------------------------------
    src_path = os.path.join(checkpoint_dir, "model.safetensors")
    state_dict = load_file(src_path)
    print(f"Loaded {len(state_dict)} keys from {src_path}")

    # --- 2. Remap keys --------------------------------------------------------
    new_sd: dict[str, torch.Tensor] = {}
    dropped, remapped = 0, 0

    for key, tensor in state_dict.items():
        # Drop: redundant post-layernorm copy (only used for intermediate layers)
        if key.startswith("encoder_wrapper.post_ln."):
            dropped += 1
            continue

        # Drop: frozen backbone's original MAP head (replaced by trained head)
        if key.startswith("encoder_wrapper.model.vision_model.head."):
            dropped += 1
            continue

        # Remap: encoder backbone weights
        if key.startswith("encoder_wrapper.model.vision_model."):
            new_key = key.replace(
                "encoder_wrapper.model.vision_model.",
                "backbone.vision_model.",
            )
            new_sd[new_key] = tensor.to(torch.float16)
            remapped += 1
            continue

        # Remap: trained MAP head → backbone's head slot
        if key.startswith("head.map_pooler."):
            new_key = key.replace(
                "head.map_pooler.",
                "backbone.vision_model.head.",
            )
            new_sd[new_key] = tensor.to(torch.float16)
            remapped += 1
            continue

        print(f"  WARNING: unmapped key '{key}' — skipping")
        dropped += 1

    print(f"Remapped {remapped} keys, dropped {dropped} keys")
    print(f"Output state dict has {len(new_sd)} keys")

    # --- 3. Save weights ------------------------------------------------------
    out_safetensors = os.path.join(output_dir, "model.safetensors")
    save_file(new_sd, out_safetensors)
    size_mb = os.path.getsize(out_safetensors) / (1024 * 1024)
    print(f"Saved {out_safetensors} ({size_mb:.1f} MB)")

    # --- 4. Save config -------------------------------------------------------
    # Import modeling module first so register_for_auto_class() runs
    # and auto_map gets written into config.json
    from configuration_nearid import NearIDConfig
    import modeling_nearid  # noqa: F401  — triggers AutoClass registration

    config = NearIDConfig()
    config.architectures = ["NearIDModel"]
    config.auto_map = {
        "AutoConfig": "configuration_nearid.NearIDConfig",
        "AutoModel": "modeling_nearid.NearIDModel",
    }
    config.save_pretrained(output_dir)
    print(f"Saved config.json to {output_dir}")

    # --- 5. Save image processor ----------------------------------------------
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(
        "google/siglip2-so400m-patch14-384", use_fast=False
    )
    proc.save_pretrained(output_dir)
    print(f"Saved preprocessor_config.json to {output_dir}")

    # --- 6. Copy Python source files (needed for trust_remote_code) -----------
    src_dir = Path(__file__).parent
    for fname in ("configuration_nearid.py", "modeling_nearid.py"):
        src_file = src_dir / fname
        dst_file = Path(output_dir) / fname
        if src_file.resolve() != dst_file.resolve():
            shutil.copy2(src_file, dst_file)
            print(f"Copied {fname} → {output_dir}")

    print(f"\nConversion complete → {output_dir}")


def verify(output_dir: str, checkpoint_dir: str) -> None:
    """Load both models, run a forward pass, compare embeddings."""
    import sys

    # Add src/ to path for EncodeIDModel
    src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    sys.path.insert(0, src_dir)

    from config import EncodeIDConfig
    from models_dist import EncodeIDModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load original model ---
    print("\n--- Verification ---")
    print(f"Loading original EncodeIDModel from {checkpoint_dir}...")
    old_model = EncodeIDModel.from_pretrained(checkpoint_dir, trust_remote_code=True)
    old_model.to(device=device)
    old_model.eval()

    # --- Load new model ---
    print(f"Loading NearIDModel from {output_dir}...")

    # Import from the output dir (which has the copied .py files)
    sys.path.insert(0, output_dir)
    from modeling_nearid import NearIDModel

    new_model = NearIDModel.from_pretrained(output_dir, trust_remote_code=True)
    new_model.to(device=device)
    new_model.eval()

    # --- Forward pass ---
    dummy = torch.randn(2, 3, 384, 384, device=device)
    old_dtype = next(old_model.encoder_wrapper.parameters()).dtype
    new_dtype = next(new_model.backbone.parameters()).dtype

    with torch.inference_mode():
        # Old model interface
        old_out = old_model({"pixel_values_anchor": dummy.to(old_dtype)}, side="anchor")
        old_emb = torch.nn.functional.normalize(old_out, p=2, dim=-1).float()

        # New model interface
        new_out = new_model(pixel_values=dummy.to(new_dtype))
        new_emb = new_out.image_embeds.float()

    max_diff = (old_emb - new_emb).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(old_emb, new_emb, dim=-1).min().item()

    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Min cosine similarity:   {cos_sim:.6f}")

    if max_diff < 1e-2 and cos_sim > 0.999:
        print("PASSED: Outputs match within fp16 tolerance.")
    else:
        print("FAILED: Outputs diverge beyond acceptable tolerance!")
        raise ValueError(f"Verification failed: max_diff={max_diff}, cos_sim={cos_sim}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert EncodeID → NearID checkpoint")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to EncodeID checkpoint directory (e.g. .../checkpoint-3300)",
    )
    parser.add_argument(
        "--output",
        default="nearid/weights",
        help="Output directory for the converted model (default: nearid/weights)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification after conversion (compares old vs new model outputs)",
    )
    args = parser.parse_args()

    # Must be run from project root so relative imports work
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

    convert(args.checkpoint, args.output)

    if args.verify:
        verify(args.output, args.checkpoint)
