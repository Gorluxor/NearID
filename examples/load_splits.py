"""Load the official NearID splits and score a positive pair against a distractor.

Everything here comes from the HuggingFace Hub plus the JSON index files in
``splits/`` — no local dataset copies are needed.

    python examples/load_splits.py
"""

import json

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel

POSITIVES = "Aleksandar/NearID"
DISTRACTORS = "Aleksandar/NearID-Flux"
MODEL = "Aleksandar/nearid-siglip2"

# ---------------------------------------------------------------------------
# 1. Load the test split
# ---------------------------------------------------------------------------
# Split files list 0-based ROW POSITIONS into the unshuffled `train` split.
# They are not `id` values: `id` is sparse and inherited from the source pool.
with open("splits/test.json") as f:
    test_idx = json.load(f)

positives = load_dataset(POSITIVES, split="train")
distractors = load_dataset(DISTRACTORS, split="train")

assert len(positives) == len(distractors) == 19386, "datasets must stay index-aligned"

pos = positives.select(test_idx)
neg = distractors.select(test_idx)

print(f"test split: {len(pos)} samples")
assert pos["id"] == neg["id"], "positives and distractors must line up row-for-row"

# The dataset also carries a `split` column, so this is equivalent:
#   pos = positives.filter(lambda r: r["split"] == "test")
assert set(pos["split"]) == {"test"}

# ---------------------------------------------------------------------------
# 2. Embed one sample: two views of the identity + a near-identity distractor
# ---------------------------------------------------------------------------
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).eval()
processor = AutoImageProcessor.from_pretrained(MODEL)

sample_pos, sample_neg = pos[0], neg[0]
images = [
    sample_pos["images1"],  # anchor view
    sample_pos["images2"],  # same identity, different background
    sample_neg["nimg1"],    # different instance, SAME background as the anchor
]

with torch.no_grad():
    inputs = processor(images=images, return_tensors="pt")
    emb = model.get_image_features(**inputs)  # [3, 1152], L2-normalised

anchor, positive, distractor = emb
sim_pos = float(anchor @ positive)
sim_neg = float(anchor @ distractor)

print(f"\ncategory: {sample_pos['category_description']}")
print(f"  anchor vs positive view  (same identity, diff. background): {sim_pos:.4f}")
print(f"  anchor vs distractor     (diff. identity, same background): {sim_neg:.4f}")
print(f"  margin: {sim_pos - sim_neg:+.4f}  -> {'CORRECT' if sim_pos > sim_neg else 'FAILED'}")

# ---------------------------------------------------------------------------
# 3. Masks
# ---------------------------------------------------------------------------
# Foreground masks ship alongside each view as `masks1`/`masks2`/`masks3`,
# aligned with the matching `images*` column. They are what `--masks` uses
# during evaluation to score the object region only.
print(f"\nmask available for view 1: {sample_pos['masks1'] is not None}")
