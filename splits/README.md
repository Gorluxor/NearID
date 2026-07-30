# NearID Train / Val / Test Splits

Official splits used for every NearID result in the paper. Each file is a JSON list of
**0-based row positions** into the `train` split of
[`Aleksandar/NearID`](https://huggingface.co/datasets/Aleksandar/NearID) (19,386 rows).

| File | Samples | Purpose |
|---|---|---|
| [`train.json`](train.json) | 18,786 | Training set |
| [`val.json`](val.json) | 100 | Online validation during training (`--data.val_indices_path`) |
| [`test.json`](test.json) | 500 | Held-out evaluation, reported as NearID-Bench (`--findx`) |

The three splits are pairwise disjoint and their union is the full 19,386 rows.

## Important: positions, not `id` values

The `id` column of the dataset is **not** equal to the row position (ids are sparse,
inherited from the SynCD source pool, and range up to 50,145). Split entries index rows
by position, so always select with `Dataset.select(...)` on the unshuffled dataset:

```python
import json
from datasets import load_dataset

ds = load_dataset("Aleksandar/NearID", split="train")   # 19,386 rows, do not shuffle
test = ds.select(json.load(open("splits/test.json")))    # 500 rows
```

## Distractor datasets are index-aligned

All near-identity distractor datasets (`Aleksandar/NearID-Flux`, `-SDXL`, `-Qwen`,
`-PowerPaint`, ...) have the same 19,386 rows in the same order as the positives, with
matching `id` values. The same split indices therefore apply unchanged to any distractor
source:

```python
negatives = load_dataset("Aleksandar/NearID-Flux", split="train")
test_neg = negatives.select(json.load(open("splits/test.json")))
assert test["id"] == test_neg["id"]
```

## Usage

Training:

```bash
accelerate launch -m training.train \
    --data.train_indices_path "splits/train.json" \
    --data.val_indices_path   "splits/val.json" \
    --data.test_indices_path  "splits/test.json" \
    ...
```

Evaluation:

```bash
python -m evaluation.sim_test --findx "splits/test.json" --split train ...
```

See [`examples/load_splits.py`](../examples/load_splits.py) for a runnable end-to-end
example that loads a split and scores a positive pair against a near-identity distractor.
