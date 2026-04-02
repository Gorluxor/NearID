# Evaluating NearID

## Overview

NearID is evaluated on three complementary benchmarks:

1. **NearID-Bench:** Object-level near-identity discrimination (SSR, PA)
2. **MTG:** Part-level discrimination + oracle alignment (MO, MOpair, SSR, PA)
3. **DreamBench++:** Human-judgment alignment (MH correlation)

## Prerequisites

```bash
pip install -e ".[eval]"
```

## Step 1: Compute Similarities

Use `sim_test.py` to compute per-sample similarity scores:

```bash
# Using the released HuggingFace model
CUDA_VISIBLE_DEVICES=0 python -m evaluation.sim_test \
    --mode fullneg \
    --model "Aleksandar/nearid-siglip2" \
    --ds "Aleksandar/NearID" \
    --ds_neg "path/to/negative_source" \
    --split train \
    --findx "splits/test.json" \
    --output_folder "runs/evals/" \
    --batch_size 64

# Using a local training checkpoint
CUDA_VISIBLE_DEVICES=0 python -m evaluation.sim_test \
    --mode fullneg \
    --model "./runs/trains/checkpoint-3300" \
    --ds "Aleksandar/NearID" \
    --ds_neg "path/to/negative_source" \
    --split train \
    --findx "splits/test.json" \
    --output_folder "runs/evals/" \
    --batch_size 64
```

### Modes

| Mode | Description |
|------|-------------|
| `positives` | Intra-positive similarities only |
| `full` | Positives + cross (pos vs neg) |
| `fullneg` | Positives + cross + neg intra (full evaluation) |
| `mtg` | MTG dataset evaluation |

### Output

CSV files with columns: `id`, `category`, `sim_model`, `source_folder`, `mask_type`, and similarity scores for each pair type.

## Step 2: Aggregate Tables

Pool results into SSR/PA tables:

```bash
python -m evaluation.gen_tables \
    --root "./runs/evals/" \
    --split testall \
    --out_path "outputs/tables" \
    --overlap primary
```

This generates LaTeX tables and CSVs under `outputs/tables/`.

## Metrics

### NearID-Bench Metrics

| Metric | Code Name | Description |
|--------|-----------|-------------|
| **SSR** | `SSRm` | Success Separation Rate (AND-based: all pairwise margins must win) |
| **PA** | `PA` | Pairwise Accuracy (win rate across all margin trials) |

### MTG Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| **MO** | `\MO` | Metric-Oracle correlation |
| **MOpair** | `\MOpair` | Oracle correlation under same-background pair constraint |

### DreamBench++ Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| **MH** | `\MH` | Metric-Human Pearson correlation (Fisher-z averaged) |

## DreamBench++ Evaluation

For human-alignment evaluation on DreamBench++, follow the setup in the [DreamBench++ repository](https://github.com/peng-navi/dreambench-plus):

```bash
cd thirdparty/dreambench_plus
python dreambench_plus/pearson.py
```

## Expected Results (Table 1)

| Scoring Model | NearID SSR | NearID PA | MTG MO | MTG MOpair | MTG SSR | MTG PA | DB++ MH |
|---|---|---|---|---|---|---|---|
| CLIP ViT-L/14 | 10.31 | 20.92 | 0.239 | 0.484 | 0.0 | 0.0 | 0.493 |
| DINOv2 ViT-L/14 | 20.43 | 34.55 | 0.324 | 0.519 | 0.0 | 0.0 | 0.492 |
| SigLIP2 (backbone) | 30.74 | 48.81 | 0.180 | 0.366 | 0.0 | 0.0 | 0.516 |
| VSM | 32.13 | 46.70 | 0.394 | 0.445 | 7.0 | 24.5 | 0.190 |
| **NearID (Ours)** | **99.17** | **99.71** | **0.465** | 0.486 | **35.0** | **46.5** | **0.545** |
