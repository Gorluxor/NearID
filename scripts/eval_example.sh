#!/usr/bin/env bash
# NearID evaluation example — reproduces the NearID SSR / PA columns of Table 1.
#
# Everything is pulled from the HuggingFace Hub; no local dataset copies needed.
#
# Prerequisites:
#   conda env create -f environment.yaml
#   conda activate nearid
#   pip install -e ".[eval]"
set -euo pipefail

MODEL="Aleksandar/nearid-siglip2"   # or a local checkpoint dir, e.g. ./runs/trains/checkpoint-3300
GPU="${CUDA_VISIBLE_DEVICES:-0}"

# The seven distractor sources pooled for Table 1. FluxC / FluxC_1024 are held
# out of the reported average (see docs/EVALUATION.md).
SOURCES=(
    "Aleksandar/NearID-Flux"
    "Aleksandar/NearID-Flux_1024"
    "Aleksandar/NearID-Qwen"
    "Aleksandar/NearID-Qwen_1328"
    "Aleksandar/NearID-PowerPaint"
    "Aleksandar/NearID-SDXL"
    "Aleksandar/NearID-SDXL_1024"
)

# --- Step 1: per-sample similarities, one CSV per distractor source ---
for SRC in "${SOURCES[@]}"; do
    echo "=== $SRC ==="
    CUDA_VISIBLE_DEVICES="$GPU" python -m evaluation.sim_test \
        --mode fullneg \
        --model "$MODEL" \
        --ds "Aleksandar/NearID" \
        --ds_neg "$SRC" \
        --split train \
        --findx "splits/test.json" \
        --output_folder "runs/evals/" \
        --batch_size 64
done

# --- Step 2: pool into SSR / PA tables ---
python -m evaluation.gen_tables \
    --root "./runs/evals/" \
    --split testall \
    --out_path "outputs/tables" \
    --overlap primary

# --- Step 3 (optional): MTG part-level discrimination + oracle alignment ---
CUDA_VISIBLE_DEVICES="$GPU" python -m evaluation.sim_test \
    --mode mtg \
    --model "$MODEL" \
    --output_folder "runs/evals/" \
    --batch_size 64
