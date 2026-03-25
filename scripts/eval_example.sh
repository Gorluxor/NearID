#!/usr/bin/env bash
# NearID evaluation example
#
# Step 1: Compute per-sample similarities (generates CSV)
# Step 2: Aggregate into pooled SSR/PA tables
#
# Prerequisites:
#   conda env create -f environment.yaml
#   conda activate nearid
#   pip install -e ".[eval]"

# --- Step 1: Similarity computation ---
# Using the released HuggingFace model:
CUDA_VISIBLE_DEVICES=0 python -m evaluation.sim_test \
    --mode fullneg \
    --model "Aleksandar/nearid-siglip2" \
    --ds "Aleksandar/EncodeID" \
    --ds_neg "path/to/negative_source" \
    --split train \
    --findx "splits/test.json" \
    --output_folder "runs/evals/" \
    --batch_size 64

# Or using a local training checkpoint:
# CUDA_VISIBLE_DEVICES=0 python -m evaluation.sim_test \
#     --mode fullneg \
#     --model "./runs/trains/checkpoint-3300" \
#     --ds "Aleksandar/EncodeID" \
#     --ds_neg "path/to/negative_source" \
#     --split train \
#     --findx "splits/test.json" \
#     --output_folder "runs/evals/" \
#     --batch_size 64

# --- Step 2: Table aggregation ---
python -m evaluation.gen_tables \
    --root "./runs/evals/" \
    --split testall \
    --out_path "outputs/tables" \
    --overlap primary
