#!/usr/bin/env bash
# NearID training example (R7 recipe — primary model from the paper)
#
# Prerequisites:
#   conda env create -f environment.yaml
#   conda activate nearid
#
# Data: The NearID dataset is loaded from HuggingFace Hub (Aleksandar/NearID).
#       Negative sources should be provided as HF dataset paths or local directories.

CUDA_VISIBLE_DEVICES=0 accelerate launch -m training.train \
    --loss_config "infonce_ext:1.0" \
    --backbone siglip2 \
    --pretrained_backbone "google/siglip2-so400m-patch14-384" \
    --head_type map \
    --head_out_dim 1152 \
    --lr 1e-4 \
    --epochs 11 \
    --data.batch_size 128 \
    --data.train_path "Aleksandar/NearID" \
    --data.neg_paths "[path/to/neg_source_1,path/to/neg_source_2]" \
    --data.val_indices_path "splits/val.json" \
    --data.test_indices_path "splits/test.json" \
    --data.mtg_train_path "abdo-eldesokey/mtg-dataset" \
    --data.mtg_repeat 4 \
    --data.mtg_margin 0.1 \
    --output_dir "./runs/trains" \
    --save_steps 100 \
    --eval_steps 100 \
    --mixed_precision fp16 \
    --wandb.project "NearID" \
    --wandb.mode "online"
