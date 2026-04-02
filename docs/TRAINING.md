# Training NearID

## Prerequisites

1. **Environment setup:**
   ```bash
   conda env create -f environment.yaml
   conda activate nearid
   pip install -e ".[train]"
   ```

2. **Data:** The NearID dataset is loaded from HuggingFace Hub (`Aleksandar/NearID`). Negative source datasets should be provided as HuggingFace dataset paths or local directories.

3. **Hardware:** Single NVIDIA A100 GPU (or equivalent with 40GB+ VRAM). The model trains with mixed precision (fp16) and uses ~15M trainable parameters.

## Training Command (R7 Recipe)

This is the primary training recipe used to produce the results in the paper:

```bash
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
    --output_dir "./runs/trains" \
    --save_steps 100 \
    --eval_steps 100 \
    --mixed_precision fp16
```

## Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--loss_config` | `infonce_ext:1.0` | Loss function. Format: `name:weight[+name:weight]` |
| `--backbone` | `siglip2` | Backbone encoder. Options: `siglip2`, `siglip`, `clip`, `dinov2`, `vit`, `qwen3vl` |
| `--head_type` | `map` | Pooling head type. `map` = Multi-head Attention Pooling (recommended) |
| `--head_out_dim` | `1152` | Output embedding dimensionality |
| `--lr` | `1e-4` | Learning rate |
| `--epochs` | `11` | Number of training epochs |
| `--data.batch_size` | `256` | Per-GPU batch size |
| `--data.mtg_train_path` | `None` | Path to MTG dataset for joint training |
| `--data.mtg_repeat` | `1` | How many times to repeat MTG in MultiDataset |

## Available Loss Functions

All losses are defined in `training/losses.py`:

| Loss | Config String | Description |
|------|--------------|-------------|
| InfoNCE (Extended) | `infonce_ext:1.0` | Primary loss used in the paper |
| Circle Loss | `circle:1.0` | Unified perspective loss |
| Triplet Margin | `triplet:1.0` | FaceNet-style triplet loss |
| SigLIP | `siglip:1.0` | Sigmoid loss for contrastive learning |
| ArcFace | `arcface:1.0` | Angular margin loss |

Losses can be combined: `--loss_config "infonce_ext:1.0+circle:0.5"`

## Architecture

- **Backbone:** Frozen SigLIP2 SO400M ViT/14 @ 384px (~413M parameters, frozen)
- **Head:** Multi-head Attention Pooling (MAP), initialized from SigLIP2 weights (~15M parameters, trained)
- **Output:** L2-normalized 1152-d embeddings

## Checkpoints

Checkpoints are saved every `--save_steps` steps to `{output_dir}/checkpoint-{step}/`. Each checkpoint includes:
- `model.safetensors` (model weights)
- `config.json` (model configuration)
- `preprocessor_config.json` (image processor config)

## Converting to HuggingFace Release Format

After training, convert the NearID checkpoint to the clean NearID format for release:

```bash
python -m training.convert_checkpoint \
    --input_path ./runs/trains/checkpoint-3300 \
    --output_path ./nearid_release \
    --verify
```

This produces a self-contained checkpoint compatible with:
```python
model = AutoModel.from_pretrained("./nearid_release", trust_remote_code=True)
```

## Expected Results

With the R7 recipe above, expect approximately:
- **NearID SSR:** 99.2%
- **NearID PA:** 99.7%
- **DB++ MH:** 0.545
- **Training time:** ~6.5 hours on 1x A100
