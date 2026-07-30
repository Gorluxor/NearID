# NearID: Identity Representation Learning via Near-identity Distractors

[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-red.svg)](https://arxiv.org/abs/2604.01973)
[![arXiv](https://img.shields.io/badge/arXiv-2604.01973-b31b1b.svg)](https://arxiv.org/abs/2604.01973)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-nearid--siglip2-yellow)](https://huggingface.co/Aleksandar/nearid-siglip2)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-NearID-blue)](https://huggingface.co/datasets/Aleksandar/NearID)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Project Page](https://img.shields.io/badge/Project%20Page-NearID-blue)](https://gorluxor.github.io/NearID)
[![KAUST](https://img.shields.io/badge/KAUST-009B4D)](https://www.kaust.edu.sa/)
[![Snap Research](https://img.shields.io/badge/Snap_Research-FFFC00?logoColor=black)](https://research.snap.com/)

<p align="center">
  <img src="assets/nearid_teaser.png" width="100%" alt="NearID overview">
</p>

NearID produces **identity-aware image embeddings** that remain stable across background and context changes while correctly rejecting *near-identity distractors* — visually similar but different instances placed in the same context. It is designed for evaluating identity preservation in personalized image generation.

## News

- **Jul 2026** — Code, dataset splits and the full reproduction pipeline released.
- **Jul 2026** — 🎉 **NearID is accepted to ECCV 2026** (Malmö, Sept 8–13).
- **Mar 2026** — Model and datasets released on the HuggingFace Hub.

## What is released

| | Component | Where |
|---|---|---|
| ✅ | **Model weights** | [`Aleksandar/nearid-siglip2`](https://huggingface.co/Aleksandar/nearid-siglip2) |
| ✅ | **NearID dataset** (19K identities) | [`Aleksandar/NearID`](https://huggingface.co/datasets/Aleksandar/NearID) |
| ✅ | **Near-identity distractors** (9 sources, 316K images) | [HF datasets](#datasets) |
| ✅ | **Official train/val/test splits** | [`splits/`](splits) |
| ✅ | **Inference code** | [`nearid/`](nearid), [`examples/`](examples) |
| ✅ | **Training code** | [`training/`](training), [docs/TRAINING.md](docs/TRAINING.md) |
| ✅ | **Evaluation code** (NearID-bench SSR/PA, MTG) | [`evaluation/`](evaluation), [docs/EVALUATION.md](docs/EVALUATION.md) |
| ✅ | **DreamBench++ alignment pipeline + rating files** | [`dreambench_plus/`](dreambench_plus) |
| ✅ | **End-to-end reproduction of Table 1** | [REPRODUCTION.md](REPRODUCTION.md) |
| ⬜ | Distractor *generation* pipeline (inpainting ensemble) | not released; the generated datasets above are |
| ⬜ | VSM baseline dependency | external: [Mind-the-Glitch](https://github.com/abdo-eldesokey/mind-the-glitch) |

Every number in Table 1 is reproducible from this repository — see
[REPRODUCTION.md](REPRODUCTION.md) for the exact commands, expected values and tolerances.

## Quick Start

Scoring images needs nothing from this repository — the model carries its own
implementation on the Hub:

```bash
pip install "transformers>=4.45" torch pillow
```

```python
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

model = AutoModel.from_pretrained("Aleksandar/nearid-siglip2", trust_remote_code=True)
processor = AutoImageProcessor.from_pretrained("Aleksandar/nearid-siglip2")

inputs = processor(images=Image.open("photo.jpg"), return_tensors="pt")
embedding = model.get_image_features(**inputs)  # [1, 1152], L2-normalised
```

## Results

### Near-Identity Discrimination & Alignment (Table 1)

| Scoring Model | NearID SSR | NearID PA | MTG MO | MTG MOpair | MTG SSR | MTG PA | DB++ MH |
|---|---|---|---|---|---|---|---|
| CLIP ViT-L/14 | 10.31 | 20.92 | 0.239 | 0.484 | 0.0 | 0.0 | 0.493 |
| DINOv2 ViT-L/14 | 20.43 | 34.55 | 0.324 | **0.519** | 0.0 | 0.0 | 0.492 |
| SigLIP2 (backbone) | 30.74 | 48.81 | 0.180 | 0.366 | 0.0 | 0.0 | 0.516 |
| VSM | 32.13 | 46.70 | 0.394 | 0.445 | 7.0 | 24.5 | 0.190 |
| **NearID (Ours)** | **99.17** | **99.71** | **0.465** | 0.486 | **35.0** | **46.5** | **0.545** |

SSR and PA are averaged across seven inpainting settings (three excluded from training). MO/MOpair = metric-to-oracle correlation; MH = metric-to-human correlation (Fisher-z averaged).

**Every number above is reproducible from this repository** — see [REPRODUCTION.md](REPRODUCTION.md) for the exact commands, expected outputs, and tolerances.

## Installation

NearID is not on PyPI — install from a clone.

**Inference only** (identity embeddings for your own images):

```bash
git clone https://github.com/Gorluxor/NearID.git && cd NearID
pip install -e .
```

**Training, evaluation and reproduction** (adds `training/`, `evaluation/`, `dreambench_plus/`):

```bash
git clone https://github.com/Gorluxor/NearID.git && cd NearID
conda env create -f environment.yaml
conda activate nearid
pip install -e ".[all]"
```

Reproducing the DB++ column needs neither a GPU nor any download — the rating
files are in the repository:

```bash
python -m dreambench_plus.pearson
```

## Usage

### Pairwise Similarity

```python
import torch

emb_a = model.get_image_features(**processor(images=img_a, return_tensors="pt"))
emb_b = model.get_image_features(**processor(images=img_b, return_tensors="pt"))

similarity = (emb_a @ emb_b.T).item()  # cosine similarity
```

### Batch Inference

```python
images = [Image.open(p) for p in image_paths]
inputs = processor(images=images, return_tensors="pt", padding=True)
embeddings = model.get_image_features(**inputs)  # [B, 1152]

sim_matrix = embeddings @ embeddings.T
```

## Architecture

| Property | Value |
|----------|-------|
| Base model | `google/siglip2-so400m-patch14-384` |
| Backbone | SigLIP2 SO400M ViT/14 @ 384px (**frozen**) |
| Pooling head | Multi-head Attention Pooling (MAP), initialised from SigLIP2 (**trained**) |
| Embedding dim | 1152 |
| Total parameters | ~428M |
| Trainable parameters | ~15M (head-only) |
| Input resolution | 384 x 384 |

## Training

Train with the NearID loss (extended InfoNCE with near-identity distractor ranking):

```bash
accelerate launch -m training.train \
    --loss_config "infonce_ext:1.0" \
    --head_type map --head_out_dim 1152 \
    --lr 1e-4 --epochs 11 --data.batch_size 128
```

See [docs/TRAINING.md](docs/TRAINING.md) for the full guide.

## Evaluation

```bash
# Step 1: compute similarities on the held-out test split
python -m evaluation.sim_test \
    --mode fullneg --model "Aleksandar/nearid-siglip2" \
    --ds "Aleksandar/NearID" --ds_neg "Aleksandar/NearID-Flux" \
    --split train --findx "splits/test.json" \
    --output_folder runs/evals/ --batch_size 64

# Step 2: aggregate tables
python -m evaluation.gen_tables --root runs/evals/ --overlap primary

# DreamBench++ human alignment (CPU, seconds — ratings are included)
python -m dreambench_plus.pearson
```

All seven sources at once: `bash scripts/eval_example.sh`.

See [REPRODUCTION.md](REPRODUCTION.md) to reproduce Table 1 exactly, and [docs/EVALUATION.md](docs/EVALUATION.md) for the full guide.

## Datasets

The NearID benchmark consists of multi-view positives and near-identity distractors generated by an ensemble of inpainting pipelines. All datasets are released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

| Dataset | Description | HuggingFace |
|---|---|---|
| **NearID** | Multi-view positives (anchor + positive views) | [`Aleksandar/NearID`](https://huggingface.co/datasets/Aleksandar/NearID) |
| NearID-Flux | Near-identity distractors via FLUX.1 | [`Aleksandar/NearID-Flux`](https://huggingface.co/datasets/Aleksandar/NearID-Flux) |
| NearID-Flux_1024 | FLUX.1 @ 1024px | [`Aleksandar/NearID-Flux_1024`](https://huggingface.co/datasets/Aleksandar/NearID-Flux_1024) |
| NearID-FluxC | FLUX.1 Canny-guided | [`Aleksandar/NearID-FluxC`](https://huggingface.co/datasets/Aleksandar/NearID-FluxC) |
| NearID-FluxC_1024 | FLUX.1 Canny-guided @ 1024px | [`Aleksandar/NearID-FluxC_1024`](https://huggingface.co/datasets/Aleksandar/NearID-FluxC_1024) |
| NearID-PowerPaint | PowerPaint inpainting | [`Aleksandar/NearID-PowerPaint`](https://huggingface.co/datasets/Aleksandar/NearID-PowerPaint) |
| NearID-Qwen | Qwen-based inpainting | [`Aleksandar/NearID-Qwen`](https://huggingface.co/datasets/Aleksandar/NearID-Qwen) |
| NearID-Qwen_1328 | Qwen-based @ 1328px | [`Aleksandar/NearID-Qwen_1328`](https://huggingface.co/datasets/Aleksandar/NearID-Qwen_1328) |
| NearID-SDXL | Stable Diffusion XL inpainting | [`Aleksandar/NearID-SDXL`](https://huggingface.co/datasets/Aleksandar/NearID-SDXL) |
| NearID-SDXL_1024 | SDXL @ 1024px | [`Aleksandar/NearID-SDXL_1024`](https://huggingface.co/datasets/Aleksandar/NearID-SDXL_1024) |

```python
from datasets import load_dataset

positives = load_dataset("Aleksandar/NearID", split="train")      # 19,386 rows
negatives = load_dataset("Aleksandar/NearID-Flux", split="train") # index-aligned
```

Every distractor source has the same 19,386 rows in the same order as the positives, so a single set of indices selects matching subsets from all of them.

### Train / Val / Test Splits

The official partition used for every result in the paper lives in [`splits/`](splits): **18,786 / 100 / 500** samples. Each file is a JSON list of 0-based **row positions** into the unshuffled `train` split — not `id` values, which are sparse.

```python
import json
test = positives.select(json.load(open("splits/test.json")))   # 500 rows
# equivalent: positives.filter(lambda r: r["split"] == "test")
```

See [`splits/README.md`](splits/README.md) for details and [`examples/load_splits.py`](examples/load_splits.py) for a runnable end-to-end example.

## Model Zoo

| Model | HuggingFace Hub | SSR | PA | MH |
|-------|----------------|-----|----|----|
| NearID (SigLIP2 + MAP) | [`Aleksandar/nearid-siglip2`](https://huggingface.co/Aleksandar/nearid-siglip2) | 99.17 | 99.71 | 0.545 |

## Citation

```bibtex
@inproceedings{cvejic2026nearid,
  title={NearID: Identity Representation Learning via Near-identity Distractors},
  author={Cvejic, Aleksandar and Abdal, Rameen and Eldesokey, Abdelrahman and Ghanem, Bernard and Wonka, Peter},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## Acknowledgements

This work was supported by King Abdullah University of Science and Technology (KAUST) and Snap Inc.

## License

- **Code & model weights:** Apache License 2.0. See [LICENSE](LICENSE).
- **Datasets:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Derived from [SynCD](https://github.com/nupurkmr9/syncd) (MIT License).
