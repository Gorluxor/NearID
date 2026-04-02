---
dataset_info:
  features:
    - name: id
      dtype: int64
    - name: category
      dtype: string
    - name: category_description
      dtype: string
    - name: img1
      dtype: image
    - name: img2
      dtype: image
    - name: img3
      dtype: image
    - name: n_images
      dtype: int64
    - name: objaverse_id
      dtype: string
    - name: prompts1
      dtype: string
    - name: prompts2
      dtype: string
    - name: prompts3
      dtype: string
    - name: quality
      dtype: string
  splits:
    - name: train
task_categories:
  - image-classification
  - image-to-image
language:
  - en
license: cc-by-4.0
tags:
  - nearid
  - identity-embedding
  - multi-view
  - synthetic
  - metric-learning
pretty_name: "NearID (Multi-View Identity Dataset)"
size_categories:
  - 10K<n<100K
---

# NearID — Multi-View Identity Dataset

[![Model](https://img.shields.io/badge/Model-nearid--siglip2-blue)](https://huggingface.co/Aleksandar/nearid-siglip2) [![Paper](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX) [![Project Page](https://img.shields.io/badge/🌐-Project_Page-blue)](https://gorluxor.github.io/NearID/) [![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Gorluxor/NearID) [![KAUST](https://img.shields.io/badge/KAUST-009B4D)](https://www.kaust.edu.sa/) [![Snap Research](https://img.shields.io/badge/Snap_Research-FFFC00?logoColor=black)](https://research.snap.com/)

This is the **base positives dataset** for the [NearID](https://huggingface.co/Aleksandar/nearid-siglip2) project. Each sample contains multiple views of the **same identity** rendered in different backgrounds/contexts.

Near-identity distractors (different but similar instances in matched context) are available as separate datasets listed below. Together, they form the NearID training and evaluation benchmark.

## Quick Start

```python
from datasets import load_dataset

# Load base positives
ds = load_dataset("Aleksandar/NearID")

# Load a negative source for contrastive training/evaluation
neg = load_dataset("Aleksandar/NearID-Flux")
```

## Dataset Structure

| Column | Type | Description |
|---|---|---|
| `id` | int64 | Sample ID (shared across all NearID datasets) |
| `category` | string | Object category |
| `category_description` | string | Natural language description of the identity |
| `img1`, `img2`, `img3` | image | Multi-view images of the same identity in different contexts |
| `n_images` | int64 | Number of valid views |
| `objaverse_id` | string | Source Objaverse object identifier |
| `prompts1`–`prompts3` | string | Generation prompts for each view |
| `quality` | string | Quality label |

## All NearID Datasets

| Dataset | Description | Resolution |
|---|---|---|
| [Aleksandar/NearID](https://huggingface.co/datasets/Aleksandar/NearID) | Multi-view positives (anchor + positive views) | Base |
| [Aleksandar/NearID-Flux](https://huggingface.co/datasets/Aleksandar/NearID-Flux) | Near-identity distractors via FLUX.1 inpainting | 512×512 |
| [Aleksandar/NearID-Flux_1024](https://huggingface.co/datasets/Aleksandar/NearID-Flux_1024) | Near-identity distractors via FLUX.1 inpainting | 1024×1024 |
| [Aleksandar/NearID-FluxC](https://huggingface.co/datasets/Aleksandar/NearID-FluxC) | Near-identity distractors via FLUX.1 Canny-guided inpainting | 512×512 |
| [Aleksandar/NearID-FluxC_1024](https://huggingface.co/datasets/Aleksandar/NearID-FluxC_1024) | Near-identity distractors via FLUX.1 Canny-guided inpainting | 1024×1024 |
| [Aleksandar/NearID-PowerPaint](https://huggingface.co/datasets/Aleksandar/NearID-PowerPaint) | Near-identity distractors via PowerPaint inpainting | 512×512 |
| [Aleksandar/NearID-Qwen](https://huggingface.co/datasets/Aleksandar/NearID-Qwen) | Near-identity distractors via Qwen-based inpainting | 512×512 |
| [Aleksandar/NearID-Qwen_1328](https://huggingface.co/datasets/Aleksandar/NearID-Qwen_1328) | Near-identity distractors via Qwen-based inpainting | 1328×1328 |
| [Aleksandar/NearID-SDXL](https://huggingface.co/datasets/Aleksandar/NearID-SDXL) | Near-identity distractors via Stable Diffusion XL inpainting | 512×512 |
| [Aleksandar/NearID-SDXL_1024](https://huggingface.co/datasets/Aleksandar/NearID-SDXL_1024) | Near-identity distractors via Stable Diffusion XL inpainting | 1024×1024 |

## Related

- **Model:** [Aleksandar/nearid-siglip2](https://huggingface.co/Aleksandar/nearid-siglip2) — NearID identity embedding model
- **Paper:** [NearID: Identity Representation Learning via Near-identity Distractors](https://arxiv.org/abs/XXXX.XXXXX)
- **Code:** [github.com/Gorluxor/NearID](https://github.com/Gorluxor/NearID)

## License & Attribution

This dataset is released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). It is derived from the [SynCD](https://github.com/nupurkmr9/syncd) dataset (MIT License, Copyright 2022 SynCD). If you use this dataset, please cite both NearID and SynCD.

## Citation

```bibtex
@article{cvejic2026nearid,
  title={NearID: Identity Representation Learning via Near-identity Distractors},
  author={Cvejic, Aleksandar and Abdal, Rameen and Eldesokey, Abdelrahman and Ghanem, Bernard and Wonka, Peter}
}
```
