# DreamBench++ Metric-Human Alignment (DB++ MH)

Reproduces the **DB++ MH** column of Table 1: the Pearson correlation between each
automatic identity metric and DreamBench++ human annotations, Fisher-z averaged
across the 7 personalisation methods.

## Reproduce the paper numbers

No images and no GPU needed — the published per-image score JSONs in `ratings/`
(3.7 MB) are all the correlation step requires:

```bash
python -m dreambench_plus.pearson
```

DreamBench++ has four categories. The headline MH quoted in Table 1 and Table 2 of the
paper is the **`object`** category (585 of the 1350 prompts) — the domain NearID is
trained for. Expected `Average` column of `correlation_table_object.csv`:

| Metric | MH (object) |
|---|---|
| Human–Human (ceiling) | 0.648 |
| GPT-4o | 0.597 |
| DINOv2 | 0.492 |
| CLIP-I | 0.493 |
| SigLIP2 (backbone) | 0.515 |
| VSM | 0.190 |
| Qwen3-VL-30B | 0.549 |
| **NearID** | **0.545** |

One table per category is written to `outputs/dreambenchpp/`. For the per-category
numbers plus the bar and radar figures (PDF and PNG):

```bash
python -m dreambench_plus.per_category              # table + plots
python -m dreambench_plus.per_category --annotate   # also label values on the plots
```

NearID versus the frozen backbone, matching the per-category radar figure in the paper:

| Category | prompts | SigLIP2 | NearID | Δ |
|---|---|---|---|---|
| `object` | 585 | 0.515 | 0.545 | **+0.029** |
| `live_subject_animal` | 405 | 0.333 | 0.435 | **+0.102** |
| `live_subject_human` | 180 | 0.376 | 0.440 | **+0.063** |
| `style` | 180 | 0.441 | 0.349 | −0.092 |

NearID improves on object, animal and human, and regresses on `style` — a category
entirely absent from training. The paper reports this explicitly (per-category radar figure) as evidence
that the gains reflect identity learning rather than uniform score inflation.

### Aggregating across categories

```bash
python -m dreambench_plus.pearson --full
```

| Aggregation | SigLIP2 | NearID |
|---|---|---|
| `object` (reported) | 0.515 | **0.545** |
| Fisher-z across the 4 category aggregates | 0.419 | 0.445 |
| single correlation pooled over all 1350 prompts | 0.427 | 0.407 |

The pooled row inverts the ordering even though NearID wins three of four categories:
the categories have different score scales, so one pooled Pearson folds
between-category variance into the estimate. The per-category breakdown is the
meaningful comparison, and is what the paper reports.

## Re-score with a different checkpoint

Only needed to score a checkpoint other than the released model, or a method not
covered here. Fetch the generated images from the DreamBench++ release
([Full Samples, 7 methods](https://drive.google.com/file/d/177GVdYtf0eAOpJO1F4pnoIB87TdVfw4R/view?usp=sharing)):

```bash
pip install gdown && gdown 177GVdYtf0eAOpJO1F4pnoIB87TdVfw4R && unzip samples.zip
```

Then score and re-correlate:

```bash
python -m dreambench_plus.score_nearid --sample_root samples
python -m dreambench_plus.pearson
```

`score_nearid.py` writes `ratings/data_nearid_rating/<method>.json`, using the same
convention as the DreamBench++ CLIP-I / DINO-I scripts: `100 * cosine(f_reference,
f_generated)`, keyed by the generated image's relative path with `/` replaced by `-`.

Re-scoring does not reproduce the published JSONs bit-exactly: the released weights
load in fp32 while the published scores were computed in fp16. On `emu2` the two
agree at `r = 0.9996` (mean 70.99 vs 71.25), which leaves MH unchanged at three
decimal places.

The reference images, prompts, and the remaining rating sets are linked from the
[DreamBench++ repository](https://github.com/yuangpeng/dreambench_plus#data).

## What is in `ratings/`

| Directory | Source | Contents |
|---|---|---|
| `data_human_rating/merged_data/group{1,2}/` | DreamBench++ | Human concept-preservation ratings, two independent annotator groups |
| `data_gpt_rating/concept_preservation_full/` | DreamBench++ | GPT-4o ratings |
| `data_dino_rating/` | DreamBench++ | DINOv2 image similarity |
| `data_clipi_rating/` | DreamBench++ | CLIP-I image similarity |
| `data_backbone_rating/` | this work | Frozen SigLIP2, mean-pooled tokens — the reported baseline |
| `data_backbone_maphead_rating/` | this work | Frozen SigLIP2 through its own MAP head — see below |
| `data_vsm_rating/` | this work | VSM ([Mind-the-Glitch](https://huggingface.co/abdo-eldesokey/mind-the-glitch)) |
| `data_qwen3vl_30b_rating/` | this work | Qwen3-VL-30B-A3B-Instruct judge |
| `data_nearid_rating/` | this work | NearID ([`Aleksandar/nearid-siglip2`](https://huggingface.co/Aleksandar/nearid-siglip2)) |

Each file maps 1350 image keys to a score, for one of the 7 methods.

### What the SigLIP2 baseline is, precisely

NearID does not *add* a pooling head to a headless encoder. SigLIP2 already ends in
a Multi-head Attention Pooling (MAP) head; NearID keeps the trunk frozen and
**retrains that MAP head**, initialised from SigLIP2's own weights.

Given the same frozen trunk, there are two ways to read out a baseline embedding.
Both are shipped, so you can check either without touching a GPU:

```python
tokens = post_layernorm(encoder(patch_embed(image)))      # [B, 729, 1152], frozen

# (a) mean pooling — the MAP head is skipped entirely
emb = tokens.mean(dim=1)

# (b) native MAP head — a learned query attends over the tokens
attended = MultiheadAttention(Q=probe, K=tokens, V=tokens)
emb      = attended + MLP(LayerNorm(attended))
```

NearID is architecturally identical to (b) — same head, same 15M parameters, same
position in the graph — with those weights **retrained** rather than pretrained.

| Row in `pearson.py` | Readout | ratings directory |
|---|---|---|
| `SigLIP2-Backbone` | (a) mean pooling | `data_backbone_rating/` |
| `SigLIP2-MAPHead` | (b) native MAP head | `data_backbone_maphead_rating/` |
| `NearID` | (b) with retrained weights | `data_nearid_rating/` |

**The paper's DB++ column uses (a).** Per-category MH, all reproducible with
`python -m dreambench_plus.pearson --full`:

| Row | object | animal | human | style | All (Fisher-z) |
|---|---|---|---|---|---|
| `SigLIP2-Backbone` — mean pooling *(reported)* | 0.515 | 0.333 | 0.376 | 0.441 | 0.419 |
| `SigLIP2-MAPHead` — native MAP head | 0.489 | 0.369 | 0.397 | 0.398 | 0.415 |
| **`NearID`** | **0.545** | **0.435** | **0.440** | 0.349 | **0.445** |

Two things worth noting:

1. **The reported baseline is the harder one.** On `object`, mean pooling (0.515)
   scores above the native MAP head (0.489), so the margin the paper quotes
   (+0.029) is conservative; measured against the like-for-like MAP-head baseline
   it would be +0.056. NearID leads on the all-category Fisher-z either way.
2. `evaluation/sim_test.py` scores its SigLIP2 baseline with `get_image_features`,
   i.e. readout (b). So the "SigLIP2 (backbone)" row of Table 1 uses the MAP-head
   readout for the NearID and MTG columns and the mean-pooled readout for the DB++
   column. NearID leads under either.

### Regenerating the two baselines

The shipped JSONs are enough for the correlations above. To recompute them from the
DreamBench++ images (both readouts come from one forward pass, so the comparison is
exact):

```bash
python -m dreambench_plus.score_backbone --sample_root samples --out_dir /tmp/check
```

`data_backbone_rating` was originally computed in fp16 and is quantised to 0.25/0.5
steps; `score_backbone.py` runs fp32, so expect agreement at r ≈ 0.9996 rather than
bit-equality. Verified: recomputing the mean-pool readout reproduces the published
per-category values to within 0.001 in all four categories.

## Attribution

The human, GPT, DINO and CLIP-I rating files are redistributed from
[DreamBench++](https://github.com/yuangpeng/dreambench_plus) (Apache License 2.0).
If you use them, please cite DreamBench++ as well:

```bibtex
@article{peng2024dreambench,
  title={DreamBench++: A Human-Aligned Benchmark for Personalized Image Generation},
  author={Peng, Yuang and Cui, Yuxin and Tang, Haomiao and Qi, Zekun and Dong, Runpei
          and Bai, Jing and Han, Chunrui and Ge, Zheng and Zhang, Xiangyu and Xia, Shu-Tao},
  journal={arXiv preprint arXiv:2406.16855},
  year={2024}
}
```
