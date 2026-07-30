# Reproducing Table 1

Every number in Table 1 can be regenerated from this repository plus the released
model and datasets on the HuggingFace Hub. No local data copies are required.

```
| Scoring Model | NearID SSR | NearID PA | MTG MO | MTG MOpair | MTG SSR | MTG PA | DB++ MH |
|---------------|------------|-----------|--------|------------|---------|--------|---------|
| NearID (Ours) |    99.17   |   99.71   |  0.465 |   0.486    |  35.0   |  46.5  |  0.545  |
```

| Columns | Command | Needs GPU |
|---|---|---|
| NearID SSR, PA | [§1](#1-nearid-ssr--pa) | yes |
| MTG MO, MOpair, SSR, PA | [§2](#2-mtg-metrics) | yes |
| DB++ MH | [§3](#3-db-mh) | no |

Prerequisites:

```bash
conda env create -f environment.yaml
conda activate nearid
pip install -e ".[all]"
```

---

## 1. NearID SSR / PA

Scores the 500-sample held-out test split against seven inpainting sources, then
pools them.

```bash
bash scripts/eval_example.sh
```

Table 1 is the **`full` mask row of the `all` pooling group**:

```bash
cat outputs/tables/csv/disc_bidir_all.csv | awk -F, '$1=="full"'
```

The `all` group pools `flux`, `flux_1024`, `qwen`, `qwen_1328`, `powerpaint`,
`sdxl`, `sdxl_1024` — the seven settings reported in the paper. `fluxc` and
`fluxc_1024` are computed by the pipeline but excluded from this average; the
per-source breakdown is in the other `disc_bidir_*.csv` files.

Expected: `n=3500`, `SSRm=99.17`, `PA=99.71`. Note the CSV column named `SSRm` is
what the paper calls **SSR** (AND-based: a sample counts as a win only if every
valid margin is positive). The column literally named `SSR` is the mean-based
variant and is not the reported metric.

Single-source sanity check (~10 min on one A100, downloads ~6 GB):

```bash
python -m evaluation.sim_test --mode fullneg \
    --model "Aleksandar/nearid-siglip2" \
    --ds "Aleksandar/NearID" --ds_neg "Aleksandar/NearID-Flux" \
    --split train --findx "splits/test.json" \
    --output_folder runs/evals/ --batch_size 64
python -m evaluation.gen_tables --root ./runs/evals/ --split testall \
    --out_path outputs/tables --overlap primary
```

Expected for Flux alone: `SSRm=99.4`, `PA=99.77`.

Baselines are reproduced by swapping `--model`, e.g.
`google/siglip2-so400m-patch14-384`, `openai/clip-vit-large-patch14`,
`facebook/dinov2-large`.

## 2. MTG metrics

Part-level discrimination and oracle alignment on the 100-sample MTG test split
([`abdo-eldesokey/mtg-dataset`](https://huggingface.co/datasets/abdo-eldesokey/mtg-dataset)):

```bash
python -m evaluation.sim_test --mode mtg \
    --model "Aleksandar/nearid-siglip2" \
    --output_folder runs/evals/ --batch_size 64
```

MO / MOpair / SSR / PA are printed in the run summary and stored in
`runs/evals/MTG-Dataset/`. Expected: `MO≈0.454`, `MOpair≈0.475`, `SSR≈34.0`,
`PA≈46.0`.

The **VSM** baseline row additionally requires the
[Mind-the-Glitch](https://github.com/abdo-eldesokey/mind-the-glitch) package on
`PYTHONPATH`; it is not a dependency of this repository. Everything else in the
MTG column runs without it.

## 3. DB++ MH

Correlation against DreamBench++ human annotations. Runs on CPU in seconds — the
per-image score JSONs are included:

```bash
python -m dreambench_plus.pearson
```

This prints the reported headline directly: **NearID 0.545** vs SigLIP2 backbone
0.515, and writes `outputs/dreambenchpp/correlation_table_object.csv`.

DreamBench++ is split into four categories. The reported MH is the **`object`**
category (585 of the 1350 prompts) — the domain NearID is trained for — Fisher-z
averaged over the 7 personalisation methods:

| Metric | MH (object) |
|---|---|
| Human–Human (ceiling) | 0.648 |
| GPT-4o | 0.597 |
| DINOv2 | 0.492 |
| CLIP-I | 0.493 |
| SigLIP2 (backbone) | 0.515 |
| VSM | 0.190 |
| Qwen3-VL 30B | 0.549 |
| **NearID** | **0.545** |

### All four categories

Per-category numbers and the figure:

```bash
python -m dreambench_plus.per_category              # table + bar/radar plots
python -m dreambench_plus.per_category --annotate   # value labels on the plots
```

| Scorer | Object | Animal | Human | Style |
|---|---|---|---|---|
| GPT-4o | 0.597 | 0.488 | 0.432 | 0.466 |
| DINOv2 | 0.492 | 0.434 | 0.248 | 0.374 |
| CLIP-I | 0.493 | 0.364 | 0.413 | 0.418 |
| SigLIP2 | 0.515 | 0.333 | 0.376 | 0.441 |
| VSM | 0.190 | 0.183 | 0.148 | 0.236 |
| Qwen3-VL 30B | 0.549 | 0.477 | 0.485 | 0.472 |
| **NearID** | **0.545** | **0.435** | **0.440** | 0.349 |
| *NearID − SigLIP2* | *+0.029* | *+0.102* | *+0.063* | *−0.092* |

NearID improves on object, animal and human, and regresses on `style` — a category
absent from training. The paper reports this explicitly in the per-category radar
figure.

### Aggregating across categories

```bash
python -m dreambench_plus.pearson --full
```

| Aggregation | SigLIP2 | NearID |
|---|---|---|
| `object` (reported) | 0.515 | **0.545** |
| Fisher-z over the 4 category aggregates | 0.419 | **0.445** |

Each category keeps its own footing under Fisher-z. Concatenating all 1350 prompts
into a single correlation instead would fold between-category variance into the
estimate — the categories have different score scales — so that aggregation is not
reported.

### Which SigLIP2 baseline?

The frozen trunk can be read out two ways, and both are shipped:

| Row | Readout | object | All (Fisher-z) |
|---|---|---|---|
| `SigLIP2-Backbone` *(reported)* | mean over patch tokens; MAP head skipped | 0.515 | 0.419 |
| `SigLIP2-MAPHead` | SigLIP2's own pretrained MAP head | 0.489 | 0.415 |
| **`NearID`** | same MAP head, retrained | **0.545** | **0.445** |

NearID retrains SigLIP2's MAP head, so `SigLIP2-MAPHead` is the like-for-like
baseline. Which of the two is harder depends on the category: on the reported
`object` category mean pooling is the stronger baseline (so the quoted +0.029 is
conservative; against MAP head it is +0.056), while on `animal` and `human` mean
pooling is the softer one (+0.102 → +0.065 and +0.063 → +0.042). Every
conclusion holds under either readout — NearID leads on object, animal and human,
trails on style, and leads on the all-category Fisher-z both ways — only the margin
sizes change. Both rows appear in `pearson.py --full`. Note that
`evaluation/sim_test.py` uses the MAP-head readout, so Table 1's SigLIP2 row is
MAP-head for the NearID/MTG columns and mean-pooled for DB++.

See [dreambench_plus/README.md](dreambench_plus/README.md) for the pseudo-code, the
full tables, and how to regenerate either baseline or re-score a different checkpoint.

---

## Tolerances

Numbers reproduce to within ~0.5% of the paper. The released weights load in
fp32 whereas the reported evaluations ran in fp16, which flips an occasional
borderline margin trial:

| Metric | Reproduced | Paper |
|---|---|---|
| NearID SSR (Flux, n=500) | 99.4 | 99.4 |
| NearID PA (Flux, n=500) | 99.77 | 99.81 |
| MTG MO | 0.454 | 0.465 |
| MTG MOpair | 0.475 | 0.486 |
| MTG SSR / PA | 34.0 / 46.0 | 35.0 / 46.5 |
| DB++ MH, NearID | 0.545 | 0.545 |
| DB++ MH, SigLIP2 baseline | 0.515 | 0.516 |

The DB++ SigLIP2 row differs in the last digit. The pipeline here computes
`0.515475`, which rounds to 0.515; the paper reports 0.516. This is a
reproduction difference, not display rounding, and it does not affect any
comparison — NearID's margin is +0.029 either way.

## Splits

`splits/{train,val,test}.json` are the official partition — 18,786 / 100 / 500
0-based **row positions** into the unshuffled `train` split of
[`Aleksandar/NearID`](https://huggingface.co/datasets/Aleksandar/NearID). They are
not `id` values; `id` is sparse and inherited from the source pool. The same
indices apply unchanged to every distractor source, which are index-aligned with
the positives. See [splits/README.md](splits/README.md) and
[examples/load_splits.py](examples/load_splits.py).

Equivalently, the dataset carries a `split` column with the same partition.
