"""Metric-Human (MH) correlation on DreamBench++ concept preservation.

Reproduces the DB++ MH column of Table 1. For each of the 7 DreamBench++
personalisation methods, computes the Pearson correlation between each
automatic metric and both human annotator groups, then averages across
methods with a Fisher z-transform.

    python -m dreambench_plus.pearson                       # one table per category
    python -m dreambench_plus.pearson --filter_keys_regex object

Reads only the small per-image score JSONs under ``ratings/``; the DreamBench++
generated images are not needed. See README.md to recompute NearID scores from
the images instead of using the published ones.
"""

import json
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fire import Fire
from scipy import stats

# The 7 DreamBench++ personalisation methods that carry human annotations.
METHOD_LIST = [
    "textual_inversion_sd",
    "dreambooth_sd",
    "dreambooth_lora_sdxl",
    "blip_diffusion",
    "emu2",
    "ip_adapter_plus_vit_h_sdxl",
    "ip_adapter_vit_g_sdxl",
]

# Scoring models compared against the human annotations.
# name -> ratings subdirectory
SCORERS: Dict[str, str] = {
    # Frozen SigLIP2, MAP head skipped (mean over patch tokens). This is the
    # "SigLIP2 (backbone)" number reported in the DB++ column of Table 1.
    "SigLIP2-Backbone": "data_backbone_rating",
    # Frozen SigLIP2 read out through its own pretrained MAP head. Not in the
    # paper; included because it is the like-for-like baseline against NearID,
    # which retrains exactly this head. See README.md and score_backbone.py.
    "SigLIP2-MAPHead": "data_backbone_maphead_rating",
    "VSM": "data_vsm_rating",
    "Qwen3VL-30B": "data_qwen3vl_30b_rating",
    "NearID": "data_nearid_rating",
}


def fisher_z_mean(rs) -> float:
    """Meta-analytic average of Pearson correlations via Fisher z-transform.

    Averaging in z-space avoids the downward bias of naive arithmetic
    averaging of correlations (Silver & Dunlap, 1987).
    """
    rs = np.asarray(rs, dtype=float)
    rs = rs[~np.isnan(rs)]
    if len(rs) == 0:
        return float("nan")
    return float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.9999, 0.9999)))))


def pearson(x, y) -> float:
    r = stats.pearsonr(x, y)[0]
    return round(float(r), 4)


def get_rating(path: str, filter_keys_regex: Optional[str] = None) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    if filter_keys_regex is not None:
        keys = [k for k in data.keys() if re.match(filter_keys_regex, k)]
    else:
        keys = list(data.keys())
    rating = np.array([data[k] for k in keys])
    assert len(rating) > 0, f"No ratings in {path} matching regex {filter_keys_regex!r}"
    return rating


# --- Key reconciliation ------------------------------------------------------
# Rating files are compared position-by-position, so their key orders must agree.
# Two harmless naming differences exist across the released files:
#
#   1. Metric files derive keys from image filenames and carry a trailing image
#      index, e.g. "..._kitten-0_0" where the human file has "..._kitten-0".
#   2. DreamBench++ object 20 is "magic_cube" in the human rating files but
#      "rubik's_cube" in the sample folders, hence in every metric file
#      (including DreamBench++'s own DINO / CLIP-I). Same 9 images either way.
#
# Neither changes ordering, so positional pairing is already correct. This check
# proves it rather than assuming it, and fails loudly if a future file diverges.
KEY_ALIASES = {"rubik's_cube": "magic_cube"}


def canonical_key(k: str) -> str:
    k = re.sub(r"_\d+$", "", k)
    for src, dst in KEY_ALIASES.items():
        k = k.replace(src, dst)
    return k


def assert_keys_aligned(reference_path: str, other_path: str) -> None:
    """Fail if two rating files do not enumerate the same items in the same order."""
    with open(reference_path) as f:
        ref = [canonical_key(k) for k in json.load(f)]
    with open(other_path) as f:
        other = [canonical_key(k) for k in json.load(f)]
    if ref != other:
        first = next(
            (i for i, (a, b) in enumerate(zip(ref, other)) if a != b), min(len(ref), len(other))
        )
        raise AssertionError(
            f"Key order mismatch between\n  {reference_path}\n  {other_path}\n"
            f"  lengths {len(ref)} vs {len(other)}; first divergence at index {first}"
        )


def main(
    _dir: str = "dreambench_plus/ratings",
    out_dir: str = "outputs/dreambenchpp",
    filter_keys_regex: Optional[str] = None,
    subset_regexes: Optional[List[Optional[str]]] = None,
    full: bool = False,
):
    """Compute DB++ metric-human correlations.

    Args:
        full: also report the two alternative aggregations over all four
            categories (see the summary printed at the end, and README.md).
    """
    if subset_regexes is None:
        subset_regexes = (
            ["object", "style", "live_subject_animal", "live_subject_human"]
            if filter_keys_regex is None
            else [filter_keys_regex]
        )

    rows = []
    for subset in subset_regexes:
        print(f"------ Concept preservation, subset={subset!r} ------")
        for method in METHOD_LIST:
            ref = f"{_dir}/data_human_rating/merged_data/group1/{method}-cp.json"
            for other in [
                f"{_dir}/data_human_rating/merged_data/group2/{method}-cp.json",
                f"{_dir}/data_gpt_rating/concept_preservation_full/{method}.json",
                f"{_dir}/data_dino_rating/{method}.json",
                f"{_dir}/data_clipi_rating/{method}.json",
                *(f"{_dir}/{s}/{method}.json" for s in SCORERS.values()),
            ]:
                if os.path.isfile(other):
                    assert_keys_aligned(ref, other)

            h1 = get_rating(ref, subset)
            h2 = get_rating(f"{_dir}/data_human_rating/merged_data/group2/{method}-cp.json", subset)

            print(f"Method: {method}")

            def add(label: str, r1: float, r2: float):
                rows.append({
                    "method": method,
                    "corr_label": label,
                    "subset": subset,
                    "subset_size": len(h1),
                    "score": (r1 + r2) / 2,
                    "pm": abs(r1 - r2) / 2,
                })
                print(f"  {label}: {(r1 + r2) / 2:.3f}±{abs(r1 - r2) / 2:.3f}")

            # human-vs-human agreement is the ceiling, so it has no ± pair
            rows.append({
                "method": method,
                "corr_label": "human1 vs human2",
                "subset": subset,
                "subset_size": len(h1),
                "score": pearson(h1, h2),
                "pm": np.nan,
            })
            print(f"  human1 vs human2: {pearson(h1, h2):.3f}")

            for label, sub in [
                ("human vs gpt", "data_gpt_rating/concept_preservation_full"),
                ("human vs dino", "data_dino_rating"),
                ("human vs clip", "data_clipi_rating"),
            ]:
                f = f"{_dir}/{sub}/{method}.json"
                r = get_rating(f, subset)
                add(label, pearson(h1, r), pearson(h2, r))

            for name, sub in SCORERS.items():
                f = f"{_dir}/{sub}/{method}.json"
                if not os.path.isfile(f):
                    print(f"  {name}: file not found, skipping ({f})")
                    continue
                r = get_rating(f, subset)
                add(f"human vs {name}", pearson(h1, r), pearson(h2, r))
            print()

    df = pd.DataFrame(rows)

    # ---- Fisher-z average across the 7 methods ----
    avg = []
    for subset in df["subset"].dropna().unique():
        for label in df.loc[df["subset"] == subset, "corr_label"].unique():
            mask = (df["subset"] == subset) & (df["corr_label"] == label)
            scores = df.loc[mask, "score"].dropna().values
            if len(scores) == 0:
                continue
            avg.append({
                "method": "Average",
                "corr_label": label,
                "subset": subset,
                "subset_size": df.loc[mask, "subset_size"].dropna().mean(),
                "score": fisher_z_mean(scores),
                "pm": np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else np.nan,
            })
    df = pd.concat([df, pd.DataFrame(avg)], ignore_index=True)

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "correlation_results.csv"), index=False)

    # ---- Per-subset tables (rows=metric, cols=method) ----
    def fmt(s, p):
        if pd.isna(s):
            return ""
        return f"{float(s):.3f}" if pd.isna(p) else f"{float(s):.3f}±{float(p):.3f}"

    df["score_pm"] = [fmt(s, p) for s, p in zip(df["score"], df["pm"])]

    row_order = ["human1 vs human2", "human vs gpt", "human vs dino", "human vs clip"] + [
        f"human vs {k}" for k in SCORERS
    ]
    col_order = METHOD_LIST + ["Average"]
    pretty = {
        "human1 vs human2": "Human-Human",
        "human vs gpt": "Human-GPT",
        "human vs dino": "Human-DINO",
        "human vs clip": "Human-CLIP",
    }

    for subset in list(df["subset"].dropna().unique()):
        sdf = df[df["subset"] == subset]
        table = sdf.pivot_table(index="corr_label", columns="method", values="score_pm", aggfunc="first")
        table = table.reindex(index=[r for r in row_order if r in table.index])
        table = table.reindex(columns=[c for c in col_order if c in table.columns])
        table = table.rename(index=pretty).rename(index=lambda s: s.replace("human vs ", "Human-"))

        print(f"\n------ subset={subset!r} ------")
        print(table.to_string())
        table.to_csv(os.path.join(out_dir, f"correlation_table_{subset}.csv"))
        with open(os.path.join(out_dir, f"correlation_table_{subset}.tex"), "w") as f:
            f.write(table.to_latex(escape=True))

    # ---- Per-category NearID vs. frozen backbone, as reported in the paper ----
    # Table 2 / Table 1 quote the `object` category; the radar figure gives all four.
    piv = df[df["method"] == "Average"].set_index(["subset", "corr_label"])["score"]
    print("\n------ NearID vs. SigLIP2 backbone, per category ------")
    for subset in subset_regexes:
        try:
            d = piv.loc[(subset, "human vs NearID")] - piv.loc[(subset, "human vs SigLIP2-Backbone")]
        except KeyError:
            continue
        print(f"  {subset:22s} {d:+.3f}")
    print(
        "  NearID improves on object, animal and human, and regresses on style —\n"
        "  a category absent from training. See the per-category radar figure in the paper."
    )

    # ---- Headline ------------------------------------------------------------
    scorer_labels = ["Human-Human", "Human-GPT", "Human-DINO", "Human-CLIP"] + [
        f"Human-{k}" for k in SCORERS
    ]
    print("\n" + "=" * 62)
    print("DB++ MH — `object` category (this is the number reported in the paper)")
    print("=" * 62)
    for lab in scorer_labels:
        key = {
            "Human-Human": "human1 vs human2",
            "Human-GPT": "human vs gpt",
            "Human-DINO": "human vs dino",
            "Human-CLIP": "human vs clip",
        }.get(lab, "human vs " + lab.replace("Human-", ""))
        try:
            print(f"  {lab:24s} {piv.loc[('object', key)]:.3f}")
        except KeyError:
            pass

    if full:
        # Whole-benchmark aggregate: Fisher-z over the four per-category values.
        # This keeps every category on its own footing, unlike concatenating all
        # 1350 prompts into one correlation.
        print("\n" + "=" * 62)
        print("All categories — Fisher-z over the 4 category aggregates")
        print("=" * 62)
        for lab in scorer_labels:
            key = {
                "Human-Human": "human1 vs human2",
                "Human-GPT": "human vs gpt",
                "Human-DINO": "human vs dino",
                "Human-CLIP": "human vs clip",
            }.get(lab, "human vs " + lab.replace("Human-", ""))
            vals = [piv.loc[(s, key)] for s in subset_regexes if (s, key) in piv.index]
            if vals:
                print(f"  {lab:24s} {fisher_z_mean(vals):.3f}")

    print(f"\nWrote tables to {out_dir}/")
    print("Per-category plot: python -m dreambench_plus.per_category")


if __name__ == "__main__":
    Fire(main)
