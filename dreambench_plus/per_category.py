"""DB++ per-category human-alignment radar / bar figure.

This is the script that produced the per-category figure in the paper. The
plotting code is unchanged; the only difference is the data source: correlations
are recomputed here from the per-image rating JSONs in ``ratings/`` instead of
being read from pre-existing correlation CSVs, so the figure is reproducible from
this repository alone. No model, no GPU and no generated images are needed.

The argument the figure makes: NearID improves alignment despite training only on
rigid objects. Animal and Human correlation increase; Style decreases, which is
expected and interpretable.

Usage
-----
    python -m dreambench_plus.per_category
    python -m dreambench_plus.per_category --plot_type radar
    python -m dreambench_plus.per_category --models "NearID,SigLIP2-Backbone"
    python -m dreambench_plus.per_category --annotate False   # drop vertex labels
    python -m dreambench_plus.per_category --usetex False     # if LaTeX is unavailable

Writes PDF (vector, for LaTeX) and PNG of each figure, plus a CSV of the numbers,
and prints the same numbers to the console.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire
from matplotlib.projections.polar import PolarAxes
from scipy import stats

from .pearson import METHOD_LIST, assert_keys_aligned, fisher_z_mean

CATEGORIES = ["object", "live_subject_animal", "live_subject_human", "style"]
CATEGORY_LABELS = ["Object", "Animal", "Human", "Style"]

# The NearID row is scored with the publicly released checkpoint.
MODEL_ID = "Aleksandar/nearid-siglip2"

# Model key -> ratings subdirectory.
RATING_DIRS: Dict[str, str] = {
    "NearID": "data_nearid_rating",
    "SigLIP2-Backbone": "data_backbone_rating",
    "SigLIP2-MAPHead": "data_backbone_maphead_rating",
    "VSM": "data_vsm_rating",
    "DINO": "data_dino_rating",
    "GPT": "data_gpt_rating/concept_preservation_full",
    "CLIP": "data_clipi_rating",
    "Qwen3VL-30B": "data_qwen3vl_30b_rating",
}

# Key -> display name for figures.
DISPLAY_NAMES = {
    "NearID": "NearID (Ours)",
    "SigLIP2-Backbone": "SigLIP2",
    "SigLIP2-MAPHead": "SigLIP2 (MAP head)",
    "GPT": "GPT-4o",
    "DINO": "DINOv2",
    "CLIP": "CLIP",
    "VSM": "VSM",
    "Qwen3VL-30B": "Qwen3-VL 30B",
}

# Model whose trace is repeated (dashed) on every subplot for reference.
OURS_KEY = "NearID"
OURS_COLOR = "#E03C3C"  # warm red — prominent but not aggressive, good print contrast

# Distinct colors per model (curated for print + colorblind friendliness).
MODEL_COLORS = {
    "NearID":           OURS_COLOR,  # red — ours, prominent
    "SigLIP2-Backbone": "#1F77B4",   # blue — baseline
    "SigLIP2-MAPHead":  "#17BECF",   # cyan — alternative baseline readout
    "GPT":              "#FF7F0E",   # orange
    "DINO":             "#2CA02C",   # green
    "CLIP":             "#9467BD",   # purple
    "VSM":              "#8C564B",   # brown
    "Qwen3VL-30B":      "#CCB974",
}

# The model set used for the figure in the paper.
DEFAULT_MODELS = "NearID,SigLIP2-Backbone,VSM,DINO,GPT"


def _set_style(usetex: bool) -> None:
    """ECCV LNCS publication-quality settings (see CLAUDE.md)."""
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


# ---------------------------------------------------------------------------
# Data: recomputed from the rating JSONs (this replaces the CSV reader)
# ---------------------------------------------------------------------------
def _vals(path: str, rx: str) -> np.ndarray:
    with open(path) as f:
        d = json.load(f)
    return np.array([v for k, v in d.items() if re.match(rx, k)], dtype=float)


def load_per_category(models: List[str], ratings_dir: str) -> pd.DataFrame:
    """Per-category Fisher-z mean correlation for each model, over the 7 methods.

    Mirrors the `Average` column that pearson.py writes into
    correlation_table_<category>.csv, computed straight from the rating files.
    """
    rows = []
    for cat in CATEGORIES:
        for model in models:
            sub = RATING_DIRS.get(model)
            if sub is None:
                print(f"[WARNING] unknown model {model!r}; known: {list(RATING_DIRS)}")
                continue
            rs = []
            for m in METHOD_LIST:
                f = f"{ratings_dir}/{sub}/{m}.json"
                if not os.path.isfile(f):
                    continue
                ref = f"{ratings_dir}/data_human_rating/merged_data/group1/{m}-cp.json"
                assert_keys_aligned(ref, f)
                h1 = _vals(ref, cat)
                h2 = _vals(f"{ratings_dir}/data_human_rating/merged_data/group2/{m}-cp.json", cat)
                r = _vals(f, cat)
                rs.append((stats.pearsonr(h1, r)[0] + stats.pearsonr(h2, r)[0]) / 2)
            if not rs:
                print(f"[WARNING] no rating files for {model!r}, skipping.")
                continue
            rows.append({
                "model": model,
                "category": cat,
                "category_label": CATEGORY_LABELS[CATEGORIES.index(cat)],
                # Rounded to 3 decimals, which is the precision the paper reports
                # and the precision the original figure script read back out of
                # correlation_table_<category>.csv. Keeping it makes the vertex
                # labels identical to the published figure (e.g. 0.545 -> "0.55").
                "mean": round(fisher_z_mean(rs), 3),
                # Standard error across the 7 methods, matching pearson.py.
                "std": round(float(np.std(rs, ddof=1) / np.sqrt(len(rs))), 3) if len(rs) > 1 else 0.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting — ported unchanged from the figure script used for the paper
# ---------------------------------------------------------------------------
def plot_grouped_bar(data: pd.DataFrame, out_path: Path):
    """Grouped bar chart: categories on x-axis, models as grouped bars."""
    models = data["model"].unique()
    cats = CATEGORY_LABELS
    n_models = len(models)
    n_cats = len(cats)

    x = np.arange(n_cats)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, model in enumerate(models):
        display = DISPLAY_NAMES.get(model, model)
        color = MODEL_COLORS.get(model, "#333333")
        subset = data[data["model"] == model]
        means = [subset[subset["category_label"] == c]["mean"].values[0] if c in subset["category_label"].values else 0 for c in cats]
        stds = [subset[subset["category_label"] == c]["std"].values[0] if c in subset["category_label"].values else 0 for c in cats]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=display,
               color=color, capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_ylabel(r"$\mathcal{M}_H$ (Pearson correlation with human judgments)")
    ax.set_xlabel("DreamBench++ Category")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(data["mean"]) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".png")), dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path.with_suffix('.png')}")
    plt.close(fig)


def _get_model_values(data: pd.DataFrame, model: str, cats: List[str]) -> List[float]:
    """Extract ordered values for a model across category labels."""
    subset = data[data["model"] == model]
    return [
        subset[subset["category_label"] == c]["mean"].values[0]
        if c in subset["category_label"].values else 0.0
        for c in cats
    ]


def plot_radar(data: pd.DataFrame, out_path: Path, ours_key: str = OURS_KEY,
               annotate: bool = True):
    """
    Multi-subplot radar chart (CVPR/ECCV style).

    Layout: one polar subplot per model, arranged horizontally.
    - First subplot: "Ours" with solid fill.
    - Subsequent subplots: each model with its own solid trace, plus a dashed
      pink overlay of "Ours" for direct visual comparison.
    - Numeric values annotated at each vertex.
    """
    models = list(data["model"].unique())
    cats = CATEGORY_LABELS
    n_cats = len(cats)
    n_models = len(models)

    # Ensure ours is first
    if ours_key in models:
        models.remove(ours_key)
        models.insert(0, ours_key)

    # Rotate axes 45° off cardinal directions so labels sit diagonally,
    # avoiding text collisions between adjacent subplots → more compact figure
    angle_offset = np.pi / 4  # 45°
    angles = (np.linspace(0, 2 * np.pi, n_cats, endpoint=False) + angle_offset).tolist()
    angles += angles[:1]
    ours_values = _get_model_values(data, ours_key, cats) if ours_key in data["model"].values else None

    # Global y-limits (shared across subplots for fair comparison)
    all_means = data["mean"].values
    y_max = max(all_means) * 1.15
    y_min = min(0.0, min(all_means) - 0.05)

    ours_closed = None
    if ours_values is not None:
        ours_closed = ours_values + ours_values[:1]

    # --- Figure setup ---
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(3.2 * n_models, 3.6),
        subplot_kw=dict(polar=True),
    )
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # type:ignore
    assert len(axes) == n_models, "Number of subplots must match number of models"
    for idx, model in enumerate(models):
        ax = axes[idx]
        assert isinstance(ax, PolarAxes), "Expected ax to be a PolarAxes"
        display = DISPLAY_NAMES.get(model, model)
        color = MODEL_COLORS.get(model, "#333333")
        values = _get_model_values(data, model, cats)
        values_closed = values + values[:1]

        # Grid styling
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats, fontsize=8, fontweight="medium")
        # Radial grid at every 0.25 interval
        r_ticks = np.arange(0.25, y_max + 0.01, 0.25)
        ax.set_rticks(r_ticks)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", labelsize=0)
        ax.spines["polar"].set_linewidth(0.4)
        ax.grid(color="grey", linewidth=0.3, alpha=0.5)

        # Dashed "Ours" overlay on every subplot (except ours itself, where
        # we draw the solid version)
        if ours_closed is not None and model != ours_key:
            ax.plot(
                angles, ours_closed,
                linestyle="--", linewidth=2.0,
                color=OURS_COLOR,
                alpha=1.0, zorder=2,
            )
            ax.fill(angles, ours_closed, alpha=0.06, color=OURS_COLOR)

        # Model's own trace
        ax.plot(angles, values_closed, "o-", linewidth=2.0,
                markersize=4, color=color, zorder=3)
        ax.fill(angles, values_closed, alpha=0.18, color=color)

        # Annotate numeric values at each vertex (pushed further out)
        if annotate:
            for angle, val in zip(angles[:-1], values):
                ax.text(
                    angle, val + 0.06, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color="#333333",
                )

        # Subplot title (model name). Under usetex matplotlib ignores
        # fontweight, so emphasise through TeX instead.
        title = display or model
        if plt.rcParams["text.usetex"]:
            title = r"\textbf{%s}" % title
        ax.set_title(title, fontsize=10, fontweight="bold", pad=16)

    fig.tight_layout(w_pad=0.5)

    # Save headless PDF (no suptitle — annotations via LaTeX)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path} (headless PDF)")

    # Save labeled PNG (with color fidelity, for preview)
    fig.savefig(str(out_path.with_suffix(".png")), dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path.with_suffix('.png')}")
    plt.close(fig)


def main(
    models: str = DEFAULT_MODELS,
    ratings_dir: str = "dreambench_plus/ratings",
    out_dir: str = "outputs/dreambenchpp",
    plot_type: str = "both",
    annotate: bool = True,
    usetex: bool = True,
):
    """Print the per-category numbers and render the figure.

    Args:
        models: comma-separated model keys. Default is the set used in the paper.
        plot_type: "radar", "bar" or "both".
        annotate: label the value at each radar vertex, as in the paper.
        usetex: render text with LaTeX. Set False if no LaTeX install is present.
    """
    _set_style(usetex)

    model_list = [m.strip() for m in models.split(",")]
    data = load_per_category(model_list, ratings_dir)
    if data.empty:
        raise SystemExit(f"No data loaded. Checked {ratings_dir}/")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- numbers to console ----
    wide = data.pivot(index="model", columns="category_label", values="mean")
    wide = wide.reindex(columns=CATEGORY_LABELS)
    wide = wide.reindex([m for m in model_list if m in wide.index])

    print("Per-category metric--human correlation (Fisher-z over 7 methods)")
    print(f"NearID row scored with {MODEL_ID}\n")
    shown = wide.copy()
    shown.index = [DISPLAY_NAMES.get(m, m) for m in shown.index]
    # Fisher-z over the four categories: the whole-benchmark aggregate.
    shown["All (Fisher-z)"] = [fisher_z_mean(wide.loc[m].values) for m in wide.index]
    print(shown.round(3).to_string())

    if OURS_KEY in wide.index and "SigLIP2-Backbone" in wide.index:
        ours_disp = DISPLAY_NAMES[OURS_KEY]
        base_disp = DISPLAY_NAMES["SigLIP2-Backbone"]
        print(f"\n{ours_disp} minus frozen {base_disp} backbone:")
        for c in list(CATEGORY_LABELS) + ["All (Fisher-z)"]:
            print(f"  {c:16s} {shown.loc[ours_disp, c] - shown.loc[base_disp, c]:+.3f}")

    csv_path = out / "per_category_mh.csv"
    data.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ---- figures ----
    if plot_type in ("radar", "both"):
        plot_radar(data, out / "dbpp_per_category_radar.pdf", annotate=annotate)
    if plot_type in ("bar", "both"):
        plot_grouped_bar(data, out / "dbpp_per_category_bar.pdf")


if __name__ == "__main__":
    Fire(main)
