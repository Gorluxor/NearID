from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# expects df_to_latex in your metric_util module
from .metrics import df_to_latex


COMMON_METHOD_TAGS = {
    "flux_1024_only": ["flux_1024"],
    "qwen_methodtag_avg": ["flux", "flux_1024", "powerpaint", "sdxl", "sdxl_1024"],
    "flux_and_sdxl_1k": ["flux_1024", "sdxl_1024"],
    "all": ["flux", "flux_1024", "powerpaint", "sdxl", "sdxl_1024", "qwen", "qwen_1328"],
    "low_res": ["flux", "qwen", "sdxl"],
    "full_res": ["flux_1024", "sdxl_1024", "qwen_1328"],
    "TOT": ["flux", "flux_1024", "qwen", "qwen_1328", "powerpaint", "sdxl", "sdxl_1024", "fluxc", "fluxc_1024", "sdxl_1024"],
    "OTH": ["powerpaint", "sdxl", "sdxl_1024", "fluxc", "fluxc_1024", "sdxl_1024"],
    "PP": ["powerpaint"],
    "SDXL": ["sdxl", "sdxl_1024"],
    "Flux": ["flux", "flux_1024"],
    "Qwen": ["qwen", "qwen_1328"],
    "FluxC": ["fluxc", "fluxc_1024"],
}


def latex_tables_by_sim_model(
    summary_df: pd.DataFrame,
    name_prefix: str,
    caption_prefix: str,
    outdir: Path,
    cols: list[str],
    sort_cols: list[str] | None = None,
    group_col: str = "sim_model",
):
    if not isinstance(outdir, Path):
        outdir = Path(outdir)
        
    outdir.mkdir(parents=True, exist_ok=True)

    if sort_cols is None:
        # sensible default if columns exist
        sort_cols = [c for c in [group_col, "method_tag", "mask_type"] if c in summary_df.columns]

    # keep only available columns
    keep_cols = [c for c in cols if c in summary_df.columns]
    if group_col not in keep_cols:
        keep_cols = [group_col] + keep_cols

    d = summary_df.loc[:, keep_cols].copy()
    if sort_cols:
        d = d.sort_values([c for c in sort_cols if c in d.columns])

    for sm, g in d.groupby(group_col, dropna=False, observed=True):
        g2 = g.drop(columns=[group_col], errors="ignore")

        tex = df_to_latex(
            g2,
            description=f"{caption_prefix} ({sm})",
            ignore_cols=(),
            label=f"tab:{name_prefix}_{sm}",
        )
        (outdir / f"{name_prefix}_{sm}.tex").write_text(tex, encoding="utf-8")
        csv_dir = outdir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        g2.to_csv(csv_dir / f"{name_prefix}_{sm}.csv", index=False)


def export_qwen_reference_table_fixed_methods(
    summary_df: pd.DataFrame,
    name: str,
    caption: str,
    outdir: Path,
    cols: list[str] | None = None,
    sort_by: list[str] | None = None,
    vlm_order: list[str] | None = None,
    vlm_models: set[str] | None = None,
    keep_method_tags: list[str] | None = None,
) -> pd.DataFrame:
    """
    Micro-style export: filter/sort the existing summary (per method_tag rows),
    then write LaTeX.

    summary_df must contain at least:
      method_tag, sim_model, mask_type, and your chosen numeric columns (mean/std/winrate,...)
    """
    if sort_by is None:
        sort_by = ["sim_model", "method_tag", "mask_type"]
    if cols is None:
        cols = ["sim_model", "mask_type", "method_tag", "n", "mean", "median", "std", "winrate"]
        cols = [c for c in cols if c in summary_df.columns]

    d = summary_df.copy()
    d = d.dropna(subset=["method_tag", "sim_model", "mask_type"])

    if vlm_order is not None:
        d["sim_model"] = pd.Categorical(d["sim_model"].astype(str), categories=vlm_order, ordered=True)

    if vlm_models is not None:
        d = d.loc[d["sim_model"].astype(str).isin(vlm_models)]

    if keep_method_tags is not None:
        d = d.loc[d["method_tag"].astype(str).isin(keep_method_tags)]

    tbl = d.loc[:, [c for c in cols if c in d.columns]].sort_values(sort_by)

    tex = df_to_latex(
        tbl,
        description=caption,
        ignore_cols=(),
        label=f"tab:{name}",
    )
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{name}.tex").write_text(tex, encoding="utf-8")
    csv_dir = outdir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(csv_dir / f"{name}.csv", index=False)
    return tbl


def _macro_mix_std(means: np.ndarray, stds: np.ndarray, ddof: int = 0) -> float:
    """
    Equal-weight mixture std from per-method means/stds:
      Var = E[Var_within] + Var_between = mean(stds^2) + var(means)
    """
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    msk = np.isfinite(means) & np.isfinite(stds)
    means, stds = means[msk], stds[msk]
    if means.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(stds ** 2) + np.var(means, ddof=ddof)))


def export_qwen_reference_table_fixed_methods_macro(
    summary_df: pd.DataFrame,
    name: str,
    caption: str,
    outdir: Path,
    cols: list[str] | None = None,
    sort_by: list[str] | None = None,
    vlm_order: list[str] | None = None,
    vlm_models: set[str] | None = None,
    keep_method_tags: list[str] | None = None,
) -> pd.DataFrame:
    """
    Macro-style export: aggregate across method_tag (equal weight per method_tag),
    producing one row per (mask_type, sim_model), then write LaTeX.

    Requires summary_df to contain:
      method_tag, sim_model, mask_type, and overall columns: mean, median, std, winrate, n (n optional)
    """
    if sort_by is None:
        sort_by = ["mask_type", "sim_model"]
    if cols is None:
        cols = ["mask_type", "sim_model", "mean", "median", "std", "winrate", "n_methods", "n_total"]

    d = summary_df.copy()
    d = d.dropna(subset=["method_tag", "sim_model", "mask_type"])

    # enforce sim model ordering if requested
    if vlm_order is not None:
        d["sim_model"] = pd.Categorical(d["sim_model"].astype(str), categories=vlm_order, ordered=True)

    # filters
    if vlm_models is not None:
        d = d.loc[d["sim_model"].astype(str).isin(vlm_models)]
    if keep_method_tags is not None:
        d = d.loc[d["method_tag"].astype(str).isin(keep_method_tags)]

    # ensure numeric
    for c in ["n", "mean", "median", "std", "winrate"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    def agg_block(g: pd.DataFrame) -> pd.Series:
        means = g["mean"].to_numpy(dtype=float)
        meds = g["median"].to_numpy(dtype=float) if "median" in g else np.full(len(g), np.nan)
        stds = g["std"].to_numpy(dtype=float) if "std" in g else np.full(len(g), np.nan)
        wins = g["winrate"].to_numpy(dtype=float) if "winrate" in g else np.full(len(g), np.nan)

        return pd.Series({
            "n_methods": int(g["method_tag"].nunique()),
            "n_total": float(np.nansum(g["n"].to_numpy(dtype=float))) if "n" in g else float(len(g)),
            "mean": float(np.nanmean(means)),
            "median": float(np.nanmedian(meds)),
            "std": _macro_mix_std(means, stds, ddof=0),
            "winrate": float(np.nanmean(wins)),
        })

    tbl = (
        d.groupby(["mask_type", "sim_model"], dropna=False, observed=True)
         .apply(agg_block, include_groups=False)
         .reset_index()
         .sort_values(sort_by)
    )

    tbl_out = tbl.loc[:, [c for c in cols if c in tbl.columns]]

    tex = df_to_latex(
        tbl_out,
        description=caption,
        ignore_cols=(),
        label=f"tab:{name}",
    )
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{name}.tex").write_text(tex, encoding="utf-8")
    csv_dir = outdir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    tbl_out.to_csv(csv_dir / f"{name}.csv", index=False)
    return tbl_out





def _pooled_std_from_groups(n, means, stds, ddof=1):
    n = np.asarray(n, dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    mask = np.isfinite(n) & np.isfinite(means) & np.isfinite(stds) & (n > 0)
    n, means, stds = n[mask], means[mask], stds[mask]
    if n.size == 0:
        return np.nan

    N = n.sum()
    if N <= ddof:
        return np.nan

    grand_mean = np.sum(n * means) / N
    ss_within = np.sum((n - ddof) * (stds ** 2))
    ss_between = np.sum(n * (means - grand_mean) ** 2)

    var = (ss_within + ss_between) / (N - ddof)
    return float(np.sqrt(var))

import numpy as np
import pandas as pd

def export_qwen_reference_table_fixed_methods_pooled(
    summary_df: pd.DataFrame,
    name: str,
    caption: str,
    outdir,
    vlm_order: list[str],
    vlm_models: set[str],
    keep_method_tags: list[str],
    sort_by=None,
    hide_n: bool = True,
    cols: list[str] | None = None,
):
    """
    Pooled/micro over method_tag using only summary rows, but for new headline metrics:
      - SSR pooled by SSR_n
      - PA pooled by PA_trials

    Output rows: (mask_type, sim_model).

    Formatting rule for export:
      - if value > 0 -> round to 1 decimal
      - else         -> round to 3 decimals
    """
    if cols is None:
        # include supports so pooling is well-defined / debuggable
        cols = ["SSRm", "PA", "SSR", "SSRm_n", "PA_trials", "SSR_n", "n"]
    if sort_by is None:
        sort_by = ["mask_type", "sim_model"]

    d = summary_df.copy()
    d = d.dropna(subset=["method_tag", "sim_model", "mask_type"])

    d = d[d["sim_model"].astype(str).isin(vlm_models)].copy()
    d = d[d["method_tag"].astype(str).isin(keep_method_tags)].copy()
    d["sim_model"] = pd.Categorical(
        d["sim_model"].astype(str), categories=vlm_order, ordered=True
    )

    # coerce numeric where present
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    def _wavg(x: np.ndarray, w: np.ndarray) -> float:
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        return float(np.sum(w[m] * x[m]) / np.sum(w[m])) if np.any(m) else np.nan

    def agg_block(g: pd.DataFrame) -> pd.Series:
        # supports
        ssr_n = (
            g["SSR_n"].to_numpy(dtype=float)
            if "SSR_n" in g
            else np.full(len(g), np.nan)
        )
        ssrm_n = (
            g["SSRm_n"].to_numpy(dtype=float)
            if "SSRm_n" in g
            else np.full(len(g), np.nan)
        )
        pa_tr = (
            g["PA_trials"].to_numpy(dtype=float)
            if "PA_trials" in g
            else np.full(len(g), np.nan)
        )

        # headline metrics (likely already in percent)
        ssr = (
            g["SSR"].to_numpy(dtype=float)
            if "SSR" in g
            else np.full(len(g), np.nan)
        )
        ssrm = (
            g["SSRm"].to_numpy(dtype=float)
            if "SSRm" in g
            else np.full(len(g), np.nan)
        )
        pa = (
            g["PA"].to_numpy(dtype=float)
            if "PA" in g
            else np.full(len(g), np.nan)
        )

        ssr_pool = _wavg(ssr, ssr_n)
        ssrm_pool = _wavg(ssrm, ssrm_n)
        pa_pool = _wavg(pa, pa_tr)

        # optional: keep a total n (old) if present
        n_total = (
            float(np.nansum(g["n"].to_numpy(dtype=float)))
            if "n" in g
            else float(len(g))
        )

        return pd.Series(
            {
                "n": n_total,
                "SSRm": ssrm_pool,
                "PA": pa_pool,
                "SSR": ssr_pool,
                # keep pooled supports for sanity checks (you can drop later if you want)
                "SSRm_n": float(np.nansum(ssrm_n)),
                "PA_trials": float(np.nansum(pa_tr)),
                "SSR_n": float(np.nansum(ssr_n)),
            }
        )

    tbl = (
        d.groupby(["mask_type", "sim_model"], dropna=False, observed=True)
        .apply(agg_block, include_groups=False)
        .reset_index()
        .sort_values(sort_by)
    )

    # what to show in the final table
    tbl_out = tbl.copy()
    if hide_n:
        tbl_out = tbl_out.drop(columns=["n"], errors="ignore")

    # --------- dynamic rounding for export (ONLY affects LaTeX output) ----------
    def _fmt_dynamic(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        x = float(x)
        return f"{x:.1f}" if x > 0 else f"{x:.3f}"

    # Apply to metric columns you want formatted this way
    for c in ["SSRm", "SSR", "PA"]:
        if c in tbl_out.columns:
            tbl_out[c] = tbl_out[c].map(_fmt_dynamic)

    # Optional: render supports as integers (comment out if you prefer floats)
    for c in ["SSRm_n", "SSR_n", "PA_trials", "n"]:
        if c in tbl_out.columns:
            tbl_out[c] = tbl_out[c].map(lambda v: "" if pd.isna(v) else f"{int(round(float(v)))}")
    # --------------------------------------------------------------------------

    tex = df_to_latex(
        tbl_out,
        description=caption,
        ignore_cols=(),
        label=f"tab:{name}",
    )
    (outdir / f"{name}.tex").write_text(tex, encoding="utf-8")
    csv_dir = outdir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(csv_dir / f"{name}.csv", index=False)  # raw numeric (pre-formatting)
    print(f"Exported {name} with {len(tbl_out)} rows to {outdir / f'{name}.tex'} + csv/")
    return tbl_out
