# metric_utils.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from glob import glob

import numpy as np
import pandas as pd

import re
from typing import Optional, Dict, Any
import os
# -----------------------------
# Optional CSV discovery helpers (as in your notebook)
# -----------------------------

SIMS_ROOT = "outputs/sims"

SIM_MODELS = {
    "clip_base": "openai~clip-vit-base-patch32",
    "clip": "openai~clip-vit-large-patch14",
    "siglip2": "google~siglip2-so400m-patch14-384",
    # VLM (new): note the "vlm:" prefix
    "qwen3vl_4b": "vlm:Qwen~Qwen3-VL-4B-Instruct",
    "qwen3vl_8b": "vlm:Qwen~Qwen3-VL-8B-Instruct",
    "qwen3vl_30b": "vlm:Qwen~Qwen3-VL-30B-A3B-Instruct",
    # TODO: Add later VSM (Mind the glitch), and AlphaCLIP, might need different env
    "dinov2": "facebook~dinov2-large",
    "vsm":  "vsm:vsm",
    # NearID trained checkpoints (R1-R9) — auto-discovered via nearid: prefix
    # Short tags from shorten_model_tag(): "nearid:<ExperimentName>~<step>"
}

# Families whose CSVs should be auto-discovered via glob (no need to list each token)
_AUTO_DISCOVER_FAMILIES = {"encodeid"}

MASK_TYPES = {
    "full": "image",
    "fg": "foreground",
}



def fix_vlm_scores(df: pd.DataFrame, vlm_scale: float = 10.0, rm_cl: list[str] = ["neg_sim"], verbose=False) -> pd.DataFrame:
    """Correct VLM scores from 1-10 scale to 0-1 scale."""
    df = df.loc[:, ~df.columns.str.contains(
        r"_(?:confidence|match_cues|conflict_cues|bg_used)$",
        regex=True
    )]
    # SIM_MODEL will have qwen3vl_4b for example
    vlm_mask = df["sim_model"].astype(str).str.contains(r"(?:^vl:|vl|VL)", case=False, na=False)
    score_cols = df.columns[df.columns.str.match(r"^(sim|cross|neg_sim)_\d\d$")]
    df.loc[vlm_mask, score_cols] = (
        df.loc[vlm_mask, score_cols] # type: ignore
          .apply(pd.to_numeric, errors="coerce")
          .div(vlm_scale)
    )
    col_to_remove = df.columns[df.columns.str.match(r"^(?:" + "|".join(rm_cl) + r")_\d\d$")]
    df = df.drop(columns=col_to_remove)
    if verbose:
        print(f"Removed columns: {list(col_to_remove)}")

    return df


def parse_method_folder(folder_name: str) -> Optional[dict]:
    """
    Supports only:
      - Dataset-Method-Note   e.g. SynCD-Flux-1024
      - Dataset-Method        e.g. SynCD-Flux

    Returns:
      {"dataset": str, "gen_method": str, "note": Optional[str]}
    """
    parts = [p.strip() for p in folder_name.strip().split("-") if p.strip()]
    if len(parts) < 2:
        return None

    dataset = parts[0]
    gen_method = parts[1].lower()
    note = parts[2] if len(parts) >= 3 else None
    return {"dataset": dataset, "gen_method": gen_method, "note": note}


def load_sims_from_folder(
    folder_path: str,
    folder_info: dict,
    mask_types: dict = MASK_TYPES,
    sim_models: dict = SIM_MODELS,
    split: str = "testall",
    mode: str = "full",
) -> List[pd.DataFrame]:
    """
    Load all CSV combinations from a single method folder.

    Supports both:
      - standard similarity CSVs:
          sims_{mask_prefix}_{split}_{mode}_{model_token}.csv
      - VLM similarity CSVs (when sim_models value starts with "vlm:"):
          sims_vlm_{mask_prefix}_{split}_{mode}_{model_token}.csv

    Adds metadata:
      - sim_model, mask_type, gen_method, gen_note, method_tag, source_folder
    """
    dfs: List[pd.DataFrame] = []

    gen_method = str(folder_info.get("gen_method", "")).lower()
    gen_note = folder_info.get("note", None)

    # Stable method tag (never "_None")
    if gen_note is not None and str(gen_note).strip() != "":
        method_tag = f"{gen_method}_{gen_note}"
    else:
        method_tag = f"{gen_method}"

    for mask_short, mask_prefix in mask_types.items():
        for sim_short, sim_pattern in sim_models.items():
            # ---- support VLM patterns via "vlm:" prefix ----
            sim_pattern_str = str(sim_pattern)

            # Support families: sims_{family}_... for any "<family>:<token>"
            m = re.match(r"^(?P<family>[a-zA-Z0-9_]+):(?P<token>.+)$", sim_pattern_str)

            if m:
                family = m.group("family")          # e.g. "vlm" or "vsm"
                model_token = m.group("token")      # e.g. "Qwen~..." or "vsm"
                csv_prefix = f"sims_{family}"       # "sims_vlm" / "sims_vsm"
            else:
                model_token = sim_pattern_str
                csv_prefix = "sims"

            csv_name = f"{csv_prefix}_{mask_prefix}_{split}_{mode}_{model_token}.csv"

            # ------------------------------------------------

            csv_path = os.path.join(folder_path, csv_name)
            if not os.path.exists(csv_path): # This does not error if some CSVs are missing, just skips them
                continue

            df = pd.read_csv(csv_path)

            df["sim_model"] = sim_short
            df["mask_type"] = mask_short

            df["gen_method"] = gen_method
            df["gen_note"] = gen_note
            df["method_tag"] = method_tag

            df["source_folder"] = os.path.basename(folder_path)

            dfs.append(df)

    # --- Auto-discover families (e.g. encodeid) via glob ---
    for family in _AUTO_DISCOVER_FAMILIES:
        for mask_short, mask_prefix in mask_types.items():
            pattern = os.path.join(
                folder_path,
                f"sims_{family}_{mask_prefix}_{split}_{mode}_*.csv",
            )
            for csv_path in sorted(glob(pattern)):
                fname = os.path.basename(csv_path)
                # Extract model_token from:
                #   sims_{family}_{mask}_{split}_{mode}_{model_token}.csv
                prefix_str = f"sims_{family}_{mask_prefix}_{split}_{mode}_"
                if not fname.startswith(prefix_str):
                    continue
                model_token = fname[len(prefix_str):-4]  # strip prefix and .csv

                # Skip if already loaded via explicit SIM_MODELS entry
                explicit_key = f"{family}:{model_token}"
                if explicit_key in sim_models.values():
                    continue

                # Use the model_token as sim_model name (e.g. "MAPInfoNCEExt~3300")
                sim_short = model_token

                df = pd.read_csv(csv_path)
                df["sim_model"] = sim_short
                df["mask_type"] = mask_short
                df["gen_method"] = gen_method
                df["gen_note"] = gen_note
                df["method_tag"] = method_tag
                df["source_folder"] = os.path.basename(folder_path)
                dfs.append(df)

    return dfs

from glob import glob

def load_all_sims(root: str = SIMS_ROOT, split: str = "testall", mode: str = "full", verbose=False) -> pd.DataFrame:
    # check if the folder root exists
    if not os.path.exists(root) or not os.path.isdir(root):
        raise ValueError(f"Root folder {root} does not exist or is not a directory.")
    
    all_dfs: List[pd.DataFrame] = []

    # find all folders inside the root
    all_folders = [f for f in glob(os.path.join(root, "*")) if os.path.isdir(f)]
    if verbose:
        print(f"Found {len(all_folders)} folders in {root}: {[os.path.basename(f) for f in all_folders]}")
    for folder in sorted(all_folders):
        info = parse_method_folder(os.path.basename(folder))
        if info is None:
            continue
        all_dfs.extend(load_sims_from_folder(folder, info, split=split, mode=mode))

    if not all_dfs:
        raise ValueError(f"No CSV files found in {root}")
    return pd.concat(all_dfs, axis=0, ignore_index=True)


# -----------------------------
# Core helpers
# -----------------------------

GROUPS_DEFAULT: List[Tuple[int, int]] = [(0, 1), (0, 2), (1, 2)]


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def try_resolve_sim_col(df: pd.DataFrame, i: int, j: int) -> Optional[str]:
    """
    Return the existing sim column for (i,j) allowing either orientation:
      sim_ij or sim_ji.
    If neither exists, return None.
    """
    a = f"sim_{i}{j}"
    b = f"sim_{j}{i}"
    if a in df.columns:
        return a
    if b in df.columns:
        return b
    return None


def add_sim_vs_cross_bidir_with_overall(
    df: pd.DataFrame,
    groups: Sequence[Tuple[int, int]] = GROUPS_DEFAULT,
    prefix: str = "m_sim_vs_cross_bidir",
    win_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Bidirectional swap margins.

    For each unordered pair (i<j):
      term1 = sim_ij - cross_ij
      term2 = sim_ij - cross_ji

    Adds per-term:
      {prefix}_{ij}_c{ij}_margin, {prefix}_{ij}_c{ij}_win
      {prefix}_{ij}_c{ji}_margin, {prefix}_{ij}_c{ji}_win

    Adds overall per-sample:
      {prefix}_overall_margin = mean(all per-term margins, skipna)
      {prefix}_overall_win    = overall_margin > win_threshold
      {prefix}_n_terms        = count of finite per-term margins
    """
    df = df.copy()
    margin_cols: List[str] = []

    for i, j in groups:
        # We interpret groups as unordered pairs; if sim doesn't exist (e.g., n_pos=2),
        # we skip gracefully.
        sim_col = try_resolve_sim_col(df, i, j)
        if sim_col is None:
            continue

        sim_val = _to_num(df[sim_col])

        # Two directed cross comparisons: cross_ij and cross_ji (each may or may not exist)
        for a, b in [(i, j), (j, i)]:
            cross_col = f"cross_{a}{b}"
            if cross_col not in df.columns:
                continue

            m_col = f"{prefix}_{i}{j}_c{a}{b}_margin"
            w_col = f"{prefix}_{i}{j}_c{a}{b}_win"

            df[m_col] = sim_val - _to_num(df[cross_col])
            # df[w_col] = df[m_col] > win_threshold # replaced to handle NaNs correctly
            df[w_col] = (df[m_col] > win_threshold).astype("boolean")
            df.loc[~np.isfinite(df[m_col].to_numpy(dtype=float)), w_col] = pd.NA

            margin_cols.append(m_col)

    if margin_cols:
        # NOTE(Alex): Initially, had mean, but switched to abs mean to better reflect bidirectional nature
        df[f"{prefix}_overall_margin"] = df[margin_cols].mean(axis=1, skipna=True)
        df[f"{prefix}_overall_amargin"] = df[margin_cols].abs().mean(axis=1, skipna=True)
        # count finite terms per row (useful sanity check)
        df[f"{prefix}_n_terms"] = np.isfinite(df[margin_cols].to_numpy(dtype=float)).sum(axis=1)
        # df[f"{prefix}_overall_win"] = df[f"{prefix}_overall_margin"] > win_threshold
        df[f"{prefix}_overall_win"] = (df[f"{prefix}_overall_margin"] > win_threshold).astype("boolean")
        df.loc[df[f"{prefix}_n_terms"] == 0, f"{prefix}_overall_win"] = pd.NA

    else:
        df[f"{prefix}_overall_margin"] = np.nan
        df[f"{prefix}_n_terms"] = 0
        df[f"{prefix}_overall_win"] = False

    return df

def add_disc_sim_vs_crossii_bidir_with_overall(
    df: pd.DataFrame,
    groups: Sequence[Tuple[int, int]] = GROUPS_DEFAULT,  # [(0,1),(0,2),(1,2)]
    prefix: str = "disc_sim_vs_crossii_bidir",
    win_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Pairwise discriminability, directed (bidirectional) terms.

    For each unordered pair (i<j), define directed terms:
      d_{i->j} = sim_{ij} - cross_{ii}
      d_{j->i} = sim_{ij} - cross_{jj}
    (sim_{ij} is resolved via try_resolve_sim_col, so sim_ji also works.)

    Adds per-term:
      {prefix}_{i}{j}_a{i}_margin, {prefix}_{i}{j}_a{i}_win   (anchor i, direction i->j)
      {prefix}_{i}{j}_a{j}_margin, {prefix}_{i}{j}_a{j}_win   (anchor j, direction j->i)

    Adds per-sample overall:
      {prefix}_overall_margin  = mean(all per-term margins, skipna)
      {prefix}_overall_amargin = mean(abs(per-term margins), skipna)
      {prefix}_n_terms         = count of finite per-term margins
      {prefix}_overall_win     = overall_margin > win_threshold
    """
    df = df.copy()
    margin_cols: List[str] = []

    for i, j in groups:
        sim_col = try_resolve_sim_col(df, i, j)
        if sim_col is None:
            continue

        sim_val = _to_num(df[sim_col])

        # two directed anchors: i (i->j) and j (j->i)
        for anchor in (i, j):
            cross_col = f"cross_{anchor}{anchor}"
            if cross_col not in df.columns:
                continue

            m_col = f"{prefix}_{i}{j}_a{anchor}_margin"
            w_col = f"{prefix}_{i}{j}_a{anchor}_win"

            df[m_col] = sim_val - _to_num(df[cross_col])
            # df[w_col] = df[m_col] > win_threshold # incorrect handling of NaNs, correction below
            df[w_col] = (df[m_col] > win_threshold).astype("boolean")
            df.loc[~np.isfinite(df[m_col].to_numpy(dtype=float)), w_col] = pd.NA
            margin_cols.append(m_col)

    if margin_cols:
        df[f"{prefix}_overall_margin"] = df[margin_cols].mean(axis=1, skipna=True)
        df[f"{prefix}_overall_amargin"] = df[margin_cols].abs().mean(axis=1, skipna=True)
        df[f"{prefix}_n_terms"] = np.isfinite(df[margin_cols].to_numpy(dtype=float)).sum(axis=1)
        # df[f"{prefix}_overall_win"] = df[f"{prefix}_overall_margin"] > win_threshold
        df[f"{prefix}_overall_win"] = (df[f"{prefix}_overall_margin"] > win_threshold).astype("boolean")
        df.loc[df[f"{prefix}_n_terms"] == 0, f"{prefix}_overall_win"] = pd.NA
    else:
        df[f"{prefix}_overall_margin"] = np.nan
        df[f"{prefix}_overall_amargin"] = np.nan
        df[f"{prefix}_n_terms"] = 0
        df[f"{prefix}_overall_win"] = False

    return df



# Backwards-compatible name (your old notebook name), now upgraded to bidirectional:
def add_sim_vs_crossji_with_overall(
    df: pd.DataFrame,
    groups: Sequence[Tuple[int, int]] = GROUPS_DEFAULT,
    prefix: str = "m_sim_vs_crossji",
    win_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    DEPRECATED name retained for convenience.
    Historically this only computed sim_ij - cross_ji for i<j.
    Now it computes BOTH sim_ij-cross_ij and sim_ij-cross_ji (bidirectional swap).
    """
    return add_sim_vs_cross_bidir_with_overall(df, groups=groups, prefix=prefix, win_threshold=win_threshold)


def add_disc_sim_vs_crossii(
    df: pd.DataFrame,
    pos_indices: Sequence[int] = (0, 1, 2),
    prefix: str = "disc_sim_vs_crossii",
    win_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Discriminability margin per i:
      d_i = mean_j!=i(sim_ij) - cross_ii

    Adds:
      {prefix}_{i}_sim_mean
      {prefix}_{i}_margin
      {prefix}_{i}_win

    And overall:
      {prefix}_overall_margin = mean(d_i over i, skipna)
      {prefix}_overall_win    = overall_margin > win_threshold
      {prefix}_n_terms        = count of finite d_i
    """
    df = df.copy()
    margin_cols: List[str] = []

    for i in pos_indices:
        # gather all available sims between i and other positives
        sim_cols: List[str] = []
        for j in pos_indices:
            if j == i:
                continue
            c = try_resolve_sim_col(df, i, j)
            if c is not None:
                sim_cols.append(c)

        cross_col = f"cross_{i}{i}"
        if not sim_cols or cross_col not in df.columns:
            continue

        sim_mean_col = f"{prefix}_{i}_sim_mean"
        m_col = f"{prefix}_{i}_margin"
        w_col = f"{prefix}_{i}_win"

        df[sim_mean_col] = df[sim_cols].apply(_to_num).mean(axis=1, skipna=True)
        df[m_col] = _to_num(df[sim_mean_col]) - _to_num(df[cross_col])
        # df[w_col] = df[m_col] > win_threshold # incorrect handling of NaNs, correction below
        df[w_col] = (df[m_col] > win_threshold).astype("boolean")
        df.loc[~np.isfinite(df[m_col].to_numpy(dtype=float)), w_col] = pd.NA

        margin_cols.append(m_col)

    if margin_cols:
        # NOTE(Alex): Using mean here doesn't care about directionality (for winrate later)
        df[f"{prefix}_overall_margin"] = df[margin_cols].mean(axis=1, skipna=True)
        df[f'{prefix}_overall_amargin'] = df[margin_cols].abs().mean(axis=1, skipna=True)

        df[f"{prefix}_n_terms"] = np.isfinite(df[margin_cols].to_numpy(dtype=float)).sum(axis=1)
        # df[f"{prefix}_overall_win"] = df[f"{prefix}_overall_margin"] > win_threshold
        df[f"{prefix}_overall_win"] = (df[f"{prefix}_overall_margin"] > win_threshold).astype("boolean")
        df.loc[df[f"{prefix}_n_terms"] == 0, f"{prefix}_overall_win"] = pd.NA
    else:
        df[f"{prefix}_overall_margin"] = np.nan
        df[f"{prefix}_n_terms"] = 0
        df[f"{prefix}_overall_win"] = False

    return df



def summarize_margin_family(
    df: pd.DataFrame,
    prefix: str,
    by: Sequence[str] = ("method", "type"),
    id_col: str = "sample_id",
    winrate_as_percent: bool = True,
    ddof: int = 1,
    overall_margin_col: Optional[str] = None,
    overall_win_col: Optional[str] = None,
    harmonic_eps: float = 1e-12,
    add_f1_alias: bool = True,
) -> pd.DataFrame:
    """
    Summarize discriminability margin family for a given prefix.

    Adds two headline rates:
      - SSR (Option A): sample success rate computed from overall_win_col
          SSR = mean_s 1[ mean_t m_{s,t} > 0 ]  (as already encoded in {prefix}_overall_win)
      - PA (Option B): pooled pairwise accuracy over per-term wins
          PA = sum_{s,t} 1[m_{s,t} > 0] / sum_{s} (#valid terms)

    Also adds a combined harmonic mean:
      - H = 2 * SSR * PA / (SSR + PA)
      - optionally an alias column 'F1' (same formula) for convenience.

    Notes:
      - PA uses only non-NA win trials.
      - SSR uses only non-NA overall wins.
      - If winrate_as_percent=True, SSR/PA/H/(F1) are expressed in percent.
    """
    df = df.copy()

    # --- Resolve overall columns ---
    margin_cols = [c for c in df.columns if c.startswith(prefix + "_") and c.endswith("_margin")]

    if overall_margin_col is None:
        overall_margin_col = f"{prefix}_overall_amargin"
    if overall_win_col is None:
        overall_win_col = f"{prefix}_overall_win"

    # overall margin is not a "per-term" margin; remove if present
    if overall_margin_col in margin_cols:
        margin_cols.remove(overall_margin_col)

    missing_overall = [c for c in (overall_margin_col, overall_win_col) if c not in df.columns]
    if missing_overall:
        raise KeyError(
            f"Missing overall cols for prefix='{prefix}': {missing_overall}. "
            f"Available: {[c for c in df.columns if c.startswith(prefix + '_')]}"
        )

    if not margin_cols:
        raise ValueError(
            f"No per-comparison margin cols found for prefix='{prefix}'. "
            f"Expected columns like '{prefix}_..._margin'."
        )

    # --- Collect per-term win columns (paired with margin_cols) ---
    win_cols: List[str] = []
    for m in sorted(margin_cols):
        w = m.replace("_margin", "_win")
        if w not in df.columns:
            raise KeyError(f"Missing win column '{w}' for margin '{m}'")
        win_cols.append(w)

    # --- Row-level pooled wins/trials for PA computation (mask-aware) ---
    # Coerce boolean -> numeric; keep NA as NaN
    w_num = df[win_cols].apply(pd.to_numeric, errors="coerce")
    df["__pa_wins"] = w_num.sum(axis=1, skipna=True)
    df["__pa_trials"] = w_num.notna().sum(axis=1).astype(float)

    # If a row has no valid term-wins, make it NA so it doesn't contribute
    df.loc[df["__pa_trials"] <= 0, ["__pa_wins", "__pa_trials"]] = np.nan

    # SSRm per row: 1.0 if ALL valid terms won, 0.0 if any lost, NaN if no valid terms
    df["__ssr_all"] = np.where(
        df["__pa_trials"].notna(),
        (df["__pa_wins"] == df["__pa_trials"]).astype(float),
        np.nan,
    )

    # --- Aggregations ---
    agg_dict: Dict[str, Any] = {}

    # Per-term summaries (kept for diagnostics)
    for m in sorted(margin_cols):
        w = m.replace("_margin", "_win")
        key = m.replace(f"{prefix}_", "").replace("_margin", "")
        agg_dict[f"{key}_mean"] = (m, "mean")
        agg_dict[f"{key}_median"] = (m, "median")
        agg_dict[f"{key}_std"] = (m, (lambda s, ddof=ddof: s.std(ddof=ddof)))
        agg_dict[f"{key}_winrate"] = (w, "mean")
        agg_dict[f"{key}_winrate_n"] = (w, (lambda s: float(s.notna().sum())))

    # Overall margin stats
    agg_dict["mean_n"] = (overall_margin_col, (lambda s: float(s.notna().sum())))
    agg_dict["mean"] = (overall_margin_col, "mean")
    agg_dict["median"] = (overall_margin_col, "median")
    agg_dict["std"] = (overall_margin_col, (lambda s, ddof=ddof: s.std(ddof=ddof)))

    # SSR (Option A): mean-based — sample wins if mean(margins) > 0
    agg_dict["SSR_n"] = (overall_win_col, (lambda s: float(s.notna().sum())))
    agg_dict["SSR"] = (overall_win_col, "mean")

    # SSRm (Option A-strict): AND-based — sample wins if ALL valid terms > 0
    agg_dict["SSRm_n"] = ("__ssr_all", (lambda s: float(s.notna().sum())))
    agg_dict["SSRm"] = ("__ssr_all", "mean")

    # PA (Option B): pooled wins / pooled trials within group
    agg_dict["PA_wins"] = ("__pa_wins", "sum")
    agg_dict["PA_trials"] = ("__pa_trials", "sum")

    # group size
    agg_dict["n"] = (id_col, "size") if id_col in df.columns else (overall_margin_col, "size")

    summary = df.groupby(list(by), dropna=False).agg(**agg_dict).reset_index()

    # --- Compute group-level PA, H, (F1 alias) ---
    summary["PA"] = summary["PA_wins"] / summary["PA_trials"]
    summary.loc[~np.isfinite(summary["PA"].to_numpy(dtype=float)), "PA"] = np.nan

    # Harmonic mean of SSR and PA
    # (defined on [0,1]; if in percent later, we convert after)
    ssr = summary["SSR"].astype(float)
    pa = summary["PA"].astype(float)
    denom = (ssr + pa).astype(float)

    summary["H"] = np.where(
        np.isfinite(denom.to_numpy()) & (denom.to_numpy() > 0),
        2.0 * ssr * pa / (denom + harmonic_eps),
        np.nan,
    )

    if add_f1_alias:
        summary["F1"] = summary["H"]

    # Hm: harmonic mean of SSRm and PA (strict counterpart to H)
    ssrm_s = summary["SSRm"].astype(float)
    denom_m = (ssrm_s + pa).astype(float)
    summary["Hm"] = np.where(
        np.isfinite(denom_m.to_numpy()) & (denom_m.to_numpy() > 0),
        2.0 * ssrm_s * pa / (denom_m + harmonic_eps),
        np.nan,
    )
    if add_f1_alias:
        summary["F1m"] = summary["Hm"]

    # --- Convert to percent if requested ---
    if winrate_as_percent:
        # Per-term winrates, plus SSR/SSRm/PA/H/Hm/F1/F1m
        for c in summary.columns:
            if c.endswith("_winrate") or c in {"SSR", "SSRm", "PA", "H", "Hm", "F1", "F1m"}:
                summary[c] = 100.0 * summary[c]

    return summary


# -----------------------------
# LaTeX table export (moved here)
# -----------------------------

def df_to_latex(
    df: pd.DataFrame,
    description: str,
    ignore_cols: Sequence[str] = ("n",),
    float_digits: int = 4,
    winrate_digits: int = 1,
    label: Optional[str] = None,
    index: bool = False,
    wrap_table: bool = True,
    pretty: bool = True,
    siunitx: bool = True,
    arraystretch: float = 1.15,
    tabcolsep_pt: int = 8,
    font_cmd: str = r"\small",
) -> str:
    out = df.copy().drop(columns=[c for c in ignore_cols if c in df.columns], errors="ignore")

    winrate_cols = {c for c in out.columns if "winrate" in str(c).lower() or "n" == str(c).lower()}
    # int_cols = {c for c in out.columns if pd.api.types.is_integer_dtype(out[c])} # TODO: fix if its int to not have .000
    formatters: Dict[str, Any] = {}
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            d = winrate_digits if c in winrate_cols else float_digits
            formatters[c] = (lambda x, d=d: "" if pd.isna(x) else f"{float(x):.{d}f}")

    tabular = out.to_latex(index=index, escape=True, formatters=formatters).strip()

    pretty_block = ""
    if pretty:
        pretty_block = (
            f"\\renewcommand{{\\arraystretch}}{{{arraystretch}}}\n"
            f"\\setlength{{\\tabcolsep}}{{{tabcolsep_pt}pt}}\n"
        )
        if font_cmd:
            pretty_block += f"{font_cmd}\n"

    if pretty and siunitx:
        col_formats = []
        for c in out.columns:
            if pd.api.types.is_numeric_dtype(out[c]):
                if c in winrate_cols:
                    col_formats.append(f"S[table-format=3.{winrate_digits}]")
                else:
                    col_formats.append(f"S[table-format=1.{float_digits}]")
            else:
                col_formats.append("l")
        new_spec = "".join(col_formats)

        tabular = re.sub(
            r"\\begin\{tabular\}\{[^}]*\}",
            r"\\begin{tabular}{" + new_spec + r"}",
            tabular,
            count=1,
        )

        # wrap numeric headers so siunitx renders them as text
        lines = tabular.splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if line.strip() == r"\toprule":
                header_idx = i + 1
                break
        if header_idx is not None and header_idx < len(lines):
            header_line = lines[header_idx]
            if "&" in header_line and r"\\" in header_line:
                cells = [c.strip() for c in header_line.split("&")]
                for k, col in enumerate(out.columns):
                    if pd.api.types.is_numeric_dtype(out[col]):
                        cells[k] = "{" + cells[k].replace(r"\\", "").strip() + "}"
                if not cells[-1].endswith(r"\\"):
                    cells[-1] = cells[-1] + r" \\"
                lines[header_idx] = " & ".join(cells)
                tabular = "\n".join(lines)

    if not wrap_table:
        return (pretty_block + tabular) if pretty else tabular
    cap = ""
    if description:
        _desc = description.replace("_", r"\_")
        cap = f"\\caption{{{_desc}}}" # note: not f-string to avoid escaping issues
    lab = f"\\label{{{label}}}" if label else ""

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"{cap}\n"
        f"{lab}\n"
        f"{pretty_block}"
        f"{tabular}\n"
        "\\end{table}\n"
    )


# -----------------------------
# Scalar summaries (for W&B or logs)
# -----------------------------

def compute_scalar_summaries(
    df: pd.DataFrame,
    prefix: str = "summary",
    include_prefixes: Tuple[str, ...] = ("sim_", "cross_", "neg_sim_", "m_", "disc_"),
    ddof: int = 0,
    harmonic_eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute global scalar summaries for numeric metric-like columns and win columns.
    - Numeric cols: mean/std/min/max/n
    - Bool '*_win' cols: winrate (%), n

    Additionally logs (if available) the paired SSR/PA/H/F1 for discriminability-style families:
      - SSR: mean({family}_overall_win)
      - PA: pooled mean over term wins in that family (sum wins / sum trials)
      - H and F1: harmonic mean of SSR and PA

    Returns a flat dict suitable for W&B logging.
    """
    scalars: Dict[str, float] = {f"{prefix}/n_rows": float(len(df))}

    # ---- standard per-column summaries ----
    for c in df.columns:
        name = str(c)
        if not name.startswith(include_prefixes):
            continue

        s = df[c]

        if name.endswith("_win"):
            winrate = float(pd.to_numeric(s, errors="coerce").mean() * 100.0)
            scalars[f"{prefix}/{name}_winrate"] = winrate
            scalars[f"{prefix}/{name}_n"] = float(s.notna().sum())
            continue

        x = pd.to_numeric(s, errors="coerce").astype(float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue

        scalars[f"{prefix}/{name}_mean"] = float(x.mean())
        scalars[f"{prefix}/{name}_std"] = float(x.std(ddof=ddof))
        scalars[f"{prefix}/{name}_min"] = float(x.min())
        scalars[f"{prefix}/{name}_max"] = float(x.max())
        scalars[f"{prefix}/{name}_n"] = float(len(x))

    # ---- NEW: detect discriminability families and compute SSR/PA/H/F1 ----
    # Family key is everything up to "_overall_win"
    overall_win_cols = [c for c in df.columns if str(c).endswith("_overall_win")]
    for ow in overall_win_cols:
        ow_name = str(ow)

        # only apply to included prefixes
        if not ow_name.startswith(include_prefixes):
            continue

        family = ow_name[: -len("_overall_win")]  # e.g. "disc_m_sim_vs_crossii_bidir"
        term_win_cols = [
            c for c in df.columns
            if str(c).startswith(family + "_") and str(c).endswith("_win") and str(c) != ow_name
        ]
        if not term_win_cols:
            continue

        # SSR: sample-level overall success
        ssr = pd.to_numeric(df[ow], errors="coerce")
        ssr_val = float(ssr.mean())  # in [0,1]
        ssr_n = float(ssr.notna().sum())

        # PA: pooled over valid term wins
        w_num = df[term_win_cols].apply(pd.to_numeric, errors="coerce")
        wins = float(w_num.sum(axis=1, skipna=True).sum(skipna=True))
        trials = float(w_num.notna().sum(axis=1).sum())
        pa_val = (wins / trials) if trials > 0 else float("nan")

        # harmonic mean of SSR and PA
        denom = ssr_val + pa_val
        h_val = (2.0 * ssr_val * pa_val / (denom + harmonic_eps)) if (np.isfinite(denom) and denom > 0) else float("nan")

        # SSRm: AND-based — sample wins if ALL valid terms > 0
        row_wins  = w_num.sum(axis=1, skipna=True)
        row_trials = w_num.notna().sum(axis=1)
        valid_rows = row_trials > 0
        ssrm_val = (
            float((row_wins[valid_rows] == row_trials[valid_rows]).sum()) / float(valid_rows.sum())
            if valid_rows.any() else float("nan")
        )
        ssrm_n = float(valid_rows.sum())

        # Hm: harmonic mean of SSRm and PA
        hm_denom = ssrm_val + pa_val
        hm_val = (
            (2.0 * ssrm_val * pa_val / (hm_denom + harmonic_eps))
            if np.isfinite(hm_denom) and hm_denom > 0 else float("nan")
        )

        # log under a stable key
        scalars[f"{prefix}/{family}/SSR"] = 100.0 * ssr_val
        scalars[f"{prefix}/{family}/SSR_n"] = ssr_n
        scalars[f"{prefix}/{family}/SSRm"]   = 100.0 * ssrm_val if np.isfinite(ssrm_val) else float("nan")
        scalars[f"{prefix}/{family}/SSRm_n"] = ssrm_n
        scalars[f"{prefix}/{family}/PA"] = 100.0 * pa_val if np.isfinite(pa_val) else float("nan")
        scalars[f"{prefix}/{family}/PA_trials"] = trials
        scalars[f"{prefix}/{family}/H"] = 100.0 * h_val if np.isfinite(h_val) else float("nan")
        scalars[f"{prefix}/{family}/Hm"]    = 100.0 * hm_val if np.isfinite(hm_val) else float("nan")
        scalars[f"{prefix}/{family}/F1"] = 100.0 * h_val if np.isfinite(h_val) else float("nan")

    return scalars


# -----------------------------
# W&B helper (imports wandb only when used)
# -----------------------------

def wandb_log_full_df_and_summaries(
    run,
    df: pd.DataFrame,
    table_key: str = "results/full_df",
    include_prefixes: Tuple[str, ...] = ("sim_", "cross_", "neg_sim_", "m_", "disc_"),
    step: Optional[int] = None,
) -> Dict[str, float]:
    if run is None:
        return {}

    import wandb  # local import by design

    payload = {
        table_key: wandb.Table(dataframe=df),
    }
    wandb.log(payload, step=step)

    res = compute_scalar_summaries(df, prefix="latex", include_prefixes=include_prefixes)
    wandb.log(res, step=step)

    return res


# -----
# Example notebook usage
# -----
# import pandas as pd
# from metric_utils import (
#     load_all_sims,
#     add_sim_vs_crossji_with_overall,   # now bidirectional (upgraded)
#     add_disc_sim_vs_crossii,
#     summarize_margin_family,
#     df_to_latex,
# )

# df = load_all_sims()
# df["method"] = df["sim_model"]
# df["type"]   = df["mask_type"]

# # Apply metrics
# df = add_sim_vs_crossji_with_overall(df, prefix="m_sim_vs_crossji")  # now includes both cross_ij and cross_ji
# df = add_disc_sim_vs_crossii(df, prefix="disc_sim_vs_crossii")

# # Summaries
# GROUP_BY = ["gen_method", "gen_steps", "sim_model", "mask_type"]
# summary_swap = summarize_margin_family(df, prefix="m_sim_vs_crossji", by=GROUP_BY)
# summary_disc = summarize_margin_family(df, prefix="disc_sim_vs_crossii", by=GROUP_BY)

# Example wandb main script usage
# import pandas as pd
# from metric_utils import (
#     add_sim_vs_crossji_with_overall,
#     add_disc_sim_vs_crossii,
#     wandb_log_full_df_and_summaries,
# )

# df = pd.DataFrame(results)

# # add derived metrics before logging
# df = add_sim_vs_crossji_with_overall(df, prefix="m_sim_vs_crossji")
# df = add_disc_sim_vs_crossii(df, prefix="disc_sim_vs_crossii")

# # log exactly one table + scalars
# wandb_log_full_df_and_summaries(run, df, table_key="results/full_df")

