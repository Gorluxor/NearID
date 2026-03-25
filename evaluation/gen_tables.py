"""
gen_min.py — MCN disc_bidir tables: SSRm / PA pooled per COMMON_METHOD_TAGS group.

Mirrors generate_tables.py load/fix/summarize pipeline but computes only
disc_m_sim_vs_crossii_bidir, then calls export_qwen_reference_table_fixed_methods_pooled
for every group in COMMON_METHOD_TAGS.

Loads from two roots:
  - root         (EncodeID5R): trained encodeid checkpoints + VSM
  - baseline_root (EncodeID5): frozen baselines (CLIP, SigLIP2, DINOv2, Qwen3-VL*)

Models present in root take priority; baseline_root fills in any model not already loaded.

Column mapping (paper ↔ code):
  Paper "SSR" = code SSRm  (AND-based: ALL pairwise margins must be positive)
  Paper "PA"  = code PA    (pooled pairwise accuracy: wins / trials across margins)

  Code "SSR" (OR-based: ANY margin positive) is exported for reference
  but is NOT used in the paper. It was mistakenly used for the paper's PA
  column during initial submission — the correct metric is code PA.

Usage
-----
    python scripts/gen_min.py --overlap primary
    python scripts/gen_min.py --root ./runs/evals/EncodeID5R --baseline_root ./runs/evals/EncodeID5 --overlap primary
    python scripts/gen_min.py --out_path outputs/tables_min --overlap primary
"""

import os
import pandas as pd
from pathlib import Path

from .metrics import (
    load_all_sims,
    fix_vlm_scores,
    add_disc_sim_vs_crossii_bidir_with_overall,
    summarize_margin_family,
)
from .table_utils import (
    COMMON_METHOD_TAGS,
    export_qwen_reference_table_fixed_methods_pooled,
)

DISC_BIDIR = "disc_m_sim_vs_crossii_bidir"
GROUP_BY   = ["method_tag", "sim_model", "mask_type"]

# Display order: baselines first, trained checkpoints sorted alphabetically after.
_BASELINE_ORDER = ["qwen3vl_4b", "qwen3vl_8b", "qwen3vl_30b", "clip", "siglip2", "dinov2", "vsm"]


def _load_and_fix(root: str, split: str, mode: str, verbose: bool) -> pd.DataFrame | None:
    """Load + fix a single root; return None if no CSVs found."""
    try:
        df = load_all_sims(root=root, split=split, mode=mode, verbose=verbose)
    except ValueError as e:
        if verbose:
            print(f"[WARNING] {e} — skipping {root}")
        return None
    df = fix_vlm_scores(df, verbose=verbose)
    # Drop old long-path encodeid rows; keep short tags like MAPInfoNCEExt~3300
    long_mask = df["sim_model"].str.contains(r"runs~trains~", na=False)
    if long_mask.any():
        if verbose:
            print(f"[INFO] {root}: dropping {long_mask.sum()} long-path duplicate encodeid rows.")
        df = df[~long_mask].copy()
    return df


def main(
    root:          str  = "./runs/evals/EncodeID5R",
    baseline_root: str  = "./runs/evals/EncodeID5",
    split:         str  = "testall",
    mode:          str  = "fullneg",
    out_path:      str  = "outputs/tables_min",
    overlap:       str  = None, # type:ignore
    verbose:       bool = True,
):
    """
    Args:
        overlap: How to resolve models that appear in BOTH roots.
                 - "primary":  keep primary root only, drop baseline (old default behavior).
                 - "baseline": keep baseline root only, drop primary.
                 - "union":    keep both and concatenate (use when each root has
                               different sources/mask_types for the same model).
                 - None:       (default) raise an error listing the overlapping models,
                               so you must explicitly choose.
    """
    OUTDIR = Path(out_path)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load primary root (trained checkpoints + VSM)
    # ------------------------------------------------------------------
    df_primary = _load_and_fix(root, split, mode, verbose)
    if df_primary is None:
        raise ValueError(f"No CSVs found in primary root: {root}")

    primary_models = set(df_primary["sim_model"].unique())

    # ------------------------------------------------------------------
    # 2. Load baseline root (CLIP, SigLIP2, DINOv2, Qwen3-VL*, VSM)
    # ------------------------------------------------------------------
    df_baseline = _load_and_fix(baseline_root, split, mode, verbose)

    if df_baseline is not None:
        baseline_models = set(df_baseline["sim_model"].unique())
        overlapping = primary_models & baseline_models

        if overlapping:
            # ── Overlap detected: require explicit resolution ──
            if overlap is None:
                # Build a diagnostic message showing row counts per root
                diag_lines = []
                for m in sorted(overlapping):
                    n_pri = len(df_primary[df_primary["sim_model"] == m])
                    n_bas = len(df_baseline[df_baseline["sim_model"] == m])
                    srcs_pri = sorted(df_primary.loc[df_primary["sim_model"] == m, "method_tag"].unique())
                    srcs_bas = sorted(df_baseline.loc[df_baseline["sim_model"] == m, "method_tag"].unique())
                    diag_lines.append(
                        f"  {m}:\n"
                        f"    primary   ({root}): {n_pri} rows, sources={srcs_pri}\n"
                        f"    baseline  ({baseline_root}): {n_bas} rows, sources={srcs_bas}"
                    )
                diag = "\n".join(diag_lines)
                raise ValueError(
                    f"\n{'='*72}\n"
                    f"OVERLAP: {len(overlapping)} model(s) found in BOTH roots:\n\n"
                    f"{diag}\n\n"
                    f"This causes SILENT DATA LOSS — the baseline data for these models\n"
                    f"is dropped entirely, even if it covers different sources or mask types.\n"
                    f"(This previously caused the Qwen3-VL-30B 2/7 source pooling bug.)\n\n"
                    f"Fix: pass --overlap=<strategy> to choose how to resolve:\n"
                    f"  --overlap=primary   keep primary only, drop baseline (old behavior)\n"
                    f"  --overlap=baseline  keep baseline only, drop primary\n"
                    f"  --overlap=union     keep both (when each root has different sources)\n"
                    f"\n"
                    f"Or remove the stray CSVs from the wrong root.\n"
                    f"{'='*72}"
                )

            overlap = overlap.lower()
            if overlap == "primary":
                if verbose:
                    print(f"\n[OVERLAP] Keeping primary root for: {sorted(overlapping)}")
                df_baseline = df_baseline[~df_baseline["sim_model"].isin(overlapping)]
            elif overlap == "baseline":
                if verbose:
                    print(f"\n[OVERLAP] Keeping baseline root for: {sorted(overlapping)}")
                df_primary = df_primary[~df_primary["sim_model"].isin(overlapping)]
            elif overlap == "union":
                if verbose:
                    print(f"\n[OVERLAP] Keeping union of both roots for: {sorted(overlapping)}")
                # Keep everything — no filtering
            else:
                raise ValueError(f"Unknown --overlap value: {overlap!r}. Must be primary|baseline|union.")

        # Drop baseline models already fully handled
        remaining_overlap = set(df_primary["sim_model"].unique()) & set(df_baseline["sim_model"].unique())
        if remaining_overlap and overlap != "union":
            df_baseline = df_baseline[~df_baseline["sim_model"].isin(remaining_overlap)]

        # Only keep baseline models not in primary
        new_mask = ~df_baseline["sim_model"].isin(set(df_primary["sim_model"].unique()))
        new_baseline = df_baseline[new_mask]
        # For union mode, also keep overlapping rows
        if overlap == "union" and overlapping:
            union_rows = df_baseline[df_baseline["sim_model"].isin(overlapping)]
            df_baseline = pd.concat([new_baseline, union_rows], ignore_index=True).drop_duplicates()
        else:
            df_baseline = new_baseline

        if verbose:
            new_models = sorted(df_baseline["sim_model"].unique())
            print(f"\nBaseline models added from {baseline_root}: {new_models}")

    # ------------------------------------------------------------------
    # 3. Merge
    # ------------------------------------------------------------------
    frames = [df_primary] + ([df_baseline] if df_baseline is not None and not df_baseline.empty else [])
    df = pd.concat(frames, ignore_index=True)

    if verbose:
        print(f"\nsim_models  : {sorted(df['sim_model'].unique())}")
        print(f"method_tags : {sorted(df['method_tag'].unique())}")
        print(f"mask_types  : {sorted(df['mask_type'].unique())}")

    # ------------------------------------------------------------------
    # 4. Compute disc_bidir  (no mask_type pre-filter — mirrors generate_tables.py)
    # ------------------------------------------------------------------
    df = add_disc_sim_vs_crossii_bidir_with_overall(df, prefix=DISC_BIDIR)

    # ------------------------------------------------------------------
    # 5. Summarize per (method_tag, sim_model, mask_type)
    # ------------------------------------------------------------------
    summary = summarize_margin_family(df, prefix=DISC_BIDIR, by=GROUP_BY)

    # ------------------------------------------------------------------
    # 6. Model order: baselines first, trained checkpoints after
    # ------------------------------------------------------------------
    all_models  = sorted(df["sim_model"].unique().tolist())
    baselines   = [m for m in _BASELINE_ORDER if m in all_models]
    checkpoints = sorted(m for m in all_models if m not in set(_BASELINE_ORDER))
    VLM_ORDER  = baselines + checkpoints
    VLM_MODELS = set(VLM_ORDER)

    if verbose:
        print(f"\nTable row order: {VLM_ORDER}")

    # ------------------------------------------------------------------
    # 7. Export one pooled table per COMMON_METHOD_TAGS group
    #    (mirrors generate_tables.py's for _common in COMMON_METHOD_TAGS loop)
    # ------------------------------------------------------------------
    for _common, keep_tags in COMMON_METHOD_TAGS.items():
        _methods_used = "Methods: " + ", ".join(keep_tags) + "."
        export_qwen_reference_table_fixed_methods_pooled(
            summary,
            name=f"disc_bidir_{_common}",
            caption=(
                f"Disc-bidir: pooled (micro over samples) across method tags. "
                f"{_methods_used}"
            ),
            outdir=OUTDIR,
            vlm_order=VLM_ORDER,
            vlm_models=VLM_MODELS,
            keep_method_tags=keep_tags,
        )


if __name__ == "__main__":
    import fire
    fire.Fire(main)
