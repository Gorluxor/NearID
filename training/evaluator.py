import logging
import pandas as pd
import numpy as np
import wandb
from pathlib import Path
import torch

from evaluation.inference import NearIDInference
from evaluation.metrics import (
    add_sim_vs_crossji_with_overall,
    add_disc_sim_vs_crossii,
    add_disc_sim_vs_crossii_bidir_with_overall,
    summarize_margin_family,
    wandb_log_full_df_and_summaries
)
from evaluation.table_utils import latex_tables_by_sim_model

logger = logging.getLogger(__name__)

# v1 = NaN→False (legacy); v2 = NaN-aware wins, both SSR (mean-based) and SSRm (AND-based)
_MTG_METRIC_VERSION = 2

def _calculate_oracle(obj_mask_pil, part_mask_pil):
    """Calculates ratio and oracle score from PIL masks."""
    if obj_mask_pil is None or part_mask_pil is None:
        return 1.0, 0.0 # Fallback if masks are missing
        
    obj_arr = np.array(obj_mask_pil.convert("L")) > 0
    part_arr = np.array(part_mask_pil.convert("L")) > 0
    
    sum_obj = obj_arr.sum()
    sum_part = part_arr.sum()
    
    # Avoid division by zero; if part is 0, ratio is 1.0
    ratio = float(sum_part / sum_obj) if sum_obj > 0 else 1.0
    oracle = 1.0 - ratio
    return ratio, oracle

def log_latex_artifact(step: int, split: str, outdir: Path) -> None:
    """
    Logs all .tex files under outdir as a W&B artifact.
    Works without Accelerate.
    """
    tex_files = sorted(outdir.rglob("*.tex"))
    wandb.log({f"eval/{split}/latex_tex_n": len(tex_files)}, step=step)

    if not tex_files:
        print(f"[eval] No .tex files found under: {outdir}")
        return

    art = wandb.Artifact(
        name=f"latex_tables_{split}",
        type="latex_tables",
        metadata={"split": split, "step": step, "dir": str(outdir)},
    )

    # step-aware path inside artifact (prevents overwrite in UI)
    for fp in tex_files:
        art.add_file(str(fp), name=f"step_{step}/{fp.name}")
    assert wandb.run is not None, "W&B run context is not active. Cannot log artifact."
    wandb.run.log_artifact(art, aliases=[f"step_{step}", "latest"])
    art.wait()
    print(f"[eval] Logged {len(tex_files)} .tex files to W&B artifact: {art.name}")



class IdentityEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir) / "eval_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, checkpoint_path: str, dataloader, step: int, split_name: str, neg_name: str):
        # 1. Run Inference Engine
        engine = NearIDInference(checkpoint_path, device="cuda")
        data = engine.get_embeddings(dataloader)
        
        embs = data["embeddings"] 
        p_mask = data["pos_mask"]
        n_mask = data["neg_mask"]
        meta = data["metadata"]
        
        results = []
        for i in range(len(embs)):
            # 2. Build detailed Row
            row = {
                **meta[i],
                "source_folder": neg_name,
                "sim_model": self.cfg.backbone,
                "mask_type": "full",
                "gen_method": self.cfg.wandb.name,
                "method_tag": f"step_{step}",
                "n_pos": int(p_mask[i].sum()),
                "n_neg": int(n_mask[i].sum())
            }
            # import pdb; pdb.set_trace()  # Debugging checkpoint to verify data loading and row construction
            # 3. Calculate Sim/Cross pairs (Dot products on normalized embs = Cosine Sim)
            # Positive Pairs
            for r in range(3):
                for c in range(3):
                    if r == c: continue
                    if p_mask[i, r] and p_mask[i, c]:
                        row[f"sim_{r}{c}"] = float((embs[i, r] * embs[i, c]).sum())
                    else:
                        row[f"sim_{r}{c}"] = np.nan
            # --- CROSS_IJ (Pos vs Neg Pairs) ---
            # Pos slots: 0, 1, 2 | Neg slots: 3, 4, 5
            # Cross-modality (Pos vs Neg)
            for r in range(3):
                for c in range(3):
                    neg_idx = c + 3
                    if p_mask[i, r] and n_mask[i, c]:
                        row[f"cross_{r}{c}"] = float((embs[i, r] * embs[i, neg_idx]).sum())
                    else:
                        row[f"cross_{r}{c}"] = np.nan
            
            results.append(row)

        df = pd.DataFrame(results)

        # Save raw similarities CSV (same column layout as sim_test.py output) for debugging
        csv_dir = self.output_dir / f"step_{step}"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"sims_{split_name}_{neg_name}.csv"
        # Keep only the similarity columns (no derived metric columns yet)
        sim_cols = [c for c in df.columns if c.startswith("sim_") or c.startswith("cross_")]
        meta_cols = ["sample_id", "category", "n_pos", "n_neg", "source_folder", "sim_model", "mask_type", "method_tag"]
        df[[c for c in meta_cols + sim_cols if c in df.columns]].to_csv(csv_path, index=False)
        logger.info("Saved raw sims CSV: %s (%d rows)", csv_path, len(df))

        # 4. Metric Aggregation
        df = add_sim_vs_crossji_with_overall(df, prefix="m_sim_vs_crossji")
        df = add_disc_sim_vs_crossii(df, prefix="disc_sim_vs_crossii")
        df = add_disc_sim_vs_crossii_bidir_with_overall(df, prefix="disc_m_sim_vs_crossii_bidir")

        GROUP_BY = ["method_tag", "sim_model"]
        df_to_log = df.replace({pd.NA: None}) 
        summary = summarize_margin_family(df, prefix="disc_m_sim_vs_crossii_bidir", by=GROUP_BY)

        # 5. Logging
        table_key = f"eval/{split_name}/{neg_name}/full"
        table_final = wandb_log_full_df_and_summaries(wandb.run, df_to_log, table_key=table_key, step=step)
        
        # Export LaTeX for reports
        table_path = self.output_dir / f"step_{step}"
        latex_tables_by_sim_model(
            summary, 
            f"{split_name}_{neg_name}_report", 
            f"Eval {split_name} {neg_name} Step {step}", 
            table_path, 
            ["sim_model", "method_tag", "mean", "median", "PA", "SSR", "SSRm"]
        )
        
        return summary, table_final
    

def process(processor, img, dtype, return_tensors="pt", device="cuda"):
    return processor(images=img, return_tensors=return_tensors).pixel_values.to(device, dtype)

class MTGOnlineEvaluator:
    """Evaluates NearID on the MTG Dataset during the training loop."""
    def __init__(self, split="test"):
        from datasets import load_dataset
        self.dataset = load_dataset("abdo-eldesokey/mtg-dataset", split=split)

    @torch.inference_mode()
    def run(self, checkpoint_path: str, step: int, device: str = "cuda"):
        from inference import NearIDInference
        import torch.nn.functional as F
        from tqdm.auto import tqdm
        import wandb
        # Uses the same aligned inference engine
        engine = NearIDInference(checkpoint_path, device=device)
        model = engine.model.to(device)  # type: ignore
        processor = model.processor
        dtype = next(model.parameters()).dtype # Match precision (Bfloat16/Fp16)

        results = []
        for i, row in enumerate(tqdm(self.dataset, desc="MTG Online Eval", leave=False)):
            r1, o1 = _calculate_oracle(row['image_1_object_mask'], row['image_1_part_mask'])
            r2, o2 = _calculate_oracle(row['image_2_object_mask'], row['image_2_part_mask'])
            
            # Batch process for speed
            batch_pv = torch.cat(
                [process(processor, row[k], dtype, device=device) for 
                k in ['image_1_original', 'image_2_original', 'image_1_inpainted', 'image_2_inpainted']], 
                dim=0)
            device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
            with torch.autocast(device_type=device_type, dtype=dtype):
                # Pass directly to model's forward
                embs = F.normalize(model({"pixel_values_anchor": batch_pv}, side="anchor"), p=2, dim=-1)

            # 4. Extract 5 target similarities
            sim_1_2 = float((embs[0] * embs[1]).sum())
            sim_1_1p = float((embs[0] * embs[2]).sum())
            sim_1_2p = float((embs[0] * embs[3]).sum())
            sim_2_2p = float((embs[1] * embs[3]).sum())
            sim_2_1p = float((embs[1] * embs[2]).sum())

            # 5. Margins: sim(1,2) vs sim(1,1p) and sim(1,2) vs sim(2,2p)
            m1 = sim_1_2 - sim_1_1p
            m2 = sim_1_2 - sim_2_2p

            # NaN-aware wins: pd.NA excluded from SSR/PA; NaN > 0 = False is wrong
            _win1 = pd.NA if np.isnan(m1) else bool(m1 > 0)
            _win2 = pd.NA if np.isnan(m2) else bool(m2 > 0)
            # SSRm: sample wins if ALL *valid* terms won (partial rows included, aligned with MCN)
            _valid_wins = [w for w in (_win1, _win2) if not pd.isna(w)]
            _overall = pd.NA if len(_valid_wins) == 0 else bool(all(_valid_wins))
            results.append({
                "sample_id": i,
                "mtg_ratio_1": r1, "mtg_oracle_1": o1,
                "mtg_ratio_2": r2, "mtg_oracle_2": o2,
                "mtg_sim_1_2": sim_1_2,
                "mtg_sim_1_1p": sim_1_1p,
                "mtg_sim_1_2p": sim_1_2p,
                "mtg_sim_2_2p": sim_2_2p,
                "mtg_sim_2_1p": sim_2_1p,
                "mtg_margin_1": m1,
                "mtg_margin_2": m2,
                "mtg_win_1": _win1,
                "mtg_win_2": _win2,
                "mtg_overall_win": _overall,
            })

        df = pd.DataFrame(results)
        
        # --- Pearson Correlations with Oracle ---
        # Helper to safely calculate correlation and handle NaNs if variance is zero
        def _safe_corr(col_sim, col_oracle):
            r = df[col_sim].corr(df[col_oracle])
            return float(r) if pd.notna(r) else 0.0

        r_1_1p = _safe_corr("mtg_sim_1_1p", "mtg_oracle_1")
        r_2_2p = _safe_corr("mtg_sim_2_2p", "mtg_oracle_2")
        r_1_2p = _safe_corr("mtg_sim_1_2p", "mtg_oracle_2")
        r_2_1p = _safe_corr("mtg_sim_2_1p", "mtg_oracle_1")

        # Fisher's z-transform for mathematically sound averaging
        rs = np.array([r_1_1p, r_2_2p, r_1_2p, r_2_1p])
        rs_clipped = np.clip(rs, -0.999, 0.999) # Prevent inf in arctanh
        mean_r = float(np.tanh(np.mean(np.arctanh(rs_clipped))))

        # Fisher's z-transform for mathematically sound averaging of only r_1_1p and r_2_2p (strictly sim vs its own oracle)
        rs_strict = np.array([r_1_1p, r_2_2p])
        rs_strict_clipped = np.clip(rs_strict, -0.999, 0.999)
        r_strict_mean = float(np.tanh(np.mean(np.arctanh(rs_strict_clipped))))

        # --- NaN-aware SSR / SSRm / PA (metric_version=2) ---
        # SSRm (AND-based): sample wins if ALL *valid* margins > 0 (partial rows included, aligned with MCN)
        ssrm = float(pd.to_numeric(df["mtg_overall_win"], errors="coerce").mean()) * 100.0
        # SSR (mean-based, skipna=True → aligned with MCN): sample wins if mean of available margins > 0
        _m1 = pd.to_numeric(df["mtg_margin_1"], errors="coerce")
        _m2 = pd.to_numeric(df["mtg_margin_2"], errors="coerce")
        _mean_marg = pd.DataFrame({"m1": _m1, "m2": _m2}).mean(axis=1, skipna=True)
        _ssr_wins = np.where(_mean_marg.notna(), (_mean_marg > 0).astype(float), np.nan)
        ssr = float(np.nanmean(_ssr_wins)) * 100.0 if not np.all(np.isnan(_ssr_wins)) else float("nan")
        # PA (pooled): total wins / total valid trials (NaN excluded from denominator)
        _w1 = pd.to_numeric(df["mtg_win_1"], errors="coerce")
        _w2 = pd.to_numeric(df["mtg_win_2"], errors="coerce")
        _total_wins = float(_w1.sum(skipna=True) + _w2.sum(skipna=True))
        _total_trials = float(_w1.notna().sum() + _w2.notna().sum())
        pa = (_total_wins / _total_trials * 100.0) if _total_trials > 0 else float("nan")

        # Hm: harmonic mean of SSRm and PA
        _hm_s = ssrm / 100.0
        _hm_p = pa / 100.0
        _hm_d = _hm_s + _hm_p
        hm = (2.0 * _hm_s * _hm_p / (_hm_d + 1e-12)) * 100.0 if (np.isfinite(_hm_d) and _hm_d > 0) else float("nan")

        # Validity breakdown
        _w1v = pd.to_numeric(df["mtg_win_1"], errors="coerce").notna()
        _w2v = pd.to_numeric(df["mtg_win_2"], errors="coerce").notna()
        n_total        = len(df)
        n_both_valid   = int((_w1v & _w2v).sum())
        n_partial      = int((_w1v ^ _w2v).sum())
        n_both_invalid = n_total - n_both_valid - n_partial
        if n_both_invalid > 0 or n_partial > 0:
            logger.warning(
                "MTG: %d samples have invalid margins (both_invalid=%d, partial=%d). "
                "This is unexpected for embedding-based evaluation.",
                n_both_invalid + n_partial, n_both_invalid, n_partial,
            )

        # Support counts
        _ssr_n   = int(np.sum(~np.isnan(_ssr_wins)))
        _ssrm_n  = int(pd.to_numeric(df["mtg_overall_win"], errors="coerce").notna().sum())
        _pa_trials = int(_total_trials)
        if _ssr_n != _ssrm_n:
            logger.warning("SSR_n (%d) != SSRm_n (%d) — unexpected (both should count rows with ≥1 valid term)", _ssr_n, _ssrm_n)

        # Correlation N per pair
        n_11p = int(df[["mtg_sim_1_1p", "mtg_oracle_1"]].dropna().shape[0])
        n_22p = int(df[["mtg_sim_2_2p", "mtg_oracle_2"]].dropna().shape[0])
        n_12p = int(df[["mtg_sim_1_2p", "mtg_oracle_2"]].dropna().shape[0])
        n_21p = int(df[["mtg_sim_2_1p", "mtg_oracle_1"]].dropna().shape[0])

        summary = {
            "eval/MTG/Corr_Mean": mean_r,
            "eval/MTG/Corr_Strict_Mean": r_strict_mean,
            "eval/MTG/Corr_1_1p": r_1_1p,
            "eval/MTG/Corr_2_2p": r_2_2p,
            "eval/MTG/Corr_1_2p": r_1_2p,
            "eval/MTG/Corr_2_1p": r_2_1p,
            "eval/MTG/SSRm": ssrm,           # AND-based (v2; previously the only "SSR")
            "eval/MTG/SSR": ssr,             # mean-based (v2, consistent with MCN)
            "eval/MTG/PA": pa,
            "eval/MTG/Hm": hm,
            "eval/MTG/SSR_n": _ssr_n,
            "eval/MTG/SSRm_n": _ssrm_n,
            "eval/MTG/PA_trials": _pa_trials,
            "eval/MTG/N_total": n_total,
            "eval/MTG/N_both_valid": n_both_valid,
            "eval/MTG/N_partial": n_partial,
            "eval/MTG/Corr_N_11p": n_11p,
            "eval/MTG/Corr_N_22p": n_22p,
            "eval/MTG/Corr_N_12p": n_12p,
            "eval/MTG/Corr_N_21p": n_21p,
            "eval/MTG/metric_version": _MTG_METRIC_VERSION,
        }
        
        if wandb.run is not None:
            wandb.log({**summary, f"eval/MTG/table_step_{step}": wandb.Table(dataframe=df)}, step=step)

        return summary, df