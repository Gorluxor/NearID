from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn.functional as dist_nn

"""
Distributed Metric & Ranking Losses Implementation
==================================================

This module provides production-ready, distributed-aware implementations of state-of-the-art 
loss functions for Metric Learning, Contrastive Learning, and Preference Optimization.

All losses inherit from `StandardizedLoss` and support:
1. Distributed Training (Gradients preserved across GPUs via torch.distributed.nn.functional).
2. Tiered/Extended Variants (Enforcing hierarchy: Pos > Hard Neg > Unrelated).
3. Automatic Input Scaling & Normalization.

References & Implementations
----------------------------

| Loss Variant              | Class Name(s)                                    | Paper / ArXiv                                | Reference Code (GitHub)                      |
| :------------------------ | :----------------------------------------------- | :------------------------------------------- | :------------------------------------------- |
| **ImageReward** | `BaselineImageRewardLoss`<br>`ExtendedImageRewardLoss` | [ImageReward: Learning Human Preferences](https://arxiv.org/abs/2304.05977) | [THUDM/ImageReward](https://github.com/THUDM/ImageReward) |
| **InfoNCE / CLIP** | `BaselineInfoNCELoss`<br>`ExtendedInfoNCELoss`<br>`SymmetricContrastiveLoss` | [Learning Transferable Visual Models (CLIP)](https://arxiv.org/abs/2103.00020) | [OpenAI/CLIP](https://github.com/openai/CLIP) |
| **Circle Loss** | `BaselineCircleLoss`<br>`ExtendedCircleLoss`     | [Circle Loss: A Unified Perspective](https://arxiv.org/abs/2002.10857) | [TinyZeaMays/CircleLoss](https://github.com/TinyZeaMays/CircleLoss) |
| **Triplet Margin** | `BaselineTripletMarginLoss`<br>`ExtendedTripletMarginLoss` | [FaceNet: A Unified Embedding](https://arxiv.org/abs/1503.03832) | [adambielski/siamese-triplet](https://github.com/adambielski/siamese-triplet) |
| **SigLIP** | `BaselineSigLIPLoss`<br>`ExtendedSigLIPLoss`     | [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) | [google-research/big_vision](https://github.com/google-research/big_vision) |
| **ArcFace** | `BaselineAngularContrastiveLoss`<br>`ExtendedAngularContrastiveLoss` | [ArcFace: Additive Angular Margin Loss](https://arxiv.org/abs/1801.07698) | [deepinsight/insightface](https://github.com/deepinsight/insightface) |

Usage
-----
    criterion = BaselineInfoNCELoss()
    output = criterion(
        anchor=ref_embeddings, 
        positive=good_gen_embeddings, 
        negative=bad_gen_embeddings
    )
    loss = output.loss
    log_dict(output.metrics)
"""


# ==============================================================================
# OBSERVATIONS & GUIDELINES FOR BATCH NEGATIVES & RANKING
# ==============================================================================
#
# 1. MECHANISM: BATCH NEGATIVES
# -----------------------------
# All losses (except ImageReward) use `gather_with_grad` to access embeddings 
# from ALL GPUs. This creates a "Global Batch" of negatives.
# - Local Batch: B  ->  Global Batch: B * World_Size.
# - Implicit Negatives: For any anchor, all other (Global - 1) samples are 
#   treated as "Batch Negatives".
# - CRITICAL FOR ACCELERATE: You must use `gather_with_grad` manually in the 
#   loss, or gradients will not flow between GPUs.
#
# 2. SETTING A: CLIP-Like Baseline (0 Hard Negatives)
# ---------------------------------------------------
# Goal: Pull Positive close (Sim->1), push all Batch Negatives away (Sim->0).
# Input: negative=None
#
# - InfoNCE / SigLIP / ArcFace / Circle (Baseline & Extended):
#   These classes automatically detect `negative=None` and fall back to standard 
#   contrastive learning (Anchor vs Global Batch). This is the correct, 
#   defensible baseline for "No Hard Negatives".
#
# - Triplet (Euclidean/Original):
#   WARNING: The standard Euclidean implementation fails with `negative=None` 
#   because it requires explicit negatives. 
#   FIX: Use `BaselineCosineTripletLoss` or `ExtendedCosineTripletLoss` which 
#   support "Batch Hard Mining" (automatically finding the hardest negative 
#   in the global batch).
#
# 3. SETTING B: Tiered Ranking (Pos > Hard Neg > Batch Neg)
# ---------------------------------------------------------
# Goal: Enforce hierarchy: Sim(Pos) > Sim(HardNeg) > Sim(BatchNeg).
#       Ideal Scores: Pos (~1.0) > HardNeg (~0.5) > BatchNeg (~0.0).
# Input: negative=Hard_Neg_Embeddings
#
# - Extended InfoNCE / SigLIP:
#   Best choices. They explicitly penalize if Hard Negatives are not ranked 
#   higher than the general Batch distribution.
#   * InfoNCE_Ext: Uses `logaddexp` to maintain probability hierarchy.
#   * SigLIP_Ext: Uses `logsigmoid` to force raw score separation.
#
# - Extended Circle / ArcFace / Triplet:
#   Valid choices. They use margin-based ranking (ReLU) to enforce the gap.
#   * Circle_Ext: Requires careful margin tuning for Tier 2.
#   * Triplet_Ext (Cosine): Explicitly calculates Sim(Hard) vs Sim(Batch).
#
# 4. SCALING & METRICS (CLIP-Like Interface)
# ------------------------------------------
# To ensure consistent [-1, 1] scores regardless of loss magnitude:
# - Input: Apply L2 Normalization to all embeddings before the loss.
# - Metric: Use Cosine Similarity (Dot Product of normalized vectors).
#
# Range Interpretation:
#   1.0 : Identity / Perfect Match
#   0.0 : Orthogonal / Unrelated (Random Batch Negative)
#  -1.0 : Opposite
# ==============================================================================




# ==============================================================================
# Utils & Distributed Helpers
# ==============================================================================

def _is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """All-gather that preserves gradients (for Embeddings)."""
    if not _is_distributed():
        return x
    return torch.cat(dist_nn.all_gather(x), dim=0)

def gather_tensor(x: torch.Tensor) -> torch.Tensor:
    """All-gather without gradients (for Masks/Indices)."""
    if not _is_distributed():
        return x
    world_size = torch.distributed.get_world_size()
    tensor_list = [torch.zeros_like(x) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, x)
    return torch.cat(tensor_list, dim=0)

# def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     return x / (x.norm(dim=-1, keepdim=True) + eps)

def l2_normalize(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)

@dataclass
class LossOutput:
    loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]

class StandardizedLoss(nn.Module):
    """
    Base class enforcing a unified interface for all losses.
    
    Arguments Mapping Guide:
    -------------------------------------------------------
    Metric Losses (InfoNCE, Circle, Triplet, ArcFace):
        anchor:   (B, D)    Reference Embedding
        positive: (B, P, D) Positive Candidate Embeddings
        negative: (B, K, D) Hard Negative Candidate Embeddings
    
    Ranking Losses (ImageReward):
        anchor:   (B, P)    Scores for Positive Candidates (Ref <-> Good)
        positive: (B, K)    Scores for Hard Neg Candidates (Ref <-> Bad)
        negative: (B, M)    Scores for Unrelated Candidates (Ref <-> Random)
    -------------------------------------------------------
    """
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: Optional[torch.Tensor] = None, 
                pos_mask: Optional[torch.Tensor] = None, 
                neg_mask: Optional[torch.Tensor] = None,
                eps: float = 1e-7, 
                **kwargs) -> LossOutput:
        raise NotImplementedError

# ==============================================================================
# SECTION A: ImageReward (Score-Based)
# ==============================================================================

class BaselineImageRewardLoss(StandardizedLoss):
    def __init__(self, margin: float = 0.0, scale: float = 1.0, **kwargs):
        super().__init__()
        self.margin = margin
        self.scale = scale # Internal scaling to handle 0-1 inputs if needed

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: Optional[torch.Tensor] = None, 
                pos_mask: Optional[torch.Tensor] = None, 
                neg_mask: Optional[torch.Tensor] = None, 
                eps: float = 1e-7,
                **kwargs) -> LossOutput:
        
        # Mapping: anchor -> PosScores, positive -> NegScores
        pos_scores = anchor * self.scale
        neg_scores = positive * self.scale
        
        B, P = pos_scores.shape
        _, K = neg_scores.shape
        if pos_mask is None: pos_mask = torch.ones((B, P), device=pos_scores.device)
        if neg_mask is None: neg_mask = torch.ones((B, K), device=neg_scores.device)
        pos_mask, neg_mask = pos_mask.bool(), neg_mask.bool()

        diff = pos_scores.unsqueeze(-1) - neg_scores.unsqueeze(1)
        loss_matrix = -F.logsigmoid(diff - self.margin)
        mask = pos_mask.unsqueeze(-1) & neg_mask.unsqueeze(1)
        
        loss = torch.tensor(0.0, device=pos_scores.device, requires_grad=True)
        valid = mask.float().sum()
        if valid > 0:
            loss = (loss_matrix * mask.float()).sum() / valid

        with torch.no_grad():
            acc = ((diff > 0).float() * mask.float()).sum() / (valid + eps)
        return LossOutput(loss=loss, metrics={"irm/loss": loss.detach(), "irm/acc": acc})

class BaselineEmbeddingRewardLoss(BaselineImageRewardLoss):
    """Wraps BaselineImageRewardLoss to accept embeddings by computing cosine similarity first."""
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: Optional[torch.Tensor] = None,
                pos_mask: Optional[torch.Tensor] = None, 
                neg_mask: Optional[torch.Tensor] = None,
                eps: float = 1e-7,
                **kwargs) -> LossOutput:
        
        # 1. Normalize embeddings
        anchor_norm = F.normalize(anchor if anchor.dim() == 3 else anchor.unsqueeze(1), p=2, dim=-1)
        pos_norm = F.normalize(positive, p=2, dim=-1)
        
        # Compute positive scores: Sim(Anchor, Positive) -> (B, P)
        pos_scores = torch.matmul(anchor_norm, pos_norm.transpose(1, 2)).squeeze(1)
        
        neg_scores = None
        # 2. Compute negative scores
        if negative is not None:
            # Case A: Explicit hard negatives -> Sim(Anchor, HardNegative) -> (B, K)
            neg_norm = F.normalize(negative, p=2, dim=-1)
            neg_scores = torch.matmul(anchor_norm, neg_norm.transpose(1, 2)).squeeze(1)
        else:
            # Case B: No hard negatives -> Fallback to global batch negatives
            global_pos = gather_with_grad(pos_norm)
            flat_global = global_pos.view(-1, anchor_norm.shape[-1])
            neg_scores = torch.matmul(anchor_norm.squeeze(1), flat_global.T) # (B, M)
            
            # Create a mask to exclude self-positives from the batch negatives
            B, P = pos_scores.shape
            rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
            global_ids = torch.arange(flat_global.shape[0], device=anchor.device).unsqueeze(0).expand(B, -1)
            start_idx = (rank * B + torch.arange(B, device=anchor.device)).unsqueeze(1) * P
            own_mask = (global_ids >= start_idx) & (global_ids < start_idx + P)
            
            # Push self-positive scores to -inf so they are ignored, and update the neg_mask
            neg_scores = neg_scores.masked_fill(own_mask, -1.0) 
            neg_mask = ~own_mask if neg_mask is None else neg_mask & ~own_mask

        # 3. Pass the 2D scores to the original IRM loss
        # Note: 'anchor' maps to pos_scores and 'positive' maps to neg_scores in the baseline
        return super().forward(
            anchor=pos_scores, 
            positive=neg_scores, 
            negative=None, 
            pos_mask=pos_mask, 
            neg_mask=neg_mask, 
            eps=eps, 
            **kwargs
        )

class ExtendedImageRewardLoss(StandardizedLoss):
    def __init__(self, margin_quality: float = 0.0, 
                 margin_relevance: float = 1.0, scale: float = 1.0, **kwargs):
        super().__init__()
        self.m1 = margin_quality
        self.m2 = margin_relevance
        self.scale = scale

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: Optional[torch.Tensor] = None,
                pos_mask: Optional[torch.Tensor] = None, 
                neg_mask: Optional[torch.Tensor] = None,
                eps: float = 1e-7,
                unrelated_mask: Optional[torch.Tensor] = None, **kwargs) -> LossOutput:
        
        # Mapping: anchor -> PosScores, positive -> HardNegScores, negative -> UnrelatedScores
        pos_scores = anchor * self.scale
        hard_neg_scores = positive * self.scale
        unrelated_scores = negative * self.scale if negative is not None else None
        
        B, P = pos_scores.shape
        _, K = hard_neg_scores.shape
        if pos_mask is None: pos_mask = torch.ones((B, P), device=pos_scores.device)
        if neg_mask is None: neg_mask = torch.ones((B, K), device=hard_neg_scores.device)
        pos_mask, neg_mask = pos_mask.bool(), neg_mask.bool()

        # Tier 1: Quality (Pos > Hard)
        diff_q = pos_scores.unsqueeze(-1) - hard_neg_scores.unsqueeze(1)
        loss_q = -F.logsigmoid(diff_q - self.m1)
        mask_q = pos_mask.unsqueeze(-1) & neg_mask.unsqueeze(1)
        
        l_qual = torch.tensor(0.0, device=pos_scores.device, requires_grad=True)
        if mask_q.any():
            l_qual = (loss_q * mask_q.float()).sum() / mask_q.sum()

        # Tier 2: Relevance (Hard > Unrelated)
        l_rel = torch.tensor(0.0, device=pos_scores.device, requires_grad=True)
        metrics = {}
        if unrelated_scores is not None:
            _, M = unrelated_scores.shape
            if unrelated_mask is None: unrelated_mask = torch.ones((B, M), device=unrelated_scores.device)
            unrelated_mask = unrelated_mask.bool()
            
            diff_r = hard_neg_scores.unsqueeze(-1) - unrelated_scores.unsqueeze(1)
            loss_r = -F.logsigmoid(diff_r - self.m2)
            mask_r = neg_mask.unsqueeze(-1) & unrelated_mask.unsqueeze(1)
            
            if mask_r.any():
                l_rel = (loss_r * mask_r.float()).sum() / mask_r.sum()
            
            metrics["irm_ext/rel_acc"] = ((diff_r > 0).float() * mask_r.float()).sum() / (mask_r.sum() + eps)

        total = l_qual + l_rel
        metrics.update({
            "irm_ext/loss": total.detach(), 
            "irm_ext/qual_acc": ((diff_q > 0).float() * mask_q.float()).sum() / (mask_q.sum() + eps)
        })
        return LossOutput(loss=total, metrics=metrics)


class ExtendedEmbeddingRewardLoss(ExtendedImageRewardLoss):
    """Wraps ExtendedImageRewardLoss to accept embeddings by computing cosine similarity first."""
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: Optional[torch.Tensor] = None,
                pos_mask: Optional[torch.Tensor] = None, 
                neg_mask: Optional[torch.Tensor] = None,
                eps: float = 1e-7,
                unrelated_mask: Optional[torch.Tensor] = None, **kwargs) -> LossOutput:
        
        # 1. Normalize and compute cosine similarity scores
        anchor_norm = F.normalize(anchor if anchor.dim() == 3 else anchor.unsqueeze(1), p=2, dim=-1)
        
        pos_norm = F.normalize(positive, p=2, dim=-1)
        pos_scores = torch.matmul(anchor_norm, pos_norm.transpose(1, 2)).squeeze(1) # (B, P)
        
        hard_neg_scores = None
        if negative is not None:
            neg_norm = F.normalize(negative, p=2, dim=-1)
            hard_neg_scores = torch.matmul(anchor_norm, neg_norm.transpose(1, 2)).squeeze(1) # (B, K)
            
        # 2. Pass the 2D scores to the original IRM loss
        return super().forward(
            anchor=pos_scores, 
            positive=hard_neg_scores,  # type: ignore
            negative=None, # Update if you have an unrelated pool 
            pos_mask=pos_mask, 
            neg_mask=neg_mask, 
            eps=eps, 
            unrelated_mask=unrelated_mask, 
            **kwargs
        )

# ==============================================================================
# SECTION B: InfoNCE / Contrastive
# ==============================================================================

class BaselineInfoNCELoss(StandardizedLoss):
    def __init__(self, temperature: float = 0.07, **kwargs):
        super().__init__()
        self.T = temperature

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: Optional[torch.Tensor] = None, 
                pos_mask: Optional[torch.Tensor] = None, 
                neg_mask: Optional[torch.Tensor] = None,
                eps: float = 1e-7, 
                  **kwargs) -> LossOutput:
        
        # 1. Normalize
        if anchor.dim() == 2: anchor = anchor.unsqueeze(1)
        anchor = l2_normalize(anchor)       # (B, 1, D)
        pos_emb = l2_normalize(positive)    # (B, P, D)
        
        B, P, D = pos_emb.shape
        if pos_mask is None: pos_mask = torch.ones((B, P), device=anchor.device)
        pos_mask = pos_mask.bool()

        # 2. Global Gather (Positives become the "classes")
        # flat_global shape: (WorldSize * B * P, D)
        global_pos = gather_with_grad(pos_emb)
        flat_global = global_pos.view(-1, D)
        
        # 3. Compute Logits
        # anchor: (B, 1, D) x flat_global.T: (D, Global) -> (B, 1, Global)
        logits_all = torch.matmul(anchor, flat_global.T).squeeze(1) / self.T
        
        logits_hard = None
        if negative is not None:
            neg_emb = l2_normalize(negative)
            # anchor: (B, 1, D) x neg_emb.T: (D, K, B) -> batch matmul? No.
            # neg_emb.transpose(1, 2) is (B, D, K).
            # matmul result is (B, 1, K). Squeeze to (B, K).
            logits_hard = torch.matmul(anchor, neg_emb.transpose(1, 2)).squeeze(1) / self.T

        # 4. Calculate Loss
        rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
        losses = []

        # Vectorization Note: We can iterate P because P is usually small (1-5).
        # Fully vectorizing P implies flattening B*P, which complicates the "logits_hard" appending logic.
        # Keeping the loop over P is acceptable performance-wise.
        for p in range(P):
            valid_p = pos_mask[:, p]
            if not valid_p.any(): continue
            
            # Combine Batch Negatives + Hard Negatives
            current_logits = logits_all
            if logits_hard is not None:
                current_logits = torch.cat([logits_all, logits_hard], dim=1)
            
            # Target Indexing
            # The positive for (batch b, index p) is located at global index:
            target_idx = (rank * B + torch.arange(B, device=anchor.device)) * P + p
            
            losses.append(F.cross_entropy(current_logits[valid_p], target_idx[valid_p]))

        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=anchor.device, requires_grad=True)
        return LossOutput(loss=loss, metrics={"infonce/loss": loss.detach()})


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

# assumes you already have:
# - StandardizedLoss
# - LossOutput
# - l2_normalize(x)
# - gather_with_grad(x)
# - _neg_inf(x): returns a scalar "negative infinity" value suitable for x.dtype (e.g., finfo.min)

class ExtendedInfoNCELoss(StandardizedLoss):
    """
    Tier-1: multi-positive InfoNCE over (global positives) + (optional hard negatives)
    Tier-2: hard negatives vs batch-negative pool (global positives excluding own positives)

    Adds monitoring metrics:
      - Tier-1 top1 acc per-positive + mean
      - Tier-1 pos_minus_maxneg (with own positives excluded from maxneg)
      - Tier-1 max softmax prob / entropy (temperature sanity)
      - Tier-1 pos_logit_mean / max_logit_mean (scale sanity)
      - Tier-2 tier_acc, avg_hard_sim, avg_lse_batch (as you had)
      - counts (global_pos_count, hard_neg_count)
    """
    def __init__(self, temperature: float = 0.07, alpha: float = 0.5, **kwargs):
        super().__init__()
        self.T = float(temperature)
        self.alpha = float(alpha)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        **kwargs,
    ) -> "LossOutput":

        # --------- Pre-computation ----------
        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(1)  # (B,1,D)
        anchor = l2_normalize(anchor)      # (B,1,D)
        pos_emb = l2_normalize(positive)   # (B,P,D)

        B, P, D = pos_emb.shape

        if pos_mask is None:
            pos_mask = torch.ones((B, P), device=anchor.device, dtype=torch.bool)
        else:
            pos_mask = pos_mask.bool()

        # Global gather of positives (DDP-safe, keeps grad)
        global_pos = gather_with_grad(pos_emb)          # (G_B, P, D)
        flat_global = global_pos.view(-1, D)            # (M, D), M = (world*B)*P

        # Similarities to global positives
        sim_batch = (anchor.squeeze(1) @ flat_global.T) / self.T   # (B, M)

        # Similarities to hard negatives (optional)
        sim_hard = None
        K = 0
        if negative is not None:
            neg_emb = l2_normalize(negative)  # (B,K,D)
            _, K, _ = neg_emb.shape
            if neg_mask is None:
                neg_mask = torch.ones((B, K), device=anchor.device, dtype=torch.bool)
            else:
                neg_mask = neg_mask.bool()
            sim_hard = (anchor @ neg_emb.transpose(1, 2)).squeeze(1) / self.T  # (B, K)

        # DDP rank for target indexing + own-positive masking
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        # Build mask for "own positives" positions in flat_global: indices [start, start+P)
        # This is used for metrics (maxneg) and Tier-2 (batch negatives).
        # Note: assumes gather order is rank-major and consistent with your original implementation.
        M = flat_global.shape[0]
        global_ids = torch.arange(M, device=anchor.device).unsqueeze(0).expand(B, -1)  # (B,M)
        start_idx = (rank * B + torch.arange(B, device=anchor.device)).unsqueeze(1) * P  # (B,1)
        own_mask = (global_ids >= start_idx) & (global_ids < (start_idx + P))  # (B,M)

        ext_log: Dict[str, torch.Tensor] = {}
        ext_log["infonce_ext/global_pos_count"] = torch.tensor(M, device=anchor.device)
        ext_log["infonce_ext/hard_neg_count"] = torch.tensor(K, device=anchor.device)

        # --------- Tier 1: Pos vs (Batch + Hard) ----------
        losses_main = []
        top1_list = []

        for p in range(P):
            valid_p = pos_mask[:, p]
            if not valid_p.any():
                continue

            # target index in flattened global positives
            target = (rank * B + torch.arange(B, device=anchor.device)) * P + p  # (B,)

            # logits used by the actual loss
            logits_full = sim_batch
            if sim_hard is not None:
                logits_full = torch.cat([sim_batch, sim_hard], dim=1)  # (B, M+K)

            # compute CE loss for this positive index
            losses_main.append(F.cross_entropy(logits_full[valid_p], target[valid_p]))

            # -------- monitoring (no grad) --------
            with torch.no_grad():
                # top-1 accuracy (matching the objective logits)
                pred = logits_full[valid_p].argmax(dim=1)
                acc_p = (pred == target[valid_p]).float().mean()
                ext_log[f"infonce_ext/main_top1_acc_p{p}"] = acc_p
                top1_list.append(acc_p)

                # pos logit stats
                pos_logit = logits_full[valid_p, target[valid_p]]
                ext_log[f"infonce_ext/pos_logit_mean_p{p}"] = pos_logit.mean()

                # softmax peakedness / entropy (temperature sanity)
                probs = torch.softmax(logits_full[valid_p], dim=1)
                ext_log[f"infonce_ext/main_maxprob_p{p}"] = probs.max(dim=1).values.mean()
                ent = -(probs * (probs + eps).log()).sum(dim=1).mean()
                ext_log[f"infonce_ext/main_entropy_p{p}"] = ent

                # pos_minus_maxneg: define max "negative" from batch by masking own positives
                # (and if hard negatives exist, include them too)
                # Batch part:
                batch_logits = sim_batch[valid_p].masked_fill(own_mask[valid_p], _neg_inf(sim_batch))
                max_batch_neg = batch_logits.max(dim=1).values

                if sim_hard is not None:
                    max_hard_neg = sim_hard[valid_p].max(dim=1).values
                    max_neg = torch.maximum(max_batch_neg, max_hard_neg)
                else:
                    max_neg = max_batch_neg

                ext_log[f"infonce_ext/pos_minus_maxneg_p{p}"] = (pos_logit - max_neg).mean()

                # logit scale sanity (cheap)
                ext_log[f"infonce_ext/logits_max_mean_p{p}"] = logits_full[valid_p].max(dim=1).values.mean()

        if losses_main:
            loss_main = torch.stack(losses_main).mean()
        else:
            loss_main = torch.tensor(0.0, device=anchor.device, requires_grad=True)

        if top1_list:
            ext_log["infonce_ext/main_top1_acc"] = torch.stack(top1_list).mean()

        # --------- Tier 2: Hard Negatives vs Batch Negatives ----------
        loss_tier = torch.tensor(0.0, device=anchor.device, requires_grad=True)

        if sim_hard is not None and neg_mask is not None and neg_mask.any():
            # mask out own positives from batch pool
            neg_min = _neg_inf(sim_batch)
            sim_batch_masked = sim_batch.masked_fill(own_mask, neg_min)  # (B,M)

            # lse over batch negatives
            lse_batch = torch.logsumexp(sim_batch_masked, dim=1)  # (B,)
            lse_batch_expanded = lse_batch.unsqueeze(1).expand(-1, K)  # (B,K)

            # NLL for hard-vs-batchpool
            nll_matrix = -sim_hard + torch.logaddexp(sim_hard, lse_batch_expanded)  # (B,K)

            valid_k_count = neg_mask.sum()
            if valid_k_count > 0:
                loss_tier = (nll_matrix * neg_mask.float()).sum() / valid_k_count

            with torch.no_grad():
                ext_log["infonce_ext/tier_loss"] = loss_tier.detach()
                ext_log["infonce_ext/tier_acc"] = (
                    ((sim_hard > lse_batch_expanded).float() * neg_mask.float()).sum() / (valid_k_count + eps)
                )
                ext_log["infonce_ext/avg_hard_sim"] = (
                    (sim_hard * neg_mask.float()).sum() / (valid_k_count + eps)
                )
                ext_log["infonce_ext/avg_lse_batch"] = (
                    (lse_batch_expanded * neg_mask.float()).sum() / (valid_k_count + eps)
                )

                # optional: how often hard negatives beat the best batch negative (more interpretable)
                max_batch_neg = sim_batch_masked.max(dim=1).values  # (B,)
                max_batch_neg_exp = max_batch_neg.unsqueeze(1).expand(-1, K)
                ext_log["infonce_ext/hard_gt_maxbatch_frac"] = (
                    (((sim_hard > max_batch_neg_exp) & neg_mask).float().sum()) / (valid_k_count + eps)
                )

        # --------- Final ----------
        total = loss_main + self.alpha * loss_tier

        return LossOutput(
            loss=total,
            metrics={
                "infonce_ext/loss": total.detach(),
                "infonce_ext/main": loss_main.detach(),
                "infonce_ext/tier": loss_tier.detach(),
                **ext_log,
            },
        )


class SymmetricContrastiveLoss(StandardizedLoss):
    def __init__(self, temperature: float = 0.07, **kwargs):
        super().__init__()
        self.t2i = BaselineInfoNCELoss(temperature)
        self.T = temperature

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: Optional[torch.Tensor] = None, 
                pos_mask: Optional[torch.Tensor] = None, 
                neg_mask: Optional[torch.Tensor] = None, 
                eps: float = 1e-7, 
                **kwargs) -> LossOutput:
        
        pos_emb, neg_emb = positive, negative
        out1 = self.t2i(anchor, pos_emb, neg_emb, pos_mask, neg_mask)
        
        if anchor.dim() == 2: anchor = anchor.unsqueeze(1)
        anchor_norm = l2_normalize(anchor).squeeze(1)
        pos = l2_normalize(pos_emb)
        
        B, P, D = pos.shape
        if pos_mask is None: pos_mask = torch.ones((B, P), device=anchor.device)
        pos_mask = pos_mask.bool()

        global_anchors = gather_with_grad(anchor_norm)
        flat_pos = pos.view(-1, D)
        flat_mask = pos_mask.view(-1)
        
        logits = torch.matmul(flat_pos, global_anchors.T) / self.T
        rank = torch.distributed.get_rank() if _is_distributed() else 0
        target = (torch.arange(B, device=anchor.device) + rank * B).repeat_interleave(P)
        
        loss2 = torch.tensor(0.0, device=anchor.device, requires_grad=True)
        if flat_mask.any():
            loss2 = F.cross_entropy(logits[flat_mask], target[flat_mask])
            
        total = 0.5 * (out1.loss + loss2)
        return LossOutput(loss=total, metrics={"clip/loss": total.detach(), "clip/t2i": out1.loss.detach(), "clip/i2t": loss2.detach()})

# ==============================================================================
# SECTION C: Circle Loss
# ==============================================================================

class BaselineCircleLoss(StandardizedLoss):
    def __init__(self, m: float = 0.25, gamma: float = 64, **kwargs):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        **kwargs
    ) -> LossOutput:

        # 1. Normalize
        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(1)
        anchor = l2_normalize(anchor)      # (B, 1, D)
        pos_emb = l2_normalize(positive)   # (B, P, D)

        # 2. Compute Similarities
        sp = (anchor * pos_emb).sum(dim=-1)  # (B, P)

        # Global Batch Negatives
        global_pos = gather_with_grad(pos_emb)
        flat_global = global_pos.view(-1, anchor.shape[-1])  # (Global_Size, D)
        sn_batch = torch.matmul(anchor.squeeze(1), flat_global.T)  # (B, Global_Size)

        sn_list = [sn_batch]

        # Hard Negatives
        if negative is not None:
            neg_emb = l2_normalize(negative)  # (B, K, D)
            sn_hard = torch.matmul(anchor, neg_emb.transpose(1, 2)).squeeze(1)  # (B, K)
            sn_list.append(sn_hard)

        sn = torch.cat(sn_list, dim=1)  # (B, Global + K)

        # 3. Masking
        B, P = sp.shape
        rank = torch.distributed.get_rank() if _is_distributed() else 0

        sn_mask = torch.ones_like(sn, dtype=torch.bool)

        global_ids = torch.arange(flat_global.shape[0], device=anchor.device).unsqueeze(0).expand(B, -1)
        start_idx = (rank * B + torch.arange(B, device=anchor.device)).unsqueeze(1) * P
        own_mask = (global_ids >= start_idx) & (global_ids < start_idx + P)
        sn_mask[:, :flat_global.shape[0]].masked_fill_(own_mask, False)

        if pos_mask is None:
            pos_mask = torch.ones_like(sp, dtype=torch.bool)
        pos_mask = pos_mask.bool()

        if neg_mask is not None:
            batch_len = sn_batch.shape[1]
            sn_mask[:, batch_len:] = sn_mask[:, batch_len:] & neg_mask.bool()

        # ---------------------------------------------------------------------
        # 4. Circle Loss Computation (fp16-safe)
        # Do the logits in fp32 for stability, then reduce to a scalar loss.
        # ---------------------------------------------------------------------
        sp_f = sp.float()
        sn_f = sn.float()

        # --- Positives ---
        ap = torch.clamp_min(-sp_f.detach() + 1.0 + self.m, min=0.0)
        logit_p = -ap * (sp_f - (1.0 - self.m)) * float(self.gamma)

        neg_inf_p = _neg_inf(logit_p)  # dtype-matched (fp32)
        logit_p = logit_p.masked_fill(~pos_mask, neg_inf_p)

        # --- Negatives ---
        an = torch.clamp_min(sn_f.detach() + self.m, min=0.0)
        logit_n = an * (sn_f - self.m) * float(self.gamma)

        neg_inf_n = _neg_inf(logit_n)  # dtype-matched (fp32)
        logit_n = logit_n.masked_fill(~sn_mask, neg_inf_n)

        # Loss = Softplus( LogSumExp(Pos) + LogSumExp(Neg) )
        loss = self.soft_plus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return LossOutput(loss=loss, metrics={"circle/loss": loss.detach()})


class ExtendedCircleLoss(StandardizedLoss):
    def __init__(self, m: float = 0.25, gamma: float = 64, alpha: float = 0.5, **kwargs):
        super().__init__()
        self.base = BaselineCircleLoss(m, gamma)
        self.alpha = alpha

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: Optional[torch.Tensor] = None,
                pos_mask: Optional[torch.Tensor] = None,
                neg_mask: Optional[torch.Tensor] = None,
                eps: float = 1e-7,
                **kwargs) -> LossOutput:

        # Tier 1
        out1 = self.base(anchor, positive, negative, pos_mask, neg_mask)
        loss_main = out1.loss

        if negative is None:
            return LossOutput(loss=loss_main, metrics={"circle_ext/loss": loss_main.detach()})

        # Tier 2 (existing)
        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(1)
        anchor_n = l2_normalize(anchor, eps=eps)       # (B,1,D)

        pos_emb = l2_normalize(positive, eps=eps)      # (B,P,D)
        B, P, D = pos_emb.shape
        p_mask = pos_mask.bool() if pos_mask is not None else torch.ones((B, P), dtype=torch.bool, device=anchor_n.device)

        neg_emb = l2_normalize(negative, eps=eps)
        K = neg_emb.shape[1]
        n_mask = neg_mask.bool() if neg_mask is not None else torch.ones((B, K), dtype=torch.bool, device=anchor_n.device)

        sim_hard = torch.matmul(anchor_n, neg_emb.transpose(1, 2)).squeeze(1)  # (B,K)

        global_pos = gather_with_grad(pos_emb)
        flat_global = global_pos.view(-1, D)
        sim_batch = torch.matmul(anchor_n.squeeze(1), flat_global.T)           # (B,M)

        # mask self-positives + invalid global positives
        rank = torch.distributed.get_rank() if _is_distributed() else 0
        global_pos_mask = gather_tensor(p_mask).bool()
        flat_global_mask = global_pos_mask.view(-1)
        cand_mask_batch = flat_global_mask.unsqueeze(0).expand(B, -1)

        global_ids = torch.arange(flat_global.shape[0], device=anchor_n.device).unsqueeze(0).expand(B, -1)
        start_idx = (rank * B + torch.arange(B, device=anchor_n.device)).unsqueeze(1) * P
        own_mask = (global_ids >= start_idx) & (global_ids < (start_idx + P))
        cand_mask_batch = cand_mask_batch & (~own_mask)

        neg_inf = _neg_inf(sim_batch)
        sim_batch_clean = sim_batch.masked_fill(~cand_mask_batch, neg_inf)

        diff = sim_batch_clean.unsqueeze(1) - sim_hard.unsqueeze(2)  # (B,K,M)
        loss_mat = F.relu(diff + self.base.m)
        loss_tier = (loss_mat.mean(dim=2) * n_mask.float()).sum() / (n_mask.float().sum() + eps)

        total = loss_main + self.alpha * loss_tier

        # ---- extra monitors ----
        metrics: Dict[str, torch.Tensor] = {
            "circle_ext/loss": total.detach(),
            "circle_ext/main": loss_main.detach(),
            "circle_ext/tier": loss_tier.detach(),
        }

        with torch.no_grad():
            def _rate(ok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                num = (ok & mask).float().sum()
                den = mask.float().sum()
                return _dist_ratio(num, den, eps).detach()

            # pos sims (cosine)
            sim_pos = torch.matmul(anchor_n, pos_emb.transpose(1, 2)).squeeze(1)  # (B,P)
            pos_inf = torch.tensor(torch.finfo(sim_pos.dtype).max, device=sim_pos.device, dtype=sim_pos.dtype)

            pos_min = sim_pos.masked_fill(~p_mask, pos_inf).min(dim=1).values
            batch_max = sim_batch_clean.max(dim=1).values

            has_pos = p_mask.any(dim=1)
            has_batch = cand_mask_batch.any(dim=1)
            has_hard = n_mask.any(dim=1)

            metrics["circle_ext/has_pos_rate"] = _dist_ratio(has_pos.float().sum(), anchor_n.new_tensor(B), eps).detach()
            metrics["circle_ext/has_batch_rate"] = _dist_ratio(has_batch.float().sum(), anchor_n.new_tensor(B), eps).detach()
            metrics["circle_ext/has_hard_rate"] = _dist_ratio(has_hard.float().sum(), anchor_n.new_tensor(B), eps).detach()

            base_pb = has_pos & has_batch
            metrics["circle_ext/pos_gt_batch_rate"] = _rate(pos_min > batch_max, base_pb)
            metrics["circle_ext/pos_minus_batch_mean"] = _dist_ratio(
                ((pos_min - batch_max) * base_pb.float()).sum(), base_pb.float().sum(), eps
            ).detach()

            if has_hard.any():
                hard_max = sim_hard.masked_fill(~n_mask, neg_inf).max(dim=1).values
                trio = has_pos & has_batch & has_hard

                metrics["circle_ext/pos_gt_hard_rate"] = _rate(pos_min > hard_max, trio)
                metrics["circle_ext/hard_gt_batch_rate"] = _rate(hard_max > batch_max, trio)
                metrics["circle_ext/full_order_rate"] = _rate((pos_min > hard_max) & (hard_max > batch_max), trio)

                metrics["circle_ext/pos_minus_hard_mean"] = _dist_ratio(
                    ((pos_min - hard_max) * trio.float()).sum(), trio.float().sum(), eps
                ).detach()
                metrics["circle_ext/hard_minus_batch_mean"] = _dist_ratio(
                    ((hard_max - batch_max) * trio.float()).sum(), trio.float().sum(), eps
                ).detach()

                # per-slot: hard slot beats max batch
                batch_max_exp = batch_max.unsqueeze(1).expand_as(sim_hard)
                metrics["circle_ext/hard_gt_maxbatch_frac"] = _dist_ratio(
                    (((sim_hard > batch_max_exp) & n_mask).float().sum()),
                    (n_mask.float().sum()),
                    eps,
                ).detach()

        return LossOutput(loss=total, metrics=metrics)



# ==============================================================================
# SECTION D: Triplet Margin Loss (Euclidean)
# ==============================================================================

class BaselineTripletMarginLoss(StandardizedLoss):
    def __init__(self, margin: float = 0.2, **kwargs):
        super().__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: Optional[torch.Tensor] = None, 
                pos_mask: Optional[torch.Tensor] = None, 
                neg_mask: Optional[torch.Tensor] = None, 
                eps: float = 1e-7,
                **kwargs
                ) -> LossOutput:
        
        # 1. Normalize
        if anchor.dim() == 2: anchor = anchor.unsqueeze(1)
        anchor = l2_normalize(anchor, eps=eps)
        pos_emb = l2_normalize(positive, eps=eps)

        B, P, _ = pos_emb.shape
        if pos_mask is None: pos_mask = torch.ones((B, P), device=anchor.device)
        pos_mask = pos_mask.bool()

        # 2. Positive Distance: d(A, P)
        # (B, P)
        d_pos = (2 - 2 * torch.matmul(anchor, pos_emb.transpose(1, 2)).squeeze(1)).clamp(min=eps).sqrt()

        # 3. Determine Negative Distance: d(A, N)
        d_neg = None
        
        # Option A: Explicit Hard Negatives provided
        if negative is not None:
            neg_emb = l2_normalize(negative, eps=eps)
            _, K, _ = neg_emb.shape
            if neg_mask is None: neg_mask = torch.ones((B, K), device=anchor.device)
            
            # (B, K)
            d_neg_hard = (2 - 2 * torch.matmul(anchor, neg_emb.transpose(1, 2)).squeeze(1)).clamp(min=eps).sqrt()
            
            # For baseline, we usually compare Pos vs All Negs pairwise.
            # Expanding d_pos to (B, P, K) and d_neg to (B, P, K)
            # This matches the original implementation logic.
            d_neg = d_neg_hard
            
            # Create combined mask for the final calculation
            # mask: (B, P, K)
            final_mask = pos_mask.unsqueeze(-1) & neg_mask.bool().unsqueeze(1)
            
            # Expand distances for broadcast
            d_pos_exp = d_pos.unsqueeze(-1)      # (B, P, 1)
            d_neg_exp = d_neg.unsqueeze(1)       # (B, 1, K)

        # Option B: No explicit negatives -> Use BATCH HARD mining
        else:
            # Gather all positives from the batch (Global Batch)
            global_pos = gather_with_grad(pos_emb).view(-1, anchor.shape[-1]) # (Global*B*P, D)
            
            # Calculate distance to ALL batch items
            # (B, Global_Total)
            d_batch = (2 - 2 * torch.matmul(anchor.squeeze(1), global_pos.T)).clamp(min=eps).sqrt()
            
            # Mask out self-positives (we don't want to treat ourselves as a negative)
            rank = torch.distributed.get_rank() if _is_distributed() else 0
            global_ids = torch.arange(d_batch.shape[1], device=anchor.device).unsqueeze(0).expand(B, -1)
            start_idx = (rank * B + torch.arange(B, device=anchor.device)).unsqueeze(1) * P
            own_mask = (global_ids >= start_idx) & (global_ids < start_idx + P)
            
            # Set self-distance to infinity so it's ignored in min()
            d_batch_masked = d_batch.masked_fill(own_mask, float('inf'))
            
            # Mine the HARDEST negative (smallest distance) in the batch for each anchor
            # (B, 1)
            d_neg_hardest, _ = d_batch_masked.min(dim=1, keepdim=True)
            
            d_neg_exp = d_neg_hardest.unsqueeze(1) # (B, 1, 1)
            d_pos_exp = d_pos.unsqueeze(-1)        # (B, P, 1)
            
            # We treat this "min" as a valid negative for all P
            final_mask = pos_mask.unsqueeze(-1)    # (B, P, 1)

        # 4. Compute Loss
        # ReLU(d_pos - d_neg + margin)
        loss_mat = self.relu(d_pos_exp - d_neg_exp + self.margin)
        
        valid = final_mask.float().sum()
        loss = torch.tensor(0.0, device=anchor.device, requires_grad=True)
        
        if valid > 0:
            loss = (loss_mat * final_mask.float()).sum() / valid

        with torch.no_grad():
            acc = ((loss_mat == 0).float() * final_mask.float()).sum() / (valid + eps)

        return LossOutput(loss=loss, metrics={"triplet/loss": loss.detach(), "triplet/acc": acc})

class ExtendedTripletMarginLoss(StandardizedLoss):
    def __init__(self, margin_quality: float = 0.2, margin_relevance: float = 0.5, alpha: float = 0.5, **kwargs):
        super().__init__()
        self.m1, self.m2, self.alpha = margin_quality, margin_relevance, alpha

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: Optional[torch.Tensor] = None, 
                pos_mask: Optional[torch.Tensor] = None, 
                neg_mask: Optional[torch.Tensor] = None,
                eps: float = 1e-7,
                **kwargs) -> LossOutput:
        
        # 1. Norm and Distances
        anchor = l2_normalize(anchor if anchor.dim() == 3 else anchor.unsqueeze(1), eps=eps)
        pos_emb = l2_normalize(positive, eps=eps)
        B, P, _ = pos_emb.shape

        # Pairwise Euclidean Distance squared: ||a-b||^2 = 2 - 2*sim
        d_pos = (2 - 2 * torch.matmul(anchor, pos_emb.transpose(1, 2)).squeeze(1)).clamp(eps).sqrt()
        
        # Batch Distances (Always needed)
        global_pos = gather_with_grad(pos_emb).view(-1, anchor.shape[-1])
        # (B, Global_Batch)
        d_batch = (2 - 2 * torch.matmul(anchor.squeeze(1), global_pos.T)).clamp(eps).sqrt()

        # Mask out own positives from the batch pool
        rank = torch.distributed.get_rank() if _is_distributed() else 0
        global_ids = torch.arange(d_batch.shape[1], device=anchor.device).unsqueeze(0)
        start_idx = (rank * B + torch.arange(B, device=anchor.device)).unsqueeze(1) * P
        own_mask = (global_ids >= start_idx) & (global_ids < start_idx + P)
        
        pos_inf = torch.tensor(torch.finfo(d_batch.dtype).max, device=d_batch.device, dtype=d_batch.dtype)
        d_batch_clean = d_batch.masked_fill(own_mask, pos_inf) # Very far away
        
        # Mine hardest batch negative (closest)
        # (B,)
        d_batch_hardest = d_batch_clean.min(dim=1).values

        if pos_mask is None: pos_mask = torch.ones((B, P), device=anchor.device)
        
        # --- LOGIC BRANCH ---
        
        # Case A: Hard Negatives Provided -> Full Tiered Logic
        if negative is not None:
            neg_emb = l2_normalize(negative, eps=eps)
            _, K, _ = neg_emb.shape
            if neg_mask is None: neg_mask = torch.ones((B, K), device=anchor.device)
            
            d_hard = (2 - 2 * torch.matmul(anchor, neg_emb.transpose(1, 2)).squeeze(1)).clamp(eps).sqrt()

            # Tier 1: Pos < Hard
            # (B, P, K)
            l_mat_1 = F.relu(d_pos.unsqueeze(-1) - d_hard.unsqueeze(1) + self.m1)
            mask_1 = pos_mask.bool().unsqueeze(-1) & neg_mask.bool().unsqueeze(1)
            loss_1 = (l_mat_1 * mask_1.float()).sum() / (mask_1.sum() + eps)

            # Tier 2: Hard < Batch
            # (B, K, Global_Batch)
            # Use ALL batch negatives or just the hardest? 
            # Original code broadcasted against all batch negs. Let's keep that but use the clean version.
            l_mat_2 = F.relu(d_hard.unsqueeze(-1) - d_batch_clean.unsqueeze(1) + self.m2)
            mask_2 = neg_mask.bool().unsqueeze(-1) # Broadcasts over Global_Batch
            loss_2 = (l_mat_2 * mask_2.float()).sum() / (mask_2.sum() * (d_batch.shape[1] - P) + eps)
            
            metrics = {"triplet_ext/loss": (loss_1 + self.alpha * loss_2).detach(), "triplet_ext/tier1": loss_1.detach(), "triplet_ext/tier2": loss_2.detach()}
            return LossOutput(loss=loss_1 + self.alpha * loss_2, metrics=metrics)

        # Case B: No Hard Negatives -> Fallback to Baseline (Pos < Batch Hardest)
        else:
            # (B, P)
            # Compare every positive against the single hardest negative in the batch
            l_mat_1 = F.relu(d_pos - d_batch_hardest.unsqueeze(1) + self.m1)
            mask_1 = pos_mask.bool()
            loss_1 = (l_mat_1 * mask_1.float()).sum() / (mask_1.sum() + eps)
            
            return LossOutput(loss=loss_1, metrics={"triplet_ext/loss": loss_1.detach(), "triplet_ext/tier1": loss_1.detach(), "triplet_ext/tier2": torch.tensor(0.0, device=anchor.device)})

# ==============================================================================
# SECTION D-Alt: Cosine Triplet Loss (CLIP-Compatible)
# ==============================================================================

# ======================================================================
# FaceNet-style (semi-hard) Triplet Loss in COSINE space
# - Single code path: always mine negatives from the GLOBAL batch pool,
#   and (optionally) AUGMENT that pool with user-provided negatives.
# - Default mining: SEMI-HARD if available, else fallback to HARDEST.
# - Includes distributed-safe monitoring metrics.
# ======================================================================


class BaselineCosineTripletLoss(StandardizedLoss):
    def __init__(self, margin: float = 0.2, **kwargs):
        super().__init__()
        self.margin = float(margin)
        self.relu = nn.ReLU()

    @staticmethod
    def _get_rank() -> int:
        return torch.distributed.get_rank() if _is_distributed() else 0

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        **kwargs
    ) -> LossOutput:

        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(1)  # (B,1,D)
        anchor = l2_normalize(anchor, eps=eps)
        pos_emb = l2_normalize(positive, eps=eps)  # (B,P,D)

        B, P, D = pos_emb.shape
        if pos_mask is None:
            pos_mask = torch.ones((B, P), device=anchor.device, dtype=torch.bool)
        else:
            pos_mask = pos_mask.bool()

        # ---- global batch pool ----
        global_pos = gather_with_grad(pos_emb)                     # (B_global,P,D)
        flat_global = global_pos.view(-1, D)                       # (N_batch,D)
        sim_batch = torch.matmul(anchor.squeeze(1), flat_global.T) # (B,N_batch)

        global_pos_mask = gather_tensor(pos_mask).bool()           # (B_global,P)
        flat_global_mask = global_pos_mask.view(-1)                # (N_batch,)
        cand_mask_batch = flat_global_mask.unsqueeze(0).expand(B, -1)

        # exclude self-positives
        rank = self._get_rank()
        N_batch = flat_global.shape[0]
        global_ids = torch.arange(N_batch, device=anchor.device).unsqueeze(0).expand(B, -1)
        start_idx = (rank * B + torch.arange(B, device=anchor.device)).unsqueeze(1) * P
        own_mask = (global_ids >= start_idx) & (global_ids < start_idx + P)
        cand_mask_batch = cand_mask_batch & (~own_mask)

        cand_sims = sim_batch
        cand_mask = cand_mask_batch

        # ---- append explicit negatives if K>0 ----
        if negative is not None:
            if negative.dim() == 2:
                # If someone accidentally passes (B,D), treat as K=1
                negative = negative.unsqueeze(1)
            K = negative.shape[1]
            if K > 0:
                neg_emb = l2_normalize(negative, eps=eps)  # (B,K,D)
                sim_hard = torch.matmul(anchor, neg_emb.transpose(1, 2)).squeeze(1)  # (B,K)

                cand_mask_hard = torch.ones((B, K), device=anchor.device, dtype=torch.bool) \
                    if neg_mask is None else neg_mask.bool()

                cand_sims = torch.cat([cand_sims, sim_hard], dim=1)
                cand_mask = torch.cat([cand_mask, cand_mask_hard], dim=1)

        sim_pos = torch.matmul(anchor, pos_emb.transpose(1, 2)).squeeze(1)  # (B,P)

        neg_inf = _neg_inf(cand_sims)
        total_loss_sum = anchor.new_tensor(0.0)
        total_count = anchor.new_tensor(0.0)
        total_acc_sum = anchor.new_tensor(0.0)

        # monitors
        semi_found_sum = anchor.new_tensor(0.0)
        semi_found_den = anchor.new_tensor(0.0)
        margin_sat_sum = anchor.new_tensor(0.0)
        margin_sat_den = anchor.new_tensor(0.0)
        viol_sum = anchor.new_tensor(0.0)
        viol_den = anchor.new_tensor(0.0)
        gap_sum = anchor.new_tensor(0.0)
        gap_den = anchor.new_tensor(0.0)

        has_any_candidate = cand_mask.any(dim=1)  # (B,)

        for p in range(P):
            valid = pos_mask[:, p] & has_any_candidate
            if not valid.any():
                continue

            s_pos = sim_pos[:, p]
            s_pos_col = s_pos.unsqueeze(1)

            semi_mask = cand_mask & (cand_sims < s_pos_col) & (cand_sims > (s_pos_col - self.margin))
            semi_any = semi_mask.any(dim=1)

            semi_best = cand_sims.masked_fill(~semi_mask, neg_inf).max(dim=1).values
            hard_best = cand_sims.masked_fill(~cand_mask, neg_inf).max(dim=1).values
            chosen_neg = torch.where(semi_any, semi_best, hard_best)

            loss_vec = self.relu(chosen_neg - s_pos + self.margin)

            v = valid.float()
            total_loss_sum = total_loss_sum + (loss_vec * v).sum()
            total_count = total_count + v.sum()

            with torch.no_grad():
                total_acc_sum = total_acc_sum + ((loss_vec == 0).float() * v).sum()

                semi_found_sum = semi_found_sum + (semi_any.float() * v).sum()
                semi_found_den = semi_found_den + v.sum()

                gap = (s_pos - chosen_neg)
                gap_sum = gap_sum + (gap * v).sum()
                gap_den = gap_den + v.sum()

                margin_sat_sum = margin_sat_sum + ((gap >= self.margin).float() * v).sum()
                margin_sat_den = margin_sat_den + v.sum()

                viol_sum = viol_sum + ((chosen_neg >= s_pos).float() * v).sum()
                viol_den = viol_den + v.sum()

        loss = anchor.new_tensor(0.0, requires_grad=True)
        if total_count > 0:
            loss = total_loss_sum / (total_count + eps)

        # distributed-safe metrics
        acc = _dist_ratio(total_acc_sum, total_count, eps).detach()
        semi_hard_rate = _dist_ratio(semi_found_sum, semi_found_den, eps).detach()
        pos_minus_neg_mean = _dist_ratio(gap_sum, gap_den, eps).detach()
        margin_sat_rate = _dist_ratio(margin_sat_sum, margin_sat_den, eps).detach()
        viol_rate = _dist_ratio(viol_sum, viol_den, eps).detach()

        return LossOutput(
            loss=loss,
            metrics={
                "triplet/loss": loss.detach(),
                "triplet/acc": acc,
                "triplet/semi_hard_rate": semi_hard_rate,
                "triplet/pos_minus_neg_mean": pos_minus_neg_mean,
                "triplet/margin_sat_rate": margin_sat_rate,
                "triplet/viol_rate": viol_rate,
            },
        )


class ExtendedCosineTripletLoss(StandardizedLoss):
    def __init__(self, 
                margin_quality: float = 0.2,  # static fallback for Pos vs Hard
                margin_relevance: float = 0.1, # additional margin for Hard vs Batch 
                alpha: float = 0.5, # weight for the second tier loss
                dyn: float = 0.0, # >0.0 uses dynamic margins from the dataset
                **kwargs):
        super().__init__()
        self.m1 = float(margin_quality)
        self.m2 = float(margin_relevance)
        self.alpha = float(alpha)
        self.dyn = float(dyn)
        self.relu = nn.ReLU()

    @staticmethod
    def _get_rank() -> int:
        return torch.distributed.get_rank() if _is_distributed() else 0

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        margin: Optional[torch.Tensor] = None,  
        **kwargs
    ) -> LossOutput:

        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(1)
        anchor = l2_normalize(anchor, eps=eps)
        pos_emb = l2_normalize(positive, eps=eps)

        B, P, D = pos_emb.shape
        if pos_mask is None:
            pos_mask = torch.ones((B, P), device=anchor.device, dtype=torch.bool)
        else:
            pos_mask = pos_mask.bool()

        # ---- batch pool ----
        global_pos = gather_with_grad(pos_emb)
        flat_global = global_pos.view(-1, D)
        sim_batch = torch.matmul(anchor.squeeze(1), flat_global.T)  # (B,N_batch)

        global_pos_mask = gather_tensor(pos_mask).bool()
        flat_global_mask = global_pos_mask.view(-1)
        cand_mask_batch = flat_global_mask.unsqueeze(0).expand(B, -1)

        rank = self._get_rank()
        N_batch = flat_global.shape[0]
        global_ids = torch.arange(N_batch, device=anchor.device).unsqueeze(0).expand(B, -1)
        start_idx = (rank * B + torch.arange(B, device=anchor.device)).unsqueeze(1) * P
        own_mask = (global_ids >= start_idx) & (global_ids < start_idx + P)
        cand_mask_batch = cand_mask_batch & (~own_mask)

        cand_sims = sim_batch
        cand_mask = cand_mask_batch
        sim_hard = None
        hard_mask = None

        # ---- append explicit negatives if K>0 ----
        if negative is not None:
            if negative.dim() == 2:
                negative = negative.unsqueeze(1)
            K = negative.shape[1]
            if K > 0:
                neg_emb = l2_normalize(negative, eps=eps)  # (B,K,D)
                sim_hard = torch.matmul(anchor, neg_emb.transpose(1, 2)).squeeze(1)  # (B,K)

                hard_mask = torch.ones((B, K), device=anchor.device, dtype=torch.bool) \
                    if neg_mask is None else neg_mask.bool()

                cand_sims = torch.cat([cand_sims, sim_hard], dim=1)
                cand_mask = torch.cat([cand_mask, hard_mask], dim=1)

        sim_pos = torch.matmul(anchor, pos_emb.transpose(1, 2)).squeeze(1)  # (B,P)
        neg_inf = _neg_inf(cand_sims)

        # ---- DYNAMIC MARGIN RESOLUTION (m1_eff) ----
        if self.dyn > 0.0 and margin is not None:
            m1_raw = margin.to(anchor.device) * self.dyn
            # If margin is per-slot (B, K), find the hardest negative's margin
            if m1_raw.dim() == 2 and m1_raw.shape[1] > 1 and sim_hard is not None and hard_mask is not None:
                hard_idx = sim_hard.masked_fill(~hard_mask, neg_inf).argmax(dim=1)  # (B,)
                m1_eff = m1_raw.gather(1, hard_idx.unsqueeze(1))  # (B, 1)
            else:
                m1_eff = m1_raw.unsqueeze(1) if m1_raw.dim() == 1 else m1_raw # Ensure (B, 1)
        else:
            m1_eff = self.m1  # Fallback to scalar

        m1_1d = m1_eff.squeeze(1) if isinstance(m1_eff, torch.Tensor) else m1_eff # For 1D math

        # ---- Tier 1 Calculation ----
        tier1_sum = anchor.new_tensor(0.0)
        tier1_count = anchor.new_tensor(0.0)

        semi_found_sum = anchor.new_tensor(0.0)
        semi_found_den = anchor.new_tensor(0.0)
        viol_sum = anchor.new_tensor(0.0)
        viol_den = anchor.new_tensor(0.0)

        has_any_candidate = cand_mask.any(dim=1)

        for p in range(P):
            valid = pos_mask[:, p] & has_any_candidate
            if not valid.any():
                continue

            s_pos = sim_pos[:, p]
            s_pos_col = s_pos.unsqueeze(1)

            # Use dynamic m1_eff for semi-hard mining
            semi_mask = cand_mask & (cand_sims < s_pos_col) & (cand_sims > (s_pos_col - m1_eff))
            semi_any = semi_mask.any(dim=1)

            semi_best = cand_sims.masked_fill(~semi_mask, neg_inf).max(dim=1).values
            hard_best = cand_sims.masked_fill(~cand_mask, neg_inf).max(dim=1).values
            chosen_neg = torch.where(semi_any, semi_best, hard_best)

            # Use dynamic m1_1d for loss calculation
            loss_vec = self.relu(chosen_neg - s_pos + m1_1d)
            v = valid.float()

            tier1_sum = tier1_sum + (loss_vec * v).sum()
            tier1_count = tier1_count + v.sum()

            with torch.no_grad():
                semi_found_sum = semi_found_sum + (semi_any.float() * v).sum()
                semi_found_den = semi_found_den + v.sum()
                viol_sum = viol_sum + ((chosen_neg >= s_pos).float() * v).sum()
                viol_den = viol_den + v.sum()

        loss_tier1 = anchor.new_tensor(0.0, requires_grad=True)
        if tier1_count > 0:
            loss_tier1 = tier1_sum / (tier1_count + eps)

        # ---- tier 2: hard > batch ----
        loss_tier2 = anchor.new_tensor(0.0, requires_grad=True)
        if sim_hard is not None and hard_mask is not None and sim_hard.shape[1] > 0 and hard_mask.any():
            sim_batch_hardest = sim_batch.masked_fill(~cand_mask_batch, neg_inf).max(dim=1).values  # (B,)
            l2_mat = self.relu(sim_batch_hardest.unsqueeze(1) - sim_hard + self.m2)  # (B,K)

            denom = hard_mask.float().sum()
            if denom > 0:
                loss_tier2 = (l2_mat * hard_mask.float()).sum() / (denom + eps)

        total = loss_tier1 + self.alpha * loss_tier2

        # ---- ordering monitors: pos > hard > batch ----
        metrics_extra: Dict[str, torch.Tensor] = {}

        has_pos = pos_mask.any(dim=1)
        has_batch = cand_mask_batch.any(dim=1)
        batch_max = sim_batch.masked_fill(~cand_mask_batch, neg_inf).max(dim=1).values
        
        # --- FIX: Track both hardest (pos_min) and easiest (pos_max) positives ---
        pos_inf = torch.tensor(torch.finfo(sim_pos.dtype).max, device=sim_pos.device, dtype=sim_pos.dtype)
        pos_min = sim_pos.masked_fill(~pos_mask, pos_inf).min(dim=1).values
        pos_max = sim_pos.masked_fill(~pos_mask, neg_inf).max(dim=1).values

        if sim_hard is not None and hard_mask is not None and sim_hard.shape[1] > 0:
            has_hard = hard_mask.any(dim=1)
            hard_max = sim_hard.masked_fill(~hard_mask, neg_inf).max(dim=1).values

            base_mask = has_pos & has_batch & has_hard
            base_den = base_mask.float().sum()

            if base_den > 0:
                # Compare the HARDEST positive against the HARDEST negative (strict, drives the loss)
                pos_gt_hard = ((pos_min > hard_max) & base_mask).float()
                
                # Compare the EASIEST positive against the HARDEST negative (optimistic/legacy)
                pos_gt_hard_easiest = ((pos_max > hard_max) & base_mask).float()
                
                hard_gt_batch = ((hard_max > batch_max) & base_mask).float()
                full_order = (((pos_min > hard_max) & (hard_max > batch_max)) & base_mask).float()

                # Use dynamic m1_1d for metric tracking
                pos_gt_hard_m1 = ((pos_min >= hard_max + m1_1d) & base_mask).float()
                hard_gt_batch_m2 = ((hard_max >= batch_max + self.m2) & base_mask).float()
                full_order_margin = (((pos_min >= hard_max + m1_1d) & (hard_max >= batch_max + self.m2)) & base_mask).float()

                gap_pos_hard = (pos_min - hard_max) * base_mask.float()
                gap_hard_batch = (hard_max - batch_max) * base_mask.float()

                metrics_extra.update({
                    "triplet_ext/pos_gt_hard_rate": _dist_ratio(pos_gt_hard.sum(), base_den, eps).detach(),
                    "triplet_ext/pos_gt_hard_rate_easiest": _dist_ratio(pos_gt_hard_easiest.sum(), base_den, eps).detach(),
                    "triplet_ext/hard_gt_batch_rate": _dist_ratio(hard_gt_batch.sum(), base_den, eps).detach(),
                    "triplet_ext/full_order_rate": _dist_ratio(full_order.sum(), base_den, eps).detach(),

                    "triplet_ext/pos_gt_hard_m1_rate": _dist_ratio(pos_gt_hard_m1.sum(), base_den, eps).detach(),
                    "triplet_ext/hard_gt_batch_m2_rate": _dist_ratio(hard_gt_batch_m2.sum(), base_den, eps).detach(),
                    "triplet_ext/full_order_margin_rate": _dist_ratio(full_order_margin.sum(), base_den, eps).detach(),

                    "triplet_ext/gap_pos_hard_mean": _dist_ratio(gap_pos_hard.sum(), base_den, eps).detach(),
                    "triplet_ext/gap_hard_batch_mean": _dist_ratio(gap_hard_batch.sum(), base_den, eps).detach(),
                })

        semi_hard_rate = _dist_ratio(semi_found_sum, semi_found_den, eps).detach()
        tier1_viol_rate = _dist_ratio(viol_sum, viol_den, eps).detach()

        has_pos = pos_mask.any(dim=1)       # (B,)
        valid_anchor = has_pos              # you can also AND other conditions if needed
        no_cand = valid_anchor & (~has_any_candidate)
        num_no_cand = no_cand.float().sum()
        den = valid_anchor.float().sum()

        metrics = {
            "triplet_ext/loss": total.detach(),
            "triplet_ext/tier1": loss_tier1.detach(),
            "triplet_ext/tier2": loss_tier2.detach(),
            "triplet_ext/semi_hard_rate": semi_hard_rate,
            "triplet_ext/tier1_viol_rate": tier1_viol_rate,
            "triplet_ext/no_cand_rate": _dist_ratio(num_no_cand, den, eps).detach(),
            # Log the effective dynamic margin to W&B
            "triplet_ext/m1_eff_mean": m1_1d.mean().detach() if isinstance(m1_1d, torch.Tensor) else anchor.new_tensor(m1_1d),
        }
        metrics.update(metrics_extra)

        return LossOutput(loss=total, metrics=metrics)





class ExtendedCosineTripletLossV3(StandardizedLoss):
    """
    Triplet-Ext V3 enforcing C1-C5 explicitly.
    """

    def __init__(
        self,
        m_pos_soft: float = 0.2,
        m_soft_batch: float = 0.1,
        m_pos_batch: float = 0.2,
        w_align: float = 0.1,
        w_pos_soft: float = 1.0,
        w_soft_batch: float = 0.5,
        w_pos_batch: float = 0.5,
        w_order: float = 0.5,
        s_ps: float = 32.0,
        s_sb: float = 32.0,
        s_pb: float = 32.0,
        s_ord: float = 32.0,
        **kwargs,
    ):
        super().__init__()
        self.m1 = float(m_pos_soft)
        self.m2 = float(m_soft_batch)
        self.mpb = float(m_pos_batch)

        self.w_align = float(w_align)
        self.w_pos_soft = float(w_pos_soft)
        self.w_soft_batch = float(w_soft_batch)
        self.w_pos_batch = float(w_pos_batch)
        self.w_order = float(w_order)

        self.s_ps = float(s_ps)
        self.s_sb = float(s_sb)
        self.s_pb = float(s_pb)
        self.s_ord = float(s_ord)

    @staticmethod
    def _get_rank() -> int:
        return torch.distributed.get_rank() if _is_distributed() else 0

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        **kwargs,
    ) -> LossOutput:
        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(1)
        anchor = l2_normalize(anchor, eps=eps)
        pos = l2_normalize(positive, eps=eps)

        B, P, D = pos.shape
        device = anchor.device

        p_mask = pos_mask.bool() if pos_mask is not None else torch.ones((B, P), device=device, dtype=torch.bool)
        has_pos = p_mask.any(dim=1)

        sim_pos = torch.matmul(anchor, pos.transpose(1, 2)).squeeze(1)

        neg_inf = _neg_inf(sim_pos)
        pos_inf = torch.tensor(torch.finfo(sim_pos.dtype).max, device=device, dtype=sim_pos.dtype)

        pos_min = sim_pos.masked_fill(~p_mask, pos_inf).min(dim=1).values
        pos_max = sim_pos.masked_fill(~p_mask, neg_inf).max(dim=1).values

        global_pos = gather_with_grad(pos)
        flat_global = global_pos.view(-1, D)
        sim_batch = torch.matmul(anchor.squeeze(1), flat_global.T)

        global_pos_mask = gather_tensor(p_mask).bool()
        flat_global_mask = global_pos_mask.view(-1)
        cand_mask_batch = flat_global_mask.unsqueeze(0).expand(B, -1)

        rank = self._get_rank()
        n_batch = flat_global.shape[0]
        global_ids = torch.arange(n_batch, device=device).unsqueeze(0).expand(B, -1)
        start_idx = (rank * B + torch.arange(B, device=device)).unsqueeze(1) * P
        own_mask = (global_ids >= start_idx) & (global_ids < start_idx + P)
        cand_mask_batch = cand_mask_batch & (~own_mask)

        has_batch = cand_mask_batch.any(dim=1)
        batch_max = sim_batch.masked_fill(~cand_mask_batch, neg_inf).max(dim=1).values

        hard_max = anchor.new_full((B,), fill_value=neg_inf.item())
        has_hard = anchor.new_zeros((B,), dtype=torch.bool)

        if negative is not None:
            if negative.dim() == 2:
                negative = negative.unsqueeze(1)
            if negative.shape[1] > 0:
                neg = l2_normalize(negative, eps=eps)
                k = neg.shape[1]
                n_mask = neg_mask.bool() if neg_mask is not None else torch.ones((B, k), device=device, dtype=torch.bool)
                sim_hard = torch.matmul(anchor, neg.transpose(1, 2)).squeeze(1)
                has_hard = n_mask.any(dim=1)
                hard_max = sim_hard.masked_fill(~n_mask, neg_inf).max(dim=1).values

        l_align = anchor.new_tensor(0.0, requires_grad=True)
        if p_mask.any():
            l_align = (1.0 - sim_pos[p_mask]).mean()

        l_pos_soft = anchor.new_tensor(0.0, requires_grad=True)
        valid_ps = has_pos & has_hard
        if valid_ps.any():
            diff = (hard_max - pos_min + self.m1) * self.s_ps
            l_pos_soft = F.softplus(diff[valid_ps]).mean()

        l_soft_batch = anchor.new_tensor(0.0, requires_grad=True)
        valid_sb = has_hard & has_batch
        if valid_sb.any():
            diff = (batch_max - hard_max + self.m2) * self.s_sb
            l_soft_batch = F.softplus(diff[valid_sb]).mean()

        l_pos_batch = anchor.new_tensor(0.0, requires_grad=True)
        valid_pb = has_pos & has_batch
        if valid_pb.any():
            diff = (batch_max - pos_min + self.mpb) * self.s_pb
            l_pos_batch = F.softplus(diff[valid_pb]).mean()

        l_order = anchor.new_tensor(0.0, requires_grad=True)
        valid_order = has_pos & has_batch & has_hard
        if valid_order.any():
            d1 = (hard_max - pos_min) * self.s_ord
            d2 = (batch_max - hard_max) * self.s_ord
            l_order = (F.softplus(d1[valid_order]) + F.softplus(d2[valid_order])).mean()

        total = (
            self.w_align * l_align
            + self.w_pos_soft * l_pos_soft
            + self.w_soft_batch * l_soft_batch
            + self.w_pos_batch * l_pos_batch
            + self.w_order * l_order
        )

        with torch.no_grad():
            def _rate(ok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                num = (ok & mask).float().sum()
                den = mask.float().sum()
                return _dist_ratio(num, den, eps).detach()

            trio = has_pos & has_batch & has_hard
            duo_pb = has_pos & has_batch
            duo_ps = has_pos & has_hard
            duo_sb = has_hard & has_batch

            pos_gt_hard = pos_min > hard_max
            hard_gt_batch = hard_max > batch_max
            full_order = pos_gt_hard & hard_gt_batch

            pos_gt_hard_m = pos_min >= (hard_max + self.m1)
            hard_gt_batch_m = hard_max >= (batch_max + self.m2)
            pos_gt_batch_m = pos_min >= (batch_max + self.mpb)
            full_order_m = pos_gt_hard_m & hard_gt_batch_m & pos_gt_batch_m

            metrics = {
                "triplet_ext/loss": total.detach(),
                "triplet_ext/C1_align": l_align.detach(),
                "triplet_ext/C2_pos_soft": l_pos_soft.detach(),
                "triplet_ext/C3_soft_batch": l_soft_batch.detach(),
                "triplet_ext/C4_pos_batch": l_pos_batch.detach(),
                "triplet_ext/C5_order": l_order.detach(),
                "triplet_ext/has_pos_rate": _dist_ratio(has_pos.float().sum(), anchor.new_tensor(B), eps).detach(),
                "triplet_ext/has_batch_rate": _dist_ratio(has_batch.float().sum(), anchor.new_tensor(B), eps).detach(),
                "triplet_ext/has_hard_rate": _dist_ratio(has_hard.float().sum(), anchor.new_tensor(B), eps).detach(),
                "triplet_ext/pos_gt_hard_rate": _rate(pos_gt_hard, duo_ps),
                "triplet_ext/hard_gt_batch_rate": _rate(hard_gt_batch, duo_sb),
                "triplet_ext/full_order_rate": _rate(full_order, trio),
                "triplet_ext/pos_gt_hard_m_rate": _rate(pos_gt_hard_m, duo_ps),
                "triplet_ext/hard_gt_batch_m_rate": _rate(hard_gt_batch_m, duo_sb),
                "triplet_ext/pos_gt_batch_m_rate": _rate(pos_gt_batch_m, duo_pb),
                "triplet_ext/full_order_margin_rate": _rate(full_order_m, trio),
                "triplet_ext/gap_posmin_hardmax_mean": _dist_ratio(
                    ((pos_min - hard_max) * duo_ps.float()).sum(), duo_ps.float().sum(), eps
                ).detach(),
                "triplet_ext/gap_hardmax_batchmax_mean": _dist_ratio(
                    ((hard_max - batch_max) * duo_sb.float()).sum(), duo_sb.float().sum(), eps
                ).detach(),
                "triplet_ext/gap_posmin_batchmax_mean": _dist_ratio(
                    ((pos_min - batch_max) * duo_pb.float()).sum(), duo_pb.float().sum(), eps
                ).detach(),
                "triplet_ext/m1": anchor.new_tensor(self.m1).detach(),
                "triplet_ext/m2": anchor.new_tensor(self.m2).detach(),
                "triplet_ext/mpb": anchor.new_tensor(self.mpb).detach(),
                "triplet_ext/w_align": anchor.new_tensor(self.w_align).detach(),
                "triplet_ext/w_pos_soft": anchor.new_tensor(self.w_pos_soft).detach(),
                "triplet_ext/w_soft_batch": anchor.new_tensor(self.w_soft_batch).detach(),
                "triplet_ext/w_pos_batch": anchor.new_tensor(self.w_pos_batch).detach(),
                "triplet_ext/w_order": anchor.new_tensor(self.w_order).detach(),
                "triplet_ext/pos_max_mean": _dist_ratio((pos_max * has_pos.float()).sum(), has_pos.float().sum(), eps).detach(),
            }

        return LossOutput(loss=total, metrics=metrics)


# ==============================================================================
# SECTION E: SigLIP
# ==============================================================================

class BaselineSigLIPLoss(StandardizedLoss):
    def __init__(self, temperature: float = 0.07, bias: float = -10.0, 
                 learnable: bool = False, **kwargs):
        super().__init__()
        self.t_init = temperature
        self.b_init = bias
        if learnable:
            self.t = nn.Parameter(torch.tensor(1.0 / temperature))
            self.b = nn.Parameter(torch.tensor(bias))
        else:
            self.register_buffer("t", torch.tensor(1.0 / temperature))
            self.register_buffer("b", torch.tensor(bias))

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: Optional[torch.Tensor] = None, 
                pos_mask: Optional[torch.Tensor] = None, 
                neg_mask: Optional[torch.Tensor] = None, 
                eps: float = 1e-7,
                **kwargs) -> LossOutput:
        
        pos_emb, neg_emb = positive, negative
        if anchor.dim() == 2: anchor = anchor.unsqueeze(1)
        anchor = l2_normalize(anchor)
        pos_emb = l2_normalize(pos_emb)
        if neg_emb is not None: neg_emb = l2_normalize(neg_emb)

        B, P, D = pos_emb.shape
        if pos_mask is None: pos_mask = torch.ones((B, P), device=anchor.device)
        pos_mask = pos_mask.bool()

        global_pos = gather_with_grad(pos_emb)
        flat_global = global_pos.view(-1, D)
        
        global_mask = gather_tensor(pos_mask).view(-1)
        
        logits = torch.matmul(anchor.squeeze(1), flat_global.T) * self.t + self.b
        labels = -torch.ones_like(logits)
        rank = torch.distributed.get_rank() if _is_distributed() else 0
        
        for p in range(P):
            global_indices = (rank * B + torch.arange(B, device=anchor.device)) * P + p
            rows = torch.arange(B, device=anchor.device)
            valid = pos_mask[:, p]
            if valid.any():
                labels[rows[valid], global_indices[valid]] = 1.0

        loss_hard = torch.tensor(0.0, device=anchor.device)
        if neg_emb is not None:
            logits_hard = torch.matmul(anchor, neg_emb.transpose(1, 2)).squeeze(1) * self.t + self.b
            if neg_mask is not None and neg_mask.any():
                loss_hard = F.binary_cross_entropy_with_logits(
                    logits_hard[neg_mask.bool()], 
                    torch.zeros_like(logits_hard[neg_mask.bool()])
                )

        col_weights = global_mask.float().unsqueeze(0)
        bce_targets = (labels > 0).float()
        loss_dense = F.binary_cross_entropy_with_logits(logits, bce_targets, reduction='none')
        loss_dense = loss_dense * col_weights
        
        num_valid = col_weights.sum() * B
        loss = loss_dense.sum() / (num_valid + eps) + loss_hard
        return LossOutput(loss=loss, metrics={"siglip/loss": loss.detach()})

# ==============================================================================
# SECTION F: Angular Contrastive (arcface-like margin)
# ==============================================================================

def _rank() -> int:
    return torch.distributed.get_rank() if _is_distributed() else 0


def _neg_large(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


def _arcface_apply_margin(cos: torch.Tensor, margin: float, eps: float = 1e-4) -> torch.Tensor:
    if margin == 0.0:
        return cos
    cos = cos.clamp(-1.0 + eps, 1.0 - eps)
    sin = torch.sqrt(torch.clamp(1.0 - cos * cos, min=0.0))
    cos_m = math.cos(margin)
    sin_m = math.sin(margin)
    cos_theta_m = cos * cos_m - sin * sin_m
    return cos_theta_m.clamp(-1.0 + eps, 1.0 - eps)


class BaselineAngularContrastiveLoss(StandardizedLoss):
    """
    Multi-positive + multi-negative ArcFace-style contrastive classification.
    """

    def __init__(self, temperature: float = 0.07, margin: float = 0.5, scale: float = 64.0, **kwargs):
        super().__init__()
        self.temperature = float(temperature)
        self.margin = float(margin)
        self.scale = float(scale)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        **kwargs,
    ) -> LossOutput:
        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(1)
        elif anchor.dim() == 3 and anchor.shape[1] != 1:
            anchor = anchor[:, :1, :]

        if positive.dim() != 3:
            raise ValueError(f"positive must be (B,P,D). Got {tuple(positive.shape)}")

        B, P, _ = positive.shape

        K = 0
        if negative is not None:
            if negative.dim() == 2:
                negative = negative.unsqueeze(1)
            if negative.dim() != 3:
                raise ValueError(f"negative must be (B,K,D). Got {tuple(negative.shape)}")
            K = negative.shape[1]

        device = anchor.device
        p_mask = pos_mask.bool() if pos_mask is not None else torch.ones((B, P), dtype=torch.bool, device=device)
        if negative is not None:
            n_mask = neg_mask.bool() if neg_mask is not None else torch.ones((B, K), dtype=torch.bool, device=device)
        else:
            n_mask = None

        if not p_mask.any():
            zero = anchor.new_tensor(0.0, requires_grad=True)
            return LossOutput(
                loss=zero,
                metrics={
                    "arcface/base_loss": zero.detach(),
                    "arcface/valid_pos_frac": zero.detach(),
                },
            )

        anchor_n = l2_normalize(anchor, eps=eps)
        pos_n = l2_normalize(positive, eps=eps)
        if negative is not None:
            neg_n = l2_normalize(negative, eps=eps)

        cos_ap = torch.matmul(anchor_n.float(), pos_n.float().transpose(1, 2)).squeeze(1)
        cos_ap_m = _arcface_apply_margin(cos_ap, self.margin, eps=1e-4)

        if negative is None or K == 0:
            logits_pos = (cos_ap_m / max(self.temperature, 1e-6)) * self.scale
            loss_pos = F.softplus(-logits_pos[p_mask]).mean()
            total = loss_pos
            return LossOutput(
                loss=total,
                metrics={
                    "arcface/base_loss": total.detach(),
                    "arcface/pos_only": total.detach(),
                    "arcface/valid_pos_frac": p_mask.float().mean().detach(),
                },
            )

        cos_an = torch.matmul(anchor_n.float(), neg_n.float().transpose(1, 2)).squeeze(1) # type: ignore
        cos_an_exp = cos_an.unsqueeze(1).expand(B, P, K)
        logits = torch.cat([cos_ap_m.unsqueeze(-1), cos_an_exp], dim=-1)

        neg_large = _neg_large(logits.dtype)
        if n_mask is not None:
            n_ok = n_mask.unsqueeze(1).expand(B, P, K)
            logits[..., 1:] = logits[..., 1:].masked_fill(~n_ok, neg_large)

        logit_scale = self.scale / max(self.temperature, 1e-6)
        logits = logits * logit_scale

        logits_flat = logits.view(B * P, 1 + K)
        p_mask_flat = p_mask.view(B * P)
        targets = torch.zeros(int(p_mask_flat.sum().item()), dtype=torch.long, device=device)
        loss_main = F.cross_entropy(logits_flat[p_mask_flat], targets)

        metrics = {
            "arcface/base_loss": loss_main.detach(),
            "arcface/valid_pos_frac": p_mask.float().mean().detach(),
            "arcface/valid_neg_frac": (n_mask.float().mean().detach() if n_mask is not None else anchor.new_tensor(1.0)),
        }
        return LossOutput(loss=loss_main, metrics=metrics)


class ExtendedAngularContrastiveLoss(StandardizedLoss):
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.5,
        scale: float = 64.0,
        alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.base = BaselineAngularContrastiveLoss(temperature=temperature, margin=margin, scale=scale)
        self.alpha = float(alpha)
        self.s = float(scale)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        **kwargs,
    ) -> LossOutput:
        out1 = self.base(anchor, positive, negative, pos_mask, neg_mask, eps=eps)
        loss_main = out1.loss

        if negative is None:
            return out1

        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(1)
        elif anchor.dim() == 3 and anchor.shape[1] != 1:
            anchor = anchor[:, :1, :]

        if negative.dim() == 2:
            negative = negative.unsqueeze(1)

        anchor_n = l2_normalize(anchor, eps=eps)
        pos_emb = l2_normalize(positive, eps=eps)
        neg_emb = l2_normalize(negative, eps=eps)

        B, P, D = pos_emb.shape
        K = neg_emb.shape[1]
        device = anchor_n.device

        p_mask = pos_mask.bool() if pos_mask is not None else torch.ones((B, P), dtype=torch.bool, device=device)
        n_mask = neg_mask.bool() if neg_mask is not None else torch.ones((B, K), dtype=torch.bool, device=device)

        cos_hard = torch.matmul(anchor_n.float(), neg_emb.float().transpose(1, 2)).squeeze(1)

        flat_global = gather_with_grad(pos_emb).view(-1, D)
        cos_batch = torch.matmul(anchor_n.squeeze(1).float(), flat_global.float().T)

        global_pos_mask = gather_tensor(p_mask).bool()
        flat_global_mask = global_pos_mask.view(-1)
        cand_mask_batch = flat_global_mask.unsqueeze(0).expand(B, -1)

        rank = _rank()
        global_ids = torch.arange(flat_global.shape[0], device=device).unsqueeze(0).expand(B, -1)
        start_idx = (rank * B + torch.arange(B, device=device)).unsqueeze(1) * P
        own_mask = (global_ids >= start_idx) & (global_ids < (start_idx + P))
        cand_mask_batch = cand_mask_batch & (~own_mask)

        neg_large = _neg_large(cos_batch.dtype)
        cos_batch_clean = cos_batch.masked_fill(~cand_mask_batch, neg_large)
        has_any_batch = cand_mask_batch.any(dim=1)

        losses_tier = []
        logit_scale = self.s
        for k in range(K):
            valid_k = n_mask[:, k] & has_any_batch
            if not valid_k.any():
                continue
            logits_hard = cos_hard[:, k].unsqueeze(1)
            logits = torch.cat([logits_hard, cos_batch_clean], dim=1) * logit_scale
            tgt = torch.zeros(int(valid_k.sum().item()), dtype=torch.long, device=device)
            losses_tier.append(F.cross_entropy(logits[valid_k], tgt))

        if losses_tier:
            loss_tier = torch.stack(losses_tier).mean()
        else:
            loss_tier = anchor_n.new_tensor(0.0, requires_grad=True)

        total = loss_main + self.alpha * loss_tier

        metrics: Dict[str, torch.Tensor] = dict(out1.metrics)
        metrics.update(
            {
                "arcface_ext/loss": total.detach(),
                "arcface_ext/main": loss_main.detach(),
                "arcface_ext/tier2": loss_tier.detach(),
                "arcface_ext/has_batch_rate": _dist_ratio(has_any_batch.float().sum(), anchor_n.new_tensor(B)).detach(),
                "arcface_ext/has_pos_rate": _dist_ratio(p_mask.any(dim=1).float().sum(), anchor_n.new_tensor(B)).detach(),
                "arcface_ext/has_hard_rate": _dist_ratio(n_mask.any(dim=1).float().sum(), anchor_n.new_tensor(B)).detach(),
            }
        )
        return LossOutput(loss=total, metrics=metrics)


class ExtendedAngularContrastiveLossV3(StandardizedLoss):
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.5,
        scale: float = 64.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        scale3: float = 32.0,
        **kwargs,
    ):
        super().__init__()
        self.ext2 = ExtendedAngularContrastiveLoss(
            temperature=temperature, margin=margin, scale=scale, alpha=alpha
        )
        self.beta = float(beta)
        self.scale3 = float(scale3)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        **kwargs,
    ) -> LossOutput:
        out2 = self.ext2(anchor, positive, negative, pos_mask, neg_mask, eps=eps, **kwargs)
        if negative is None:
            return out2

        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(1)
        elif anchor.dim() == 3 and anchor.shape[1] != 1:
            anchor = anchor[:, :1, :]

        if negative.dim() == 2:
            negative = negative.unsqueeze(1)

        anchor_n = l2_normalize(anchor, eps=eps)
        pos_n = l2_normalize(positive, eps=eps)
        neg_n = l2_normalize(negative, eps=eps)

        B, P, _ = pos_n.shape
        K = neg_n.shape[1]
        device = anchor_n.device

        p_mask = pos_mask.bool() if pos_mask is not None else torch.ones((B, P), dtype=torch.bool, device=device)
        n_mask = neg_mask.bool() if neg_mask is not None else torch.ones((B, K), dtype=torch.bool, device=device)

        sim_pos = torch.matmul(anchor_n.float(), pos_n.float().transpose(1, 2)).squeeze(1)
        sim_hard = torch.matmul(anchor_n.float(), neg_n.float().transpose(1, 2)).squeeze(1)

        pos_inf = torch.tensor(torch.finfo(sim_pos.dtype).max, device=device, dtype=sim_pos.dtype)
        pos_min = sim_pos.masked_fill(~p_mask, pos_inf).min(dim=1).values

        neg_large = _neg_large(sim_hard.dtype)
        hard_max = sim_hard.masked_fill(~n_mask, neg_large).max(dim=1).values

        valid_rows = p_mask.any(dim=1) & n_mask.any(dim=1)
        if valid_rows.any():
            diff = (hard_max - pos_min) * self.scale3
            loss3 = F.softplus(diff[valid_rows]).mean()
        else:
            loss3 = anchor_n.new_tensor(0.0, requires_grad=True)

        total = out2.loss + self.beta * loss3

        metrics = dict(out2.metrics)
        metrics.update(
            {
                "arcface_ext/loss": total.detach(),
                "arcface_ext/tier3": loss3.detach(),
                "arcface_ext/tier3_valid_rate": _dist_ratio(valid_rows.float().sum(), anchor_n.new_tensor(B)).detach(),
            }
        )
        return LossOutput(loss=total, metrics=metrics)


AngularContrastiveLoss = BaselineAngularContrastiveLoss
ArcFaceExtendedV2 = ExtendedAngularContrastiveLoss
ArcFaceExtendedV3 = ExtendedAngularContrastiveLossV3

class ExtendedSigLIPLoss(StandardizedLoss):
    def __init__(self, temperature: float = 0.07, bias: float = -10.0,
                 learnable: bool = False, alpha: float = 0.5, **kwargs):
        super().__init__()
        self.alpha = alpha
        t_val = torch.tensor(1.0 / temperature)
        b_val = torch.tensor(bias)
        if learnable:
            self.t = nn.Parameter(t_val)
            self.b = nn.Parameter(b_val)
        else:
            self.register_buffer("t", t_val)
            self.register_buffer("b", b_val)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: Optional[torch.Tensor] = None,
                pos_mask: Optional[torch.Tensor] = None,
                neg_mask: Optional[torch.Tensor] = None,
                eps: float = 1e-7,
                **kwargs) -> LossOutput:

        # ---- normalize / shape ----
        anchor = l2_normalize(anchor if anchor.dim() == 3 else anchor.unsqueeze(1))  # (B,1,D)
        pos_emb = l2_normalize(positive)                                            # (B,P,D)
        B, P, D = pos_emb.shape

        p_mask = pos_mask.bool() if pos_mask is not None else torch.ones((B, P), dtype=torch.bool, device=anchor.device)

        # ---- global pool ----
        global_pos = gather_with_grad(pos_emb)   # (B_global,P,D)
        flat_global = global_pos.view(-1, D)     # (M,D), M = world*B*P

        # logits over global pool (scaled + biased)
        logits = torch.matmul(anchor.squeeze(1), flat_global.T) * self.t + self.b  # (B,M)

        # label matrix: +1 at own positives, -1 elsewhere
        labels = torch.full_like(logits, -1.0)

        rank = torch.distributed.get_rank() if _is_distributed() else 0
        row_indices = torch.arange(B, device=anchor.device).unsqueeze(1).expand(-1, P)
        col_indices = (rank * B + torch.arange(B, device=anchor.device)).unsqueeze(1) * P + torch.arange(P, device=anchor.device)

        # set +1 for valid positives only
        labels[row_indices[p_mask], col_indices[p_mask]] = 1.0

        # column weights: only count columns that correspond to valid global positives
        global_mask = gather_tensor(p_mask).view(-1).float().unsqueeze(0)  # (1,M)

        # ---- main SigLIP loss ----
        loss_main_mat = F.binary_cross_entropy_with_logits(logits, (labels > 0).float(), reduction="none")
        loss_main = (loss_main_mat * global_mask).sum() / (global_mask.sum() * B + eps)

        # ---- explicit negatives (optional) ----
        loss_hard = torch.tensor(0.0, device=anchor.device, requires_grad=True)
        loss_tier = torch.tensor(0.0, device=anchor.device, requires_grad=True)

        logits_hard = None
        sim_hard = None
        n_mask = None
        K = 0

        if negative is not None:
            if negative.dim() == 2:
                negative = negative.unsqueeze(1)
            K = negative.shape[1]
            if K > 0:
                neg_emb = l2_normalize(negative)  # (B,K,D)
                n_mask = neg_mask.bool() if neg_mask is not None else torch.ones((B, K), dtype=torch.bool, device=anchor.device)

                logits_hard = torch.matmul(anchor, neg_emb.transpose(1, 2)).squeeze(1) * self.t + self.b  # (B,K)
                sim_hard = torch.matmul(anchor, neg_emb.transpose(1, 2)).squeeze(1)                       # (B,K) cosine

                if n_mask.any():
                    loss_hard = F.binary_cross_entropy_with_logits(
                        logits_hard[n_mask], torch.zeros_like(logits_hard[n_mask])
                    )

                # ---- Tier: hard > batch (ranking) ----
                # build batch cosine for ranking baseline
                cos_batch = torch.matmul(anchor.squeeze(1), flat_global.T)  # (B,M) cosine

                neg_inf = _neg_inf(cos_batch)
                # mask own positives in batch pool
                cos_batch[row_indices, col_indices] = neg_inf

                # also mask invalid global positives
                valid_cols = gather_tensor(p_mask).view(-1).bool()  # (M,)
                cos_batch = cos_batch.masked_fill(~valid_cols.unsqueeze(0), neg_inf)

                # LSE over batch (scaled, biased)
                lse_batch = torch.logsumexp(cos_batch * self.t + self.b, dim=1, keepdim=True)  # (B,1)
                ranking_logits = logits_hard - lse_batch  # (B,K)

                if n_mask.any():
                    loss_tier = -F.logsigmoid(ranking_logits[n_mask]).mean()

        total = loss_main + loss_hard + self.alpha * loss_tier

        # ---- extra monitors (distributed-safe) ----
        metrics: Dict[str, torch.Tensor] = {
            "siglip_ext/loss": total.detach(),
            "siglip_ext/main": loss_main.detach(),
            "siglip_ext/hard": loss_hard.detach(),
            "siglip_ext/tier": loss_tier.detach(),
        }

        with torch.no_grad():
            def _rate(ok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                num = (ok & mask).float().sum()
                den = mask.float().sum()
                return _dist_ratio(num, den, eps).detach()

            # construct batch candidate mask (valid global positives excluding own positives)
            global_pos_mask = gather_tensor(p_mask).bool()     # (B_global,P)
            flat_global_mask = global_pos_mask.view(-1)        # (M,)
            cand_mask_batch = flat_global_mask.unsqueeze(0).expand(B, -1)  # (B,M)

            global_ids = torch.arange(flat_global.shape[0], device=anchor.device).unsqueeze(0).expand(B, -1)
            own_mask = (global_ids >= (rank * B + torch.arange(B, device=anchor.device)).unsqueeze(1) * P) & \
                       (global_ids <  ((rank * B + torch.arange(B, device=anchor.device)).unsqueeze(1) * P + P))

            cand_mask_batch = cand_mask_batch & (~own_mask)

            # pos logits (for own positives)
            idx_mat = (rank * B + torch.arange(B, device=anchor.device)).unsqueeze(1) * P + torch.arange(P, device=anchor.device)
            pos_logits = logits.gather(1, idx_mat)  # (B,P)

            neg_inf_logits = _neg_inf(logits)
            pos_inf_logits = torch.tensor(torch.finfo(logits.dtype).max, device=logits.device, dtype=logits.dtype)

            pos_min_logit = pos_logits.masked_fill(~p_mask, pos_inf_logits).min(dim=1).values
            pos_max_logit = pos_logits.masked_fill(~p_mask, neg_inf_logits).max(dim=1).values

            # batch max logit
            batch_max_logit = logits.masked_fill(~cand_mask_batch, neg_inf_logits).max(dim=1).values

            has_pos = p_mask.any(dim=1)
            has_batch = cand_mask_batch.any(dim=1)

            metrics["siglip_ext/has_pos_rate"] = _dist_ratio(has_pos.float().sum(), anchor.new_tensor(B), eps).detach()
            metrics["siglip_ext/has_batch_rate"] = _dist_ratio(has_batch.float().sum(), anchor.new_tensor(B), eps).detach()

            # pos > batch
            base_mask = has_pos & has_batch
            metrics["siglip_ext/pos_gt_batch_rate"] = _rate(pos_min_logit > batch_max_logit, base_mask)
            metrics["siglip_ext/pos_minus_maxbatch_mean"] = _dist_ratio(
                ((pos_min_logit - batch_max_logit) * base_mask.float()).sum(),
                base_mask.float().sum(),
                eps,
            ).detach()

            # hard-related (only if K>0)
            if logits_hard is not None and n_mask is not None and n_mask.any():
                has_hard = n_mask.any(dim=1)
                metrics["siglip_ext/has_hard_rate"] = _dist_ratio(has_hard.float().sum(), anchor.new_tensor(B), eps).detach()

                hard_max_logit = logits_hard.masked_fill(~n_mask, neg_inf_logits).max(dim=1).values

                trio_mask = has_pos & has_batch & has_hard

                metrics["siglip_ext/pos_gt_hard_rate"] = _rate(pos_min_logit > hard_max_logit, trio_mask)
                metrics["siglip_ext/hard_gt_batch_rate"] = _rate(hard_max_logit > batch_max_logit, trio_mask)
                metrics["siglip_ext/full_order_rate"] = _rate((pos_min_logit > hard_max_logit) & (hard_max_logit > batch_max_logit), trio_mask)

                metrics["siglip_ext/pos_minus_hard_mean"] = _dist_ratio(
                    ((pos_min_logit - hard_max_logit) * trio_mask.float()).sum(),
                    trio_mask.float().sum(),
                    eps,
                ).detach()

                metrics["siglip_ext/hard_minus_batch_mean"] = _dist_ratio(
                    ((hard_max_logit - batch_max_logit) * trio_mask.float()).sum(),
                    trio_mask.float().sum(),
                    eps,
                ).detach()

                # fraction of hard neg slots beating max batch neg
                batch_max_exp = batch_max_logit.unsqueeze(1).expand_as(logits_hard)
                valid_k = n_mask
                metrics["siglip_ext/hard_gt_maxbatch_frac"] = _dist_ratio(
                    (((logits_hard > batch_max_exp) & valid_k).float().sum()),
                    valid_k.float().sum(),
                    eps,
                ).detach()

                # tier acc: hard > lse_batch (if tier computed)
                # (approx: ranking_logits > 0 for valid slots)
                # compute only if we had tier
                # NOTE: ranking_logits lives only inside the earlier block; recompute cheaply:
                cos_batch = torch.matmul(anchor.squeeze(1), flat_global.T)
                neg_inf = _neg_inf(cos_batch)
                cos_batch[row_indices, col_indices] = neg_inf
                valid_cols = flat_global_mask.bool()
                cos_batch = cos_batch.masked_fill(~valid_cols.unsqueeze(0), neg_inf)
                lse_batch = torch.logsumexp(cos_batch * self.t + self.b, dim=1, keepdim=True)
                ranking_logits = logits_hard - lse_batch
                metrics["siglip_ext/tier_acc"] = _dist_ratio(
                    (((ranking_logits > 0) & valid_k).float().sum()),
                    valid_k.float().sum(),
                    eps,
                ).detach()

        return LossOutput(loss=total, metrics=metrics)




# assumes you already have these in your module:
# - _is_distributed
# - gather_with_grad
# - gather_tensor
# - l2_normalize
# - LossOutput
# - StandardizedLoss

def _neg_inf(x: torch.Tensor) -> torch.Tensor:
    # return torch.finfo(dtype).min
    return torch.tensor(torch.finfo(x.dtype).min, device=x.device, dtype=x.dtype)

def _masked_logsumexp(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x: (..., N)
    mask: same shape as x, bool
    returns logsumexp over dim with masked-out entries removed.
    NOTE: rows with all mask=False will return -inf.
    """
    x = x.masked_fill(~mask, _neg_inf(x))
    return torch.logsumexp(x, dim=dim)

def _dist_sum(x: torch.Tensor) -> torch.Tensor:
    if not _is_distributed():
        return x
    x = x.clone()
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x

def _dist_ratio(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return _dist_sum(num) / (_dist_sum(den) + eps)


class BaselineSymmetricContrastiveLoss(StandardizedLoss):
    """
    CLIP-style symmetric contrastive loss with MULTIPLE positives per anchor.

    Inputs (your convention):
      anchor:   (B,D) or (B,1,D)
      positive: (B,P,D)
      negative: optional (B,K,D) -> treated as additional negatives (same strength as batch)
      pos_mask: (B,P) optional
      neg_mask: (B,K) optional

    t2i (anchor -> positives-as-classes):
      - Global candidate set = all positives in global batch (flattened) [+ optional explicit negatives]
      - Multi-positive numerator: logsumexp over the anchor’s own valid P positives
      - Denominator: logsumexp over all valid candidates

    i2t (positive -> anchors-as-classes):
      - Each valid positive must classify back to its anchor in global anchor pool (standard CE).
    """

    def __init__(self, temperature: float = 0.07, **kwargs):
        super().__init__()
        self.T = float(temperature)

    @staticmethod
    def _get_rank() -> int:
        return torch.distributed.get_rank() if _is_distributed() else 0

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        **kwargs
    ) -> LossOutput:

        # ---- normalize / shape ----
        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(1)          # (B,1,D)
        anchor = l2_normalize(anchor, eps=eps)    # (B,1,D)

        pos_emb = l2_normalize(positive, eps=eps) # (B,P,D)
        B, P, D = pos_emb.shape

        if pos_mask is None:
            pos_mask = torch.ones((B, P), device=pos_emb.device, dtype=torch.bool)
        else:
            pos_mask = pos_mask.bool()

        rank = self._get_rank()

        # ---------------------------------------------------------------------
        # t2i: anchor -> candidate positives (global) [+ optional explicit negs]
        # ---------------------------------------------------------------------
        global_pos = gather_with_grad(pos_emb)              # (B_global,P,D)
        flat_global = global_pos.view(-1, D)                # (N_batch,D)
        logits_batch = (anchor.squeeze(1) @ flat_global.T) / self.T  # (B,N_batch)

        global_pos_mask = gather_tensor(pos_mask).bool()    # (B_global,P)
        flat_global_mask = global_pos_mask.view(-1)         # (N_batch,)
        cand_mask_batch = flat_global_mask.unsqueeze(0).expand(B, -1)  # (B,N_batch)

        logits_all = logits_batch
        cand_mask_all = cand_mask_batch

        # Optional explicit negatives: appended as extra candidate columns
        if negative is not None:
            if negative.dim() == 2:
                negative = negative.unsqueeze(1)
            K = negative.shape[1]
            if K > 0:
                neg_emb = l2_normalize(negative, eps=eps)   # (B,K,D)
                logits_hard = (anchor @ neg_emb.transpose(1, 2)).squeeze(1) / self.T  # (B,K)

                if neg_mask is None:
                    hard_mask = torch.ones((B, K), device=pos_emb.device, dtype=torch.bool)
                else:
                    hard_mask = neg_mask.bool()

                logits_all = torch.cat([logits_all, logits_hard], dim=1)      # (B,N_batch+K)
                cand_mask_all = torch.cat([cand_mask_all, hard_mask], dim=1)  # (B,N_batch+K)

        # Gather indices of "this anchor’s own positives" in the global candidate list.
        # In gather order, local sample i is at global sample (rank*B + i).
        base = (rank * B + torch.arange(B, device=pos_emb.device)) * P        # (B,)
        idx_mat = base.unsqueeze(1) + torch.arange(P, device=pos_emb.device)  # (B,P)

        # pull logits at those positions: (B,P)
        logits_own = logits_batch.gather(1, idx_mat)

        # mask invalid positives
        logits_own = logits_own.masked_fill(~pos_mask, _neg_inf(logits_own))

        # numerator: logsumexp over own valid positives
        num = torch.logsumexp(logits_own, dim=1)  # (B,)

        # denominator: logsumexp over all valid candidates
        den = _masked_logsumexp(logits_all, cand_mask_all, dim=1)  # (B,)

        # only anchors with at least one positive and at least one candidate
        has_pos = pos_mask.any(dim=1)
        has_cand = cand_mask_all.any(dim=1)
        valid_t2i = has_pos & has_cand & torch.isfinite(num) & torch.isfinite(den)

        loss_t2i = anchor.new_tensor(0.0, requires_grad=True)
        if valid_t2i.any():
            loss_t2i = (den[valid_t2i] - num[valid_t2i]).mean()

        # ---------------------------------------------------------------------
        # i2t: each positive -> global anchors (standard CE)
        # ---------------------------------------------------------------------
        anchor_q = anchor.squeeze(1)                             # (B,D)
        global_anchors = gather_with_grad(anchor_q)              # (B_global,D)

        flat_pos = pos_emb.reshape(B * P, D)                     # (B*P,D)
        flat_mask = pos_mask.reshape(B * P)                      # (B*P,)

        logits_i2t = (flat_pos @ global_anchors.T) / self.T      # (B*P, B_global)

        target = (rank * B + torch.arange(B, device=pos_emb.device)).repeat_interleave(P)  # (B*P,)

        loss_i2t = anchor.new_tensor(0.0, requires_grad=True)
        if flat_mask.any():
            loss_i2t = F.cross_entropy(logits_i2t[flat_mask], target[flat_mask])

        total = 0.5 * (loss_t2i + loss_i2t)

        # light, distributed-safe metrics
        with torch.no_grad():
            t2i_count = valid_t2i.float().sum()
            i2t_count = flat_mask.float().sum()
            t2i_rate = _dist_ratio(t2i_count, anchor.new_tensor(B), eps).detach()
            i2t_rate = _dist_ratio(i2t_count, anchor.new_tensor(B * P), eps).detach()

        return LossOutput(
            loss=total,
            metrics={
                "clip/loss": total.detach(),
                "clip/t2i": loss_t2i.detach(),
                "clip/i2t": loss_i2t.detach(),
                "clip/t2i_valid_rate": t2i_rate,
                "clip/i2t_valid_rate": i2t_rate,
            },
        )


class ExtendedSymmetricContrastiveLoss(StandardizedLoss):
    """
    Extended symmetric contrastive loss:
    - same as BaselineSymmetricContrastiveLoss
    - BUT explicit hard negatives are added with STRONGER weight in the denominator

    Mechanism: exp(logit_hard) is multiplied by hard_weight
      => logit_hard' = logit_hard + log(hard_weight)

    This is a clean way to make hard negatives contribute more strongly than
    regular batch negatives without changing the basic CLIP objective.
    """

    def __init__(self, temperature: float = 0.07, hard_weight: float = 4.0, **kwargs):
        super().__init__()
        self.T = float(temperature)
        if hard_weight <= 0:
            raise ValueError("hard_weight must be > 0")
        self.hard_weight = float(hard_weight)

    @staticmethod
    def _get_rank() -> int:
        return torch.distributed.get_rank() if _is_distributed() else 0

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        **kwargs
    ) -> LossOutput:

        # ---- normalize / shape ----
        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(1)
        anchor = l2_normalize(anchor, eps=eps)

        pos_emb = l2_normalize(positive, eps=eps)
        B, P, D = pos_emb.shape

        if pos_mask is None:
            pos_mask = torch.ones((B, P), device=pos_emb.device, dtype=torch.bool)
        else:
            pos_mask = pos_mask.bool()

        rank = self._get_rank()

        # ---------------------------------------------------------------------
        # t2i (multi-positive) with hard-negative reweighting in the denominator
        # ---------------------------------------------------------------------
        global_pos = gather_with_grad(pos_emb)
        flat_global = global_pos.view(-1, D)
        logits_batch = (anchor.squeeze(1) @ flat_global.T) / self.T  # (B,N_batch)

        global_pos_mask = gather_tensor(pos_mask).bool()
        flat_global_mask = global_pos_mask.view(-1)
        cand_mask_batch = flat_global_mask.unsqueeze(0).expand(B, -1)

        logits_all = logits_batch
        cand_mask_all = cand_mask_batch

        hard_present = anchor.new_tensor(0.0)
        if negative is not None:
            if negative.dim() == 2:
                negative = negative.unsqueeze(1)
            K = negative.shape[1]
            if K > 0:
                neg_emb = l2_normalize(negative, eps=eps)
                logits_hard = (anchor @ neg_emb.transpose(1, 2)).squeeze(1) / self.T  # (B,K)

                if neg_mask is None:
                    hard_mask = torch.ones((B, K), device=pos_emb.device, dtype=torch.bool)
                else:
                    hard_mask = neg_mask.bool()

                # reweight hard negatives in denom
                logits_hard = logits_hard + math.log(self.hard_weight)

                logits_all = torch.cat([logits_all, logits_hard], dim=1)
                cand_mask_all = torch.cat([cand_mask_all, hard_mask], dim=1)

                hard_present = hard_mask.any(dim=1).float().mean()

        base = (rank * B + torch.arange(B, device=pos_emb.device)) * P
        idx_mat = base.unsqueeze(1) + torch.arange(P, device=pos_emb.device)
        logits_own = logits_batch.gather(1, idx_mat).masked_fill(~pos_mask, _neg_inf(logits_batch))

        num = torch.logsumexp(logits_own, dim=1)
        den = _masked_logsumexp(logits_all, cand_mask_all, dim=1)

        has_pos = pos_mask.any(dim=1)
        has_cand = cand_mask_all.any(dim=1)
        valid_t2i = has_pos & has_cand & torch.isfinite(num) & torch.isfinite(den)

        loss_t2i = anchor.new_tensor(0.0, requires_grad=True)
        if valid_t2i.any():
            loss_t2i = (den[valid_t2i] - num[valid_t2i]).mean()

        # ---------------------------------------------------------------------
        # i2t (same as baseline)
        # ---------------------------------------------------------------------
        anchor_q = anchor.squeeze(1)
        global_anchors = gather_with_grad(anchor_q)

        flat_pos = pos_emb.reshape(B * P, D)
        flat_mask = pos_mask.reshape(B * P)

        logits_i2t = (flat_pos @ global_anchors.T) / self.T
        target = (rank * B + torch.arange(B, device=pos_emb.device)).repeat_interleave(P)

        loss_i2t = anchor.new_tensor(0.0, requires_grad=True)
        if flat_mask.any():
            loss_i2t = F.cross_entropy(logits_i2t[flat_mask], target[flat_mask])

        total = 0.5 * (loss_t2i + loss_i2t)

        with torch.no_grad():
            t2i_count = valid_t2i.float().sum()
            t2i_rate = _dist_ratio(t2i_count, anchor.new_tensor(B), eps).detach()

        return LossOutput(
            loss=total,
            metrics={
                "clip_ext/loss": total.detach(),
                "clip_ext/t2i": loss_t2i.detach(),
                "clip_ext/i2t": loss_i2t.detach(),
                "clip_ext/t2i_valid_rate": t2i_rate,
                "clip_ext/hard_present_rate": hard_present.detach(),
                "clip_ext/hard_weight": anchor.new_tensor(self.hard_weight).detach(),
            },
        )


class CLIPOneLoss(StandardizedLoss):
    """
    CLIP-style symmetric InfoNCE with EXACTLY ONE positive per anchor.
    No hard negatives: negatives are the other items in the (distributed) global batch.

    Inputs:
      anchor:   (B,D) or (B,1,D)
      positive: (B,D) or (B,1,D) or (B,P,D)  (if P>1, we select one per anchor)
      pos_mask: (B,P) optional, used only when positive is (B,P,D)

    This aligns to CLIP training:
      loss = 0.5 * ( CE(text->image) + CE(image->text) )
    """

    def __init__(
        self,
        temperature: float = 0.07,
        select: str = "random",          # {"random","first"}
        learnable_scale: bool = False,   # CLIP uses learnable logit_scale; optional
        clamp_max: float = 100.0,        # CLIP clamps exp(logit_scale) to 100
        include_negatives: bool = False, # append MCN distractors as extra negatives in softmax
        **kwargs
    ):
        super().__init__()
        self.T = float(temperature)
        self.select = str(select)
        self.clamp_max = float(clamp_max)
        self.include_negatives = bool(include_negatives)

        if learnable_scale:
            # store logit_scale like CLIP (log of scale)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / self.T)))
        else:
            self.register_buffer("logit_scale", torch.tensor(math.log(1.0 / self.T)))

    @staticmethod
    def _get_rank() -> int:
        return torch.distributed.get_rank() if _is_distributed() else 0

    def _pick_one_positive(
        self,
        positive: torch.Tensor,
        pos_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          pos_one: (B,D)
          valid:   (B,) bool
        """
        if positive.dim() == 2:
            B = positive.shape[0]
            return positive, torch.ones((B,), device=positive.device, dtype=torch.bool)

        if positive.dim() == 3 and positive.shape[1] == 1:
            B = positive.shape[0]
            return positive[:, 0], torch.ones((B,), device=positive.device, dtype=torch.bool)

        if positive.dim() != 3:
            raise ValueError(f"positive must be (B,D), (B,1,D) or (B,P,D). Got {tuple(positive.shape)}")

        B, P, D = positive.shape
        if P <= 0:
            return positive.new_zeros((B, D)), torch.zeros((B,), device=positive.device, dtype=torch.bool)

        if self.select == "first":
            idx = torch.zeros((B,), device=positive.device, dtype=torch.long)
            valid = torch.ones((B,), device=positive.device, dtype=torch.bool)
            if pos_mask is not None:
                pm = pos_mask.bool()
                valid = pm.any(dim=1)
                # pick first True per row; fallback 0 if none (then masked out by valid)
                idx = pm.float().argmax(dim=1)
            pos_one = positive[torch.arange(B, device=positive.device), idx]
            return pos_one, valid

        # default: random
        if pos_mask is None:
            idx = torch.randint(0, P, (B,), device=positive.device)
            valid = torch.ones((B,), device=positive.device, dtype=torch.bool)
            pos_one = positive[torch.arange(B, device=positive.device), idx]
            return pos_one, valid

        pm = pos_mask.bool()
        weights = pm.float()
        has = weights.sum(dim=1) > 0
        # multinomial errors if a row sums to 0 -> make it safe; we'll mask those rows out via `has`
        weights_safe = weights.clone()
        weights_safe[~has] = 1.0
        idx = torch.multinomial(weights_safe, 1).squeeze(1)
        pos_one = positive[torch.arange(B, device=positive.device), idx]
        return pos_one, has

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,  # used when include_negatives=True
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        **kwargs
    ) -> LossOutput:
        # shape + normalize
        if anchor.dim() == 3 and anchor.shape[1] == 1:
            anchor = anchor[:, 0]
        elif anchor.dim() != 2:
            raise ValueError(f"anchor must be (B,D) or (B,1,D). Got {tuple(anchor.shape)}")

        pos_one, valid = self._pick_one_positive(positive, pos_mask)

        anchor = l2_normalize(anchor, eps=eps)     # (B,D)
        pos_one = l2_normalize(pos_one, eps=eps)   # (B,D)

        B, D = anchor.shape
        if B == 0:
            z = anchor.new_tensor(0.0, requires_grad=True)
            return LossOutput(loss=z, metrics={"clip_one/loss": z.detach()})

        # distributed global pools (with grad)
        global_anchor = gather_with_grad(anchor)   # (N,D)
        global_pos = gather_with_grad(pos_one)     # (N,D)

        # CLIP logit scale
        scale = self.logit_scale.exp().clamp(max=self.clamp_max)

        logits_t2i = (anchor @ global_pos.T) * scale  # (B,N)
        logits_i2t = (pos_one @ global_anchor.T) * scale  # (B,N)

        # Optionally append MCN distractors as extra negatives in softmax
        K = 0
        if self.include_negatives and negative is not None:
            neg_emb = l2_normalize(negative, eps=eps)  # (B,K,D)
            if neg_emb.dim() == 3:
                K = neg_emb.shape[1]
                # Compute sim to distractors: (B, K)
                sim_neg_t2i = torch.einsum('bd,bkd->bk', anchor, neg_emb) * scale
                sim_neg_i2t = torch.einsum('bd,bkd->bk', pos_one, neg_emb) * scale
                # Mask invalid negatives
                if neg_mask is not None:
                    neg_inf = _neg_inf(sim_neg_t2i)
                    sim_neg_t2i = sim_neg_t2i.masked_fill(~neg_mask.bool(), neg_inf)
                    sim_neg_i2t = sim_neg_i2t.masked_fill(~neg_mask.bool(), neg_inf)
                # Append: logits become (B, N+K)
                logits_t2i = torch.cat([logits_t2i, sim_neg_t2i], dim=1)
                logits_i2t = torch.cat([logits_i2t, sim_neg_i2t], dim=1)

        rank = self._get_rank()
        target = rank * B + torch.arange(B, device=anchor.device)

        # ignore rows with no valid positive (if using pos_mask and some rows empty)
        loss_t2i = anchor.new_tensor(0.0, requires_grad=True)
        loss_i2t = anchor.new_tensor(0.0, requires_grad=True)

        if valid.any():
            loss_t2i = F.cross_entropy(logits_t2i[valid], target[valid])
            loss_i2t = F.cross_entropy(logits_i2t[valid], target[valid])

        total = 0.5 * (loss_t2i + loss_i2t)

        # metrics (distributed-safe)
        with torch.no_grad():
            vcount = valid.float().sum()

            t2i_ok = (logits_t2i.argmax(dim=1) == target) & valid
            i2t_ok = (logits_i2t.argmax(dim=1) == target) & valid

            t2i_acc = _dist_ratio(t2i_ok.float().sum(), vcount, eps).detach()
            i2t_acc = _dist_ratio(i2t_ok.float().sum(), vcount, eps).detach()
            valid_rate = _dist_ratio(vcount, anchor.new_tensor(B), eps).detach()

        return LossOutput(
            loss=total,
            metrics={
                "clip_one/loss": total.detach(),
                "clip_one/t2i": loss_t2i.detach(),
                "clip_one/i2t": loss_i2t.detach(),
                "clip_one/t2i_acc": t2i_acc,
                "clip_one/i2t_acc": i2t_acc,
                "clip_one/valid_rate": valid_rate,
                "clip_one/logit_scale": scale.detach(),
                "clip_one/neg_count": torch.tensor(K, device=anchor.device),
            },
        )

# ==============================================================================
# SECTION G: Combined / Weighted Loss
# ==============================================================================

import inspect

def _filter_kwargs_for_ctor(cls, kwargs: dict) -> dict:
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    # if ctor has **kwargs, we can pass everything
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    return kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in allowed}

class CombinedLoss(StandardizedLoss):
    """
    Combines multiple losses with configurable weights.
    
    Usage:
        # String format: "name:weight+name:weight" or "name:weight:kwarg=val"
        loss = CombinedLoss("infonce:1.0+circle:0.5+triplet:0.2")
        loss = CombinedLoss("infonce:1.0:temperature=0.1+circle:0.5:m=0.3")
        
        # Dict format: {name: weight} or {name: {"weight": w, **kwargs}}
        loss = CombinedLoss({"infonce": 1.0, "circle": 0.5})
        loss = CombinedLoss({"infonce": {"weight": 1.0, "temperature": 0.1}})
        
        # List format: [(name, weight, kwargs), ...]
        loss = CombinedLoss([("infonce", 1.0, {"temperature": 0.1})])
    """
    
    def __init__(self, 
                 config: Union[str, Dict, list], 
                 normalize_weights: bool = False,
                 **kwargs):
        super().__init__()
        self.losses, self.weights, self.names = [], [], []
        self._parse_config(config)
        if normalize_weights:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
        self.losses = nn.ModuleList(self.losses)
    
    def _parse_config(self, config: Union[str, Dict, list]):
        if isinstance(config, str):
            for part in config.split("+"):
                tokens = part.strip().split(":")
                name, weight = tokens[0], float(tokens[1]) if len(tokens) > 1 else 1.0
                kwargs = dict(t.split("=") for t in tokens[2:]) if len(tokens) > 2 else {}
                kwargs = {k: self._cast(v) for k, v in kwargs.items()}
                self._add(name, weight, kwargs)
        elif isinstance(config, dict):
            for name, val in config.items():
                if isinstance(val, (int, float)):
                    self._add(name, float(val), {})
                else:
                    w = val.pop("weight", 1.0)
                    self._add(name, w, val)
        elif isinstance(config, list):
            for item in config:
                name, weight, kwargs = item if len(item) == 3 else (*item, {})
                self._add(name, weight, kwargs)
    
    def _add(self, name: str, weight: float, kwargs: dict):
        if name not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}")
        self.names.append(name)
        self.weights.append(weight)
        self.losses.append(LOSS_REGISTRY[name](**kwargs))
    
    @staticmethod
    def _cast(v: str):
        try: return int(v)
        except ValueError:
            try: return float(v)
            except ValueError: return v.lower() in ("true", "1", "yes") if v.lower() in ("true", "false", "1", "0", "yes", "no") else v

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: Optional[torch.Tensor] = None,
                pos_mask: Optional[torch.Tensor] = None,
                neg_mask: Optional[torch.Tensor] = None,
                eps: float = 1e-7,
                **kwargs) -> LossOutput:
        
        losses = []
        all_metrics = {}
        
        for name, weight, loss_fn in zip(self.names, self.weights, self.losses):
            out = loss_fn(anchor, positive, negative, pos_mask, neg_mask, **kwargs)
            losses.append(weight * out.loss)
            for k, v in out.metrics.items():
                # Avoid double-prefixing (e.g., "infonce/infonce/loss" → "infonce/loss")
                key = k if k.startswith(f"{name}/") else f"{name}/{k}"
                all_metrics[key] = v
        
        # Stack and sum for proper gradient flow (Accelerate/DDP compatible)
        total_loss = torch.stack(losses).sum() if losses else anchor.new_tensor(0.0)
        
        all_metrics["combined/loss"] = total_loss.detach()
        return LossOutput(loss=total_loss, metrics=all_metrics)

    def __repr__(self):
        parts = [f"{n}:{w:.2f}" for n, w in zip(self.names, self.weights)]
        return f"CombinedLoss({'+'.join(parts)})"

class MarginCosineTripletLoss(StandardizedLoss):
    """
    Cosine-space margin loss with the exact tiering we discussed:

      Tier 1 (quality):    pos > hard + m_pos_hard     (default m_pos_hard = 0.1)
      Tier 2 (separation): hard > batch_max + m_hard_batch   (default m_hard_batch = 0.5)

    where:
      - pos is the (masked) max similarity over P positives for each anchor
      - hard is the (masked) max similarity over K hard negatives for each anchor
      - batch_max is the max similarity to the global-batch pool (gathered positives),
        excluding the anchor's own positives

    Inputs (your convention):
      anchor:   (B,D) or (B,1,D)
      positive: (B,P,D)
      negative: optional (B,K,D) hard negatives
      pos_mask: optional (B,P)
      neg_mask: optional (B,K)

    Notes:
      - Uses global batch negatives (world-size aware) via gather_with_grad/gather_tensor.
      - If negative is None or K==0, we fall back to a simpler objective:
            pos > batch_max + m_pos_hard
    """

    def __init__(self, m_pos_hard: float = 0.1, m_hard_batch: float = 0.05, alpha: float = 0.5):
        super().__init__()
        self.m1 = float(m_pos_hard)
        self.m2 = float(m_hard_batch)
        self.alpha = float(alpha)
        self.relu = nn.ReLU()

    @staticmethod
    def _get_rank() -> int:
        return torch.distributed.get_rank() if _is_distributed() else 0

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        margin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LossOutput:

        if margin is not None:
            m1 = margin.to(anchor.device)  # (B,K) per-slot or (B,) per-sample
            if m1.dim() == 1:
                m1 = m1.unsqueeze(1)  # (B,) -> (B,1) for backward compat
        else:
            m1 = self.m1

        # ---- shape + normalize ----
        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(1)  # (B,1,D)
        anchor = l2_normalize(anchor, eps=eps)  # (B,1,D)

        pos_emb = l2_normalize(positive, eps=eps)  # (B,P,D)
        B, P, D = pos_emb.shape

        if pos_mask is None:
            pos_mask = torch.ones((B, P), device=pos_emb.device, dtype=torch.bool)
        else:
            pos_mask = pos_mask.bool()

        # pos sims: (B,P)
        sim_pos = torch.matmul(anchor, pos_emb.transpose(1, 2)).squeeze(1)
        neg_inf = _neg_inf(sim_pos)
        pos_inf = torch.tensor(torch.finfo(sim_pos.dtype).max, device=sim_pos.device, dtype=sim_pos.dtype)

        pos_min = sim_pos.masked_fill(~pos_mask, pos_inf).min(dim=1).values
        pos_max = pos_min # Note(Alex): This is fix to use the strongest counter example, not easiest
        # pos_max per anchor (masked): (B,)
        # pos_max = sim_pos.masked_fill(~pos_mask, neg_inf).max(dim=1).values
        has_pos = pos_mask.any(dim=1)

        # ---- global batch pool (batch negatives) ----
        global_pos = gather_with_grad(pos_emb)          # (B_global,P,D)
        flat_global = global_pos.view(-1, D)            # (N_batch,D)
        sim_batch = torch.matmul(anchor.squeeze(1), flat_global.T)  # (B,N_batch)

        global_pos_mask = gather_tensor(pos_mask).bool()  # (B_global,P)
        flat_global_mask = global_pos_mask.view(-1)       # (N_batch,)
        cand_mask_batch = flat_global_mask.unsqueeze(0).expand(B, -1)  # (B,N_batch)

        # exclude self-positives from the batch pool
        rank = self._get_rank()
        N_batch = flat_global.shape[0]
        global_ids = torch.arange(N_batch, device=pos_emb.device).unsqueeze(0).expand(B, -1)
        start_idx = (rank * B + torch.arange(B, device=pos_emb.device)).unsqueeze(1) * P
        own_mask = (global_ids >= start_idx) & (global_ids < start_idx + P)
        cand_mask_batch = cand_mask_batch & (~own_mask)

        # batch_max per anchor: (B,)
        batch_max = sim_batch.masked_fill(~cand_mask_batch, neg_inf).max(dim=1).values
        has_batch = cand_mask_batch.any(dim=1)

        # ---- hard negatives ----
        has_hard = anchor.new_zeros((B,), dtype=torch.bool)
        hard_max = anchor.new_full((B,), fill_value=neg_inf.item())
        sim_hard = None  # saved for per-slot margin argmax

        if negative is not None:
            if negative.dim() == 2:
                negative = negative.unsqueeze(1)  # (B,1,D)
            K = negative.shape[1]
            if K > 0:
                neg_emb = l2_normalize(negative, eps=eps)  # (B,K,D)
                sim_hard = torch.matmul(anchor, neg_emb.transpose(1, 2)).squeeze(1)  # (B,K)

                if neg_mask is None:
                    neg_mask = torch.ones((B, K), device=pos_emb.device, dtype=torch.bool)
                else:
                    neg_mask = neg_mask.bool()

                has_hard = neg_mask.any(dim=1)
                hard_max = sim_hard.masked_fill(~neg_mask, neg_inf).max(dim=1).values

        # ---- resolve effective per-anchor margin (m1_eff) ----
        # m1 can be: scalar, (B,1), or (B,K) for per-slot margins
        if isinstance(m1, torch.Tensor) and m1.dim() == 2 and m1.shape[1] > 1 and sim_hard is not None and neg_mask is not None:
            # Per-slot margin: pick margin of the hardest negative per anchor
            hard_idx = sim_hard.masked_fill(~neg_mask, neg_inf).argmax(dim=1)  # (B,)
            m1_eff = m1.gather(1, hard_idx.unsqueeze(1)).squeeze(1)  # (B,)
        elif isinstance(m1, torch.Tensor):
            m1_eff = m1.squeeze(-1) if m1.dim() == 2 else m1  # (B,) or scalar
        else:
            m1_eff = m1  # scalar float

        # ---- losses ----
        # Tier 1: pos > hard + m1  (or pos > batch + m1 if no hard)
        loss1 = anchor.new_tensor(0.0, requires_grad=True)
        loss2 = anchor.new_tensor(0.0, requires_grad=True)

        if has_hard.any():
            valid1 = has_pos & has_hard
            if valid1.any():
                l1_vec = self.relu(hard_max - pos_max + m1_eff)
                loss1 = (l1_vec[valid1]).mean()

            # Tier 2: hard > batch_max + m2
            valid2 = has_hard & has_batch
            if valid2.any():
                l2_vec = self.relu(batch_max - hard_max + self.m2)
                loss2 = (l2_vec[valid2]).mean()
        else:
            # Fallback: no explicit hard negatives -> just separate from batch
            valid1 = has_pos & has_batch
            if valid1.any():
                l1_vec = self.relu(batch_max - pos_max + m1_eff)
                loss1 = (l1_vec[valid1]).mean()

        total = loss1 + self.alpha * loss2

        # ---- lightweight, distributed-safe monitors ----
        with torch.no_grad():
            # rates measured on anchors where the constraint is applicable
            def _rate(ok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                num = (ok & mask).float().sum()
                den = mask.float().sum()
                return _dist_ratio(num, den, eps).detach()

            valid1_mask = (has_pos & (has_hard if has_hard.any() else has_batch))
            if has_hard.any():
                ok1 = pos_max >= (hard_max + m1_eff)
            else:
                ok1 = pos_max >= (batch_max + m1_eff)

            valid2_mask = has_hard & has_batch
            ok2 = hard_max >= (batch_max + self.m2)

            tier1_sat = _rate(ok1, valid1_mask)
            tier2_sat = _rate(ok2, valid2_mask)

            pos_minus_hard = _dist_ratio(
                ((pos_max - hard_max) * (has_pos & has_hard).float()).sum(),
                (has_pos & has_hard).float().sum(),
                eps,
            ).detach() if has_hard.any() else anchor.new_tensor(float("nan"))

            hard_minus_batch = _dist_ratio(
                ((hard_max - batch_max) * valid2_mask.float()).sum(),
                valid2_mask.float().sum(),
                eps,
            ).detach() if valid2_mask.any() else anchor.new_tensor(float("nan"))

        return LossOutput(
            loss=total,
            metrics={
                "margin_triplet/loss": total.detach(),
                "margin_triplet/tier1": loss1.detach(),
                "margin_triplet/tier2": loss2.detach(),
                "margin_triplet/tier1_sat_rate": tier1_sat,
                "margin_triplet/tier2_sat_rate": tier2_sat,
                "margin_triplet/pos_minus_hard_mean": pos_minus_hard,
                "margin_triplet/hard_minus_batch_mean": hard_minus_batch,
                "margin_triplet/m1": (m1_eff.float().mean().detach() if isinstance(m1_eff, torch.Tensor) else anchor.new_tensor(m1_eff).detach()),
                "margin_triplet/m2": anchor.new_tensor(self.m2).detach(),
                "margin_triplet/alpha": anchor.new_tensor(self.alpha).detach(),
            },
        )



# ==============================================================================
# SECTION H: Oracle Losses (Margin-as-Target)
# ==============================================================================
#
# DESIGN RATIONALE — DECOMPOSING CONTRASTIVE + CALIBRATION + RANKING
# -------------------------------------------------------------------
#
# Standard contrastive losses (InfoNCE, ArcFace, Circle) optimize a BINARY
# decision boundary: push negatives below some threshold, pull positives above.
# This is analogous to classification — the loss provides no gradient for
# *how* similar a negative should be, only that it should be *less* similar
# than the positive.
#
# Our evaluation metric (Pearson correlation with an oracle score) measures
# LINEAR AGREEMENT — whether cos_sim(anchor, edited) tracks the degree of
# editing.  A model that perfectly separates pos/neg but assigns random
# similarity *magnitudes* within the negative set will score well on SSR
# but poorly on Pearson.
#
# We therefore decompose the training objective into three orthogonal terms
# that can be independently ablated:
#
#   ┌──────────────────────────────────────────────────────────────────┐
#   │  ROLE          │  LOSS                │  WHAT IT OPTIMIZES       │
#   │────────────────│──────────────────────│──────────────────────────│
#   │  Alignment     │  clip_one / infonce  │  anchor ↔ positive       │
#   │                │  (any contrastive)   │  binary separation       │
#   │────────────────│──────────────────────│──────────────────────────│
#   │  Pos Calib.    │  oracle_pos          │  cos_sim(a, pos) → 1.0   │
#   │  (Optional)    │                      │  saturate clean pairs    │
#   │────────────────│──────────────────────│──────────────────────────│
#   │  Neg Calib.    │  oracle_neg          │  cos_sim(a, neg_k)       │
#   │                │                      │    → 1 − margin_k        │
#   │                │                      │  continuous calibration   │
#   │────────────────│──────────────────────│──────────────────────────│
#   │  Ordering      │  oracle_rank         │  sim(less_edit)          │
#   │                │                      │    > sim(more_edit)      │
#   │                │                      │  monotonicity            │
#   └──────────────────────────────────────────────────────────────────┘
#
# EXAMPLE CLI ABLATION TABLE:
#   Baseline:           --loss_config "clip_one:1.0"
#   + neg calibration:  --loss_config "clip_one:1.0+oracle_neg:0.3"
#   + ranking only:     --loss_config "clip_one:1.0+oracle_rank:0.2"
#   + both:             --loss_config "clip_one:1.0+oracle_neg:0.3+oracle_rank:0.2"
#   + pos saturation:   --loss_config "clip_one:1.0+oracle_neg:0.3+oracle_pos:0.1"
#   Full:               --loss_config "clip_one:1.0+oracle_neg:0.3+oracle_pos:0.1+oracle_rank:0.2"
#
# ORACLE TARGET MAPPING:
#   margin = ratio = part_mask_area / obj_mask_area  (for MTG)
#   margin = heuristic per generation method          (for EncodeID)
#
#   target_sim(b, k) = 1 − margin(b, k)
#     ratio = 0   → target = 1.0   (nothing edited  → identical)
#     ratio = 0.5 → target = 0.5   (half edited     → moderate similarity)
#     ratio = 1.0 → target = 0.0   (fully replaced  → unrelated)
#
# LITERATURE JUSTIFICATION:
#
#   Calibration (oracle_neg / oracle_pos):
#     Directly regressing cosine similarity to a continuous target is
#     analogous to knowledge distillation [Hinton et al., 2015] where one
#     regresses student scores to teacher scores.  In metric learning,
#     Soft Contrastive Learning [Thoma et al., CVPR 2023] generalises
#     contrastive labels to continuous similarities derived from metadata.
#     The SoftTriple Loss [Qian et al., ICCV 2019] similarly uses soft
#     class assignments.  Our oracle score is a geometric ground-truth
#     (mask area ratio), making it a physically-grounded distillation
#     target.
#
#   Ranking (oracle_rank):
#     Pairwise ranking via log-sigmoid is exactly the RankNet formulation
#     [Burges et al., ICML 2005], later extended to LambdaRank [Burges,
#     2010].  In metric learning, Ranked List Loss [Wang et al., CVPR
#     2019] and differentiable AP losses (SmoothAP [Brown et al., ECCV
#     2020], FastAP [Cakir et al., CVPR 2019]) directly optimise ranking
#     quality.  Our within-sample pairwise ranking is the simplest and
#     most defensible variant: no listwise approximation is needed because
#     K=3 slots produce at most 3 pairs, and the oracle ordering is exact
#     (not noisy labels).
#
#   Decomposition (independent ablation):
#     Multi-task loss decomposition with per-term weights is standard
#     practice [Kendall et al., CVPR 2018; Chen et al., 2018].  Keeping
#     alignment (contrastive) separate from calibration (regression) and
#     ordering (ranking) ensures each term has a clear, non-conflicting
#     gradient signal.  ArcFace itself separates the angular margin
#     (geometric prior) from the cross-entropy (alignment) — our approach
#     extends this to arbitrary per-sample oracle scores.
#
# ==============================================================================


# ---- Shared utility for oracle losses ----

def _oracle_pearson(x: torch.Tensor, y: torch.Tensor,
                    eps: float = 1e-7) -> torch.Tensor:
    """Differentiable Pearson correlation between 1-D tensors (detached for logging)."""
    if x.numel() < 2:
        return x.new_tensor(float("nan"))
    xm = x - x.mean()
    ym = y - y.mean()
    num = (xm * ym).sum()
    den = (xm.norm() * ym.norm()).clamp(min=eps)
    return (num / den).detach()


def _oracle_regression_core(
    sims: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "mse",
    beta: float = 0.1,
) -> torch.Tensor:
    """Shared regression computation used by both oracle_pos and oracle_neg."""
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(sims, targets, beta=beta)
    return F.mse_loss(sims, targets)


# --------------------------------------------------------------------------
# OraclePositiveAlignLoss  (anchor ↔ positive calibration)
# --------------------------------------------------------------------------

class OraclePositiveAlignLoss(StandardizedLoss):
    """Regress ``cos_sim(anchor, pos_k) → 1.0`` for every valid positive slot.

    **Why separate from the contrastive loss?**
    Contrastive objectives (InfoNCE, CLIP) optimise *relative* ranking: the
    positive logit must be higher than all negative logits.  They do NOT
    constrain the *absolute* magnitude of positive similarity — a model that
    assigns ``sim = 0.6`` to all positives while assigning ``sim = 0.2`` to
    all negatives satisfies InfoNCE perfectly.

    ``oracle_pos`` adds an explicit pressure for ``sim(pos) ≈ 1.0``, ensuring
    the upper end of the similarity range is calibrated.  This is the positive-
    side analogue of a temperature calibration objective.

    Parameters
    ----------
    loss_type : ``"mse"`` | ``"smooth_l1"``
    beta : SmoothL1 beta (only used when loss_type="smooth_l1").

    References
    ----------
    * Thoma et al., "Soft Contrastive Learning for Visual Localization" (Neurips 2020)
    """

    def __init__(self, loss_type: str = "mse", beta: float = 0.1, **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        **kwargs,                       # absorbs margin, etc.
    ) -> LossOutput:
        a = l2_normalize(anchor, eps=eps)
        if a.dim() == 3:
            a = a.squeeze(1)                # (B, D)

        if positive is None or positive.shape[-2] == 0:
            return LossOutput(
                loss=a.new_tensor(0.0, requires_grad=True),
                metrics={"oracle_pos/skipped": a.new_tensor(1.0)},
            )

        pos = l2_normalize(positive, eps=eps)   # (B, P, D)
        B, P, _ = pos.shape
        if pos_mask is None:
            pos_mask = torch.ones((B, P), device=a.device, dtype=torch.bool)
        pos_mask = pos_mask.bool()

        # cos_sim(anchor, pos_k)
        sim_pos = torch.matmul(a.unsqueeze(1), pos.transpose(1, 2)).squeeze(1)  # (B, P)
        target_pos = torch.ones_like(sim_pos)

        flat_mask = pos_mask.reshape(-1)
        if not flat_mask.any():
            return LossOutput(
                loss=a.new_tensor(0.0, requires_grad=True),
                metrics={"oracle_pos/n_pairs": a.new_tensor(0.0)},
            )

        s = sim_pos.reshape(-1)[flat_mask]
        t = target_pos.reshape(-1)[flat_mask]

        loss = _oracle_regression_core(s, t, self.loss_type, self.beta)

        with torch.no_grad():
            mae = (s - t).abs().mean()

        return LossOutput(
            loss=loss,
            metrics={
                "oracle_pos/loss": loss.detach(),
                "oracle_pos/mae": mae,
                "oracle_pos/mean_sim": s.mean(),
                "oracle_pos/n_pairs": a.new_tensor(float(s.numel())),
            },
        )


# --------------------------------------------------------------------------
# OracleNegCalibrationLoss  (anchor ↔ negative ⟶ oracle target)
# --------------------------------------------------------------------------

class OracleNegCalibrationLoss(StandardizedLoss):
    """Regress ``cos_sim(anchor, neg_k) → 1 − margin_k`` for valid negative slots.

    This is the core "margin-as-target" loss.  Unlike contrastive losses that
    use the margin to shift a decision boundary, this loss treats it as a
    **regression target** for the raw cosine similarity.  The training signal
    is *continuous*: a negative with ``ratio = 0.1`` should yield
    ``sim ≈ 0.9``, while ``ratio = 0.6`` should yield ``sim ≈ 0.4``.

    This directly optimises for Pearson correlation with the oracle because
    the predicted similarity is trained to be a linear function of the oracle.

    Parameters
    ----------
    loss_type : ``"mse"`` | ``"smooth_l1"``
    beta : SmoothL1 beta (only used when loss_type="smooth_l1").

    References
    ----------
    * Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
      — regressing student outputs to teacher outputs (continuous targets).
    * Thoma et al., "Soft Contrastive Learning" (CVPR 2023)
      — contrastive learning with continuous similarity labels.
    * Qian et al., "SoftTriple Loss" (ICCV 2019)
      — soft class assignments in metric learning.
    """

    def __init__(self, loss_type: str = "mse", beta: float = 0.1, **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.beta = beta

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        margin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LossOutput:
        if margin is None or negative is None or negative.shape[-2] == 0:
            return LossOutput(
                loss=anchor.new_tensor(0.0, requires_grad=True),
                metrics={"oracle_neg/skipped": anchor.new_tensor(1.0)},
            )

        margin = margin.to(anchor.device).float()   # (B, 3)
        B = anchor.shape[0]
        a = l2_normalize(anchor, eps=eps)
        if a.dim() == 3:
            a = a.squeeze(1)

        neg = l2_normalize(negative, eps=eps)        # (B, K, D)
        K = neg.shape[1]
        if neg_mask is None:
            neg_mask = torch.ones((B, K), device=a.device, dtype=torch.bool)
        neg_mask = neg_mask.bool()

        sim_neg = torch.matmul(a.unsqueeze(1), neg.transpose(1, 2)).squeeze(1)  # (B, K)
        target_neg = (1.0 - margin[:, :K]).clamp(0.0, 1.0)                     # (B, K)

        flat_mask = neg_mask.reshape(-1)
        if not flat_mask.any():
            return LossOutput(
                loss=anchor.new_tensor(0.0, requires_grad=True),
                metrics={"oracle_neg/n_pairs": anchor.new_tensor(0.0)},
            )

        s = sim_neg.reshape(-1)[flat_mask]
        t = target_neg.reshape(-1)[flat_mask]

        loss = _oracle_regression_core(s, t, self.loss_type, self.beta)

        with torch.no_grad():
            mae = (s - t).abs().mean()
            pearson = _oracle_pearson(s, t, eps)

        return LossOutput(
            loss=loss,
            metrics={
                "oracle_neg/loss": loss.detach(),
                "oracle_neg/mae": mae,
                "oracle_neg/pearson": pearson,
                "oracle_neg/n_pairs": anchor.new_tensor(float(s.numel())),
                "oracle_neg/mean_target": t.mean(),
                "oracle_neg/mean_sim": s.mean(),
            },
        )


# --------------------------------------------------------------------------
# OracleRegressionLoss  (backward-compatible: neg + optional pos)
# --------------------------------------------------------------------------

class OracleRegressionLoss(StandardizedLoss):
    """Combined oracle regression over negatives (and optionally positives).

    **Prefer using ``oracle_neg`` and ``oracle_pos`` separately** for cleaner
    ablation.  This class is kept for backward compatibility and dispatches
    to the atomic losses internally.

    Parameters
    ----------
    loss_type : ``"mse"`` | ``"smooth_l1"``
    beta : SmoothL1 beta.
    include_pos : bool
        If True, also regress ``cos_sim(anchor, pos_k) → 1.0``.
    """

    def __init__(self, loss_type: str = "mse", beta: float = 0.1,
                 include_pos: bool = False, **kwargs):
        super().__init__()
        self._neg = OracleNegCalibrationLoss(loss_type=loss_type, beta=beta)
        self._pos = OraclePositiveAlignLoss(loss_type=loss_type, beta=beta) if include_pos else None

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        margin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LossOutput:
        out_neg = self._neg(anchor, positive, negative, pos_mask, neg_mask,
                            eps=eps, margin=margin, **kwargs)

        if self._pos is not None:
            out_pos = self._pos(anchor, positive, negative, pos_mask, neg_mask,
                                eps=eps, **kwargs)
            total = out_neg.loss + out_pos.loss
            metrics = {**out_neg.metrics, **out_pos.metrics,
                       "oracle_reg/loss": total.detach()}
        else:
            total = out_neg.loss
            metrics = {**out_neg.metrics, "oracle_reg/loss": total.detach()}

        return LossOutput(loss=total, metrics=metrics)


# --------------------------------------------------------------------------
# OracleRankingLoss  (pairwise ordering among negatives)
# --------------------------------------------------------------------------

class OracleRankingLoss(StandardizedLoss):
    """Pairwise ranking loss: enforce similarity ordering matches the oracle.

    For every pair of valid negative slots ``(k1, k2)`` in the same sample
    where ``margin[k1] < margin[k2]`` (k1 was less edited), enforce:

        ``cos_sim(anchor, neg_k1) > cos_sim(anchor, neg_k2) + ranking_margin``

    This teaches **monotonicity**: less editing → higher similarity.

    **Theoretical defence**

    Contrastive losses provide *no gradient* for ordering *within* the
    negative set — all negatives are pushed away equally.  When the evaluation
    metric is Pearson correlation (linear agreement with an oracle), the model
    must learn a monotonic mapping from edit severity to similarity.  Pairwise
    ranking directly optimises this.

    The soft (log-sigmoid) variant is exactly the **RankNet** objective
    [Burges et al., ICML 2005]: ``−log σ(s_i − s_j)`` where ``s_i > s_j``
    is the desired ordering.  This is the foundational building block of
    Learning-to-Rank (LTR), later extended to LambdaRank [Burges, 2010]
    and LambdaMART.

    In metric learning, the same idea appears as:
    * **Ranked List Loss** [Wang et al., CVPR 2019]
    * **SmoothAP** [Brown et al., ECCV 2020] — differentiable Average
      Precision computed via pairwise sigmoid comparisons (exactly our inner
      loop).
    * **FastAP** [Cakir et al., CVPR 2019]

    With K = 3 slots per sample, the loop generates at most 3 pairs (very
    cheap), and the oracle ordering is exact (not noisy).

    Parameters
    ----------
    ranking_margin : float
        Minimum desired separation between similarities.
    soft : bool
        True → log-sigmoid (RankNet); False → hinge (margin-based).
    cross_sample : bool
        True → also form pairs across batch samples (mixes identities; use
        with care).  Default False = within-sample only.

    References
    ----------
    * Burges et al., "Learning to Rank using Gradient Descent" (ICML 2005)
    * Burges, "From RankNet to LambdaRank to LambdaMART" (MSR-TR, 2010)
    * Wang et al., "Ranked List Loss for Deep Metric Learning" (CVPR 2019)
    * Brown et al., "Smooth-AP" (ECCV 2020)
    * Cakir et al., "Deep Metric Learning to Rank" (CVPR 2019)
    """

    def __init__(self, ranking_margin: float = 0.05, soft: bool = True,
                 cross_sample: bool = False, **kwargs):
        super().__init__()
        self.ranking_margin = ranking_margin
        self.soft = soft
        self.cross_sample = cross_sample

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        margin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LossOutput:
        if margin is None or negative is None or negative.shape[-2] < 2:
            return LossOutput(
                loss=anchor.new_tensor(0.0, requires_grad=True),
                metrics={"oracle_rank/skipped": anchor.new_tensor(1.0)},
            )

        margin = margin.to(anchor.device).float()       # (B, 3)
        B, K, D = negative.shape
        a = l2_normalize(anchor, eps=eps)                # (B, D)
        if a.dim() == 3:
            a = a.squeeze(1)
        neg = l2_normalize(negative, eps=eps)            # (B, K, D)

        if neg_mask is None:
            neg_mask = torch.ones((B, K), device=a.device, dtype=torch.bool)
        neg_mask = neg_mask.bool()

        # cos_sim(anchor, neg_k) for each slot
        sim = torch.matmul(a.unsqueeze(1), neg.transpose(1, 2)).squeeze(1)  # (B, K)
        margin_neg = margin[:, :K]  # (B, K)

        if self.cross_sample:
            loss, n_pairs, n_correct = self._cross_sample_ranking(
                sim, margin_neg, neg_mask, eps
            )
        else:
            loss, n_pairs, n_correct = self._within_sample_ranking(
                sim, margin_neg, neg_mask, eps
            )

        with torch.no_grad():
            acc = n_correct / max(n_pairs, 1)

        return LossOutput(
            loss=loss,
            metrics={
                "oracle_rank/loss": loss.detach(),
                "oracle_rank/n_pairs": anchor.new_tensor(float(n_pairs)),
                "oracle_rank/accuracy": anchor.new_tensor(acc),
            },
        )

    def _within_sample_ranking(
        self,
        sim: torch.Tensor,       # (B, K)
        margin: torch.Tensor,    # (B, K)
        mask: torch.Tensor,      # (B, K) bool
        eps: float,
    ):
        """Form pairs (k1, k2) within each sample where margin[k1] ≠ margin[k2]."""
        B, K = sim.shape
        total_loss = sim.new_tensor(0.0, requires_grad=True)
        n_pairs = 0
        n_correct = 0

        pair_losses = []
        for k1 in range(K):
            for k2 in range(k1 + 1, K):
                valid = mask[:, k1] & mask[:, k2]  # (B,)
                if not valid.any():
                    continue

                m1 = margin[valid, k1]
                m2 = margin[valid, k2]
                s1 = sim[valid, k1]
                s2 = sim[valid, k2]

                diff_m = m1 - m2       # negative ⟹ k1 less edited
                diff_s = s1 - s2       # positive ⟹ k1 more similar (good if m1 < m2)

                not_tie = diff_m.abs() > eps
                if not not_tie.any():
                    continue

                diff_m = diff_m[not_tie]
                diff_s = diff_s[not_tie]

                target_sign = -diff_m.sign()           # +1 if m1 < m2
                ordered_diff = target_sign * diff_s    # > 0 ⟹ correct order

                if self.soft:
                    pair_loss = -F.logsigmoid(ordered_diff - self.ranking_margin)
                else:
                    pair_loss = F.relu(self.ranking_margin - ordered_diff)

                pair_losses.append(pair_loss)
                n_pairs += pair_loss.numel()
                n_correct += (ordered_diff > 0).sum().item()

        if pair_losses:
            total_loss = torch.cat(pair_losses).mean()

        return total_loss, n_pairs, n_correct

    def _cross_sample_ranking(
        self,
        sim: torch.Tensor,       # (B, K)
        margin: torch.Tensor,    # (B, K)
        mask: torch.Tensor,      # (B, K) bool
        eps: float,
    ):
        """Form pairs across all (sample, slot) combinations in the batch."""
        flat_sim = sim.reshape(-1)
        flat_margin = margin.reshape(-1)
        flat_mask = mask.reshape(-1)

        valid = flat_mask.nonzero(as_tuple=True)[0]
        if valid.numel() < 2:
            return sim.new_tensor(0.0, requires_grad=True), 0, 0

        v_sim = flat_sim[valid]
        v_margin = flat_margin[valid]

        diff_m = v_margin.unsqueeze(1) - v_margin.unsqueeze(0)
        diff_s = v_sim.unsqueeze(1) - v_sim.unsqueeze(0)

        upper = torch.triu(torch.ones_like(diff_m, dtype=torch.bool), diagonal=1)
        not_tie = diff_m.abs() > eps
        pair_mask = upper & not_tie

        if not pair_mask.any():
            return sim.new_tensor(0.0, requires_grad=True), 0, 0

        dm = diff_m[pair_mask]
        ds = diff_s[pair_mask]

        target_sign = -dm.sign()
        ordered_diff = target_sign * ds

        if self.soft:
            loss = -F.logsigmoid(ordered_diff - self.ranking_margin).mean()
        else:
            loss = F.relu(self.ranking_margin - ordered_diff).mean()

        n_pairs = ordered_diff.numel()
        n_correct = (ordered_diff > 0).sum().item()

        return loss, n_pairs, n_correct

def _pos_cohesion_loss(
    positive: torch.Tensor,
    pos_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
    mode: str = "prototype",  # {"prototype","pairwise"}
) -> torch.Tensor:
    """
    Enforce positives to be close to each other (within the same sample).

    positive: (B, P, D)
    pos_mask: (B, P) bool or None

    Returns:
      scalar loss (requires grad)

    Modes:
      - "prototype": minimize 1 - cos(p_i, mean(p)_norm) for valid positives
        (cheap, stable, strongly pulls positives together)
      - "pairwise": minimize mean_{i<j} (1 - cos(p_i, p_j)) over valid pairs
        (stronger but O(P^2), still fine for small P)
    """
    if positive is None or positive.dim() != 3 or positive.shape[1] <= 1:
        # P <= 1 => nothing to "cohere"
        return positive.new_tensor(0.0, requires_grad=True)

    B, P, D = positive.shape
    pos = l2_normalize(positive, eps=eps)  # (B,P,D)

    if pos_mask is None:
        mask = torch.ones((B, P), device=pos.device, dtype=torch.bool)
    else:
        mask = pos_mask.to(pos.device).bool()

    # if a row has <2 valid positives => cohesion loss 0 for that row
    cnt = mask.sum(dim=1)  # (B,)
    valid_row = cnt >= 2

    if not valid_row.any():
        return pos.new_tensor(0.0, requires_grad=True)

    if mode == "prototype":
        # mean of valid positives (masked), then normalize => prototype
        w = mask.float().unsqueeze(-1)  # (B,P,1)
        mean = (pos * w).sum(dim=1) / (w.sum(dim=1).clamp_min(1.0))  # (B,D)
        proto = F.normalize(mean, p=2, dim=-1, eps=eps)              # (B,D)

        # cos(p_i, proto) for valid positives
        sim = (pos * proto.unsqueeze(1)).sum(dim=-1)                 # (B,P)
        loss_mat = (1.0 - sim) * mask.float()                        # (B,P)

        denom = mask.float().sum().clamp_min(1.0)
        # only count rows with >=2 positives (others contribute zeros via valid_row masking)
        row_keep = valid_row.float().unsqueeze(1)
        loss = (loss_mat * row_keep).sum() / denom

        return loss

    if mode == "pairwise":
        # pairwise cosine among positives: (B,P,P)
        sim_pp = torch.matmul(pos, pos.transpose(1, 2))  # (B,P,P)

        # build valid pair mask (i<j and both valid)
        m = mask
        pair_mask = (m.unsqueeze(2) & m.unsqueeze(1))  # (B,P,P)
        tri = torch.triu(torch.ones((P, P), device=pos.device, dtype=torch.bool), diagonal=1)
        pair_mask = pair_mask & tri.unsqueeze(0)

        if not pair_mask.any():
            return pos.new_tensor(0.0, requires_grad=True)

        loss_pairs = (1.0 - sim_pp)[pair_mask]  # (num_pairs,)
        return loss_pairs.mean()

    raise ValueError(f"Unknown cohesion mode: {mode}")


class ExtendedInfoNCELossV3(ExtendedInfoNCELoss):
    """
    V3 = InfoNCE_ext + explicit positive cohesion term (positives pull together).

    Adds:
      L_total = L_infonce_ext + beta_pos * L_pos_cohesion

    Notes:
    - Cohesion is WITHIN-SAMPLE only (no distributed gather needed).
    - Works for multi-positive (P>=2). If P<2, term is 0.
    - Uses cosine geometry (assumes l2_normalize).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        alpha: float = 0.5,         # inherited: weight for tier2 hard-vs-batch
        beta_pos: float = 0.1,      # NEW: weight for cohesion
        cohesion_mode: str = "prototype",  # {"prototype","pairwise"}
        **kwargs,
    ):
        super().__init__(temperature=temperature, alpha=alpha, **kwargs)
        self.beta_pos = float(beta_pos)
        self.cohesion_mode = str(cohesion_mode)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        **kwargs,
    ) -> "LossOutput":
        out = super().forward(
            anchor=anchor,
            positive=positive,
            negative=negative,
            pos_mask=pos_mask,
            neg_mask=neg_mask,
            eps=eps,
            **kwargs,
        )

        l_pos = _pos_cohesion_loss(
            positive=positive,
            pos_mask=pos_mask,
            eps=eps,
            mode=self.cohesion_mode,
        )

        total = out.loss + self.beta_pos * l_pos

        metrics: Dict[str, torch.Tensor] = dict(out.metrics)
        metrics.update(
            {
                "infonce_ext_v3/loss": total.detach(),
                "infonce_ext_v3/pos_cohesion": l_pos.detach(),
                "infonce_ext_v3/beta_pos": anchor.new_tensor(self.beta_pos).detach(),
            }
        )
        # keep original infonce_ext/* metrics intact too

        return LossOutput(loss=total, metrics=metrics)


class MarginCosineTripletLossV3(MarginCosineTripletLoss):
    """
    V3 = MarginTriplet (tiered) + explicit positive cohesion term.

    Adds:
      L_total = L_margin_triplet + beta_pos * L_pos_cohesion

    Why this helps:
      MarginTriplet mostly enforces pos > (hard/batch) + margin, but does not
      explicitly reduce variance among multiple positives. This term makes the
      entire positive set compact (reducing cross-subject overlaps in projections).
    """

    def __init__(
        self,
        m_pos_hard: float = 0.1,
        m_hard_batch: float = 0.05,
        alpha: float = 0.5,          # inherited: weight for tier2
        beta_pos: float = 0.1,       # NEW: weight for cohesion
        cohesion_mode: str = "prototype",
        **kwargs,
    ):
        super().__init__(m_pos_hard=m_pos_hard, m_hard_batch=m_hard_batch, alpha=alpha)
        self.beta_pos = float(beta_pos)
        self.cohesion_mode = str(cohesion_mode)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        pos_mask: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
        margin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "LossOutput":
        out = super().forward(
            anchor=anchor,
            positive=positive,
            negative=negative,
            pos_mask=pos_mask,
            neg_mask=neg_mask,
            eps=eps,
            margin=margin,
            **kwargs,
        )

        l_pos = _pos_cohesion_loss(
            positive=positive,
            pos_mask=pos_mask,
            eps=eps,
            mode=self.cohesion_mode,
        )

        total = out.loss + self.beta_pos * l_pos

        metrics: Dict[str, torch.Tensor] = dict(out.metrics)
        metrics.update(
            {
                "margin_triplet_v3/loss": total.detach(),
                "margin_triplet_v3/pos_cohesion": l_pos.detach(),
                "margin_triplet_v3/beta_pos": anchor.new_tensor(self.beta_pos).detach(),
            }
        )
        # keep original margin_triplet/* metrics intact too

        return LossOutput(loss=total, metrics=metrics)


# Registry mapping short names to loss classes
LOSS_REGISTRY: Dict[str, type] = {
    # "irm": BaselineImageRewardLoss,
    # "irm_ext": ExtendedImageRewardLoss,
    "irm": BaselineEmbeddingRewardLoss, # general version of IRM that can be applied to any embedding-based loss
    "irm_ext": ExtendedEmbeddingRewardLoss, # more general version of IRM extension that can be applied to any embedding-based loss
    "infonce": BaselineInfoNCELoss,
    "infonce_ext": ExtendedInfoNCELoss,
    "infonce_ext_v3": ExtendedInfoNCELossV3,
    "clip_base": SymmetricContrastiveLoss,
    "clip_one": CLIPOneLoss,
    "clip": ExtendedSymmetricContrastiveLoss,
    "circle": BaselineCircleLoss,
    "circle_ext": ExtendedCircleLoss,
    # Note: Triplet losses are commented out for now, left for completeness, but not included in the registry.
    # This is because we want to operate in unified space, so try the Cosine versions of this loss
    # "triplet": BaselineTripletMarginLoss,
    # "triplet_ext": ExtendedTripletMarginLoss,
    "triplet": BaselineCosineTripletLoss,
    "triplet_ext": ExtendedCosineTripletLoss,
    "triplet_ext_v3": ExtendedCosineTripletLossV3,
    "siglip": BaselineSigLIPLoss,
    "siglip_ext": ExtendedSigLIPLoss,
    "arcface": BaselineAngularContrastiveLoss,
    "arcface_ext": ExtendedAngularContrastiveLoss,
    "arcface_ext_v2": ArcFaceExtendedV2,
    "arcface_ext_v3": ArcFaceExtendedV3,
    "margin_triplet": MarginCosineTripletLoss,
    "margin_triplet_v3": MarginCosineTripletLossV3,
    # Oracle losses — decomposed for clean ablation
    "oracle_pos":  OraclePositiveAlignLoss,      # cos_sim(a, pos) → 1.0
    "oracle_neg":  OracleNegCalibrationLoss,      # cos_sim(a, neg) → 1 − margin
    "oracle_rank": OracleRankingLoss,             # monotonicity: less edit → higher sim
    "oracle_reg":  OracleRegressionLoss,          # backward-compat: neg + optional pos
}
