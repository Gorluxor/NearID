"""NearID training script.

Usage:
    python -m training.train --help
    accelerate launch -m training.train --loss_config "infonce_ext:1.0" --epochs 11
"""
from __future__ import annotations
import os
import re
import ast
import logging
from datetime import datetime
from typing import Any, Dict, Tuple
from pathlib import Path

import torch
import simple_parsing
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import get_scheduler
from datasets import load_dataset

from .config import TrainConfig, NearIDConfig
from .models import NearIDModel
from .data.nearid_dataset import (
    NearIDDataset, collate_nearid, pack_for_losses_dist,
    NearIDDataConfig, load_indices_json,
)
from .data.mtg_dataset import MTGTrainDataset, MultiDataset
from . import losses as losses_dist
from .evaluator import IdentityEvaluator

logger = logging.getLogger(__name__)


def build_eval_loaders(
    *,
    pos_ds,
    processor,
    cfg,
    neg_ds_list_val,
    neg_names_val=None,
    neg_ds_list_test=None,
    neg_names_test=None,
) -> Dict[Tuple[str, str], torch.utils.data.DataLoader]:
    eval_loaders = {}

    split_specs = [
        ("val", cfg.data.val_indices_path, neg_ds_list_val, neg_names_val),
        ("test", cfg.data.test_indices_path, neg_ds_list_test or [], neg_names_test),
    ]

    for split, path, neg_ds_list, neg_names in split_specs:
        if not path:
            continue

        indices = load_indices_json(path)
        assert len(indices) > 0, f"{split}: empty indices list from {path}"

        if neg_names is None:
            neg_names = [f"neg[{i}]" for i in range(len(neg_ds_list))]

        assert len(neg_names) == len(neg_ds_list)

        for neg_name, neg_ds in zip(neg_names, neg_ds_list):
            e_ds = NearIDDataset(
                pos_ds,
                hf_negs=[neg_ds],
                neg_names=[neg_name],
                processor=processor,
                indices=indices,
                config=NearIDDataConfig(
                    mask_prob=0.0,
                    shuffle_anchor=False,
                    per_slot_neg_dataset=True,
                    margin_map=cfg.data.margin_map,
                    margin_field=cfg.data.margin_field,
                    flip_prob=0.0,
                    color_jitter_prob=0.0,
                    translate_prob=0.0,
                ),
            )

            eval_loaders[(split, neg_name)] = torch.utils.data.DataLoader(
                e_ds,
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                collate_fn=collate_nearid,
                pin_memory=False,
            )
            logger.info(f"Created eval loader for split={split} with neg='{neg_name}' ({len(e_ds)} samples)")

    return eval_loaders


def _normalize_neg_paths(xs):
    if xs is None:
        return []

    if isinstance(xs, (list, tuple)) and len(xs) == 1 and isinstance(xs[0], str):
        s0 = xs[0].strip()
        if (s0.startswith("[") and s0.endswith("]")) or (s0.startswith("(") and s0.endswith(")")):
            try:
                parsed = ast.literal_eval(s0)
                if isinstance(parsed, (list, tuple)):
                    xs = list(parsed)
            except Exception:
                pass

    if isinstance(xs, str):
        s = xs.strip()
        if s.lower() in {"", "none", "null"}:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    xs = list(parsed)
                else:
                    xs = [s]
            except Exception:
                xs = [s]
        else:
            xs = [p.strip() for p in s.split(",") if p.strip()]

    out = []
    for x in xs:
        if x is None:
            continue
        x = str(x).strip()
        if x.lower() in {"", "none", "null"}:
            continue
        out.append(x)
    return out


def _slug(s: str) -> str:
    s = s.strip().replace("/", "-")
    s = re.sub(r"[^a-zA-Z0-9._+-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def runner_name_from_cfg(cfg: TrainConfig) -> str:
    parts = ["NearID", cfg.backbone, "head_only"]
    loss_part = cfg.loss_config.replace(":", "").replace("+", "")
    parts.append(f"loss{loss_part}")
    return "-".join(parts)


def run_suffix_from_cfg(cfg: TrainConfig) -> str:
    nneg = len(cfg.data.neg_paths) if cfg.data.neg_paths is not None else 0
    nneg_val = len(cfg.data.neg_paths_val) if cfg.data.neg_paths_val else 0
    nneg_test = len(cfg.data.neg_paths_test) if cfg.data.neg_paths_test else 0
    lr_part = f"{cfg.lr:.2e}".replace("+", "")
    uniq = datetime.now().strftime("%y%m%d-%H%M%S")
    base_name = runner_name_from_cfg(cfg)
    return f"{base_name}-lr{lr_part}-Nneg{nneg}-V{nneg_val}-T{nneg_test}-{uniq}"


def save_checkpoint(accelerator, model, output_dir, step, cfg):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"checkpoint-{step}")

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            save_path, safe_serialization=True,
            state_dict=accelerator.get_state_dict(model)
        )
        proc = getattr(unwrapped_model, "processor", None)
        if proc is not None and hasattr(proc, "save_pretrained"):
            proc.save_pretrained(save_path)

        if cfg.push_to_hub and cfg.hub_model_id:
            unwrapped_model.push_to_hub(cfg.hub_model_id, commit_message=f"Step {step}")


def main(cfg: TrainConfig):
    # 1. Setup Accelerator & Logging
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="wandb"
    )

    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        config=cfg.to_dict(),
        init_kwargs={
            "wandb": {
                "entity": cfg.wandb.entity,
                "name": runner_name_from_cfg(cfg),
                "tags": cfg.wandb.tags,
                "mode": cfg.wandb.mode,
            }
        }
    )

    cfg.data.neg_paths = _normalize_neg_paths(cfg.data.neg_paths)
    base_out = cfg.output_dir
    cfg.output_dir = os.path.join(base_out, run_suffix_from_cfg(cfg))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if accelerator.is_local_main_process else logging.WARN,
    )

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        cfg.save_yaml(os.path.join(cfg.output_dir, "config.yaml"))

        try:
            import wandb as _wb
            if _wb.run is not None:
                wandb_id_path = os.path.join(cfg.output_dir, "wandb_run_id.txt")
                with open(wandb_id_path, "w") as f:
                    f.write(_wb.run.id)
                logger.info(f"W&B run ID: {_wb.run.id}")
        except Exception:
            pass

    logger.info(f"{cfg.data.neg_paths=}")
    logger.info(f"{cfg.data.neg_paths_val=}")
    logger.info(f"{cfg.data.neg_paths_test=}")

    # 2. Model Setup
    model_config = NearIDConfig(
        backbone=cfg.backbone,
        pretrained_backbone=cfg.pretrained_backbone,
        pooling=cfg.pooling,
        head_type=cfg.head_type,
        layer_idx=getattr(cfg, "layer_idx", -1),
        head_hidden_dim=cfg.head_hidden_dim,
        head_out_dim=cfg.head_out_dim,
        head_layers=cfg.head_layers,
        head_dropout=cfg.head_dropout,
        use_batchnorm=cfg.use_batchnorm,
        use_layernorm=cfg.use_layernorm,
    )
    logger.info(f"Model config: {model_config}")
    with accelerator.main_process_first():
        model = NearIDModel(model_config)

    # Head-only training: freeze backbone
    for p in model.encoder_wrapper.parameters():
        p.requires_grad = False

    if accelerator.is_main_process:
        try:
            from torchinfo import summary
            model_stats = summary(model, depth=4,
                col_names=["num_params", "params_percent", "trainable"], verbose=0)
            logger.info(f"Model Architecture Summary:\n{model_stats}")
        except ImportError:
            pass

        logger.info("--- Actively Training Parameters ---")
        for name, p in model.named_parameters():
            if p.requires_grad:
                logger.info(f"Trainable: {name} | Shape: {list(p.shape)}")

    # 3. Data Setup
    data_cfg = NearIDDataConfig(
        mask_prob=cfg.data.mask_prob,
        shuffle_anchor=cfg.data.shuffle_anchor,
        per_slot_neg_dataset=cfg.data.per_slot_neg_dataset,
        margin_map=cfg.data.margin_map,
        margin_field=cfg.data.margin_field,
        mask_prob_apn=cfg.data.mask_prob_apn,
        flip_prob=cfg.data.flip_prob,
        color_jitter_prob=cfg.data.color_jitter_prob,
        translate_prob=cfg.data.translate_prob,
        translate_fraction=cfg.data.translate_fraction,
        scale_range_min=cfg.data.scale_range_min,
        scale_range_max=cfg.data.scale_range_max,
        fail_hard=True,
    )

    has_encodeid_negs = bool(cfg.data.neg_paths and len(cfg.data.neg_paths) > 0)
    has_mtg = bool(cfg.data.mtg_train_path and cfg.data.mtg_train_path.strip())

    if not has_encodeid_negs and not has_mtg:
        raise ValueError(
            "At least one training source must be set: "
            "data.neg_paths (NearID negatives) and/or data.mtg_train_path (MTG)."
        )

    # Evaluation Setup
    eval_loaders = {}
    evaluator = IdentityEvaluator(cfg)
    pos_ds = load_dataset(cfg.data.train_path, split="train")

    neg_ds_list_val = cfg.data.neg_paths_val if cfg.data.neg_paths_val and len(cfg.data.neg_paths_val) > 0 else []
    neg_ds_list_test = cfg.data.neg_paths_test if cfg.data.neg_paths_test and len(cfg.data.neg_paths_test) > 0 else []

    eval_loaders = build_eval_loaders(
        pos_ds=pos_ds,
        processor=model.processor,
        cfg=cfg,
        neg_ds_list_val=neg_ds_list_val,
        neg_names_val=[a.split('/')[-1] for a in neg_ds_list_val],
        neg_ds_list_test=neg_ds_list_test,
        neg_names_test=[a.split('/')[-1] for a in neg_ds_list_test],
    )
    logger.info(f"Created {len(eval_loaders)} evaluation loaders.")

    # Training dataset
    train_dataset = None
    if has_encodeid_negs:
        neg_ds_list = cfg.data.neg_paths
        train_dataset = NearIDDataset(
            pos_ds,
            hf_negs=neg_ds_list,
            processor=model.processor,
            config=data_cfg,
            neg_names=[a.split('/')[-1] for a in neg_ds_list],
            indices=load_indices_json(cfg.data.train_indices_path) if cfg.data.train_indices_path else None
        )

    # Optional: MTG joint training
    if has_mtg:
        mtg_dataset = MTGTrainDataset(
            processor=model.processor,
            config=NearIDDataConfig(
                mask_prob=cfg.data.mask_prob,
                mask_prob_apn=cfg.data.mask_prob_apn,
                shuffle_anchor=cfg.data.shuffle_anchor,
                per_slot_neg_dataset=True,
                flip_prob=cfg.data.flip_prob,
                color_jitter_prob=cfg.data.color_jitter_prob,
                translate_prob=cfg.data.translate_prob,
                translate_fraction=cfg.data.translate_fraction,
                scale_range_min=cfg.data.scale_range_min,
                scale_range_max=cfg.data.scale_range_max,
            ),
            split=cfg.data.mtg_split,
            hf_path=cfg.data.mtg_train_path,
            mtg_margin=cfg.data.mtg_margin,
            mtg_min=cfg.data.mtg_min,
            mtg_factor=cfg.data.mtg_factor,
        )
        mtg_repeat = max(1, getattr(cfg.data, "mtg_repeat", 1))
        if train_dataset is not None:
            mtg_list = [mtg_dataset] * mtg_repeat
            train_dataset = MultiDataset([train_dataset] + mtg_list)
        else:
            train_dataset = MultiDataset([mtg_dataset] * mtg_repeat) if mtg_repeat > 1 else mtg_dataset

    assert train_dataset is not None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_nearid,
        pin_memory=False,
    )

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    num_training_steps = cfg.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=num_training_steps
    )

    # 5. Loss Setup
    loss_fn = losses_dist.CombinedLoss(cfg.loss_config)

    # Prepare
    model, optimizer, train_loader, lr_scheduler, loss_fn = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler, loss_fn
    )

    logger.info(f"Output dir: {cfg.output_dir}")
    logger.info(f"Starting training: {cfg.epochs} epochs | Loss: {cfg.loss_config}")

    # 6. Training Loop
    global_step = 0
    postfix = {}

    for epoch in range(cfg.epochs):
        try:
            model.train()
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)

            pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}")

            for batch in pbar:
                if batch is None:
                    continue

                with accelerator.accumulate(model):
                    pv = batch["pixel_values"]
                    B, S = pv.shape[:2]
                    flat_pv = pv.view(B * S, *pv.shape[2:])

                    inputs = {"pixel_values_anchor": flat_pv}
                    embeddings_flat = model(inputs, side="anchor")
                    embeddings = embeddings_flat.view(B, S, -1)

                    anchor, positive, negative, pos_mask, neg_mask = pack_for_losses_dist(
                        embeddings, batch["pos_mask"], batch["neg_mask"]
                    )

                    loss_out = loss_fn(
                        anchor=anchor, positive=positive, negative=negative,
                        pos_mask=pos_mask, neg_mask=neg_mask,
                        margin=batch.get("margin", None)
                    )
                    loss = loss_out.loss

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    if global_step % 10 == 0:
                        metrics = {"train/loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
                        if hasattr(loss_out, "metrics"):
                            for k, v in loss_out.metrics.items():
                                metrics[f"train/{k}"] = v.item() if torch.is_tensor(v) else v
                        accelerator.log(metrics, step=global_step)
                        postfix["loss"] = f"{loss.item():.4f}"
                        pbar.set_postfix(postfix)

                if global_step % cfg.save_steps == 0 and global_step > 0:
                    save_checkpoint(accelerator, model, cfg.output_dir, global_step, cfg)

                    if accelerator.is_main_process:
                        checkpoint_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        for (split, neg_name), loader in eval_loaders.items():
                            try:
                                summary, table_final = evaluator.run(
                                    checkpoint_path=checkpoint_path,
                                    dataloader=loader,
                                    step=global_step,
                                    split_name=split,
                                    neg_name=neg_name,
                                )

                                SSR_KEY = "latex/disc_m_sim_vs_crossii_bidir/SSRm"
                                PA_KEY = "latex/disc_m_sim_vs_crossii_bidir/PA"
                                ssr = table_final.get(SSR_KEY, None)
                                pa = table_final.get(PA_KEY, None)

                                log_dict = {}
                                if ssr is not None:
                                    log_dict[f"SSRm/{split}/{neg_name}/SSRm"] = float(ssr)
                                if pa is not None:
                                    log_dict[f"PA/{split}/{neg_name}/PA"] = float(pa)
                                accelerator.log(log_dict, step=global_step)

                                postfix["eval"] = f"{split}/{neg_name}"
                                postfix["SSRm"] = f"{ssr:.1f}" if ssr is not None else "NA"
                                postfix["PA"] = f"{pa:.1f}" if pa is not None else "NA"
                                pbar.set_postfix(postfix)
                            except Exception as e:
                                raise e

                        model.to(accelerator.device)
                        accelerator.free_memory()

                    accelerator.wait_for_everyone()

                if cfg.dry_run and accelerator.is_main_process:
                    break
            if cfg.dry_run and accelerator.is_main_process:
                break
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected. Saving checkpoint...")
            save_checkpoint(accelerator, model, cfg.output_dir, global_step, cfg)
            return

    accelerator.end_training()


if __name__ == "__main__":
    config = simple_parsing.parse(TrainConfig)
    main(config)
