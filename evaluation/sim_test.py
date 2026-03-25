"""
FULL WORKING EXAMPLE (REFactored): SigLIP2 intra-sample similarity + optional VLM judge,
with a *batched* DataLoader GPU forward pass.

Modes (NEW):
- positives : positives only (intra positives)
- full      : positives + cross (pos vs neg)   [no neg intra]
- fullneg   : positives + cross + neg intra

Key idea:
- DataLoader returns variable-length tensors per sample: (n_i, 3, H, W)
- Flatten into one big batch (sum n_i, 3, H, W)
- One model forward
- Slice embeddings back per sample and compute similarities

# ┌────────────────────────────────────────────────────────────────────────┐
# │ BIG NOTE on output CSV organization (2026-03-09)                      │
# │                                                                        │
# │ Output CSVs go to --output_folder (default: runs/evals/).              │
# │                                                                        │
# │ scripts/gen_min.py pools results from TWO roots:                       │
# │   - EncodeID5R (primary): trained checkpoints + VSM                    │
# │   - EncodeID5  (baseline): frozen baselines (CLIP, SigLIP2, DINOv2,   │
# │                             Qwen3-VL 4B/8B/30B)                        │
# │                                                                        │
# │ gen_min.py deduplicates by sim_model NAME — if a model appears in      │
# │ EncodeID5R (even for 1 source), ALL data for that model from EncodeID5 │
# │ is SILENTLY DROPPED. This caused the Qwen3-VL-30B pooling bug where   │
# │ only 2/7 sources were included (n=998 instead of 3500).               │
# │                                                                        │
# │ RULE: Baseline VLM outputs (Qwen3-VL-*) must go ONLY into EncodeID5,  │
# │       NEVER into EncodeID5R. Trained checkpoints go into EncodeID5R.   │
# │       Stray duplicates in the wrong root cause silent data loss.       │
# └────────────────────────────────────────────────────────────────────────┘

Tested structure for HuggingFace datasets where columns are:
- images1, images2, images3 : PIL.Image OR dict {"bytes": ...} / {"path": ...} OR HF Image feature
- negatives aligned columns: nimg1, nimg2, nimg3 (optionally absent in positives mode)
- id, category optional
"""

import io
import json
import os
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Union

import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
import wandb

# -----------------------------------------------------------------------------
# Optional Qwen3-VL availability
# -----------------------------------------------------------------------------

try:
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import Qwen3VLMoeForConditionalGeneration
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False

import json_repair

# -----------------------------------------------------------------------------
# Optional VSM (Mind-the-Glitch) availability
# -----------------------------------------------------------------------------

try:
    from configs.config import load_dataset_generation_config as _mtg_load_dataset_cfg  # type: ignore
    import mtg.models as _mtg_models  # type: ignore
    from mtg.utils.aux_models_manager import ModelManager as _MTGModelManager  # type: ignore
    from mtg.utils.correspondence import compute_vsm_metric as _compute_vsm_metric  # type: ignore
    VSM_AVAILABLE = True
except ImportError:
    VSM_AVAILABLE = False

# -----------------------------------------------------------------------------
# Optional EncodeID (trained model) availability
# -----------------------------------------------------------------------------

try:
    from training.models import EncodeIDModel, EncodeIDConfig  # type: ignore
    from transformers import AutoConfig
    try:
        AutoConfig.register("encode_id", EncodeIDConfig)
        AutoModel.register(EncodeIDConfig, EncodeIDModel)
    except Exception:
        pass  # Already registered
    ENCODEID_AVAILABLE = True
except ImportError:
    ENCODEID_AVAILABLE = False

# Also support loading NearIDModel from HuggingFace Hub
try:
    from nearid import NearIDModel, NearIDConfig  # type: ignore
    NEARID_AVAILABLE = True
except ImportError:
    NEARID_AVAILABLE = False

# -----------------------------------------------------------------------------
# Mode spec (minimal branching)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeSpec:
    mtg_eval: bool 
    load_pos: bool
    load_neg: bool
    do_cross: bool
    do_neg_intra: bool

_MODE_SPECS: Dict[str, ModeSpec] = {
    "positives": ModeSpec(load_pos=True,  load_neg=False, do_cross=False, do_neg_intra=False, mtg_eval=False),
    "full":      ModeSpec(load_pos=True,  load_neg=True,  do_cross=True,  do_neg_intra=False, mtg_eval=False),
    "fullneg":   ModeSpec(load_pos=True,  load_neg=True,  do_cross=True,  do_neg_intra=True, mtg_eval=False),
    "mtg":       ModeSpec(load_pos=True,  load_neg=True,  do_cross=True,  do_neg_intra=False, mtg_eval=True),
}

_MODE_ALIASES: Dict[str, str] = {
    "full_neg": "fullneg",
    "negatives": "fullneg",  # legacy: previously "negatives" meant "neg intra"; we now map to fullneg
}

# v1 = NaN→False (legacy); v2 = NaN-aware wins, both SSR (mean-based) and SSRm (AND-based)
_MTG_METRIC_VERSION = 2

def get_mode_spec(mode: str) -> Tuple[str, ModeSpec]:
    m = _MODE_ALIASES.get(mode, mode)
    if m not in _MODE_SPECS:
        raise ValueError(f"Unknown mode='{mode}'. Valid: {sorted(_MODE_SPECS.keys())} (aliases: {sorted(_MODE_ALIASES.keys())})")
    return m, _MODE_SPECS[m]


# -----------------------------------------------------------------------------
# Model type detection (auto-detect VLM vs embedding model)
# -----------------------------------------------------------------------------

_VLM_MODEL_PATTERNS = (
    "Qwen3-VL",
    "Qwen2-VL",
    "Qwen-VL",
    "LLaVA",
    "InternVL",
    "CogVLM",
)

def is_vlm_model(model_id: str) -> bool:
    """Detect if model_id refers to a VLM (vision-language model) vs an embedding model."""
    model_id_upper = model_id.upper()
    for pattern in _VLM_MODEL_PATTERNS:
        if pattern.upper() in model_id_upper:
            return True
    return False


def is_encodeid_checkpoint(path: str) -> bool:
    """Detect if path is a local EncodeID checkpoint (has config.json with model_type='encode_id')."""
    if not os.path.isdir(path):
        return False
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        return cfg.get("model_type") == "encode_id"
    except Exception:
        return False


def _read_wandb_id(model_path: str) -> str:
    """Try to read persisted W&B run ID from the run directory.

    Looks for 'wandb_run_id.txt' in the run folder (parent of checkpoint-*).
    Returns the W&B ID string, or "" if not found.
    """
    parts = model_path.rstrip("/").split("/")
    # Walk up from checkpoint-* to the CLIPID-* run dir
    for i, p in enumerate(parts):
        if p.startswith("checkpoint-"):
            run_dir = os.path.join(*parts[:i]) if i > 0 else "."
            id_file = os.path.join(run_dir, "wandb_run_id.txt")
            if os.path.isfile(id_file):
                try:
                    with open(id_file) as f:
                        return f.read().strip()
                except Exception:
                    pass
            break
    return ""


def shorten_model_tag(model_path: str, is_encodeid: bool = False) -> str:
    """Produce a short, filesystem-safe model identifier for output filenames.

    For EncodeID checkpoints like:
      runs/trains/runs/SigLIP2_MAPInfoNCEExt/CLIPID-...-260301-070712/checkpoint-3300
    returns: "MAPInfoNCEExt~3300"

    When multiple runs share the same experiment folder (e.g. different lr),
    appends the run timestamp to disambiguate:
      "MAPInfoNCEExt~3300_260301-070712"

    For HuggingFace model IDs like "google/siglip2-so400m-patch14-384":
    returns: "google~siglip2-so400m-patch14-384" (original behaviour)
    """
    if is_encodeid:
        parts = model_path.rstrip("/").split("/")
        # Find checkpoint-N part
        ckpt_step = ""
        for p in reversed(parts):
            if p.startswith("checkpoint-"):
                ckpt_step = p.replace("checkpoint-", "")
                break
        # Find the experiment folder (e.g. SigLIP2_MAPInfoNCEExt)
        # Convention: it's the parent of the CLIPID-* run folder
        exp_name = ""
        exp_folder_idx = -1
        for i, p in enumerate(parts):
            if p.startswith("SigLIP2_"):
                exp_name = p.replace("SigLIP2_", "")
                exp_folder_idx = i
                break
        if not exp_name:
            # Fallback: use last non-checkpoint directory component
            for p in reversed(parts):
                if not p.startswith("checkpoint-"):
                    exp_name = p[:60]  # cap length
                    break

        # Extract run timestamp from CLIPID-* folder for disambiguation.
        # Multiple runs under the same experiment folder (e.g. different lr)
        # would otherwise produce identical tags and overwrite each other's CSVs.
        run_ts = ""
        for p in parts:
            if p.startswith("CLIPID-"):
                # Timestamp is the last hyphen-separated segment(s): e.g. "260301-070712"
                segs = p.split("-")
                if len(segs) >= 2:
                    run_ts = f"{segs[-2]}-{segs[-1]}"
                break

        tag = f"{exp_name}~{ckpt_step}" if ckpt_step else exp_name

        # Check if the experiment folder has multiple CLIPID-* run subdirectories.
        # If so, append the run timestamp to prevent filename collisions.
        if exp_folder_idx >= 0:
            exp_folder = os.path.join(*parts[:exp_folder_idx + 1])
            if os.path.isdir(exp_folder):
                run_dirs = [d for d in os.listdir(exp_folder) if d.startswith("CLIPID-")]
                if len(run_dirs) > 1 and run_ts:
                    tag = f"{tag}_{run_ts}"

        # Log W&B ID if available (for traceability)
        wandb_id = _read_wandb_id(model_path)
        if wandb_id:
            print(f"  [model_tag] {tag} (W&B: {wandb_id})")

        return tag
    # Default: HF-style slash→tilde
    return model_path.replace("/", "~")


# -----------------------------------------------------------------------------
# W&B helpers (single parameter: "entity/project"; default disabled via --wandb flag)
# -----------------------------------------------------------------------------

def _parse_wandb_entity_project(s: str) -> Tuple[Optional[str], str]:
    s = (s or "").strip()
    if "/" in s:
        entity, project = s.split("/", 1)
        entity = entity.strip() or None
        project = project.strip() or "default"
        return entity, project
    return None, (s or "default")


def _wandb_init_if_enabled(args, extra_config: Dict[str, Any]):
    if not getattr(args, "wandb", False):
        return None
    if getattr(args, "wandb_mode", "online") == "disabled":
        return None

    try:
        import wandb
    except Exception as e:
        print(f"[wandb] Could not import wandb ({type(e).__name__}: {e}). Continuing without W&B.")
        return None

    entity, project = _parse_wandb_entity_project(getattr(args, "wandb_project", "NearID"))
    tags = [t.strip() for t in (getattr(args, "wandb_tags", "") or "").split(",") if t.strip()]

    ds_neg_folder = extra_config.get("ds_neg_folder", "")
    findx_tag = os.path.splitext(os.path.basename(getattr(args, "findx", "") or ""))[0] if getattr(args, "findx", None) else args.split
    run_name = f"{ds_neg_folder}|{findx_tag}|{args.mode}|{'masks' if args.masks else 'nomasks'}|{'vlm' if args.vlm else 'emb'}"
    #run_name = f"{ds_neg_folder}|{args.split}|{args.mode}|{'masks' if args.masks else 'nomasks'}|{'vlm' if args.vlm else 'emb'}"

    run = wandb.init(
        project=project,
        entity=entity,
        group=getattr(args, "wandb_group", None),
        name=run_name,
        tags=tags,
        mode=getattr(args, "wandb_mode", "online"),
        config={**vars(args), **extra_config},
    )
    return run

# -----------------------------------------------------------------------------
# Dataset combine helpers
# -----------------------------------------------------------------------------

def add_null_cols(ds, cols):
    n = len(ds)
    for c in cols:
        if c not in ds.column_names:
            ds = ds.add_column(c, [None] * n)
    return ds


def fast_combine_aligned(main_ds: DatasetDict, neg_ds: DatasetDict, cols=("nimg1", "nimg2", "nimg3")) -> DatasetDict:
    cols_to_add = neg_ds.select_columns(list(cols))
    out = {}

    for split in main_ds.keys():
        if split not in neg_ds:
            out[split] = add_null_cols(main_ds[split], cols)
        else:
            out[split] = concatenate_datasets([main_ds[split], cols_to_add[split]], axis=1)

    return DatasetDict(out)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _hf_image_to_pil(x: Any) -> Optional[Image.Image]:
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, dict):
        if x.get("bytes") is not None:
            return Image.open(io.BytesIO(x["bytes"]))
        if x.get("path") is not None:
            return Image.open(x["path"])
    if hasattr(x, "convert"):  # duck-typing for PIL-like
        return x
    print(f"Warning: Unknown image type: {type(x)}")
    return None


# -----------------------------------------------------------------------------
# SigLIP2 similarity calculator
# -----------------------------------------------------------------------------

class SigLIP2SimilarityCalculator:
    def __init__(
        self,
        model_id: str = "google/siglip2-so400m-patch14-384",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device == "cuda" else torch.float32

        print(f"Loading {model_id} on {self.device} ({self.dtype}) ...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        except Exception:
            from transformers import AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device).eval()
        self._has_get_image_features = hasattr(self.model, "get_image_features")

    @torch.no_grad()
    def encode_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.device == "cuda":
            with torch.amp.autocast(dtype=self.dtype, device_type="cuda"):
                emb = self._forward(pixel_values)
        else:
            emb = self._forward(pixel_values)

        emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        return emb

    def _forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._has_get_image_features:
            # CLIP / SigLIP path
            return self.model.get_image_features(pixel_values=pixel_values)
        # DINOv2 / generic ViT path: use CLS token (pooler_output)
        out = self.model(pixel_values=pixel_values)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        # fallback: CLS token from last_hidden_state
        return out.last_hidden_state[:, 0]


# -----------------------------------------------------------------------------
# EncodeID similarity calculator (trained checkpoint)
# -----------------------------------------------------------------------------

class EncodeIDSimilarityCalculator:
    """Drop-in calculator for EncodeID trained checkpoints.
    
    Duck-types with SigLIP2SimilarityCalculator: exposes .processor, .device,
    .dtype, and .encode_pixel_values(pixel_values) -> normalized embeddings.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        if not ENCODEID_AVAILABLE:
            raise ImportError(
                "EncodeID dependencies not available. "
                "Ensure src/ contains models_dist.py, models.py, config.py."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading EncodeID checkpoint from {checkpoint_path} on {self.device} ...")
        self.model = EncodeIDModel.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
        self.model.to(device=self.device)  # type: ignore
        self.model.eval()

        # Match precision to encoder weights
        self.dtype = next(self.model.encoder_wrapper.parameters()).dtype
        self.processor = self.model.processor

        head_out = self.model.config.head_out_dim
        backbone = self.model.config.backbone

        # Read persisted W&B run ID for traceability
        self.wandb_id = _read_wandb_id(checkpoint_path)
        wandb_str = f", W&B: {self.wandb_id}" if self.wandb_id else ""
        print(f"  backbone={backbone}, head_out_dim={head_out}, dtype={self.dtype}{wandb_str}")

    @torch.no_grad()
    def encode_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)
        device_type = self.device if isinstance(self.device, str) else str(self.device).split(":")[0]
        with torch.autocast(device_type=device_type, dtype=self.dtype):
            out = self.model({"pixel_values_anchor": pixel_values}, side="anchor")
        out = out / (out.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        return out


# -----------------------------------------------------------------------------
# VSM (Mind-the-Glitch) Calculator
# -----------------------------------------------------------------------------

class VSMCalculator:
    """Wrapper around Mind-the-Glitch (MTG) model for VSM metric computation."""

    def __init__(self, device: Optional[str] = None, semantic_threshold: float = 0.6):
        if not VSM_AVAILABLE:
            raise ImportError(
                "Mind-the-Glitch dependencies not available. "
                "Ensure thirdparty/mind-the-glitch is set up correctly."
            )

        from pathlib import Path
        from omegaconf import OmegaConf
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from torchvision import transforms

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.semantic_threshold = semantic_threshold

        # Download model files from HF Hub
        mtg_ckpt_pth = Path(hf_hub_download(repo_id="abdo-eldesokey/mind-the-glitch", filename="mtg_weights.safetensors"))
        mtg_config_pth = Path(hf_hub_download(repo_id="abdo-eldesokey/mind-the-glitch", filename="experiment_cfg.yaml"))
        training_cfg = OmegaConf.load(mtg_config_pth)
        dataset_cfg = _mtg_load_dataset_cfg("automated")

        self.dtype = getattr(torch, training_cfg.dtype, torch.float32)
        model_name = training_cfg.model.name

        # Load auxiliary models (only CleanDIFT needed for inference)
        print("Loading MTG auxiliary models (CleanDIFT)...")
        models_manager = _MTGModelManager(self.device, dataset_cfg, load_cleandift=True, load_groundingsam=False)
        cleandift_model = models_manager.cleandift_model

        self.transform = transforms.Compose([
            transforms.Resize(dataset_cfg.img_size),
            transforms.ToTensor(),
        ])

        # Load MTG model
        print(f"Loading MTG model '{model_name}' from {mtg_ckpt_pth}")
        self.model = getattr(_mtg_models, model_name)(cleandift_model, training_cfg, self.device, self.dtype)
        if mtg_ckpt_pth.suffix == ".safetensors":
            self.model.load_state_dict(load_file(mtg_ckpt_pth))
        else:
            self.model.load_state_dict(torch.load(mtg_ckpt_pth, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("MTG model loaded successfully.")

    @torch.no_grad()
    def compute_vsm_pair(
        self,
        img1: Image.Image,
        img2: Image.Image,
        mask1: Image.Image,
        mask2: Image.Image,
    ) -> Dict[str, float]:
        """Compute VSM metric for a single image pair.

        Returns dict with keys 'vsm_0.5', 'vsm_0.6', 'vsm_0.7'.
        """
        model_out = self.model.inference(
            img1, img2,
            img1_obj_mask_p=mask1,
            img2_obj_mask_p=mask2,
            transform=self.transform,
            return_correspondences=True,
        )

        vsm_dict, _ = _compute_vsm_metric(
            model_out["source_points_s"],
            model_out["target_points_s"],
            model_out["matching_score_maps_max_s"],
            model_out["source_points_v"],
            model_out["target_points_v"],
            model_out["matching_score_maps_max_v"],
            semantic_threshold=self.semantic_threshold,
            visualize_matches=False,
        )
        return vsm_dict


# -----------------------------------------------------------------------------
# Qwen3-VL Judge Calculator (VLM-based pairwise scoring)
# -----------------------------------------------------------------------------

class Qwen3VLJudgeCalculator:
    """
    Qwen3-VL judge: (imgA, imgB) + template prompt -> JSON with {"score": <number>, ...}
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        template_path: str = "vl_template_v2.txt", # Note(Alex): Important V2 as says to be concise
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 128,  # WARNING: 128 may truncate MTG JSON output (~262 tokens). Use >=512 for MTG mode.
    ):
        if not QWEN3_AVAILABLE:
            raise ImportError(
                "Qwen3VLForConditionalGeneration not available. "
                "Install transformers version that supports Qwen3-VL."
            )

        print(f"Loading {model_id} with device_map={device_map}, dtype={dtype} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        if "30B" in model_id:
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            ).eval()

        if "30B" in model_id:
            print(f"[NOTICE] 30B model detected — overriding max_new_tokens {max_new_tokens} → 512")
            self.max_new_tokens = 512
        else:
            self.max_new_tokens = max_new_tokens
        self.model_id = model_id

        with open(template_path, "r", encoding="utf-8") as f:
            self.template = f.read().strip()
        print(f"Loaded template from {template_path} ({len(self.template)} chars)")

        # self._json_re = re.compile(r"\{.*\}", re.DOTALL)
        self._score_re = re.compile(r'(?i)"score"\s*:\s*([-+]?\d+(?:\.\d+)?)')

    def _parse_single_output(self, out_text: str) -> Dict[str, Any]:
        """
        Robust parse:
        1) Try json_repair to get an object (works for many LLM-ish errors).
        2) If missing score, regex-extract score from the *repaired JSON string* (or raw).
        3) Always return stable keys; store cues as JSON-strings for CSV.
        """
        result = {
            "score": float("nan"),
            "confidence": "error",
            "match_cues": "[]",
            "conflict_cues": "[]",
            "background_used": False,
            "invalid_json": False,   # json_repair couldn't produce a dict
            "invalid_score": False,  # score still NaN after all fallbacks
        }

        text = (out_text or "").strip()
        if not text:
            return result

        obj = None
        repaired_str = ""

        # 1) Try json_repair -> object
        try:
            # return_objects=True is faster + avoids an extra json.loads
            obj = json_repair.repair_json(text, return_objects=True, ensure_ascii=False)
            # If super broken, library may return "" or None
            if not isinstance(obj, dict):
                obj = None
        except Exception:
            obj = None
        result["invalid_json"] = (obj is None)
        # Also produce a repaired string for regex fallback (cheap + helps when score got "fixed away")
        try:
            repaired_str = json_repair.repair_json(text, ensure_ascii=False)
        except Exception:
            repaired_str = ""

        # 2) Get score from object if present; else regex from repaired/raw text
        score_key = None
        if isinstance(obj, dict):
            for key in ("score", "Score", "SCORE", "similarity", "Similarity"):
                if key in obj:
                    score_key = key
                    break
            if score_key is not None:
                try:
                    result["score"] = float(obj[score_key])
                except Exception:
                    pass

        if not np.isfinite(result["score"]):
            blob = repaired_str or text
            m = self._score_re.search(blob)
            if m:
                try:
                    result["score"] = float(m.group(1))
                except Exception:
                    pass

        # 3) Fill remaining fields if we have a dict
        if isinstance(obj, dict):
            conf = obj.get("confidence", result["confidence"])
            result["confidence"] = str(conf)

            result["background_used"] = bool(obj.get("background_used", False))

            match = obj.get("match_cues", [])
            conflict = obj.get("conflict_cues", [])

            # normalize to list[str]
            def _as_list_str(x):
                if x is None:
                    return []
                if isinstance(x, list):
                    return [str(v) for v in x]
                if isinstance(x, str):
                    s = x.strip()
                    # sometimes model returns list-as-string
                    if s.startswith("[") and s.endswith("]"):
                        try:
                            v = json.loads(s)
                            if isinstance(v, list):
                                return [str(t) for t in v]
                        except Exception:
                            pass
                    return [x]
                return [str(x)]

            match_l = _as_list_str(match)
            conflict_l = _as_list_str(conflict)

            result["match_cues"] = json.dumps(match_l, ensure_ascii=False)
            result["conflict_cues"] = json.dumps(conflict_l, ensure_ascii=False)

        # Clip score and set invalid_score flag (applies to both dict-parsed and regex-fallback)
        if np.isfinite(result["score"]):
            result["score"] = float(np.clip(result["score"], 0.0, 10.0))
            result["invalid_score"] = False
        else:
            result["invalid_score"] = True

        return result


    @torch.no_grad()
    def judge_pairs_batch(
        self,
        pairs: List[Tuple[Image.Image, Image.Image]],
        return_raw: bool = False,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[str]]]:
        if not pairs:
            return ([], []) if return_raw else []

        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"

        try:
            all_messages = []
            for img_a, img_b in pairs:
                img_a = img_a.convert("RGB")
                img_b = img_b.convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_a},
                            {"type": "image", "image": img_b},
                            {"type": "text", "text": self.template},
                        ],
                    }
                ]
                all_messages.append(messages)

            inputs = self.processor.apply_chat_template(
                all_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            results = [self._parse_single_output(text) for text in output_texts]
            if return_raw:
                return results, output_texts
            return results
        finally:
            self.processor.tokenizer.padding_side = original_padding_side


# -----------------------------------------------------------------------------
# Mask utilities
# -----------------------------------------------------------------------------

def _clean_mask_pil(mask_pil: Image.Image) -> Image.Image:
    mask_gray = np.array(mask_pil.convert("L"))
    blurred = cv2.GaussianBlur(mask_gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [contours[0]], -1, 255, thickness=cv2.FILLED) # type: ignore
        mask = clean

    return Image.fromarray(mask, mode="L")


def binarize_mask(mask_pil: Image.Image) -> Image.Image:
    m = mask_pil.convert("L")
    arr = (np.array(m) > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def dilate_mask(mask_pil: Image.Image, k: int = 3) -> Image.Image:
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be an odd integer >= 1 (e.g., 3, 5, 7).")
    m = mask_pil.convert("L")
    m = binarize_mask(m)
    m = m.filter(ImageFilter.MaxFilter(k))
    return binarize_mask(m)


def apply_mask_to_image_pil(
    img_pil: Image.Image,
    mask_pil: Image.Image,
    keep: str = "foreground",
    fill: str = "black",
) -> Image.Image:
    img = img_pil.convert("RGB")
    m = mask_pil.convert("L")

    if m.size != img.size:
        m = m.resize(img.size, resample=Image.Resampling.NEAREST)

    m = binarize_mask(m)
    m01 = (np.array(m) > 0)
    arr = np.array(img).astype(np.uint8)

    if fill == "black":
        fill_rgb = np.array([0, 0, 0], dtype=np.uint8)
    elif fill == "white":
        fill_rgb = np.array([255, 255, 255], dtype=np.uint8)
    else:
        raise ValueError("fill must be 'black' or 'white'")

    if keep == "foreground":
        keep01 = m01
    elif keep == "background":
        keep01 = ~m01
    else:
        raise ValueError("keep must be 'foreground' or 'background'")

    out = arr.copy()
    out[~keep01] = fill_rgb
    return Image.fromarray(out, mode="RGB")


# -----------------------------------------------------------------------------
# Dataset wrapper for SynCD-like structure
# -----------------------------------------------------------------------------

class SynCDSimilarityDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        processor=None,
        mode: str = "full",
        use_masks: bool = False,
        mask_keep: str = "foreground",
        mask_fill: str = "black",
        mask_dilate: bool = False,
        mask_dilate_k: int = 3,
        max_samples: Optional[int] = None,
        mask_clean_fn=_clean_mask_pil,
        return_pil: bool = False,
    ):
        resolved_mode, spec = get_mode_spec(mode)
        self.mode = resolved_mode
        self.spec = spec

        self.hf_dataset = hf_dataset
        self.processor = processor
        self.use_masks = use_masks
        self.mask_keep = mask_keep
        self.mask_fill = mask_fill
        self.mask_dilate = mask_dilate
        self.mask_dilate_k = mask_dilate_k
        self.mask_clean_fn = mask_clean_fn
        self.return_pil = return_pil or (processor is None)

        self.max_samples = max_samples
        if self.max_samples is not None:
            self.hf_dataset = self.hf_dataset.select(range(min(self.max_samples, len(self.hf_dataset))))
            print(f"Limiting to {self.max_samples} samples")

    def __len__(self):
        return len(self.hf_dataset)

    def _load_image_set(self, row, prefix: str, apply_masks: bool = False, debug: bool = False) -> List[Image.Image]:
        images: List[Image.Image] = []
        for j in range(1, 4):
            key = f"{prefix}{j}"
            raw = row.get(key)
            if debug:
                print(f"  [DEBUG _load_image_set] {key}: type={type(raw)}, is_none={raw is None}")
            img = _hf_image_to_pil(raw)
            if img is None:
                continue
            img = img.convert("RGB")
            try:
                img.load()
            except Exception:
                continue

            if apply_masks and self.use_masks:
                mk = _hf_image_to_pil(row.get(f"masks{j}"))
                if mk is None:
                    continue
                mk = mk.convert("L")
                try:
                    mk.load()
                except Exception:
                    continue

                mk = self.mask_clean_fn(mk)
                mk = binarize_mask(mk)
                if self.mask_dilate:
                    mk = dilate_mask(mk, self.mask_dilate_k)
                img = apply_mask_to_image_pil(img, mk, keep=self.mask_keep, fill=self.mask_fill)

            images.append(img)
        return images

    def _process_images(self, images: List[Image.Image]) -> torch.Tensor:
        processed = self.processor(images=images, return_tensors="pt")
        return processed.pixel_values

    def __getitem__(self, idx: int):
        try:
            row = self.hf_dataset[idx]
            spec = self.spec

            pos_images: List[Image.Image] = []
            neg_images: List[Image.Image] = []

            if spec.load_pos:
                pos_images = self._load_image_set(row, "images", apply_masks=self.use_masks)
                # we require at least 2 positives for intra (and to keep positives consistent)
                if len(pos_images) < 2:
                    return None

            if spec.load_neg:
                neg_images = self._load_image_set(row, "nimg", apply_masks=self.use_masks)
                # for cross in full/fullneg, require at least 1 negative
                if spec.do_cross and len(neg_images) < 1:
                    return None

            all_images = pos_images + neg_images

            out: Dict[str, Any] = {
                "sample_id": row.get("id", idx),
                "category": row.get("category", ""),
                "n_pos": len(pos_images),
                "n_neg": len(neg_images),
                "n_images": len(all_images),
            }

            if self.return_pil:
                out["images"] = all_images
            else:
                out["pixel_values"] = self._process_images(all_images)

            return out

        except Exception as e:
            print(f"[DEBUG __getitem__] Exception at idx {idx}: {type(e).__name__}: {e}")
            return None


def collate_similarity_batch(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    original_len = len(batch)
    batch = [b for b in batch if b is not None]
    if not batch:
        print(f"[DEBUG collate] Empty batch after filtering! Original had {original_len} items")
        return None

    result: Dict[str, Any] = {
        "n_images": [b["n_images"] for b in batch],
        "n_pos": [b["n_pos"] for b in batch],
        "n_neg": [b["n_neg"] for b in batch],
        "sample_ids": [b["sample_id"] for b in batch],
        "categories": [b["category"] for b in batch],
    }

    if "pixel_values" in batch[0]:
        result["pixel_values"] = [b["pixel_values"] for b in batch]
    if "images" in batch[0]:
        result["images"] = [b["images"] for b in batch]

    return result


# -----------------------------------------------------------------------------
# Similarity computation helpers
# -----------------------------------------------------------------------------

def _get_dataset_mode(dataset) -> str:
    """Extract mode from a Dataset or Subset wrapper."""
    if hasattr(dataset, "mode"):
        return dataset.mode
    # Handle torch.utils.data.Subset
    if hasattr(dataset, "dataset"):
        return _get_dataset_mode(dataset.dataset)
    raise AttributeError(f"Cannot find 'mode' attribute in dataset of type {type(dataset)}")


def _compute_intra_similarities(emb: torch.Tensor, n_imgs: int, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if n_imgs >= 2:
        out[f"{prefix}_01"] = float((emb[0] * emb[1]).sum())
    if n_imgs >= 3:
        out[f"{prefix}_02"] = float((emb[0] * emb[2]).sum())
        out[f"{prefix}_12"] = float((emb[1] * emb[2]).sum())
    return out


def _compute_cross_similarities(pos_emb: torch.Tensor, neg_emb: torch.Tensor, n_pos: int, n_neg: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i in range(n_pos):
        for j in range(n_neg):
            out[f"cross_{i}{j}"] = float((pos_emb[i] * neg_emb[j]).sum())
    return out


def _calculate_oracle(obj_mask_pil, part_mask_pil):
    if obj_mask_pil is None or part_mask_pil is None:
        return 1.0, 0.0
    obj_arr = np.array(obj_mask_pil.convert("L")) > 0
    part_arr = np.array(part_mask_pil.convert("L")) > 0
    sum_obj, sum_part = obj_arr.sum(), part_arr.sum()
    ratio = float(sum_part / sum_obj) if sum_obj > 0 else 1.0

    return ratio, 1.0 - ratio

@torch.no_grad()
def compute_mtg_similarities(
    hf_dataset,
    calculator,
    is_vsm=False,
    is_vlm=False,
    indices=None,
    use_masks=False,
    mask_keep="foreground",
    run=None,
    output_dir: Optional[str] = None,
    debug_samples: int = 50,
) -> List[Dict[str, Any]]:
    
    from datetime import datetime

    sample_indices = indices if indices is not None else range(len(hf_dataset))
    results = []
    vlm_nan_samples, vlm_total_nan_scores = 0, 0
    vlm_key_nan_counts: Dict[str, int] = {k: 0 for k in ["1_2", "1_1p", "1_2p", "2_2p", "2_1p"]}

    samples_saved = 0
    debug_dir = None
    if is_vlm and output_dir is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = os.path.join(output_dir, f"debug_mtg_{timestamp}")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Saving MTG VLM debug JSONs for first {debug_samples} samples to: {debug_dir}")

    pbar = tqdm(sample_indices, desc="MTG Benchmark")
    for i, idx in enumerate(pbar):
        row = hf_dataset[idx]
        
        # 1. Load raw masks (MTG masks are already clean 'L' mode)
        m1_raw = _hf_image_to_pil(row['image_1_object_mask']).convert("L")
        m2_raw = _hf_image_to_pil(row['image_2_object_mask']).convert("L")
        
        # 2. Oracle scores
        r1, o1 = _calculate_oracle(m1_raw, _hf_image_to_pil(row['image_1_part_mask']))
        r2, o2 = _calculate_oracle(m2_raw, _hf_image_to_pil(row['image_2_part_mask']))

        # 3. Load Images
        i1 = _hf_image_to_pil(row['image_1_original']).convert("RGB")
        i2 = _hf_image_to_pil(row['image_2_original']).convert("RGB")
        i1p = _hf_image_to_pil(row['image_1_inpainted']).convert("RGB")
        i2p = _hf_image_to_pil(row['image_2_inpainted']).convert("RGB")
        
        # 4. Apply Masking Logic (Simplified)
        if use_masks:
            i1 = apply_mask_to_image_pil(i1, m1_raw, keep=mask_keep, fill="black")
            i1p = apply_mask_to_image_pil(i1p, m1_raw, keep=mask_keep, fill="black")
            
            i2 = apply_mask_to_image_pil(i2, m2_raw, keep=mask_keep, fill="black")
            i2p = apply_mask_to_image_pil(i2p, m2_raw, keep=mask_keep, fill="black")
            
            # DEBUG
            # write into a temp folder to inspect
            # temp_dir = "temp_debug_masks"
            # os.makedirs(temp_dir, exist_ok=True)
            # i1.save(os.path.join(temp_dir, f"{idx}_i1.png"))
            # i1p.save(os.path.join(temp_dir, f"{idx}_i1p.png"))
            # i2.save(os.path.join(temp_dir, f"{idx}_i2.png"))
            # i2p.save(os.path.join(temp_dir, f"{idx}_i2p.png"))
            # import pdb; pdb.set_trace()

        # 5. Calculate Metrics
        sims = {}
        if is_vsm:
            # VSM needs the mask references regardless of whether the images are pre-blacked out
            sims["mtg_sim_1_2"] = calculator.compute_vsm_pair(i1, i2, m1_raw, m2_raw).get("vsm_0.6", np.nan)
            sims["mtg_sim_1_1p"] = calculator.compute_vsm_pair(i1, i1p, m1_raw, m1_raw).get("vsm_0.6", np.nan)
            sims["mtg_sim_1_2p"] = calculator.compute_vsm_pair(i1, i2p, m1_raw, m2_raw).get("vsm_0.6", np.nan)
            sims["mtg_sim_2_2p"] = calculator.compute_vsm_pair(i2, i2p, m2_raw, m2_raw).get("vsm_0.6", np.nan)
            sims["mtg_sim_2_1p"] = calculator.compute_vsm_pair(i2, i1p, m2_raw, m1_raw).get("vsm_0.6", np.nan)
            
        elif is_vlm:
            _pair_keys = ["1_2", "1_1p", "1_2p", "2_2p", "2_1p"]
            pairs = [(i1, i2), (i1, i1p), (i1, i2p), (i2, i2p), (i2, i1p)]
            need_debug = debug_dir is not None and samples_saved < debug_samples
            if need_debug:
                res, raw_outputs = calculator.judge_pairs_batch(pairs, return_raw=True)
            else:
                res = calculator.judge_pairs_batch(pairs)
                raw_outputs = None
            for _k, _r in zip(_pair_keys, res):
                sims[f"mtg_sim_{_k}"] = _r["score"]
                sims[f"mtg_sim_{_k}_confidence"] = _r.get("confidence", "error")
                sims[f"mtg_sim_{_k}_invalid_json"] = _r.get("invalid_json", False)
                sims[f"mtg_sim_{_k}_invalid_score"] = _r.get("invalid_score", False)
            # NaN tracking for progress bar
            _nan_in = sum(1 for _k in _pair_keys if np.isnan(sims[f"mtg_sim_{_k}"]))
            vlm_total_nan_scores += _nan_in
            for _k in _pair_keys:
                if np.isnan(sims[f"mtg_sim_{_k}"]):
                    vlm_key_nan_counts[_k] += 1
            if _nan_in > 0:
                vlm_nan_samples += 1
                pbar.set_postfix_str(
                    f"VLM NaN: {vlm_nan_samples} samples / {vlm_total_nan_scores} pairs"
                )
            # Early NaN guard: warn loudly after first sample or after 10 samples
            n_so_far = i + 1
            if _nan_in > 0 and n_so_far == 1:
                print(
                    f"\n[ALERT] First MTG sample produced {_nan_in}/5 NaN scores! "
                    f"This usually means max_new_tokens is too low (need >=512 for VLM JSON). "
                    f"Per-key NaN: {vlm_key_nan_counts}"
                )
            if n_so_far == 10 and vlm_nan_samples > 0:
                nan_rate = vlm_nan_samples / n_so_far
                print(
                    f"\n[ALERT] After 10 samples: {vlm_nan_samples}/10 ({nan_rate:.0%}) have NaN. "
                    f"Results will be UNRELIABLE. Consider aborting and re-running with "
                    f"--max_new_tokens 512. Per-key: {vlm_key_nan_counts}"
                )
            if need_debug:
                debug_data = {
                    "sample_id": idx,
                    "pair_keys": _pair_keys,
                    "raw_outputs": raw_outputs,
                    "parsed_results": [dict(r) for r in res],
                }
                debug_path = os.path.join(debug_dir, f"sample_{samples_saved:04d}_{idx}.json")
                with open(debug_path, "w") as f:
                    json.dump(debug_data, f, indent=2, default=str)
                samples_saved += 1
            
        else: # Standard Embeddings
            pv = calculator.processor(images=[i1, i2, i1p, i2p], return_tensors="pt").pixel_values.to(calculator.device, calculator.dtype)
            emb = calculator.encode_pixel_values(pv)
            sims["mtg_sim_1_2"] = float((emb[0] * emb[1]).sum())
            sims["mtg_sim_1_1p"] = float((emb[0] * emb[2]).sum())
            sims["mtg_sim_1_2p"] = float((emb[0] * emb[3]).sum())
            sims["mtg_sim_2_2p"] = float((emb[1] * emb[3]).sum())
            sims["mtg_sim_2_1p"] = float((emb[1] * emb[2]).sum())
            
        m1 = sims["mtg_sim_1_2"] - sims["mtg_sim_1_1p"]
        m2 = sims["mtg_sim_1_2"] - sims["mtg_sim_2_2p"]
        # NaN-aware wins: pd.NA excluded from SSR/PA; NaN > 0 = False is wrong
        _win1 = pd.NA if np.isnan(m1) else bool(m1 > 0)
        _win2 = pd.NA if np.isnan(m2) else bool(m2 > 0)
        # SSRm: sample wins if ALL *valid* terms won (partial rows included, aligned with MCN)
        _valid_wins = [w for w in (_win1, _win2) if not pd.isna(w)]
        _overall = pd.NA if len(_valid_wins) == 0 else bool(all(_valid_wins))

        results.append({
            "sample_id": idx,
            "mtg_ratio_1": r1, "mtg_oracle_1": o1,
            "mtg_ratio_2": r2, "mtg_oracle_2": o2,
            **sims,
            "mtg_margin_1": m1, "mtg_margin_2": m2,
            "mtg_win_1": _win1, "mtg_win_2": _win2,
            "mtg_overall_win": _overall,
        })
        
        if run is not None and (i + 1) % 10 == 0:
            try:
                import wandb
                wandb.log({"progress/samples": i + 1})
            except Exception:
                pass
            
    # VLM NaN summary warning
    if is_vlm and vlm_nan_samples > 0:
        n_processed = len(results)
        print(
            f"\n[WARNING MTG VLM] {vlm_nan_samples}/{n_processed} samples had ≥1 NaN score "
            f"({vlm_total_nan_scores}/{5 * n_processed} pair-level NaNs). "
            f"Score coverage: {(1 - vlm_total_nan_scores / max(5 * n_processed, 1)) * 100:.1f}%"
        )

    df = pd.DataFrame(results)

    # --- Validity breakdown (per-sample margin coverage) ---
    _w1v = pd.to_numeric(df["mtg_win_1"], errors="coerce").notna()
    _w2v = pd.to_numeric(df["mtg_win_2"], errors="coerce").notna()
    n_total        = len(df)
    n_both_valid   = int((_w1v & _w2v).sum())
    n_partial      = int((_w1v ^ _w2v).sum())   # exactly 1 valid margin
    n_both_invalid = n_total - n_both_valid - n_partial
    print(f"MTG support: {n_total} total | {n_both_valid} both-valid | "
          f"{n_partial} partial | {n_both_invalid} both-invalid")
    if n_partial > 0:
        print(f"[INFO MTG] {n_partial} samples have only 1 valid margin — "
              f"included in SSR/SSRm (using valid terms only, aligned with MCN)")

    # --- Pearson Correlations with Oracle ---
    def _safe_corr(col_sim, col_oracle):
        r = df[col_sim].corr(df[col_oracle])
        return float(r) if pd.notna(r) else 0.0

    r_1_1p = _safe_corr("mtg_sim_1_1p", "mtg_oracle_1")
    r_2_2p = _safe_corr("mtg_sim_2_2p", "mtg_oracle_2")
    r_1_2p = _safe_corr("mtg_sim_1_2p", "mtg_oracle_2")
    r_2_1p = _safe_corr("mtg_sim_2_1p", "mtg_oracle_1")

    # Effective N per correlation (pairwise complete observations)
    n_11p = int(df[["mtg_sim_1_1p", "mtg_oracle_1"]].dropna().shape[0])
    n_22p = int(df[["mtg_sim_2_2p", "mtg_oracle_2"]].dropna().shape[0])
    n_12p = int(df[["mtg_sim_1_2p", "mtg_oracle_2"]].dropna().shape[0])
    n_21p = int(df[["mtg_sim_2_1p", "mtg_oracle_1"]].dropna().shape[0])

    # Fisher's z-transform
    rs = np.array([r_1_1p, r_2_2p, r_1_2p, r_2_1p])
    rs_clipped = np.clip(rs, -0.999, 0.999)
    mean_r = float(np.tanh(np.mean(np.arctanh(rs_clipped))))

    # Fisher's z-transform of strict r_1_1p and r_2_2p for secondary metric
    rs_strict = np.array([r_1_1p, r_2_2p])
    rs_strict_clipped = np.clip(rs_strict, -0.999, 0.999)
    r_strict_mean = float(np.tanh(np.mean(np.arctanh(rs_strict_clipped))))

    # --- NaN-aware SSR / SSRm / PA / Hm (metric_version=2) ---
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
    # Hm: harmonic mean of SSRm and PA (strict counterpart to harmonic H of SSR/PA)
    _hm_s, _hm_p = ssrm / 100.0, pa / 100.0
    _hm_d = _hm_s + _hm_p
    hm = (2.0 * _hm_s * _hm_p / (_hm_d + 1e-12)) * 100.0 if (np.isfinite(_hm_d) and _hm_d > 0) else float("nan")

    # Support counts (denominators for each metric)
    _ssr_n  = int(np.sum(~np.isnan(_ssr_wins)))
    _ssrm_n = int(pd.to_numeric(df["mtg_overall_win"], errors="coerce").notna().sum())
    _pa_trials = int(_total_trials)
    if _ssr_n != _ssrm_n:
        print(f"[WARNING MTG] SSR_n ({_ssr_n}) != SSRm_n ({_ssrm_n}) — unexpected (both should count rows with ≥1 valid term)")

    print(f"\n--- MTG Final Results ---")
    print(f"Mean Correlation (Fisher z): {mean_r:.4f}")
    print(f"Strict Mean Correlation (1-1p & 2-2p): {r_strict_mean:.4f}")
    print(f"  ├─ corr(1, 1p): {r_1_1p:.4f}  [N={n_11p}]")
    print(f"  ├─ corr(2, 2p): {r_2_2p:.4f}  [N={n_22p}]")
    print(f"  ├─ corr(1, 2p): {r_1_2p:.4f}  [N={n_12p}]")
    print(f"  └─ corr(2, 1p): {r_2_1p:.4f}  [N={n_21p}]")
    _fmt = lambda v: f"{v:.2f}%" if np.isfinite(v) else "NaN"
    print(f"Secondary -> SSR: {_fmt(ssr)} | SSRm: {_fmt(ssrm)} | PA: {_fmt(pa)} | Hm: {_fmt(hm)}")
    print(f"Support   -> SSR_n={_ssr_n} | SSRm_n={_ssrm_n} | PA_trials={_pa_trials} | "
          f"partial={n_partial} | both-invalid={n_both_invalid}")

    if run is not None:
        import wandb
        _wb: Dict[str, Any] = {
            "MTG/Corr_Mean": mean_r,
            "MTG/Corr_Strict_Mean": r_strict_mean,
            "MTG/Corr_1_1p": r_1_1p,
            "MTG/Corr_2_2p": r_2_2p,
            "MTG/Corr_1_2p": r_1_2p,
            "MTG/Corr_2_1p": r_2_1p,
            "MTG/Corr_N_11p": n_11p,
            "MTG/Corr_N_22p": n_22p,
            "MTG/Corr_N_12p": n_12p,
            "MTG/Corr_N_21p": n_21p,
            "MTG/SSR": ssr,    # mean-based, skipna=True (v2, aligned with MCN)
            "MTG/SSRm": ssrm,  # AND-based (v2)
            "MTG/PA": pa,
            "MTG/Hm": hm,
            "MTG/SSR_n": _ssr_n,
            "MTG/SSRm_n": _ssrm_n,
            "MTG/PA_trials": _pa_trials,
            "MTG/N_total": n_total,
            "MTG/N_both_valid": n_both_valid,
            "MTG/N_partial": n_partial,
            "MTG/metric_version": _MTG_METRIC_VERSION,
        }
        if is_vlm:
            _wb["MTG/vlm_nan_samples"] = vlm_nan_samples
            _wb["MTG/vlm_total_nan_scores"] = vlm_total_nan_scores
            _total_pairs = n_total * len(vlm_key_nan_counts)
            _wb["MTG/vlm_coverage_pct"] = (
                100.0 * (1 - vlm_total_nan_scores / _total_pairs) if _total_pairs > 0 else float("nan")
            )
            for _k, _cnt in vlm_key_nan_counts.items():
                _wb[f"MTG/vlm_nan_pct_{_k}"] = 100.0 * _cnt / n_total if n_total > 0 else float("nan")
        wandb.summary.update(_wb)

    return results

# -----------------------------------------------------------------------------
# Batched dataloader similarity computation (embedding path)
# -----------------------------------------------------------------------------

@torch.no_grad()
def compute_similarities_with_dataloader_batched(
    dataset: SynCDSimilarityDataset,
    calculator: SigLIP2SimilarityCalculator,
    batch_size: int = 4,
    num_workers: int = 4,
    run=None,
    stream_table=None,
) -> List[Dict[str, Any]]:
    mode = _get_dataset_mode(dataset)
    _, spec = get_mode_spec(mode)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_similarity_batch,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        # persistent_workers=(num_workers > 0),
        persistent_workers=False # Had issue where termination from wandb kept stuff in VRAM
    )

    device = calculator.device
    dtype = calculator.dtype

    results: List[Dict[str, Any]] = []
    samples_seen = 0

    # stream logging: keep bounded memory
    table_cols = ["sample_id", "category", "metric", "value"]
    table_rows: List[List[Any]] = []

    def _append_rows(out: Dict[str, Any]) -> None:
        if run is None:
            return
        sid = out["sample_id"]
        cat = out["category"]
        for k, v in out.items():
            if k.startswith(("sim_", "neg_sim_", "cross_")):
                table_rows.append([sid, cat, k, float(v)])
    try:
        for batch in tqdm(dataloader, desc=f"Computing similarities [{mode}] (batched GPU)"):
            if batch is None:
                print("Warning: empty batch encountered, skipping.")
                continue

            pv_list: List[torch.Tensor] = batch["pixel_values"]
            n_list: List[int] = batch["n_images"]
            n_pos_list: List[int] = batch["n_pos"]
            n_neg_list: List[int] = batch["n_neg"]
            ids: List[Any] = batch["sample_ids"]
            cats: List[Any] = batch["categories"]

            offsets: List[Tuple[int, int]] = []
            cursor = 0
            for n in n_list:
                offsets.append((cursor, cursor + n))
                cursor += n

            pixel_values = torch.cat(pv_list, dim=0).to(device, dtype=dtype, non_blocking=True)
            emb = calculator.encode_pixel_values(pixel_values)

            for (s, e), n_pos, n_neg, sample_id, category in zip(offsets, n_pos_list, n_neg_list, ids, cats):
                sample_emb = emb[s:e]
                pos_emb = sample_emb[:n_pos] if n_pos > 0 else sample_emb[:0]
                neg_emb = sample_emb[n_pos:n_pos + n_neg] if n_neg > 0 else sample_emb[:0]

                out: Dict[str, Any] = {
                    "sample_id": sample_id,
                    "category": category,
                    "n_pos": int(n_pos),
                    "n_neg": int(n_neg),
                }

                # positives intra
                if spec.load_pos and n_pos >= 2:
                    out.update(_compute_intra_similarities(pos_emb, n_pos, prefix="sim"))

                # cross
                if spec.do_cross and n_pos >= 1 and n_neg >= 1:
                    out.update(_compute_cross_similarities(pos_emb, neg_emb, n_pos, n_neg))

                # negatives intra (only in fullneg)
                if spec.do_neg_intra and n_neg >= 2:
                    out.update(_compute_intra_similarities(neg_emb, n_neg, prefix="neg_sim"))

                results.append(out)
                _append_rows(out)

                samples_seen += 1

            # (optional) also log progress once per batch
            if run is not None:
                try:
                    import wandb
                    wandb.log({"progress/samples": samples_seen})
                except Exception:
                    pass
    except KeyboardInterrupt:
        print(f'Recieved SIGINT stopping')

    return results


# -----------------------------------------------------------------------------
# VLM Judge-based similarity computation (batched per sample)
# -----------------------------------------------------------------------------

def _add_result_to_output(out: Dict[str, Any], prefix: str, result: Dict[str, Any]) -> None:
    out[prefix] = result["score"]
    out[f"{prefix}_confidence"] = result["confidence"]
    out[f"{prefix}_match_cues"] = result["match_cues"]
    out[f"{prefix}_conflict_cues"] = result["conflict_cues"]
    out[f"{prefix}_bg_used"] = result["background_used"]


@torch.no_grad()
def compute_similarities_with_vlm_judge(
    dataset: SynCDSimilarityDataset,
    calculator: Qwen3VLJudgeCalculator,
    batch_size: int = 16,
    num_workers: int = 4,
    output_dir: Optional[str] = None,
    debug_samples: int = 50,
    run=None,                 # NEW: optional wandb run
    stream_every: int = 256,  # NEW: log stream table every N samples
) -> List[Dict[str, Any]]:
    """
    Adds optional W&B streaming logging:
      - logs a streaming table: results/vlm_stream (metric-level rows)
      - logs progress: progress/samples
    """
    from datetime import datetime

    mode = _get_dataset_mode(dataset)
    _, spec = get_mode_spec(mode)

    debug_dir = None
    if output_dir is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = os.path.join(output_dir, f"debug_{timestamp}")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Saving debug JSONs for first {debug_samples} samples to: {debug_dir}")

    samples_saved = 0
    samples_seen = 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_similarity_batch,
        pin_memory=False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False # (num_workers > 0),
    )

    results: List[Dict[str, Any]] = []

    # NEW: counters
    invalid_json = 0      # alias for in_r (samples with any invalid_json)
    invalid_scores = 0    # alias for in_s (samples with any invalid_score)
    in_r = 0              # samples where >=1 pair had invalid_json
    in_s = 0              # samples where >=1 pair had invalid_score
    Tin_r = 0             # total pairs with invalid_json
    Tin_s = 0             # total pairs with invalid_score

    pbar = tqdm(dataloader, desc=f"Computing VLM judge scores [{mode}] (batched)")
    try:
        for batch in pbar:
            if batch is None:
                print("Warning: empty batch encountered, skipping.")
                continue

            img_lists: List[List[Image.Image]] = batch["images"]
            n_pos_list: List[int] = batch["n_pos"]
            n_neg_list: List[int] = batch["n_neg"]
            ids = batch["sample_ids"]
            cats = batch["categories"]

            for imgs, n_pos, n_neg, sample_id, category in zip(img_lists, n_pos_list, n_neg_list, ids, cats):
                pos_imgs = imgs[:n_pos]
                neg_imgs = imgs[n_pos:n_pos + n_neg]

                out: Dict[str, Any] = {
                    "sample_id": sample_id,
                    "category": category,
                    "n_pos": int(n_pos),
                    "n_neg": int(n_neg),
                }

                pairs: List[Tuple[Image.Image, Image.Image]] = []
                pair_keys: List[str] = []

                # positives intra
                if spec.load_pos and n_pos >= 2:
                    pairs.append((pos_imgs[0], pos_imgs[1])); pair_keys.append("sim_01")
                if spec.load_pos and n_pos >= 3:
                    pairs.append((pos_imgs[0], pos_imgs[2])); pair_keys.append("sim_02")
                    pairs.append((pos_imgs[1], pos_imgs[2])); pair_keys.append("sim_12")

                # cross
                if spec.do_cross and n_pos >= 1 and n_neg >= 1:
                    for i in range(n_pos):
                        for j in range(n_neg):
                            pairs.append((pos_imgs[i], neg_imgs[j]))
                            pair_keys.append(f"cross_{i}{j}")

                # negatives intra (only in fullneg)
                if spec.do_neg_intra and n_neg >= 2:
                    pairs.append((neg_imgs[0], neg_imgs[1])); pair_keys.append("neg_sim_01")
                if spec.do_neg_intra and n_neg >= 3:
                    pairs.append((neg_imgs[0], neg_imgs[2])); pair_keys.append("neg_sim_02")
                    pairs.append((neg_imgs[1], neg_imgs[2])); pair_keys.append("neg_sim_12")

                if pairs:
                    need_debug = debug_dir is not None and samples_saved < debug_samples
                    if need_debug:
                        batch_results, raw_outputs = calculator.judge_pairs_batch(pairs, return_raw=True)
                    else:
                        batch_results = calculator.judge_pairs_batch(pairs)
                        raw_outputs = None

                    for key, r in zip(pair_keys, batch_results):
                        _add_result_to_output(out, key, r)

                    # NEW: update counters (pair-level + sample-level)
                    pair_bad_json = sum(1 for r in batch_results if r.get("invalid_json", False))
                    pair_bad_score = sum(1 for r in batch_results if r.get("invalid_score", False))
                    Tin_r += pair_bad_json
                    Tin_s += pair_bad_score

                    sample_bad_json = (pair_bad_json > 0)
                    sample_bad_score = (pair_bad_score > 0)
                    if sample_bad_json:
                        in_r += 1
                        invalid_json += 1
                    if sample_bad_score:
                        in_s += 1
                        invalid_scores += 1

                    if need_debug:
                        debug_data = {
                            "sample_id": sample_id,
                            "category": category,
                            "mode": mode,
                            "n_pos": int(n_pos),
                            "n_neg": int(n_neg),
                            "pair_keys": pair_keys,
                            "raw_outputs": raw_outputs,
                            "parsed_results": batch_results,
                        }
                        debug_path = os.path.join(debug_dir, f"sample_{samples_saved:04d}_{sample_id}.json")
                        with open(debug_path, "w") as f:
                            json.dump(debug_data, f, indent=2, default=str)
                        samples_saved += 1

                    results.append(out)
                    samples_seen += 1

            # NEW: show sample-level / total-level as "two numbers"
            pbar.set_postfix_str(
                f"json(in_r/Tin_r)={in_r}/{Tin_r}  score(in_s/Tin_s)={in_s}/{Tin_s}"
            )
            
    except KeyboardInterrupt:
        print(f'Recieved SIGINT stopping')
    # --- FINAL LOGGING (W&B) ---
    if run is not None:
        try:
            import wandb
            run.summary.update({
                "vlm/in_r": int(in_r),
                "vlm/in_s": int(in_s),
                "vlm/Tin_r": int(Tin_r),
                "vlm/Tin_s": int(Tin_s),
                "vlm/invalid_json": int(invalid_json),
                "vlm/invalid_scores": int(invalid_scores),
            })
            wandb.log({
                "vlm/final_in_r": int(in_r),
                "vlm/final_in_s": int(in_s),
                "vlm/final_Tin_r": int(Tin_r),
                "vlm/final_Tin_s": int(Tin_s),
            })
        except Exception as e:
            print(f"Error while updating summary: {e}")
    return results



# -----------------------------------------------------------------------------
# VSM similarity computation (pairwise, iterates HF dataset directly)
# -----------------------------------------------------------------------------

def _make_white_mask(img: Image.Image) -> Image.Image:
    """Create full-image white mask (fallback when no mask is available)."""
    return Image.new("L", img.size, 255)


@torch.no_grad()
def compute_similarities_with_vsm(
    hf_dataset,
    calculator,  # VSMCalculator
    mode: str = "positives",
    max_samples: Optional[int] = None,
    indices: Optional[List[int]] = None,
    run=None,
) -> List[Dict[str, Any]]:
    """Compute VSM (Visual Similarity Metric) for all image pairs in the dataset.

    VSM processes one pair at a time (compute-heavy), so we iterate directly
    over the HF dataset rows instead of using a batched DataLoader.
    """
    _, spec = get_mode_spec(mode)

    n = len(hf_dataset)
    if max_samples is not None:
        n = min(n, max_samples)

    if indices is not None:
        sample_indices = indices
    else:
        sample_indices = list(range(n))

    results: List[Dict[str, Any]] = []
    _VSM_PAIRS = [(0, 1), (0, 2), (1, 2)]

    for idx in tqdm(sample_indices, desc=f"Computing VSM [{mode}]"):
        row = hf_dataset[idx]
        sample_id = row.get("id", idx)
        category = row.get("category", "")

        # Load positive images and masks
        pos_images: List[Image.Image] = []
        pos_masks: List[Image.Image] = []
        for j in range(1, 4):
            img = _hf_image_to_pil(row.get(f"images{j}"))
            if img is None:
                continue
            img = img.convert("RGB")
            mask = _hf_image_to_pil(row.get(f"masks{j}"))
            mask = mask.convert("L") if mask is not None else _make_white_mask(img)
            pos_images.append(img)
            pos_masks.append(mask)

        if len(pos_images) < 2:
            continue

        # Load negative images and masks
        neg_images: List[Image.Image] = []
        neg_masks: List[Image.Image] = []
        if spec.load_neg:
            for j in range(1, 4):
                img = _hf_image_to_pil(row.get(f"nimg{j}"))
                if img is None:
                    continue
                img = img.convert("RGB")
                mask = _hf_image_to_pil(row.get(f"nmasks{j}"))
                mask = mask.convert("L") if mask is not None else _make_white_mask(img)
                neg_images.append(img)
                neg_masks.append(mask)

        n_pos = len(pos_images)
        n_neg = len(neg_images)

        out: Dict[str, Any] = {
            "sample_id": sample_id,
            "category": category,
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

        # Positive intra-pairs
        if spec.load_pos and n_pos >= 2:
            for i, j in _VSM_PAIRS:
                if j >= n_pos:
                    continue
                try:
                    vsm = calculator.compute_vsm_pair(pos_images[i], pos_images[j], pos_masks[i], pos_masks[j])
                    out[f"sim_{i}{j}"] = vsm.get("vsm_0.6", float("nan"))
                    for k, v in vsm.items():
                        out[f"sim_{i}{j}_{k}"] = v
                except Exception as e:
                    print(f"[VSM] Error computing sim_{i}{j} for sample {sample_id}: {e}")

        # Cross pairs (pos vs neg)
        if spec.do_cross and n_pos >= 1 and n_neg >= 1:
            for i in range(n_pos):
                for j in range(n_neg):
                    try:
                        vsm = calculator.compute_vsm_pair(pos_images[i], neg_images[j], pos_masks[i], neg_masks[j])
                        out[f"cross_{i}{j}"] = vsm.get("vsm_0.6", float("nan"))
                        for k, v in vsm.items():
                            out[f"cross_{i}{j}_{k}"] = v
                    except Exception as e:
                        print(f"[VSM] Error computing cross_{i}{j} for sample {sample_id}: {e}")

        # Negative intra-pairs
        if spec.do_neg_intra and n_neg >= 2:
            for i, j in _VSM_PAIRS:
                if j >= n_neg:
                    continue
                try:
                    vsm = calculator.compute_vsm_pair(neg_images[i], neg_images[j], neg_masks[i], neg_masks[j])
                    out[f"neg_sim_{i}{j}"] = vsm.get("vsm_0.6", float("nan"))
                    for k, v in vsm.items():
                        out[f"neg_sim_{i}{j}_{k}"] = v
                except Exception as e:
                    print(f"[VSM] Error computing neg_sim_{i}{j} for sample {sample_id}: {e}")

        results.append(out)

        if run is not None:
            try:
                import wandb
                wandb.log({"progress/samples": len(results)})
            except Exception:
                pass

    return results


# -----------------------------------------------------------------------------
# Dataset loader helper (auto local vs HF)
# -----------------------------------------------------------------------------

def load_dataset_auto(path: str, split: str | None = None):
    if os.path.exists(path):
        ds = load_from_disk(path)
        # DatasetDict vs Dataset can differ; handle both
        if hasattr(ds, "column_names"):
            print(f"Loading local dataset from: {path} with columns: {ds.column_names}")
        else:
            if isinstance(ds, DatasetDict):
                print(f"Loading local dataset dict from: {path} with splits: {list(ds.keys())}")
            else: # fallback generic print
                print(f"Loading local dataset from: {path} with type: {type(ds)}")
        if split:
            return ds[split]
        return ds

    print(f"Loading HuggingFace dataset: {path}")
    if split:
        return load_dataset(path, split=split)
    return load_dataset(path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    s = v.lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")

# Note command below to see which processes are running VRAM if by change change to persistant=True and kill the process
# fuser -k -9 /dev/nvidia-uvm
# nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | sort -k2 -nr

# Example run
# DS_NEG=./data/ordered/SynCD-Flux-1024 CUDA_DEV=0 OUT_DIR=./runs/ordered_sims ./sim_test.sh 


def _load_indices_json(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Support either: [1,2,3] or {"indices":[1,2,3]}
    if isinstance(obj, dict) and "indices" in obj:
        obj = obj["indices"]

    if not isinstance(obj, list):
        raise ValueError(f"--findx must be a JSON list of ints or {{'indices': [...]}}. Got: {type(obj)}")

    idxs: List[int] = []
    for x in obj:
        try:
            idxs.append(int(x))
        except Exception:
            raise ValueError(f"Non-integer index in --findx: {x!r}")

    # Alert if duplicates found, but still return unique sorted list
    if len(idxs) != len(set(idxs)):
        raise ValueError(f"Duplicate indices found in --findx: {idxs}")

    return idxs



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute image similarities on SynCD dataset")
    parser.add_argument("--mode", type=str, default="fullneg", choices=["positives", "full", "fullneg", "full_neg", "negatives", "mtg"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_folder", type=str, default="outputs/sims/")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="google/siglip2-so400m-patch14-384",
        help="Model checkpoint (e.g., google/siglip2-so400m-patch14-384, Qwen/Qwen3-VL-4B-Instruct)",
    )
    parser.add_argument(
        "--vlm",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Force VLM mode (True/False). If not specified, auto-detected from model name.",
    )
    parser.add_argument("--vl_template", type=str, default="vl_template_v2.txt") 
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max tokens for VLM generation. WARNING: MTG JSON output is ~262 tokens; "
                             "use >=512 for MTG mode. 30B auto-overrides to 512, but 4B/8B do not.")
    # parser.add_argument("--masks", action="store_true")
    parser.add_argument("--masks", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--mask_keep", type=str, default="foreground", choices=["foreground", "background"])
    parser.add_argument("--ds_neg", type=str, default="./data/EncodeID/EncodeID-Flux") # prior: default="Aleksandar/SynCDIntraNeg"
    parser.add_argument("--ds", type=str, default="Aleksandar/EncodeID") # prior: default="Aleksandar/SynCD"
    parser.add_argument("--split", type=str, default="train")
    

    # W&B: default disabled (only enabled when --wandb is passed)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "NearID"),
        help="Single string 'entity/project' (default: NearID)",
    )
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated tags")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=os.environ.get("WANDB_MODE", "online"),
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--findx",
        type=str,
        default=None,
        help="Optional JSON file with a list of dataset indices to run (e.g., [0, 5, 10]).",
    )

    args = parser.parse_args()

    resolved_mode, mode_spec = get_mode_spec(args.mode)
    args.mode = resolved_mode  # normalize for naming/logging

    # Auto-detect VLM mode from model name if not explicitly set
    _vlm = args.vlm
    args.vlm = is_vlm_model(args.model)
    print(f"[NOTICE] Auto-detected VLM mode: {args.vlm} (even though {_vlm} passed) (model: {args.model})")
        
    # if model is VSM, then we use that specific calculator
    _VSM_mode = "vsm" in args.model.lower()

    # Auto-detect EncodeID checkpoint (local dir with config.json model_type=encode_id)
    _encodeid_mode = is_encodeid_checkpoint(args.model)
    if _encodeid_mode:
        print(f"[NOTICE] Detected EncodeID checkpoint at: {args.model}")

    findx_tag = None
    if args.findx:
        # e.g., runs/splits/val_all.json -> val_all
        findx_tag = os.path.splitext(os.path.basename(args.findx))[0]
    split_tag = findx_tag or args.split

    # Output path
    ds_neg_folder = "MTG-Dataset" if args.mode == "mtg" else os.path.basename(args.ds_neg.rstrip("/"))
    _mask = "image" if not args.masks else args.mask_keep
    _max = "" if args.mode == "mtg" else ("all" if args.max_samples is None else str(args.max_samples))
    _method = "_vlm" if args.vlm else "_vsm" if _VSM_mode else "_encodeid" if _encodeid_mode else ""
    _model_tag = shorten_model_tag(args.model, is_encodeid=_encodeid_mode)
    _base = args.output.split(".csv")[0] if args.output else "sims"
    _base = _base + f"{_method}_{_mask}_{split_tag}{_max}_{args.mode}_{_model_tag}"
    output_file = os.path.join(args.output_folder, ds_neg_folder, f"{_base}.csv")
    print(f"Output will be: {output_file}")
    
    if args.mode != "mtg":
        print(f"Loading dataset from: {args.ds} (neg: {args.ds_neg})")

    # Read training W&B ID if available (for cross-referencing eval → training run)
    _train_wandb_id = _read_wandb_id(args.model) if _encodeid_mode else ""
    if _train_wandb_id:
        print(f"Training W&B ID: {_train_wandb_id}")

    # W&B init (optional)
    run = _wandb_init_if_enabled(
        args,
        extra_config={
            "ds_neg_folder": ds_neg_folder,
            "output_file": output_file,
            "mask_variant": _mask,
            "method": "vlm" if args.vlm else "embedding",
            "mtg_eval": args.mode == "mtg",
            "train_wandb_id": _train_wandb_id,
        },
    )

    if run is not None:
        import wandb
        wandb.define_metric("sample_idx")                        # the x-axis
        wandb.define_metric("sample/*", step_metric="sample_idx") # all per-sample metrics use it

    # 1) Load dataset
    print("Loading dataset...")
    if args.mode == "mtg":
        # BYPASS custom SynCD logic and load MTG directly
        hf_ds = load_dataset("abdo-eldesokey/mtg-dataset", split="test")
        print("Loaded MTG dataset directly.")
        # if args.max_samples is not None:
        #     hf_ds = hf_ds.select(range(min(args.max_samples, len(hf_ds))))
    elif mode_spec.load_neg:
        ds_main = load_dataset_auto(args.ds)
        ds_neg = load_dataset_auto(args.ds_neg)
        ds_combined = fast_combine_aligned(ds_main, ds_neg)
        hf_ds = ds_combined[args.split]
        print("Combined dataset columns:", hf_ds.column_names)
    else:
        hf_ds = load_dataset_auto(args.ds, args.split)

    print("Dataset size:", len(hf_ds))
    if run is not None:
        try:
            wandb.log({"data/n_samples": int(len(hf_ds))})
        except Exception:
            pass

    # 2) Initialize + compute
    if args.mode == "mtg":
        print(f"\nUsing MTG benchmark mode with {args.model}")
        
        # Instantiate correct calculator based on the model provided
        if args.vlm:
            print(f"  vl_template: {args.vl_template}")
            calc = Qwen3VLJudgeCalculator(
                model_id=args.model, template_path=args.vl_template,
                dtype="auto", device_map="auto", max_new_tokens=args.max_new_tokens
            )
        elif _VSM_mode:
            calc = VSMCalculator(semantic_threshold=0.6)
        elif _encodeid_mode:
            calc = EncodeIDSimilarityCalculator(checkpoint_path=args.model)
        else:
            calc = SigLIP2SimilarityCalculator(model_id=args.model, dtype=torch.float16)

        indices_to_run = None
        if args.findx is not None:
            indices_to_run = _load_indices_json(args.findx)
            print(f"Filtering to {len(indices_to_run)} specified indices from {args.findx}")

        results = compute_mtg_similarities(
            hf_dataset=hf_ds,
            calculator=calc,
            is_vsm=_VSM_mode,
            is_vlm=args.vlm,
            indices=indices_to_run,
            use_masks=args.masks,
            mask_keep=args.mask_keep,
            run=run,
            output_dir=os.path.dirname(output_file),
            debug_samples=50,
        )

    elif args.vlm:
        print(f"\nUsing VLM judge mode with {args.model}")
        print(f"  vl_template: {args.vl_template}")
        calc = Qwen3VLJudgeCalculator(
            model_id=args.model,
            template_path=args.vl_template,
            dtype="auto",
            device_map="auto",
            max_new_tokens=args.max_new_tokens,
        )

        dataset = SynCDSimilarityDataset(
            hf_dataset=hf_ds,
            processor=None,
            mode=args.mode,
            use_masks=args.masks,
            mask_keep=args.mask_keep,
            mask_fill="black",
            mask_dilate=False,
            mask_dilate_k=3,
            max_samples=args.max_samples,
            return_pil=True,
        )
        print(f"Created dataset with {len(dataset)} samples (VLM mode)")

        if args.findx is not None:
            indices_to_run = _load_indices_json(args.findx)
            dataset = torch.utils.data.Subset(dataset, indices_to_run)
            print(f"Filtering dataset to {len(indices_to_run)} specified indices from {args.findx}, len(dataset) is now {len(dataset)}")


        results = compute_similarities_with_vlm_judge(
            dataset=dataset,
            calculator=calc,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_dir=os.path.dirname(output_file),
            debug_samples=50,
            run=run,              # NEW
            stream_every=4,     # optional
        )
    elif _VSM_mode:
        print(f"\nUsing VSM (Mind-the-Glitch) mode")
        calc = VSMCalculator(semantic_threshold=0.6)

        # For VSM we iterate the HF dataset directly (pairwise compute, no DataLoader batching)
        indices_to_run = None
        if args.findx is not None:
            indices_to_run = _load_indices_json(args.findx)
            print(f"Filtering to {len(indices_to_run)} specified indices from {args.findx}")

        results = compute_similarities_with_vsm(
            hf_dataset=hf_ds,
            calculator=calc,
            mode=args.mode,
            max_samples=args.max_samples,
            indices=indices_to_run,
            run=run,
        )
    else:
        if _encodeid_mode:
            print(f"\nUsing EncodeID embedding mode with {args.model}")
            calc = EncodeIDSimilarityCalculator(checkpoint_path=args.model)
        else:
            print(f"\nUsing embedding similarity mode with {args.model}")
            calc = SigLIP2SimilarityCalculator(model_id=args.model, dtype=torch.float16)

        dataset = SynCDSimilarityDataset(
            hf_dataset=hf_ds,
            processor=calc.processor,
            mode=args.mode,
            use_masks=args.masks,
            mask_keep=args.mask_keep,
            mask_fill="black",
            mask_dilate=False,
            mask_dilate_k=3,
            max_samples=args.max_samples,
            return_pil=False,
        )
        print(f"Created dataset with {len(dataset)} samples (embedding mode)")

        if args.findx is not None:
            indices_to_run = _load_indices_json(args.findx)
            dataset = torch.utils.data.Subset(dataset, indices_to_run)
            print(f"Filtering dataset to {len(indices_to_run)} specified indices from {args.findx}, len(dataset) is now {len(dataset)}")

        results = compute_similarities_with_dataloader_batched(
            dataset=dataset,
            calculator=calc,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            run=run,
            stream_table=None,
        )

    print(f"\nComputed similarities for {len(results)} samples")

    # W&B: log summaries + full table (optional)
    # _wandb_log_numeric_summaries(run, results)
    # _wandb_log_results_table(run, results, table_name="results/full")

    # Save CSV

    df = pd.DataFrame(results)
    print("\nDataFrame head:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("\nDescriptive statistics:")
    print(df.describe(include="all"))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nWrote {output_file}")

    # log the full DataFrame directly 
    if run is not None:
        try:
            import wandb
            table = wandb.Table(dataframe=df)
            run.log({"results/full_df": table})
        except Exception as e: 
            print(f'Failed to log full DataFrame to W&B. Error: {e}')



    # Finish W&B run
    if run is not None:
        try:
            import wandb
            import gc
            wandb.finish()
            gc.collect()
            torch.cuda.empty_cache()
            del calc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'Failed to finish W&B run. Error: {e}')