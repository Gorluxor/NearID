from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    CLIPVisionModel,
    Dinov2Model,
    PreTrainedModel,
    SiglipModel,
    SiglipVisionModel,
    ViTMAEModel,
    ViTModel,
    Qwen2VLModel,
)

# Handle Qwen3 import alias
try:
    from transformers import Qwen3VLModel
except ImportError:
    Qwen3VLModel = Qwen2VLModel

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

from .config import NearIDConfig


# ==============================================================================
# 0. Best Practices Registry (Single Source of Truth)
# ==============================================================================

BACKBONE_REGISTRY = {
    "siglip2": {
        "repo_id": "google/siglip2-so400m-patch14-384",
        "dtype": torch.float16,
        "pooling": "mean",
    },
    "siglip": {
        "repo_id": "google/siglip-so400m-patch14-384",
        "dtype": torch.float32,
        "pooling": "mean",
    },
    "qwen2.5vl": {
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "dtype": torch.bfloat16,
        "pooling": "image_mean",
    },
    "qwen3vl": {
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "dtype": torch.bfloat16,
        "pooling": "image_mean",
    },
    "clip": {
        "repo_id": "openai/clip-vit-large-patch14",
        "dtype": torch.float16,
        "pooling": "cls",
    },
    "vit": {
        "repo_id": "facebook/vit-mae-large",
        "dtype": torch.float32,
        "pooling": "cls",
    },
    "dinov2": {
        "repo_id": "facebook/dinov2-large",
        "dtype": torch.float32,
        "pooling": "cls",
    },
}

def get_backbone_defaults(name: str) -> Dict[str, Any]:
    name = name.lower()
    if name not in BACKBONE_REGISTRY:
        return {"dtype": torch.float32, "pooling": "cls", "repo_id": None}
    return BACKBONE_REGISTRY[name]


# ==============================================================================
# 1. Universal Projection Head
# ==============================================================================

class UniversalProjectionHead(nn.Module):
    """
    1. Identity (num_layers=0) -> Raw features.
    2. Linear Probe (num_layers=1) -> CLIP-style projection.
    3. MLP (num_layers>1) -> SSL/SimCLR style projection.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        use_batchnorm: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        if use_layernorm and use_batchnorm:
            use_layernorm = False
        if num_layers == 0:
            self.net = nn.Identity()
            return

        if num_layers == 1:
            self.net = nn.Linear(in_dim, out_dim, bias=use_bias)
            return

        if hidden_dim is None:
            hidden_dim = in_dim

        layers = []
        d = in_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim, bias=use_bias))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            if use_batchnorm:
                layers.append(nn.SyncBatchNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim

        layers.append(nn.Linear(d, out_dim, bias=use_bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==============================================================================
# 2. Encoders
# ==============================================================================

@dataclass
class EncoderOutput:
    embedding: torch.Tensor

class BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def embed_dim(self) -> int:
        raise NotImplementedError

    def forward(self, batch: Dict[str, Any], side: str) -> EncoderOutput:
        raise NotImplementedError


class VisionEncoderHF(BaseEncoder):
    """ViT / DINOv2 / MAE"""
    def __init__(self, model: nn.Module, pool: str = "cls"):
        super().__init__()
        self.model = model
        self.pool = pool
        self._embed_dim = self.model.config.hidden_size

    @property
    def embed_dim(self) -> int:
        return int(self._embed_dim)

    def forward(self, batch: Dict[str, Any], side: str) -> EncoderOutput:
        pixel_values = batch[f"pixel_values_{side}"]
        out = self.model(pixel_values=pixel_values, return_dict=True)
        hs = out.last_hidden_state

        if self.pool == "cls":
            emb = hs[:, 0]
        elif self.pool == "mean":
            emb = hs[:, 1:].mean(dim=1) if hs.shape[1] > 1 else hs.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")
        return EncoderOutput(embedding=emb)


class CLIPRawEncoder(BaseEncoder):
    def __init__(self, model: CLIPVisionModel, pool: str = "cls"):
        super().__init__()
        self.model = model
        self.pool = pool
        self._embed_dim = self.model.config.hidden_size

    @property
    def embed_dim(self) -> int:
        return int(self._embed_dim)

    def forward(self, batch: Dict[str, Any], side: str) -> EncoderOutput:
        pixel_values = batch[f"pixel_values_{side}"]
        out = self.model(pixel_values=pixel_values, return_dict=True)

        if self.pool == "pooler" and hasattr(out, "pooler_output"):
            emb = out.pooler_output
        elif self.pool == "cls":
            emb = out.last_hidden_state[:, 0]
        elif self.pool == "mean":
            emb = out.last_hidden_state[:, 1:].mean(dim=1)
        else:
            raise ValueError(f"Unknown or unsupported pool type for CLIP: {self.pool}")

        return EncoderOutput(embedding=emb)

from transformers.models.siglip.modeling_siglip import SiglipMultiheadAttentionPoolingHead

class StandaloneSiglipMAPHead(nn.Module):
    def __init__(self, config, pretrained_state_dict=None):
        super().__init__()
        self.map_pooler = SiglipMultiheadAttentionPoolingHead(config)

        if pretrained_state_dict is not None:
            self.map_pooler.load_state_dict(pretrained_state_dict, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        target_dtype = next(self.map_pooler.parameters()).dtype
        return self.map_pooler(x.to(target_dtype)).to(x_dtype)


class SiglipRawEncoder(BaseEncoder):
    def __init__(self, model: SiglipVisionModel, pool: str = "mean", layer_idx: int = -1):
        super().__init__()
        self.model = model
        self.pool = pool
        self.layer_idx = layer_idx
        self._embed_dim = self.model.config.hidden_size

        vision_trunk: nn.Module = getattr(self.model, "vision_model", self.model)
        self.post_ln = nn.LayerNorm(self._embed_dim, eps=self.model.config.layer_norm_eps)
        self.post_ln.load_state_dict(vision_trunk.post_layernorm.state_dict())  # type:ignore

    @property
    def embed_dim(self) -> int:
        return int(self._embed_dim)

    def forward(self, batch: Dict[str, Any], side: str) -> EncoderOutput:
        pixel_values = batch[f"pixel_values_{side}"]
        out = self.model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)

        if self.layer_idx == -1:
            hs = out.last_hidden_state
        else:
            hs = out.hidden_states[self.layer_idx]
            hs = self.post_ln(hs)

        if not torch.isfinite(hs).all():
            raise RuntimeError(f"Non-finite hs. dtype={hs.dtype} max={hs.abs().max().item()}")

        if self.pool == "none":
            emb = hs
        elif self.pool == "mean":
            emb = hs.mean(dim=1)
        elif self.pool == "cls":
            if hs.shape[1] <= 1:
                raise ValueError("SigLIP model returned sequence length <= 1. 'cls' pooling invalid.")
            emb = hs[:, 0]
        elif self.pool == "last":
            emb = hs[:, -1]
        else:
            raise ValueError(f"Unknown pool type for SigLIP: {self.pool}")

        return EncoderOutput(embedding=emb)


def _first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for y in x:
            t = _first_tensor(y)
            if t is not None:
                return t
    return None

class Qwen3VLEncoder(BaseEncoder):
    """Qwen3VL / Qwen2VL encoder."""
    def __init__(self, model: nn.Module, pool: str = "image_mean"):
        super().__init__()
        self.model = model
        self.pool = pool

        cfg = model.config
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self._embed_dim = int(cfg.text_config.hidden_size)
        elif hasattr(cfg, "hidden_size"):
            self._embed_dim = int(cfg.hidden_size)
        else:
            raise AttributeError("Cannot infer embed dim for Qwen model.")

        self.image_token_id = getattr(model.config, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = 151655

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, batch: Dict[str, Any], side: str) -> EncoderOutput:
        inputs = {
            "input_ids": batch.get(f"input_ids_{side}"),
            "attention_mask": batch.get(f"attention_mask_{side}"),
            "pixel_values": batch.get(f"pixel_values_{side}"),
            "image_grid_thw": batch.get(f"image_grid_thw_{side}"),
        }
        inputs = {k: v for k, v in inputs.items() if v is not None}

        base = self.model.model if hasattr(self.model, "model") else self.model

        input_ids = inputs.get("input_ids")
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        no_image_tokens = False
        if input_ids is None:
            no_image_tokens = True
        else:
            if (input_ids == self.image_token_id).sum().item() == 0:
                no_image_tokens = True

        if pixel_values is not None and no_image_tokens:
            def call_get_image_features(fn, pixel_values, image_grid_thw):
                try:
                    if image_grid_thw is None:
                        return fn(pixel_values=pixel_values)
                    return fn(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
                except TypeError:
                    if image_grid_thw is None:
                        return fn(pixel_values)
                    return fn(pixel_values, image_grid_thw)

            if hasattr(self.model, "get_image_features"):
                vision_out = call_get_image_features(self.model.get_image_features, pixel_values, image_grid_thw)
            elif hasattr(base, "get_image_features"):
                vision_out = call_get_image_features(base.get_image_features, pixel_values, image_grid_thw)
            else:
                raise AttributeError("Qwen model does not expose get_image_features()")

            pooler = getattr(vision_out, "pooler_output", None)
            if pooler is not None:
                if isinstance(pooler, (list, tuple)):
                    pooled = []
                    for p in pooler:
                        if p.dim() == 1:
                            pooled.append(p)
                        elif p.dim() == 2:
                            pooled.append(p.mean(dim=0))
                        else:
                            pooled.append(p.reshape(-1, p.shape[-1]).mean(dim=0))
                    emb = torch.stack(pooled, dim=0)
                else:
                    if pooler.dim() == 1:
                        emb = pooler.unsqueeze(0)
                    elif pooler.dim() == 2:
                        emb = pooler
                    elif pooler.dim() == 3:
                        emb = pooler.mean(dim=1)
                    else:
                        emb = pooler.mean(dim=(1, 2))
            else:
                lhs = getattr(vision_out, "last_hidden_state", None)
                if not torch.is_tensor(lhs):
                    lhs = _first_tensor(lhs if lhs is not None else vision_out)
                assert lhs is not None and torch.is_tensor(lhs)
                if input_ids is not None:
                    B = input_ids.shape[0]
                elif image_grid_thw is not None:
                    B = image_grid_thw.shape[0]
                else:
                    B = 1

                if lhs.dim() == 2:
                    emb = lhs if lhs.shape[0] == B else lhs.mean(dim=0, keepdim=True)
                elif lhs.dim() == 3:
                    emb = lhs.mean(dim=1)
                elif lhs.dim() == 4:
                    emb = lhs.mean(dim=(1, 2))
                else:
                    raise ValueError(f"Unexpected tensor shape from vision_out: {lhs.shape}")
            return EncoderOutput(embedding=emb)

        # Multimodal forward
        out = base(**inputs, return_dict=True, output_hidden_states=False)
        hs = out.last_hidden_state

        if self.pool == "last":
            mask = inputs.get("attention_mask")
            if mask is not None:
                last_idx = mask.sum(dim=1) - 1
                emb = hs[torch.arange(hs.shape[0], device=hs.device), last_idx]
            else:
                emb = hs[:, -1]

        elif self.pool == "image_mean":
            if input_ids is None:
                raise ValueError("pool='image_mean' requires input_ids")
            img_mask = (input_ids == self.image_token_id).float()
            denom = torch.clamp(img_mask.sum(dim=1, keepdim=True), min=1e-9)
            emb = (hs * img_mask.unsqueeze(-1)).sum(dim=1) / denom

        else:
            raise ValueError(f"Unknown pool type for Qwen3VL: {self.pool}")

        return EncoderOutput(embedding=emb)


# ==============================================================================
# 3. Model Bundle
# ==============================================================================

@dataclass
class ModelBundle:
    encoder: BaseEncoder
    head: nn.Module
    processor: Any
    is_multimodal: bool

    def forward(self, batch: Dict[str, Any], side: str = "anchor") -> torch.Tensor:
        key_map = {
            "pixel_values": f"pixel_values_{side}",
            "input_ids": f"input_ids_{side}",
            "attention_mask": f"attention_mask_{side}",
            "image_grid_thw": f"image_grid_thw_{side}",
        }

        batch_size = 0
        num_views = 1
        has_views = False

        ids = batch.get(key_map["input_ids"])
        if ids is not None:
            batch_size = ids.shape[0]
            if ids.dim() == 3:
                has_views = True
                num_views = ids.shape[1]

        pv = batch.get(key_map["pixel_values"])
        if pv is not None:
            if batch_size == 0:
                batch_size = pv.shape[0]
            if pv.dim() == 5:
                has_views = True
                num_views = pv.shape[1]

        if batch_size == 0:
            raise ValueError(f"No valid input keys found for side '{side}'")

        target_bs = batch_size * num_views
        encoder_input = {}

        for generic_key, specific_key in key_map.items():
            data = batch.get(specific_key)
            if data is None:
                continue

            if generic_key == "image_grid_thw":
                if isinstance(data, (list, tuple)):
                    assert pv is not None
                    data = torch.tensor(data, device=ids.device if ids is not None else pv.device, dtype=torch.long)
                elif isinstance(data, torch.Tensor):
                    data = data.to(torch.long)

            current_bs = data.shape[0]

            if current_bs == target_bs:
                encoder_input[specific_key] = data
            elif has_views and current_bs == batch_size and data.shape[1] == num_views:
                encoder_input[specific_key] = data.reshape(-1, *data.shape[2:])
            elif current_bs == batch_size:
                if has_views:
                    encoder_input[specific_key] = data.repeat_interleave(num_views, dim=0)
                else:
                    encoder_input[specific_key] = data
            elif generic_key == "pixel_values" and self.is_multimodal and data.dim() == 2:
                encoder_input[specific_key] = data
            else:
                raise ValueError(f"Shape mismatch for {specific_key}: {data.shape}")

        enc_out = self.encoder(encoder_input, side)
        proj_emb = self.head(enc_out.embedding)

        if has_views:
            if proj_emb.shape[0] == target_bs:
                proj_emb = proj_emb.reshape(batch_size, num_views, -1)

        return proj_emb


def build_encoder_and_processors(
    backbone: str,
    pretrained: str,
    *,
    trust_remote_code: bool = False,
    vision_pool: Optional[str] = None,
    qwen_pool: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[BaseEncoder, Any, bool]:

    backbone = backbone.lower()
    defaults = get_backbone_defaults(backbone)

    if dtype is None:
        dtype = defaults["dtype"]
    if vision_pool is None:
        vision_pool = defaults["pooling"]
    if qwen_pool is None:
        qwen_pool = defaults["pooling"]

    assert vision_pool in ("cls", "mean", "pooler", "last", "none"), f"Invalid vision_pool: {vision_pool}"

    if backbone == "vit":
        model = ViTModel.from_pretrained(pretrained, dtype=dtype)
        enc = VisionEncoderHF(model, pool=vision_pool)
        proc = AutoImageProcessor.from_pretrained(pretrained, use_fast=False)
        return enc, proc, False

    if backbone == "dinov2":
        model = Dinov2Model.from_pretrained(pretrained, dtype=dtype)
        enc = VisionEncoderHF(model, pool=vision_pool)
        proc = AutoImageProcessor.from_pretrained(pretrained, use_fast=False)
        return enc, proc, False

    if backbone == "clip":
        model = CLIPVisionModel.from_pretrained(pretrained, dtype=dtype)
        enc = CLIPRawEncoder(model, pool=vision_pool)
        proc = AutoProcessor.from_pretrained(pretrained, use_fast=False)
        return enc, proc, False

    if backbone == "siglip2":
        model = SiglipVisionModel.from_pretrained(pretrained, dtype=dtype)
        enc = SiglipRawEncoder(model, pool=vision_pool)
        proc = AutoProcessor.from_pretrained(pretrained, use_fast=False)
        return enc, proc, False

    if backbone in ("vit-mae", "vitmae", "mae"):
        model = ViTMAEModel.from_pretrained(pretrained, dtype=dtype)
        enc = VisionEncoderHF(model, pool=vision_pool)
        proc = AutoImageProcessor.from_pretrained(pretrained, use_fast=False)
        return enc, proc, False

    if "qwen" in backbone:
        assert qwen_pool in ("last", "image_mean"), f"Invalid default qwen_pool: {qwen_pool}"
        if Qwen3VLForConditionalGeneration is not None:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained, dtype=dtype, trust_remote_code=trust_remote_code, attn_implementation="eager",
            )
        else:
            model = Qwen3VLModel.from_pretrained(
                pretrained, dtype=dtype, trust_remote_code=trust_remote_code, attn_implementation="eager",
            )
        enc = Qwen3VLEncoder(model, pool=qwen_pool)
        proc = AutoProcessor.from_pretrained(pretrained, trust_remote_code=trust_remote_code, use_fast=False)
        return enc, proc, True

    raise ValueError(f"Unknown backbone: {backbone}")


# ==============================================================================
# 4. NearIDModel (HuggingFace PreTrainedModel for training)
# ==============================================================================

class NearIDPreTrainedModel(PreTrainedModel):
    config_class = NearIDConfig
    base_model_prefix = "nearid"
    _keys_to_ignore_on_load_missing = [r"encoder_wrapper.*"]

class NearIDModel(NearIDPreTrainedModel):
    def __init__(self, config: NearIDConfig):
        super().__init__(config)
        self.config = config

        # Handle pooling override for MAP head
        vision_pool = config.pooling
        if config.head_type == "map":
            vision_pool = "none"

        # Build Backbone & Processor
        self.encoder_wrapper, self.processor, _ = build_encoder_and_processors(
            backbone=config.backbone,
            pretrained=config.pretrained_backbone,
            vision_pool=vision_pool,
            qwen_pool=vision_pool,
            trust_remote_code=True
        )
        if hasattr(self.encoder_wrapper, "layer_idx"):
            self.encoder_wrapper.layer_idx = int(config.layer_idx)

        # Projection Head
        _map_pooler_sd = None
        if config.head_type == "map":
            vision_trunk = getattr(self.encoder_wrapper.model, "vision_model", self.encoder_wrapper.model)

            if hasattr(vision_trunk, "head"):
                head = getattr(vision_trunk, "head", None)
                assert head is not None
                assert isinstance(head, nn.Module)
                _map_pooler_sd = {k: v.clone() for k, v in head.state_dict().items()}

                map_head = StandaloneSiglipMAPHead(
                    config=self.encoder_wrapper.model.config
                )

                if config.head_out_dim != self.encoder_wrapper.embed_dim:
                    self.head = nn.Sequential(
                        map_head,
                        nn.Linear(self.encoder_wrapper.embed_dim, config.head_out_dim, bias=True)
                    )
                else:
                    self.head = map_head
            else:
                raise ValueError(f"Requested MAP head, but could not find '.head' inside {type(vision_trunk).__name__}.")
        else:
            self.head = UniversalProjectionHead(
                in_dim=self.encoder_wrapper.embed_dim,
                out_dim=config.head_out_dim,
                hidden_dim=config.head_hidden_dim,
                num_layers=config.head_layers,
                dropout=config.head_dropout
            )

        self.post_init()

        # Bootstrap MAP weights from backbone on fresh init
        if config.head_type == "map" and _map_pooler_sd is not None:
            is_fresh_init = getattr(config, "_name_or_path", "") in ("", None)
            if is_fresh_init:
                map_part = self.head[0] if isinstance(self.head, nn.Sequential) else self.head
                assert isinstance(map_part, StandaloneSiglipMAPHead)
                map_part.map_pooler.load_state_dict(_map_pooler_sd, strict=True)

    def save_pretrained(self, save_directory, *args, **kwargs):
        super().save_pretrained(save_directory, *args, **kwargs)
        proc = getattr(self, "processor", None)
        if proc is not None and hasattr(proc, "save_pretrained"):
            proc.save_pretrained(save_directory)

    def forward(self, batch: Dict[str, Any], side: str = "anchor") -> torch.Tensor:
        enc_out = self.encoder_wrapper(batch, side=side)
        features = self.head(enc_out.embedding)
        return features

# Register with HF Auto classes
AutoConfig.register("nearid", NearIDConfig)
AutoModel.register(NearIDConfig, NearIDModel)
NearIDConfig.register_for_auto_class()
NearIDModel.register_for_auto_class("AutoModel")
