"""NearID: Identity embedding model built on SigLIP2 + MAP head."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from transformers.utils import ModelOutput

try:
    from .configuration_nearid import NearIDConfig
except ImportError:
    from configuration_nearid import NearIDConfig


@dataclass
class NearIDModelOutput(ModelOutput):
    """Output of :class:`NearIDModel`.

    Attributes:
        image_embeds: L2-normalised identity embeddings ``[B, embed_dim]``.
        last_hidden_state: Patch-level features ``[B, seq_len, hidden_size]``.
        pooler_output: MAP head output *before* normalisation ``[B, embed_dim]``.
        hidden_states: Intermediate hidden states (if requested).
        attentions: Attention weights (if requested).
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class NearIDPreTrainedModel(PreTrainedModel):
    config_class = NearIDConfig
    base_model_prefix = "nearid"
    main_input_name = "pixel_values"
    _no_split_modules = ["SiglipEncoderLayer", "SiglipMultiheadAttentionPoolingHead"]

    def _init_weights(self, module: nn.Module) -> None:
        """Weight init is a no-op — all weights are loaded from checkpoint."""


class NearIDModel(NearIDPreTrainedModel):
    """NearID identity embedding model.

    Frozen SigLIP2 vision encoder with a trained Multi-head Attention Pooling
    (MAP) head.  Produces L2-normalised 1152-d embeddings suitable for cosine
    similarity scoring.

    Example::

        from transformers import AutoModel, AutoImageProcessor
        from PIL import Image

        model = AutoModel.from_pretrained("nearid-siglip2", trust_remote_code=True)
        processor = AutoImageProcessor.from_pretrained("nearid-siglip2")

        inputs = processor(images=Image.open("photo.jpg"), return_tensors="pt")
        emb = model(**inputs).image_embeds          # [1, 1152]
        emb = model.get_image_features(**inputs)     # same, tensor shortcut
    """

    def __init__(self, config: NearIDConfig):
        super().__init__(config)
        vcfg = SiglipVisionConfig(**config.vision_config)
        self.backbone = SiglipVisionModel(vcfg)
        self.post_init()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        normalize: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Return identity embeddings as a plain tensor.

        Args:
            pixel_values: ``[B, C, H, W]`` preprocessed images.
            normalize: L2-normalise the output (default ``True``).

        Returns:
            ``torch.Tensor`` of shape ``[B, embed_dim]``.
        """
        out = self.backbone(
            pixel_values=pixel_values,
            return_dict=True,
        )
        pooled = out.pooler_output
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[NearIDModelOutput, Tuple]:
        """
        Args:
            pixel_values: ``[B, C, H, W]`` preprocessed images.
            output_hidden_states: Return intermediate hidden states.
            output_attentions: Return attention weights.
            return_dict: Return :class:`NearIDModelOutput` instead of tuple.

        Returns:
            :class:`NearIDModelOutput` with ``image_embeds`` (normalised),
            ``pooler_output`` (raw), and ``last_hidden_state``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        out = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        pooler_output = out.pooler_output
        if self.config.normalize_embeddings:
            image_embeds = F.normalize(pooler_output, p=2, dim=-1)
        else:
            image_embeds = pooler_output

        if not return_dict:
            return (image_embeds, out.last_hidden_state, pooler_output)

        return NearIDModelOutput(
            image_embeds=image_embeds,
            last_hidden_state=out.last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=getattr(out, "hidden_states", None),
            attentions=getattr(out, "attentions", None),
        )


# ---------------------------------------------------------------------------
# AutoClass registration
# ---------------------------------------------------------------------------
AutoConfig.register("nearid", NearIDConfig)
AutoModel.register(NearIDConfig, NearIDModel)
NearIDConfig.register_for_auto_class()
NearIDModel.register_for_auto_class("AutoModel")
