"""NearID model configuration."""

from typing import Optional

from transformers import PretrainedConfig


# Default vision config matching google/siglip2-so400m-patch14-384
_DEFAULT_VISION_CONFIG = {
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_hidden_layers": 27,
    "num_attention_heads": 16,
    "image_size": 384,
    "patch_size": 14,
    "num_channels": 3,
    "hidden_act": "gelu_pytorch_tanh",
    "layer_norm_eps": 1e-6,
    "attention_dropout": 0.0,
}


class NearIDConfig(PretrainedConfig):
    """Configuration for NearIDModel.

    NearID is an identity embedding model built on a frozen SigLIP2 vision
    encoder with a trained MAP (Multi-head Attention Pooling) head. It produces
    L2-normalized embeddings for image similarity and retrieval tasks.

    Args:
        vision_config: Dictionary of SiglipVisionConfig parameters.
        embed_dim: Dimensionality of the output embedding.
        normalize_embeddings: Whether ``forward()`` returns L2-normalized
            embeddings by default.
    """

    model_type = "nearid"

    def __init__(
        self,
        vision_config: Optional[dict] = None,
        embed_dim: int = 1152,
        normalize_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_config = vision_config or dict(_DEFAULT_VISION_CONFIG)
        self.embed_dim = embed_dim
        self.normalize_embeddings = normalize_embeddings
