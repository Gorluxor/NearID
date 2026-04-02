from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict
from transformers import PretrainedConfig
from simple_parsing.helpers import Serializable, list_field


# --- 1. The Hugging Face "Model" Config (Artifact) ---
class NearIDConfig(PretrainedConfig):
    model_type = "nearid"

    def __init__(
        self,
        backbone: str = "siglip2",
        pretrained_backbone: str = "google/siglip2-so400m-patch14-384",
        pooling: str = "none",
        head_type: str = "map",
        layer_idx: int = -1,
        head_hidden_dim: int = 1024,
        head_out_dim: int = 1152,
        head_layers: int = 2,
        head_dropout: float = 0.0,
        use_batchnorm: bool = True,
        use_layernorm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.pretrained_backbone = pretrained_backbone
        self.pooling = pooling
        self.head_type = head_type
        self.layer_idx = layer_idx
        self.head_hidden_dim = head_hidden_dim
        self.head_out_dim = head_out_dim
        self.head_layers = head_layers
        self.head_dropout = head_dropout
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm


# --- 2. The Training "Runtime" Config (CLI) ---

@dataclass
class WandbConfig(Serializable):
    project: str = "NearID"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = list_field("nearid", "research", "siglip2")
    mode: Literal["online", "offline", "disabled"] = "online"

@dataclass
class DataConfig(Serializable):
    train_path: str = "Aleksandar/NearID"
    neg_paths: List[str] = list_field()
    neg_paths_val: Optional[List[str]] = list_field(default=[])
    neg_paths_test: Optional[List[str]] = list_field(default=[])
    train_indices_path: Optional[str] = None
    val_indices_path: Optional[str] = None
    test_indices_path: Optional[str] = None

    batch_size: int = 256
    num_workers: int = 4
    mask_prob: float = 0.0
    shuffle_anchor: bool = True
    per_slot_neg_dataset: bool = True
    margin_map: Dict[str, float] = field(default_factory=lambda: {
        "qwen-1328": 0.050,
        "sdxl-1024": 0.075,
        "flux-1024": 0.100,
        "fluxc": 0.100,
        "fluxc-1024": 0.125,
        "flux": 0.150,
        "powerpaint": 0.150,
        "qwen": 0.175,
        "sdxl": 0.200,
        "default": 0.100
    })
    margin_field: str = "method_tag"
    mask_prob_apn: Optional[List[float]] = None

    # Augmentations
    flip_prob: float = 0.0
    color_jitter_prob: float = 0.0
    translate_prob: float = 0.0
    translate_fraction: float = 0.1
    scale_range_min: float = 0.9
    scale_range_max: float = 1.1

    # MTG joint training
    mtg_train_path: Optional[str] = None
    mtg_split: str = "train"
    mtg_margin: float = 0.1
    mtg_min: float = 1.0
    mtg_factor: float = 1.0
    mtg_repeat: int = 1

@dataclass
class TrainConfig(Serializable):
    """Master configuration for the NearID training run."""
    # Groups
    wandb: WandbConfig = field(default_factory=WandbConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Model Params
    backbone: str = "siglip2"
    pretrained_backbone: str = "google/siglip2-so400m-patch14-384"
    pooling: str = "none"
    layer_idx: int = -1

    # Loss Config
    loss_config: str = "infonce_ext:1.0"

    # Optimization
    output_dir: str = "./runs/trains"
    seed: int = 42
    epochs: int = 11
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"

    # Checkpointing & Hub
    save_steps: int = 100
    eval_steps: int = 100
    eval_start: int = 0
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    dry_run: bool = False

    # Evaluation Config
    eval_batch_size: int = 32

    # Model config
    head_type: str = "map"
    head_dropout: float = 0.0
    head_hidden_dim: int = 1024
    head_out_dim: int = 1152
    use_batchnorm: bool = True
    use_layernorm: bool = False
    head_layers: int = 2
