from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict, Any, List
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel

from training.models import EncodeIDModel
from training.config import EncodeIDConfig

# Essential for from_pretrained to find your custom logic
AutoConfig.register("encode_id", EncodeIDConfig)
AutoModel.register(EncodeIDConfig, EncodeIDModel)

class EncodeIDInference:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        
        # 1. Native HF Loading 
        # trust_remote_code=True is essential if loading from a remote hub
        self.model = EncodeIDModel.from_pretrained(
            checkpoint_path, 
            trust_remote_code=True,
            # torch_dtype="auto"  # Let HF decide the best dtype (Bf16/Fp16) based on the checkpoint and device
        )
        
        # 2. Precision Sync
        # We derive the precision from the backbone (which was set during training)
        # and cast the entire model (including the new MAP head) to match.
        # target_dtype = next(self.model.encoder_wrapper.parameters()).dtype
        self.model.to(device=self.device) # type: ignore
        self.model.eval()

    @torch.inference_mode() 
    def get_embeddings(self, dataloader) -> Dict[str, Any]:
        all_embeddings = []
        all_pos_masks = []
        all_neg_masks = []
        metadata = []

        model_dtype = next(self.model.parameters()).dtype
        for batch in tqdm(dataloader, desc="Running Inference", leave=False):
            if batch is None: continue
            
            pv = batch["pixel_values"].to(device=self.device, dtype=model_dtype)
            B, S, C, H, W = pv.shape
            # Flatten B*S for the encoder wrapper
            flat_pv = pv.view(B * S, C, H, W)
            inputs = {"pixel_values_anchor": flat_pv}
            
            # Forward + Normalize
            with torch.autocast(device_type=self.device, dtype=model_dtype):
                out = self.model(inputs, side="anchor")

            out = F.normalize(out, p=2, dim=-1, eps=1e-8)
            
            all_embeddings.append(out.view(B, S, -1).cpu())
            all_pos_masks.append(batch["pos_mask"].cpu())
            all_neg_masks.append(batch["neg_mask"].cpu())
            
            for i in range(B):
                metadata.append({
                    "sample_id": batch["sample_id"][i],
                    "category": batch["category"][i],
                })

        return {
            "embeddings": torch.cat(all_embeddings, dim=0),
            "pos_mask": torch.cat(all_pos_masks, dim=0),
            "neg_mask": torch.cat(all_neg_masks, dim=0),
            "metadata": metadata
        }