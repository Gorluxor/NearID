"""Compare two images using NearID identity embeddings."""

import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

model = AutoModel.from_pretrained("Aleksandar/nearid-siglip2", trust_remote_code=True)
processor = AutoImageProcessor.from_pretrained("Aleksandar/nearid-siglip2")

# Load two images (replace with your own)
img_a = Image.new("RGB", (384, 384), color=(200, 100, 50))
img_b = Image.new("RGB", (384, 384), color=(200, 105, 55))

# Extract embeddings
with torch.no_grad():
    emb_a = model.get_image_features(**processor(images=img_a, return_tensors="pt"))
    emb_b = model.get_image_features(**processor(images=img_b, return_tensors="pt"))

# Cosine similarity (embeddings are already L2-normalised)
similarity = (emb_a @ emb_b.T).item()
print(f"Cosine similarity: {similarity:.4f}")
