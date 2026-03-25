"""Batch inference: embed multiple images and compute a similarity matrix."""

import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from pathlib import Path

model = AutoModel.from_pretrained("Aleksandar/nearid-siglip2", trust_remote_code=True)
processor = AutoImageProcessor.from_pretrained("Aleksandar/nearid-siglip2")
model.eval()

# Load images from a directory (or replace with your own list)
# images = [Image.open(p) for p in sorted(Path("your_images/").glob("*.jpg"))]
images = [Image.new("RGB", (384, 384), color=(i * 30, 100, 200)) for i in range(5)]  # demo

# Batch process
inputs = processor(images=images, return_tensors="pt", padding=True)

with torch.no_grad():
    embeddings = model.get_image_features(**inputs)  # [N, 1152]

# Pairwise similarity matrix
sim_matrix = embeddings @ embeddings.T
print(f"Embeddings shape: {embeddings.shape}")
print(f"Similarity matrix:\n{sim_matrix.numpy()}")
