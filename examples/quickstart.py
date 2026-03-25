"""Quick start: extract a NearID identity embedding from a single image."""

from transformers import AutoModel, AutoImageProcessor
from PIL import Image

# Load model and processor from HuggingFace Hub
model = AutoModel.from_pretrained("Aleksandar/nearid-siglip2", trust_remote_code=True)
processor = AutoImageProcessor.from_pretrained("Aleksandar/nearid-siglip2")

# Process an image
image = Image.new("RGB", (384, 384), color=(128, 128, 128))  # replace with your image
inputs = processor(images=image, return_tensors="pt")

# Extract embedding
embedding = model.get_image_features(**inputs)  # [1, 1152], L2-normalised
print(f"Embedding shape: {embedding.shape}")
print(f"Embedding norm: {embedding.norm(dim=-1).item():.4f}")
