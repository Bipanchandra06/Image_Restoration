# -*- coding: utf-8 -*-
"""
Created on Sat May 24 16:06:20 2025

@author: bipan
"""

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from models.gan_model import UNetGenerator
from models.gan_model import  SimpleUNetGenerator  # Adjust import path if needed
import os

st.set_page_config(page_title="Image Restoration GAN", layout="centered")
st.title("🧠 Image Restoration using GAN")
st.write("Upload a **corrupted image**, and the GAN will restore it.")

# Load config
config = {
    "image_size": 256,
    "checkpoint_path": "models/checkpoints/checkpoint_epoch_99.pth"
}

# Load model
@st.cache_resource
def load_model(model_path, model_type="unet", device="cpu"):
    """
    Load the trained model with proper error handling
    Handles your specific saving format:
    - G_epoch{epoch}.pth (generator only)
    - checkpoint_epoch_{epoch}.pth (full checkpoint)
    
    Args:
        model_path: Path to the saved model
        model_type: Type of model ("unet" or "simple_unet")
        device: Device to load model on
    """
    try:
        # Initialize the model
        if model_type == "simple_unet":
            model = SimpleUNetGenerator().to(device)
        else:
            model = UNetGenerator().to(device)
        
        # Load the state dict
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats based on your saving convention
            if isinstance(checkpoint, dict):
                if 'generator_state_dict' in checkpoint:
                    # Full checkpoint format: checkpoint_epoch_{epoch}.pth
                    model.load_state_dict(checkpoint['generator_state_dict'])
                    epoch = checkpoint.get('epoch', 'unknown')
                    print(f"✓ Loaded full checkpoint from epoch {epoch}")
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume the entire dict is the state dict (G_epoch{epoch}.pth format)
                    model.load_state_dict(checkpoint)
                    print("✓ Loaded generator-only checkpoint")
            else:
                # Direct state dict (G_epoch{epoch}.pth format)
                model.load_state_dict(checkpoint)
                print("✓ Loaded generator-only checkpoint")
                
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model architecture matches the saved checkpoint.")
        print("Available checkpoint types:")
        print("  - G_epoch{N}.pth (generator only)")
        print("  - checkpoint_epoch_{N}.pth (full checkpoint)")
        raise

model= load_model("models/checkpoints/checkpoint_epoch_99.pth")
device="cpu"

# Image upload
uploaded_file = st.file_uploader("Upload a corrupted image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Corrupted Image", use_column_width=False)

    # Load and preprocess
    img = Image.open(uploaded_file).convert("RGB")
    original_size = img.size

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)[0].cpu().clamp(0, 1)

    # Convert to image and resize back to original size
    restored_img = transforms.ToPILImage()(output).resize(original_size)

    from io import BytesIO

    # Show result
    st.image(restored_img, caption="Restored Image", use_column_width=False)

    # Convert to JPG in memory
    buffer = BytesIO()
    restored_img.save(buffer, format="JPEG")
    buffer.seek(0)

    # Download button for JPG
    st.download_button(
        label="Download Restored Image (JPG)",
        data=buffer,
        file_name="restored.jpg",
        mime="image/jpeg"
        )

