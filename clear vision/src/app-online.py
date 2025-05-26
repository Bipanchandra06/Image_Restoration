# -*- coding: utf-8 -*-
"""
Created on Sun May 25 23:35:42 2025

@author: bipan
"""

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from models.gan_model import UNetGenerator
from models.gan_model import SimpleUNetGenerator  # Adjust import path if needed
import os
import gdown
import tempfile
import time

st.set_page_config(page_title="Image Restoration GAN", layout="centered")
st.title("🧠 Image Restoration using GAN")
st.write("Upload a **corrupted image**, and the GAN will restore it.")

# Load config with Google Drive file ID
config = {
    "image_size": 256,
    "gdrive_file_id": "1tPw9pYdMAcGmQnpHzCYrLKhxp_DdsNc9",  # Replace with your actual file ID
    "checkpoint_filename": "checkpoint_epoch_99.pth"
}

def download_model_from_gdrive(file_id, filename):
    """
    Download model from Google Drive using gdown
    
    Args:
        file_id: Google Drive file ID
        filename: Name to save the file as
    
    Returns:
        Path to downloaded file
    """
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)
        
        # Construct Google Drive URL
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # Download the file
        st.info("Downloading model from Google Drive...")
        gdown.download(url, file_path, quiet=False)
        
        if os.path.exists(file_path):
            st.success("Model downloaded successfully!")
            return file_path
        else:
            raise FileNotFoundError("Failed to download model file")
            
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        st.error("Please check your Google Drive file ID and make sure the file is publicly accessible")
        raise

# Load model
@st.cache_resource
def load_model(gdrive_file_id, filename, model_type="unet", device="cpu"):
    """
    Load the trained model from Google Drive with proper error handling
    Handles your specific saving format:
    - G_epoch{epoch}.pth (generator only)
    - checkpoint_epoch_{epoch}.pth (full checkpoint)
    
    Args:
        gdrive_file_id: Google Drive file ID
        filename: Name of the checkpoint file
        model_type: Type of model ("unet" or "simple_unet")
        device: Device to load model on
    """
    try:
        # Download model from Google Drive
        model_path = download_model_from_gdrive(gdrive_file_id, filename)
        
        # Initialize the model
        if model_type == "simple_unet":
            model = SimpleUNetGenerator().to(device)
        else:
            model = UNetGenerator().to(device)
        
        # Load the state dict
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
                
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model architecture matches the saved checkpoint.")
        print("Available checkpoint types:")
        print("  - G_epoch{N}.pth (generator only)")
        print("  - checkpoint_epoch_{N}.pth (full checkpoint)")
        raise

# Initialize model loading
device = "cpu"
try:
    model = load_model(
        config["gdrive_file_id"], 
        config["checkpoint_filename"], 
        device=device
    )
except Exception as e:
    st.error("Failed to load model. Please check the Google Drive configuration.")
    st.stop()

# Image upload
uploaded_file = st.file_uploader("Upload a corrupted image", type=["jpg", "jpeg", "png"])
start_time=time.time()
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
    in_latency=time.time()-start_time
    # Convert to image and resize back to original size
    restored_img = transforms.ToPILImage()(output).resize(original_size)

    from io import BytesIO

    # Show result
    st.image(restored_img, caption="Restored Image", use_column_width=False)
    st.info(f'Interference latency={in_latency}s')

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
