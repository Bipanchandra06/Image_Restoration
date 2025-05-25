# src/data/corruption.py

import os
import numpy as np
import yaml
import cv2
import random
from PIL import Image
import io
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageCorruption:
    """
    Module for applying various corruption methods to images
    """
    
    def __init__(self,config=None, config_path="/users/bipan/onedrive/desktop/clear vision/config/config.yaml"):
        """
        Initialize the corruption module
        
        Args:
            config_path (str): Path to configuration file
        """
        if config is None:
            with open('config/config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            self.corruption_config = self.config['corruption']
            
        # Create output directory for corrupted images
        self.output_dir = os.path.join(self.config['scraper']['download_path'], '../corrupted')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def apply_corruptions(self, image_paths, save_corrupted=True, corruption_types=None):
        """
        Apply corruptions to a list of images
        
        Args:
            image_paths (list): Paths to original images
            save_corrupted (bool): Whether to save corrupted images
            corruption_types (list): Types of corruptions to apply, defaults to config
            
        Returns:
            dict: Mapping from original paths to corrupted paths
        """
        corruption_mapping = {}
        corruption_types = corruption_types or self.corruption_config['types']
        
        for img_path in tqdm(image_paths, desc="Corrupting images"):
            try:
                # Read the image
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Could not read image: {img_path}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Choose a random corruption type
                corruption_type = random.choice(corruption_types)
                
                # Apply the corruption
                corrupted_img = self._apply_corruption(img, corruption_type)
                
                if save_corrupted:
                    # Save the corrupted image
                    filename = os.path.basename(img_path)
                    base, ext = os.path.splitext(filename)
                    corrupted_filename = f"{base}_corrupted_{corruption_type}{ext}"
                    corrupted_path = os.path.join(self.output_dir, corrupted_filename)
                    
                    # Convert back to BGR for OpenCV
                    corrupted_img_bgr = cv2.cvtColor(corrupted_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(corrupted_path, corrupted_img_bgr)
                    
                    corruption_mapping[img_path] = corrupted_path
                else:
                    corruption_mapping[img_path] = corrupted_img
                    
            except Exception as e:
                logger.error(f"Error corrupting {img_path}: {e}")
                
        return corruption_mapping
    
    def _apply_corruption(self, img, corruption_type):
        """
        Apply specific corruption to an image
        
        Args:
            img (numpy.ndarray): Input image
            corruption_type (str): Type of corruption to apply
            
        Returns:
            numpy.ndarray: Corrupted image
        """
        if corruption_type == "noise":
            return self._add_noise(img)
        elif corruption_type == "jpeg":
            return self._jpeg_compression(img)
        elif corruption_type == "blur":
            return self._add_blur(img)
        elif corruption_type == "mask":
            return self._add_mask(img)
        elif corruption_type == "combined":
            # Apply multiple corruptions
            corrupted = img.copy()
            num_corruptions = random.randint(2, 4)
            corruption_choices = ["noise", "jpeg", "blur", "mask"]
            selected_corruptions = random.sample(corruption_choices, num_corruptions)
            
            for c_type in selected_corruptions:
                corrupted = self._apply_corruption(corrupted, c_type)
                
            return corrupted
        else:
            logger.warning(f"Unknown corruption type: {corruption_type}")
            return img
            
    def _add_noise(self, img):
        """Add Gaussian or salt-and-pepper noise to image"""
        noise_type = random.choice(["gaussian", "salt_pepper"])
        
        if noise_type == "gaussian":
            # Get noise parameters
            noise_range = self.corruption_config["noise_params"]["gaussian_range"]
            std = random.uniform(noise_range[0], noise_range[1])
            
            # Generate noise
            noise = np.random.normal(0, std, img.shape).astype(np.float32)
            noisy_img = np.clip(img.astype(np.float32) + noise * 255, 0, 255).astype(np.uint8)
            return noisy_img
            
        else:  # salt_pepper
            # Get noise parameters
            s_p_range = self.corruption_config["noise_params"]["salt_pepper_range"]
            prob = random.uniform(s_p_range[0], s_p_range[1])
            
            # Generate a salt and pepper mask
            mask = np.random.random(img.shape[:2])
            salt = (mask < prob/2).astype(np.uint8)
            pepper = (mask > 1 - prob/2).astype(np.uint8)
            
            # Apply salt and pepper noise
            noisy_img = img.copy()
            noisy_img[salt > 0] = 255
            noisy_img[pepper > 0] = 0
            
            return noisy_img
    
    def _jpeg_compression(self, img):
        """Apply JPEG compression artifacts"""
        # Get compression quality
        quality_range = self.corruption_config["jpeg_params"]["quality_range"]
        quality = random.randint(quality_range[0], quality_range[1])
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        
        # Apply JPEG compression
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_img = np.array(Image.open(buffer))
        
        return compressed_img
        
    def _add_blur(self, img):
        """Apply Gaussian blur to image"""
        # Get blur parameters
        kernel_range = self.corruption_config["blur_params"]["kernel_range"]
        sigma_range = self.corruption_config["blur_params"]["sigma_range"]
        
        # Ensure kernel size is odd
        kernel_size = random.randint(kernel_range[0], kernel_range[1])
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        
        # Apply Gaussian blur
        blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        
        return blurred_img
        
    def _add_mask(self, img):
        """Add random masks (simulating occlusion)"""
        # Get mask parameters
        ratio_range = self.corruption_config["mask_params"]["ratio_range"]
        num_boxes_range = self.corruption_config["mask_params"]["num_boxes_range"]
        
        # Create a copy of the image
        masked_img = img.copy()
        h, w = img.shape[:2]
        
        # Generate random boxes
        num_boxes = random.randint(num_boxes_range[0], num_boxes_range[1])
        
        for _ in range(num_boxes):
            # Random box size
            ratio = random.uniform(ratio_range[0], ratio_range[1])
            box_w = int(w * ratio)
            box_h = int(h * ratio)
            
            # Random position
            x = random.randint(0, w - box_w)
            y = random.randint(0, h - box_h)
            
            # Random color for the box (black, white, or random)
            color_type = random.choice(["black", "white", "random"])
            
            if color_type == "black":
                color = (0, 0, 0)
            elif color_type == "white":
                color = (255, 255, 255)
            else:
                color = tuple(random.randint(0, 255) for _ in range(3))
                
            # Apply mask
            masked_img[y:y+box_h, x:x+box_w] = color
            
        return masked_img

    def generate_paired_dataset(self, original_dir, corrupted_dir, output_dir):
        """
        Create a dataset of paired original and corrupted images
        
        Args:
            original_dir (str): Directory with original images
            corrupted_dir (str): Directory for corrupted images
            output_dir (str): Output directory for paired dataset
            
        Returns:
            list: Paths to paired images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get original images
        original_paths = [
            os.path.join(original_dir, f) 
            for f in os.listdir(original_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        # Apply corruptions
        corruption_mapping = self.apply_corruptions(
            original_paths, 
            save_corrupted=True
        )
        
        # Create a file with paired paths
        pairs_file = os.path.join(output_dir, "image_pairs.txt")
        with open(pairs_file, 'w') as f:
            for orig_path, corrupted_path in corruption_mapping.items():
                f.write(f"{orig_path},{corrupted_path}\n")
                
        return list(corruption_mapping.items())


# Script to run the corruption module independently
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply corruptions to images')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory with input images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for corrupted images')
    parser.add_argument('--corruption_types', nargs='+', 
                        choices=['noise', 'jpeg', 'blur', 'mask', 'combined'],
                        help='Types of corruptions to apply')
    
    args = parser.parse_args()
    
    corruption_module = ImageCorruption(config_path=args.config)
    
    # Get input images
    image_paths = [
        os.path.join(args.input_dir, f) 
        for f in os.listdir(args.input_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    # Override output directory
    corruption_module.output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Apply corruptions
    corruption_mapping = corruption_module.apply_corruptions(
        image_paths, 
        corruption_types=args.corruption_types
    )
    
    print(f"Successfully corrupted {len(corruption_mapping)} images")
