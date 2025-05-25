# -*- coding: utf-8 -*-
"""
Model Evaluation Script - Enhanced with LPIPS
Created on Fri May 23 22:27:10 2025
@author: bipan
"""

import os
import torch
import yaml
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from gan_model import UNetGenerator, SimpleUNetGenerator  # Import both generators
from gan_training import ImagePairDataset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.transforms.functional import to_pil_image
import argparse
from PIL import Image
import lpips
from datetime import datetime

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

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for metric calculation"""
    # Ensure tensor is in [0, 1] range
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = torch.clamp((tensor + 1) / 2, 0, 1)  # Convert [-1,1] to [0,1]
    
    # Convert to numpy
    numpy_array = tensor.detach().cpu().numpy()
    
    # Handle different tensor formats
    if len(numpy_array.shape) == 4:  # Batch dimension
        numpy_array = numpy_array[0]
    if len(numpy_array.shape) == 3:  # Channel first to channel last
        numpy_array = np.transpose(numpy_array, (1, 2, 0))
    
    return numpy_array

def prepare_tensor_for_lpips(tensor):
    """Prepare tensor for LPIPS calculation (expects [-1, 1] range)"""
    # LPIPS expects input in [-1, 1] range
    if tensor.min() >= 0:  # If tensor is in [0, 1] range
        tensor = tensor * 2.0 - 1.0  # Convert to [-1, 1]
    return tensor

def save_metrics_to_file(metrics_dict, output_path):
    """Save evaluation metrics to a text file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Path: {metrics_dict.get('model_path', 'N/A')}\n")
        f.write(f"Model Type: {metrics_dict.get('model_type', 'N/A')}\n")
        f.write(f"Samples Evaluated: {metrics_dict.get('num_samples', 0)}\n")
        f.write("-" * 60 + "\n\n")
        
        # SSIM Results
        if 'ssim_scores' in metrics_dict and metrics_dict['ssim_scores']:
            ssim_scores = metrics_dict['ssim_scores']
            f.write("STRUCTURAL SIMILARITY INDEX (SSIM)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean SSIM: {np.mean(ssim_scores):.6f}\n")
            f.write(f"Std SSIM:  {np.std(ssim_scores):.6f}\n")
            f.write(f"Median SSIM: {np.median(ssim_scores):.6f}\n")
        
        
        # PSNR Results
        if 'psnr_scores' in metrics_dict and metrics_dict['psnr_scores']:
            psnr_scores = metrics_dict['psnr_scores']
            f.write("PEAK SIGNAL-TO-NOISE RATIO (PSNR)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean PSNR: {np.mean(psnr_scores):.4f} dB\n")
            f.write(f"Std PSNR:  {np.std(psnr_scores):.4f} dB\n")
            f.write(f"Median PSNR: {np.median(psnr_scores):.4f} dB\n")
          
        
        # LPIPS Results
        if 'lpips_scores' in metrics_dict and metrics_dict['lpips_scores']:
            lpips_scores = metrics_dict['lpips_scores']
            f.write("LEARNED PERCEPTUAL IMAGE PATCH SIMILARITY (LPIPS)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean LPIPS: {np.mean(lpips_scores):.6f}\n")
            f.write(f"Std LPIPS:  {np.std(lpips_scores):.6f}\n")
            f.write(f"Median LPIPS: {np.median(lpips_scores):.6f}\n")
        
        
        # Individual sample scores (optional detailed output)
        if metrics_dict.get('save_detailed', False):
            f.write("DETAILED SAMPLE SCORES\n")
            f.write("-" * 40 + "\n")
            f.write("Sample\tSSIM\t\tPSNR\t\tLPIPS\n")
            f.write("-" * 40 + "\n")
            
            ssim_scores = metrics_dict.get('ssim_scores', [])
            psnr_scores = metrics_dict.get('psnr_scores', [])
            lpips_scores = metrics_dict.get('lpips_scores', [])
            
            for i in range(len(ssim_scores)):
                ssim_val = ssim_scores[i] if i < len(ssim_scores) else 'N/A'
                psnr_val = psnr_scores[i] if i < len(psnr_scores) else 'N/A'
                lpips_val = lpips_scores[i] if i < len(lpips_scores) else 'N/A'
                
                f.write(f"{i+1:04d}\t{ssim_val:.6f}\t{psnr_val:.4f}\t\t{lpips_val:.6f}\n")
        
        f.write("\n" + "="*60 + "\n")

def evaluate_model(config_path="config/gan_config.yaml", 
                  model_path=None, 
                  model_type="unet",
                  output_dir="results/eval_outputs",
                  save_images=True,
                  lpips_net='alex',
                  save_detailed=False):
    """
    Evaluate the trained model with SSIM, PSNR, and LPIPS metrics
    
    Args:
        lpips_net: LPIPS network to use ('alex', 'vgg', 'squeeze')
        save_detailed: Whether to save detailed per-sample scores
    """
    # Load configuration
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Using default configuration...")
        config = {
            "val_pairs_path": "data/val_pairs.txt",
            "corrupted_image_root": "data/corrupted",
            "raw_image_root": "data/raw",
            "image_size": 128,
            "checkpoint_dir": "checkpoints"
        }

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize LPIPS metric
    try:
        print(f"Initializing LPIPS with {lpips_net} network...")
        lpips_fn = lpips.LPIPS(net=lpips_net).to(device)
        print("✓ LPIPS initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize LPIPS: {e}")
        print("LPIPS evaluation will be skipped. Install lpips package: pip install lpips")
        lpips_fn = None

    # Load model
    if model_path is None:
        # Try to find the latest checkpoint based on your saving convention
        checkpoint_dir = config["checkpoint_dir"]
        
        # Look for the highest epoch number
        epoch_files = []
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.startswith("G_epoch") and file.endswith(".pth"):
                    try:
                        epoch_num = int(file.replace("G_epoch", "").replace(".pth", ""))
                        epoch_files.append((epoch_num, os.path.join(checkpoint_dir, file)))
                    except ValueError:
                        continue
        
        if epoch_files:
            # Use the highest epoch number
            epoch_files.sort(key=lambda x: x[0], reverse=True)
            model_path = epoch_files[0][1]
            print(f"Auto-detected latest model: {model_path}")
        else:
            # Fallback to common names
            alternatives = [
                os.path.join(checkpoint_dir, "G_epoch100.pth"),
                os.path.join(checkpoint_dir, "G_epoch50.pth"),
                os.path.join(checkpoint_dir, "checkpoint_epoch_100.pth"),
                os.path.join(checkpoint_dir, "generator_final.pth")
            ]
            for alt_path in alternatives:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break

    model = load_model(model_path, model_type, device)

    # Create dataset and dataloader
    try:
        dataset = ImagePairDataset(
            config["val_pairs_path"],
            config["corrupted_image_root"],
            config["raw_image_root"],
            config["image_size"]
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f"✓ Loaded {len(dataset)} validation samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Create output directory
    if save_images:
        os.makedirs(output_dir, exist_ok=True)

    # Evaluation metrics storage
    ssim_scores, psnr_scores, lpips_scores = [], [], []

    print("\n--- Starting Evaluation ---")
    
    with torch.no_grad():
        for idx, (corrupted, target) in enumerate(loader):
            corrupted = corrupted.to(device)
            target = target.to(device)
            
            # Generate output
            generated = model(corrupted)
            
            # Save images if requested
            if save_images:
                # Save generated image
                save_image(
                    generated * 0.5 + 0.5,  # Convert [-1,1] to [0,1]
                    os.path.join(output_dir, f"generated_{idx:04d}.png")
                )
                
                # Save input and target for comparison
                save_image(
                    corrupted * 0.5 + 0.5,
                    os.path.join(output_dir, f"input_{idx:04d}.png")
                )
                save_image(
                    target * 0.5 + 0.5,
                    os.path.join(output_dir, f"target_{idx:04d}.png")
                )

            # Convert tensors to numpy for SSIM/PSNR calculation
            try:
                gen_np = tensor_to_numpy(generated)
                target_np = tensor_to_numpy(target)
                
                # Ensure both arrays are in [0,1] range and same shape
                gen_np = np.clip(gen_np, 0, 1)
                target_np = np.clip(target_np, 0, 1)
                
                # Calculate SSIM and PSNR
                if gen_np.shape == target_np.shape:
                    ssim_score = ssim(target_np, gen_np, 
                                    channel_axis=-1 if len(gen_np.shape) == 3 else None,
                                    data_range=1.0)
                    psnr_score = psnr(target_np, gen_np, data_range=1.0)
                    
                    ssim_scores.append(ssim_score)
                    psnr_scores.append(psnr_score)
                else:
                    print(f"Shape mismatch at sample {idx}: {gen_np.shape} vs {target_np.shape}")
                    continue
                    
            except Exception as e:
                print(f"Error calculating SSIM/PSNR for sample {idx}: {e}")
                continue
            
            # Calculate LPIPS if available
            if lpips_fn is not None:
                try:
                    # Prepare tensors for LPIPS (expects [-1, 1] range)
                    gen_lpips = prepare_tensor_for_lpips(generated)
                    target_lpips = prepare_tensor_for_lpips(target)
                    
                    # Calculate LPIPS
                    lpips_score = lpips_fn(gen_lpips, target_lpips).item()
                    lpips_scores.append(lpips_score)
                    
                except Exception as e:
                    print(f"Error calculating LPIPS for sample {idx}: {e}")
                    continue
                
            # Progress indicator
            if (idx + 1) % 10 == 0:
                metrics_str = f"SSIM: {np.mean(ssim_scores):.4f}" if ssim_scores else ""
                if psnr_scores:
                    metrics_str += f", PSNR: {np.mean(psnr_scores):.2f}"
                if lpips_scores:
                    metrics_str += f", LPIPS: {np.mean(lpips_scores):.4f}"
                print(f"Processed {idx + 1}/{len(loader)} samples... ({metrics_str})")

    # Prepare metrics dictionary
    metrics_dict = {
        'model_path': model_path,
        'model_type': model_type,
        'num_samples': len(ssim_scores),
        'ssim_scores': ssim_scores,
        'psnr_scores': psnr_scores,
        'lpips_scores': lpips_scores,
        'save_detailed': save_detailed
    }

    # Print and save results
    if ssim_scores and psnr_scores:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Samples evaluated: {len(ssim_scores)}")
        print(f"Average SSIM: {np.mean(ssim_scores):.6f} ± {np.std(ssim_scores):.6f}")
        print(f"Average PSNR: {np.mean(psnr_scores):.4f} ± {np.std(psnr_scores):.4f} dB")
        if lpips_scores:
            print(f"Average LPIPS: {np.mean(lpips_scores):.6f} ± {np.std(lpips_scores):.6f}")
        print(f"Median SSIM: {np.median(ssim_scores):.6f}")
        print(f"Median PSNR: {np.median(psnr_scores):.4f} dB")
        if lpips_scores:
            print(f"Median LPIPS: {np.median(lpips_scores):.6f}")
        print("="*60)
        
        # Save metrics to file
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(results_dir, f"evaluation_metrics_{timestamp}.txt")
        
        save_metrics_to_file(metrics_dict, metrics_file)
        print(f"✓ Metrics saved to: {metrics_file}")
        
        if save_images:
            print(f"✓ Images saved to: {output_dir}")
    else:
        print("❌ No metrics could be calculated. Check your data and model.")

def test_single_image(model_path, image_path, model_type="unet", output_path="test_output.png"):
    """
    Test the model on a single image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(model_path, model_type, device)
    
    # Load and preprocess image
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Convert to [-1,1]
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate output
    with torch.no_grad():
        output = model(input_tensor)
    
    # Save result
    save_image(output * 0.5 + 0.5, output_path)
    print(f"✓ Result saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GAN model with SSIM, PSNR, and LPIPS")
    parser.add_argument("--config", default="config/gan_config.yaml", help="Config file path")
    parser.add_argument("--model", help="Model checkpoint path")
    parser.add_argument("--model_type", choices=["unet", "simple_unet"], default="unet", help="Model architecture type")
    parser.add_argument("--output_dir", default="results/eval_outputs", help="Output directory for images")
    parser.add_argument("--test_image", help="Path to single test image")
    parser.add_argument("--no_save", action="store_true", help="Don't save output images")
    parser.add_argument("--lpips_net", choices=["alex", "vgg", "squeeze"], default="alex", help="LPIPS network backbone")
    parser.add_argument("--detailed", action="store_true", help="Save detailed per-sample scores")
    
    args = parser.parse_args()
    
    if args.test_image:
        # Test single image
        if args.model is None:
            print("Please provide --model path for single image testing")
        else:
            test_single_image(args.model, args.test_image, args.model_type)
    else:
        # Full evaluation
        evaluate_model(
            config_path=args.config,
            model_path=args.model,
            model_type=args.model_type,
            output_dir=args.output_dir,
            save_images=not args.no_save,
            lpips_net=args.lpips_net,
            save_detailed=args.detailed
        )