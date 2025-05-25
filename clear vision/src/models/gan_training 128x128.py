# -*- coding: utf-8 -*-
"""
Improved GAN Training Code with fixes for common issues
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
from gan_model import UNetGenerator, PatchDiscriminator
from torchvision import transforms
from pathlib import Path

class ImagePairDataset(Dataset):
    def __init__(self, pair_file, corrupted_root, raw_root, image_size=(256, 256)):
        # Ensure image_size is a tuple of two ints
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        self.pairs = []
        with open(pair_file, "r") as f:
            for line in f:
                raw_path, corrupted_path = line.strip().split(',')
                raw_path = os.path.normpath(raw_path.strip())
                corrupted_path = os.path.normpath(corrupted_path.strip())
                self.pairs.append((raw_path, corrupted_path))

        self.corrupted_root = corrupted_root
        self.raw_root = raw_root

        # Improved transform pipeline with normalization
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        raw_rel, corrupted_rel = self.pairs[idx]

        # Only keep the filename (strip full/relative paths)
        raw_filename = os.path.basename(raw_rel)
        corrupted_filename = os.path.basename(corrupted_rel)

        raw_path = Path(self.raw_root) / raw_filename
        corrupted_path = Path(self.corrupted_root) / corrupted_filename

        try:
            raw_img = Image.open(raw_path).convert("RGB")
            corrupted_img = Image.open(corrupted_path).convert("RGB")
        except Exception as e:
            print(f"Error loading images: {raw_path}, {corrupted_path}")
            raise e

        # Apply transforms (resize + to tensor + normalize)
        raw_img = self.transform(raw_img)
        corrupted_img = self.transform(corrupted_img)

        return corrupted_img, raw_img


def train():
    with open("config/gan_config.yaml") as f:
        config = yaml.safe_load(f)

    # Ensure image_size from config is a tuple
    image_size = config.get("image_size", (256, 256))
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)

    # Initialize weights properly
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    G.apply(weights_init)
    D.apply(weights_init)

    # Use MSE loss for more stable training
    criterion_GAN = torch.nn.MSELoss()  # Changed from BCELoss
    criterion_L1 = torch.nn.L1Loss()

    # Different learning rates for G and D
    lr_g = config.get("learning_rate_g", config.get("learning_rate", 0.0002))
    lr_d = config.get("learning_rate_d", config.get("learning_rate", 0.0002))
    
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(config.get("beta1", 0.5), 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(config.get("beta1", 0.5), 0.999))

    # Add learning rate schedulers
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.5)

    train_dataset = ImagePairDataset(
        config["train_pairs_path"],
        config["corrupted_image_root"],
        config["raw_image_root"],
        image_size=image_size
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=2,  # Add parallel data loading
        pin_memory=True if torch.cuda.is_available() else False
    )

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)

    # Training loop with improvements
    for epoch in range(config["epochs"]):
        G.train()
        D.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            # Create simple labels for discriminator
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Update Discriminator more frequently than Generator
            for _ in range(1):  # You can increase this to 2 or 3 if D is too weak
                D.zero_grad()
                
                # Real pairs
                pred_real = D(x, y)
                loss_D_real = criterion_GAN(pred_real, real_labels)
                
                # Fake pairs
                with torch.no_grad():
                    fake_img = G(x)
                pred_fake = D(x, fake_img)
                loss_D_fake = criterion_GAN(pred_fake, fake_labels)
                
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizer_D.step()

            # Update Generator
            G.zero_grad()
            fake_img = G(x)
            pred_fake = D(x, fake_img)
            
            # GAN loss
            loss_G_GAN = criterion_GAN(pred_fake, real_labels)
            
            # L1 loss
            lambda_L1 = config.get("lambda_L1", 100)  # Default value
            loss_G_L1 = criterion_L1(fake_img, y) * lambda_L1
            
            # Perceptual loss (optional - requires additional implementation)
            # loss_G_perceptual = perceptual_loss(fake_img, y) * lambda_perceptual
            
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{config['epochs']}], Step [{i}], "
                      f"D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}, "
                      f"G_GAN: {loss_G_GAN.item():.4f}, G_L1: {loss_G_L1.item():.4f}")
                
                # Save images with proper denormalization
                with torch.no_grad():
                    fake_img_denorm = fake_img * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
                    real_img_denorm = y * 0.5 + 0.5
                    input_img_denorm = x * 0.5 + 0.5
                    
                    # Save comparison
                    comparison = torch.cat([input_img_denorm[:4], fake_img_denorm[:4], real_img_denorm[:4]], dim=0)
                    save_image(comparison, f"{config['results_dir']}/comparison_{epoch}_{i}.png", nrow=4)

        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        print(f"Epoch [{epoch}] completed - Avg G Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")

        # Save checkpoints
        if epoch % 10 == 0 or epoch == config["epochs"] - 1:
            # Save comprehensive checkpoint
            torch.save({
                'generator_state_dict': G.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'epoch': epoch,
            }, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch}.pth")
            
            # Also save generator only (like your original code)
            torch.save(G.state_dict(), f"{config['checkpoint_dir']}/G_epoch{epoch}.pth")
            
            # Save discriminator separately too
            torch.save(D.state_dict(), f"{config['checkpoint_dir']}/D_epoch{epoch}.pth")


if __name__ == "__main__":
    train()