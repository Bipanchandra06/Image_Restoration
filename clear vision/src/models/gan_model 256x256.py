# -*- coding: utf-8 -*-
"""
Created on Fri May 23 22:25:20 2025
@author: bipan
"""
import torch
import torch.nn as nn
from torchvision import transforms

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder layers - added enc0 for 256->128
        self.enc0 = self._make_encoder_block(in_channels, 64, norm=False)  # 256->128
        self.enc1 = self._make_encoder_block(64, 64, norm=False)  # 128->64
        self.enc2 = self._make_encoder_block(64, 128)   # 64->32
        self.enc3 = self._make_encoder_block(128, 256)  # 32->16
        self.enc4 = self._make_encoder_block(256, 512)  # 16->8
        self.enc5 = self._make_encoder_block(512, 512)  # 8->4
        self.enc6 = self._make_encoder_block(512, 512)  # 4->2
        self.enc7 = self._make_encoder_block(512, 512)  # 2->1
        
        
        # Decoder layers - added dec0 for final upsampling to 256
        self.dec7 = self._make_decoder_block(512, 512, dropout=True)      # from bottleneck (enc7)
        self.dec6 = self._make_decoder_block(1024, 512, dropout=True)     # 512 + 512 skip from enc6
        self.dec5 = self._make_decoder_block(1024, 512, dropout=True)     # 512 + 512 skip from enc5
        self.dec4 = self._make_decoder_block(1024, 256)                   # 512 + 512 skip from enc4
        self.dec3 = self._make_decoder_block(512, 128)                    # 256 + 256 skip from enc3
        self.dec2 = self._make_decoder_block(256, 64)                     # 128 + 128 skip from enc2
        self.dec1 = self._make_decoder_block(128, 64)                     # 64 + 64 skip from enc1
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),              # 64 + 64 skip from enc0
            nn.Tanh()
        )
        
        self.apply(init_weights)
    
    def _make_encoder_block(self, in_channels, out_channels, norm=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _make_decoder_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder with skip connection storage
        e0 = self.enc0(x)      # 128 x 128 x 64
        e1 = self.enc1(e0)     # 64 x 64 x 64
        e2 = self.enc2(e1)     # 32 x 32 x 128
        e3 = self.enc3(e2)     # 16 x 16 x 256
        e4 = self.enc4(e3)     # 8 x 8 x 512
        e5 = self.enc5(e4)     # 4 x 4 x 512
        e6 = self.enc6(e5)     # 2 x 2 x 512
        e7 = self.enc7(e6)     # 1 x 1 x 512 (this is our bottleneck)
        
        # Decoder with skip connections
        d7 = self.dec7(e7)                          # 2 x 2 x 512
        d6 = self.dec6(torch.cat([d7, e6], dim=1)) # 4 x 4 x 512 (512+512 -> 512)
        d5 = self.dec5(torch.cat([d6, e5], dim=1)) # 8 x 8 x 512 (512+512 -> 512)
        d4 = self.dec4(torch.cat([d5, e4], dim=1)) # 16 x 16 x 256 (512+512 -> 256)
        d3 = self.dec3(torch.cat([d4, e3], dim=1)) # 32 x 32 x 128 (256+256 -> 128)
        d2 = self.dec2(torch.cat([d3, e2], dim=1)) # 64 x 64 x 64 (128+128 -> 64)
        d1 = self.dec1(torch.cat([d2, e1], dim=1)) # 128 x 128 x 64 (64+64 -> 64)
        d0 = self.dec0(torch.cat([d1, e0], dim=1)) # 256 x 256 x 3 (64+64 -> 3)
        
        return d0

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        
        # Use spectral normalization for more stable training
        def spectral_norm(layer):
            return nn.utils.spectral_norm(layer)
        
        self.model = nn.Sequential(
            # First layer without batch norm
            spectral_norm(nn.Conv2d(in_channels, 64, 4, 2, 1)),  # 256->128
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second layer
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),          # 128->64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third layer
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),         # 64->32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fourth layer
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),         # 32->16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fifth layer - reduced stride to prevent too small feature maps
            spectral_norm(nn.Conv2d(512, 512, 4, 2, 1)),         # 16->8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer - use 3x3 kernel to be safe with small feature maps
            nn.Conv2d(512, 1, 3, 1, 1)                           # 8->8
        )
        
        # Global average pooling to get single output
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.apply(init_weights)
    
    def forward(self, x, y):
        combined = torch.cat([x, y], dim=1)
        features = self.model(combined)
        # Apply global pooling and flatten to get [batch_size, 1]
        output = self.global_pool(features).view(features.size(0), -1)
        return output

# Alternative simpler U-Net if the above is too complex
class SimpleUNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder - added enc0 for 256->128
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),    # 256->128
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(64, 64, 4, 2, 1),             # 128->64
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),            # 64->32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),           # 32->16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),           # 16->8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder - added dec0 for final upsampling to 256
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8->16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),  # 16->32 (256 + 256 from skip)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),   # 32->64 (128 + 128 from skip)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64->128 (64 + 64 from skip)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),  # 128->256 (64 + 64 from skip)
            nn.Tanh()
        )
        
        self.apply(init_weights)
    
    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)    # 128x128x64
        e1 = self.enc1(e0)   # 64x64x64
        e2 = self.enc2(e1)   # 32x32x128
        e3 = self.enc3(e2)   # 16x16x256
        e4 = self.enc4(e3)   # 8x8x512
        
        # Decoder with skip connections
        d4 = self.dec4(e4)                          # 16x16x256
        d3 = self.dec3(torch.cat([d4, e3], dim=1)) # 32x32x128
        d2 = self.dec2(torch.cat([d3, e2], dim=1)) # 64x64x64
        d1 = self.dec1(torch.cat([d2, e1], dim=1)) # 128x128x64
        d0 = self.dec0(torch.cat([d1, e0], dim=1)) # 256x256x3
        
        return d0

# Additional utility function to check model compatibility with input size
def test_model_shapes(input_size=(256, 256)):
    """Test if the models work with given input size"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test generator
    gen = SimpleUNetGenerator().to(device)  # Use simpler version for testing
    test_input = torch.randn(1, 3, *input_size).to(device)
    
    try:
        gen_output = gen(test_input)
        print(f"Generator works! Input: {test_input.shape} -> Output: {gen_output.shape}")
    except Exception as e:
        print(f"Generator failed: {e}")
        return False
    
    # Test discriminator
    disc = PatchDiscriminator().to(device)
    try:
        disc_output = disc(test_input, gen_output.detach())
        print(f"Discriminator works! Output shape: {disc_output.shape}")
        return True
    except Exception as e:
        print(f"Discriminator failed: {e}")
        return False

if __name__ == "__main__":
    # Test the models
    print("Testing models with 256x256 input:")
    test_model_shapes()