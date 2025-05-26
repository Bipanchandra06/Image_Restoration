# Image Restoration using Generative Adversarial Networks (GAN)

## Problem Statement

Image degradation is common in real-world visual data due to noise, compression artifacts, occlusion, or partial corruption. This project focuses on building an image restoration system that can recover and reconstruct high-quality images from such degraded inputs using deep generative models.

The goal is to design a model that takes a corrupted image and restores it to a clean, high-fidelity version using Generative Adversarial Networks (GANs).

## Project Overview

This project implements an image restoration system using GANs to restore corrupted images with various types of degradation including blur, noise, and missing parts. The system includes a web interface for easy upload and restoration preview.

## Dataset

- **Size**: 210 images
- **Source**: Generated using Selenium-based web scraper
- **Download**: [Dataset Google Drive Link](https://drive.google.com/drive/folders/1SCC8B1wJoEqwe0sa9dULJ9HZdEj23KHg?usp=sharing)
- **Corruption Types**:
  - **Blur**: Gaussian and motion blur effects
  - **Noise**: Salt-and-pepper, Gaussian noise
  - **Masking**: Random patches/regions removed from images

## Model Architecture

**Generative Adversarial Network (GAN)**
- **Generator**: U-Net architecture for image-to-image translation
- **Discriminator**: Convolutional neural network for distinguishing real vs restored images
- **Loss Function**: Combination of adversarial loss and pixel-wise reconstruction loss

## Key Features

- ✅ **Data Collection**: Selenium-based web scraper for image dataset creation
- ✅ **Data Preprocessing**: Custom corruption algorithms for realistic degradation simulation
- ✅ **GAN Implementation**: U-Net generator with adversarial training
- ✅ **Evaluation Metrics**: PSNR, SSIM, LPIPS for comprehensive quality assessment
- ✅ **Web Interface**: Streamlit-based application for easy image upload and restoration
- ✅ **Inference Optimization**: Efficient model loading and processing pipeline
- ✅ **Web Deployment**:The app is available at [imagerestoration](https://imagerestoration-bipanchandra.streamlit.app)
## Tech Stack

### Machine Learning
- **PyTorch** - Deep learning framework
- **torchvision** - Computer vision utilities

### Web Interface
- **Streamlit** - Web application framework
- **Pillow (PIL)** - Image processing

### Data Collection
- **Selenium** - Web scraping for dataset creation

### Additional Tools
- **gdown** - Google Drive integration for model hosting

## Quick Start

### Option 1: Use Pre-trained Model
1. Clone the repository:
```bash
git clone <repository-url>
cd clear vision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download resources (optional - auto-downloads on first run):
   - **Dataset**: [Download from Google Drive](https://drive.google.com/drive/folders/1SCC8B1wJoEqwe0sa9dULJ9HZdEj23KHg?usp=sharing)
   - **Model**: [Download checkpoint file](https://drive.google.com/drive/folders/1JjQP6_1WVpcuku5EfLOjzzt67wim5YLV?usp=drive_link)

4. Run the Streamlit application:
```bash
streamlit run app-online.py
```

### Option 2: Train Your Own Model
1. Download the dataset from the Google Drive link above
2. Extract to `data/` directory
3. Run training script (add your training code)
4. Use the generated checkpoint with the web interface

## Usage

### Web Interface
1. Launch the Streamlit app
2. Upload a corrupted image (JPG, JPEG, PNG)
3. View the original and restored images side by side  
4. Download the restored image

## Evaluation Metrics

The model is evaluated using standard image quality metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality
- **SSIM (Structural Similarity Index)**: Evaluates structural similarity
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Assesses perceptual quality

## Results

[Add your specific results here, for example:]
- Median PSNR: 25.8488 dB
- Median SSIM: 0.8097
- Median LPIPS: 0.1703
- Inference latency: 190 ms per image

## File Structure

```
image-restoration-gan/
├── app.py                 # Streamlit web interface
├── models/
│   └── gan_model.py      # GAN model definitions
    └──   data/
│         ├── raw/         # Original clean images
│         └── corrupted/        # Corrupted images for training
          └── splits/
├── checkpoints/          # Trained model weights
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Model Checkpoints

The trained model is hosted on Google Drive and automatically downloaded when running the application.

**Download Links:**
- **Trained Model**: [Model Checkpoint (.pth file)](https://drive.google.com/your-model-checkpoint-link-here)
- **Dataset**: [Complete Dataset](https://drive.google.com/your-dataset-link-here)

The checkpoint includes:
- Generator state dictionary
- Training epoch information
- Model configuration parameters

**Manual Download Instructions:**
1. Download the model checkpoint from the Google Drive link
2. Place it in the `checkpoints/` directory
3. Update the `gdrive_file_id` in the app configuration if needed

## Corruption Algorithm

The project includes custom corruption algorithms that simulate realistic degradation:

1. **Blur Corruption**: Applies Gaussian and motion blur with varying intensities
2. **Noise Injection**: Adds salt-and-pepper and Gaussian noise
3. **Missing Parts**: Randomly removes patches to simulate occlusion/damage

## Future Improvements

- [ ] Implement additional model architectures (VAE, Diffusion Models)
- [ ] Expand dataset with more diverse corruption types
- [ ] Add mobile interface support
- [ ] Implement batch processing capabilities
- [ ] Add real-time video restoration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## Acknowledgments

- PyTorch team for deep learning framework
- Streamlit for web interface capabilities
- Research papers and resources that inspired the GAN architecture

## Downloads & Resources

### Google Drive Links
- 📊 **Dataset (210 images)**: [Download Complete Dataset](https://drive.google.com/your-dataset-link-here)
- 🤖 **Trained Model**: [Download Model Checkpoint](https://drive.google.com/your-model-checkpoint-link-here)
- 📋 **Project Files**: All resources are available through the links above

### File Details
- **Dataset**: Contains original and corrupted image pairs
- **Model**: `checkpoint_epoch_99.pth` - Final trained GAN model
- **Size**: Dataset (25 MB), Model (540 MB)
