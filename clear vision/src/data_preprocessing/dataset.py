import os
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import logging
import json
from pathlib import Path
import sys
import traceback

# Configure logging with higher verbosity
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageRestorationDataset(Dataset):
    """Dataset for image restoration with paired corrupted and clean images"""
    
    def __init__(self, pairs_file, transform=None, target_transform=None):
        """
        Initialize the dataset
        
        Args:
            pairs_file (str): Path to file with paired image paths
            transform (callable, optional): Transform to apply to corrupted images
            target_transform (callable, optional): Transform to apply to clean images
        """
        logger.debug(f"Initializing dataset with pairs file: {pairs_file}")
        self.transform = transform
        self.target_transform = target_transform
        
        # Load image pairs
        self.image_pairs = []
        try:
            with open(pairs_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(',')
                        if len(parts) != 2:
                            logger.warning(f"Invalid line in pairs file: {line.strip()}")
                            continue
                        clean_path, corrupted_path = parts
                        
                        # Verify files exist
                        if not os.path.exists(clean_path):
                            logger.warning(f"Clean image not found: {clean_path}")
                            continue
                        if not os.path.exists(corrupted_path):
                            logger.warning(f"Corrupted image not found: {corrupted_path}")
                            continue
                            
                        self.image_pairs.append((clean_path, corrupted_path))
        except Exception as e:
            logger.error(f"Error loading pairs file: {e}")
            traceback.print_exc()
            
        logger.info(f"Loaded {len(self.image_pairs)} valid image pairs")
                    
    def __len__(self):
        """Return the number of image pairs"""
        return len(self.image_pairs)
        
    def __getitem__(self, idx):
        """
        Get a pair of clean and corrupted images
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (corrupted_image, clean_image)
        """
        clean_path, corrupted_path = self.image_pairs[idx]
        
        try:
            # Load images
            clean_image = Image.open(clean_path).convert('RGB')
            corrupted_image = Image.open(corrupted_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                corrupted_image = self.transform(corrupted_image)
            if self.target_transform:
                clean_image = self.target_transform(clean_image)
                
            return corrupted_image, clean_image
        except Exception as e:
            logger.error(f"Error loading image pair [{idx}]: {e}")
            traceback.print_exc()
            # Return a placeholder if loading fails
            placeholder = torch.zeros((3, 256, 256))
            return placeholder, placeholder


class DatasetManager:
    """Manager for creating and handling datasets"""
    
    def __init__(self, config_path="/users/bipan/onedrive/desktop/clear vision/config/config.yaml"):
        """
        Initialize the dataset manager
        
        Args:
            config_path (str): Path to configuration file
        """
        logger.debug(f"Initializing DatasetManager with config: {config_path}")
        
        # Verify config path
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config_content = f.read()
                logger.debug(f"Config content first 100 chars: {config_content[:100]}")
                self.config = yaml.safe_load(config_content)
                
            logger.debug(f"Config keys: {list(self.config.keys())}")
            
            # Check for required sections
            required_sections = ['data', 'project', 'scraper']
            for section in required_sections:
                if section not in self.config:
                    logger.error(f"'{section}' section not found in config file")
                    raise KeyError(f"'{section}' section not found in config file")
                    
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML: {e}")
            raise
            
        # Define paths
        self.data_dir = Path(self.config['scraper']['download_path']).parent
        self.raw_dir = Path(self.config['scraper']['download_path'])
        self.corrupted_dir = self.data_dir / "corrupted"
        self.splits_dir = self.data_dir / "splits"
        
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Raw directory: {self.raw_dir}")
        logger.info(f"Corrupted directory: {self.corrupted_dir}")
        logger.info(f"Splits directory: {self.splits_dir}")
        
        # Create directories
        try:
            os.makedirs(self.splits_dir, exist_ok=True)
            logger.debug(f"Created splits directory: {self.splits_dir}")
        except Exception as e:
            logger.error(f"Error creating directory: {e}")
            traceback.print_exc()
        
        # Set image size
        self.image_size = self.config['data']['image_size']
        logger.debug(f"Image size: {self.image_size}")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
        self.target_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        
    def create_train_val_test_splits(self, pairs_file, random_seed=None):
        """
        Create train, validation, and test splits
        
        Args:
            pairs_file (str): Path to file with paired image paths
            random_seed (int, optional): Random seed for reproducibility
            
        Returns:
            tuple: (train_file, val_file, test_file) paths
        """
        logger.info(f"Creating splits from pairs file: {pairs_file}")
        
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            logger.debug(f"Set random seed: {random_seed}")
            
        # Read all pairs
        try:
            with open(pairs_file, 'r') as f:
                pairs = [line.strip() for line in f if line.strip()]
                
            logger.info(f"Read {len(pairs)} pairs from file")
            
            if not pairs:
                logger.error("No valid pairs found in file")
                raise ValueError("No valid pairs found in file")
                
            # Shuffle pairs
            indices = np.random.permutation(len(pairs))
            
            # Calculate split sizes
            train_size = int(len(pairs) * self.config['data']['train_split'])
            val_size = int(len(pairs) * self.config['data']['val_split'])
            test_size = len(pairs) - train_size - val_size
            
            logger.debug(f"Split sizes: train={train_size}, val={val_size}, test={test_size}")
            
            # Create splits
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            # Create split files
            train_file = os.path.join(self.splits_dir, "train_pairs.txt")
            val_file = os.path.join(self.splits_dir, "val_pairs.txt")
            test_file = os.path.join(self.splits_dir, "test_pairs.txt")
            
            # Write train pairs
            with open(train_file, 'w') as f:
                for idx in train_indices:
                    f.write(f"{pairs[idx]}\n")
                    
            # Write validation pairs
            with open(val_file, 'w') as f:
                for idx in val_indices:
                    f.write(f"{pairs[idx]}\n")
                    
            # Write test pairs
            with open(test_file, 'w') as f:
                for idx in test_indices:
                    f.write(f"{pairs[idx]}\n")
                    
            # Log split information
            logger.info(f"Created dataset splits: train={train_size}, val={val_size}, test={test_size}")
            
            return train_file, val_file, test_file
            
        except Exception as e:
            logger.error(f"Error creating splits: {e}")
            traceback.print_exc()
            raise
    
    def get_dataloaders(self, batch_size=None, num_workers=None, use_existing_splits=True):
        """
        Get train, validation, and test dataloaders
        
        Args:
            batch_size (int, optional): Batch size
            num_workers (int, optional): Number of worker processes
            use_existing_splits (bool): Whether to use existing splits
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        logger.info("Creating dataloaders")
        
        batch_size = batch_size or self.config['data']['batch_size']
        num_workers = num_workers or self.config['data']['num_workers']
        
        logger.debug(f"Batch size: {batch_size}, num_workers: {num_workers}")
        
        # Check if splits already exist
        train_file = os.path.join(self.splits_dir, "train_pairs.txt")
        val_file = os.path.join(self.splits_dir, "val_pairs.txt")
        test_file = os.path.join(self.splits_dir, "test_pairs.txt")
        
        pairs_file = os.path.join(self.data_dir, "image_pairs.txt")
        
        if not use_existing_splits or not all(os.path.exists(f) for f in [train_file, val_file, test_file]):
            logger.info(f"Need to create new splits (use_existing_splits={use_existing_splits})")
            # Create new splits
            if not os.path.exists(pairs_file):
                logger.error(f"Pairs file not found: {pairs_file}")
                raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
                
            train_file, val_file, test_file = self.create_train_val_test_splits(
                pairs_file, 
                random_seed=self.config['project']['seed']
            )
        else:
            logger.info("Using existing splits")
        
        # Create datasets
        try:
            train_dataset = ImageRestorationDataset(
                train_file,
                transform=self.transform,
                target_transform=self.target_transform
            )
            
            val_dataset = ImageRestorationDataset(
                val_file,
                transform=self.transform,
                target_transform=self.target_transform
            )
            
            test_dataset = ImageRestorationDataset(
                test_file,
                transform=self.transform,
                target_transform=self.target_transform
            )
            
            logger.info(f"Created datasets - train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            
            logger.info("Created dataloaders successfully")
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            logger.error(f"Error creating datasets/dataloaders: {e}")
            traceback.print_exc()
            raise
    
    def create_dataset_from_raw_images(self):
        """
        Create a dataset from raw images by applying corruptions
        
        Returns:
            str: Path to the pairs file
        """
        logger.info("Creating dataset from raw images")
        
        try:
            # Instead of directly importing, try relative import with appropriate error handling
            try:
                # Check current directory for debugging
                logger.debug(f"Current working directory: {os.getcwd()}")
                
                # Try to find corruption.py in the same directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                corruption_path = os.path.join(current_dir, "corruption.py")
                
                logger.debug(f"Looking for corruption module at: {corruption_path}")
                if not os.path.exists(corruption_path):
                    logger.warning(f"corruption.py not found at: {corruption_path}")
                
                # Get raw images
                if not os.path.exists(self.raw_dir):
                    logger.error(f"Raw directory not found: {self.raw_dir}")
                    raise FileNotFoundError(f"Raw directory not found: {self.raw_dir}")
                
                raw_image_paths = [
                    str(self.raw_dir / f) 
                    for f in os.listdir(self.raw_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                
                logger.info(f"Found {len(raw_image_paths)} raw images")
                
                if not raw_image_paths:
                    logger.error("No images found in raw directory")
                    raise ValueError("No images found in raw directory")
                    
                # We'll try both relative and absolute import approaches
                try:
                    logger.debug("Attempting to import ImageCorruption...")
                    
                    # Adjust Python path to include parent directory
                    parent_dir = os.path.dirname(current_dir)
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                        
                    from corruption import ImageCorruption
                    logger.debug("Successfully imported ImageCorruption")
                    
                except ImportError as ie:
                    logger.error(f"Import error: {ie}")
                    
                    # Try direct module loading as backup
                    logger.debug("Trying direct module loading...")
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("corruption", corruption_path)
                    corruption_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(corruption_module)
                    ImageCorruption = corruption_module.ImageCorruption
                    logger.debug("Successfully loaded ImageCorruption via spec")
                
                # Create corruption module
                logger.debug("Creating ImageCorruption instance...")
                corruption_instance = ImageCorruption(config=self.config)
                
                # Apply corruptions
                logger.info("Applying corruptions to images...")
                corruption_mapping = corruption_instance.apply_corruptions(
                    raw_image_paths, 
                    save_corrupted=True
                )
                
                # Create pairs file
                pairs_file = os.path.join(self.data_dir, "image_pairs.txt")
                with open(pairs_file, 'w') as f:
                    for orig_path, corrupted_path in corruption_mapping.items():
                        f.write(f"{orig_path},{corrupted_path}\n")
                        
                logger.info(f"Created dataset with {len(corruption_mapping)} image pairs")
                
                return pairs_file
                
            except Exception as inner_e:
                logger.error(f"Error in corruption handling: {inner_e}")
                traceback.print_exc()
                raise
                
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            traceback.print_exc()
            raise
        
    def get_dataset_stats(self):
        """
        Get dataset statistics
        
        Returns:
            dict: Dataset statistics
        """
        logger.info("Getting dataset statistics")
        
        try:
            # Create directories if they don't exist
            os.makedirs(self.raw_dir, exist_ok=True)
            os.makedirs(self.corrupted_dir, exist_ok=True)
            
            stats = {
                "num_raw_images": len([f for f in os.listdir(self.raw_dir) if os.path.isfile(os.path.join(self.raw_dir, f))]),
                "num_corrupted_images": len([f for f in os.listdir(self.corrupted_dir) if os.path.isfile(os.path.join(self.corrupted_dir, f))]),
                "splits": {}
            }
            
            # Get split stats
            for split in ["train", "val", "test"]:
                split_file = os.path.join(self.splits_dir, f"{split}_pairs.txt")
                if os.path.exists(split_file):
                    with open(split_file, 'r') as f:
                        stats["splits"][split] = len(f.readlines())
                        
            logger.info(f"Dataset stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            traceback.print_exc()
            return {"error": str(e)}


# Script to create and manage datasets
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create and manage datasets')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--create_dataset', action='store_true',
                        help='Create dataset from raw images')
    parser.add_argument('--create_splits', action='store_true',
                        help='Create train/val/test splits')
    parser.add_argument('--stats', action='store_true',
                        help='Show dataset statistics')
    parser.add_argument('--test_loader', action='store_true',
                        help='Test dataloader creation')
    
    args = parser.parse_args()
    
    # Print arguments
    print(f"Arguments: {args}")
    
    try:
        # Create dataset manager
        dataset_manager = DatasetManager(config_path=args.config)
        
        if args.create_dataset:
            # Create dataset from raw images
            pairs_file = dataset_manager.create_dataset_from_raw_images()
            print(f"Created dataset: {pairs_file}")
            
        if args.create_splits:
            # Create splits
            pairs_file = os.path.join(dataset_manager.data_dir, "image_pairs.txt")
            if not os.path.exists(pairs_file):
                print(f"Pairs file not found: {pairs_file}")
            else:
                train_file, val_file, test_file = dataset_manager.create_train_val_test_splits(
                    pairs_file,
                    random_seed=dataset_manager.config['project']['seed']
                )
                print(f"Created splits: {train_file}, {val_file}, {test_file}")
                
        if args.stats:
            # Show statistics
            stats = dataset_manager.get_dataset_stats()
            print(json.dumps(stats, indent=2))
            
        if args.test_loader:
            # Test dataloader creation
            train_loader, val_loader, test_loader = dataset_manager.get_dataloaders()
            print(f"Dataloaders created - train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}")
            
            # Test a batch
            for corrupted_batch, clean_batch in train_loader:
                print(f"Batch shapes - corrupted: {corrupted_batch.shape}, clean: {clean_batch.shape}")
                break
                
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()