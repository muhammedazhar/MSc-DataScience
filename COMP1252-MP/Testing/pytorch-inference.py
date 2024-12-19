import os
import glob
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from skimage import exposure
import logging
from typing import Tuple, Optional

# Local imports
from Solutions.helper import *
# Configure logging
setup_logging()

class DeforestationDataset(Dataset):
    """Dataset for loading temporal image pairs for deforestation detection."""
    
    def __init__(self, 
                 dataset_dir: str,
                 plot_range: range,
                 img_size: Tuple[int, int] = (224, 224),
                 cloud_threshold: float = 0.7):
        """
        Initialize the dataset.
        
        Args:
            dataset_dir: Root directory containing plot folders
            plot_range: Range of plot numbers to process
            img_size: Target image size (height, width)
            cloud_threshold: Maximum allowed cloud coverage
        """
        self.dataset_dir = Path(dataset_dir)
        self.plot_range = plot_range
        self.img_size = img_size
        self.cloud_threshold = cloud_threshold
        
        # Get valid temporal pairs
        self.pairs = self._get_temporal_pairs()
        
    def _get_temporal_pairs(self):
        """Get all valid temporal pairs across plots."""
        pairs = []
        
        for plot_id in self.plot_range:
            plot_name = f"PLOT-{plot_id:05d}"
            plot_dir = self.dataset_dir / plot_name
            
            if not plot_dir.exists():
                continue
                
            pre_dir = plot_dir / "Pre-event"
            post_dir = plot_dir / "Post-event"
            mask_dir = plot_dir / "Masks"
            
            if not all(d.exists() for d in [pre_dir, post_dir, mask_dir]):
                continue
                
            # Get temporal pairs for this plot
            plot_pairs = self._get_plot_temporal_pairs(pre_dir, post_dir, mask_dir)
            pairs.extend(plot_pairs)
            
        return pairs
    
    def _get_plot_temporal_pairs(self, pre_dir, post_dir, mask_dir):
        """Get valid temporal pairs for a single plot."""
        pairs = []
        
        pre_files = sorted(pre_dir.glob("*.npy"))
        post_files = sorted(post_dir.glob("*.npy"))
        
        for pre_file in pre_files:
            pre_date = self._extract_date(pre_file)
            pre_dt = np.datetime64(pre_date)
            
            # Find matching post-event file
            for post_file in post_files:
                post_date = self._extract_date(post_file)
                post_dt = np.datetime64(post_date)
                
                delta = (post_dt - pre_dt).astype(int)
                if 5 <= delta <= 30:
                    mask_file = mask_dir / f"{post_date}.tif"
                    if mask_file.exists():
                        pairs.append((pre_file, post_file, mask_file))
                        break
                        
        return pairs
    
    @staticmethod
    def _extract_date(filepath):
        """Extract date from filename (YYYYMMDDTHHMMSS)."""
        return Path(filepath).stem.split('T')[0]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pre_file, post_file, mask_file = self.pairs[idx]
        
        # Load and preprocess images
        pre_img = np.load(pre_file)
        post_img = np.load(post_file)
        
        # Scale images
        pre_img = pre_img.astype(np.float32) / 10000.0
        post_img = post_img.astype(np.float32) / 10000.0
        
        # Apply histogram matching
        post_matched = self._histogram_match(post_img, pre_img)
        
        # Compute difference
        diff_img = post_matched - pre_img
        
        # Stack channels
        combined = np.concatenate([pre_img, post_matched, diff_img], axis=-1)
        
        # Convert to tensor
        combined = torch.from_numpy(combined).float()
        
        # Move channels first
        combined = combined.permute(2, 0, 1)
        
        return combined, pre_file.parent.parent.name
    
    @staticmethod
    def _histogram_match(source, reference):
        """Apply histogram matching channel-wise."""
        matched = np.zeros_like(source)
        for c in range(source.shape[-1]):
            matched[..., c] = exposure.match_histograms(
                source[..., c], 
                reference[..., c]
            )
        return matched

def run_inference(
    model: torch.nn.Module,
    dataset_dir: str,
    output_dir: str,
    device: str = 'cuda',
    batch_size: int = 8,
    num_workers: int = 4,
    threshold: float = 0.5
):
    """
    Run inference on the dataset.
    
    Args:
        model: Trained PyTorch model
        dataset_dir: Root directory containing plot folders
        output_dir: Directory to save predictions
        device: Device to run inference on
        batch_size: Batch size for inference
        num_workers: Number of worker processes for data loading
        threshold: Threshold for binary prediction
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = DeforestationDataset(dataset_dir, range(1, 67))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for batch, plot_names in dataloader:
            # Move batch to device
            batch = batch.to(device)
            
            # Run inference
            predictions = model(batch)
            
            # Apply threshold
            binary_preds = (predictions > threshold).cpu().numpy()
            
            # Save predictions
            for pred, plot_name in zip(binary_preds, plot_names):
                output_path = os.path.join(output_dir, f"{plot_name}_pred.npy")
                np.save(output_path, pred)
                logger.info(f"Saved prediction: {output_path}")

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'dataset_dir': '../Datasets/Testing/TemporalStack/',
        'output_dir': '../Predictions/',
        'model_path': '../Models/unet_diff_model.pth',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 8,
        'num_workers': 4,
        'threshold': 0.5
    }
    
    # Load model
    model = torch.load(CONFIG['model_path'])
    
    # Run inference
    run_inference(
        model=model,
        dataset_dir=CONFIG['dataset_dir'],
        output_dir=CONFIG['output_dir'],
        device=CONFIG['device'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        threshold=CONFIG['threshold']
    )