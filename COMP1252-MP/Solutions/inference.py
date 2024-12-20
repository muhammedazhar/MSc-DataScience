#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
U-Net Diff Inference for Deforestation Change Detection
-------------------------------------------------------
This script loads pre-event and post-event images from a temporal dataset, constructs
a 27-channel input (9 pre-event channels, 9 post-event channels, and 9 difference channels),
applies histogram matching, normalizes the data, and runs inference using a U-Net model.
It outputs a binary change detection mask representing deforestation areas.

Dataset Structure:
../Datasets/Testing/TemporalStack/
    PLOT-00001/
        Masks/
            YYYYMMDD.tif
        Pre-event/
            YYYYMMDDTHHMMSS.npy
            stack_info.json
        Post-event/
            YYYYMMDDTHHMMSS.npy
            stack_info.json
    PLOT-00002/
    ...
    PLOT-00066/

Requirements:
- Proper temporal alignment (5-30 days difference)
- 27-channel input:
  [0:9]   -> Pre-event (RGB, NIR/SWIR, NDVI, NDMI)
  [9:18]  -> Post-event (RGB, NIR/SWIR, NDVI, NDMI)
  [18:27] -> Difference (post - pre)
- Each image scaled by 1/10000
- Apply histogram matching between pre- and post-event images
- Model input: 224x224 patches
- Outputs a binary mask (thresholded from model's sigmoid output)

Author: Azhar Muhammed
Date: December 2024
"""

import os
import glob
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import rasterio
from skimage import exposure
import logging
from typing import Tuple, Optional, Dict
from torchmetrics import Dice, F1Score

# Local imports
from helper import *
# Configure logging
setup_logging()

class ConvBlock(nn.Module):
    """Basic convolutional block for U-Net."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetDiff(nn.Module):
    """U-Net with ResNet-50 backbone for deforestation detection."""

    def __init__(self):
        super().__init__()

        # Load ResNet-50 backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Modify first conv to accept 27 channels
        self.encoder1 = nn.Sequential(
            nn.Conv2d(27, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu
        )
        self.pool = resnet.maxpool

        # Encoder blocks (from ResNet)
        self.encoder2 = resnet.layer1  # 256 channels
        self.encoder3 = resnet.layer2  # 512 channels
        self.encoder4 = resnet.layer3  # 1024 channels
        self.encoder5 = resnet.layer4  # 2048 channels

        # Decoder blocks with correct channel sizes
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder5 = ConvBlock(2048, 1024)  # 1024 + 1024 input channels

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(1024, 512)   # 512 + 512 input channels

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(512, 256)    # 256 + 256 input channels

        self.upconv2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(128, 64)     # 64 + 64 input channels

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(59, 32)      # 32 + 27 input channels

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pool1 = self.pool(enc1)

        enc2 = self.encoder2(pool1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        # Decoder with skip connections
        dec5 = self.upconv5(enc5)
        dec5 = torch.cat([dec5, enc4], dim=1)
        dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(dec5)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, x], dim=1)
        dec1 = self.decoder1(dec1)

        out = self.final_conv(dec1)
        return torch.sigmoid(out)

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

            plot_pairs = self._get_plot_temporal_pairs(pre_dir, post_dir, mask_dir)
            pairs.extend(plot_pairs)

        return pairs

    def _get_plot_temporal_pairs(self, pre_dir, post_dir, mask_dir):
        pairs = []
        # Assume one mask per plot for the deforestation event
        mask_files = list(mask_dir.glob("*.tif"))
        if not mask_files:
            return pairs

        mask_file = mask_files[0]
        mask_date_str = mask_file.stem  # e.g., '20180731'
        mask_dt = np.datetime64(mask_date_str)

        pre_files = sorted(pre_dir.glob("*.npy"))
        post_files = sorted(post_dir.glob("*.npy"))

        # Find a suitable pre_file (before mask_dt)
        suitable_pre = None
        for p in pre_files:
            p_date = np.datetime64(self._extract_date(p))
            if p_date <= mask_dt:
                suitable_pre = p
            else:
                # Once we pass mask_dt, no need to continue
                break

        # Find a suitable post_file (after mask_dt, within 5–30 days)
        suitable_post = None
        for p in post_files:
            p_date = np.datetime64(self._extract_date(p))
            delta = (p_date - mask_dt).astype(int)
            if 5 <= delta <= 30:
                suitable_post = p
                break

        if suitable_pre is not None and suitable_post is not None:
            pairs.append((suitable_pre, suitable_post, mask_file))

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
        mask = self._load_mask(mask_file)

        # Scale images
        pre_img = pre_img.astype(np.float32) / 10000.0
        post_img = post_img.astype(np.float32) / 10000.0

        # Apply histogram matching
        post_matched = self._histogram_match(post_img, pre_img)

        # Compute difference
        diff_img = post_matched - pre_img

        # Stack channels
        combined = np.concatenate([pre_img, post_matched, diff_img], axis=-1)

        # Convert to tensors
        combined = torch.from_numpy(combined).float()
        mask = torch.from_numpy(mask).float()

        # Move channels first
        combined = combined.permute(2, 0, 1)
        mask = mask.unsqueeze(0)  # Add channel dimension

        return combined, mask, pre_file.parent.parent.name

    @staticmethod
    def _load_mask(mask_file):
        """Load and preprocess mask."""
        with rasterio.open(mask_file) as src:
            mask = src.read(1)
            return (mask > 0).astype(np.float32)

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

class MetricTracker:
    """Track multiple metrics during inference."""
    def __init__(self, device):
        self.dice = Dice().to(device)
        self.f1 = F1Score(task="binary").to(device)
        self.results = {
            'dice': [],
            'f1': []
        }

    def update(self, preds, targets):
        """Update metrics with new predictions."""
        self.dice.update(preds, targets)
        self.f1.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        return {
            'dice': self.dice.compute().item(),
            'f1': self.f1.compute().item()
        }

def run_inference(
    model: torch.nn.Module,
    dataset_dir: str,
    output_dir: str,
    device: str = 'mps',
    batch_size: int = 8,
    num_workers: int = 4,
    threshold: float = 0.5
) -> Dict[str, float]:
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

    Returns:
        Dictionary containing computed metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = DeforestationDataset(dataset_dir, range(1, 67))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    model.eval()
    model.to(device)

    # Initialize metric tracker
    metrics = MetricTracker(device)

    with torch.no_grad():
        for batch, masks, plot_names in dataloader:
            # Move batch to device
            batch = batch.to(device)
            masks = masks.to(device)

            # Run inference
            predictions = model(batch)

            # Apply threshold and compute metrics
            binary_preds = (predictions > threshold)
            metrics.update(binary_preds, masks)

            # Convert to numpy and save predictions
            binary_preds = binary_preds.cpu().numpy()

            for pred, plot_name in zip(binary_preds, plot_names):
                output_path = os.path.join(output_dir, f"{plot_name}_pred.npy")
                np.save(output_path, pred)
                logging.info(f"Saved prediction: {output_path}")

    # Compute and return final metrics
    if len(dataset) == 0:
        logging.warning("No samples found in dataset. Metrics cannot be computed.")
        metrics = {'dice': 0.0, 'f1': 0.0}
    else:
        metrics = metrics.compute()
    return metrics

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'dataset_dir': '../Datasets/Testing/TemporalStack/',
        'output_dir': '../Predictions/',
        'model_path': '../Models/best_unet_diff.pth',
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'batch_size': 8,
        'num_workers': 4,
        'threshold': 0.5  # Configurable threshold for binary prediction
    }
    dataset = DeforestationDataset(CONFIG['dataset_dir'], range(1, 67))
    print("Number of samples in dataset:", len(dataset))  # Debug line
    # Set device
    device = get_device(pretty=print)
    # Load model
    model = UNetDiff()
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    model.to(device)
    model.eval()

    # Run inference and get metrics
    metrics = run_inference(
        model=model,
        dataset_dir=CONFIG['dataset_dir'],
        output_dir=CONFIG['output_dir'],
        device=CONFIG['device'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        threshold=CONFIG['threshold']
    )

    # Log metrics
    logging.info("Inference completed. Final metrics:")
    for metric_name, value in metrics.items():
        logging.info(f"{metric_name}: {value:.4f}")
