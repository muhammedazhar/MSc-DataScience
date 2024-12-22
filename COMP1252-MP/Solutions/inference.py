#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
U-Net Diff Inference for Deforestation Change Detection
-------------------------------------------------------
This script loads pre-event and post-event images from a temporal dataset, constructs
a 27-channel input (9 pre-event channels, 9 post-event channels, and 9 difference channels),
applies histogram matching, normalizes the data, and runs inference using a U-Net model.
It outputs a binary change detection mask representing deforestation areas.

Author: Azhar Muhammed
Date: December 2024
"""

# ------------------------------------------------------------
# Essential Imports
# ------------------------------------------------------------
import os
import cv2
import glob
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import torchvision.models as models
import rasterio
from skimage import exposure
import logging
from typing import Tuple, Optional, Dict
from torchmetrics import Dice, F1Score

# Local imports
from helper import *

# ------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------
setup_logging()
device = get_device()

# ------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Dataset Definition
# ------------------------------------------------------------
class DeforestationDataset(Dataset):
    """
    Dataset for loading temporal image pairs for deforestation detection.
    """
    def __init__(self, dataset_dir: str, plot_range: range, img_size: Tuple[int, int] = (224, 224), cloud_threshold: float = 0.7, transform: Optional[A.Compose] = None, split: str = 'test'):
        self.dataset_dir = Path(dataset_dir)
        self.plot_range = plot_range
        self.img_size = img_size
        self.cloud_threshold = cloud_threshold
        self.transform = transform
        self.split = split

        # Get valid temporal pairs based on mask dates
        self.pairs = self._get_temporal_pairs()

        logging.info(f"Total number of samples in dataset '{self.split}': {len(self.pairs)}")

    def _get_temporal_pairs(self):
        pairs = []

        for plot_id in self.plot_range:
            plot_name = f"PLOT-{plot_id:05d}"
            plot_dir = self.dataset_dir / plot_name

            if not plot_dir.exists():
                logging.warning(f"Plot directory does not exist: {plot_dir}")
                continue

            pre_dir = plot_dir / "Pre-event"
            post_dir = plot_dir / "Post-event"
            mask_dir = plot_dir / "Masks"

            if not all(d.exists() for d in [pre_dir, post_dir, mask_dir]):
                logging.warning(f"Missing subdirectories in plot {plot_name}. Skipping.")
                continue

            plot_pairs = self._get_plot_temporal_pairs(pre_dir, post_dir, mask_dir)
            pairs.extend(plot_pairs)

        return pairs

    def _get_plot_temporal_pairs(self, pre_dir: Path, post_dir: Path, mask_dir: Path):
        pairs = []
        mask_files = sorted(mask_dir.glob("*.tif"))

        if not mask_files:
            logging.warning(f"No mask files found in {mask_dir}. Skipping plot.")
            return pairs

        for mask_file in mask_files:
            mask_date_str = mask_file.stem  # Extract 'YYYYMMDD' from 'YYYYMMDD.tif'
            try:
                mask_dt = np.datetime64(mask_date_str)
            except ValueError as e:
                logging.error(f"Invalid mask date format in file: {mask_file}. Error: {e}")
                continue

            # Find the pre-event image closest to but before the mask date
            pre_files = sorted(pre_dir.glob("*.npy"))
            suitable_pre = None
            for pre_file in pre_files:
                pre_date = self._extract_date(pre_file)
                try:
                    pre_dt = np.datetime64(pre_date)
                except ValueError as e:
                    logging.error(f"Invalid pre-event date format in file: {pre_file}. Error: {e}")
                    continue

                if pre_dt <= mask_dt:
                    suitable_pre = pre_file
                else:
                    break  # Since pre_files are sorted, no need to check further

            # Find the post-event image closest to but after the mask date within 5-30 days
            post_files = sorted(post_dir.glob("*.npy"))
            suitable_post = None
            for post_file in post_files:
                post_date = self._extract_date(post_file)
                try:
                    post_dt = np.datetime64(post_date)
                except ValueError as e:
                    logging.error(f"Invalid post-event date format in file: {post_file}. Error: {e}")
                    continue

                delta_days = (post_dt - mask_dt).astype(int)
                if 5 <= delta_days <= 30:
                    suitable_post = post_file
                    break  # Take the first post-event within the window

            if suitable_pre and suitable_post:
                pairs.append((suitable_pre, suitable_post, mask_file))
                logging.debug(f"Pair added: Pre={suitable_pre.name}, Post={suitable_post.name}, Mask={mask_file.name}")
            else:
                logging.warning(
                    f"Could not find suitable pre/post images for mask {mask_file.name} in plot {mask_file.parent.parent.name}."
                )

        return pairs

    @staticmethod
    def _extract_date(filepath: Path) -> str:
        stem = filepath.stem  # 'YYYYMMDDTHHMMSS'
        if 'T' in stem:
            return stem.split('T')[0]
        else:
            # If 'T' is not present, assume the entire stem is the date
            return stem

    @staticmethod
    def _load_mask(mask_file: Path) -> np.ndarray:
        with rasterio.open(mask_file) as src:
            mask = src.read(1)
            return (mask > 0).astype(np.float32)

    @staticmethod
    def _histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        matched = np.zeros_like(source)
        for c in range(source.shape[-1]):
            matched[..., c] = exposure.match_histograms(
                source[..., c],
                reference[..., c]
            )
        return matched

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pre_file, post_file, mask_file = self.pairs[idx]

        try:
            # Load and preprocess images
            pre_img = np.load(pre_file).astype(np.float32) / 10000.0  # Shape: (C, H, W)
            post_img = np.load(post_file).astype(np.float32) / 10000.0  # Shape: (C, H, W)

            # Transpose to (H, W, C) for processing
            pre_img = np.transpose(pre_img, (1, 2, 0))  # (H, W, C)
            post_img = np.transpose(post_img, (1, 2, 0))  # (H, W, C)

            # Load mask
            mask = self._load_mask(mask_file)  # (H, W)

            # Histogram match post-event to pre-event
            matched_post = self._histogram_match(post_img, pre_img)

            # Compute difference image
            diff_img = matched_post - pre_img

            # Stack channels: [pre (9), post (9), diff (9)] => 27 channels total
            x = np.concatenate([pre_img, matched_post, diff_img], axis=-1)  # (H, W, 27)

            # Apply transforms if any
            if self.transform:
                # Ensure mask shape matches image
                if mask.shape != x.shape[:2]:
                    mask = cv2.resize(mask, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_NEAREST)

                transformed = self.transform(image=x, mask=mask)
                x = transformed['image']
                mask = transformed['mask']

            # Convert to torch tensors (C, H, W)
            x = torch.from_numpy(x).float().permute(2, 0, 1)  # (27, H, W)
            mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)

            # Optionally, include plot name for saving predictions
            plot_name = pre_file.parent.parent.name  # Assuming structure: PLOT-XXXXX/Pre-event/...

            return x, mask, plot_name

        except Exception as e:
            logging.error(f"Error processing files:")
            logging.error(f"Pre-event: {pre_file}")
            logging.error(f"Post-event: {post_file}")
            logging.error(f"Mask: {mask_file}")
            logging.error(f"Error: {e}")
            raise e

# ------------------------------------------------------------
# Metric Tracking
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Inference Routine
# ------------------------------------------------------------
def run_inference(
    model: torch.nn.Module,
    dataset_dir: str,
    output_dir: str,
    device: str = device,
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

# ------------------------------------------------------------
# Main Inference Execution
# ------------------------------------------------------------
if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'dataset_dir': '../Datasets/Testing/TemporalStack/',
        'output_dir': '../Predictions/',
        'model_path': '../Models/best_unet_diff.pth',
        'device': device,
        'batch_size': 8,
        'num_workers': 4,
        'threshold': 0.5  # Configurable threshold for binary prediction
    }
    dataset = DeforestationDataset(CONFIG['dataset_dir'], range(1, 67))
    print("Number of samples in dataset:", len(dataset))  # Debug line
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
