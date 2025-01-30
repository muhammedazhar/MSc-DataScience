#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
U-Net Model for Deforestation Detection
---------------------------------------
The model is designed to be trained on satellite imagery to detect
deforestation. It uses a U-Net architecture with a ResNet-50 backbone to
process the input images.

Author: Azhar Muhammed
Date: December 2024
"""

# -----------------------------------------------------------------------------
# Essential Imports
# -----------------------------------------------------------------------------
import cv2
import shutil
import rasterio
import torch
import torch.nn as nn
import torchvision.models as models
import torch.multiprocessing as mp
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
from skimage import exposure
import albumentations as A
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Local Imports
# -----------------------------------------------------------------------------
from helper import *

# -----------------------------------------------------------------------------
# Logging and device configuration using helper script
# -----------------------------------------------------------------------------
mp.set_sharing_strategy('file_system')
filename = os.path.splitext(os.path.basename(__file__))[0]
# Set up logger using the imported function
logger, file_logger = setup_logging(file='None')
device = get_device()

# -----------------------------------------------------------------------------
# Traiing Configuration
# -----------------------------------------------------------------------------
TRAIN_CONFIG = {
    'batch_size': 8,
    'num_epochs': 200,
    'learning_rate': 1e-2,
    'dataset_dir': '../Datasets/Testing/TemporalStacks',
    'device': device,
    'num_workers': 2,      # Adjust based on the system's load
    'lr_milestones': [10, 40, 80, 150],
    'lr_gamma': 0.1
}


# -----------------------------------------------------------------------------
# Model Components
# -----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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


# -----------------------------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.2, dice_weight=0.8):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, predictions, targets):
        bce = self.bce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def create_optimizer_and_scheduler(model, config):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['lr_milestones'],
        gamma=config['lr_gamma']
    )
    return optimizer, scheduler


# -----------------------------------------------------------------------------
# Data Augmentations
# -----------------------------------------------------------------------------
train_transform = A.Compose([
    A.RandomCrop(int(224 * 0.7), int(224 * 0.7)),
    A.Resize(224, 224),
    A.RandomBrightnessContrast(p=0.5),
    A.ElasticTransform(p=0.5),
    A.GridDistortion(p=0.5),
    # A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.5) # Uncomment for more augmentation
], is_check_shapes=False)

val_transform = A.Compose([
    A.Resize(224, 224)
], is_check_shapes=False)


# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, config):
    """Train and validate the model."""
    model = model.to(config['device'])
    criterion = CombinedLoss(bce_weight=0.2, dice_weight=0.8)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0

        # Training loop with progress bar
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]", leave=False)
        for data, target in train_pbar:
            data, target = data.to(
                config['device']), target.to(config['device'])
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({"Train Loss": f"{loss.item():.4f}"})

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]", leave=False)
            for data, target in val_pbar:
                data, target = data.to(
                    config['device']), target.to(config['device'])
                output = model(data)
                v_loss = criterion(output, target).item()
                val_loss += v_loss
                val_pbar.set_postfix({"Val Loss": f"{v_loss:.4f}"})

        scheduler.step()

        # Compute average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Print epoch results (concise)
        logger.info(
            f"Epoch {epoch+1}/{config['num_epochs']}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '../Models/best_unet_diff.pth')
            logger.info("Model improved and saved.")


# -----------------------------------------------------------------------------
# Dataset Definition
# -----------------------------------------------------------------------------
class TemporalStackDataset(Dataset):
    """
    A dataset class for loading temporal stack data (pre-event, post-event, and mask).
    """

    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.pairs = self._get_temporal_pairs()

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

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _extract_date(filepath):
        """Extract date from filename (YYYYMMDDTHHMMSS)."""
        return Path(filepath).stem.split('T')[0]

    def _get_temporal_pairs(self):
        """Find valid temporal pairs of pre-event and post-event images based on time difference."""
        pairs = []
        plot_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        # Split into train/val sets
        split_idx = int(0.8 * len(plot_dirs))
        plot_dirs = plot_dirs[:split_idx] if self.split == 'train' else plot_dirs[split_idx:]

        for plot_dir in plot_dirs:
            pre_dir = plot_dir / 'Pre-event'
            post_dir = plot_dir / 'Post-event'
            mask_dir = plot_dir / 'Masks'

            if not all(p.exists() for p in [pre_dir, post_dir, mask_dir]):
                continue

            # Get temporal pairs (5-30 days apart)
            for pre_file in sorted(pre_dir.glob('*.npy')):
                pre_date = self._extract_date(pre_file)
                pre_dt = np.datetime64(pre_date)

                for post_file in sorted(post_dir.glob('*.npy')):
                    post_date = self._extract_date(post_file)
                    post_dt = np.datetime64(post_date)

                    delta = (post_dt - pre_dt).astype(int)
                    if 5 <= delta <= 30:
                        mask_file = mask_dir / f"{post_date}.tif"
                        if mask_file.exists():
                            pairs.append((pre_file, post_file, mask_file))
                            break
        return pairs

    def __getitem__(self, idx):
        pre_file, post_file, mask_file = self.pairs[idx]

        try:
            pre_img = np.load(pre_file).astype(
                np.float32) / 10000.0  # (C, H, W)
            post_img = np.load(post_file).astype(np.float32) / 10000.0

            # Transpose to (H, W, C) for processing
            pre_img = np.transpose(pre_img, (1, 2, 0))
            post_img = np.transpose(post_img, (1, 2, 0))

            logger.debug(f"Pre-image shape after transpose: {pre_img.shape}")
            logger.debug(f"Post-image shape after transpose: {post_img.shape}")

            with rasterio.open(mask_file) as src:
                mask = src.read(1)
                mask = (mask > 0).astype(np.float32)
                logger.debug(f"Mask shape: {mask.shape}")

            # Histogram match post-event to pre-event
            matched_post = self._histogram_match(post_img, pre_img)

            # Compute difference image
            diff_img = matched_post - pre_img

            # Stack channels: [pre (9), post (9), diff (9)] => 27 channels total
            x = np.concatenate([pre_img, matched_post, diff_img], axis=-1)
            logger.debug(f"Stacked input shape: {x.shape}")

            if self.transform:
                # Ensure mask shape matches image
                if mask.shape != x.shape[:2]:
                    mask = cv2.resize(
                        mask, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_NEAREST)

                transformed = self.transform(image=x, mask=mask)
                x = transformed['image']
                mask = transformed['mask']

                logger.debug(f"Transformed input shape: {x.shape}")
                logger.debug(f"Transformed mask shape: {mask.shape}")

            # Convert to torch tensors (C, H, W)
            x = torch.from_numpy(x).float().permute(2, 0, 1)
            mask = torch.from_numpy(mask).float().unsqueeze(0)

            return x, mask

        except Exception as e:
            logger.error("Error processing files:")
            logger.error(f"Pre-event: {pre_file}")
            logger.error(f"Post-event: {post_file}")
            logger.error(f"Mask: {mask_file}")
            raise e


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def remove_incomplete_plots(root_dir):
    """Remove plots that do not have required pre and post-event data."""
    root = Path(root_dir)
    plot_dirs = list(root.glob('PLOT-*'))
    for plot_dir in plot_dirs:
        pre_event_dir = plot_dir / 'Pre-event'
        post_event_dir = plot_dir / 'Post-event'

        pre_files = list(pre_event_dir.glob('*.npy'))
        post_files = list(post_event_dir.glob('*.npy'))

        if not pre_files or not post_files:
            logger.info(f"Removing {plot_dir} due to missing data.")
            shutil.rmtree(plot_dir)


def inspect_data_shapes(root_dir):
    """Inspect shapes of datasets for debugging and ensure consistency."""
    root = Path(root_dir)
    plot_dirs = list(root.glob('PLOT-*'))
    plot_dirs.sort(key=lambda x: int(x.name.split('-')[-1]))
    for plot_dir in plot_dirs:
        logger.debug(f"Inspecting {plot_dir.name}")
        mask_dir = plot_dir / 'Masks'
        mask_files = list(mask_dir.glob('*.tif'))
        if mask_files:
            mask_img = Image.open(mask_files[0])
            mask_shape = np.array(mask_img).shape
            logger.debug(f"Mask shape: {mask_shape}")

        pre_dir = plot_dir / 'Pre-event'
        pre_files = list(pre_dir.glob('*.npy'))
        if pre_files:
            pre_shape = np.load(pre_files[0]).shape
            logger.debug(f"Pre-event shape: {pre_shape}")

        post_dir = plot_dir / 'Post-event'
        post_files = list(post_dir.glob('*.npy'))
        if post_files:
            post_shape = np.load(post_files[0]).shape
            logger.debug(f"Post-event shape: {post_shape}")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    setup_logging(file=filename)  # Print file name in logs
    get_device(pretty='print')  # Print environment info once
    remove_incomplete_plots(TRAIN_CONFIG['dataset_dir'])
    inspect_data_shapes(TRAIN_CONFIG['dataset_dir'])

    # Create datasets
    train_dataset = TemporalStackDataset(
        root_dir=TRAIN_CONFIG['dataset_dir'],
        transform=train_transform,
        split='train'
    )

    val_dataset = TemporalStackDataset(
        root_dir=TRAIN_CONFIG['dataset_dir'],
        transform=val_transform,
        split='val'
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAIN_CONFIG['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers']
    )

    # Create and train model
    model = UNetDiff()

    # Display model summary
    batch_size = TRAIN_CONFIG['batch_size']
    summary(model, input_size=(batch_size, 27, 224, 224))

    train_model(model, train_loader, val_loader, TRAIN_CONFIG)


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        raise
