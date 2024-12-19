# Essential imports
import cv2
import shutil
import rasterio
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
from skimage import exposure

# Print the PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Running on a local machine
if torch.backends.mps.is_available():
    device = 'mps'
    message = "Apple Silicon Metal Performance Shader (MPS) Support"
    print(f"\n{message} \n{'-' * len(message)}")
    print(f"Apple MPS built status : {torch.backends.mps.is_built()}")
    print(f"Apple MPS availability : {torch.backends.mps.is_available()}")
    print(f"{'-' * len(message)}")
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# TODO: Add support for AMD ROCm GPU if needed

# Print the device being used
print(f"Using device: {device.upper()}\n")

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
    def __init__(self):
        super().__init__()

        # Load ResNet-50 backbone with proper weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Encoder (modified ResNet-18)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(27, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu
        )
        self.pool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(128, 64)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(96, 32)

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(80, 16)

        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)  # 64 channels
        pool1 = self.pool(enc1)

        enc2 = self.encoder2(pool1)  # 64 channels
        enc3 = self.encoder3(enc2)  # 128 channels

        # Decoder with skip connections
        dec3 = self.upconv3(enc3)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, x], dim=1)
        dec1 = self.decoder1(dec1)

        # Final 1x1 convolution
        out = self.final_conv(dec1)
        return torch.sigmoid(out)

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

def create_optimizer_and_scheduler(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 40, 80, 150],
        gamma=0.1
    )

    return optimizer, scheduler

import albumentations as A
from torch.utils.data import Dataset, DataLoader

# Configuration
TRAIN_CONFIG = {
    'batch_size': 8,
    'num_epochs': 200,
    'learning_rate': 1e-2,
    'dataset_dir': '../Datasets/Testing/TemporalStacks',
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'num_workers': 4,
    'lr_milestones': [10, 40, 80, 150],
    'lr_gamma': 0.1
}

# Augmentations for training
train_transform = A.Compose([
    A.RandomCrop(int(224 * 0.7), int(224 * 0.7)),
    A.Resize(224, 224),
    A.RandomBrightnessContrast(p=0.5),
    A.ElasticTransform(p=0.5),
    A.GridDistortion(p=0.5),
    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.5)
], is_check_shapes=False)  # Disable shape checking temporarily

# Validation transform - only resize if needed
val_transform = A.Compose([
    A.Resize(224, 224)
], is_check_shapes=False)  # Disable shape checking temporarily

def train_model(model, train_loader, val_loader, config):
    model = model.to(config['device'])
    criterion = CombinedLoss(bce_weight=0.2, dice_weight=0.8)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['lr_milestones'],
        gamma=config['lr_gamma']
    )

    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(config['device'])
            target = target.to(config['device'])

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(config['device'])
                target = target.to(config['device'])
                output = model(data)
                val_loss += criterion(output, target).item()

        scheduler.step()

        # Print epoch results
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f'Epoch {epoch+1}/{config["num_epochs"]}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_unet_diff.pth')

import numpy as np
from pathlib import Path

class TemporalStackDataset(Dataset):
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
        pairs = []
        plot_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        # Split into train/val
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
            # Load data
            pre_img = np.load(pre_file).astype(np.float32) / 10000.0  # Shape: (C, H, W)
            post_img = np.load(post_file).astype(np.float32) / 10000.0  # Shape: (C, H, W)

            # Transpose to (H, W, C) for processing
            pre_img = np.transpose(pre_img, (1, 2, 0))  # Shape: (H, W, C)
            post_img = np.transpose(post_img, (1, 2, 0))  # Shape: (H, W, C)

            print(f"Pre-image shape after transpose: {pre_img.shape}")
            print(f"Post-image shape after transpose: {post_img.shape}")

            # Load and binarize mask
            with rasterio.open(mask_file) as src:
                mask = src.read(1)
                mask = (mask > 0).astype(np.float32)
                print(f"Mask shape: {mask.shape}")

            # Apply histogram matching
            matched_post = self._histogram_match(post_img, pre_img)

            # Compute difference image
            diff_img = matched_post - pre_img

            # Stack channels: [pre (9), post (9), diff (9)]
            x = np.concatenate([pre_img, matched_post, diff_img], axis=-1)  # Shape: (H, W, 27)
            print(f"Stacked input shape: {x.shape}")

            if self.transform:
                # Ensure mask shape matches image shape before transform
                if mask.shape != x.shape[:2]:  # Compare spatial dimensions only
                    print(f"Reshaping mask from {mask.shape} to {x.shape[:2]}")
                    mask = cv2.resize(mask, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_NEAREST)

                transformed = self.transform(image=x, mask=mask)
                x = transformed['image']
                mask = transformed['mask']
                print(f"Transformed input shape: {x.shape}")
                print(f"Transformed mask shape: {mask.shape}")

            # Convert to torch tensors and transpose back to (C, H, W)
            x = torch.from_numpy(x).float()  # Shape: (H, W, C)
            x = x.permute(2, 0, 1)  # Shape: (C, H, W)
            mask = torch.from_numpy(mask).float().unsqueeze(0)  # Shape: (1, H, W)

            return x, mask

        except Exception as e:
            print(f"Error processing files:")
            print(f"Pre-event: {pre_file}")
            print(f"Post-event: {post_file}")
            print(f"Mask: {mask_file}")
            raise e

def remove_incomplete_plots(root_dir):
    root = Path(root_dir)
    plot_dirs = list(root.glob('PLOT-*'))

    for plot_dir in plot_dirs:
        pre_event_dir = plot_dir / 'Pre-event'
        post_event_dir = plot_dir / 'Post-event'

        pre_files = list(pre_event_dir.glob('*.npy'))
        post_files = list(post_event_dir.glob('*.npy'))

        if not pre_files or not post_files:
            print(f"Removing {plot_dir} due to missing data.")
            shutil.rmtree(plot_dir)

def inspect_data_shapes(root_dir):
    root = Path(root_dir)
    plot_dirs = list(root.glob('PLOT-*'))
    # Sort plot directories numerically
    plot_dirs.sort(key=lambda x: int(x.name.split('-')[-1]))
    for plot_dir in plot_dirs:
        print(f"\nInspecting {plot_dir.name}")

        # Check masks
        mask_dir = plot_dir / 'Masks'
        mask_files = list(mask_dir.glob('*.tif'))
        if mask_files:
            mask_img = Image.open(mask_files[0])
            mask_shape = np.array(mask_img).shape
            print(f"Mask shape       : {mask_shape}")

        # Check pre-event
        pre_dir = plot_dir / 'Pre-event'
        pre_files = list(pre_dir.glob('*.npy'))
        if pre_files:
            pre_shape = np.load(pre_files[0]).shape
            print(f"Pre-event shape  : {pre_shape}")

        # Check post-event
        post_dir = plot_dir / 'Post-event'
        post_files = list(post_dir.glob('*.npy'))
        if post_files:
            post_shape = np.load(post_files[0]).shape
            print(f"Post-event shape : {post_shape}")

# Call the function before data inspection
remove_incomplete_plots('../Datasets/Testing/TemporalStacks')

# Run inspection
inspect_data_shapes('../Datasets/Testing/TemporalStacks')

if __name__ == "__main__":
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

    # Create model
    model = UNetDiff()

    # Train model
    train_model(model, train_loader, val_loader, TRAIN_CONFIG)
