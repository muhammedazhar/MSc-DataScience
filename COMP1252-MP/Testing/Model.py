# Essential imports
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image

# Print the PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Running on a local machine
if torch.backends.mps.is_available():
    device = 'mps'
    message = "Apple Silicon Metal Performance Shader (MPS) Support"
    print(f"\n{message} \n{'-' * len(message)}")
    print(f"Apple MPS built status : {torch.backends.mps.is_built()}")
    print(f"Apple MPS available    : {torch.backends.mps.is_available()}")
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

        # ResNet-18 backbone
        resnet = models.resnet18(pretrained=True)

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

# Define augmentations
train_transform = A.Compose([
    A.RandomCrop(int(224 * 0.7), int(224 * 0.7)),
    A.Resize(224, 224),
    A.RandomBrightnessContrast(p=0.5),
    A.ElasticTransform(p=0.5),
    A.GridDistortion(p=0.5),
    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.5)
])

def train_model(model, train_loader, val_loader, num_epochs=200):
    model.summary()
    model = model.to(device)

    criterion = CombinedLoss()
    optimizer, scheduler = create_optimizer_and_scheduler(model)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

        scheduler.step()

        # Print epoch results
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f'Epoch {epoch+1}/{num_epochs}:')
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

        # Get all plot directories
        plot_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        # Split train/val
        split_idx = int(0.8 * len(plot_dirs))
        self.plot_dirs = plot_dirs[:split_idx] if split == 'train' else plot_dirs[split_idx:]

    def __len__(self):
        return len(self.plot_dirs)

    def load_temporal_stack(self, stack_dir):
        npy_files = sorted(list(stack_dir.glob('*.npy')))
        if not npy_files:
            raise ValueError(f"No .npy files found in {stack_dir}")
        return np.concatenate([np.load(f) for f in npy_files], axis=-1)

    def __getitem__(self, idx):
        plot_dir = self.plot_dirs[idx]

        # Load stacks
        pre_event_dir = plot_dir / 'Pre-event'
        post_event_dir = plot_dir / 'Post-event'

        try:
            pre_stack = self.load_temporal_stack(pre_event_dir)
            post_stack = self.load_temporal_stack(post_event_dir)

            # Concatenate pre and post stacks along the time/channel dimension
            x = np.concatenate([pre_stack, post_stack], axis=0)  # Shape: (channels, height, width)

            # Transpose to (height, width, channels) for Albumentations
            x = np.transpose(x, (1, 2, 0))  # Now x.shape is (height, width, channels)

            # Create dummy mask with the same spatial dimensions
            h, w, _ = x.shape
            mask = np.zeros((h, w, 1), dtype=np.float32)

            if self.transform:
                transformed = self.transform(image=x, mask=mask)
                x = transformed['image']
                mask = transformed['mask']

            # Convert to torch tensors
            x = torch.from_numpy(x).float().permute(2, 0, 1)  # Shape: (channels, height, width)
            mask = torch.from_numpy(mask).float().permute(2, 0, 1)

            return x, mask

        except Exception as e:
            print(f"Error loading data from {plot_dir}: {str(e)}")
            raise

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

# Initialize datasets with fewer workers for debugging
train_dataset = TemporalStackDataset(
    root_dir='../Datasets/Testing/TemporalStacks',
    transform=train_transform,
    split='train'
)

val_dataset = TemporalStackDataset(
    root_dir='../Datasets/Testing/TemporalStacks',
    transform=None,
    split='val'
)

# Create data loaders with fewer workers
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0  # Start with 0 for debugging
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0  # Start with 0 for debugging
)

# Test loading one batch
try:
    test_batch = next(iter(train_loader))
    print("Test batch shape:", test_batch[0].shape)
except Exception as e:
    print("Error loading test batch:", str(e))
