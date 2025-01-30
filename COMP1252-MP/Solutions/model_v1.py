#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
U-Net Model for Image Segmentation
----------------------------------
This script implements a U-Net model for image segmentation using PyTorch. It
defines a U-Net architecture with skip connections and batch normalization. The
model is designed for semantic segmentation tasks.

Author: Azhar Muhammed
Date: October 2024
"""

# -----------------------------------------------------------------------------
# Essential Imports
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import random
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import time  # type: ignore

# -----------------------------------------------------------------------------
# Local imports
# -----------------------------------------------------------------------------
from helper import setup_logging, get_device

# -----------------------------------------------------------------------------
# Constants and Configuration
# -----------------------------------------------------------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

filename = os.path.splitext(os.path.basename(__file__))[0]
# Set up logger using the imported function
logger, file_logger = setup_logging(file=filename)

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
DEVICE = get_device(pretty='print')

# -----------------------------------------------------------------------------
# Custom Dataset Definition
# -----------------------------------------------------------------------------
class NucleiDataset(Dataset):
    def __init__(self, images, masks=None, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()
        image = image.permute(2, 0, 1)  # Change from HWC to CHW format

        if self.masks is not None:
            mask = torch.from_numpy(self.masks[idx]).float()
            mask = mask.permute(2, 0, 1)
            return image, mask
        return image

# -----------------------------------------------------------------------------
# U-Net Model Definition
# -----------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self._make_layer(IMG_CHANNELS, 16)
        self.enc2 = self._make_layer(16, 32)
        self.enc3 = self._make_layer(32, 64)
        self.enc4 = self._make_layer(64, 128)
        self.enc5 = self._make_layer(128, 256)

        # Decoder
        self.up6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec6 = self._make_layer(256, 128)  # 256 because of concatenation

        self.up7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec7 = self._make_layer(128, 64)

        self.up8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec8 = self._make_layer(64, 32)

        self.up9 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec9 = self._make_layer(32, 16)

        self.final = nn.Conv2d(16, 1, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.1)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Normalize input
        x = x / 255.0

        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool(c1)

        c2 = self.enc2(p1)
        p2 = self.pool(c2)

        c3 = self.enc3(p2)
        p3 = self.pool(c3)

        c4 = self.enc4(p3)
        p4 = self.pool(c4)

        c5 = self.enc5(p4)

        # Decoder
        up6 = self.up6(c5)
        merge6 = torch.cat([up6, c4], dim=1)
        c6 = self.dec6(merge6)

        up7 = self.up7(c6)
        merge7 = torch.cat([up7, c3], dim=1)
        c7 = self.dec7(merge7)

        up8 = self.up8(c7)
        merge8 = torch.cat([up8, c2], dim=1)
        c8 = self.dec8(merge8)

        up9 = self.up9(c8)
        merge9 = torch.cat([up9, c1], dim=1)
        c9 = self.dec9(merge9)

        out = torch.sigmoid(self.final(c9))

        return out

# -----------------------------------------------------------------------------
# Evaluation Metrics
# -----------------------------------------------------------------------------
def calculate_metrics(pred, target):
    pred = (pred > 0.5).float()

    # Pixel-wise accuracy
    accuracy = (pred == target).float().mean()

    # IoU (Intersection over Union)
    intersection = (pred * target).sum()
    union = (pred + target).bool().float().sum()
    iou = (intersection + 1e-7) / (union - intersection + 1e-7)  # Add small epsilon to avoid division by zero

    return accuracy.item(), iou.item()

def calculate_pixel_accuracy(pred, target):
    pred = (pred > 0.5).float()
    correct = (pred == target).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy

def calculate_iou(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    return iou

# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_val_loss = float('inf')
    best_epoch = 0
    best_val_acc = 0
    best_val_iou = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_iou': [],
        'val_loss': [], 'val_acc': [], 'val_iou': []
    }

    start_training_time = time.time()

    for epoch in range(num_epochs):
        text = f'Epoch {epoch+1}/{num_epochs}'
        print(f'\n{text}\n{"-" * len(text)}')

        # Training phase
        model.train()
        train_loss = 0
        train_acc = 0
        train_iou = 0

        for inputs, masks in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            # Calculate metrics
            acc = calculate_pixel_accuracy(outputs, masks)
            iou = calculate_iou(outputs, masks)
            train_loss += loss.item()
            train_acc += acc.item()
            train_iou += iou.item()

        # Average metrics over batches
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_iou /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_acc = 0
        val_iou = 0

        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs = inputs.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, masks)

                # Calculate metrics
                acc = calculate_pixel_accuracy(outputs, masks)
                iou = calculate_iou(outputs, masks)
                val_loss += loss.item()
                val_acc += acc.item()
                val_iou += iou.item()

            # Average metrics over batches
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            val_iou /= len(val_loader)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_iou = val_iou
                best_epoch = epoch + 1
                torch.save(model.state_dict(), '../Models/Testing-Model.pth')

        # Store metrics in history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_iou'].append(val_iou)

        # Print metrics
        print(f'Train Loss : {train_loss:.4f} - Train Acc : {train_acc:.4f} - Train IoU : {train_iou:.4f}')
        print(f'Val Loss   : {val_loss:.4f} - Val Acc   : {val_acc:.4f} - Val IoU   : {val_iou:.4f}')

    total_training_time = time.time() - start_training_time
    print('\nTraining completed!')
    print('Best Model Performance:')
    print(f'- Validation Loss: {best_val_loss:.4f}')
    print(f'- Validation Accuracy: {best_val_acc:.4f}')
    print(f'- Validation IoU: {best_val_iou:.4f}\n')
    print(f'Best epoch: {best_epoch}')

    return model, history, total_training_time

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    """Main execution function"""
    # Load and preprocess data (using existing data loading code)
    TRAIN_PATH = '../Datasets/Testing/Test-UNetModel/stage1_train/'
    TEST_PATH = '../Datasets/Testing/Test-UNetModel/stage1_test/'

    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]

    # Load training data
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

    print('Loading training images and masks')
    loading_time = time.time()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img

        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask

    loading_time = time.time() - loading_time
    # Create datasets and dataloaders
    train_size = int(0.9 * len(X_train))
    train_dataset = NucleiDataset(X_train[:train_size], Y_train[:train_size])
    val_dataset = NucleiDataset(X_train[train_size:], Y_train[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialize model, criterion, and optimizer
    model = UNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # Train model and get history
    model, history, training_time = train_model(model, train_loader, val_loader, criterion, optimizer)

    # Print time metrics
    print(f'Training Time: {int(training_time // 60)}m {int(training_time % 60)}s')
    print(f'Loading Time: {int(loading_time // 60)}m {int(loading_time % 60)}s')

    # Visualize results
    save_dir = '../Docs/Diagrams/'
    file_prefix = f'{filename}'

    # Make predictions
    model.eval()
    test_dataset = NucleiDataset(X_train[train_size:], Y_train[train_size:])  # Include masks
    test_loader = DataLoader(test_dataset, batch_size=16)

    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            # Unpack the batch - since we included masks in test_dataset
            inputs, _ = batch
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    predictions = np.array(predictions)
    predictions = (predictions > 0.5).astype(np.uint8)

    # Visualize results
    plt.figure(figsize=(15, 5))

    # Plot Loss
    plt.subplot(131)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(132)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot IoU
    plt.subplot(133)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('Model IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{file_prefix}_performance_metrics.png'))
    plt.close()

    # Visualize sample predictions
    idx = random.randint(0, len(predictions) - 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(X_train[train_size + idx])
    plt.title('Original Image')

    plt.subplot(132)
    plt.imshow(np.squeeze(Y_train[train_size + idx]), cmap='gray')
    plt.title('True Mask')

    plt.subplot(133)
    plt.imshow(np.squeeze(predictions[idx]), cmap='gray')
    plt.title('Predicted Mask')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{file_prefix}_sample_prediction.png'))
    plt.close()


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        raise
