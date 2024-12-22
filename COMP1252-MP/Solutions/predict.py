#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction Script for Deforestation Detection
---------------------------------------------
This script loads a trained model and uses it to make predictions on a set of
of input images. It also provides visualization of the predictions.

Author: Azhar Muhammed
Date: December 2024
"""

# ------------------------------------------------------------
# Essential Imports
# ------------------------------------------------------------
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import exposure
from datetime import datetime
import rasterio
from tabulate import tabulate
from sklearn.metrics import precision_score, recall_score, f1_score

# ------------------------------------------------------------
# Local Imports
# ------------------------------------------------------------
from helper import *
from model_v2 import UNetDiff

# ------------------------------------------------------------
# Data Loading and Preprocessing Functions
# ------------------------------------------------------------
def load_and_preprocess_data(pre_file, post_file):
    """Load and preprocess image pairs."""
    # Load numpy arrays
    pre_img = np.load(pre_file).astype(np.float32) / 10000.0  # Scale by 1/10000
    post_img = np.load(post_file).astype(np.float32) / 10000.0

    # Transpose to (H, W, C) for processing
    pre_img = np.transpose(pre_img, (1, 2, 0))
    post_img = np.transpose(post_img, (1, 2, 0))

    # Histogram matching
    matched_post = np.zeros_like(post_img)
    for c in range(post_img.shape[-1]):
        matched_post[..., c] = exposure.match_histograms(
            post_img[..., c],
            pre_img[..., c]
        )

    # Compute difference
    diff_img = matched_post - pre_img

    # Stack channels
    x = np.concatenate([pre_img, matched_post, diff_img], axis=-1)

    # Convert to torch tensor (C, H, W)
    x = torch.from_numpy(x).float().permute(2, 0, 1)
    return x.unsqueeze(0)  # Add batch dimension

def load_mask(mask_file):
    """Load and preprocess mask."""
    with rasterio.open(mask_file) as src:
        mask = src.read(1)
        return (mask > 0).astype(np.float32)

# ------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------
def calculate_metrics(true_mask, pred_mask):
    """Calculate comprehensive metrics for the prediction."""
    # Flatten masks for metric calculation
    y_true = true_mask.flatten()
    y_pred = pred_mask.flatten()

    # Calculate intersection and union for Dice/IoU
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)

    # Calculate metrics
    dice = (2.0 * intersection) / union if union > 0 else 0
    iou = intersection / (union - intersection) if (union - intersection) > 0 else 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'Dice Score': dice,
        'IoU Score': iou,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# ------------------------------------------------------------
# Main Prediction and Visualization Pipeline
# ------------------------------------------------------------
def predict_and_visualize():
    """Main function for deforestation detection and visualization."""
    text = "Deforestation Detection Analysis"
    print(f"\n{text}\n{'-' * len(text)}")

    # Setup paths
    plot_dir = Path("../Datasets/Testing/TemporalStacks/PLOT-00006")
    model_path = Path("../Models/best_unet_diff.pth")
    save_dir = Path("../Docs/Diagrams")
    save_dir.mkdir(exist_ok=True)

    # Get event date from mask
    mask_file = next(plot_dir.glob("Masks/*.tif"))
    event_date = datetime.strptime(mask_file.stem, "%Y%m%d")
    print(f"Event Date: {event_date.strftime('%Y-%m-%d')}")

    # Load model
    device = get_device()
    print(f"Using device: {device.upper()}")

    model = UNetDiff()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    print("Model loaded successfully\n")

    # Load pre-event image
    pre_file = next(plot_dir.glob("Pre-event/*.npy"))
    print(f"Pre-event image: {pre_file.name}")

    # Get all post-event images after the event date
    post_files = []
    for post_file in sorted(plot_dir.glob("Post-event/*.npy")):
        post_date = datetime.strptime(post_file.stem.split('T')[0], "%Y%m%d")
        if post_date >= event_date:
            post_files.append(post_file)

    print(f"\nAnalyzing {len(post_files)} post-event images...")

    # Load true mask
    true_mask = load_mask(mask_file)

    # Store all results for comparison
    results = []
    best_dice = 0
    # best_prediction = None
    best_input = None
    detected_date = None

    with torch.no_grad():
        for post_file in post_files:
            post_date = datetime.strptime(post_file.stem.split('T')[0], "%Y%m%d")
            print(f"Processing: {post_file.name}")

            # Prepare input
            x = load_and_preprocess_data(pre_file, post_file)
            x = x.to(device)

            # Predict
            pred = model(x)
            pred = pred.cpu().numpy()[0, 0]
            pred_mask = (pred > 0.5).astype(np.float32)

            # Calculate metrics
            metrics = calculate_metrics(true_mask, pred_mask)

            # Store results
            results.append({
                'Date': post_date.strftime('%Y-%m-%d'),
                **metrics
            })

            if metrics['Dice Score'] > best_dice:
                best_dice = metrics['Dice Score']
                # best_prediction = pred_mask
                best_input = np.load(post_file)[0]
                detected_date = post_date

    # Print results table
    text = "Results for All Post-Event Images"
    print(f"\n{text}\n{'=' * len(text)}")
    headers = ['Date', 'Dice Score', 'IoU Score', 'Precision', 'Recall', 'F1 Score']
    table_data = [[r['Date']] + [f"{r[h]:.3f}" for h in headers[1:]] for r in results]
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Print detection date
    text = "Detection Date"
    print(f"\n{text}\n{'-' * len(text)}")
    print(f"Date: {detected_date.strftime('%Y-%m-%d')}")
    print(f"Dice Score: {best_dice:.3f}")

    # Create visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot original image
    axes[0].imshow(best_input, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot true mask
    axes[1].imshow(true_mask, cmap='binary')
    axes[1].set_title(f'Mask with dice {best_dice:.3f}')
    axes[1].axis('off')

    # Plot predicted mask
    # axes[2].imshow(best_prediction, cmap='binary')
    # axes[2].set_title(f'Predicted Mask\nDice: {best_dice:.3f}')
    # axes[2].axis('off')

    plt.tight_layout()
    save_path = save_dir / f'deforestation_prediction_{detected_date.strftime("%Y%m%d")}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nVisualization saved to: {save_path}")
    print("\nAnalysis complete!")

# ------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    predict_and_visualize()
