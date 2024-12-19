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
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from skimage import exposure
import tensorflow as tf
from tensorflow.keras.models import load_model
import traceback

##############################################
# Configuration
##############################################
DATASET_DIR = '../Datasets/Testing/TemporalStack/'
PLOT_PREFIX = 'PLOT-'
PLOT_RANGE = range(1, 67)  # PLOT-00001 to PLOT-00066
IMG_SIZE = (224, 224)      # Height, Width
N_CHANNELS = 27            # 9 pre + 9 post + 9 diff
THRESHOLD = 0.5
MODEL_PATH = '../Models/unet_diff_model.h5'  # Path to your trained U-Net model
OUTPUT_DIR = '../Predictions/'


##############################################
# Utility Functions
##############################################
def safe_load_npy(file_path):
    """
    Safely load a .npy file and handle errors.
    """
    try:
        arr = np.load(file_path)
        return arr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_temporal_pairs(pre_dir, post_dir):
    """
    Get aligned pre-event and post-event image pairs based on temporal proximity (5-30 days apart).
    This function assumes that filenames contain date information (YYYYMMDDTHHMMSS).
    You will need to adapt this logic to your dataset's actual temporal alignment rules.

    Returns a list of tuples: (pre_image_path, post_image_path, mask_date)
    """
    pre_files = sorted(glob.glob(os.path.join(pre_dir, '*.npy')))
    post_files = sorted(glob.glob(os.path.join(post_dir, '*.npy')))

    # Extract dates from filenames (example: 20180726T084009 -> 20180726)
    def extract_date(fname):
        base = os.path.basename(fname)
        date_str = base.split('T')[0]
        return date_str

    pairs = []
    for pre_file in pre_files:
        pre_date = extract_date(pre_file)
        pre_dt = np.datetime64(pre_date)

        # Find a suitable post-event file within 5 to 30 days after pre_date
        for post_file in post_files:
            post_date = extract_date(post_file)
            post_dt = np.datetime64(post_date)
            delta = (post_dt - pre_dt).astype(int)

            if 5 <= delta <= 30:
                # Found a suitable pair
                # Use the post_date as mask_date or choose the known deforestation event date
                mask_date = post_date + '.tif'  # Example: 20180731.tif
                pairs.append((pre_file, post_file, mask_date))
                break

    return pairs


def load_mask(mask_dir, mask_name):
    """
    Load a mask GeoTIFF file and return as a binary numpy array.
    Assumes single-band binary mask.
    """
    mask_path = os.path.join(mask_dir, mask_name)
    if not os.path.exists(mask_path):
        print(f"Mask file {mask_path} not found.")
        return None
    
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
        # Ensure binary: Assuming values >0 indicate deforestation
        mask_bin = (mask_data > 0).astype(np.uint8)
    return mask_bin


def histogram_match(source, reference):
    """
    Apply histogram matching to source image to match the reference image distribution.
    Both should be (H, W, C).
    """
    matched = np.zeros_like(source, dtype=source.dtype)
    for c in range(source.shape[-1]):
        matched[..., c] = exposure.match_histograms(source[..., c], reference[..., c])
    return matched


def preprocess_image(pre_img, post_img):
    """
    Preprocess and stack pre, post, and diff images:
    1. Ensure both images are scaled by 1/10000.
    2. Apply histogram matching on post_img to match pre_img distribution.
    3. Compute diff_img = post_img - pre_img.
    4. Concatenate to form a 27-channel image.
    """
    # Scale
    pre_scaled = pre_img.astype(np.float32) / 10000.0
    post_scaled = post_img.astype(np.float32) / 10000.0

    # Histogram matching (post to pre)
    post_matched = histogram_match(post_scaled, pre_scaled)

    # Compute difference
    diff_img = post_matched - pre_scaled

    # Stack: [pre (9), post (9), diff (9)]
    combined = np.concatenate([pre_scaled, post_matched, diff_img], axis=-1)
    return combined


def load_and_preprocess_triplet(pre_file, post_file, mask_file, mask_dir):
    """
    Load pre-event, post-event, and mask data, then preprocess.
    Returns:
        combined (H, W, 27)
        mask (H, W)
    """
    pre_img = safe_load_npy(pre_file)
    post_img = safe_load_npy(post_file)

    # Basic checks
    if pre_img is None or post_img is None:
        raise ValueError("Pre or Post image could not be loaded.")

    if pre_img.shape[:2] != post_img.shape[:2]:
        raise ValueError("Pre and Post images do not have the same spatial dimensions.")

    # Load mask
    mask = load_mask(mask_dir, mask_file)
    if mask is None:
        # If no mask is found, return None or handle differently
        mask = np.zeros(pre_img.shape[:2], dtype=np.uint8)

    # Ensure correct size (224x224) - for inference, you might crop or pad if needed
    # Here we assume images are already 224x224. If not, implement resize/crop logic.
    if pre_img.shape[:2] != IMG_SIZE:
        # Resize or crop, here we just raise an error for simplicity
        raise ValueError(f"Images are not {IMG_SIZE}. Got {pre_img.shape[:2]}")

    # Preprocess
    combined = preprocess_image(pre_img, post_img)

    return combined, mask


def save_prediction(pred_mask, out_path):
    """
    Save prediction mask as a .npy file or a GeoTIFF if desired.
    For demonstration, we save as .npy.
    """
    np.save(out_path, pred_mask)


##############################################
# U-Net Model (Load or define)
##############################################
# Assuming a U-Net model saved at MODEL_PATH
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)
model.summary()


##############################################
# Inference Pipeline
##############################################
def run_inference_on_dataset():
    """
    Run inference on the entire dataset (PLOT-00001 to PLOT-00066).
    Processes each temporal pair, produces predictions, and saves results.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for plot_id in PLOT_RANGE:
        plot_name = f"{PLOT_PREFIX}{plot_id:05d}"
        plot_dir = os.path.join(DATASET_DIR, plot_name)
        if not os.path.exists(plot_dir):
            print(f"Plot directory {plot_dir} does not exist, skipping.")
            continue

        pre_dir = os.path.join(plot_dir, "Pre-event")
        post_dir = os.path.join(plot_dir, "Post-event")
        mask_dir = os.path.join(plot_dir, "Masks")

        if not os.path.exists(pre_dir) or not os.path.exists(post_dir) or not os.path.exists(mask_dir):
            print(f"Missing subdirectories in {plot_dir}, skipping.")
            continue

        try:
            pairs = get_temporal_pairs(pre_dir, post_dir)
            if not pairs:
                print(f"No valid temporal pairs found in {plot_dir}, skipping.")
                continue

            for (pre_file, post_file, mask_file) in pairs:
                try:
                    combined, true_mask = load_and_preprocess_triplet(pre_file, post_file, mask_file, mask_dir)
                    # combined shape: (224,224,27)
                    # Expand dims for model (batch size = 1)
                    input_tensor = np.expand_dims(combined, axis=0)  # (1,224,224,27)

                    # Predict
                    preds = model.predict(input_tensor, verbose=0)
                    # preds shape: (1,224,224,1)
                    pred_mask = (preds[0,...,0] > THRESHOLD).astype(np.uint8)

                    # Save prediction
                    base_name = os.path.splitext(os.path.basename(mask_file))[0]
                    out_path = os.path.join(OUTPUT_DIR, f"{plot_name}_{base_name}_pred.npy")
                    save_prediction(pred_mask, out_path)
                    print(f"Saved prediction: {out_path}")

                except Exception as e:
                    print(f"Error processing {pre_file} and {post_file} for {plot_name}: {e}")
                    traceback.print_exc()

        except Exception as e:
            print(f"Error processing plot {plot_name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    run_inference_on_dataset()