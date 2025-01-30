#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal Image Stacking Script
------------------------------
Uses load_geojson_dates() to read the latest .geojson file in a 'Samples' folder,
extracting event dates for each PLOT-xxxxx. Organizes .npy files from 'Tiles' into
Pre-event or Post-event subfolders.

Author: Azhar Muhammed
Date: December 2024
"""


# -----------------------------------------------------------------------------
# Essential Imports
# -----------------------------------------------------------------------------
import os
import re
import json
import shutil
import numpy as np
from glob import glob
from datetime import datetime

# -----------------------------------------------------------------------------
# Local Imports
# -----------------------------------------------------------------------------
from helper import setup_logging

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TILES_DIR = os.path.normpath("../Datasets/Testing/Tiles")
SAMPLES_DIR = os.path.normpath("../Datasets/Testing/Samples")
OUTPUT_BASE_DIR = os.path.normpath("../Datasets/Testing/TemporalStacks")

# Regex to help identify date substrings like 20190606T083601 in folder name
FOLDER_DATE_PATTERN = r"\d{8}T\d{6}"
DATE_FORMAT = "%Y%m%dT%H%M%S"

filename = os.path.splitext(os.path.basename(__file__))[0]
# Set up logging using the imported function
logger, file_logger = setup_logging(file=filename)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def load_geojson_dates(samples_data_dir, print_loading=False):
    """
    Load event dates from the most recent .geojson file in 'samples_data_dir'.
    Returns a dictionary of { "00007": datetime_object, ... } with no 'PLOT-' prefix.
    """
    sample_files = glob(os.path.join(samples_data_dir, '*.geojson'))
    if not sample_files:
        raise FileNotFoundError("No .geojson files found in the Samples directory.")
    
    latest_file = max(sample_files, key=os.path.getctime)
    if print_loading:
        logger.info(f"Loading events from {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    event_dates = {}
    for feature in data.get('features', []):
        plot_id_raw = feature['properties']['name'].replace('PLOT-', '').strip()
        event_date_str = feature['properties'].get('img_date')
        if event_date_str:
            dt_obj = datetime.strptime(event_date_str, "%Y-%m-%d")
            event_dates[plot_id_raw] = dt_obj
    
    return event_dates


def parse_date_from_folder(folder_name):
    """
    Extract the date (YYYYmmddTHHMMSS) from the folder name using a regex.
    Returns a datetime object, or None if not found.
    """
    match = re.search(FOLDER_DATE_PATTERN, folder_name)
    if match:
        date_str = match.group(0)  # e.g. 20190606T083601
        try:
            return datetime.strptime(date_str, DATE_FORMAT)
        except ValueError as e:
            logger.error(f"Error parsing date from folder {folder_name}: {e}")
    return None


def parse_plot_id(file_name):
    """
    Extract the plot ID from the file name, e.g. "PLOT-00007.npy" -> "00007".
    Returns an empty string if not found.
    """
    match = re.search(r"PLOT-(\d+)", file_name)
    if match:
        return match.group(1)  # e.g. '00007'
    return ""

def update_stack_info_json(
    json_path: str,
    dest_filename: str,
    event_date: datetime,
    stack_type: str,
    tile_date: datetime = None,
    tile_array_path: str = None
):
    """
    Append 'file_name' to the 'files' list in stack_info.json. Creates or updates it.
    Also store 'event_date' and 'stack_type'.
    """
    data = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"JSON decode error in {json_path}; overwriting with new data.")
    
    # Ensure minimal structure
    if "num_images" not in data:
        data["num_images"] = 0
    if "event_date" not in data:
        data["event_date"] = event_date.strftime("%Y-%m-%d")
    if "stack_type" not in data:
        data["stack_type"] = stack_type
    if "image_dates" not in data:
        data["image_dates"] = []
    if "files" not in data:
        data["files"] = []
    if "shapes" not in data:
        data["shapes"] = []

    if dest_filename not in data["files"]:
        data["files"].append(dest_filename)
        data["num_images"] += 1

        if tile_date:
            data["image_dates"].append(tile_date.strftime("%Y-%m-%d"))

        if tile_array_path and os.path.exists(tile_array_path):
            arr = np.load(tile_array_path, mmap_mode='r')
            data["shapes"].append(arr.shape)
        else:
            data["shapes"].append(None)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------
def main():
    """ Single pass from Tiles -> TemporalStacks, using event dates from the latest .geojson. """
    try:
        event_dates = load_geojson_dates(SAMPLES_DIR, print_loading=True)
        logger.info(f"Loaded event dates for {len(event_dates)} plots.")
    except FileNotFoundError as err:
        logger.error(err)
        return

    for root, dirs, files in os.walk(TILES_DIR):
        folder_name = os.path.basename(root)
        folder_date = parse_date_from_folder(folder_name)
        if folder_date is None:
            continue
        
        for file_name in files:
            if not file_name.lower().endswith(".npy"):
                continue
            
            numeric_id = parse_plot_id(file_name)
            if not numeric_id:
                logger.warning(f"Could not parse plot ID in {file_name}, skipping.")
                continue
            
            event_dt = event_dates.get(numeric_id)
            if not event_dt:
                logger.warning(f"No event date found for PLOT-{numeric_id}, skipping file {file_name}.")
                continue

            is_pre_event = folder_date < event_dt
            subfolder_name = "Pre-event" if is_pre_event else "Post-event"
            stack_type = "Pre" if is_pre_event else "Post"

            plot_dir = os.path.join(OUTPUT_BASE_DIR, f"PLOT-{numeric_id}")
            target_dir = os.path.join(plot_dir, subfolder_name)
            os.makedirs(target_dir, exist_ok=True)

            date_str = folder_date.strftime(DATE_FORMAT)
            dest_filename = f"{date_str}.npy"
            src_path = os.path.join(root, file_name)
            dest_path = os.path.join(target_dir, dest_filename)

            shutil.copy2(src_path, dest_path)
            logger.info(f"Copied {file_name} => {dest_filename} in {target_dir}")

            json_path = os.path.join(target_dir, "stack_info.json")
            update_stack_info_json(
                json_path=json_path,
                dest_filename=dest_filename,
                event_date=event_dt,
                stack_type=stack_type,
                tile_date=folder_date,
                tile_array_path=dest_path
            )

    logger.info("Completed organizing .npy files into TemporalStacks.")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        raise