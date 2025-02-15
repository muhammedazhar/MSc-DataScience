#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mask Generator Script
---------------------
This script generates raster masks from GeoJSON features for forest change
detection. It processes temporal geometric data and creates corresponding mask
files in GeoTIFF format.

Author: Azhar Muhammed
Date: October 2024
"""

# -----------------------------------------------------------------------------
# Essential Imports
# -----------------------------------------------------------------------------
import os
import sys
from glob import glob
import geopandas as gpd
import rasterio
from rasterio import features

# -----------------------------------------------------------------------------
# Local Imports
# -----------------------------------------------------------------------------
from helper import setup_logging

# -----------------------------------------------------------------------------
# Constants and Configuration
# -----------------------------------------------------------------------------
filename = os.path.splitext(os.path.basename(__file__))[0]
# Set up logger using the imported function
logger, file_logger = setup_logging(file=filename)

RASTER_WIDTH = 224
RASTER_HEIGHT = 224
MASK_VALUE = 255
MASK_DTYPE = 'uint8'

# -----------------------------------------------------------------------------
# Mask Creation Function
# -----------------------------------------------------------------------------
def create_mask(geometry, width=RASTER_WIDTH, height=RASTER_HEIGHT, crs=None):
    """
    Create a raster mask from geometry.
    """
    try:
        minx, miny, maxx, maxy = geometry.bounds
        transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

        mask = features.rasterize(
            [(geometry, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=MASK_DTYPE
        )
        mask = mask * MASK_VALUE

        return mask, transform

    except Exception as e:
        logger.error(f"Error creating mask: {str(e)}")
        raise

# -----------------------------------------------------------------------------
# Metadata Generation Function
# -----------------------------------------------------------------------------
def get_output_metadata(height, width, crs, transform):
    """
    Generate raster metadata for output files.
    """
    return {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': MASK_DTYPE,
        'crs': crs,
        'transform': transform
    }

# -----------------------------------------------------------------------------
# GeoJSON Processing Function
# -----------------------------------------------------------------------------
def process_geojson(samples_data_dir, base_dir):
    """
    Process GeoJSON files and generate masks.
    """
    try:
        # Find most recent GeoJSON file
        sample_files = glob(os.path.join(samples_data_dir, '*.geojson'))
        if not sample_files:
            raise FileNotFoundError("No .geojson files found in the Samples directory.")

        latest_file = max(sample_files, key=os.path.getctime)
        logger.info(f"Processing GeoJSON file: {os.path.basename(latest_file)}")

        # Load and process GeoJSON
        gdf = gpd.read_file(latest_file)

        for idx, row in gdf.iterrows():
            name = row['name']
            date_str = row['img_date'].strftime('%Y%m%d')
            geometry = row['geometry']

            # Setup output directory
            mask_dir = os.path.join(base_dir, name, 'Masks')
            os.makedirs(mask_dir, exist_ok=True)
            mask_path = os.path.join(mask_dir, f"{date_str}.tif")

            # Generate and save mask
            mask, transform = create_mask(geometry, crs=gdf.crs)
            out_meta = get_output_metadata(RASTER_HEIGHT, RASTER_WIDTH, gdf.crs, transform)

            with rasterio.open(mask_path, 'w', **out_meta) as dest:
                dest.write(mask, 1)

            logger.info(f"Created mask for {name} - {row['img_date']} as {date_str}.tif file")

    except Exception as e:
        logger.error(f"Error processing GeoJSON: {str(e)}")
        raise

# -----------------------------------------------------------------------------
# Main Execution Function
# -----------------------------------------------------------------------------
def main():
    """Main execution function"""

    # Define directories
    samples_data_dir = "../Datasets/Testing/Samples"
    base_dir = '../Datasets/Testing/TemporalStacks'

    try:
        process_geojson(samples_data_dir, base_dir)
        logger.info("Mask generation completed successfully")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

# -----------------------------------------------------------------------------
# Script Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        raise
