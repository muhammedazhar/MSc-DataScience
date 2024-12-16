#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentinel-2 Image Processor
--------------------------
This script processes Sentinel-2 satellite imagery to create image patches
based on GeoJSON geometries. It handles band loading, resampling, and index
computation.

Author: Azhar Muhammed
Date: December 2024
"""

import os
import gc
import re
import logging
from pathlib import Path
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.windows import from_bounds
from shapely.geometry import box
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../Docs/Logs/patcher.log'),
        logging.StreamHandler()
    ]
)

class SentinelPatchProcessor:
    def __init__(self, patch_size=224):
        """
        Initialize the Sentinel-2 patch processor.

        Args:
            patch_size (int): Size of the output patches (default: 224)
        """
        self.patch_size = patch_size

    def get_tile_id(self, sentinel_path):
        """Extract tile ID from Sentinel path"""
        match = re.search(r'T\d{2}[A-Z]{3}', str(sentinel_path))
        return match.group(0) if match else None

    def get_tile_bounds(self, sentinel_path):
        """Get the geographical bounds of a Sentinel tile"""
        sample_band = next(Path(sentinel_path).glob('GRANULE/*/IMG_DATA/*B02.jp2'))
        with rasterio.open(sample_band) as src:
            bounds = box(*src.bounds)
        return bounds

    def group_geometries_by_tile(self, geojson_path, sentinel_path):
        """Group geometries based on which tile they intersect with"""
        gdf = gpd.read_file(geojson_path)
        tile_bounds = self.get_tile_bounds(sentinel_path)
        tile_id = self.get_tile_id(sentinel_path)

        # Transform geometries to tile CRS if needed
        with rasterio.open(next(Path(sentinel_path).glob('GRANULE/*/IMG_DATA/*B02.jp2'))) as src:
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

        # Filter geometries that intersect with this tile
        mask = gdf.geometry.intersects(tile_bounds)
        intersecting_count = mask.sum()

        if intersecting_count == 0:
            logging.debug(f"Tile {tile_id} has no intersecting geometries out of {len(gdf)} total geometries")
            return None

        # Clip geometries to tile bounds
        tile_geometries = gdf[mask].copy()
        tile_geometries['geometry'] = tile_geometries.geometry.intersection(tile_bounds)

        logging.debug(f"Tile {tile_id} has {len(tile_geometries)} intersecting geometries")
        return tile_geometries

    def load_bands(self, sentinel_path):
        """Load and stack Sentinel-2 bands"""
        band_paths = list(Path(sentinel_path).glob('GRANULE/*/IMG_DATA/*.jp2'))
        band_data = {}
        required_bands = ['B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12']

        for band_path in band_paths:
            band_name = re.search(r'B\d{2}|B8A', band_path.name)
            if band_name:
                band_name = band_name.group(0)
                if band_name in required_bands:
                    try:
                        with rasterio.open(band_path) as src:
                            band_data[band_name] = src.read(1)
                            if band_name == 'B02':
                                self.meta = src.meta.copy()
                            logging.info(f"Loaded band {band_name} from {self.get_tile_id(sentinel_path)}")
                    except Exception as e:
                        logging.error(f"Error loading {band_name} from {sentinel_path}: {str(e)}")

        if len(band_data) != len(required_bands):
            missing_bands = set(required_bands) - set(band_data.keys())
            logging.error(f"Missing required bands for {self.get_tile_id(sentinel_path)}: {missing_bands}")
            return None

        return band_data

    def validate_bands(self, band_data, required_bands):
        """Validate that all required bands are present"""
        missing_bands = [band for band in required_bands if band not in band_data]
        if missing_bands:
            raise ValueError(f"Missing required bands: {missing_bands}")

    def resample_to_10m(self, band_data):
        """Resample all bands to 10m resolution"""
        try:
            # Get shape from a 10m band (B02)
            target_shape = band_data['B02'].shape
            logging.debug(f"Target shape for resampling: {target_shape}")

            # Bands that need resampling (20m bands)
            bands_to_resample = ['B8A', 'B11', 'B12']

            for band in bands_to_resample:
                if band in band_data and band_data[band].shape != target_shape:
                    logging.info(f"Resampling {band} to 10m resolution")
                    band_data[band] = self._resample_array(
                        band_data[band],
                        target_shape
                    )
                    logging.debug(f"Resampled {band} shape: {band_data[band].shape}")

            return band_data

        except Exception as e:
            logging.error(f"Error in resample_to_10m: {str(e)}")
            raise

    def _resample_array(self, array, target_shape):
        """Helper function to resample arrays using bilinear interpolation"""
        try:
            # Convert array to PIL Image for resampling
            img = Image.fromarray(array)

            # Resize to target shape (note the order: width, height)
            resized = img.resize(
                (target_shape[1], target_shape[0]),  # PIL uses (width, height)
                resample=Image.BILINEAR
            )

            # Convert back to numpy array
            return np.array(resized)

        except Exception as e:
            logging.error(f"Error in _resample_array: {str(e)}")
            raise

    def compute_indices(self, band_data):
        """
        Compute NDVI and NDMI indices from Sentinel-2 bands with safe division
        """
        try:
            # Validate required bands
            required_bands = ['B04', 'B08', 'B8A', 'B11']
            self.validate_bands(band_data, required_bands)

            # Calculate NDVI safely
            nir_red_sum = band_data['B08'] + band_data['B04']
            nir_red_diff = band_data['B08'] - band_data['B04']

            # Use np.divide with where condition to handle zeros
            ndvi = np.divide(
                nir_red_diff,
                nir_red_sum,
                out=np.zeros_like(nir_red_diff, dtype=np.float32),
                where=nir_red_sum != 0
            )

            # Calculate NDMI safely
            nir_swir_sum = band_data['B8A'] + band_data['B11']
            nir_swir_diff = band_data['B8A'] - band_data['B11']

            # Use np.divide with where condition to handle zeros
            ndmi = np.divide(
                nir_swir_diff,
                nir_swir_sum,
                out=np.zeros_like(nir_swir_diff, dtype=np.float32),
                where=nir_swir_sum != 0
            )

            # Add bounds to prevent extreme values
            ndvi = np.clip(ndvi, -1, 1)
            ndmi = np.clip(ndmi, -1, 1)

            # Replace NaN values with 0
            ndvi = np.nan_to_num(ndvi, nan=0.0)
            ndmi = np.nan_to_num(ndmi, nan=0.0)

            logging.info(f"Successfully computed NDVI and NDMI indices")
            logging.debug(f"NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
            logging.debug(f"NDMI range: [{ndmi.min():.3f}, {ndmi.max():.3f}]")

            return ndvi, ndmi

        except Exception as e:
            logging.error(f"Error computing indices: {str(e)}")
            raise

    def create_patches(self, stacked_bands, geometries, output_dir):
        """
        Create and save image patches for each geometry using the 'name' property.
        Only add _P{number} suffix when multiple patches exist for the same name.
        """
        try:
            # Convert output_dir to absolute path
            output_dir = Path(output_dir).resolve()
            os.makedirs(output_dir, exist_ok=True)

            # First, count total patches for each name
            name_total_patches = geometries['name'].value_counts().to_dict()
            current_patch_num = {}

            # Add debug logging
            logging.info(f"Output directory: {output_dir}")
            logging.info(f"Total patches per name: {name_total_patches}")

            for idx, geometry in geometries.iterrows():
                try:
                    plot_name = geometry['name']

                    if plot_name not in current_patch_num:
                        current_patch_num[plot_name] = 1
                    else:
                        current_patch_num[plot_name] += 1

                    if name_total_patches[plot_name] > 1:
                        patch_name = f"{plot_name}_P{current_patch_num[plot_name]}"
                    else:
                        patch_name = plot_name

                    # Create the full output path
                    output_path = output_dir / f"{patch_name}.npy"
                    logging.debug(f"Attempting to save to: {output_path}")

                    bounds = geometry.geometry.bounds
                    window = from_bounds(*bounds, transform=self.meta['transform'])

                    patch = stacked_bands[
                        :,
                        int(window.row_off):int(window.row_off + self.patch_size),
                        int(window.col_off):int(window.col_off + self.patch_size)
                    ]

                    if patch.shape[1:] == (self.patch_size, self.patch_size):
                        np.save(output_path, patch)
                        logging.info(f"Saved patch {patch_name} to {output_path}")
                    else:
                        logging.warning(f"Skipping patch {patch_name} due to incorrect size: {patch.shape[1:]}")

                except KeyError as ke:
                    logging.error(f"'name' property not found in geometry at index {idx}: {str(ke)}")
                    continue
                except Exception as e:
                    logging.error(f"Error processing patch for geometry at index {idx}: {str(e)}")
                    logging.error(f"Full error: {str(e)}", exc_info=True)
                    continue

        except Exception as e:
            logging.error(f"Error in create_patches: {str(e)}")
            raise

    def process_imagery(self, sentinel_path, geojson_path, output_dir):
        """Process imagery considering tile boundaries"""
        try:
            # Input validation
            if not os.path.exists(sentinel_path):
                raise FileNotFoundError(f"Sentinel path does not exist: {sentinel_path}")
            if not os.path.exists(geojson_path):
                raise FileNotFoundError(f"GeoJSON path does not exist: {geojson_path}")
            if not os.path.isdir(sentinel_path):
                raise ValueError(f"Sentinel path must be a directory: {sentinel_path}")

            tile_id = self.get_tile_id(sentinel_path)
            if not tile_id:
                logging.error(f"Could not determine tile ID for {sentinel_path}")
                return

            # Get geometries (we already know they exist from main())
            tile_geometries = self.group_geometries_by_tile(geojson_path, sentinel_path)

            # Load and process bands
            band_data = self.load_bands(sentinel_path)
            if band_data is None:
                return

            band_data = self.resample_to_10m(band_data)
            ndvi, ndmi = self.compute_indices(band_data)

            stacked_bands = np.stack([
                band_data['B02'], band_data['B03'], band_data['B04'],
                band_data['B08'], band_data['B8A'], band_data['B11'],
                band_data['B12'], ndvi, ndmi
            ])

            self.create_patches(stacked_bands, tile_geometries, output_dir)
            logging.info(f"Successfully processed tile {tile_id}")

        except Exception as e:
            logging.error(f"Error processing tile {tile_id}: {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        # Initialize processor
        processor = SentinelPatchProcessor(patch_size=224)

        # Get latest geojson
        samples_path = Path("../Datasets/Testing/Samples")
        geojson_files = list(samples_path.glob("*.geojson"))
        if not geojson_files:
            raise FileNotFoundError(f"No .geojson files found in {samples_path}")
        latest_geojson = max(geojson_files, key=lambda p: p.stat().st_mtime)

        # Get all .SAFE directories
        base_path = Path("../Datasets/Sentinel-2")
        safe_dirs = list(base_path.glob("*/*.SAFE"))

        # Track processing statistics
        processed_count = 0
        error_count = 0
        skipped_count = 0

        # Process each Sentinel-2 scene
        for safe_dir in tqdm(safe_dirs, desc="Processing Sentinel-2 images"):
            try:
                # Check if there are intersecting geometries first
                tile_geometries = processor.group_geometries_by_tile(
                    str(latest_geojson),
                    str(safe_dir)
                )

                if tile_geometries is None or tile_geometries.empty:
                    skipped_count += 1
                    continue

                # Only create output directory if we have geometries to process
                output_dir = Path("../Datasets/Testing/Tiles") / safe_dir.parent.name

                # Check if already processed
                if output_dir.exists() and list(output_dir.glob("*.npy")):
                    logging.info(f"Skipping {safe_dir.name} - already processed")
                    continue

                # Create directory only when needed
                output_dir.mkdir(parents=True, exist_ok=True)

                # Process the imagery
                processor.process_imagery(
                    sentinel_path=str(safe_dir),
                    geojson_path=str(latest_geojson),
                    output_dir=output_dir
                )
                processed_count += 1

            except Exception as e:
                logging.error(f"Failed to process {safe_dir}: {str(e)}", exc_info=True)
                error_count += 1
                continue

            # Clear memory
            gc.collect()

        logging.info(
            f"Processing complete. Processed: {processed_count}, "
            f"Skipped: {skipped_count}, Errors: {error_count}"
        )

    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}", exc_info=True)
        raise
