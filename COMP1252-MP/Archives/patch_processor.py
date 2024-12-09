import rasterio
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import os
from pathlib import Path
import logging
from PIL import Image
import re

class SentinelPatchProcessor:
    def __init__(self, patch_size=224):
        self.patch_size = patch_size
        logging.basicConfig(level=logging.INFO)
        
    def load_bands(self, sentinel_path):
        """Load and stack Sentinel-2 bands"""
        band_paths = Path(sentinel_path).glob('GRANULE/*/IMG_DATA/*.jp2')
        band_data = {}
        required_bands = ['B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12']
        
        for band_path in band_paths:
            band_name = re.search(r'B\d{2}|B8A', band_path.name)
            if band_name:
                band_name = band_name.group(0)
                if band_name in required_bands:
                    with rasterio.open(band_path) as src:
                        band_data[band_name] = src.read(1)
                        if band_name == 'B02':
                            self.meta = src.meta.copy()
                        logging.info(f"Loaded band {band_name}")
        
        if len(band_data) != len(required_bands):
            missing_bands = set(required_bands) - set(band_data.keys())
            logging.error(f"Processing failed: {missing_bands}")
            raise ValueError(f"Missing required bands: {missing_bands}")
            
        return band_data

    # Rest of the class implementation remains the same
    def compute_indices(self, band_data):
        """Compute NDVI and NDMI indices"""
        ndvi = (band_data['B08'] - band_data['B04']) / (band_data['B08'] + band_data['B04'])
        ndmi = (band_data['B8A'] - band_data['B11']) / (band_data['B8A'] + band_data['B11'])
        return ndvi, ndmi

    def resample_to_10m(self, band_data):
        """Resample all bands to 10m resolution"""
        target_shape = band_data['B02'].shape
        for band in ['B8A', 'B11', 'B12']:
            if band_data[band].shape != target_shape:
                band_data[band] = self._resample_array(
                    band_data[band],
                    target_shape
                )
        return band_data

    def _resample_array(self, array, target_shape):
        """Helper function to resample arrays"""
        return np.array(Image.fromarray(array).resize(
            (target_shape[1], target_shape[0]),
            resample=Image.BILINEAR
        ))

    def create_patches(self, stacked_bands, geojson_path, output_dir):
        """Create and save image patches"""
        gdf = gpd.read_file(geojson_path)
        gdf = gdf.to_crs(self.meta['crs'])
        
        for idx, geometry in enumerate(gdf.geometry):
            try:
                bounds = geometry.bounds
                window = rasterio.windows.from_bounds(
                    *bounds,
                    transform=self.meta['transform']
                )
                
                patch = stacked_bands[
                    :,
                    int(window.row_off):int(window.row_off + self.patch_size),
                    int(window.col_off):int(window.col_off + self.patch_size)
                ]
                
                if patch.shape[1:] == (self.patch_size, self.patch_size):
                    self._save_patch(patch, idx, output_dir)
                    
            except Exception as e:
                logging.error(f"Error processing patch {idx}: {str(e)}")
                
    def _save_patch(self, patch, idx, output_dir):
        """Save individual patches"""
        output_path = Path(output_dir) / f"patch_{idx}.npy"
        np.save(output_path, patch)
        logging.info(f"Saved patch {idx} to {output_path}")
        
    def process_imagery(self, sentinel_path, geojson_path, output_dir):
        """Main processing pipeline"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Processing imagery from {sentinel_path}")
            
            band_data = self.load_bands(sentinel_path)
            band_data = self.resample_to_10m(band_data)
            
            ndvi, ndmi = self.compute_indices(band_data)
            
            stacked_bands = np.stack([
                band_data['B02'], band_data['B03'], band_data['B04'],
                band_data['B08'], band_data['B8A'], band_data['B11'],
                band_data['B12'], ndvi, ndmi
            ])
            
            self.create_patches(stacked_bands, geojson_path, output_dir)
            logging.info("Processing completed successfully")
            
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
