import rasterio
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from rasterio.windows import from_bounds
import re
from pathlib import Path
import logging
import os
from PIL import Image
from tqdm.notebook import tqdm

class SentinelPatchProcessor:
    def __init__(self, patch_size=224):
        """
        Initialize the Sentinel-2 patch processor.
        
        Args:
            patch_size (int): Size of the output patches (default: 224)
        """
        self.patch_size = patch_size
        logging.basicConfig(
            level=logging.INFO,
            filename='processing.log',
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
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
        tile_geometries = gdf[mask].copy()
        
        # Clip geometries to tile bounds
        tile_geometries['geometry'] = tile_geometries.geometry.intersection(tile_bounds)
        
        return tile_geometries if not tile_geometries.empty else None

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
        Compute NDVI and NDMI indices from Sentinel-2 bands
        """
        try:
            # Validate required bands
            required_bands = ['B04', 'B08', 'B8A', 'B11']
            self.validate_bands(band_data, required_bands)
            
            # Calculate NDVI
            nir_red_sum = band_data['B08'] + band_data['B04']
            nir_red_diff = band_data['B08'] - band_data['B04']
            
            # Handle division by zero
            ndvi = np.where(
                nir_red_sum != 0,
                nir_red_diff / nir_red_sum,
                0  # Set to 0 where sum is 0
            )
            
            # Calculate NDMI
            nir_swir_sum = band_data['B8A'] + band_data['B11']
            nir_swir_diff = band_data['B8A'] - band_data['B11']
            
            # Handle division by zero
            ndmi = np.where(
                nir_swir_sum != 0,
                nir_swir_diff / nir_swir_sum,
                0  # Set to 0 where sum is 0
            )
            
            # Add bounds to prevent extreme values
            ndvi = np.clip(ndvi, -1, 1)
            ndmi = np.clip(ndmi, -1, 1)
            
            logging.info(f"Successfully computed NDVI and NDMI indices")
            logging.debug(f"NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
            logging.debug(f"NDMI range: [{ndmi.min():.3f}, {ndmi.max():.3f}]")
            
            return ndvi, ndmi
            
        except Exception as e:
            logging.error(f"Error computing indices: {str(e)}")
            raise

    def create_patches(self, stacked_bands, geometries, output_dir):
        """Create and save image patches for each geometry"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for idx, geometry in geometries.iterrows():
                try:
                    bounds = geometry.geometry.bounds
                    window = from_bounds(*bounds, transform=self.meta['transform'])
                    
                    patch = stacked_bands[
                        :,
                        int(window.row_off):int(window.row_off + self.patch_size),
                        int(window.col_off):int(window.col_off + self.patch_size)
                    ]
                    
                    if patch.shape[1:] == (self.patch_size, self.patch_size):
                        output_path = Path(output_dir) / f"patch_{idx}.npy"
                        np.save(output_path, patch)
                        logging.info(f"Saved patch {idx} to {output_path}")
                    else:
                        logging.warning(f"Skipping patch {idx} due to incorrect size: {patch.shape[1:]}")
                        
                except Exception as e:
                    logging.error(f"Error processing patch {idx}: {str(e)}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error in create_patches: {str(e)}")
            raise

    def process_imagery(self, sentinel_path, geojson_path, output_dir):
        """Process imagery considering tile boundaries"""
        try:
            tile_id = self.get_tile_id(sentinel_path)
            if not tile_id:
                logging.error(f"Could not determine tile ID for {sentinel_path}")
                return

            # Group geometries by tile
            tile_geometries = self.group_geometries_by_tile(geojson_path, sentinel_path)
            if tile_geometries is None:
                logging.info(f"No geometries intersect with tile {tile_id}")
                return

            # Create output directory for this tile
            tile_output_dir = Path(output_dir) / tile_id
            os.makedirs(tile_output_dir, exist_ok=True)

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
            
            self.create_patches(stacked_bands, tile_geometries, tile_output_dir)
            logging.info(f"Successfully processed tile {tile_id}")
            
        except Exception as e:
            logging.error(f"Error processing tile {tile_id}: {str(e)}")
            raise
