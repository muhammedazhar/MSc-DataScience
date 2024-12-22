import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from s2cloudless import S2PixelCloudDetector
from shapely.geometry import box
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('../Docs/Logs/main.log'), logging.StreamHandler()]
)

class SentinelPatchProcessor:
    def __init__(self, patch_size: int = 224, cloud_coverage_threshold: float = 0.3, max_workers: int = 4):
        """
        Initialize the Sentinel-2 patch processor with cloud filtering.
        
        Args:
            patch_size (int): Size of the output patches (default: 224)
            cloud_coverage_threshold (float): Maximum allowed cloud coverage (0.0 - 1.0)
            max_workers (int): Maximum number of parallel workers for processing
        """
        self.patch_size = patch_size
        self.cloud_coverage_threshold = cloud_coverage_threshold
        self.max_workers = max_workers
        self.cloud_detector = S2PixelCloudDetector(threshold=0.4)
        self.required_bands = ['B01', 'B02', 'B03', 'B04', 'B05',
                             'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        self.meta = None

    def get_tile_id(self, sentinel_path: str) -> Optional[str]:
        """Extract tile ID from Sentinel path"""
        match = re.search(r'T\d{2}[A-Z]{3}', str(sentinel_path))
        return match.group(0) if match else None

    def get_tile_bounds(self, sentinel_path: str) -> box:
        """Get the geographical bounds of a Sentinel tile"""
        sample_band = next(Path(sentinel_path).glob('GRANULE/*/IMG_DATA/*B02.jp2'))
        with rasterio.open(sample_band) as src:
            bounds = box(*src.bounds)
        return bounds

    def group_geometries_by_tile(self, geojson_path: str, sentinel_path: str) -> Optional[gpd.GeoDataFrame]:
        """Group geometries based on which tile they intersect with"""
        gdf = gpd.read_file(geojson_path)
        tile_bounds = self.get_tile_bounds(sentinel_path)
        tile_id = self.get_tile_id(sentinel_path)

        with rasterio.open(next(Path(sentinel_path).glob('GRANULE/*/IMG_DATA/*B02.jp2'))) as src:
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

        mask = gdf.geometry.intersects(tile_bounds)
        tile_geometries = gdf[mask].copy()
        tile_geometries['geometry'] = tile_geometries.geometry.intersection(tile_bounds)

        return tile_geometries if not tile_geometries.empty else None

    def load_bands(self, sentinel_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Load required Sentinel-2 bands in parallel"""
        band_paths = list(Path(sentinel_path).glob('GRANULE/*/IMG_DATA/*.jp2'))
        band_data = {}

        def load_band(band_path: Path) -> Tuple[str, np.ndarray]:
            band_name_match = re.search(r'B\d{2}|B8A', band_path.name)
            if not band_name_match:
                return None
            
            band_name = band_name_match.group(0)
            if band_name not in self.required_bands:
                return None

            try:
                with rasterio.open(band_path) as src:
                    data = src.read(1)
                    if band_name == 'B02' and self.meta is None:
                        self.meta = src.meta.copy()
                    return (band_name, data)
            except Exception as e:
                logging.error(f"Error loading {band_name}: {str(e)}")
                return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(load_band, path) for path in band_paths]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    band_name, data = result
                    band_data[band_name] = data

        missing_bands = set(self.required_bands) - set(band_data.keys())
        if missing_bands:
            logging.error(f"Missing required bands: {missing_bands}")
            return None

        return band_data

    def resample_to_10m(self, band_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Resample all bands to 10m resolution in parallel"""
        target_shape = band_data['B02'].shape

        def resample_band(band_tuple: Tuple[str, np.ndarray]) -> Tuple[str, np.ndarray]:
            band_name, band_array = band_tuple
            if band_array.shape != target_shape:
                return (band_name, self._resample_array(band_array, target_shape))
            return (band_name, band_array)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(resample_band, band_data.items()))
            
        return dict(results)

    def _resample_array(self, array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Helper function to resample arrays using bilinear interpolation"""
        img = Image.fromarray(array)
        resized = img.resize((target_shape[1], target_shape[0]), resample=Image.BILINEAR)
        return np.array(resized)

    def compute_indices(self, band_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute NDVI and NDMI indices from Sentinel-2 bands"""
        required_bands_indices = ['B04', 'B08', 'B8A', 'B11']
        self.validate_bands(band_data, required_bands_indices)

        nir_red_sum = band_data['B08'] + band_data['B04']
        nir_red_diff = band_data['B08'] - band_data['B04']
        ndvi = np.divide(nir_red_diff, nir_red_sum,
                        out=np.zeros_like(nir_red_diff, dtype=np.float32),
                        where=nir_red_sum != 0)

        nir_swir_sum = band_data['B8A'] + band_data['B11']
        nir_swir_diff = band_data['B8A'] - band_data['B11']
        ndmi = np.divide(nir_swir_diff, nir_swir_sum,
                        out=np.zeros_like(nir_swir_diff, dtype=np.float32),
                        where=nir_swir_sum != 0)

        return np.clip(ndvi, -1, 1), np.clip(ndmi, -1, 1)

    def validate_bands(self, band_data: Dict[str, np.ndarray], required_bands: List[str]) -> None:
        """Validate that all required bands are present"""
        missing_bands = [band for band in required_bands if band not in band_data]
        if missing_bands:
            raise ValueError(f"Missing required bands: {missing_bands}")

    def detect_clouds(self, band_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute cloud mask and cloud probability"""
        cloud_bands_order = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        stacked_cloud_bands = np.stack([band_data[b] for b in cloud_bands_order], axis=-1) / 10000.0
        cloud_probs = self.cloud_detector.get_cloud_probability_maps(np.expand_dims(stacked_cloud_bands, axis=0))[0]
        return cloud_probs > self.cloud_detector.threshold

    def create_patches(self, stacked_bands: np.ndarray, cloud_mask: np.ndarray,
                      geometries: gpd.GeoDataFrame, output_dir: str) -> None:
        """Create and save image patches for each geometry"""
        os.makedirs(output_dir, exist_ok=True)

        def process_geometry(geometry_tuple: Tuple[int, gpd.GeoSeries]) -> None:
            idx, geometry = geometry_tuple
            plot_name = geometry['name']
            windows = self._generate_windows(geometry.geometry, self.meta['transform'])

            for i, window in enumerate(windows, 1):
                row_off, col_off = int(window.row_off), int(window.col_off)
                height, width = int(window.height), int(window.width)

                patch = stacked_bands[:, row_off:row_off + height, col_off:col_off + width]
                cloud_patch = cloud_mask[row_off:row_off + height, col_off:col_off + width]

                if patch.shape[1:] != (self.patch_size, self.patch_size):
                    continue

                if np.mean(cloud_patch) > self.cloud_coverage_threshold:
                    continue

                output_filename = f"{plot_name}_P{i}.npy" if len(windows) > 1 else f"{plot_name}.npy"
                output_path = Path(output_dir) / output_filename
                np.save(output_path, patch)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(executor.map(process_geometry, geometries.iterrows()))

    def _generate_windows(self, geometry, transform) -> List[Window]:
        """Generate a list of windows covering the geometry"""
        minx, miny, maxx, maxy = geometry.bounds
        min_col, max_row = ~transform * (minx, miny)
        max_col, min_row = ~transform * (maxx, maxy)

        min_row, max_row = int(np.floor(min_row)), int(np.ceil(max_row))
        min_col, max_col = int(np.floor(min_col)), int(np.ceil(max_col))

        windows = []
        for row_off in range(min_row, max_row, self.patch_size):
            for col_off in range(min_col, max_col, self.patch_size):
                window = Window(col_off=col_off, row_off=row_off,
                              width=self.patch_size, height=self.patch_size)
                window_transform = rasterio.windows.transform(window, transform)
                window_bounds = rasterio.transform.array_bounds(self.patch_size,
                                                              self.patch_size,
                                                              window_transform)
                if geometry.intersects(box(*window_bounds)):
                    windows.append(window)

        return windows

    def process_imagery(self, sentinel_path: str, geojson_path: str, output_dir: str) -> None:
        """Process imagery considering tile boundaries and filter out cloudy patches"""
        tile_id = self.get_tile_id(sentinel_path)
        if not tile_id:
            logging.error(f"Could not determine tile ID for {sentinel_path}")
            return

        tile_geometries = self.group_geometries_by_tile(geojson_path, sentinel_path)
        if tile_geometries is None:
            logging.info(f"No geometries intersect with tile {tile_id}")
            return

        band_data = self.load_bands(sentinel_path)
        if band_data is None:
            return

        band_data = self.resample_to_10m(band_data)
        cloud_mask = self.detect_clouds(band_data)
        ndvi, ndmi = self.compute_indices(band_data)

        stacked_bands = np.stack([
            band_data['B02'], band_data['B03'], band_data['B04'],
            band_data['B08'], band_data['B8A'], band_data['B11'],
            band_data['B12'], ndvi, ndmi
        ])

        self.create_patches(stacked_bands, cloud_mask, tile_geometries, output_dir)

def main():
    # Get full path of current script
    current_file = __file__

    # Extract just the filename
    filename = os.path.basename(current_file)

    # If you need filename without extension
    filename_no_ext = os.path.splitext(filename)[0]
    logging.info(f"Running {filename_no_ext} script...")

    processor = SentinelPatchProcessor(patch_size=224, cloud_coverage_threshold=0.3)
    base_path = Path("../Datasets/Sentinel-2")
    output_base = Path("../Datasets/Testing/Processed")
    logging.info(f"Output base is set to {output_base}")
    safe_dirs = list(base_path.glob("*/*.SAFE"))
    logging.info(f"Found {len(safe_dirs)} .SAFE directories")

    if not safe_dirs:
        raise FileNotFoundError(f"No .SAFE directories found in {base_path}")

    results = {'total': len(safe_dirs), 'processed': 0, 'failed': 0}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for safe_dir in safe_dirs:
            product_name = safe_dir.parent.name
            output_dir = output_base / product_name
            sample_files = glob('../Datasets/Testing/Samples/*.geojson')
            
            if not sample_files:
                raise FileNotFoundError("No .geojson files found in Testing/Samples/")
            
            latest_file = max(sample_files, key=os.path.getctime)
            future = executor.submit(
                processor.process_imagery,
                str(safe_dir),
                latest_file,
                str(output_dir)
            )
            futures.append((future, product_name))

        for future, product_name in tqdm(futures, desc="Processing Sentinel-2 images"):
            try:
                future.result()
                results['processed'] += 1
                logging.info(f"Successfully processed {product_name}")
            except Exception as e:
                results['failed'] += 1
                logging.error(f"Failed to process {product_name}: {str(e)}")

    logging.info("\nProcessing Summary:")
    logging.info(f"Total scenes: {results['total']}")
    logging.info(f"Successfully processed: {results['processed']}")
    logging.info(f"Failed: {results['failed']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
