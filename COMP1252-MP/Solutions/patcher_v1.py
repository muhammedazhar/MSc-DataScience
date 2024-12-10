import logging
import os
import re
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from shapely.geometry import box
from tqdm import tqdm

logging.basicConfig(
    filename=Path("../Docs/Logs/main.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SentinelPatchProcessor:
    def __init__(self, patch_size: int = 224, max_workers: int = 4):
        self.patch_size = patch_size
        self.max_workers = max_workers
        self.required_bands = ['B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12']
        self.meta = None

    def _resample_array(self, array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Fixed resampling function with proper image mode handling"""
        # Convert to float32 to preserve precision
        array = array.astype(np.float32)
        
        # Normalize to 0-255 range for PIL
        min_val, max_val = array.min(), array.max()
        if max_val > min_val:
            normalized = ((array - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
        else:
            normalized = np.zeros(array.shape, dtype=np.uint8)

        # Create PIL image and resize
        img = Image.fromarray(normalized, mode='L')
        resized = img.resize(
            (target_shape[1], target_shape[0]), 
            resample=Image.BILINEAR
        )
        
        # Convert back to original scale
        result = np.array(resized, dtype=np.float32)
        if max_val > min_val:
            result = (result * (max_val - min_val) / 255 + min_val)
        
        return result

    def process_batch(self, items: List[Tuple[Path, str]], output_base: Path, geojson_path: str) -> Dict[str, int]:
        """Process a batch of Sentinel images"""
        results = {'processed': 0, 'failed': 0}
        
        for safe_dir, product_name in items:
            try:
                output_dir = output_base / product_name
                self.process_imagery(str(safe_dir), geojson_path, str(output_dir))
                results['processed'] += 1
                logging.info(f"Successfully processed {product_name}")
            except Exception as e:
                results['failed'] += 1
                logging.error(f"Failed to process {product_name}: {str(e)}")
                
        return results

    def process_imagery(self, safe_dir: str, geojson_path: str, output_dir: str) -> None:
        """Process Sentinel imagery to create patches
        
        Args:
            safe_dir: Path to .SAFE directory
            geojson_path: Path to geojson with ROI polygons
            output_dir: Output directory for patches
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Read ROI polygons
        gdf = gpd.read_file(geojson_path)
        
        # Get band files
        granule_dir = glob(os.path.join(safe_dir, "GRANULE/*/"))[0]
        band_files = {
            band: glob(os.path.join(granule_dir, f"IMG_DATA/*_{band}.jp2"))[0]
            for band in self.required_bands
        }
        
        # Process each polygon
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for idx, row in gdf.iterrows():
                poly = row.geometry
                futures.append(
                    executor.submit(
                        self._process_polygon,
                        poly,
                        band_files,
                        output_dir,
                        idx
                    )
                )
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()  # Raises any exceptions that occurred

    def _process_polygon(self, poly, band_files: Dict[str, str], 
                        output_dir: str, poly_idx: int) -> None:
        """Process a single polygon ROI"""
        bounds = poly.bounds
        window = self._get_raster_window(bounds, band_files['B02'])
        
        # Read and resample all bands
        patch_data = {}
        for band, filepath in band_files.items():
            with rasterio.open(filepath) as src:
                data = src.read(1, window=window)
                if data.shape != (self.patch_size, self.patch_size):
                    data = self._resample_array(data, (self.patch_size, self.patch_size))
                patch_data[band] = data
        
        # Save each band
        for band, data in patch_data.items():
            output_path = os.path.join(output_dir, f"patch_{poly_idx}_{band}.npy")
            np.save(output_path, data)

    def _get_raster_window(self, bounds, raster_path: str) -> Window:
        """Get raster window for given bounds"""
        with rasterio.open(raster_path) as src:
            window = src.window(*bounds)
            return window

def main():
    current_file = Path(__file__).name
    filename_no_ext = Path(__file__).stem
    logging.info(f"Running {filename_no_ext} script...")

    # Initialize paths
    base_path = Path("../Datasets/Sentinel-2")
    output_base = Path("../Datasets/Testing/Tiles")
    samples_path = Path("../Datasets/Testing/Samples")
    
    # Validate paths
    if not base_path.exists():
        raise FileNotFoundError(f"Base path {base_path} does not exist")
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Get Sentinel directories
    safe_dirs = list(base_path.glob("*/*.SAFE"))
    if not safe_dirs:
        raise FileNotFoundError(f"No .SAFE directories found in {base_path}")
    
    # Get latest geojson
    geojson_files = list(samples_path.glob("*.geojson"))
    if not geojson_files:
        raise FileNotFoundError(f"No .geojson files found in {samples_path}")
    latest_geojson = max(geojson_files, key=lambda p: p.stat().st_mtime)

    # Prepare batches for multiprocessing
    cpu_count = mp.cpu_count()
    batch_size = max(1, len(safe_dirs) // cpu_count)
    batches = [
        [
            (safe_dir, safe_dir.parent.name) 
            for safe_dir in safe_dirs[i:i + batch_size]
        ]
        for i in range(0, len(safe_dirs), batch_size)
    ]

    # Process using multiprocessing
    processor = SentinelPatchProcessor(patch_size=224, max_workers=8)
    
    with mp.Pool(processes=cpu_count) as pool:
        process_func = partial(
            processor.process_batch,
            output_base=output_base,
            geojson_path=str(latest_geojson)
        )
        
        total_results = {'processed': 0, 'failed': 0}
        
        for batch_result in tqdm(
            pool.imap_unordered(process_func, batches),
            total=len(batches),
            desc="Processing Sentinel-2 batches"
        ):
            total_results['processed'] += batch_result['processed']
            total_results['failed'] += batch_result['failed']

    # Log final results
    logging.info("\nProcessing Summary:")
    logging.info(f"Total scenes: {len(safe_dirs)}")
    logging.info(f"Successfully processed: {total_results['processed']}")
    logging.info(f"Failed: {total_results['failed']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise
