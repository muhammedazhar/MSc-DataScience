import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm
from s2cloudless import S2PixelCloudDetector

class CloudProcessor:
    def __init__(self, base_path: str, cloud_threshold: float = 0.1):
        self.base_path = Path(base_path)
        self.cloud_threshold = cloud_threshold
        # Corrected band list and order
        self.s2cloudless_bands = ['B02', 'B03', 'B04', 'B05', 'B08', 'B8A', 'B11', 'B12']
        self.cloud_detector = S2PixelCloudDetector(threshold=0.4)
        
        # Verify base path exists
        if not self.base_path.exists():
            raise ValueError(f"Base path does not exist: {self.base_path}")

    def get_safe_dir(self, patch_file: Path) -> Path:
        """Get corresponding SAFE directory for a patch"""
        # Extract product name from patch path
        # Updated to account for the new directory structure
        product_name = patch_file.parent.name

        # Search for matching SAFE directory
        safe_pattern = f"{product_name}.SAFE"
        safe_dirs = list(self.base_path.glob(safe_pattern))
        
        if not safe_dirs:
            # Try searching one level deeper
            safe_dirs = list(self.base_path.glob(f"*/{safe_pattern}"))
            
        if not safe_dirs:
            raise ValueError(f"No SAFE directory found for product: {product_name}")
            
        return safe_dirs[0]

    def load_bands_for_patch(self, safe_dir: Path) -> Dict[str, np.ndarray]:
        """Load required bands for cloud detection with proper alignment"""
        bands = {}
        # Use B02 as reference band
        ref_band = 'B02'
        ref_files = list(safe_dir.glob(f'GRANULE/*/IMG_DATA/*{ref_band}*.jp2'))
        if not ref_files:
            raise FileNotFoundError(f"Reference band {ref_band} not found in {safe_dir}")
        ref_band_file = ref_files[0]
        with rasterio.open(ref_band_file) as ref_src:
            ref_data = ref_src.read(1)
            ref_transform = ref_src.transform
            ref_crs = ref_src.crs
            ref_shape = ref_src.shape

        for band in self.s2cloudless_bands:
            band_files = list(safe_dir.glob(f'GRANULE/*/IMG_DATA/*{band}*.jp2'))
            if not band_files:
                raise FileNotFoundError(f"Band {band} not found in {safe_dir}")
            band_file = band_files[0]
            with rasterio.open(band_file) as src:
                if src.crs != ref_crs:
                    raise ValueError(f"Band {band} CRS does not match reference CRS")
                # Resample band to reference resolution and shape
                band_data = src.read(
                    out_shape=(1, ref_shape[0], ref_shape[1]),
                    resampling=Resampling.bilinear
                )[0]
            bands[band] = band_data
        return bands

    def detect_clouds(self, bands: Dict[str, np.ndarray]) -> Tuple[bool, float]:
        """Run cloud detection on band data"""
        # Stack bands in required order
        stacked = np.stack([bands[b] for b in self.s2cloudless_bands], axis=2) / 10000.0  # Shape: (H, W, C)
        data = np.expand_dims(stacked, axis=0)  # Shape: (1, H, W, C)
        
        # Get cloud probabilities
        cloud_probs = self.cloud_detector.get_cloud_probability_maps(data)
        cloud_mask = cloud_probs > self.cloud_detector.threshold
        cloud_percentage = float(np.mean(cloud_mask))
        
        return cloud_percentage <= self.cloud_threshold, cloud_percentage

    def process_patches(self, input_dir: str, output_dir: str):
        """Process all patches"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Get all patches
        patches = list(input_path.rglob("*.npy"))
        
        results = {
            'total': len(patches),
            'processed': 0,
            'kept': 0,
            'errors': 0
        }
        
        for patch_file in tqdm(patches, desc="Processing patches"):
            try:
                # Get original SAFE directory
                safe_dir = self.get_safe_dir(patch_file)
                
                # Load bands and detect clouds
                bands = self.load_bands_for_patch(safe_dir)
                keep_patch, cloud_pct = self.detect_clouds(bands)
                
                results['processed'] += 1
                
                if keep_patch:
                    # Create output directory structure
                    out_file = output_path / patch_file.relative_to(input_path)
                    out_file.parent.mkdir(exist_ok=True, parents=True)
                    
                    # Copy patch to output
                    patch_data = np.load(patch_file)
                    np.save(out_file, patch_data)
                    results['kept'] += 1
                    
                logging.info(f"Processed {patch_file.name}: {cloud_pct:.1%} clouds, kept: {keep_patch}")
                    
            except Exception as e:
                results['errors'] += 1
                logging.error(f"Error processing {patch_file}: {str(e)}")
                continue
        
        # Print summary
        logging.info(f"\nProcessing Summary:")
        logging.info(f"Total patches: {results['total']}")
        logging.info(f"Successfully processed: {results['processed']}")
        logging.info(f"Kept (low clouds): {results['kept']}")
        logging.info(f"Errors: {results['errors']}")