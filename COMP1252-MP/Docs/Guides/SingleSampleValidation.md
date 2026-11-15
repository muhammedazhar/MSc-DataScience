# One aligned satellite-image example

The first repair milestone is a small, inspected image pair with a geographically
correct target. The original preprocessing and training scripts are unchanged.
This workflow does not start training or regenerate the full dataset.

## Reproduce

From the project root, use the existing project environment:

```bash
conda run -n COMP1252-MP python -B Solutions/validate_sample.py
```

Defaults select **PLOT-00026, 31 July to 5 August 2018**, tile **36UXA**.
The crop is **64 x 64 pixels at the original 10 m resolution**, covering
640 x 640 metres. This keeps nearby clouds outside the inspected example.
It is a smaller physical crop, not a resized 2.24 km image.

Generated files go to:

```text
Datasets/Validation/PLOT-00026_2018-07-31_2018-08-05_64px/
```

Existing output is never overwritten. For another reproducibility run, choose a
new destination with `--output Datasets/Validation/recheck-PLOT-00026`.
All newly generated validation artifacts are excluded from Git.

## What the example contains

| File | Meaning |
| --- | --- |
| `pre_features.tif`, `post_features.tif` | Float32, nine-channel observations on the same geographic grid |
| `pre_cumulative_loss.tif`, `post_cumulative_loss.tif` | Cumulative annotated clearcut footprints through each observation date |
| `change_mask.tif` | New annotated footprint in the later mask: 0 = no new annotated loss, 1 = new loss, 255 = invalid |
| `valid_pixels.tif` | 1 where both observations pass available product QA and band-value checks |
| `annotations.geojson` | Applicable source polygons, their original feature indices, and observation dates |
| `comparison.png` | Shared-contrast RGB images and the reference annotation overlay, including a close-up |
| `metadata.json` | Source hashes, acquisition names, grid, checks, channel ranges, review status, and measured resources |

Channel order is B02, B03, B04, B08, B8A, B11, B12, NDVI, NDMI.
The seven reflectance bands are divided by the product's quantification value.
NDVI and NDMI are computed with floating-point arithmetic and retain their signed
scale. These files are already scaled: the old loader's blanket division by
10,000 must not be applied to them. Invalid feature pixels contain NaN;
downstream training must honor the validity mask and target value 255.

The later cumulative annotation mask minus the earlier mask supplies the
pair target. All applicable polygons in the crop are considered, with exact
duplicate geometry/date records removed. The individual sampled polygon only
selects the location. Labels are rasterized on the image grid using pixel
centres; their geographic bounds are never stretched to the crop dimensions.

The two dates and all seven source bands are resampled onto one explicit grid.
Only that small target is read, one band and one observation at a time. Product
cloud, nodata, saturation, defect, and technical QA polygons are checked where
present. Their CRS and polygon holes are preserved; invalid QA geometries are
repaired and counted. Missing cloud QA causes refusal. Nonzero L1C radiometric
offsets are currently unsupported and cause refusal instead of silent scaling.

## Inspection result

The accepted example has **4,096 valid pixels** and **28 newly annotated loss
pixels**: 2,800 square metres (0.28 hectares) after rasterization. The source
polygon covers approximately 2,728.64 square metres; the small difference is
expected at 10 m raster boundaries. All six output rasters share **EPSG:32636**,
the same transform, shape, and 10 m pixel spacing.

Visual inspection shows an existing narrow clearing before and a wider bare
patch afterwards. The reference mask follows the newly annotated expansion.
The selected crop has no obvious occluding cloud at the target. The earlier
image has a different colour cast/contrast; this is a visual consistency check,
not independent proof of every annotation boundary or an exact felling date.
Observation dates bracket the reviewed change; 5 August is the annotation image
date, not a claim that the trees were felled that day.

Earlier candidates are kept under `Datasets/Validation/Rejected/` with rejection
reasons in their metadata. PLOT-00001 was rejected from QA before extraction.
PLOT-00004 was too hazy for a strong before-state assessment. PLOT-00058 and the
larger PLOT-00026 crop exposed visible clouds/shadows missed by the supplied
cloud polygons. This is why visual inspection is required even when QA says
zero cloud pixels.

## Mac capacity and safeguards

The Mac has 18 GiB of RAM. The final extraction took **2.04 seconds**, with a
worker high-water memory footprint of **145.3 MiB** and at least **4.18 GiB of
available system memory** at monitor samples. Its artifacts occupy about
**0.32 MiB**.

Default operating limits are:

- One worker, native thread limits of one, and a lower scheduling priority
  (`nice` increment 10).
- No GPU, model loading, whole-tile arrays, or parallel scene processing.
- 32 MiB GDAL cache and 16 MiB warp buffer.
- A 512 MiB worker RSS budget, 1.5 GiB system-memory reserve, and 180-second
  runtime budget.
- Monitoring every 250 ms. The worker is stopped if a sampled limit is exceeded;
  its final high-water memory is also checked before publishing output. These
  are monitored safeguards, not an instantaneous operating-system memory cap.
- Isolated staging: a failed or interrupted run discards its own temporary
  output and leaves the original datasets and existing results intact.
- Crop sizes are restricted to multiples of 32 from 64 to 256 pixels. There is
  no batch-processing option.

## Verification and remaining scope

Run the small regression suite with:

```bash
conda run -n COMP1252-MP python -B -m unittest discover -s Tests -v
```

Tests use tiny synthetic rasters and mocked workers. They cover signed index
arithmetic, separate scaling, temporal targets, GML holes/CRS/altitudes, mixed
band resolutions, nodata/saturation, and resource-limit cleanup.

Before expanding the dataset, confirm which original baseline and temporal
annotation files were merged into the local GeoJSON. A zero target means no
new loss is annotated; it does not establish that an area is verified unchanged.
The existing training, inference, split leakage, and prediction-output bugs
remain for subsequent milestones.

Source references:

- [Dataset and observation-date definition](https://github.com/QuantuMobileSoftware/forest_change_detection#dataset)
- [Original cumulative-mask construction](https://github.com/QuantuMobileSoftware/forest_change_detection/blob/master/time_dependent/segmentation/preprocessing/binary_mask_converter.py)
- [Original pairwise change targets](https://github.com/QuantuMobileSoftware/forest_change_detection/blob/master/time_dependent/segmentation/preprocessing/image_difference.py)
- [Paper, DOI 10.1109/JSTARS.2020.3034186](https://doi.org/10.1109/JSTARS.2020.3034186)
- [Rasterio windowed reading](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html)
- [Rasterio virtual crop/reprojection parameters](https://rasterio.readthedocs.io/en/stable/api/rasterio.vrt.html)
