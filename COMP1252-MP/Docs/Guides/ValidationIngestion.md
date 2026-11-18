# Sentinel-2 minimal pair scene ingestion and local registration guide

> Review correction (18 November 2026): V9 verified the seven-band inventory only; the actual frozen pixel-QA prerequisites were incomplete.
> See [the review](../Reviews/AntigravityReview/2026-11-18-review.md) and current continuation.


**Scene ingestion and local SAFE registration completed: minimal pair (2018-07-26 and 2018-07-31) in tile `36UXB` on relative orbit `R064` successfully ingested from AWS Open Data and registered in `Datasets/Sentinel-2/`, bringing total local registered scenes from 50 to 52.**
Both scenes pass frozen reader verification (`inventory_scenes`) with `metadata_usable == True` and zero radiometric offset. Both fresh validation events now have full core crop and QA context containment on both pair dates.

| Scene Date | Satellite | Relative Orbit | Tile | Source | Baseline | Offset | Transferred Size | Registered Status |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| 2018-07-26 | S2B | `R064` | `36UXB` | AWS L1C | `02.06` | 0 DN | 461.38 MiB | `metadata_usable: True` |
| 2018-07-31 | S2A | `R064` | `36UXB` | AWS L1C | `02.06` | 0 DN | 472.34 MiB | `metadata_usable: True` |

## Key results and verification

1. **Successful SAFE directory assembly and registration**:
   - Both scenes assembled into standard Sentinel-2 SAFE layout with 7 required JP2 bands (`B02`, `B03`, `B04`, `B08`, `B8A`, `B11`, `B12`), product metadata (`MTD_MSIL1C.xml`), tile metadata (`MTD_TL.xml`), and cloud QA masks (`MSK_CLOUDS_B00.gml`).
   - `inventory_scenes()` in `Solutions/annotation_scene_inventory.py` registered both scenes without issues (`missing_bands: []`, `cloud_qa_missing: False`, `cloud_qa_well_formed: True`, `quantification_value: 10000.0`).
   - Total local registered scenes in `Datasets/Sentinel-2/` increased from 50 to 52.

2. **Resolution of original spatial coverage blocks**:
   - `event-71d8da17cbbfd9babdee`: Core crop `[663640, 5605000, 664280, 5605640]` and QA context `[662440, 5603800, 665480, 5606840]` are completely contained within tile `36UXB` bounds `[600000, 5590200, 709800, 5700000]` on both `2018-07-26` and `2018-07-31`.
   - `event-ce1c1df8d02be33bb3c8`: Core crop `[663380, 5604740, 664020, 5605380]` and QA context `[662180, 5603540, 665220, 5606580]` are completely contained within tile `36UXB` bounds on both dates.
   - The prior blocks `missing_post_scene_coverage` and `crop_crosses_tile_edge` are resolved for this pair.

3. **Guarded resource execution**:
   - Downloaded 933.72 MiB (979,071,178 bytes) across 20 files in 45.89 s (~21.3 MB/s average transfer speed).
   - Guarded worker high-water RSS: **79.66 MiB** (limit: 512 MiB).
   - Total elapsed execution time: **46.48 s** (limit: 300 s).
   - Available Mac system memory reserve remained at **3.47 GiB** (limit: >=1.5 GiB).

## Scientific constraints preserved

- Both events share one geographic group (`group-96c27871459d03c08bcc`); they cannot count as two independent validation locations.
- Zero satellite raster pixel reads were performed during ingestion; verification was strictly metadata and bounds containment.
- No model training, threshold tuning or label inference has been executed.
- Quality screening (cloud/shadow/haze masks) and blind manual review screening remain mandatory before any validation use.

**324 tests pass.** Guarded pre-execution proof: `Docs/Reviews/ValidationIngestionV9/checks/pre_execution_tests/`.
Evidence: `Docs/Reviews/ValidationIngestionV9/`; continuation: `Docs/Handoffs/2026-11-17-validation-ingestion.md`.
