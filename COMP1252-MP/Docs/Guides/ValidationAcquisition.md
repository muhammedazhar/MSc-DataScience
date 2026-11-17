# Source acquisition and reader adaptation guide for validation scenes

**Acquisition feasibility confirmed: minimal pair (2018-07-26 and 2018-07-31) in tile `36UXB` on relative orbit `R064` is available via public AWS open data under Baseline `02.06` with zero radiometric offset.**
Assembled SAFE directories match the frozen scene reader specification. Total download size for the 7 required bands + metadata across both scenes is 870.07 MiB (0.85 GiB), safely within the user's 32 GiB Mac capacity reserve.

| Scene Date | Satellite | Relative Orbit | Tile | Source | Baseline | Offset | Required 7 Bands Size |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| 2018-07-26 | S2B | `R064` | `36UXB` | AWS L1C | `02.06` | 0 DN | 440.00 MiB |
| 2018-07-31 | S2A | `R064` | `36UXB` | AWS L1C | `02.06` | 0 DN | 430.07 MiB |

## Key findings and architecture

1. **Direct compatibility with frozen reader**:
   - The AWS open data archive (`sentinel-s2-l1c`) provides the original datatakes processed under Baseline `02.06` (`N0206`), matching the exact configuration of all 50 local scenes in `Datasets/Sentinel-2/`.
   - Radiometric offsets are zero (`RADIO_ADD_OFFSET` absent). The frozen reader accepts the metadata with `nonzero_offset == False` and `metadata_usable == True`.
   - Access is public over HTTP without authentication credentials.

2. **Copernicus Data Space Ecosystem (CDSE) comparison**:
   - Reprocessed under Baseline `05.00` with a -1,000 DN radiometric offset.
   - Requires OAuth authentication.
   - Requires downloading monolithic ~750 MB zip archives containing all 13 bands (including unneeded 60m bands B01, B09, B10).
   - Recommended source is therefore AWS open data, saving ~500 MB per scene and avoiding offset adaptation.

3. **SAFE Assembly Structure**:
   - `Solutions/adapt_validation_scenes.py` maps S3 asset URLs to the standard Sentinel-2 SAFE layout:
     - `MTD_MSIL1C.xml` (product metadata)
     - `GRANULE/<granule_id>/MTD_TL.xml` (tile metadata)
     - `GRANULE/<granule_id>/QI_DATA/MSK_CLOUDS_B00.gml` (cloud QA mask)
     - `GRANULE/<granule_id>/IMG_DATA/T36UXB_<datetime>_<band>.jp2` (7 required bands: B02, B03, B04, B08, B8A, B11, B12)
   - Tested using mock SAFE skeletons; all reader gates pass cleanly.

## Scientific constraints preserved

- Both events share one geographic group (`group-96c27871459d03c08bcc`); they cannot count as two independent validation locations.
- No training, threshold tuning or uncurated evaluation has been run.
- Partial labels remain 0 forest, 1 clearing, 255 unknown. Complete train/test footprint isolation is preserved.

**319 tests pass.** The guarded planning worker used **17.89 MiB in 0.26 s**, with 3.74 GiB minimum available memory, one nice+10 CPU worker, threads 1, no GPU and the unchanged 512 MiB/1.5 GiB memory guard. Full suite used 360.80 MiB in 2.54 s.

Evidence: `Docs/Reviews/ValidationAcquisitionV8/`; continuation: `Docs/Handoffs/2026-11-17-validation-acquisition.md`.
