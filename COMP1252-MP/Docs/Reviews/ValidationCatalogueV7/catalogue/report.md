# Source catalogue search and compatibility audit

**External catalogues confirm full spatial coverage of both fixed crops and QA context by Sentinel-2 tile 36UXB on relative orbit R064 across all 7 original pair dates. Copernicus products use baseline 05.00 with -1000 DN radiometric offset (incompatible with frozen reader without offset support; 4.68 GiB for 7 scenes). AWS Earth Search retains original baseline 02.06 matching local data, but distributes COG assets requiring assembly. No download, raster read, QA model, or label was executed; pairs remain blocked.**

## Summary

```json
{
  "fixed_events": 2,
  "fixed_original_pairs": 12,
  "fixed_pair_dates": [
    "2018-07-26",
    "2018-07-31",
    "2018-08-05",
    "2018-08-10",
    "2018-08-15",
    "2018-08-20",
    "2018-08-25"
  ],
  "timeline_window_end": "2018-09-14",
  "copernicus_total_products_found": 21,
  "copernicus_covering_fixed_date_products": 7,
  "earthsearch_total_products_found": 32,
  "earthsearch_covering_fixed_date_products": 12,
  "covering_tile_id": "36UXB",
  "relative_orbit": "R064",
  "copernicus_processing_baseline": "05.00 (N0500)",
  "copernicus_radiometric_offset": -1000,
  "copernicus_frozen_reader_compatible": false,
  "copernicus_reader_blocker": "nonzero_radiometric_offset_unsupported",
  "copernicus_7_scene_download_size_bytes": 5030496204,
  "copernicus_7_scene_download_size_gib": 4.69,
  "copernicus_21_scene_timeline_size_bytes": 15561169602,
  "copernicus_21_scene_timeline_size_gib": 14.49,
  "earthsearch_processing_baseline": "02.06 (N0206)",
  "earthsearch_radiometric_offset": 0,
  "earthsearch_frozen_reader_compatible": false,
  "earthsearch_reader_blocker": "cog_s3_packaging_requires_safe_reassembly",
  "all_12_pairs_have_covering_imagery_in_catalogue": true,
  "catalogue_spatial_coverage_status": "Tile 36UXB on relative orbit R064 fully covers all core and context grids for all 7 dates.",
  "acquisition_decision": "No downloads or QA executed. CDSE N0500 requires radiometric offset adaptation (-1000 DN); Earth Search N0206 requires S3/COG ingestion pipeline."
}
```

## Pair-Level Coverage and Compatibility

| Event | Pre Date | Post Date | Covering Tile | Orbit | CDSE Available | CDSE Baseline | Offset | ES Available | ES Baseline |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| `event-71d8da` | 2018-07-26 | 2018-07-31 | `36UXB` | `R064` | True | `N0500` | -1000 DN | True | `N0500` |
| `event-71d8da` | 2018-07-26 | 2018-08-05 | `36UXB` | `R064` | True | `N0500` | -1000 DN | True | `N0500` |
| `event-71d8da` | 2018-07-26 | 2018-08-10 | `36UXB` | `R064` | True | `N0500` | -1000 DN | True | `N0500` |
| `event-71d8da` | 2018-07-26 | 2018-08-15 | `36UXB` | `R064` | True | `N0500` | -1000 DN | True | `N0500` |
| `event-71d8da` | 2018-07-26 | 2018-08-20 | `36UXB` | `R064` | True | `N0500` | -1000 DN | True | `N0500` |
| `event-71d8da` | 2018-07-26 | 2018-08-25 | `36UXB` | `R064` | True | `N0500` | -1000 DN | True | `N0500` |
| `event-ce1c1d` | 2018-07-26 | 2018-07-31 | `36UXB` | `R064` | True | `N0500` | -1000 DN | True | `N0500` |
| `event-ce1c1d` | 2018-07-26 | 2018-08-05 | `36UXB` | `R064` | True | `N0500` | -1000 DN | True | `N0500` |
| `event-ce1c1d` | 2018-07-26 | 2018-08-10 | `36UXB` | `R064` | True | `N0500` | -1000 DN | True | `N0500` |
| `event-ce1c1d` | 2018-07-26 | 2018-08-15 | `36UXB` | `R064` | True | `N0500` | -1000 DN | True | `N0500` |
| `event-ce1c1d` | 2018-07-26 | 2018-08-20 | `36UXB` | `R064` | True | `N0500` | -1000 DN | True | `N0500` |
| `event-ce1c1d` | 2018-07-26 | 2018-08-25 | `36UXB` | `R064` | True | `N0500` | -1000 DN | True | `N0500` |

