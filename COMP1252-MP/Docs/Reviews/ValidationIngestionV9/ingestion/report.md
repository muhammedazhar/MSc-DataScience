# Source ingestion and local scene registration report

**Minimal validation pair scenes for 2018-07-26 and 2018-07-31 in tile 36UXB were successfully ingested from AWS open data and registered in Datasets/Sentinel-2/ (total 52 local scenes). Both scenes pass frozen reader verification with metadata_usable True and zero radiometric offset. Both fresh validation events now have full core crop and QA context containment on both pair dates.**

## Summary

```json
{
  "total_registered_scenes": 52,
  "prior_registered_scenes": 50,
  "new_registered_scenes": 2,
  "ingested_scene_ids": [
    "S2B_MSIL1C_20180726T084009_N0206_R064_T36UXB_20180726T104227",
    "S2A_MSIL1C_20180731T083601_N0206_R064_T36UXB_20180731T110233"
  ],
  "tile_id": "36UXB",
  "relative_orbit": "R064",
  "processing_baseline": "02.06 (N0206)",
  "radiometric_offset": 0,
  "all_new_scenes_metadata_usable": true,
  "total_bytes_transferred": 979071178,
  "download_elapsed_seconds": 45.89,
  "minimal_pair_dates": [
    "2018-07-26",
    "2018-07-31"
  ],
  "events_evaluated": 2,
  "both_events_minimal_pair_covered": true,
  "coverage_verdict": "Both fresh events (event-71d8da17cbbfd9babdee and event-ce1c1df8d02be33bb3c8) now have full core crop and QA context coverage for 2018-07-26 and 2018-07-31.",
  "blocker_status": "Original missing_post_scene_coverage and crop_crosses_tile_edge blocks are resolved for the 2018-07-26/2018-07-31 pair. QA and manual review screening remain required before any validation use."
}
```

## Event Minimal Pair Coverage

| Event | Pre (2018-07-26) Covered | Post (2018-07-31) Covered | Tile | Orbit | Both Covered |
| --- | --- | --- | --- | --- | --- |
| `event-71d8da17cbbfd9babdee` | True | True | `36UXB` | `R064` | **True** |
| `event-ce1c1df8d02be33bb3c8` | True | True | `36UXB` | `R064` | **True** |

