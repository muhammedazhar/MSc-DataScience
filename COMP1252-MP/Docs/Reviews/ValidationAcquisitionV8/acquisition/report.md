# Source acquisition and reader adaptation feasibility

**Acquisition feasibility confirmed: minimal pair (2018-07-26 and 2018-07-31) in tile 36UXB on orbit R064 is available via public AWS open data under Baseline 02.06 with zero radiometric offset. Total download size for the 7 required bands + metadata across both scenes is 868.42 MiB (0.85 GiB), safely within the 32 GiB Mac capacity reserve. Assembled SAFE directories match the frozen reader specification.**

## Summary

```json
{
  "plan_scope": "Minimal validation pair acquisition and SAFE assembly feasibility",
  "target_events": [
    {
      "event_id": "event-71d8da17cbbfd9babdee",
      "annotation_image_date": "2018-07-31",
      "crop_grid": {
        "crs": "EPSG:32636",
        "transform": [
          10.0,
          0.0,
          663640.0,
          0.0,
          -10.0,
          5605640.0
        ],
        "width": 64,
        "height": 64,
        "bounds": [
          663640.0,
          5605000.0,
          664280.0,
          5605640.0
        ]
      },
      "required_qa_context_bounds": [
        662440.0,
        5603800.0,
        665480.0,
        5606840.0
      ],
      "later_persistence_window_end": "2018-09-14"
    },
    {
      "event_id": "event-ce1c1df8d02be33bb3c8",
      "annotation_image_date": "2018-07-31",
      "crop_grid": {
        "crs": "EPSG:32636",
        "transform": [
          10.0,
          0.0,
          663380.0,
          0.0,
          -10.0,
          5605380.0
        ],
        "width": 64,
        "height": 64,
        "bounds": [
          663380.0,
          5604740.0,
          664020.0,
          5605380.0
        ]
      },
      "required_qa_context_bounds": [
        662180.0,
        5603540.0,
        665220.0,
        5606580.0
      ],
      "later_persistence_window_end": "2018-09-14"
    }
  ],
  "minimal_pair_dates": [
    "2018-07-26",
    "2018-07-31"
  ],
  "target_tile": "36UXB",
  "target_orbit": "R064",
  "source_repository": "AWS Open Data (sentinel-s2-l1c)",
  "processing_baseline": "02.06 (N0206)",
  "radiometric_offset": 0,
  "quantification_value": 10000.0,
  "total_pair_download_size_bytes": 912332678,
  "total_pair_download_size_mib": 870.07,
  "total_pair_download_size_gib": 0.85,
  "available_disk_space_gib": 32.0,
  "capacity_safety_margin_gib": 31.15,
  "reader_compatibility_assessment": "AWS open data Baseline 02.06 matches the local 50-scene inventory; zero radiometric offset means no reader modification is required.",
  "cdse_alternative_assessment": "CDSE Baseline 05.00 has -1000 DN offset requiring reader adaptation, and requires OAuth credentials.",
  "recommended_acquisition_source": "AWS Open Data (sentinel-s2-l1c, baseline 02.06)"
}
```

## Minimal Pair Ingestion Roadmap

| Date | Satellite | Orbit | Tile | Source | Baseline | Offset | Required Bands Size | Layout |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| 2018-07-26 | S2B | R064 | 36UXB | AWS L1C | 02.06 | 0 DN | 440.0 MiB | SAFE directory |
| 2018-07-31 | S2A | R064 | 36UXB | AWS L1C | 02.06 | 0 DN | 430.07 MiB | SAFE directory |

**Combined download size**: 870.07 MiB (0.85 GiB).

