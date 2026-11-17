# Source catalogue audit for two validation events

**External source catalogues confirm full spatial coverage of both fixed validation crops and their QA context by Sentinel-2 tile `36UXB` on relative orbit `R064` across all 7 original pair dates.**
No imagery was downloaded, no satellite raster pixels were opened, no QA masks or caches were created, and no cloud or forest models were run. The original blocking status of the 12 pairs remains unchanged.

| Event suffix | Original crop bounds, EPSG:32636 | Covering tile | Relative orbit | Dates covered |
| --- | --- | --- | --- | ---: |
| `71d8da17cbbfd9babdee` | `[663640,5605000,664280,5605640]` | `36UXB` | `R064` | 7 of 7 |
| `ce1c1df8d02be33bb3c8` | `[663380,5604740,664020,5605380]` | `36UXB` | `R064` | 7 of 7 |

Combined required QA context bounds (with 1,200 m margin): `[662180,5603540,665480,5606840]`, EPSG:32636.
Tile `36UXB` spans northing 5,590,240 to 5,700,040 m, containing both core crops and the full context halo with >13 km margin to any tile border.

## Catalogue findings and reader compatibility

Two external catalogues were audited using WGS84 spatial geometry and observation dates:

1. **Copernicus Data Space Ecosystem (CDSE, official ESA)**:
   - 21 products found covering the timeline (2018-07-26 to 2018-09-14).
   - All 7 pair dates (`2018-07-26`, `2018-07-31`, `2018-08-05`, `2018-08-10`, `2018-08-15`, `2018-08-20`, `2018-08-25`) exist in `36UXB` on relative orbit `R064`.
   - Download size: 5,030,496,204 bytes (4.69 GiB) for the 7 pair scenes; 14.49 GiB for all 21 timeline scenes.
   - Processing baseline: `05.00` (`N0500`).
   - Compatibility blocker: Copernicus Baseline 04.00+ / 05.00 includes a -1,000 DN radiometric offset (`RADIO_ADD_OFFSET` = -1000). The project's frozen L1C scene reader (`Solutions/annotation_scene_inventory.py`) flags this as `nonzero_radiometric_offset_unsupported`. Downloading CDSE scenes would require an offset adaptation before the frozen reader accepts them.

2. **AWS Earth Search STAC (Element 84)**:
   - 32 items found across the timeline.
   - Retains original processing baseline `02.06` (`N0206`) matching the local `36UXA` scenes, with zero radiometric offset.
   - Format: distributed as individual cloud-optimized GeoTIFF (COG) assets on S3, rather than monolithic `.SAFE` directories required by the frozen reader.

## Scientific constraints preserved

- Both events share one geographic group (`group-96c27871459d03c08bcc`); they cannot count as two independent validation locations.
- Spatial footprint coverage alone does not establish cloud-free QA, clear forest baseline, or a verified clearing transition.
- Original pairs remain blocked; no substitute source or download has been executed.
- No validation labels, validation dataset, or accuracy scores exist.

**316 tests pass.** The guarded audit worker used **37.59 MiB in 5.34 s**, with 3.67 GiB minimum available memory, one nice+10 CPU worker, threads 1, no GPU and the unchanged 512 MiB/1.5 GiB memory guard. Full suite used 355.30 MiB in 3.06 s.

Evidence: `Docs/Reviews/ValidationCatalogueV7/`; continuation: `Docs/Handoffs/2026-11-17-validation-catalogue.md`.
