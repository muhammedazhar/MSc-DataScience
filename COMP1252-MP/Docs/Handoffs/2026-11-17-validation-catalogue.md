# Source catalogue search and compatibility continuation

> Historical milestone: compatibility/readiness claims are qualified by the
> [18 November review](../Reviews/AntigravityReview/2026-11-18-review.md).
> Continue from `2026-11-18-validation-source-review.md`; preserve these receipts.


## Current state: approved catalogue audit complete; tile 36UXB covers fixed crops

Keep **`codex/curated-deforestation-pilot`**. Commit every verified logical
change with this handoff updated and push to the SAME GitHub branch. Root
`CONTINUE.md` is tool-neutral for Codex, Gemini Antigravity or another agent.
The user requested bounded work to conserve the remaining usage allowance.

The V7 catalogue audit is COMPLETE: Sentinel-2 tile **`36UXB`** on relative orbit
**`R064`** fully covers both fixed crops and the entire required QA context halo
(1,200 m padding, `[662180,5603540,665480,5606840]`, EPSG:32636) across ALL 7
original pair dates. All 12 original blocked pairs have pre and post scenes available.
No imagery, raster pixels, QA/cache arrays, cloud/forest models, downloads or
labels were used/approved. No grid or original flag changed. No worker is active.

Checkpoints pushed: `0cce5b5` predeclared protocol, `2bd8108` tested implementation
and source freeze. The completion commit contains all catalogue evidence, reports
and integrity checks. Previous milestone `ba9c1a6`: local coverage feasibility check.
Read `2026-11-17-validation-coverage.md` and linked prior handoffs for provenance,
model, QA, splits and the recorded Torch runtime discrepancy.

## Fixed scope, findings and reader compatibility

Protocol: `Docs/Reviews/ValidationCatalogueV7/protocol.json`, committed before
catalogue query execution. Exact V5/V6 planning set: 12 blocked temporal anchor-pairs
from two fresh events in ONE geographic group `group-96c27871459d03c08bcc`:

| Event | Original crop bounds, EPSG:32636 | Covering tile | Relative orbit |
| --- | --- | --- | --- |
| `event-71d8da17cbbfd9babdee` | `[663640,5605000,664280,5605640]` | `36UXB` | `R064` |
| `event-ce1c1df8d02be33bb3c8` | `[663380,5604740,664020,5605380]` | `36UXB` | `R064` |

All 7 fixed dates (`2018-07-26`, `2018-07-31`, `2018-08-05`, `2018-08-10`, `2018-08-15`,
`2018-08-20`, `2018-08-25`) are verified in external catalogues. Tile 36UXB northing
spans 5,590,240 to 5,700,040 m; the core union (5,604,740–5,605,640 m) and context halo
(5,603,540–5,606,840 m) lie safely inside, over 13 km north of the southern tile edge.

Two external catalogues were audited with the following compatibility findings:

1. **Copernicus Data Space Ecosystem (CDSE, official ESA archive)**:
   - 21 products found across the timeline window (2018-07-26 to 2018-09-14).
   - 7 products cover the 7 fixed pair dates, all in `36UXB` on orbit `R064`.
   - Download size: 5,030,496,204 bytes (4.69 GiB) for the 7 pair scenes; 14.49 GiB for all 21 scenes.
   - Processing baseline: `05.00` (`N0500`).
   - Compatibility blocker: Baseline 05.00 includes a -1,000 DN radiometric offset (`RADIO_ADD_OFFSET` = -1000). The project's frozen L1C scene reader (`Solutions/annotation_scene_inventory.py`) flags this as `nonzero_radiometric_offset_unsupported`. Using CDSE products requires offset handling before the frozen reader accepts them.

2. **AWS Earth Search STAC (Element 84)**:
   - 32 items found across the timeline.
   - Retains original baseline `02.06` (`N0206`) matching local `36UXA` data, with zero radiometric offset.
   - Compatibility blocker: Assets are distributed as individual Cloud-Optimized GeoTIFFs (COGs) on S3 rather than monolithic `.SAFE` directories required by the current reader.

## Durable evidence, verification and resources

Evidence: `Docs/Reviews/ValidationCatalogueV7/`:

- `protocol.json`: 5 direct source receipts, inherited V6/V5 identities and all
  frozen implementations, plus the tested new source hash.
- `catalogue/{catalogue.json,metadata.json,report.md}`: all 21 CDSE and 32 Earth Search
  products, pair-level evaluation matrix, exact download sizes and resource receipt.
- `checks/pre_execution_tests/`: full 316-test proof and logs.
- `checks/final_integrity.json`: confirms 5 direct receipts, 35 frozen implementations,
  all 12 pair evaluations, unknown QA, zero downloads and zero active workers.

New code: `Solutions/audit_validation_catalogue.py`; tests:
`Tests/test_validation_catalogue.py` (five). They cover pure-Python UTM-to-WGS84 ellipsoid
projection, ray-casting point-in-polygon containment, bounding-box-in-polygon sampling,
product name parsing and baseline/offset compatibility classification.
The script imports NO `torch`, `rasterio` or `numpy`, remaining strictly metadata-only.

**316 tests pass**, previous 311 plus five new, no failures/errors/skips. Full suite
3.06 s, sampled peak 355.30 MiB (high-water 356.58 MiB), minimum 3.58 GiB available.
Actual catalogue audit 5.34 s, peak 37.59 MiB, minimum 3.67 GiB available. One nice+10
CPU worker at a time, native threads 1, no GPU; unchanged
`validate_sample.run_guarded`, 512 MiB RSS, 1.5 GiB available-memory reserve, 300 s
maximum (tests 120 s). No source changed after the passing suite.
Local completed copy: `Datasets/ValidationPilot/v7/catalogue/` (ignored).

## Next proposed bounded task (not executed)

Predeclare an acquisition feasibility and reader adaptation protocol:
1. Design a non-destructive reader adapter in `Solutions/` supporting Sentinel-2 Baseline 04.00+/05.00 radiometric offsets (subtracting -1,000 DN or offsetting reflectances) without altering the frozen baseline-02.06 reader behavior.
2. If download is authorized, download the minimal pre/post scene pair (`2018-07-26` and `2018-07-31` from CDSE, ~1.4 GiB total) under strict Mac capacity bounds (one guarded worker, rate-limited, checksum-verified).
3. Do not run training, threshold tuning or uncurated evaluation. Retain 0 forest / 1 clearing / 255 ignore partial labels and train/test geographic isolation. Both events remain in one geographic group and cannot count as two independent locations.

No validation labels/dataset, label lock, first-visible interval or accuracy
exists. Preserve V1–V7 outcomes, all frozen sources, exposed-group exclusions,
0 forest/1 clearing/255 unknown masks and complete train/test footprint isolation.

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Git root is its parent. Check branch/status and active commands before new work;
use one guarded low-priority worker, never overwrite outputs or duplicate a job.
