# Source acquisition and reader adaptation continuation

> Historical milestone: compatibility/readiness claims are qualified by the
> [18 November review](../Reviews/AntigravityReview/2026-11-18-review.md).
> Continue from `2026-11-18-validation-source-review.md`; preserve these receipts.


## Current state: approved acquisition feasibility check complete; minimal pair verified

Keep **`codex/curated-deforestation-pilot`**. Commit every verified logical
change with this handoff updated and push to the SAME GitHub branch. Root
`CONTINUE.md` is tool-neutral for Codex, Gemini Antigravity or another agent.
The user requested bounded work to conserve the remaining usage allowance.

The V8 acquisition feasibility check is COMPLETE: minimal validation pair
(`2018-07-26` and `2018-07-31`) in tile **`36UXB`** on relative orbit **`R064`**
is available via public AWS open data under Baseline **`02.06`** (`N0206`) with
zero radiometric offset. The assembled SAFE directory structure passes the frozen
scene reader (`annotation_scene_inventory.py`) specification with `metadata_usable == True`,
valid quantification value (10,000), well-formed cloud QA GML and exact 10 m grid
bounds in EPSG:32636 (`[600000, 5590200, 709800, 5700000]`).
Total download size for the 7 required bands + metadata across both scenes is
**870.07 MiB** (0.85 GiB), safely within the user's 32 GiB Mac capacity reserve.
No raster pixels, QA/cache arrays, cloud/forest models, full downloads or
labels were used/approved. No grid or original flag changed. No worker is active.

Checkpoints pushed: `5ff0adb` predeclared protocol, `e6f2600` tested implementation
and source freeze. The completion commit contains all acquisition plan evidence, reports
and integrity checks. Previous milestone `9f4145d`: source catalogue audit.
Read `2026-11-17-validation-catalogue.md` and linked prior handoffs for provenance,
model, QA, splits and the recorded Torch runtime discrepancy.

## Minimal pair roadmap and reader compatibility

Protocol: `Docs/Reviews/ValidationAcquisitionV8/protocol.json`, committed before
acquisition planning execution. Exact minimal pair target:

| Date | Satellite | Relative Orbit | Tile | Source | Baseline | Offset | Required 7 Bands Size |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| 2018-07-26 | S2B | `R064` | `36UXB` | AWS L1C | `02.06` | 0 DN | 440.00 MiB |
| 2018-07-31 | S2A | `R064` | `36UXB` | AWS L1C | `02.06` | 0 DN | 430.07 MiB |

- **AWS Open Data vs CDSE**:
  - AWS open data provides the original datatakes processed under Baseline `02.06` (`N0206`), identical to the 50 local scenes. Zero radiometric offset means no reader modification is required.
  - CDSE provides Baseline `05.00` (`N0500`) with a -1,000 DN offset, triggering `nonzero_radiometric_offset_unsupported` in the frozen reader, and requires OAuth authentication.
  - AWS open data is the recommended, cleaner, zero-offset acquisition path.
- **Mac capacity bounds**:
  - Total download size: 870.07 MiB (0.85 GiB).
  - Available disk space on Mac: 32.0 GiB (capacity reserve remaining: 31.15 GiB).

## Durable evidence, verification and resources

Evidence: `Docs/Reviews/ValidationAcquisitionV8/`:

- `protocol.json`: 5 direct source receipts, inherited V7/V6 identities and all
  frozen implementations, plus the tested new source hash.
- `acquisition/{acquisition_plan.json,metadata.json,report.md}`: minimal pair asset manifests,
  exact byte sizes, S3 URL mappings, SAFE layout specification and resource receipt.
- `checks/pre_execution_tests/`: full 319-test proof and logs.
- `checks/final_integrity.json`: confirms 5 direct receipts, 36 frozen implementations,
  both planned scenes, zero downloads and zero active workers.

New code: `Solutions/adapt_validation_scenes.py`; tests:
`Tests/test_validation_acquisition.py` (three). They cover SAFE manifest construction,
offset adaptation analysis, skeleton directory creation and frozen reader validation.
The script imports NO `torch`, `rasterio` or `numpy`, remaining strictly metadata-only.

**319 tests pass**, previous 316 plus three new, no failures/errors/skips. Full suite
2.54 s, sampled peak 360.80 MiB (high-water 362.91 MiB), minimum 3.48 GiB available.
Actual planning audit 0.26 s, peak 17.89 MiB, minimum 3.74 GiB available. One nice+10
CPU worker at a time, native threads 1, no GPU; unchanged
`validate_sample.run_guarded`, 512 MiB RSS, 1.5 GiB available-memory reserve, 300 s
maximum (tests 120 s). No source changed after the passing suite.
Local completed copy: `Datasets/ValidationPilot/v8/acquisition/` (ignored).

## Next proposed bounded task (not executed)

Predeclare an isolated, rate-limited download execution protocol for the minimal pair:
1. Download the 7 required JP2 bands and verified metadata for `2018-07-26` and `2018-07-31` from public AWS open data into `Datasets/Sentinel-2/` (assembled into standard SAFE format, 870.07 MiB total).
2. Run `inventory_scenes()` on `Datasets/Sentinel-2/` to formally register the 2 new scenes alongside the existing 50 scenes (total 52 scenes).
3. Re-run coverage and candidate extraction on the 2 fresh events (`event-71d8da17cbbfd9babdee` and `event-ce1c1df8d02be33bb3c8`) to see if candidate crops are covered and pass quality screening.
4. Do not run training, threshold tuning or uncurated evaluation. Retain 0 forest / 1 clearing / 255 ignore partial labels and train/test geographic isolation. Both events remain in one geographic group and cannot count as two independent locations.

No validation labels/dataset, label lock, first-visible interval or accuracy
exists. Preserve V1–V8 outcomes, all frozen sources, exposed-group exclusions,
0 forest/1 clearing/255 unknown masks and complete train/test footprint isolation.

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Git root is its parent. Check branch/status and active commands before new work;
use one guarded low-priority worker, never overwrite outputs or duplicate a job.
