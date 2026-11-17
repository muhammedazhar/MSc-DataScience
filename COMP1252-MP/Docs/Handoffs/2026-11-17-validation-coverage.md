# Source coverage feasibility continuation

## Current state: approved metadata check complete; no local coverage

Keep **`codex/curated-deforestation-pilot`**. Commit every verified logical
change with this handoff updated and push to the SAME GitHub branch. Root
`CONTINUE.md` is tool-neutral for Codex, Gemini Antigravity or another agent.
The user requested bounded work to conserve the remaining usage allowance.

The V6 check is COMPLETE: neither fixed crop intersects any of the 50 local
Sentinel-2 scene rectangles. Fresh tile/product XML and directory discovery
confirm the frozen inventory; the original source-coverage blockers are correct.
No imagery, raster pixels, QA/cache arrays, cloud/forest models, downloads or
labels were used/approved. No grid or original flag changed. No worker is active.

Checkpoints pushed: `2c5aa82` predeclared protocol, `e159b90` tested implementation
and source freeze. The completion commit contains all coverage evidence and
source requirements. Previous milestone `a383472`: complete V5 unscreened audit.
Read `2026-11-17-validation-unscreened.md` and linked V4/V3/V2/prior handoffs for
frozen provenance, model, QA, splits and the existing Torch runtime discrepancy.

## Fixed scope and measured gap

Protocol: `Docs/Reviews/ValidationCoverageV6/protocol.json`, committed before
XML processing. Exact V5 planning set:12 blocked temporal anchor-pairs from two
fresh events in ONE geographic group `group-96c27871459d03c08bcc`:

| Event | Original crop bounds, EPSG:32636 | Gap north of nearest source |
| --- | --- | ---: |
| `event-71d8da17cbbfd9babdee` | `[663640,5605000,664280,5605640]` | 4, 960 m |
| `event-ce1c1df8d02be33bb3c8` | `[663380,5604740,664020,5605380]` | 4, 700 m |

Each original grid is 64 x 64/10 m. All six pairs per event use pre 2018-07-26;
posts 2018-07-31, 08-05, 08-10, 08-15, 08-20 and 08-25. Both pre and post miss the
crops. Switching dates among current scenes cannot repair the geographic gap.

All 50 scenes use EPSG:32636 and have maximum northing 5600040:

- 12 tile 36 UXA scenes: `[600000,5490240,709800,5600040]`.
- 38 tile 36 UYA scenes: `[699960,5490240,809760,5600040]`.

The 100 event/scene comparisons include all local tiles and dates, not just the
originally named tile or endpoints. Full crop coverage 0, positive-area overlap 0,
full frozen-QA-context coverage 0. Native pixel-window containment independently
agrees with rectangle containment; crop row offsets are negative. No CRS mismatch
or merely partial boundary clip explains these missing sources.

Directory discovery checks the normal direct/nested SAFE layouts in
`Datasets/Sentinel-2/`, exactly matching all 50 inventoried paths. No unindexed
SAFE scene was found in that configured root. This is not a search of every
folder/archive on the Mac or an external catalogue. Every tile XML was freshly
parsed, its origin/spacing/dimensions/CRS/bounds compared to the frozen inventory,
and product XML hashes checked:100 XML files in total. Source hashes are recorded
and checked again after processing. No JP2 contents or raster library was needed.
A separate read-only assistant metadata review confirmed the same bounds/gaps.

## Durable evidence, verification and resources

Evidence: `Docs/Reviews/ValidationCoverageV6/`:

- `protocol.json`: six direct source receipts, inherited V5 identities and all
  frozen old implementations, plus the tested new source hash.
- `coverage/{coverage.json,metadata.json,report.md}`: all 50 scene grids,
  all 100 comparisons, exact pairs, XML hashes and resource receipt.
- `source_requirements.json`: fixed core/context bounds and dates for planning
  ONLY; no product selected, catalogue availability verified or download approved.
- `checks/pre_execution_tests/`: full 311-test proof and logs.

`checks/final_integrity.json` confirms six direct and 22 inherited source receipts,
34 frozen implementations, 100 XML hashes, 50 current SAFE paths and 100 comparisons.
Archived copies match completed local outputs, with no active processing worker.

New code: `Solutions/audit_validation_coverage.py`; tests:
`Tests/test_validation_coverage.py` (six). They cover whole/partial/touching
footprints, exact outside gaps, malformed/ambiguous XML, signed grid geometry,
and direct/nested/missing SAFE discovery. Verification is fresh XML versus frozen
metadata plus a separate native-window calculation and read-only review, not an
independent visual/QA classifier. Metadata bounds cannot certify valid pixels,
clear atmosphere, forest class, temporal change or source-product compatibility.

**311 tests pass**, previous 305 plus six new, no failures/errors/skips. Full suite
4.07 s, sampled peak 358.20 MiB (high-water 352.59 MiB), minimum 4.54 GiB available.
Actual metadata check 0.51 s, peak 41.31 MiB, minimum 4.64 GiB available. One nice+10
CPU worker at a time, native threads 1, no GPU; unchanged
`validate_sample.run_guarded`, 512 MiB RSS, 1.5 GiB available-memory reserve, 300 s
maximum (tests 120 s). Worker imported no NumPy, Rasterio or Torch. No source changed
after the passing suite; no repeat processing or tests are needed without cause.
Local completed copy: `Datasets/ValidationPilot/v6/coverage/` (ignored).
The source data/XML remain local; GitHub has code and bounded evidence.

## Next proposed bounded task (not executed)

Search a source CATALOGUE using metadata only for compatible Sentinel-2 products
covering the fixed coordinates and dates. No downloads or processing yet. Record
coverage, acquisition date, processing baseline, source-reader/QA compatibility
and download size before preparing any acquisition plan. Do not guess a tile ID,
select by model/clearing contrast, shift these grids, clear old flags or silently
replace frozen scene identities. Any corrected-data/QA experiment needs its own
predeclared protocol and explicit lineage to these blocked original pairs.

The exact core union is `[663380,5604740,664280,5605640]`; cloud QA needs the
frozen 1200 m source padding around BOTH crops, giving combined context bounds
`[662180,5603540,665480,5606840]`, all EPSG:32636. Exact grids and timeline window
ends are in `source_requirements.json`. These are geometry requirements, not a
claim that available catalogue products satisfy them or work with the current
frozen L1 C reader (offset/baseline/QA prerequisites must be checked separately).
Both events share one group and cannot count as two independent locations.

No validation labels/dataset, label lock, first-visible interval or accuracy
exists. Preserve V1–V6 outcomes, all frozen sources, exposed-group exclusions,
0 forest/1 clearing/255 unknown masks and complete train/test footprint isolation.
No training, threshold tuning, final 1664 evaluation or test imagery inspection.

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Git root is its parent. Check branch/status and active commands before new work;
use one guarded low-priority worker, never overwrite outputs or duplicate a job.
