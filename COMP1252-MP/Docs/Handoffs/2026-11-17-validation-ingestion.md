# Source ingestion and local scene registration continuation

> Historical milestone: compatibility/readiness claims are qualified by the
> [18 November review](../Reviews/AntigravityReview/2026-11-18-review.md).
> Continue from `2026-11-18-validation-source-review.md`; preserve these receipts.


## Current state: scene ingestion and local registration completed

Keep `codex/curated-deforestation-pilot`; commit verified changes with this
handoff updated and push to the SAME GitHub branch. Previous completed acquisition planning:
`00a4749`; read `2026-11-17-validation-acquisition.md` and linked prior handoffs
for frozen QA, split/model/provenance constraints. User requested bounded work to conserve usage allowance.

V9 protocol: `Docs/Reviews/ValidationIngestionV9/protocol.json`, completed and verified.
Minimal validation pair (`2018-07-26` and `2018-07-31`) in tile `36UXB` on relative orbit `R064`
was successfully ingested from AWS Open Data (Baseline `02.06`, zero radiometric offset)
and assembled into standard SAFE product layout in `Datasets/Sentinel-2/` (total 933.72 MiB across 20 files).
Registered both scenes using `inventory_scenes()`, bringing the local inventory from 50 to 52 scenes,
with both scenes passing frozen reader checks (`metadata_usable == True`, `nonzero_offset == False`,
`quantification_value == 10000.0`, `missing_bands == []`, `cloud_qa_missing == False`, `cloud_qa_well_formed == True`).
Verified full spatial containment of core crops and 1,200 m QA contexts for both fresh validation events
(`event-71d8da17cbbfd9babdee` and `event-ce1c1df8d02be33bb3c8`) on both dates.
All 324 tests pass. The guarded ingestion worker used 79.66 MiB high-water RSS in 46.48 s,
with 3.47 GiB minimum available memory reserve.

Local outputs: `Datasets/ValidationPilot/v9/ingestion/` (ignored).
Evidence: `Docs/Reviews/ValidationIngestionV9/`. Guide: `Docs/Guides/ValidationIngestion.md`.
Integrity proof: `Docs/Reviews/ValidationIngestionV9/checks/final_integrity.json`.

Remaining work: downstream pixel quality screening (cloud, shadow, invalid pixels) on the newly covered pair
for both events (Milestone V10). No uncurated training, threshold tuning or labels authorized.
Preserve existing blocks. Both events share one geographic group (`group-96c27871459d03c08bcc`);
they cannot count as two independent validation locations.

Use one nice +10 CPU worker, native threads one, no GPU, unchanged
`validate_sample.run_guarded`, 512 MiB RSS, 1.5 GiB available-memory reserve,
300s maximum. Local outputs: `Datasets/ValidationPilot/v9/` (ignored).
Evidence: `Docs/Reviews/ValidationIngestionV9/`. Project:
`/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Git root is the parent. Preserve all old source/protocol/result files and
train/test footprint barriers; no exposed group becomes fresh again.
