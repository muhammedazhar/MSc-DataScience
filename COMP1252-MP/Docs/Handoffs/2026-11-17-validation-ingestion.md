# Source ingestion and local scene registration continuation

## Current state: approved scene ingestion and registration in progress

Keep `codex/curated-deforestation-pilot`; commit verified changes with this
handoff updated and push to the SAME GitHub branch. Previous completed acquisition planning:
`00a4749`; read `2026-11-17-validation-acquisition.md` and linked prior handoffs
for frozen QA, split/model/provenance constraints. User requested bounded work to conserve usage allowance.

V9 protocol: `Docs/Reviews/ValidationIngestionV9/protocol.json`, predeclared before
scene download execution. Target: download the 7 required JP2 bands and verified XML/GML
metadata for the minimal validation pair (`2018-07-26` and `2018-07-31`) in tile `36UXB`
from public AWS open data into `Datasets/Sentinel-2/` (assembled into standard SAFE format, 870.07 MiB total).
Register both scenes using `inventory_scenes()`, bringing the local inventory from 50 to 52 scenes,
and verify that both fresh events now have full core and QA context coverage.

Protocol checkpoint `1fc954e` is pushed. The new scene ingestion worker implementation
`Solutions/ingest_validation_scenes.py` and five unit/regression tests in
`Tests/test_validation_ingestion.py` are complete; source is frozen in the protocol
(`0f368c5d9b63048ccc689838cd21cedc44a6aa55b15ebfc4e5ae4af3dd42cb01`).
All 324 tests pass: test suite 1.74 s, sampled peak 353.5 MiB (high-water 354.67 MiB),
minimum 3.81 GiB available. Proof: V9 `checks/pre_execution_tests/`.
No real ingestion worker has run yet at this checkpoint.

Remaining: run one guarded worker, archive report, verify final integrity and update
this handoff. Preserve existing blocks. No uncurated training, threshold tuning or labels authorized
in this step. No download worker has run at this checkpoint.

Use one nice +10 CPU worker, native threads one, no GPU, unchanged
`validate_sample.run_guarded`, 512 MiB RSS, 1.5 GiB available-memory reserve,
300s maximum. Local outputs: `Datasets/ValidationPilot/v9/` (ignored).
Evidence: `Docs/Reviews/ValidationIngestionV9/`. Project:
`/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Git root is the parent. Preserve all old source/protocol/result files and
train/test footprint barriers; no exposed group becomes fresh again.
