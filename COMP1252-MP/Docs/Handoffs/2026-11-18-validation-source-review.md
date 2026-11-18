# Antigravity source review and repair continuation

## Current state: review complete; source completion in progress

Continue on `codex/curated-deforestation-pilot`; commit verified changes with
this handoff updated and push to the SAME branch. Previous Antigravity checkpoint
`2cb99f4` is preserved. Read `2026-11-17-validation-ingestion.md`, acquisition/
catalogue handoffs and linked V1-V6 provenance/QA/split/model constraints.

Review: `Docs/Reviews/AntigravityReview/2026-11-18-review.md`. Critical correction:
V9's seven-band inventory success is NOT frozen pixel-QA readiness. Both actual
SAFEs lack B01/B05/B09/B10 and44 required quality GMLs each; actual
`qa_scene_metadata` isFalse. V8/V9 ingestion also lacks trusted reuse checks and
transactional final publication. Selected products' actual L1C/N0206/zero-offset
identities and coverage are supported; old source/protocol/results remain frozen.
Do not rerun archived V8/V9 workers to repair this.

V10 protocol: `Docs/Reviews/ValidationSourceRepairV10/protocol.json`, fixing
exact96supplements/74,541,437bytes from provider listing sizes/ETags. Original20
asset source sizes and V9 SHA receipts are retained. Stage outside Sentinel2
discovery, verify size/MD5/SHA, measure diskfree>=2GiB, require actual QA metadata
on BOTH scenes, publish with journal/recovery and retain original backups.
No invented empty quality masks or original source/grid/label changes.

Remaining: finish/test new supplement transaction, freeze source, run one guarded
completion; then separately predeclare bounded QA-only screening for two explicit
derived source pairs. Grids/group/anchors/temporal label IDs stay identical to
frozen parents. New source scene paths/IDs and derived IDs must be explicit;
original missing-coverage pairs stay blocked. No class review, forest prediction,
training or labels in this phase. All27-channel eligibility is a later check.

Both new implementations and their tests are complete, outside frozen originals:
`complete_validation_qa_inputs.py` (8 tests) and `screen_repaired_validation.py`
(9 tests). All341tests pass, no errors/failures; worker7.94s, peak298.27MiB,
minimum2.70GiB available. Full proof and focused8-test receipt are archived under
`Docs/Reviews/ValidationSourceRepairV10/checks/`. V10 source hash is frozen.
Review/protocol checkpoint `ddb55da` is pushed; this checkpoint adds tested code.
No supplement download or QA job has run at this checkpoint. Existing2scenes
and prior20asset receipts are retained; no active pixel/model job.

Use one nice+10 CPU worker, threads1, no GPU, unchanged `validate_sample.run_guarded`,
512MiB RSS,1.5GiB available-system-memory reserve,300s maximum. Working paths:
`Datasets/ValidationPilot/v10/` and later `v11/` (ignored). Persistent transaction/
old backups must be OUTSIDE guard temporary-report staging. Project:
`/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Both events share group96c and are not two independent validation locations.
Preserve all exposed-group/train/test footprint barriers and0/1/255 masks.
