# Unscreened validation support inventory

## Current state: approved inventory in progress

Continue on `codex/curated-deforestation-pilot`; commit verified changes with
this handoff updated and push to the same branch. Read the previous completed
`2026-11-17-validation-manual.md` and its linked handoffs for provenance and
frozen QA/model/split constraints. Previous milestone: `43500cc`, 299 tests.

V5 protocol: `Docs/Reviews/ValidationUnscreenedV5/protocol.json`, predeclared
before geometry processing. The exact V2 complement has 606 pairs, 630 anchors,
67 events and 9 groups. Metadata inspection finds ALL pairs already blocked;
retain those flags rather than treating absent QA as the only obstacle.
No images, caches, satellite pixels or models are needed for this task.

Protocol checkpoint `61feefc` is pushed. New inventory and verification modes in
`Solutions/audit_validation_unscreened.py` are tested and source-hash frozen.
All 305 tests pass (299 previous + 6 new): 3.82s worker, peak342.75MiB,
minimum3.46GiB available. Evidence: `checks/pre_execution_tests/` under V5 Docs.
No actual inventory or imagery work has run at this checkpoint. Verification
independently reconstructs catalogue projection, cumulative masks and NumPy
erosion; metadata gate/summary helpers are shared, not an independent class/QA
certification. This deliberately reuses existing helpers to keep the task bounded.

Remaining: run one guarded inventory and its geometry verification; preserve every pair/anchor,
all original blockers, unknown QA, support masks/counts, full footprint isolation,
later metadata opportunities and all V1/V4 group exclusions. No shortlist or
labels. Geometry support must never override provenance/coverage blocks.

Use one nice +10 worker, native threads one, no GPU, unchanged
`validate_sample.run_guarded`, 512 MiB RSS, 1.5 GiB available-memory reserve,
300 seconds maximum. Local outputs: `Datasets/ValidationPilot/v5/` (ignored).
Durable evidence: `Docs/Reviews/ValidationUnscreenedV5/`. All old source files
remain frozen. Never overwrite completed outputs or duplicate active workers.

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Git root is the project parent. The V4 group `group-45f0c39e787ca8622918` is
now exposed/rejected and must be excluded alongside every V1 reviewed group.
No validation dataset, approved labels, date interval or accuracy exists.
