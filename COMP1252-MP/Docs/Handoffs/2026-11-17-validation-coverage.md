# Source coverage feasibility continuation

## Current state: approved metadata check in progress

Keep `codex/curated-deforestation-pilot`; commit verified changes with this
handoff updated and push to the SAME GitHub branch. Previous completed inventory:
`a383472`; read `2026-11-17-validation-unscreened.md` and linked prior handoffs
for frozen QA, split/model/provenance constraints. No model or imagery processing
is part of this task. User requested bounded work to conserve usage allowance.

V6 protocol: `Docs/Reviews/ValidationCoverageV6/protocol.json`, predeclared before
local XML verification. Fixed V5 events `event-71d8da17cbbfd9babdee` and
`event-ce1c1df8d02be33bb3c8`, all 12 pairs, same original grids. Initial frozen
metadata shows the crops wholly north of all 50 scene bounds; confirm against
fresh tile XML and directory discovery before treating this as a completed result.

Remaining: implement/test small metadata-only coverage check, freeze source, run
one guarded worker, archive report and update this handoff. Preserve existing
blocks. No grid shifts, source corrections, downloads, QA or labels authorized
in this step. No inventory worker has run at this checkpoint.

Use one nice +10 CPU worker, native threads one, no GPU, unchanged
`validate_sample.run_guarded`, 512 MiB RSS, 1.5 GiB available-memory reserve,
300s maximum. Local outputs: `Datasets/ValidationPilot/v6/` (ignored).
Evidence: `Docs/Reviews/ValidationCoverageV6/`. Project:
`/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Git root is the parent. Preserve all old source/protocol/result files and
train/test footprint barriers; no exposed group becomes fresh again.
