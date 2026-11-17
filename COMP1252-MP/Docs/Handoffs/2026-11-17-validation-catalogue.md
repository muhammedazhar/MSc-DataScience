# Source catalogue search continuation

## Current state: approved metadata catalogue search in progress

Keep `codex/curated-deforestation-pilot`; commit verified changes with this
handoff updated and push to the SAME GitHub branch. Previous completed coverage:
`ba9c1a6`; read `2026-11-17-validation-coverage.md` and linked prior handoffs
for frozen QA, split/model/provenance constraints. No model or imagery processing
is part of this task. User requested bounded work to conserve usage allowance.

V7 protocol: `Docs/Reviews/ValidationCatalogueV7/protocol.json`, predeclared before
catalogue query execution. Fixed V5/V6 events `event-71d8da17cbbfd9babdee` and
`event-ce1c1df8d02be33bb3c8`, all 12 original pairs, exact core union bounds
`[663380,5604740,664280,5605640]` and QA context bounds `[662180,5603540,665480,5606840]`
in EPSG:32636. Query external source catalogues (Copernicus Data Space Ecosystem and
AWS Earth Search STAC) using spatial geometry and dates only. Record coverage, acquisition
date, tile ID, processing baseline, source-reader/QA compatibility and download sizes.

Remaining: implement/test metadata-only catalogue search with offline mock tests,
freeze source, run one guarded worker, archive report, verify final integrity and update
this handoff. Preserve existing blocks. No grid shifts, downloads, raster pixel reads, QA
or labels authorized in this step. No search worker has run at this checkpoint.

Use one nice +10 CPU worker, native threads one, no GPU, unchanged
`validate_sample.run_guarded`, 512 MiB RSS, 1.5 GiB available-memory reserve,
300s maximum. Local outputs: `Datasets/ValidationPilot/v7/` (ignored).
Evidence: `Docs/Reviews/ValidationCatalogueV7/`. Project:
`/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Git root is the parent. Preserve all old source/protocol/result files and
train/test footprint barriers; no exposed group becomes fresh again.
