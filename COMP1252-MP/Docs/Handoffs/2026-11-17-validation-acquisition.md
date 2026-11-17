# Source acquisition and reader adaptation continuation

## Current state: approved acquisition feasibility check in progress

Keep `codex/curated-deforestation-pilot`; commit verified changes with this
handoff updated and push to the SAME GitHub branch. Previous completed catalogue audit:
`9f4145d`; read `2026-11-17-validation-catalogue.md` and linked prior handoffs
for frozen QA, split/model/provenance constraints. User requested bounded work to conserve usage allowance.

V8 protocol: `Docs/Reviews/ValidationAcquisitionV8/protocol.json`, predeclared before
acquisition assembly execution. Target: minimal pre/post scene pair (`2018-07-26` and `2018-07-31`)
in tile `36UXB` on relative orbit `R064`. Evaluates AWS open data Sentinel-2 L1C (baseline 02.06, zero offset,
public access) and Copernicus L1C (baseline 05.00, offset normalization). Develops SAFE assembly adapter
and tests compatibility with the project's frozen L1C reader (`annotation_scene_inventory.py`).

Remaining: implement/test SAFE assembly adapter and acquisition planner with offline mock tests,
freeze source, run one guarded worker, archive report, verify final integrity and update
this handoff. Preserve existing blocks. No uncurated training, full downloads or labels authorized
in this planning step. No acquisition worker has run at this checkpoint.

Use one nice +10 CPU worker, native threads one, no GPU, unchanged
`validate_sample.run_guarded`, 512 MiB RSS, 1.5 GiB available-memory reserve,
300s maximum. Local outputs: `Datasets/ValidationPilot/v8/` (ignored).
Evidence: `Docs/Reviews/ValidationAcquisitionV8/`. Project:
`/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Git root is the parent. Preserve all old source/protocol/result files and
train/test footprint barriers; no exposed group becomes fresh again.
