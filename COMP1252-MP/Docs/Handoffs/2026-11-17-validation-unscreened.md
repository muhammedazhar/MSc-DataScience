# Unscreened validation support inventory

## Current state: approved inventory complete; zero passes all metadata gates

Continue on `codex/curated-deforestation-pilot`; commit verified changes with
this handoff updated and push to the same GitHub branch. Root `CONTINUE.md` is
usable by Codex, Gemini Antigravity or another agent. No chat or worker recovery
is needed. User requested a focused task to conserve the remaining usage allowance.

The authorized metadata/geometry-only V5 inventory is COMPLETE, with geometry
verification passed and **305 tests passing**. No imagery, satellite JP2 pixels,
QA masks, caches, cloud/forest models or labels were used/approved. No shortlist,
validation dataset, timing interval or accuracy was produced. Do not repeat
completed processing or silently remove existing blockers.

Pushed checkpoints: `61feefc` predeclared protocol; `4b08b39` tested implementation
and source freeze. This completion commit contains both completed runs and docs.
Previous milestone `43500cc`: V4 manual rejection. Read
`2026-11-17-validation-manual.md`, `2026-11-17-validation-context.md`,
`2026-11-16-validation-support-pool.md` and their linked handoffs for provenance,
frozen QA/model/split rules and the recorded Torch runtime discrepancy.

## Complete coverage and findings

Protocol: `Docs/Reviews/ValidationUnscreenedV5/protocol.json`, committed before
geometry processing. It binds all inherited V2 inputs, V2 and V4 proof/history,
32 immutable implementations and the new source. Universe is exactly the V2
606-pair complement of published QA, not a new candidate search. All original
pairs and anchor identities are kept: 582 single-anchor pairs and 24 double-anchor
pairs yield **630 anchor-pairs, 67 events, 9 groups**. Pair-level dispositions
remain separate from per-anchor geometry so neither blockers nor extra anchors
can disappear through filtering.

Every pair has original status `blocked_for_temporal_training`. Overlapping
pair-level reasons:347 crop-crosses-tile-edge, 347 post-scene-does-not-cover-crop,
259 unverified-temporal-label-in-crop, 79 baseline-only-provenance. These counts
are not independent cases. The original status/reasons are saved verbatim.

Support is the original anchor intersected with cumulative-post MINUS
cumulative-pre temporal catalogue reference, then 3 x 3 one-pixel erosion. Raw
anchor/raw interior/reference/target/final interior counts and hashes are kept,
along with clipped pixels. Baseline-only anchors are never inserted into the
frozen temporal catalogue; even incidental overlap with another temporal event
cannot override their provenance blocker.

- 37 anchor-pairs from 6 events in 4 groups retain >=4 interior pixels.
- 25 supported anchor-pairs have baseline-only anchor provenance and retain
  both provenance/temporal-label blockers.
- 12 supported temporal anchor-pairs come from TWO events in ONE fresh group:
  `event-71d8da17cbbfd9babdee` (6), `event-ce1c1df8d02be33bb3c8` (6), both
  `group-96c27871459d03c08bcc`, source `36UXA_time-dependent`. All retain crop
  tile-edge and missing-post-coverage blockers; they are NOT approved proposals.
- Exact grids: `[663640,5605000,664280,5605640]` and
  `[663380,5604740,664020,5605380]`, original EPSG:32636/64 x 64/10 m.
- 111 anchor-pairs have temporal clipping.
- All 630 full feature grids avoid EVERY frozen train/test conservative footprint
  and all 13 actual training grids, including boundary contact.
- 566 anchor-pairs have a later metadata-usable same-year observation strictly
  after pair post and within annotation+45 days; opportunity only, not QA/class truth.
- 203 anchor-pairs are in exposed groups. All V1 reviewed groups AND V4's
  `group-45f0c39e787ca8622918` are excluded, covering six exposed groups total.
- **Zero passes all metadata/geometry gates**. All QA counts remain `None`/null,
  never zero/clear. No QA or review roster was generated from these blocked pairs.

## Verification, resources and implementation

New code: `Solutions/audit_validation_unscreened.py`, `--mode audit|verify`.
New tests: `Tests/test_validation_unscreened.py` (6). The producer uses existing
frozen target construction and SciPy erosion. Verification independently projects
the catalogue, burns cumulative masks and uses nine NumPy erosion slices. It
matches ALL630 anchor records, 606 pair dispositions, mask hashes and counts.
Metadata joins/gates and summary helpers are SHARED, so do not describe this as
fully independent metadata gate implementation or visual/QA truth certification.

All 305 tests pass: previous 299 plus six meaningful coverage/blocker/erosion/
temporal-overlap regressions. Test suite 3.82 s, peak 342.75 MiB, minimum 3.46 GiB
available. Audit 4.35 s/266.77 MiB/minimum 3.27 GiB; verification 3.90 s/236.50 MiB/
minimum 3.24 GiB. All were sequential nice+10 CPU workers, native threads 1, no GPU,
using UNCHANGED `validate_sample.run_guarded`, 512 MiB RSS and 1.5 GiB available
memory reserve, 300 s maximum (tests 120 s). No job remains active. No source changes
followed the passing full suite; no rerun is needed absent a new concern.

## Durable state and next task (not executed)

Authoritative evidence: `Docs/Reviews/ValidationUnscreenedV5/`:
`protocol.json`, `inventory/{inventory.json,metadata.json,report.md}`,
`verification/{metadata.json,report.md}`, `checks/pre_execution_tests/`,
`planning_counts.json`. Guide: `Docs/Guides/ValidationUnscreened.md`.
`checks/final_integrity.json` confirms 22 source inputs, 33 frozen source
implementations, all 606/630 identities, unknown QA, exact archived copies and
no active processing worker. Original blocker/history fingerprints are intact.
Local completed working copies: `Datasets/ValidationPilot/v5/` (ignored).
GitHub contains code/bounded evidence; full metadata and satellite data remain
on this Mac. Existing V1–V4 protocols/results and every frozen source are unchanged.

Next proposed bounded task: METADATA-ONLY coverage feasibility for the two fresh
supported temporal events above. Compare original grids, actual scene footprints,
dates and available tile inventory; identify whether suitable coverage exists
locally or additional source products would be required. Do not download/process
new imagery, shift grids, clear old flags or infer labels as part of that check.
Only a separate predeclared corrected-data/QA plan could follow if justified.
The 25 baseline-only supported rows do not provide temporal deforestation labels.

Preserve labels 0 forest/1 clearing/255 unknown, geographic separation and every exposed
group exclusion. No training, threshold tuning, validation inference or test-group
exposure. The model is frozen final 1664 at 0.5, but there is no approved validation
set for it yet. First-visible timing remains future work; annotation image dates
are not exact felling dates.

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Git root is the project parent. Before future workers check branch/status and
active commands. Use one guarded low-priority worker and the capacity limits
above. Never overwrite completed outputs or duplicate an active run.
