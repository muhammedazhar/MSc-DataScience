# Manual validation timeline continuation

## Current state: approved review in progress; protocol predeclared

Continue on `codex/curated-deforestation-pilot`. Commit each completed verified
logical change with this handoff updated and push to the SAME GitHub branch.
`CONTINUE.md` is the tool-neutral entry point for Codex, Gemini Antigravity or
another agent. Previous completed milestone: `f46085f`, independently verified
V3 target/context QA with NO automatic NDVI forest control; 278 tests passed.
Read `Docs/Handoffs/2026-11-17-validation-context.md` and its linked prior
handoffs for frozen provenance, QA, split, model and resource constraints.

The user authorized the next visual timeline review. The separate V4 protocol
is `Docs/Reviews/ValidationManualV4/protocol.json`, declared BEFORE new images,
pixel processing or visual assessment. V3's negative eligibility stays unchanged.
The new preparation implementation and eight focused tests are complete.
Its source hash is frozen in the protocol before processing. Independent
verification code/tests are still in progress in separate new files. No V4
pixel worker, preview inspection, manual verdict, labels or forest-model
inference has completed at this checkpoint. Frozen existing sources remain
unchanged. The original protocol checkpoint is `4950a0c` and is pushed.

## Fixed task and boundaries

Use the SAME fixed event `event-bf3d28f31dcf19938962`, pair
`pair-430ec0ecae32ce98efcb`, group `group-45f0c39e787ca8622918`, original
64x64/10m grid, 2019-04-02 -> 2019-04-27, annotation image date 2019-04-22.
Target36/interior4 and full two-pixel contexts136/36 are already clear/finite
at endpoints. Preserve the 56 pre-image whole-crop shadow-risk pixels.

The complete same-year +/-45-day window is 2019-03-08 through 2019-06-06,
containing 12 local scenes. Metadata-only inventory found eight existing QA
caches and four missing: March13, March18, June1 and June6. The protocol binds
all scene identities, source statistics and existing cache hashes. Preparation
may compute exactly those four missing observations using the UNCHANGED frozen
QA/cloud policy in a single guarded worker. This is quality preparation, not
deforestation prediction. Do not substitute cache absence for unavailable imagery
or omit difficult dates. No new endpoint feature extraction is needed; reuse
the V3 saved original27-channel features and validity evidence.
The four QA observations themselves require 44 fused-band window requests,
including 28 per-date feature requests for RGB/NDVI/validity, and four cloud
screening calls. Only endpoint27-channel re-extraction is zero. This receipt
wording was clarified before processing; no raw-source-zero claim is made.

Before opening images, select and COMMIT at most three control squares using
only endpoint QA, finite inputs and exclusions: largest16/12/8 then ascending
row/column. Each FULL radius2 context must avoid the full-catalogue20m exclusion,
target context and every earlier selected full context. NDVI thresholds, class
appearance and model predictions do not select the roster. No redrawing,
shrinking or replacement after inspection. Retain every candidate disposition.

Inspect all12dates as plain fixed0-.30 RGB and separate exact target/context/
control overlays plus QA. Require unambiguous target forest at pre, clearing at
post, later clear persistence and no contradictory later forest. Require one
whole control ROI to be forest at both endpoints and a later date (>=3 distinct
dates), no clear contradictory nonforest/change. Every definite visual claim
needs full-context QA-clear and finite display data; both fixed endpoints also
retain all27 finite input features. Select first qualifying locked control after
reviewing all candidates/dates. If target or every control is ambiguous, reject.
No negative-only labels from a rejected case. Assistant review is not expert
ground truth and this one-group convenience case is not a benchmark.

## Remaining work

Preparation: `Solutions/prepare_validation_manual.py`, with eight passing
`Tests/test_validation_manual_preparation.py` tests. Guard receipt/log:
`Docs/Reviews/ValidationManualV4/checks/preparation_helpers/`. Worker1.78s,
peak191.53MiB, minimum5.25GiB available. Tests cover geometry-order/no-NDVI
selection, disjoint FULL halos, flags/exclusions/nonfinite values only in a
margin, exhaustion/empty roster, exact visual contexts and capacity limits.
No real validation imagery or QA model ran during these synthetic tests.

1. Finish/test new separate independent verifier, including complete rendered
   image coverage and plain pixels matching saved RGB. Freeze its source before
   actual processing, preserving all old immutable files. New verifier file may
   be uncommitted while this work is in progress; it has not run.
2. Run one guarded preparation; retain all12 dates, raw evidence and resources.
   Commit exact control roster/artifacts BEFORE image inspection.
3. Independently verify source/array/QA/roster integrity. Inspect all plain and
   overlay images, record every date/control state with uncertainty, and verify
   the final hash-bound review disposition independently.
4. Archive proof/test receipts, update guide and handoff, commit/push completion.

No validation dataset export, label lock, label approval, training, evaluation
or accuracy is part of this V4 task. Further work depends on the reviewed result.
First-visible intervals describe observed images, not exact felling dates.

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Git root is its parent. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Use one nice+10 CPU worker, native threads1, no GPU, UNCHANGED
`validate_sample.run_guarded`, 512MiB RSS, >=1.5GiB available-system-memory
reserve, <=300s. Never run concurrent pixel/model jobs or overwrite completed
outputs. Planned local working outputs: `Datasets/ValidationPilot/v4/` (ignored).
Durable evidence: `Docs/Reviews/ValidationManualV4/`. Full source imagery/caches
remain on this Mac; GitHub stores code and bounded evidence. Preserve 0/1/255
mask semantics and all train/test footprint isolation. Existing Torch runtime/
distribution discrepancy is documented in the prior handoffs; do not reinstall.
