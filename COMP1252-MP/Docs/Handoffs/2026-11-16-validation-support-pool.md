# Broader validation support audit continuation

## Current state: authorized work in progress

Branch `codex/curated-deforestation-pilot`; commit every verified logical change
with an updated handoff and push to the same GitHub branch. Previous milestone
complete at `771d1c7`. Read `CONTINUE.md`, `AGENTS.md` and the previous validation
handoff. Do not repeat its five-case review, model training or completed audits.

Approved task: audit actual interior support over the BROADER frozen validation
QA pool, preserve exclusions and save a fixed shortlist BEFORE new image review.
Protocol: `Docs/Reviews/ValidationSupportV2/protocol.json`. It binds all input
hashes, immutable source code, selection rules and Mac limits before processing.
The broad audit is COMPLETE and archived in `Docs/Reviews/ValidationSupportV2/audit/`.
It covers all 540 completed validation QA pairs/anchors, 53 events and 14 groups,
with all 606 unscreened candidates retained separately (1146 candidates total).
100 pairs from 10 events meet the four-pixel interior minimum. Only 8 pairs from
2 events in 2 groups satisfy the quality, footprint and later-observation gates
before review-history exclusion. Both groups were previously exposed, leaving
ZERO fresh candidates and all six shortlist slots explicitly missing.

The empty shortlist is a valid audit outcome. Do not relax the protocol or add
fallback candidates to populate it. No new imagery was opened, no labels were
approved, and no validation model inference was run. The worker took 4.59s,
peak 198.77MiB, with >=4.17GiB available; no source JP2 pixels or Torch imports.
All ten rule/provenance tests pass. Independent verification remains pending;
it reconstructs geometry using published target planes and separate NumPy erosion.

The universe is all frozen validation candidate pairs. Every completed QA pair
and EVERY anchor will be audited, including bad quality and prior-rejected
locations. No area/bounding dimensions or top-three proposal restriction.
Unscreened/blocked candidates remain explicit records without invented counts.
Count raw anchor, temporally clipped target and one-pixel-eroded interior
separately. Targets depend on pair dates; do not cache by event/grid alone.

Shortlist eligibility retains automatic pass, zero endpoint crop/target QA,
actual interior >=4 and zero interior QA, full separation from train/test
conservative footprints and all 13 real training grids, and a later usable-
metadata scene in the same-year annotation+45d window. Later scene availability
is an opportunity for visual persistence review, not an approved label.
Exclude all five previously reviewed EVENTS and GROUPS from the new shortlist,
while retaining their audit rows/reasons. This preserves fresh image review.
Rank by group/event, exact annotation-date post preference, shortest interval
and pair ID, without NDVI/model scores. Allocate all three source primaries
before up to one alternate each, distinct groups throughout. Missing slots stay
missing; no fallback or forced approval. Maximum six shortlist cases.

## Remaining work

1. DONE: implemented/tested broad worker and executed the full audit without
   modifying archived QA/training/validation code. Raw output is complete; no rerun.
2. Finish/test and run the independent verifier, checking all geometry, counts,
   exclusions and deterministic empty shortlist. Retain diagnostics as planning
   evidence only; do not change the predeclared selection.
3. Save verification/test receipts and a clear report, update this handoff and
   `CONTINUE.md`, commit and push. No new imagery review in this audit milestone.

The later review task needs a separate bounded review protocol for the actual
covered source strata; do not spoof absent slots into the old V1 protocol.
No dataset, labels or model accuracy follow automatically from support counts.

## Capacity and sources

Project `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Git root is its parent. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
One nice+10 CPU worker, native threads one, no GPU, unchanged resource guard:
512 MiB RSS, 1.5 GiB available-memory reserve, maximum 300 seconds per worker.
No concurrent processing workers; atomic outputs never overwrite finished work.

Local ignored outputs: `Datasets/ValidationPilot/v2/`. Durable evidence:
`Docs/Reviews/ValidationSupportV2/`. Full audit/cache/source imagery remain local.
The protocol binds the frozen QA policy and old source hashes. Preserve all
original models, splits and 0/1/255 label semantics. Test groups stay untouched.
