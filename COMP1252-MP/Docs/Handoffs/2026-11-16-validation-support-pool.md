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
No new geometry processing, imagery inspection or validation inference has run.

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

1. Implement/test the new separate broad audit and independent verifier without
   modifying archived QA/training/validation code. Freeze executable sources.
2. Run one low-priority guarded audit, then independently verify geometry from
   published target masks, counts, all exclusions and deterministic shortlist.
3. Save results/test receipts and a clear report, update this handoff and
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
