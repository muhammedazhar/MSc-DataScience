# Small held-out validation continuation

## Current state: approved work in progress

Continue on `codex/curated-deforestation-pilot`; commit each verified logical
change with this handoff updated. Previous training milestone complete at
`439e14c`. Do not repeat training. Root `CONTINUE.md` is usable by Codex, Gemini
Antigravity or another agent without chat history.

Current task: prepare a small prediction-blind validation set from the original
frozen VALIDATION groups, then evaluate the frozen final-1664 checkpoint only
after reviews and labels are locked. Protocol:
`Docs/Reviews/ValidationPilotV1/protocol.json`. No validation model inference
has run and no validation label is approved yet.

Plan: one primary per UXA/UYA Summer/UYA Spring stratum, at most one predeclared
alternate from another group per stratum, maximum six prepared cases and three
accepted loss cases. Clear image/geometry filtering is a search aid and creates
selection bias. It does not prove labels or guarantee an eroded interior.
Full 64x64 feature footprints must avoid all frozen train/test conservative
footprints and all 13 actual training grids. No split reassignment.

Every available timeline observation requires explicit forest/clearing/obscured/
uncertain and stable-control review states. Use no predictions during review.
Accepted loss pairs need >=4 interior pixels, >=64 forest control pixels,
forest then clearing endpoints, later persistence and >=3 clear control dates.
Optional stable-only earlier pairs use a deterministic rule over reviewed
pre-event dates; supervise ONLY the reviewed forest control, never empty targets
or unknown background. These are explicitly derived temporal pairs, not frozen
candidate-pair records. They share parent validation groups, not new locations.

## Implementation and resource boundaries

Keep all existing QA, training-only curation/loader and source-bound model files
unchanged. New selector/preparer/review/export modules provide an explicit
validation role; do not spoof validation as training to bypass old gates.
Selector, preparer and blind review rules are implemented. The 18 review/
preparation tests and 6 selector tests pass under the resource guard. Selector
output is locked under `Docs/Reviews/ValidationPilotV1/selection/`: three
primaries plus two distinct-group alternates, with the missing Summer alternate
explicitly retained. Its run reads metadata only and uses no model predictions.
Source/selection hashes are frozen before image inspection. Export/audit code
is still in progress. No validation images have been inspected or approved yet.

The first preparation preflight found a new-code schema mismatch: completed QA
stores `processing_errors` as an empty list, not integer zero. It failed before
pixel processing and published no output. The new preparer now validates the
actual list contract; all 9 preparation tests pass, including its regression.
Protocol preparation-source hash was refreshed before any successful preparation.

GitHub backup is now authorized and configured: push every completed checkpoint
to `origin/codex/curated-deforestation-pilot` on the same branch. The initial
push succeeded through `6509ba5`; subsequent checkpoint pushes should follow.

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Git root is its parent. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
One low-priority CPU worker, native/Torch threads one, no GPU. Unchanged guard:
512 MiB RSS, 1.5 GiB available-memory reserve, 300s maximum, one prepared case
per worker. No simultaneous pixel/model jobs. Labels 0/1/255 retain exact
reviewed forest/clearing/ignore meanings. Never overwrite completed outputs.

Inputs: `Datasets/AnnotationAudit/v1/`, `Datasets/PixelQA/v1/`, existing
`Datasets/PixelQA/cache/`, immutable 13-case training export. The frozen QA
policy and original source hashes are bound in the protocol. Earlier details:
`Docs/Handoffs/2026-11-16-pixel-pilot.md` and linked prior handoffs.

## Remaining work

1. Finish/test metadata-only deterministic selector and separate guarded
   validation preparer/review/export path; freeze source/selection hashes.
2. Prepare at most six cases sequentially, inspect images without predictions,
   preserve all rejections, bind decisions to exact artifacts and freeze labels.
3. Verify exports and geographic/temporal/source isolation, then commit the
   label lock before any model inference.
4. Evaluate only the frozen final-1664 checkpoint at threshold .5, no retraining
   or tuning; retain all outcomes and report partial-ROI limitations.
5. Update guide, evidence and this handoff; commit completed verified work.

If interrupted, inspect committed outputs and active workers before resuming.
Do not turn unreviewed cases into approved labels or rerun finished jobs.
