# Small held-out validation continuation

## Current state: bounded review and audit complete; validation labels unavailable

Continue on `codex/curated-deforestation-pilot`. Commit each verified logical
change with this handoff updated, then push to the SAME GitHub branch at origin.
Root `CONTINUE.md` is the tool-neutral entry point for Codex, Gemini Antigravity
or another agent. The full training milestone remains complete at `439e14c`;
no training or held-out inference needs repeating.

The five-case validation roster is exhausted: **0 approved, 5 rejected**. The
rejections were independently verified; a geometry-only audit then explained
poor raster interior support across the metadata-eligible pool. There is NO
approved validation dataset, label lock or accuracy score. Preparing labels and
measuring unseen-location performance remain future work. Do not force labels,
relax margins or silently select replacements until a result passes.

Checkpoints: `6509ba5` predeclared protocol; `2dbd686` locked roster and tested
review/preparation; `18ec71c` QA schema fix; `5f41594` all five rejections;
`65aef3f` interruption backup preserving in-progress code; `501c023` independent
review verification and all 35 review-bound TIFFs. The following completion
commit contains geometry results, final 250-test proof, exporter refusal and
updated guide/continuation. Completed checkpoints are pushed to origin.

## What was selected, inspected and rejected

Protocol: `Docs/Reviews/ValidationPilotV1/protocol.json`. Selection:
`selection/selection.json` under that review directory. The metadata-only rule
used frozen validation groups, clear endpoint QA and area/bounding dimensions
as SEARCH AIDS, with no NDVI contrast or model prediction ranking. Full 64x64
feature footprints avoid EVERY frozen train/test conservative footprint and
all 13 actual training grids. No split reassignment occurred.

Three primaries and two predefined distinct-group alternates were selected.
Summer had no eligible second group under the same rules. All five complete
local timeline PNGs were opened before any validation prediction, and the
Spring primary's detailed before/after pair was inspected. The observations
retain explicit uncertainty; no rejected case supplies even negative labels.

| Candidate suffix | Source stratum | Role | Eroded pixels | Rejection |
| --- | --- | --- | ---: | --- |
| 9993 | UXA | Primary | 1 | Below fixed minimum 4 |
| 4ee5 | UYA Summer | Primary | 0 | Below fixed minimum 4 |
| 4a343 | UYA Spring | Primary | 10 | Pre-image forest not unambiguous |
| 9495 | UXA | Alternate | 0 | Below fixed minimum 4 |
| 3304 | UYA Spring | Alternate | 0 | Below fixed minimum 4 |

The Spring primary lies in a road/disturbed-ground mosaic; colour change or an
annotation date does not prove a forest-to-clearing transition. Review is
assistant inspection, not independent expert ground truth. Rejection statements
are hash-bound to exact case JSONs, all 9 artifacts and all prepared observations.

## Geometry finding and interpretation

`geometry_audit/` reconstructs the anchor-specific temporal reference and exact
one-pixel 3x3 erosion on all 33 metadata-eligible anchor-pairs from 11 events in 6
geographic groups. It selects no new candidates and reads no satellite pixels.

- UXA: 18 pairs, 6 events; none retain >=4 interior pixels.
- UYA Summer: 6 pairs, 2 events; none retain >=4 interior pixels.
- UYA Spring: 9 pairs, 3 events; 6 pairs from 2 events in 2 groups retain >=4 pixels.
- Total:27/33 pairs fail. Geometry support is NOT clearing/temporal approval.

The two support-qualified events are `event-4a34368ee9483ed8f7a7` (already
rejected for ambiguous temporal evidence) and `event-b41d67055c21412c75de`
(not visually reviewed or approved). Do not treat this list as a new roster.
The audited subset inherits earlier clear-image/area/top-candidate filtering;
it is not all 514 validation events or all 91 validation groups.

Tests show equal 1,600m2 geometries can differ: a thin 2-pixel-wide rectangle loses
all eroded pixels, while a 4x4-pixel square retains 4. Union targets can also be
large because of other events; only the selected anchor's own interior counts.
The initial area/bbox search aid did not establish minimum raster support.

## Verification and resources

**250 tests pass**: the previous 206 plus 44 validation tests. Full suite took
1.480s within a 3.31s guarded worker, peak 305.89 MiB, minimum 3.82 GiB available.
No further full-suite/model rerun is needed absent a new change or concern.

Independent review verification PASSED all 5 rejections, 53 timeline observations
and 53 distinct caches. It checked exact frozen joins, full footprint isolation,
raw/canonical case hashes, all artifacts, original target/erosion, complete
scene-inventory coverage, actual cached QA planes/counts, published endpoint QA
and source fingerprints. It used 179.73 MiB in 2.79s, no Torch/model execution or
source JP2 pixel reads. It verifies recorded evidence, not independent visual
truth. Prepared features were checked without re-extracting raw source bands.

Geometry audit PASSED in 2.55s at165.14MiB; no satellite pixel reads, model
execution, new imagery inspection or label approval. Frozen input/source hashes
were unchanged. Actual export of the all-rejected roster is also tested: it
refuses publication with an explicit error and the unchanged guard removes
staging, leaving no dataset output. No empty-set accuracy is manufactured.

Five sequential preparation workers totalled 63.99s and peaked at380.30 MiB.
Minimum available memory stayed >=2.63 GiB. All processing used one nice +10
CPU worker, native threads 1, no GPU, unchanged 512 MiB RSS / 1.5 GiB reserve guard.

## Durable evidence and implementation status

Authoritative tracked evidence: `Docs/Reviews/ValidationPilotV1/`:

- `protocol.json`, `selection/`: predeclared criteria, exact roster and source hashes.
- `prepared_cases/`: all five case JSONs, pair/timeline PNGs and 35 TIFF arrays.
- `preparation_receipts/`, `reviews.json`: measured jobs and every final rejection.
- `review_verification/`: completed independent rejection audit/report.
- `geometry_audit/`: all 33 geometry reconstructions, coverage/report and resources.
- `checks/`: focused/final test logs and actual `export_refusal.json` check.

Guide: `Docs/Guides/ValidationPilot.md`. New implemented modules:
`select_validation_cases.py`, `prepare_validation_review.py`,
`validation_review.py`, `build_validation_pilot.py`,
`verify_validation_dataset.py`, `evaluate_validation_pilot.py`,
`audit_validation_geometry.py`. Export/evaluation helpers have meaningful unit
tests but are NOT integrated against a real approved validation dataset, because
none exists. Inference requires committed approved labels/reviews/protocol and
a label-lock file before importing Torch. Do not infer execution from code presence.

One early preparation preflight found new-code `processing_errors==0` assumed
an integer, whereas the frozen QA report stores a list. It failed before pixel
processing, published nothing and was corrected/tested before successful jobs.
The independent verifier uses the same corrected list contract. Original frozen
QA code and archived model/objective sources remain unchanged.

The global TIFF ignore rule initially omitted case arrays from Git. A narrow
exception now preserves all 35 exact hash-bound Docs TIFFs. Some duplicate local
JSON/PNG outputs were also included by interruption backup 65aef3f; preserve that
history and use Docs snapshots as authoritative. Further local working outputs
are ignored under `Datasets/ValidationPilot/`. GitHub contains code and bounded
review evidence; full audit/cache/source imagery remain on this Mac and are
needed for complete numerical rechecks and source-stat verification.

## Safe continuation and next task

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Git root is its parent. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
No pixel/model job remains active. Check branch/status and existing outputs
before any new worker. Never overwrite completed results or duplicate a job.

Frozen QA policy:
`fe2472ec030f466bebbb9dc2cb033ed6175de489bcf022f650081dd8e9347560`.
Original QA/model/training-only loader/curation files are unchanged. The existing
Torch runtime/distribution discrepancy remains recorded; do not reinstall it
as part of this task. Preserve labels 0 forest, 1 clearing, 255 unknown exactly.

Completed review recheck, ONLY if specifically needed, using a NEW output path:

```bash
PILOT_PYTHON=/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python
"$PILOT_PYTHON" -B Solutions/verify_validation_dataset.py --mode reviews --reviews Docs/Reviews/ValidationPilotV1/reviews.json --output Datasets/ValidationPilot/v1/review-verification-recheck
```

Next proposed bounded task (NOT executed): audit actual anchor-specific raster
interior support over a BROADER frozen validation QA pool, avoiding reliance on
area/bbox or top-three review proposals as sufficient support. Do this before
preparing more image timelines. Keep the margin/minimum and split barriers;
predeclare a separate support-qualified roster before opening new imagery.
Retain all rejections and acknowledge convenience-sample bias. Temporal ambiguity
needs explicit review/expert adjudication rather than automatic boundary labels.
Only after approved reviews and exports are committed should the frozen final
1664 checkpoint be evaluated at .5 without retraining or tuning. Test groups stay
untouched. Temporal first-visible-date inference remains later work.
