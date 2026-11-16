# Deforestation project handoff

## Current state: reviewed pilot complete

Branch: `codex/curated-deforestation-pilot`.
Initial durable checkpoint: `4eb2387`; the following completion commit contains
final decisions, the small dataset and verification. Continue on this branch.

User continuation instruction, confirmed 16 November 2026: save each completed,
verified logical change as a commit on this same branch and refresh this handoff
with current state, validation and remaining work. If interrupted mid-change,
save an explicit incomplete checkpoint rather than implying completion. Project
`AGENTS.md` carries this instruction for subsequent agents.

Current next task remains the ignore-aware loader, loss and metrics described
below, followed by a resource-bounded baseline smoke run. No model training has
started. This continuation update changes documentation/instructions only;
validation is a clean diff check, and the previously recorded 136-test result
belongs to the completed pilot implementation.

Repository root: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience`.
Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.

The user authorized the data audit, quality scan and reviewed pilot. They also
explicitly requested a separate branch and durable handoff. Their goal is to
locate deforestation and estimate when it first becomes visible in usable
imagery. Annotation `img_date` is the image observation date, not physical
felling time. Dataset construction is complete; no forest model was trained.

## Completed outputs

- Corrected physical-grid sample validation and signed NDVI/NDMI features.
- Annotation audit: 5,696 raw records -> 3,911 unique events in 601 frozen
  geographic train/validation/test groups.
- Full QA: 6,948 pairs / 55 batches / zero processing errors. Statuses: 2,924
  pass pending visual review, 3,405 excluded, 578 visual-review, 41 label-review.
  Independent verification passed all masks, targets, 5,096 caches and links.
- Event review queue: 682 events, 7,001 links. 554 events have candidate
  proposals; 124 require alternate imagery/quality review; 4 need label review.
- Reviewed pilot: 28 candidate dispositions retained, 13 accepted examples
  from 13 distinct training groups with 13 matched stable-forest control ROIs.
  Nine UXA, three UYA Summer-source, one UYA Spring-source (June).
- Labels: 151 clearing-interior pixels, 2,640 stable-forest pixels, 50,457
  ignored pixels. Reviewed interior area 1.51 ha; this is not total event area.
- All 13 full timelines and primary pair previews were visually inspected.
  Accepted states have zero corresponding ROI QA; clearing persists in later
  clear frames and controls have >=3 clear dates including both pair endpoints.
  Observed intervals range 5–20 days and explicitly retain uncertain/cloud dates.
- Independent final pilot verification passed all 13 arrays, all 28 decisions,
  136 timeline observations/caches, feature signs/scales, exact grids/ROI masks,
  hashes/source statistics, date bounds and held-out footprint isolation.
- Full test suite: **136 passed** (known old rejection-path ResourceWarnings).

## Where to pick up

Read `Docs/Guides/CuratedPilot.md` first. The small final dataset is tracked:

- `Datasets/CuratedPilot/v1/dataset/report.md`: concise accepted/rejected table
  and direct links to original local inspection artifacts.
- `dataset/training_manifest.jsonl`: all 13 examples; array paths are relative
  to the dataset directory and can be consumed without raw satellite scenes.
- `dataset/examples/<event-id>/`: 27-channel input TIFF, partial supervision,
  valid mask and hash-rich metadata. The whole dataset is about 6.2 MiB.
- `dataset/review_decisions.jsonl`: all 28 explicit dispositions and timelines.
- `Datasets/CuratedPilot/v1/verification/{metadata.json,report.md}`: completed
  independent checks and measured resources.
- `Docs/Reviews/CuratedPilotV1/`: finalized `reviews.json`, selection/replacement
  provenance and all 28 case JSONs, PNGs and ROI masks. Its prepared-case index
  maps tracked evidence to original local preparation paths. Historical draft
  notes are clearly superseded by final reviews.

Large raw imagery/caches and full audit/QA outputs remain local under ignored
`Datasets/Sentinel-2/`, `Datasets/AnnotationAudit/v1/`, `Datasets/PixelQA/v1/`.
Prepared review batches remain local under
`Datasets/CuratedPilot/v1/{review_parts,replacement_parts}/`. They are finished;
no processing worker remains. Only the small final dataset/verification and
tracked evidence were intentionally added to Git; original data/models remain
unchanged. A fresh computer needs source datasets to regenerate crops, but the
tracked final TIFFs and evidence can be inspected/used directly.

## Important scientific and implementation constraints

Partial labels are **1** reviewed clearing interior, **0** reviewed forest ROI,
and **255** everywhere else. `valid_mask` is exactly `label != 255`. Never turn
unknown background, empty targets or quality exclusions into negative labels.
The 13 controls lie inside original training grids; they are not independent
held-out negative examples or proof of permanent stability.

The original binary trainer is **incompatible with 255** until its loader,
loss and metrics explicitly ignore those pixels. It also has previously found
issues with datetime parsing, uint16 index subtraction, augmentation clipping,
channel scaling, inference paths/masks and truthful prediction display. Do not
feed this pilot into that trainer unchanged. The appropriate next milestone is
an ignore-aware loader/loss/metrics and a small baseline smoke run; retain
geographic separation and do not claim independent accuracy from this pilot.

The current pilot is strong-signal/clear-image biased. The source called Spring
has qualifying June examples; it does not establish seasonal coverage. Exact
selection filter order is saved in selection_order_clarification.json. Only
windowed local observations support `(last observed forest, first clearing]`;
these are not exact felling dates or globally earliest sightings.

The user requires Mac responsiveness: one `nice +10` worker, native threads 1,
no GPU, monitored 512 MiB worker budget, 1.5 GiB available-system-memory reserve,
300-second per-worker limit. Use `validate_sample.run_guarded` for new pixel
work. Actual largest pilot preparation worker: 417.6 MiB. Export: 29.51s /
207.5 MiB. Independent verification: 1.02s / about 119 MiB. The earlier full QA
scan used max 464.9 MiB and 2h49m sequential worker time.

## Frozen QA identity

Keep policy
`fe2472ec030f466bebbb9dc2cb033ed6175de489bcf022f650081dd8e9347560`
unchanged when reusing completed QA/cache. It hashes these existing files:
`assess_pixel_quality.py`, `pixelqa_masks.py`, `pixelqa_scene_metadata.py`,
`validate_sample.py`, `annotation_scene_inventory.py`, plus config/package/model
versions. Editing them invalidates QA resumption. New curation code was added
separately: `prepare_curated_pilot.py`, `curated_pilot_review.py`,
`curated_pilot_exports.py`, `build_curated_pilot.py`, `verify_curated_pilot.py`.

One baseline geometry, `event-8179ed3a1e4286448a17`, becomes invalid after
projection and emitted distance warnings in preparation. Provenance proves it
is >407m from every initial crop and >40km from replacements, beyond the 20m
exclusion margin. New export code conservatively bbox-filters and repairs
relevant invalid geometries without changing the stored catalogue or frozen
QA files. The warning did not alter any selected control exclusion.

## Commands

Run from the project root with the existing environment:

```bash
/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python -B -m unittest discover -s Tests

/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python -B Solutions/verify_curated_pilot.py --output Datasets/CuratedPilot/v1/verification-recheck
```

The independent verifier checks local original preparation/cache/source
fingerprints, so it requires those local inputs. Use a fresh output directory;
completed guarded outputs are intentionally not overwritten. No full dataset
reprocessing or model training is needed to inspect this completed milestone.
