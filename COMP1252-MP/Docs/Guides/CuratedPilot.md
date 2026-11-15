# Reviewed deforestation pilot

The completed pilot contains **13 image pairs from 13 training geographic
groups**, with **13 matched stable-forest control regions**. Of 28 candidate
cases assessed, 15 were rejected and retained with their reasons. Some were
rejected by geometry gates before imagery inspection; the assessment count is
not a claim that every rejected image was visually reviewed.

The dataset, decisions, source fingerprints, previews and verification are
saved on `codex/curated-deforestation-pilot`. This is a small development pilot,
not an independent evaluation set or expert-certified ground truth.

## Files and review evidence

- `Datasets/CuratedPilot/v1/dataset/report.md`: accepted intervals, rejection
  reasons, counts and links to inspection images.
- `Datasets/CuratedPilot/v1/dataset/training_manifest.jsonl`: 13 accepted
  examples with relative paths, frozen grids and visibility/control evidence.
- `Datasets/CuratedPilot/v1/dataset/review_decisions.jsonl`: all 28 dispositions
  and explicit states for every prepared observation.
- `Datasets/CuratedPilot/v1/dataset/examples/<event-id>/`: input features,
  supervision, valid mask and provenance metadata.
- `Datasets/CuratedPilot/v1/verification/`: independent verification report.
- `Docs/Reviews/CuratedPilotV1/`: tracked selections, all 28 prepared cases,
  preview images, masks and finalized review statements. Historical draft
  notes are explicitly superseded by `reviews.json`.

The small final dataset and verification are intentionally tracked despite the
broader generated-directory ignore rule. Large source imagery, preparation
outputs and observation caches remain local. The tracked preview copies allow
another agent to inspect evidence without regenerating satellite crops. The
tracked final feature TIFFs can be consumed without rereading source scenes.

## What the labels mean

Every example retains the original **64 x 64 pixels at 10 m** geographic grid.
The 27 float32 feature channels are:

```text
pre:        B02 B03 B04 B08 B8A B11 B12 NDVI NDMI
post:       B02 B03 B04 B08 B8A B11 B12 NDVI NDMI
difference: the same nine channels, post minus pre
```

Reflectance bands are divided by their product quantification. Signed NDVI,
NDMI and differences retain their own scales and are not clipped. Each export
is reread and checked against its in-memory arrays and frozen CRS/transform.

`supervision_mask.tif` is **partial supervision**:

| Value | Meaning | Pixels across pilot |
| --- | --- | ---: |
| 1 | Reviewed clearing interior | 151 |
| 0 | Reviewed stable-forest control region | 2,640 |
| 255 | Unreviewed or uncertain; ignore | 50,457 |

`valid_mask.tif` is exactly `supervision_mask != 255`. Positive regions use a
one-pixel eroded interior of the anchor-specific temporal reference; this pilot
requires at least four surviving interior pixels. Control squares are 16, 12
or 8 pixels wide, remain inside existing training grids, and avoid all known
catalogue loss footprints across dates/provenance, with a conservative margin.
Every accepted supervised pixel is finite, valid and free of the two dates'
cloud, buffer, shadow-risk, haze-risk and invalid flags. Automatic QA is also
checked against explicit visual inspection.

The 151 positive pixels represent **1.51 ha of supervised interiors**, not the
total deforested area of the events. Controls represent **26.4 ha** of reviewed
forest regions. Pixel counts do not supply independent sample counts.

## Timeline interpretation

Each prepared timeline includes every available local same-year observation
within 45 days of the annotation image date. The assistant recorded forest,
visible clearing, obscured or uncertain states for the target, and stable
forest, obscured or uncertain states for the proposed control. A clear claim
must also agree with zero numeric QA flags on its reviewed region.

Accepted examples require forest at the exported pre-image, clearing at the
post-image, at least one later clear image confirming persistence, and at least
three clear control dates including both pair endpoints. No later forest claim
may contradict persistent loss. Observation gaps and clouds remain explicit.

The observed interval is **(last observed forest, first observed clearing]**.
Its width ranges from **5 to 20 days** in this pilot. For example, the event
labelled on 21 July 2019 has forest visible on 1 July, so its evidence-based
interval is (1 July, 21 July], even though its exported pair starts on 26 June.
Another event retains uncertain 16 June and obscured 21 June observations,
giving the conservative interval (11 June, 26 June]. These are bounded local
observations, not exact felling dates or globally earliest sightings.

## Selection and limitations

The pilot favors clear crops with visible spectral contrast and sufficiently
large interiors. It is therefore biased toward easier examples. It contains
nine 36UXA cases, three 36UYA Summer-source cases and one 36UYA Spring-source
case. The Spring-source case is from June; source balance is not seasonal
coverage. The selection and replacement files retain the exact filtering
order, stable ranking, excluded groups and minimum-support criteria.

Controls are forest regions within the same training crops, not separate
held-out negative image pairs. They establish stability at inspected dates,
not indefinite absence of logging. Unannotated pixels and empty targets are
never silently promoted to negative labels. Human/expert adjudication and a
larger independent validation set remain necessary for performance claims.

All accepted grids remain disjoint from the original held-out conservative
footprints. Original imagery, annotations, QA products, split plans and forest
models are unchanged. The five files defining the completed QA policy remain
frozen. Relevant invalid projected catalogue geometries are repaired only
inside the new export exclusion helper; the stored catalogue is preserved.

## Reproduction and verification

Use the existing project environment from the project root. On this Mac:

```bash
PILOT_PYTHON=/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python
```

The selection/provenance is tracked under `Docs/Reviews/CuratedPilotV1/`; local
prepared-batch metadata and the source datasets are needed to regenerate the
feature exports. Prepared reviews are immutable and bound to case, PNG and ROI
hashes. Use a new output directory to repeat a run:

```bash
"$PILOT_PYTHON" -B Solutions/build_curated_pilot.py \
  --reviews Datasets/CuratedPilot/v1/reviews.json \
  --output Datasets/CuratedPilot/v1/dataset-recheck

"$PILOT_PYTHON" -B Solutions/verify_curated_pilot.py \
  --output Datasets/CuratedPilot/v1/verification-recheck

"$PILOT_PYTHON" -B -m unittest discover -s Tests
```

The independent check passed all 13 exports, all 28 dispositions and their
timeline evidence. It checks actual cache masks, signed differences, index
scaling, exact arrays/grids, reviewed ROI labels, timestamps, source statistics
and held-out footprint isolation. It reads no source JP2 pixels and runs no
model predictions. The full suite passed **136 tests**; earlier QA rejection
tests still emit their documented file-closing warnings.

Processing remains sequential, low priority and CPU-only, with native threads
limited to one, a monitored 512 MiB worker budget and a 1.5 GiB available-memory
reserve. The largest preparation worker used **417.6 MiB**. Feature export took
**29.51 seconds** at **207.5 MiB**; independent verification took **1.02 seconds**
at approximately **119 MiB**. No forest model was trained.

## Next implementation step

Implement an ignore-aware loader, loss and metrics before a baseline smoke
run. The original binary trainer is incompatible with 255 labels. For BCE,
select only `labels != 255` before computing targets/loss; metrics must use the
same reviewed-pixel mask. Preserve all channel scales and signed differences.
This pilot supports pipeline checks and learning experiments; it cannot provide
an unbiased accuracy estimate or independent test score.
