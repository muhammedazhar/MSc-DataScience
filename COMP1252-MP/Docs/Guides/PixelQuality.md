# Per-crop image-quality screening

This stage screens the frozen image-pair candidates from the annotation audit.
It preserves their pair, event, geographic-group and split identities. The
packaged s2cloudless model supplies cloud probabilities; the deforestation
models are not loaded or trained.

The current candidate pool contains 6,948 pairs. Completion and result counts
come from `Datasets/PixelQA/v1/summary/metadata.json` and the generated
`quality_report.md`. A run is complete only when the summary verifies exact
coverage of that frozen pool. Partial batches are not a completed assessment.

## Run and resume

Use the existing project environment from the project root. The default command
runs a small pilot:

```bash
conda run -n COMP1252-MP python -B Solutions/assess_pixel_quality.py
```

The pilot selects 24 group-diverse pairs across the three fixed splits and two
tiles, plus three previously inspected PLOT controls. Selection is deterministic
and does not depend on observed quality outcomes. Its previews are written to
`Datasets/PixelQA/v1/pilot/previews/`.

To assess the full pool in sequential batches and publish the summary:

```bash
conda run -n COMP1252-MP python -B Solutions/assess_pixel_quality.py --mode all
```

Defaults are 128 pairs per batch, one worker and a 300-second limit per worker.
Completed, compatible batches are reused when this command is repeated. A
stopped or failed worker discards its temporary batch output; compatible
observation-cache files remain reusable. There are no parallel scene workers.

Individual batches and the final coverage check can be run separately:

```bash
conda run -n COMP1252-MP python -B Solutions/assess_pixel_quality.py \
  --mode batch --batch-index 0

conda run -n COMP1252-MP python -B Solutions/assess_pixel_quality.py \
  --mode summarize
```

Batch indices are zero-based. Resumption checks the original batch size, seed,
selection and policy. Keep these settings unchanged within an output version.
If the policy or inputs change, select a new `--output-base`, such as
`Datasets/PixelQA/v2`, instead of mixing incompatible results. The default
`--cache-root` is `Datasets/PixelQA/cache`; compatible observations can be reused
independently of the output version.

## Geometry, bands and validity

Each core is **64 x 64 pixels at 10 m**, covering 640 x 640 metres on its frozen
geographic grid. Quality assessment adds **1,200 m of context on every side**.
The context helps detect clouds whose shadows can fall inside the core.

For each band, the reader decodes a small native raster window with an
interpolation halo, then resamples onto a shared 10 m context grid. It retains
the exact 10 m core and averages aligned 2 x 2 samples to produce the **20 m
cloud-model context grid**. Quality masks and probabilities are expanded back
to the core by nearest-neighbour repetition. Averaging or enlargement does not
increase the native information content of the 20 m and 60 m source bands.

The s2cloudless inputs use this exact order:

```text
B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12
```

Bands are converted to reflectance using the product's quantification value.
Current local products use pre-04 processing baselines with zero radiometric
offsets. Unsupported nonzero offsets and missing prerequisites remain explicit
processing failures rather than producing an all-clear fallback.

Native DN **0** and **65535** are invalid. Native validity is resampled with a
conservative minimum operation, independently of bilinear reflectance, so
invalid categories are not simply averaged into plausible values. A 20 m block
is valid only when all four contributing 10 m validity samples pass. Available
cloud, nodata, saturation, defect and technical-quality polygons are checked
for the union of the feature and cloud-model bands. Non-finite model inputs and
invalid probabilities also remain invalid.

The cloud classifier receives a finite placeholder at invalid context pixels,
but those pixels are restored to unknown/invalid afterwards. Placeholder
predictions are never used as evidence of clear observations.

## Frozen screening rules

| Component | Current rule | Interpretation |
| --- | --- | --- |
| Cloud | s2cloudless probability >= 0.4, or a supplied product-cloud polygon | Combined cloud evidence |
| Cloud buffer | Physical disk dilation by 60 m | Conservative cloud-edge exclusion |
| Suspected shadow | Cloud projection opposite the mean solar azimuth, up to 1,000 m, intersected with valid B08 reflectance < 0.15 | Unvalidated L1C review screen |
| Haze risk | B02 > 0.15 and B02 - B04 > 0.03 | Unvalidated blue-excess diagnostic |
| Invalid | Native invalid categories, product QA, missing/non-finite spectral data or invalid model probabilities | Unusable or unknown pixels |

Projection uses a north-up raster: a southern sun projects shadows north and
an eastern sun projects them west. Shifts are clipped at context edges and
never wrap. Dark forest alone is not labelled shadow; it must intersect a
projected cloud path. Buffered cloud pixels are kept separate from suspected
shadows. Missing or invalid mean solar angles cause explicit failure.

The shadow and haze thresholds are **not validated cloud/shadow ground truth
for this L1C collection**. The dark-NIR example in the official Earth Engine
tutorial uses surface reflectance, which differs from these inputs. The search
is bounded to 1 km; it cannot rule out shadows from farther clouds or all
terrain shadows. Any invalid padded context is recorded as incomplete context.

Rules were frozen before reviewing held-out imagery. Test imagery is not used
to tune thresholds. A future policy should be developed using training and
validation material and receive a new policy identity.

## Quality states and metrics

The pre/post masks are combined conservatively. Counts are recorded both for
the whole core and for the reference new-loss target. Every fraction uses all
pixels in its named region as the denominator; an empty region has `null`
fractions. The unusable count is a union, so overlapping cloud, shadow, haze
and invalid flags do not double-count pixels.

| State | Meaning |
| --- | --- |
| `needs_label_review` | The reference new-loss target rasterizes to an empty mask |
| `excluded_by_automatic_quality_screen` | Invalid pixels or cloud/buffer affect the reference target |
| `needs_visual_review` | Suspected shadow/haze affects the target, quality flags cover more than 10% of the core, or context is incomplete |
| `passed_automatic_screen_pending_visual_review` | None of the configured automatic flags requires exclusion or escalation |
| `processing_error` | Required processing failed; the recorded error must be resolved |

Every result retains `training_approved: false`. Passing the automatic screen
still requires visual assessment and temporal-label review. An attempted pair
with a processing error is not counted as successfully screened. Bright surface
features can trigger false-positive cloud predictions, and subtle cloud/haze
can be missed. These flags support inspection rather than certify exact timing
or every target boundary.

Reference targets come from temporally sourced catalogue events on the exact
frozen crop grid. Baseline-only labels are not introduced into these targets.
Targets remain observation-date annotations, not exact physical felling dates.

## Outputs

| Location | Contents |
| --- | --- |
| `Datasets/PixelQA/v1/pilot/` | Fixed pilot selection, results, georeferenced masks and inspection previews |
| `Datasets/PixelQA/v1/batches/00000/` and subsequent indices | Frozen batch selection, per-pair results, masks, policy identity and resources |
| `Datasets/PixelQA/v1/summary/quality_manifest.jsonl` | Consolidated pair-level results with their source batch paths |
| `Datasets/PixelQA/v1/summary/quality_report.md` | Actual completed-run counts, limitations and resource totals |
| `Datasets/PixelQA/v1/summary/metadata.json` | Coverage, split, uniqueness and policy checks; processing errors and measured resources |
| `Datasets/PixelQA/v1/pilot/visual_review.json` and `visual_review.md` | Assistant inspection of five training previews, with artifact hashes and unresolved quality concerns |
| `Datasets/PixelQA/v1/verification/` | Independent artifact-integrity results and the exact verification script used |
| `Datasets/PixelQA/cache/<policy-id>/*.npz` | Reusable observation masks, probabilities, RGB, NDVI and grid/source metadata |

Per-pair GeoTIFFs use the original core CRS, transform and dimensions. Their
six bands are `cloud`, `cloud_buffer`, `shadow_risk`, `haze_risk`, `invalid` and
`reference_new_loss`, each with 0/1 values. Downstream processing must read the
invalid band explicitly; these are QA artifacts rather than approved training
targets. NPZ arrays retain their geographic metadata and are loaded without
pickle. Both observations' separate quality metrics and cache references are
preserved in each result.

The generated folders are excluded from Git. Original satellite images,
annotations, split plans and existing forest models remain unchanged.

## Event-level review queue

After the full quality summary has passed its publication checks, build the
event-level review queue:

```bash
conda run -n COMP1252-MP python -B Solutions/build_quality_review_queue.py
```

The defaults read the completed
`Datasets/PixelQA/v1/summary/quality_manifest.jsonl` and the frozen
`Datasets/AnnotationAudit/v1/annotation_manifest.jsonl`. The summary's adjacent
`metadata.json` must confirm complete candidate coverage, unique pairs, one
policy and preserved splits. The postprocessor reads JSON metadata only; it
does not rerun image screening, render new previews or run a forest model.

| File under `Datasets/PixelQA/v1/review/` | Purpose |
| --- | --- |
| `review_queue.jsonl` | One record per represented anchor event, retaining all its screened candidate IDs, statuses and reasons, with up to three ranked candidates |
| `quality_result_index.jsonl` | Compact index of every screened input pair, including batch paths and references to its QA artifacts |
| `review_report.md` | Event-level queue counts, ranking method and remaining review requirements |
| `metadata.json` | Input and implementation hashes, policy identity, coverage checks, event counts and worker resources |

Existing queue output is not overwritten. Use a new `--output`, such as
`Datasets/PixelQA/v1/review-recheck`, for a separate reproducibility run. Custom
input locations are available through `--quality-manifest` and
`--annotation-manifest`; their bidirectional pair/anchor links and frozen
group, split, tile and observation dates must agree.

The top-three ranking is deterministic, using this order:

1. Quality status: automatic-screen pass, visual-review requirement,
   label-review requirement, automatic exclusion, then processing error.
2. Earliest post observation on or after the known annotation observation date.
3. Closest preceding observation before that annotation date.
4. Shortest image-pair interval.
5. Stable pair ID to break any remaining ties.

Only candidates with status `passed_automatic_screen_pending_visual_review`
or `needs_visual_review` can appear in `proposed_pair_ids`. All candidates
remain traceable, including those outside the top three. Excluded pairs and
processing errors are retained rather than disappearing from the queue. An
event with only excluded candidates receives
`requires_alternate_imagery_or_quality_review`; empty-label and
processing-recovery cases have their own states. Catalogue events with no
screened candidate in the input remain in the original annotation manifest
and are counted outside this queue.

This ranking uses the known annotation date to curate examples for inspection.
It does not infer an onset date, establish the first physical clearing date or
grant training approval. Forest-model predictions and Dice/performance scores
are not used. Every queue entry retains unreviewed visual/temporal-label states
and `training_approved: false`. Follow the stored batch paths to the original
quality artifacts; full-batch records may have no preview image.

## Cache and resumption identity

Observation cache keys include:

- The frozen quality configuration and hashes of the processing/helper code.
- The packaged cloud-model hash and s2cloudless/LightGBM/runtime package versions.
- Product/tile metadata identity, QA-file content hashes, and raster-file sizes
  and modification timestamps.
- The exact core geographic grid.

Raster files are not hashed in full, avoiding large sequential imagery reads.
Their size/timestamp fingerprint assumes that input files are preserved rather
than externally rewritten with the same size and timestamp. The in-memory
cache retains at most twelve cores; disk cache files are written atomically.

Batch and summary resumption additionally check hashes of the audit metadata,
candidate manifest and cleaned annotation catalogue, plus their frozen
selection. Source-file statistics are checked for changes. The final summary
requires unique, exact candidate coverage under one policy and consistent
pair/group/split identities. Incompatible or incomplete output is not silently
treated as current.

## Mac capacity and verification

The workflow is designed for the user's **18 GiB Mac** and its responsiveness:

- One low-priority worker, with a `nice` increment of 10.
- Native thread limits of one, explicit LightGBM `num_threads=1`, and one
  OpenCV thread. No GPU work.
- A **512 MiB RSS budget** and **1.5 GiB available-system-memory reserve**.
- A **300-second timeout per worker**, a 32 MiB GDAL cache and 16 MiB warp buffer.
- Monitoring every 250 ms and a final high-water-memory check before publishing
  the batch. Failed workers discard their own staging directories.

These are monitored safeguards, not an instantaneous operating-system memory
cap. Actual peak memory and elapsed time belong to each batch's `metadata.json`;
the summary reports the maximum worker high-water footprint and total worker
time. No guessed full-run resource measurements are recorded here.

At implementation verification, **90 small tests passed** across the project.
They include band prerequisites, solar metadata, geometry/validity alignment,
cloud thresholds and buffers, shadow direction and edge clipping, custom haze
thresholds, region denominators, scientific quality states, cache identity and
the earlier audit/resource regressions. Review-queue checks cover frozen joins,
deterministic ranking, retained exclusions and the separation of curation from
approval. Tests use small synthetic fixtures and mocked workers. Run them when
image processing is idle:

```bash
conda run -n COMP1252-MP python -B -m unittest discover -s Tests -v
```

## Completed v1 assessment

The completed run screened **6,948 pairs in 55 batches**, with **zero processing
errors**, under policy
`fe2472ec030f466bebbb9dc2cb033ed6175de489bcf022f650081dd8e9347560`.
The audit identities and geographic splits were preserved.

| Automatic result | Pairs |
| --- | ---: |
| Passed automatic screen, pending visual review | 2,924 |
| Excluded by automatic quality screen | 3,405 |
| Needs visual review | 578 |
| Needs label review because the rasterized target is empty | 41 |

The review queue represents **682 anchor events** and preserves **7,001
event-pair links**. Of these events, **554** have inspection proposals, **124**
require alternate imagery or quality review, and **4** require label review.
Another **3,229** catalogue events remain outside this screened queue; their
audit exclusions and missing observations remain recorded. Empty targets are
not verified no-change examples.

The largest scan-worker high-water footprint was **464.9 MiB**. Sequential
worker time totalled **10,127.89 seconds**, approximately **2 hours 49 minutes**;
this is worker time rather than a wall-clock guarantee. The smallest sampled
available-system-memory value across the 55 scan batches was **4.10 GiB**.
The final cached pilot took **6.61 seconds** at **321.7 MiB** and the queue
postprocessor took **0.51 seconds** at **46.3 MiB**.

Independent verification checked all **6,948** six-band binary GeoTIFF masks,
reconstructed all temporal targets from the frozen annotation IDs, recomputed
their numeric metrics and quality states, and verified **5,096** observation
caches from **37** scenes. All **682** review entries and their top-three
rankings passed. It read no source JP2 pixels and ran no model predictions.
This check took **25.95 seconds** at **182.7 MiB**, with a minimum sampled
available-memory value of **2.95 GiB**. Its report is
`Datasets/PixelQA/v1/verification/verification_report.md`.

The exact verifier used is retained with that generated report. To repeat the
check into a new output directory:

```bash
conda run -n COMP1252-MP python -B \
  Datasets/PixelQA/v1/verification/verify_pixelqa_completion.py \
  --project-root "$PWD" --output Datasets/PixelQA/v1/verification-recheck
```

The 90-test suite also passed after completion. Rejection-path summary tests
currently emit file-closing `ResourceWarning` messages; they do not fail, and
the processing source remains frozen to preserve this run's policy identity.

Five **training-split** previews were inspected without tuning the frozen
rules. Two visibly cloud-obscured pre-images agreed with automatic exclusion.
PLOT-00026 was visually consistent with a small clearing expansion, subject to
the limited resolution of its boundary. PLOT-00004 retained a suspected blue
veil/contrast concern despite zero automatic flags, so it requires haze review.
Another example had an isolated off-target cloud flag that could be a bright
surface false positive. These inspections establish neither cloud/shadow
ground truth nor exact first-visible dates; the detailed records retain
`training_approved: false`. The event queue remains unreviewed and does not
automatically import these inspection notes.

The next data step is visual and temporal-label curation of proposed examples,
plus independently verified stable-forest/no-change controls for false-alarm
evaluation. The current event-based candidate pool cannot supply those controls
merely by treating empty or excluded targets as negatives.

## Primary references

- [s2cloudless input bands, scaling and baseline harmonization](https://github.com/sentinel-hub/sentinel2-cloud-detector#input-sentinel-2-scenes)
- [LightGBM 4.5.0 thread control](https://lightgbm.readthedocs.io/en/v4.5.0/Parameters.html#num_threads)
- [Official Earth Engine cloud/shadow example and parameter limitations](https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless)
- [Rasterio windowed reading](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html)
- [Rasterio resampling methods](https://rasterio.readthedocs.io/en/stable/topics/resampling.html)
