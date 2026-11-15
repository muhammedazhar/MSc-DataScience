# Annotation provenance, cleaning and geographic splits

This milestone prepares a traceable catalogue and fixed dataset manifest before
more imagery is processed. It reads small annotation, XML and GML files and file
paths. It does not open satellite raster pixels, load models, train, or alter
the original datasets.

## Reproduce

From the project root, use the existing environment:

```bash
conda run -n COMP1252-MP python -B Solutions/audit_annotations.py
```

The default destination is `Datasets/AnnotationAudit/v1/`. Existing output is
never overwritten. A new reproducibility run can use:

```bash
conda run -n COMP1252-MP python -B Solutions/audit_annotations.py \
  --output Datasets/AnnotationAudit/recheck
```

The script verifies the original annotation snapshots against
`Datasets/AnnotationSources/sources.json`, then works offline. Generated audit
artifacts are excluded from Git; the source snapshots, scripts and tests remain
available for reproducibility.

## Verified findings

- **5,696 local rows match original author annotations**, using exact normalized
  2D geometry, tile and observation date. No coordinate rounding is used for this
  source match.
- **1,785 duplicate rows are removed**, leaving **3,911 unique events**. All
  valid local source indices remain traceable through aliases.
- **1,799 events have temporal-source membership**. **2,112 are baseline-only**
  and require temporal-label review before use for first-visible-change training.
  Some events belong to multiple original sources; those memberships are retained.
- **Two footprints have conflicting dates**, involving four events. Both dates
  remain visible and are blocked from confident temporal supervision. The
  original zero-based local indices are 79/1445 (19 September 2017/2018) and
  5512/5602 (10/30 August 2019).
- **85 original baseline records lie in tile 37UCR**, outside the local study.
  **Five original spring records lack tile IDs**. They are preserved as source
  exclusions, not silently inserted into the local catalogue. Candidate crops
  intersecting the unresolved spring footprints are blocked for review.
- All **65 existing PLOT names map to canonical events** despite tiny previous
  reprojection roundoff. PLOT-00056 and PLOT-00063 now resolve to the same event,
  group and split.

All source-feature indices in the audit are zero-based positions in their
original GeoJSON `features` arrays. Stable event IDs are the preferred join keys.

## Frozen geographic plan

The audit uses the reviewed **64 x 64 pixel, 10 m** crop plan. Full annotation
extents and conservative crop envelopes are combined; a 20 m margin covers
window snapping. Connected overlap components stay together across dates and
tile IDs. This protects against both duplicate examples and shared image pixels
crossing dataset splits.

There are **601 geographic groups**. A seeded hash (seed 42) assigns complete
groups to nominal 70/15/15 train/validation/test proportions:

| Split | Geographic groups | Canonical events |
| --- | ---: | ---: |
| Train | 416 | 2,944 |
| Validation | 91 | 514 |
| Test | 94 | 453 |

Group sizes vary, so event proportions differ from group proportions.
Independent checks found **zero cross-split footprint overlaps**. Increasing
crop size, moving centres or tiling large annotations requires recomputing this
plan. Seventy annotations exceed 640 m in width or height and are explicitly
flagged for a tiling decision.

## Image availability and candidates

All **50 local scenes** have complete supported metadata: 12 in 36UXA and 38 in
36UYA. Metadata completeness is not a claim about cloud-free pixels.

The audit lists **11,156 distinct 5–30 day bracketing pairs**. Of these, **4,208
are blocked** with explicit reasons and **6,948 can proceed to pixel review**.
**2,878 events have no local bracketing pair** within that interval. Event
manifests include observation dates, available nearby image dates and candidate
IDs, so missing coverage remains visible.

| Split | Pairs for pixel review | Distinct anchor events | Geographic groups |
| --- | ---: | ---: | ---: |
| Train | 5,583 | 541 | 94 |
| Validation | 540 | 53 | 14 |
| Test | 825 | 88 | 21 |

These are correlated pairs, not independent locations. All splits retain both
tiles and all three temporal source collections before pixel QA. Evaluation
coverage must be reassessed after quality exclusions.

No new manifest pair is marked `training_approved`. Scene-level cloud
percentages and well-formed cloud masks cannot certify usable pixels, as the
single-sample milestone already demonstrated. Crops near conflicting labels,
unverified new baseline labels or unresolved original annotations remain
blocked. Pair targets list temporally sourced label events separately from all
context annotations; an absent annotation is not a verified negative label.

## Deliverables

| File under `Datasets/AnnotationAudit/v1/` | Purpose |
| --- | --- |
| `audit_report.md` | Readable findings, exclusions and split coverage |
| `annotations_clean.geojson` | Canonical annotations, provenance, dates, flags and split IDs |
| `splits.json` | Frozen geographic groups, seed and crop assumptions |
| `annotation_manifest.jsonl` | Per-event imagery availability, candidate IDs and review counts |
| `candidate_pairs.jsonl` | Candidate scenes, dates, crop grids, labels and blocking reasons |
| `scene_inventory.jsonl` | Metadata-only local scene inventory |
| `source_aliases.jsonl` | Every valid original local row mapped to an event and split |
| `sampled_plot_aliases.jsonl` | Existing PLOT names mapped to events and splits |
| `source_provenance.json` | Original source matching and coverage |
| `duplicate_events.jsonl`, `date_conflicts.jsonl` | Preserved duplicate and ambiguous-date evidence |
| `original_source_exclusions.jsonl`, `invalid_local_records.jsonl` | Source exclusions and unusable local records |
| `metadata.json` | Checksums, checks, statistics and measured resources |

JSONL files contain one JSON object per line for streaming consumption.

## Verification and Mac capacity

Run the small synthetic regression suite with:

```bash
conda run -n COMP1252-MP python -B -m unittest discover -s Tests -v
```

**31 tests pass**, covering geometry identity, aliases, temporal conflicts,
transitive overlap, deterministic splits, calendar-day boundaries, missing
metadata, original-source uncertainty, resource safeguards and the earlier
single-sample checks. An independent output check verified all 5,696 aliases,
all pair links, split membership and 17,694 overlapping footprint edges.
A second complete run reproduced identical split assignments, pair contents,
scene inventory, aliases and source matching.

The final audit took **3.55 seconds**, with a worker high-water footprint of
**204.8 MiB** and at least **2.97 GiB of available system memory** at monitor
samples. It uses one low-priority worker, native thread limits of one, no GPU,
a 512 MiB RSS budget, a 1.5 GiB system-memory reserve, and a 180-second timeout.
The watchdog checks every 250 ms and checks final high-water memory before
publishing output. Failed runs discard only their own staging directory.
These are monitored safeguards, not an instantaneous operating-system cap.

The next stage is per-crop pixel-quality assessment and review, followed by a
small controlled dataset build using this catalogue and these frozen splits.
Original imagery, annotations and existing model files remain unchanged.
