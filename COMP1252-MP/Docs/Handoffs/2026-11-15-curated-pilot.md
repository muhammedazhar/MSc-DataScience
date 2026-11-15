# Deforestation pilot handoff

## Objective and authorization

The user wants to locate deforestation and estimate the first usable image in
which it becomes visible. The annotation `img_date` is an image observation
date, not the physical felling date. The user approved building a small reviewed
training pilot, including timeline evidence, stable-forest controls and exports.
They also explicitly requested this separate branch and a durable handoff.

Branch: `codex/curated-deforestation-pilot` (created from `master`).
Repository root: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience`.
Project root: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.

Keep Mac responsiveness: one low-priority worker (`nice +10`), native threads 1,
no GPU, monitored 512 MiB worker RSS, 1.5 GiB available-system-memory reserve,
300-second per-worker timeout. Existing `validate_sample.run_guarded` supplies
atomic staging, watchdog and cleanup. Do not start multiple pixel workers.

## Completed foundations

- Correctly aligned 64-pixel / 10-m single-sample validator and signed indices.
- Annotation provenance audit: 5,696 raw annotations -> 3,911 unique events;
  601 geographic groups with frozen train/validation/test assignments.
- Full pixel-quality screening: 6,948 pairs / 55 batches / zero errors.
  2,924 automatic passes pending review, 3,405 excluded, 578 visual-review,
  41 label-review cases. Maximum scan-worker memory 464.9 MiB.
- Review queue: 682 anchor events / 7,001 links; 554 events with proposals,
  124 alternate-imagery/quality-review and 4 label-review events.
- Independent full verification passed all 6,948 masks and reconstructed
  targets, 5,096 observation caches, 37 scenes and 682 review entries.

Read `Docs/Guides/{SingleSampleValidation,AnnotationAudit,PixelQuality}.md`.
Large generated products stay local under ignored `Datasets/Validation/`,
`Datasets/AnnotationAudit/v1/` and `Datasets/PixelQA/v1/`. Raw satellite scenes
and existing forest models were not changed.

**Preserve the frozen QA policy.** Its identity is
`fe2472ec030f466bebbb9dc2cb033ed6175de489bcf022f650081dd8e9347560`.
It hashes `assess_pixel_quality.py`, `pixelqa_masks.py`,
`pixelqa_scene_metadata.py`, `validate_sample.py`, and
`annotation_scene_inventory.py`, plus configuration/package/model versions.
Editing those processing files invalidates existing caches/resumption. New
curation helpers were deliberately added separately.

## Current pilot status — unfinished checkpoint

The new pilot code exists and has focused tests:

- `Solutions/prepare_curated_pilot.py`: guarded training-only timeline batches.
- `Solutions/curated_pilot_review.py`: explicit review/evidence validator.
- `Solutions/curated_pilot_exports.py`: seven bands/date -> correctly scaled
  pre9/post9/(post-pre)9 features; strict partial-label exports and round-trip
  verification.
- `Solutions/build_curated_pilot.py`: guarded review-bound export orchestrator.

All **28** candidate cases are prepared locally under
`Datasets/CuratedPilot/v1/{review_parts,replacement_parts}/` (17 initial and 11
replacement cases). Preparation is finished; no pixel worker should remain.
The initial preparation peak was 417.6 MiB. Each case has `case.json`,
`timeline.png`, `pair.png`, `target_mask.tif`, `positive_interior.tif` and
`proposed_control.tif`. Timelines include every local same-year observation in
the declared +/-45-day window; they do not establish global earliest clearing.

Durable branch evidence is copied to `Docs/Reviews/CuratedPilotV1/` (about
6.5 MiB). This includes all 28 case JSONs, PNGs and ROI masks, selection and
replacement provenance, the original 17 reviews, and eight draft replacement
timeline interpretations. The tracked `prepared_case_index.json` maps each
copy to its original local case path. Raw scenes/caches remain local and are
needed for final feature export; copying code alone to another computer will
not provide them. The generated `.gitignore` directories are intentional.

**Only two initial cases currently have formal accepted decisions** in local
`Datasets/CuratedPilot/v1/reviews.json` (also copied into tracked evidence):

| Event | Observed transition interval | Positive interior / stable pixels |
| --- | --- | ---: |
| `event-e691d44c335d87649310` | (2019-06-11, 2019-06-26] | 25 / 256 |
| `event-263e9027a6142dc9fc84` | (2019-08-15, 2019-08-25] | 13 / 256 |

Both timelines AND primary pairs were visually inspected by the root assistant.
An independent agent checked all review hashes, exact timeline coverage and
numeric QA compatibility. Later clear observations confirm persistence; each
forest control has >=3 clear dates. June 16 for the first event remains
explicitly uncertain and June 21 obscured. These are assistant-reviewed pilot
labels, not expert ground-truth certification or physical felling dates.

The other 15 initial cases are rejected in the formal review file: mostly
boundary-dominated targets with fewer than four interior pixels, one with no
fully clear later persistence observation, one with its first clear post-image
on the last available date. Do not relabel their unreviewed pixels as negatives.

Eight replacement timelines have been inspected, but **formal approval is still
pending their primary pair PNG inspection and validator confirmation**. Their
per-date forest/clearing/obscured/uncertain interpretations and control notes
are saved in tracked
`replacement_timeline_inspection_drafts.json`. None of those drafts is a
training approval. The remaining three replacement timelines are uninspected:

- `replacement_parts/00002/event-0012effe85aa26b69541/`
- `replacement_parts/00002/event-65fa6d8d7a0c82498340/`
- `replacement_parts/00002/event-e932f0d28b2819f8368d/`

No final `Datasets/CuratedPilot/v1/dataset/` export has been run at this
checkpoint. The accepted count must follow evidence; there is no requirement
to force twelve examples or artificial source balance.

## Next actions

1. Inspect the eight pending primary pair PNGs using the saved draft timeline
   notes, then inspect the remaining three timelines and primary pairs. Accept
   or reject based on actual forest-to-clearing visibility, later persistence
   and stable control-region evidence; metadata/NDVI alone is insufficient.
2. Append explicit observations for every prepared scene/date to local
   `reviews.json`. Each row needs `case_file`, exact `event_id`/`pair_id`,
   `visual_review` and `temporal_label_review` (`accepted`/`rejected`),
   `qa_policy_id`, reviewer/notes, all observation states, and SHA256 hashes of
   `case.json`, `timeline.png`, `pair.png`. Also retain ROI mask hashes.
   `validate_review(case, review)` derives bounds; never invent dates from
   annotation dates or NDVI. Keep all rejected cases and reasons.
3. Validate every review, preserving accepted event/group uniqueness and the
   unchanged frozen training grids. Copy finalized reviews into tracked evidence.
4. Run the guarded export once reviews are final:

   ```bash
   /Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python -B \
     Solutions/build_curated_pilot.py \
     --reviews Datasets/CuratedPilot/v1/reviews.json \
     --output Datasets/CuratedPilot/v1/dataset
   ```

5. Independently check every exported small TIFF, feature differences, exact
   frozen grid/split joins, review hashes, supervised ROI/QA intersections,
   counts and visibility/control evidence. No broad JP2 reread is necessary.
6. Complete `Docs/Guides/CuratedPilot.md`, README pointer, concise final report,
   resource measurements and this handoff; run full test suite and diff checks.
   Commit the completed milestone on this branch.

Partial supervision is essential: **1** = explicitly reviewed clearing interior;
**0** = explicitly reviewed stable-forest ROI; **255** = everything else.
Controls stay INSIDE existing training crops, avoiding new-grid split leakage.
The original binary trainer is incompatible until its loader/loss/metrics
explicitly ignore 255. Do not run it on this dataset yet. No forest-model
training has been authorized as part of this dataset-construction milestone.

## Known qualifications

- Strong-signal, clear-image candidate selection is biased and training-only.
  The qualifying Spring-source examples are from June, not seasonal coverage.
  Read the selection-order clarification instead of assuming absolute nearest
  pre-image selection regardless of the other eligibility filters.
- One baseline-only geometry, `event-8179ed3a1e4286448a17`, becomes invalid after
  projection and emits Shapely distance warnings during preparation. Its
  bounding rectangle is >407 m from every initial selected crop and >40 km
  from replacements, outside their 20 m exclusion range. Replacement provenance
  records this proof. New export code uses conservative bbox prefiltering and
  `make_valid` for relevant geometries; original catalogue/policy is untouched.
- Existing QA summary rejection-path tests emit file-closing ResourceWarnings;
  they pass. Do not casually edit frozen processing files to silence them.
- Some subagents reached service usage limits. Their completed files remain
  available; do not assume an errored agent completed any requested follow-up.

## Validation command

```bash
/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python -B \
  -m unittest discover -s Tests
```

At this checkpoint **136 tests passed** in the full suite (0.514 seconds),
including preparation, review validation, exports, orchestration and the earlier
audit/QA/resource checks. The known rejection-path ResourceWarnings remain.
Check current full-suite output if another agent changes tests.
