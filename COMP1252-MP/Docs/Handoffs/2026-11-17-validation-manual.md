# Manual validation timeline continuation

## Current state: approved review complete; case rejected

Continue on **`codex/curated-deforestation-pilot`**. Commit each completed,
verified logical change with this handoff updated and push to the SAME GitHub
branch. `CONTINUE.md` is the tool-neutral entry point for Codex, Gemini
Antigravity or another agent. No chat history or running worker needs recovering.

The authorized V4 task is DONE: all 12 local dates and all three fixed control
candidates were visually inspected before any deforestation predictions.
**One case rejected, zero labels approved, no first-visible interval or accuracy.**
The target's prior forest state and new forest-to-clearing transition are not
unambiguous. Whole-ROI control forest classes also remain uncertain. Do not force
labels or use rejected cases as negative-only samples. V3's automatic NDVI
no-control outcome and every original QA/model/split source remain unchanged.

Checkpoints, pushed to origin: `4950a0c` predeclared protocol; `fc6e976` tested
preparation; `3f70d93` tested independent verifier/source freeze/299 tests;
`4280811` complete packet and exact control roster committed BEFORE inspection.
The following completion commit contains all manual statements, final independent
review verification, guide updates and this finished handoff. Previous V3
milestone: `f46085f`. All local and remote checkpoints use the same branch name.

## Fixed case, image packet and complete disposition

Protocol: `Docs/Reviews/ValidationManualV4/protocol.json`, declared before new
pixels/images. It binds 14 source inputs, 30 immutable old implementations,
both new worker sources, complete timeline/cache plan, architecture and resources.
The fixed candidate remains:

- Event `event-bf3d28f31dcf19938962`.
- Pair `pair-430ec0ecae32ce98efcb`.
- Validation group `group-45f0c39e787ca8622918`.
- Stratum `36UYA_Spring_time-dependent`.
- Images 2019-04-02 -> 2019-04-27; annotation image date 2019-04-22.
- Original 64 x 64/10 m grid EPSG:32636, transform `[10,0,771700,0,-10,5537480]`.
- Target 36 pixels/interior 4; two-pixel square contexts 136/36, both fully
  inside, QA-clear and finite in all 27 original endpoint channels in V3.
- Original full-crop QA 56 pre-image shadow-risk pixels/0 post remains unchanged.

The complete same-year +/-45-day window 2019-03-08 through 2019-06-06 contains
12 local scenes. Eight caches were reused unchanged; four missing QA observations
(March 13/18, June 1/6) were prepared with the FROZEN cloud/quality policy. Exactly
44 fused-band window requests, 28 include-feature requests for per-date RGB/NDVI/
validity and four cloud-screening calls occurred. No endpoint 27-channel feature
stack was re-extracted, and no Torch or forest model was loaded. Zero raw-source
reads is true of the independent verifiers, NOT of this preparation worker.

At most three controls were selected BEFORE previews by largest 16/12/8 size,
then ascending row/column, using only full-context QA, finite 27/RGB/NDVI and
exclusions. No NDVI threshold, class appearance or model score entered selection.
Each full context avoids all-catalogue 20 m exclusions, target context and all
previously selected full contexts. All three are 16 x 16/256 pixels with 400-pixel
full contexts: C1 row 2 / col 2; C2 row 2 / col 22; C3 row 14 / col 42. No square was redrawn,
shrunk, replaced or selected after inspection. All three dispositions are kept.

Five committed PNGs were opened: `timeline-01.png`, `timeline-02.png`,
`timeline-03.png`, `pair.png`, `target-detail.png`, under `preparation/`.
Every date has plain fixed 0-.30 RGB, separate exact overlays and frozen QA;
nearest-neighbour target detail adds no information beyond the original 10 m pixels.

The plain target detail shows a tan/grey vertical feature and mixed/dark adjoining
pixels already on April 2, with substantially similar appearance on April 27.
March 13 also has a bright/mixed feature there. Later surroundings green, but
May 17/June 6 do not establish forest across the exact full target before clearing.
The five target-context-clear dates (March 13, April 2/27, May 17, June 6) remain
**uncertain**. Seven other target dates retain flagged-context **obscured**
abstentions. This does not prove no historical forest loss or an exact alternative
land-cover class; it establishes insufficient evidence for these labels/dates.

C1 is the strongest plausible forest patch, but its whole 256-pixel forest class
at both endpoints is uncertified in dark/smooth early 10 m RGB. C2 has an upper
bright linear margin; C3 has boundary/mixed-texture ambiguity. All clear-context
control states remain uncertain; other dates retain flagged-context abstentions.
No definite nonforest diagnosis is made. A separate assistant critique inspected
the same five images and agreed with these conservative uncertainties. Neither
assistant review is expert ground truth.

`reviews.json` binds the exact protocol, raw preparation metadata and all six
artifacts; it retains 12 target-date statements, 36 control-date statements, all
three rejected control dispositions, reasons, false approvals and null interval.
`visual_review.md` explains observations and includes the complete context-QA
count table. In particular April 12 has 10 target-context flags although its four
interior pixels' context is clear; April 22 annotation-date context has 136/136 flags.
Neither a clean small interior nor the annotation date can certify onset.

## Independent checks, tests and Mac resource proof

Packet-only verification PASSED before inspection. Final review verification
also PASSED, reconstructing the rejected case, every control disposition and
absence of an interval/selected control. Separate NumPy 25-shift halos/prefix
search reconstruct the numerical roster and per-date metrics. The verifiers
check complete scene inventory, 696 source-file size/mtime fingerprints, exact
cache source/grid/policy keys/hashes, all 27 saved endpoint features and V3 identities,
six artifacts/five views, 16 plain RGB and 14 QA panels against actual saved arrays.
Overlay image hashes, masks and source code are bound; overlay pixels are not
independently re-rendered. Integrity/decision-rule verification does not certify
visual class truth. No raw satellite bands or QA/forest models are rerun there.

**299 tests pass** (previous 278+8 preparation+13 verifier), no failures/errors/skips.
The actual-render synthetic integration initially caught coordinate labels
encroaching on one column of a plain panel. Moving label origin from x-15 to x-12
fixed it before any real image processing; strict pixel comparisons were retained.
The initial 13-test attempt failed 1 case and published no output. This history is
explicit in `checks/rendering_regression.json`. Final full-suite proof is
`checks/pre_execution_tests/`; no source changed after that successful suite.
Do not repeat finished checks without a new source change, failure or recheck need.

| Completed job | Elapsed | Peak worker RSS | Minimum available |
| --- | ---: | ---: | ---: |
| Eight focused preparation tests | 1.78 s | 191.53 MiB | 5.25 GiB |
| Full 299-test suite | 3.56 s | 357.98 MiB | 5.16 GiB |
| Complete 12-date preparation | 16.81 s | 399.52 MiB | 5.20 GiB |
| Independent packet verification | 1.27 s | 144.40 MiB | 5.14 GiB |
| Independent final-review verification | 1.27 s | 149.16 MiB | 5.11 GiB |

All jobs were sequential nice +10 CPU workers, native threads 1, no GPU, using the
UNCHANGED `validate_sample.run_guarded`, 512 MiB RSS and 1.5 GiB available-memory
reserve. Preparation/verification runtime bound is 300 s, tests 120 s. No jobs
remain active. Never rerun this preparation: its four previously missing caches
are now published and hash-bound by the completed packet.

## Durable files and safe continuation

Authoritative tracked evidence: `Docs/Reviews/ValidationManualV4/`:

- `protocol.json`: fixed predeclared source/cache/roster/review/resource contract.
- `preparation/`: safe `observations.npz`, metadata and five inspection PNGs.
- `packet_verification/`: completed independent pre-inspection proof.
- `reviews.json`, `visual_review.md`: complete hash-bound states and rejection.
- `review_verification/`: independent final rejection/interval/coverage proof.
- `checks/`: focused/full-suite receipts and pre-image rendering regression history.

`checks/final_integrity.json` confirms all 14 source inputs, 30 immutable old
implementations, two frozen new implementations, 696 timeline source-file
fingerprints and 12 cache hashes still match. Archived outputs match local
completed copies, the final review hash matches independent verification, and
no processing worker or validation dataset/label-lock output remains.

New modules: `Solutions/prepare_validation_manual.py`,
`Solutions/verify_validation_manual.py`; corresponding new tests in `Tests/`.
Guide: `Docs/Guides/ValidationManual.md`. Ignored local working copies:
`Datasets/ValidationPilot/v4/`. GitHub holds code and bounded evidence; full source
imagery, all caches and complete audit inputs remain on this Mac and are needed
for source-aware reproduction. Do not overwrite completed outputs.

Previous provenance/constraints: `Docs/Handoffs/2026-11-17-validation-context.md`,
`2026-11-16-validation-support-pool.md`, `2026-11-16-validation-pilot.md`,
`2026-11-16-pixel-pilot.md`, `2026-11-15-curated-pilot.md` and their linked handoffs.
QA policy remains
`fe2472ec030f466bebbb9dc2cb033ed6175de489bcf022f650081dd8e9347560`.
All old model/QA/objective/curation/validation implementations and both new V4
implementations are frozen by source hashes. Do not edit them during reproduction.
Preserve labels 0 reviewed forest, 1 reviewed clearing, 255 unknown, and full train/test
footprint isolation. Existing Torch runtime 2.6.0.dev20241112/distribution 2.5.1
mismatch is recorded in test receipts; do not reinstall the environment here.

## Next proposed bounded task (not executed)

Audit actual anchor-specific raster support and observation metadata for the
**606 frozen validation pairs lacking published QA**. The completed V2 audit
retained these as unknown, not failed/clear. This next inventory should be
metadata/geometry-only, with no satellite pixel reads, imagery viewing or models.
Retain all 606 records and all failure/unknown reasons; do not call support an
approval. Preserve minimum 4 eroded positive pixels, original targets, complete
feature-footprint isolation and later-observation opportunities as separate gates.

Exclude EVERY previously inspected validation group from any fresh proposal,
including this NOW EXPOSED/REJECTED `group-45f0c39e787ca8622918`, in addition to
all V1 history. A different event/pair in this group is not fresh. Keep the completed
V1/V2/V3/V4 results unchanged and test groups untouched. Do not add replacement
cases to this exhausted V4 protocol or search imagery until success.

Only after that inventory should a separate bounded quality-screening roster be
predeclared, if support exists. Any eventual validation requires explicit class/
timeline review, approved labels, independently verified export and committed
label lock before evaluating frozen final 1664 weights at 0.5. No retraining,
threshold tuning or best-checkpoint selection. First-visible timing remains
future work; annotation image dates and local observed intervals are not exact
felling dates.

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Git root is its parent. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Check branch/status and active Python commands before new work. Use one guarded
low-priority worker and the capacity limits above; never duplicate active jobs.
