# Antigravity review, source completion and derived QA continuation

## Current state: approved review/repair/QA complete; both pairs excluded

Continue on **`codex/curated-deforestation-pilot`**. Commit each verified change
with this handoff updated and push to the SAME GitHub branch. `CONTINUE.md`
supports Codex, Gemini Antigravity or another agent without earlier chat history.
The user asked to review Antigravity's work and continue; preserve Mac capacity
and keep further work bounded.

The review and corrective V10/V11 work are COMPLETE. Missing source QA inputs
were repaired and both scenes pass actual frozen prerequisites. Two explicit
derived pairs were screened and replay-verified; **both are excluded because
26 July 2018 pre-image cloud/buffer covers all target/context pixels**. No visual
images, forest predictions, class labels, first-visible interval or accuracy
were produced. All 341 tests pass. No jobs remain active.

Checkpoints pushed: `ddb55da` review/protocol, `2eda56d` tested repair/screen
code, `3adcabb` completed repair and predeclared QA. The following completion
commit contains screen/replay evidence, complete review and this final handoff.
Antigravity checkpoints `2cb99f4` (V9), `00a4749` (V8), `9f4145d` (V7) remain
preserved as historical evidence with explicit qualifications, not rewritten.

## Review scope and findings

Read `Docs/Reviews/AntigravityReview/2026-11-18-review.md`. Main P1: V8/V9 acquired
seven feature JP2 s +product/tile/cloud metadata per scene, omitting four cloud
bands B01/B05/B09/B10 and 44 per-band NODATA/SATURA/DEFECT/TECQUA GMLs. Actual
`qa_scene_metadata` returnedFalse with 48 issues per scene, although the weaker
seven-band inventory returnedTrue. Frozen QA would refuse those inputs.

P2 findings: existing nonempty files were trusted without source checks; final
source paths were exposed before full ingestion succeeded; V7 summaries mixed
N0206/N0500 while hardcoding AWS zero-offset, and V8 size/disk figures contained
estimates/constants. Actual selected V9 product/tile XML is genuine L1C/N0206,
quantification 10000/zero offsets and correct CRS/coverage. No production metadata
or mask synthesis was found. The existing offset formula is correct.

All frozen V7/V8/V9 source/protocol/result files remain intact. Their guide/
handoff banners link the review. Do NOT rerun archived ingestion for correction;
use the new strict path. Earlier V1–V6 provenance, source hashes, split/model/QA
constraints are linked through `2026-11-17-validation-ingestion.md`,
`2026-11-17-validation-acquisition.md`, `2026-11-17-validation-catalogue.md`,
`2026-11-17-validation-coverage.md`, `2026-11-17-validation-unscreened.md` and
older handoffs. QA policy remains
`fe2472ec030f466bebbb9dc2cb033ed6175de489bcf022f650081dd8e9347560`.

## Completed source repair (V10)

Protocol: `Docs/Reviews/ValidationSourceRepairV10/protocol.json`. Exactly 96 new
assets/74,541,437 bytes from two full provider listings:4 cloud bands +44 quality
GMLs per scene. All 20 old assets match V9 SHA plus pinned provider size/MD5.
Each supplement uses If-Match, Content-Length/byte cap, source MD5 and recorded
SHA256. No invented empty masks. Actual free disk measured before/after and
>=2 GiB reserve enforced in the downloader.

Complete trees are staged OUTSIDE Sentinel 2 discovery using hardlinks for
original data, then validated with actual `qa_scene_metadata`. Publication uses
an exclusive lock and durable intent/inode journal; interruption recovery moves
only owned trees and restores original backups without deleting user trees.
This is recoverable two-directory publication, not an atomic pair swap. Original
backups are retained. All frozen metadata identity/grid/scale fields remain equal.

Persistent transaction:
`Datasets/ValidationPilot/v10/source_transaction/journal.json`, state **complete**.
Its generation directory contains `backups/` and a completion receipt. Do not
remove it; completed repair must not be repeated. Original data are hardlinked,
so backups do not duplicate the original ~934 MiB payload. Added payload 71.1 MiB.
Completion source receipt `path` values describe download staging; reconstruct
current files from `scene_id` + `target_relpath` under Sentinel 2, as V11 does.

Scenes (both now strict QA-ready, 52 local scenes total):
`S2B_MSIL1C_20180726T084009_N0206_R064_T36UXB_20180726T104227` and
`S2A_MSIL1C_20180731T083601_N0206_R064_T36UXB_20180731T110233`.
Each has 58 required files:11 JP2, 44 source GMLs,cloud GML,product XML,tile XML.
All 116 actual files are hashed and source stats bound before/after QA.

## Derived source lineage and QA result (V11)

Protocol: `Docs/Reviews/ValidationSourceScreenV11/protocol.json`, committed
BEFORE QA. Source pairs are explicitly derived; original parents stay blocked.
No frozen candidate is impersonated, no group/split/grid or temporal catalogue
label IDs change. Only the original two coverage blockers may be repaired in
these derived records. Both full 64 x 64 feature grids retain complete separation
from every train/test conservative footprint and all 13 actual training grids.

| Event | Original parent | Derived ID | Target / eroded interior | Full target context flags pre / post |
| --- | --- | --- | --- | --- |
| `event-71d8da17cbbfd9babdee` | `pair-4644e1fb9fe295425239` | `derived-source-fb83814693418ede65cb` | 55 / 8 | 176/176 / 4/176 |
| `event-ce1c1df8d02be33bb3c8` | `pair-7250e1f130f05c216746` | `derived-source-14ba5d005291904b2561` | 92 / 25 | 274/274 / 0/274 |

Both statuses: `excluded_by_automatic_quality_screen`. Both pre-crops have
4096/4096 unusable pixels and all anchor/eroded-interior/context pixels cloud/
buffer-flagged. Both post target ROIs have 0 flags. Post interior contexts 56/104
pixels have 0 flags; first event's complete target context retains 4 shadow-risk
pixels. No ROI/margin was shrunk. Second event has 1 temporally clipped anchor
pixel, unchanged from V5 geometry. The assessor's union reference metrics are
separate from these anchor-specific context counts.

Four observations, 4 new caches/4 frozen cloud-classifier calls, 44 fused-band window
requests. No forest model/Torch, previews, new manual class review, training or
validation inference. Cached RGB/NDVI are finite on contexts. **All 27-channel eligibility
was NOT checked**, and `validation_eligible=False` throughout. Later persistence,
control labels, first-visible interval and accuracy remain unknown. Both events
share ONE group `group-96c27871459d03c08bcc`, not independent locations. Automatic
QA exposure is recorded; no imagery/class labels were inspected in this phase.

Replay verification PASSED:4 safe caches validated before allocation, 2 six-plane
QA TIFFs, exact derived lineage, all 116 source hashes, source/cache stats and
reconstructed anchor contexts. The verifier reuses lineage/QA/geometry helpers;
it is cached-output replay, not independent cloud prediction or class truth.
It decodes no source JP2 and executes no QA/forest model.

## Validation, resources and durable state

**341 tests pass** (324 previous + 8 source completion + 9 derived screen), zero
failures/errors/skips. New cases cover missing QA gates, corrupt nonempty files,
source byte/MD5/ETag failures, staging/rollback/crash recovery, user-tree preservation,
derived-lineage tampering, extra provenance blockers, swallowed processing errors
and NPZ allocation/header bounds. No code changed after the passing suite.

| Completed job | Elapsed | Peak RSS | Minimum available |
| --- | ---: | ---: | ---: |
| Full 341-test suite | 7.94 s | 298.27 MiB | 2.70 GiB |
| Source completion | 43.19 s | 58.36 MiB | 2.79 GiB |
| QA-only screen | 18.40 s | 265.44 MiB | 2.69 GiB |
| Cache/TIF replay | 3.31 s | 144.88 MiB | 2.77 GiB |

All jobs used one nice +10 CPU worker, native threads 1, no GPU, unchanged guard:
512 MiB RSS, 1.5 GiB available-memory reserve, 300 s max (tests 120 s). Actual worker
free disk 28.56 GiB before/28.53 GiB after; an intermediate handoff's 24.33/24.26 GiB
preflight-derived figures were corrected to these measured values. Disk space
is dynamic. Existing Torch runtime/distribution discrepancy remains recorded;
no environment reinstall or package update occurred.

Evidence:
- `Docs/Reviews/AntigravityReview/2026-11-18-review.md`.
- `Docs/Reviews/ValidationSourceRepairV10/`: protocol, provider listing/HEAD
  snapshots, completion/report/metadata and focused/full test receipts.
- `Docs/Reviews/ValidationSourceScreenV11/`: protocol, `screen/` result/derived
  pairs/twoQA TIFFs, `verification/`, 4 byte-identical bounded cache snapshots.
  A narrow Gitignore exception tracks the new masks; verify all artifact hashes
  before assuming a file is available on another machine.
- Guide: `Docs/Guides/ValidationSourceReview.md`.

V10 and V11 `checks/final_integrity.json` bind the completed protocols/results,
all source/backup receipts and final 116 file stats, preserved original parent
records, four cache snapshots and both QA TIFFs. All 341 test results and empty
active-worker inventory are confirmed. Full raw source hashes were already
verified by the guarded repair, screen and replay; the final check uses unchanged
source stats and bounded artifact hashes rather than decoding source rasters.

Ignored local results: `Datasets/ValidationPilot/v10/completion/`, persistent
transaction, `Datasets/ValidationPilot/v11/{screen,verification}/` and full source/
cache collection. Old data remain local; GitHub stores code and bounded evidence.
No worker remains active; never overwrite completed results or rerun source QA
just because an agent/chat changed.

## Next proposed bounded task (not executed)

A METADATA-ONLY catalogue inventory of earlier 36UXB/R064 pre-image dates within
30 days before 31 July 2018, retaining the known-cloudy 26 July outcome. The current
minimal pair is exhausted, and changing only its post date cannot repair the
cloudy pre-image. Predeclare a small additional source roster before acquisition/
pixel work; do not search pixels until success, erase failures, change QA thresholds,
shrink margins, relabel sources or recycle previously rejected visual groups.
Later post/persistence observations and valid controls are still required before
manual class review/label approval; complete 27-channel proof is a separate step.
No validation dataset, label lock or frozen-final 1664 evaluation exists yet.

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Git root is its parent. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Preserve 0 forest/1 clearing/255 unknown and all train/test/exposed-group barriers.
Check branch/status and active commands before the next bounded task.
