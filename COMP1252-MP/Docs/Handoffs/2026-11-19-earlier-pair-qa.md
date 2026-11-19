# Earlier pair QA continuation

## Current state: V15 complete; all four fixed pairs excluded

Continue on `codex/curated-deforestation-pilot`; commit each verified logical
change with this handoff updated and push to the SAME branch. Read
`2026-11-18-earlier-source-acquisition.md` and linked V13/V12/V11/older handoffs
for source provenance, frozen QA/model/split constraints and prior failures.
`CONTINUE.md` supports Codex, Gemini Antigravity or another agent without chat history.

The approved next step is COMPLETE: four distinct earlier-date pairs were
prepared, screened and replay-verified. **All four are excluded by automatic QA:
cloud/buffer flags cover every target and target/interior context pixel in both
21 July and 16 July 2018 pre-images.** Both pre-crops have 4096/4096 flagged pixels.
No mask/margin was shrunk, threshold changed, date substituted or failed result
erased. Zero processing errors. No manual images, forest predictions, class
labels, first-visible interval, full 27-channel proof or unseen accuracy exist.
380 tests pass. No worker is active. No further pixel/date work occurred in V15.

Checkpoints pushed: `48af981` predeclared four-case protocol; `c63618c` tested/
frozen implementation; `882fe6a` lineages/targets committed BEFORE satellite
pixels; `0920e45` all four QA exclusions and actual masks. The completion
checkpoint contains safe cache snapshots, replay/integrity and this final handoff.
All earlier source data, V10/V14 transactions and V11 exclusions remain intact.

## Fixed roster, explicit lineage and cumulative targets

Protocol: `Docs/Reviews/ValidationEarlierScreenV15/protocol.json`.
Implementation: `Solutions/screen_earlier_validation.py`.
Exactly pre 21 then pre 16 July x events A/B, fixed 31 July post. These are ONE
geographic group `group-96c27871459d03c08bcc`, not four independent locations.
Source scenes are exactly the two published V14 earlier L1C/N0206/R064/36UXB
products plus the unchanged 31 July V10 source. No acquisition/download in V15.
Original 36UXA parents and V11's cloudy 26 July derived pairs remain unchanged.

| Pre-date | Event suffix | Original parent suffix | New derived ID | Interval |
| --- | --- | --- | --- | ---: |
| 2018-07-21 | `71d8da17cbbfd9babdee` | `4644e1fb9fe295425239` | `derived-earlier-7da954e9ab94479ec0dd` | 10 days |
| 2018-07-21 | `ce1c1df8d02be33bb3c8` | `7250e1f130f05c216746` | `derived-earlier-bc76a605bd7930ec58f2` | 10 days |
| 2018-07-16 | `71d8da17cbbfd9babdee` | `4644e1fb9fe295425239` | `derived-earlier-fca71802356890bf7db8` | 15 days |
| 2018-07-16 | `ce1c1df8d02be33bb3c8` | `7250e1f130f05c216746` | `derived-earlier-badde8347026296b30b7` | 15 days |

Every new pair stores the exact original parent hash, old dates/status/blockers,
new pre-date/interval/source IDs/paths and new identity. Original grid 64x64/10 m,
group/split/anchor and annotation temporal source IDs remain unchanged. Imagery
tile 36UXB is separate from annotation source 36UXA; no provenance relabelling.
Complete crop-intersecting context/post/pre event sets were recomputed. Expanded
intervals introduce no baseline-only/conflicting/cross-group labels. All original
train/test conservative footprints and all 13 actual training grids are avoided.

Cumulative post AND NOT cumulative pre was recalculated using frozen helpers,
and checked against fresh cumulative burns/boolean difference. No interval-only
union or V11 hardcoded-date derivation/reference mask was reused. Relevant
annotation dates are 21 June, 6 July, 31 July; future 5 August stays context only.
Thus 16/21 July references legitimately equal the parent 26 July reference, with
0 changed pixels, while date-aware provenance/IDs remain distinct. Annotation
image dates are not felling timestamps; zero reference pixels are not negatives.

| Event | Cumulative pre / post / new-loss union | Anchor target / eroded interior |
| --- | ---: | ---: |
| A `71d8...` | 122 / 269 / 147 | 55 / 8 |
| B `ce1c...` | 78 / 222 / 144 | 92 / 25 |

These counts hold for both earlier dates. One-pixel 3x3 anchor erosion and full
radius 2 receptive-field context were retained. Event B still has one temporally
clipped anchor pixel, as before. Target hashes and parent comparison are saved.
The boolean check shares the rasterizer; it is not independent annotation truth.

## Automatic QA and replay outcomes

All four statuses: `excluded_by_automatic_quality_screen`. Both dates have
`cloud_or_buffer_on_target` and `crop_quality_flags_exceed_10_percent`; 21 July
also has `suspected_haze_on_target`. Model cloud flags/review heuristics are
input-quality decisions, not independently verified physical cloud/class truth.

| Pre-date | Event | Target / interior | Full target context flags pre / post | Interior context flags pre / post |
| --- | --- | ---: | ---: | ---: |
| 21 July | A | 55 / 8 | 176/176 /4/176 | 56/56 /0/56 |
| 21 July | B | 92 / 25 | 274/274 /0/274 | 104/104 /0/104 |
| 16 July | A | 55 / 8 | 176/176 /4/176 | 56/56 /0/56 |
| 16 July | B | 92 / 25 | 274/274 /0/274 | 104/104 /0/104 |

Both pre-crops flag 4096/4096; post crop flags 64/4096 for A, 88/4096 for B. Both post
anchor target ROIs flag 0. The four post-context flags for A are shadow risk. All
pre target/interior context pixels are cloud/buffer flagged. Cached RGB/NDVI
are finite on those contexts, but this is NOT complete 27-channel eligibility.
No stable control/class review, later persistence or labels have been approved.

Eight observation requests across six unique source/grid combinations: 4 hits,
4 new caches / 4 frozen cloud-model calls / 44 fused-band requests. Two original
V11 post caches were reused byte-identically; their hashes were checked before
and during reuse. New/existing caches are bounded/header-validated before array
allocation. Previous automatic exposure is retained explicitly; this group was
already automatically screened in V11. Manual exposure history is also preserved.

Replay PASSED all four exact derived pairs and freshly recomputed targets,
eight bound cache requests, four six-plane QA TIFFs, every numeric QA outcome
and anchor/interior/context count. It decoded no source JP2 windows and ran no
cloud/forest model. It reuses frozen geometry/QA helpers and cached arrays; it
is not independent cloud prediction, imagery review or class truth.

## Validation, resources and durable evidence

380 tests pass (369 prior + 11 new). They cover four deterministic date/event IDs,
parent/date/grid/blocker/collision refusal, fresh provenance and baseline-only
interval checks, overlapping cumulative-mask subtraction, safe NPZ scalar/header
allocation bounds and source/grid binding, lineage tampering and full-roster
error retention. If any pair processing fails, all four attempts/outcomes are
retained with explicit `incomplete_processing_errors` metadata; such runs cannot
replay/pass. Failed observation keys are not automatically retried. No such
processing error occurred in this execution. No frozen implementation changed.

| Completed guarded job | Elapsed | RSS high-water / sampled peak | Minimum available |
| --- | ---: | ---: | ---: |
| 380-test suite | 4.84 s | 333.52 / 334.0 MiB | 3.44 GiB |
| Lineage/geometry preparation | 2.30 s | 167.03 / 159.2 MiB | 3.23 GiB |
| Four-pair QA | 11.20 s | 360.02 / 355.1 MiB | 3.39 GiB |
| Cached replay | 2.29 s | 179.27 / 171.2 MiB | 4.05 GiB |

All jobs: one nice +10 CPU worker, native threads 1, no GPU, unchanged
`validate_sample.run_guarded`, 512 MiB RSS, 1.5 GiB available-system-memory reserve,
300 s max (tests 120 s). No workers overlapped. Preparation decodes no satellite
pixels; only screening runs the frozen cloud classifier. Torch/forest models
were never imported in these phases; full tests retain their existing CPU setup.
Initial free disk 25.54 GiB; outputs/copies are small, no new raw source imagery.
No environment reinstall or QA/model/ROI threshold changes.

Tracked evidence under `Docs/Reviews/ValidationEarlierScreenV15/`:
- `protocol.json`, `lineage/` with all four pairs/recomputed reference counts/hashes.
- `screen/` every outcome/context/receipt plus all four six-plane TIFFs. A narrow
  Gitignore exception tracks these exact masks; no preview was generated/viewed.
- `verification/` completed replay receipt/report.
- `cache_evidence/` six byte-identical NPZ snapshots / 278,139 bytes total, with
  occurrence/source/hash index; includes two unchanged V11 post observations.
- `checks/pre_execution_tests/` and `checks/final_integrity.json`.
Guide: `Docs/Guides/EarlierPairQA.md`.

Final integrity binds all 62 frozen evidence/code hashes, 174 used source hashes
checked in each guarded phase, all 174 unchanged source stats, prior 116 source
stats and both V11 exclusions; V10/V14 complete transactions/original backups;
all four masks, six safe cache snapshots and 380-test proof; zero active workers,
labels, forest models or new visual reviews. Final audit uses artifact hashes
and unchanged source stats rather than decoding/rehashing source images again.
Full27-channel eligibility remains unchecked and `validation_eligible=False`.

Completed ignored local outputs: `Datasets/ValidationPilot/v15/{lineage,screen,verification}/`.
Never overwrite/repeat these because a chat/agent changed. GitHub contains code
and bounded evidence, while full sources/cache collection and source transaction
journals remain on this Mac. Preserve all earlier evidence, source files and
0 forest/1 clearing/255 unknown semantics; frozen model 1664 at 0.5 remains unchanged.

## Next proposed bounded task (not executed)

The fixed 16/21 July roster is exhausted; all four pairs failed automatic QA.
The known-cloudy 26 July source remains excluded too. A reasonable next task is
METADATA-ONLY compatibility preflight of the remaining finite V12 nominal dates:
**11 July, 6 July, 1 July 2018** (20/25/30-day intervals before fixed 31 July post).
Predeclare that entire small roster, source/disk/request bounds and stop rule
BEFORE any new acquisition/pixel work. Retain all earlier failures. No additional
source/date acquisition or pixel screening was executed as part of V15.
Do not screen extra dates until success, recycle rejected visual groups, relax QA,
shrink margins, infer labels from cloud metadata/NDVI or report empty-set accuracy.

Later class approval still needs unambiguous before forest/after clearing,
controls, later persistence, finite 27-channel proof and a label lock before
frozen-final evaluation. Two events in this one group cannot establish multiple
independent-location accuracy. No first-visible interval or validation score yet.

SSH GitHub backup works via `ssh.github.com:443` with the same account:
`ssh://git@ssh.github.com:443/muhammedazhar/MSc-DataScience.git`, BatchMode and
StrictHostKeyChecking enabled. Push the SAME branch, then fetch tracking and
compare actual HEADs. No force push/global Git config/remote change needed.
Python `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Project `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`;
Git root is its parent. Check branch/status, existing outputs and active jobs.
