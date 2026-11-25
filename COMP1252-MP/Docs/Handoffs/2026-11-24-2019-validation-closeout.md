# 2019 Ukraine validation closeout

Continue on `codex/curated-deforestation-pilot`. Commit verified milestones
and push to GitHub on the same branch. Read the V25 and V26 handoffs for the
original crop, source, catalogue and reference evidence.

V27 fixes all four V26 metadata-ready scenes before any imagery transfer.
The two 2019 footprints, their group assignments and their original post
observations remain the comparison anchors.

| Site | Earlier observations | Fixed post observation |
| --- | --- | --- |
| March | 3 and 8 March 2019 | 2 April 2019 |
| May | 7 and 12 May 2019 | 6 June 2019 |

The [acquisition plan](../Reviews/Validation2019ScreenV27/acquisition_plan.json)
pins the exact V26 scene rows, native file manifest hashes and frozen code.
The per-date `download_inputs` directories are direct imports of the verified
V26 rows for `Solutions/download.py download --preflight`. The listed payload
is 2,208,628,047 bytes across 232 files. Available disk before download was
66.89 GiB. Use one nice +10 CPU worker and one native thread at a time, a
512 MiB worker RSS cap, a 1.5 GiB available memory reserve, a 2 GiB free-disk
reserve and the existing 300 second guard. Each run writes a distinct
`publication/YYYY-MM-DD` report.

## Verified publication

| Date | Files | Bytes | Elapsed | Peak worker RSS |
| --- | ---: | ---: | ---: | ---: |
| 2019-03-03 | 58 | 679,536,761 | 134.50 s | 59.41 MiB |
| 2019-03-08 | 58 | 610,837,722 | 108.34 s | 64.20 MiB |
| 2019-05-07 | 58 | 347,466,958 | 78.70 s | 66.28 MiB |
| 2019-05-12 | 58 | 570,786,606 | 114.15 s | 63.78 MiB |

All four downloader transactions are complete. Each publication receipt
binds 58 verified local files and actual frozen QA metadata to the original
V26 manifest. The large raw sources and transaction journals remain local.

## Pre-screen lineage

The [screening protocol](../Reviews/Validation2019ScreenV27/protocol.json)
and [lineage](../Reviews/Validation2019ScreenV27/lineage/lineage.json) bind four
distinct pairs to the frozen split, fixed post observations, unchanged crops
and the verified publication receipts. The cumulative new-loss reference
recomputed to 180 pixels at the March site with 73 eroded anchor pixels, and
46 at the May site with 13 eroded anchor pixels. All footprints passed the
split-isolation gate. The guard checked 2,262 prior source file stats and
preserved the three known cloudy 2019 comparisons and four V24 exclusions.
Lineage preparation peaked at 153.16 MiB and read zero image band windows.

## Quality screening and replay

| Earlier date | Fixed post date | Earlier crop flagged | Earlier target context flagged | Earlier interior context flagged | Disposition |
| --- | --- | ---: | ---: | ---: | --- |
| 3 March 2019 | 2 April 2019 | 1,876 / 4,096 | 146 / 392 | 103 / 244 | Suspected haze. Fails the frozen review-candidate gate |
| 8 March 2019 | 2 April 2019 | 4,096 / 4,096 | 392 / 392 | 244 / 244 | Excluded for cloud or its buffer |
| 7 May 2019 | 6 June 2019 | 4,096 / 4,096 | 137 / 137 | 65 / 65 | Excluded for cloud or its buffer |
| 12 May 2019 | 6 June 2019 | 4,096 / 4,096 | 137 / 137 | 65 / 65 | Excluded for cloud or its buffer |

Both fixed post crops and their full contexts have zero QA flags. The base
screen marks 3 March `needs_visual_review`. Its flagged target and interior
contexts leave the frozen review roster empty. The screen used four cloud
classifier calls and 44 source band windows. No processing errors occurred.
Offline replay matched the four six-plane masks, numeric outcomes, context
counts and all eight cache bindings from six distinct observations. Replay
used zero source band reads and zero cloud classifier calls.
[Screening results](../Reviews/Validation2019ScreenV27/screen/screen.json) ·
[Replay](../Reviews/Validation2019ScreenV27/verification/verification.json).

Screening and replay peaked at 320.78 and 163.11 MiB respectively. The
lowest sampled available memory during screening was 2.65 GiB. All 394
tests passed under the guard. The independent training verifier reproduced
the original 13-case fit and the final checkpoint at its frozen threshold.
[Tests](../Reviews/Validation2019ScreenV27/checks/test_run/metadata.json) ·
[Training verification](../Reviews/Validation2019ScreenV27/checks/training_verification/report.md).

## Project results

The [project results report](../ProjectResults.md) records the reproducible
pipeline, verified training fit, held-out screening outcomes and the Area B
case study. The report supports the provisional lower-opening expansion
interval (6 July, 31 July] 2018. Both 2019 first-visible intervals and held-out
accuracy remain unresolved. The study produced zero approved validation
labels. The [integrity receipt](../Reviews/Validation2019ScreenV27/checks/final_integrity.json)
binds all closeout checks. Acquisition, screening, replay and reporting are
complete for this roster.
