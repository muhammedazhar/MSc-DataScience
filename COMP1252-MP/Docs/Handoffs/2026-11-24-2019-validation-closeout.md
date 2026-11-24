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

Pending work includes publication dates absent from the table, comparison lineage,
cloud screening, replay and the project results closeout.
