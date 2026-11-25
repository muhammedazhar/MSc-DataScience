# Site-03 acquisition and crop quality screening

Continue on `codex/curated-deforestation-pilot` and push each verified
checkpoint on the same branch. The [acquisition plan](../Reviews/ValidationSite03ScreenV30/acquisition_plan.json)
pins the three completed V29 manifests, the frozen site-03 crop and its
proposed 16 × 16 control. Use the existing `Solutions/download.py` workflow
with the original per-date V29 preflight directories.

The 30 May and 4 June 2018 scenes are the frozen five-day comparison. The
10 May observation is for QA on the same proposed control. Audit the control's
full two-pixel context on all three dates. The proposed control remains
unlabelled until forest class can be reviewed. The 1,200 m source margin,
64 × 64 crop, exact coordinates and geographic split stay fixed.

The roster totals 174 files and 1,562,638,464 bytes. Before transfer the volume
had 60.31 GiB free. Keep one nice +10 CPU worker and one native thread at a
time, a 512 MiB worker RSS cap, a 1.5 GiB available-memory reserve, a 2 GiB
disk reserve and a 300 second runtime cap. Stage each scene through its
recoverable transaction and archive each publication receipt.

## Verified publication

| Date | Verified files | Bytes | Elapsed | Peak worker RSS |
| --- | ---: | ---: | ---: | ---: |
| 2018-05-10 | 58 | 511,349,457 | 22.38 s | 63.53 MiB |

Pending work includes publication dates absent from the table, lineage
verification, crop and control QA, and replay.
