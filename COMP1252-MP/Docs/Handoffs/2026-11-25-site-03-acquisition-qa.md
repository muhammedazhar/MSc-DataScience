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
| 2018-05-30 | 58 | 526,754,551 | 22.88 s | 64.30 MiB |
| 2018-06-04 | 58 | 524,534,456 | 22.39 s | 64.88 MiB |

All three transactions are complete. Each published scene passed the frozen
local QA metadata checks and has 58 verified file receipts tied to its V29
manifest. The full source scenes and transaction journals remain local.

## Frozen lineage

The [protocol](../Reviews/ValidationSite03ScreenV30/protocol.json) and
[lineage](../Reviews/ValidationSite03ScreenV30/lineage/lineage.json) bind the
five-day comparison to the original site grid and frozen split. The cumulative
mask changed from 57 pixels before to 234 after, with 177 pixels in the new-loss
reference and 115 in the eroded target. The target and eroded target contexts
have 335 and 231 pixels. The 256-pixel proposed control has a 400-pixel context.
The source checks verified 174 new receipts and 2,494 prior file fingerprints.
Lineage preparation peaked at 149.48 MiB and read zero image band windows.

## Verified automatic screening

The [screen report](../Reviews/ValidationSite03ScreenV30/screen/screen.json)
retains `derived-site03-891e388f6c6aaee1a9ea` for separate visual review. The
automatic screen reported zero unusable pixels in the full 4,096-pixel crop,
the 177-pixel new-loss target, and the 335- and 231-pixel target contexts on
both dates. Cached RGB and NDVI values were finite in both target contexts.

| Control observation | Unusable in control (256 pixels) | Unusable in full context (400 pixels) |
| --- | ---: | ---: |
| 2018-05-10 | 0 | 0 |
| 2018-05-30 | 0 | 0 |
| 2018-06-04 | 0 | 0 |

Cached RGB and NDVI values were finite throughout the control and its context.
The control's class remains unknown. The automated checks covered cloud,
cloud buffer, shadow risk, haze risk and invalid pixels under the frozen rules.

The [offline replay](../Reviews/ValidationSite03ScreenV30/verification/verification.json)
matched the exact lineage, target geometry, QA masks, numeric results, context
checks and review shortlist using the three archived cache snapshots. Live
screening used 33 band windows and three cloud classifier calls. Replay read
zero original image windows and made zero cloud classifier calls. Screening
took 9.42 s with 340.00 MiB peak sampled RSS and 4.58 GiB minimum available
memory. Replay took 1.28 s with 148.98 MiB peak worker RSS. The
[integrity check](../Reviews/ValidationSite03ScreenV30/final_integrity.json)
binds the evidence archived in Git. All 394 existing tests passed under
the CPU guard.

## Remaining work

Check all 27 feature channels in the fixed target and control contexts, then
review the two target endpoints and the three control observations without
seeing forest-model predictions. Investigate access to the independent
reference leads from V28 before drawing a class or timing conclusion. The
five-day span is a candidate comparison window. Visual class, validation
eligibility and the first-visible interval remain unresolved. No label or
forest-model output was added in V30.
