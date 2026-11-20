# Remaining July source acquisition

V17 is complete. All three L1C scenes passed file integrity and the frozen
local scene metadata checks. Their directories are published under
`Datasets/Sentinel-2/`. The 174 files total 1,664,246,549 bytes. Pixel clarity
for these dates remains untested. Continue on `codex/curated-deforestation-pilot`
and push verified checkpoints to the same branch.

| Date | Local scene directory | Files | Guarded download | Peak RSS |
| --- | --- | ---: | ---: | ---: |
| 11 July 2018 | `S2A_MSIL1C_20180711T083601_N0206_R064_T36UXB_20180711T122613` | 58 | 52.74 s | 63.03 MiB |
| 6 July 2018 | `S2B_MSIL1C_20180706T083559_N0206_R064_T36UXB_20180706T123246` | 58 | 47.42 s | 59.58 MiB |
| 1 July 2018 | `S2A_MSIL1C_20180701T083601_N0206_R064_T36UXB_20180701T104347` | 58 | 47.75 s | 66.94 MiB |

The acquisition runner uses the frozen V14 streaming and transaction
functions with a V17 journal. It followed the V16 L1C asset pins and checked
size and provider MD5 on transfer. Publication rehashed every file against
the saved SHA-256 and reran the frozen metadata checks before and after
moving the scene directories. All six local product and tile XML files matched
the V16 evidence byte for byte. The final check confirms the prior 232 file
size and mtime records, both earlier transactions, and all six cloud
exclusions. The V17 journal records 174 verified transfers, three prepared
scenes, and one completed publication. The staging directory is empty.

Acquisition and publication used one nice +10 worker at a time, one native
thread, a 512 MiB RSS cap, a 1.5 GiB available memory reserve, and a
300-second limit per run. Publication took 7.12 seconds, peaked at 51.80 MiB,
and finished with 23,820,689,408 bytes free on the source filesystem.
All 388 guarded tests passed. No workers are active.

Checkpoint `4aecdad` pins the tested wrapper, exact three-scene roster, and
resource limits. Checkpoints `84a94f8`, `094b939`, and `8431c82` preserve the
verified staging results in acquisition order. The completion checkpoint
records the published directories, asset receipts and final integrity check.
See `Docs/Reviews/ValidationRemainingAcquisitionV17/` for the protocol,
guard reports, journal snapshots and final file stats. The live transaction
and durable completion receipt are under
`Datasets/ValidationPilot/v17/source_transaction/`. Raw imagery is local.
The repository contains verified receipts and checksums. Journal schema 14
is the internal format of the reused engine. The outer V17 protocol and its
hash bind the journal to this run.

Future pixel QA should derive six explicit pairs from the three pre-images,
the fixed 31 July post-image, and the two original event grids. Recalculate
cumulative temporal references for each new date, including the 30-day
boundary and the earlier baseline on 1 July. See the V15 handoff for lineage,
split isolation and frozen cloud screening. The six pairs share one
geographic group and any first-visible interval still requires clear pixels
and class evidence.
