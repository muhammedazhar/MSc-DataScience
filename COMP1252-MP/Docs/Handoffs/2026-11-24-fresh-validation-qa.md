# Fresh validation acquisition and QA

V24 is complete. Both 21 and 16 July pre-images flag every crop pixel at
both fresh sites. All four fixed pairs are excluded by automatic QA and the
review roster is empty. Continue on
`codex/curated-deforestation-pilot` and push verified checkpoints to that branch.
The two exact 36UXA/R064/N0206 scenes are 21 and 16 July 2018. The two
original validation groups, post dates, and 64 × 64 crop grids stay fixed.
Earlier rejected pair outcomes and the seven exposed groups are retained.

Check provider metadata and all 58 listed files per scene before acquisition.
Stage each verified source separately and use the existing transaction and
guard. Derive four comparison identities, recompute cumulative reference
geometry before decoding new imagery, and retain every QA result. Use one
nice +10 CPU worker, one native thread, 512 MiB RSS cap, 1.5 GiB available
memory reserve, and at least 2 GiB disk reserve during download.

Checkpoint `62b7b76` pins the protocol and three workers. Live preflight and
offline replay passed for both real 36UXA XML pairs, complete padded footprints,
and 58 listed asset pins per scene. The listed total is 1,143,754,335 bytes.
Working memory peaked at 102.83 MiB in preflight and 90.39 MiB in replay.
Checkpoint `6bc801e` saves preflight and exact replay. The 21 July source
passed staging with 58 files and 596,775,942 bytes verified. The download
took 79.64 seconds and peaked at 101.56 MiB RSS, recorded at `5b2e405`.
The 16 July source also passed staging with 58 files and 546,978,393 bytes
verified. It took 76.94 seconds and peaked at 100.03 MiB RSS, recorded at
`fc22a7f`. Both sources are published and independently reverified under
`Datasets/Sentinel-2/`. All 116 file hashes and local XML readiness match.
The original 2,146 source fingerprints were checked and remain unchanged.
Publication verification peaked at 111.59 MiB RSS. Checkpoint `59aa363`
saves the durable publication and replay receipts. Lineage preparation is
complete. Four distinct comparisons preserve both original crops and group
assignments. Each recomputes cumulative temporal geometry from its actual
pre-date and fixed post-date and passes split isolation. The guarded lineage
worker peaked at 274.25 MiB. Checkpoint `8b7fa50` saves the exact four derived
identities, temporal masks' geometry counts and source bindings before pixel
processing. Both dates preserve the corresponding original reference masks
after cumulative recomputation.

## Crop quality results

| Pre-date | Fixed post-date | Anchor / eroded pixels | Full target context flagged pre → post | Eroded interior context flagged pre → post |
| --- | --- | ---: | ---: | ---: |
| 21 July | 31 July | 63 / 16 | 174 / 174 → 0 | 92 / 92 → 0 |
| 21 July | 10 August | 239 / 129 | 464 / 464 → 0 | 320 / 320 → 0 |
| 16 July | 31 July | 63 / 16 | 174 / 174 → 0 | 92 / 92 → 0 |
| 16 July | 10 August | 239 / 129 | 464 / 464 → 0 | 320 / 320 → 0 |

Each pre-crop has 4,096 flagged pixels. The frozen screen reports cloud or
its buffer over the target alongside suspected haze. Both unchanged post
crops remain clear. Screening read 44 fused-band windows, used four cloud
classifier calls, and reused the two frozen post caches across four pair
bindings. The four masks, numeric outcomes and eight cache bindings replay
exactly from six distinct observations. Replay used zero source band reads
and zero cloud classifier calls. There were zero processing errors.

Screening peaked at 372.27 MiB worker RSS and replay at 253.64 MiB. All 391
guarded tests passed. The workers were sequential and the memory reserve
held. The live transaction is complete and no workers are active. Raw scenes
remain local. Smaller receipts, all four masks and the six bounded QA
snapshots are archived under `Docs/Reviews/ValidationFreshSourcesV24/`.

There are still no approved validation labels or label lock. The positive
support, split isolation and successful publication are available for future
planning. Preserve these two failed dates and the original post observations.
The remaining fresh 2019 groups retained in V23 give a distinct metadata-only
alternative. Check available dates and reference imagery before fixing a new
bounded roster. Keep the already exposed groups and Area B case study separate
from new-site validation.
