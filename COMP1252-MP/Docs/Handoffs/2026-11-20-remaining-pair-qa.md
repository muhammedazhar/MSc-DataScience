# Remaining July pair QA

V18 is complete. Both 6 July to 31 July 2018 pairs passed the frozen automatic
screen and are pending visual review. Both 1 July pairs and both 11 July pairs
were excluded for cloud or its buffer. All six pairs cover two event areas
within one geographic group. Continue on `codex/curated-deforestation-pilot`
and push verified checkpoints to the same branch.

| Pre-image | Event | Cumulative pre / post / new loss pixels | Anchor / eroded interior pixels | Automatic result |
| --- | --- | ---: | ---: | --- |
| 11 July | A | 122 / 269 / 147 | 55 / 8 | Excluded |
| 11 July | B | 78 / 222 / 144 | 92 / 25 | Excluded |
| 6 July | A | 122 / 269 / 147 | 55 / 8 | Pending visual review |
| 6 July | B | 78 / 222 / 144 | 92 / 25 | Pending visual review |
| 1 July | A | 27 / 269 / 242 | 55 / 8 | Excluded |
| 1 July | B | 0 / 222 / 222 | 93 / 25 | Excluded |

Event A is `event-71d8da17cbbfd9babdee`. Event B is
`event-ce1c1df8d02be33bb3c8`. The passing 6 July pairs are
`derived-remaining-9a36abc2df4bc919d207` (A) and
`derived-remaining-e61a8a8e623222b69efb` (B). Both have zero QA flags on the
full annotated loss reference and on the eroded anchor interior and its
radius-two context. The 31 July side of A has four shadow-risk pixels in
the full target context (176 pixels), all outside the loss target and the
56-pixel eroded interior context. The 6 July pre-images have zero flags in
either full target context. B's 274-pixel full target context has zero flags
on both dates.

Every pair has a distinct ID, the original parent hash, the original crop and
split, and explicit 36UXB source paths. Training and test isolation passed.
The 6 July baseline includes annotations observed that day. The 1 July
baseline precedes those observations and adds 95 and 78 pixels to the
recomputed loss unions relative to the parent references. The 1 July pair
has a 30-day interval, the inclusive maximum of the frozen rule.

All 391 guarded tests passed, including the six distinct identities, 30-day
boundary, and cumulative subtraction. Preparation took 2.79 seconds and
peaked at 239.48 MiB. Screening took 16.02 seconds and peaked at 370.62 MiB
with at least 2.98 GiB available. It used 12 observation requests, reused two
unique 31 July caches, and made six cloud model calls and 66 fused band reads.
Replay took 3.06 seconds and peaked at 240.06 MiB. It reproduced the numeric
results, arrays, and anchor contexts for six masks and 12 cache bindings from
saved observations, with zero JP2 reads and zero model calls. The unchanged
guard used one nice +10 CPU worker, one native thread, a 512 MiB RSS cap,
a 1.5 GiB available memory reserve, and a 300-second limit per QA worker.

Checkpoint `a747afd` pins the protocol and verified lineage before screening.
Checkpoint `762dfed` saves all six outcomes and masks. The completion
checkpoint saves the replay receipt, eight cache snapshots and final
integrity evidence. See `Docs/Reviews/ValidationRemainingScreenV18/`.
Local outputs are under `Datasets/ValidationPilot/v18/`.
All 232 used source hashes were checked in each worker. Final integrity also
checked 406 distinct prior or current file stats, the three completed source
transactions, and the six prior QA exclusions. No workers are active.

The two 6 July candidates are ready for visual review of pre-event forest
and post-event clearing, including the four context shadow flags for A.
Class evidence, control review, later persistence, and 27-channel eligibility
are still needed for any label approval. No first-visible interval or
validation accuracy has been established.
