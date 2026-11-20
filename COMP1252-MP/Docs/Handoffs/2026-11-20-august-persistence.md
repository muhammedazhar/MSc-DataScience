# Area B August persistence check

V20 is complete. The lower opening visible on 31 July 2018 remains visible
on 10 August, ten days later, on the same Area B grid. The provisional
optical bracket of after 6 July and by 31 July still applies to the lower
expansion. First-visible deforestation timing remains unresolved. Continue on
`codex/curated-deforestation-pilot` and push verified checkpoints to that branch.

The local scene and cache inventories stop at 31 July on the required tile.
The fixed two-date follow-up is 5 and 10 August 2018 using the corresponding
L1C/N0206/R064/T36UXB products identified in the V7 archive. The dates were
selected chronologically before new pixel inspection. Both dates keep a
disposition, including a failure or missing source. The V7 normalized metadata
is an identity lead. Fresh raw item geometry, provider XML and listed assets
must pass the source checks before acquisition.

Use the exact Area B grid and target from V19 and the full 274-pixel target
context for the review gate. Compare the same 6 July, 31 July and later crop
with fixed-scale RGB and use NDVI as supporting evidence. Keep the older pale
opening and the new lower block distinct. The V19 interpretation, all earlier
exclusions and the visually exposed group remain recorded as they were.

The protocol and four runner implementations are pinned under
`Docs/Reviews/ValidationPersistenceV20/`. The fixed roster and syntax checks
passed. All 391 existing tests passed under the guard. Checkpoint `6feff78`
pinned the two dates and the runners before network requests.

Both 5 and 10 August passed the live metadata preflight and exact offline
replay. Raw catalogue items and four provider metadata requests per date
confirmed full footprint, matching L1C XML, and 58 pinned files per scene.
The total listed payload is 899,759,383 bytes and the conservative staging
allowance is 1,927,511,952 bytes plus a 2 GiB disk reserve. The source volume
had 26,831,347,712 bytes available at preflight. The preflight peaked at
68.19 MiB RSS. Checkpoint `ffe6fc1` saves the raw metadata and exact replay.

The 5 August scene is staged with all 58 assets verified against the provider
pins and hashed. Staging passed the local frozen metadata checks. Its guarded
download took 32.00 seconds at 68.09 MiB peak RSS. Checkpoint `09ed6e5` saves
its receipt. The 10 August scene also passed staging with 58 verified assets.
It took 41.91 seconds at 62.66 MiB peak RSS, recorded at `4cfd4f5`.

Both scenes are published under `Datasets/Sentinel-2/`. Independent
publication verification matched all 116 asset hashes, local metadata and
the completed transaction receipt. Publication and verification peaked at
52.14 and 56.53 MiB RSS. Checkpoint `0db0e35` saves publication and verification.

Crop QA and cached replay are complete. The 5 August image has 17 shadow-risk
pixels in the full 274-pixel target context and fails the visual review gate.
The target itself and its eroded interior context have zero flags. The 10
August image has zero flags across the complete crop and is ready for visual
comparison. Both dates used the original grid. Screening made two cloud-model
calls and 22 fused-band requests, with peak RSS 337.00 MiB. Replay reproduced
both QA rows, two snapshot archives and three views with no band reads or cloud
model calls at 142.50 MiB peak RSS. Checkpoint `8337af9` pins those outcomes
and the clear comparison before visual inspection.

Both reviewers opened the plain RGB comparison before detail and NDVI. The
10 August lower block remains in the same location, with the same straight
southern edge and narrow connection to the older opening. NDVI remains low
within the block. `visual_review.md` records the findings and all three views.
`reviews.json` binds the visual dispositions for both dates to the replay,
exact images and V19 review. `checks/final_integrity.json` checks the complete
archived evidence and resource receipts. All 391 tests passed. No job is active.

Keep one nice +10 CPU
worker, one native thread, 512 MiB RSS, 1.5 GiB system memory reserve and a
300-second job limit. Maintain at least 2 GiB disk reserve during acquisition.

The next bounded task is to verify Area B's full frozen 27-channel endpoint
eligibility and predefine stable-forest control support on the original grid
before class approval. Keep the spatially limited persistence result, all earlier
exclusions, and the existing exposed geographic group. The two completed source
folders and transaction are local. GitHub holds the code and bounded receipts,
snapshots and previews. Do not repeat the completed downloads or screening.
