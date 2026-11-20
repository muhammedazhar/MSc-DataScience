# Area B August persistence check

V20 checks whether the lower opening that appeared by 31 July 2018 remains
visible on the same Area B grid after July. Continue on
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
passed. All 391 existing tests passed under the guard. Live metadata, acquisition,
QA and visual dispositions are pending. Keep one nice +10 CPU
worker, one native thread, 512 MiB RSS, 1.5 GiB system memory reserve and a
300-second job limit. Maintain at least 2 GiB disk reserve during acquisition.
