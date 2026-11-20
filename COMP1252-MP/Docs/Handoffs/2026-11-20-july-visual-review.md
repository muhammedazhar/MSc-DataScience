# Visual review of the 6–31 July pairs

V19 is complete. Both 6–31 July pairs were visually reviewed. Both show
plausible expansion around older openings. Area B supports a provisional
optical bracket of **after 6 July and by 31 July 2018** for a newly visible
lower patch. First-visible deforestation timing remains unresolved for both
full targets. Label and validation eligibility remain pending.
Continue on `codex/curated-deforestation-pilot` and push verified checkpoints
to the same branch.

The fixed roster was both 6 July to 31 July 2018 pairs that passed automatic
QA. The packet uses the original 64 × 64 grids and four hash-verified V18
cached observations. The reviewers opened the matched fixed-scale plain RGB
first, followed by exact contours, nearest-neighbour detail, QA and NDVI.

See `Docs/Reviews/ValidationJulyVisualV19/protocol.json` for the predeclared
roster, inspection rule, and resource limits. `visual_review.md` and
`reviews.json` record both dispositions with the exact six image hashes.
`group-96c27871459d03c08bcc` is now visually exposed. Both crops overlap
within this one group. Retain all six previously exposed groups and this
group when excluding visually seen locations from fresh validation.

Area A's four 31 July shadow-risk cells lie in one column just southwest of
the target, outside both the target and eroded interior context. The central
and southern target brighten and lose NDVI away from those cells. Its narrower
pre-existing strip and the flagged flank leave the full transition uncertain.
Area B has QA-clear context on both dates. Its new lower pale block has
coherent NDVI decline. The upper hook was partly open on 6 July, so the lower
patch expansion bracket is spatially limited.

Checkpoint `efc484d` saves the protocol, renderer and byte-replayed packet,
committed before visual inspection. Under `Docs/Reviews/ValidationJulyVisualV19/`,
the archived `packet/` includes six boards and the input bindings and statistics.
`packet_verification/` records the exact replay of all seven artifacts.
`checks/final_integrity.json` records the verified decisions, evidence hashes,
full roster, exposure update and guard limits. Preparation and replay peaked
at 161.98 and 159.02 MiB respectively, with zero JP2 reads. All 391 offline
tests passed at 340.61 MiB peak RSS. All three guarded jobs are finished.

## Next bounded task

Inventory later observations for the same two crop grids, starting with
available metadata for a clear post-31 July observation on the frozen tile
and orbit. Predeclare a bounded persistence check before viewing any additional
pixels. The lower patch in Area B has the strongest evidence to revisit. Keep
the earlier QA exclusions, both visual dispositions and the shared exposed
group. Separately verify stable-forest control support and all 27 frozen
endpoint channels before seeking label approval. Preserve the single nice +10
CPU worker, one native thread, 512 MiB RSS cap, 1.5 GiB available memory reserve,
and 300-second limit. See `2026-11-20-remaining-pair-qa.md` and its linked
handoffs for the frozen source and QA lineage.
