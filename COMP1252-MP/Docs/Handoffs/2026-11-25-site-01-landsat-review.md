# Site-01 bounded Landsat reference review

V33 is complete. Both dated reference crops are inconclusive at the target
and the control. The nominated Sentinel-2 pair stays pending.

Use `codex/curated-deforestation-pilot` and save verified checkpoints
on the same GitHub branch. Start with the
[V32 nomination](2026-11-25-remaining-site-selection.md) and the
[V33 protocol](../Reviews/ValidationSite01ReferenceV33/protocol.json).
Keep the original 64 × 64 site-01 crop, 237-pixel target, 151-pixel eroded
interior and fixed 16 × 16 control at zero-based row 44, column 41.
The control's full two-pixel context stays fixed.

The two requested references are the exact Tier 1 USGS Landsat 8 scenes
from 12 January and 20 February 2018. The scene-locked Esri export is
43 × 43 in EPSG:32636 and covers the frozen crop. Each request is capped
at 64 KiB. Use one nice +10 CPU worker with one native thread, a 512 MiB
RSS limit and a 1.5 GiB available-memory reserve. Keep raw images and
rendered boards local. Archive the request receipts and hashes.

Inspect plain RGB first and outlines second. Check the target's local
visibility, evidence for earlier forest and a later opening, and the
full control for stable forest-like cover. The 20 February reference
can only corroborate the later appearance. The Sentinel-2 after date
remains 6 February. Support from both Landsat views is required before
the conditional native Sentinel-2 metadata preflight.

The guarded worker retrieved the 12 January and 20 February crops in
3,067 and 3,850 bytes. The [packet receipts](../Reviews/ValidationSite01ReferenceV33/packet/packet.json)
bind both scene-locked PNGs, the exact geometry and the review boards.
The [offline replay](../Reviews/ValidationSite01ReferenceV33/packet_verification.json)
reproduced the boards and packet byte for byte. Peak RSS was 131.81 MiB
for preparation and 130.41 MiB for replay.

## Visual disposition

Two independent reviewers examined the plain and then outlined views. Pale
veiling crosses the January target and weakens terrain detail. Bright bands
and muted patches cross the February target and control. Both reviews leave
earlier forest, later opening and whole-square stable forest unresolved.
The [visual review](../Reviews/ValidationSite01ReferenceV33/visual_review.md)
and [disposition](../Reviews/ValidationSite01ReferenceV33/reviews.json)
preserve the exact dates, areas and ambiguity. Add
`group-c4f441ae55f683f915fa` to future visual-exposure exclusions.

The conditional Sentinel-2 preflight did not advance. No full Sentinel-2
scene was transferred and no labels or first-visible interval were produced.
All 394 existing tests passed. The
[integrity check](../Reviews/ValidationSite01ReferenceV33/final_integrity.json)
binds the archived evidence and the local preview hashes. Site-02 remains
the untouched V32 reserve with weaker independent reference metadata.
