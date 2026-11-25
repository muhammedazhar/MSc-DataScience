# Site-02 bounded Landsat reference review

V34 is complete. Both site-02 reference crops are inconclusive at the
target and the fixed control. The Sentinel-2 reserve remains pending.

Use `codex/curated-deforestation-pilot` and push verified checkpoints on
the same branch. Following the site-01 disposition, advance the untouched
V32 reserve to a small independent reference review. The
[V34 protocol](../Reviews/ValidationSite02ReferenceV34/protocol.json) pins
the 16 November–6 December 2017 Sentinel-2 option, the original 64 × 64
site-02 crop, the 194-pixel target and the 131-pixel eroded interior.
The V32 alternate 16 × 16 control is at zero-based row 5, column 45. Review
its full 400-pixel context.

Use the scene-locked Landsat views for 16 November 2017 and 3 January 2018.
The Tier 1 before reference was captured about 24 minutes before the
Sentinel-2 observation. The Tier 2 later reference was captured 28 days
after the Sentinel-2 endpoint. It provides coarse later context and has
91.21% scene cloud. Geometric and crop clarity uncertainty remain.
The [January metadata receipt](../Reviews/ValidationSite02ReferenceV34/late-access.json)
confirms the exact EPSG:32636 extent and public 43 × 43 export route.

Review the plain view before the outlines. Check crop clarity, whether the
target looks forested on 16 November, whether a coherent opening appears
in the later view, and whether the fixed control appears homogeneously
forested on both dates. If unresolved, retain the outcome and keep the
Sentinel-2 preflight pending. The two crop requests are capped at 64 KiB
each. Use one nice +10 CPU worker, one native thread, 512 MiB RSS and
1.5 GiB of available memory. The raw crops and boards stay local.

The guarded worker retrieved the two crops in 3,645 and 2,851 bytes.
The [packet receipts](../Reviews/ValidationSite02ReferenceV34/packet/packet.json)
bind the exact source crops, masks and boards. The
[offline replay](../Reviews/ValidationSite02ReferenceV34/packet_verification.json)
reproduced both boards and the packet byte for byte. Peak RSS was
144.81 MiB for preparation and 139.61 MiB for replay.

## Visual disposition

Two reviewers found mottled veiling and a diffuse dark streak across the
November target. A dark patch reaches into the November control. In the
January crop, a broad smooth dark area coincides with the target and pale
cover obscures the control. The
[visual review](../Reviews/ValidationSite02ReferenceV34/visual_review.md)
and [disposition](../Reviews/ValidationSite02ReferenceV34/reviews.json)
retain all observations and the two dates. Add
`group-d75dc0544bf1a0e519ad` to future visual-exposure exclusions.

All 394 existing tests passed. The
[integrity check](../Reviews/ValidationSite02ReferenceV34/final_integrity.json)
binds the indexed evidence and local previews. No native Sentinel-2
preflight or scene transfer occurred. No held-out labels or first-visible
interval were approved. The three groups on the original V28 shortlist
have now been visually examined. Future candidate selection must use the
updated exposure history and seek a clearer dated before reference.
