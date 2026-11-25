# Site-01 bounded Landsat reference review

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

The guarded downloads, offline replay and visual review are pending.
