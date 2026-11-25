# Site-03 feature audit and model-blind review

V31 is complete. Site-03 is excluded from held-out evaluation because most of
the opening was already visible before the pair and the fixed control is mixed.

Work on `codex/curated-deforestation-pilot` and push verified checkpoints to
the same branch. Start with the [V30 handoff](2026-11-25-site-03-acquisition-qa.md)
and keep its exact crop, dates, source lineage, target masks, control and
geographic split. V31 covers the 177-pixel target, 115-pixel eroded interior
and their full two-pixel contexts. The 256-pixel control and its 400-pixel
context stay fixed.

Audit the 27 frozen feature channels on the 30 May–4 June pair in the target,
interior and control contexts and on the 10–30 May pair in the control context.
Use one nice +10 CPU worker with one native thread, the unchanged 512 MiB cap,
1.5 GiB available-memory reserve and 300-second cap. Replay from archived
arrays. All model weights, splits and class labels remain frozen.

The [visual protocol](../Reviews/ValidationSite03ReviewV31/visual_protocol.json)
pins the two target dates and three control dates. Open the plain matched RGB
boards before the outlines and vegetation index. Inspect for earlier
forest-like cover and a coherent new opening on 4 June. Require the entire
control to remain consistent with forest across all three observations.
Investigate traceable dated reference access from V28. Preserve class
uncertainty and record visual exposure separately from validation status.

## Verified inputs and packet

The [feature audit](../Reviews/ValidationSite03ReviewV31/inputs/audit.json)
found all 27 channels finite on both target contexts and the control context
on its two pairs. The [replay](../Reviews/ValidationSite03ReviewV31/input_verification/verification.json)
matched both tensors and all context checks. Feature extraction used 28
bounded band windows and peaked at 188.45 MiB. Its replay used zero original
band windows and peaked at 152.41 MiB.

The [packet](../Reviews/ValidationSite03ReviewV31/packet/packet.json) and its
[replay](../Reviews/ValidationSite03ReviewV31/packet_verification/metadata.json)
bind all six boards and their V30 archive evidence. Packet preparation and
byte-identical replay used no original band windows and peaked at 146.50
and 145.12 MiB.

## Model-blind disposition

Both reviewers saw the pale opening over most of the fixed target and eroded
interior on 30 May. The June difference is mainly at the margins. The proposed
control has a stable vegetated main block, with a persistent pale strip across
its west side and a mixed northwest corner. The whole square cannot serve
as a confirmed stable-forest control. The [visual review](../Reviews/ValidationSite03ReviewV31/visual_review.md)
and [dispositions](../Reviews/ValidationSite03ReviewV31/reviews.json) preserve
the fixed geometry, exclude the pair and control from evaluation and record
the new visual exposure of `group-a85fa2022a25e1a72e2c`. No labels were approved.

## Independent reference review

The [access report](../Reviews/ValidationSite03ReviewV31/references/access.json)
confirms two public Landsat 8 scene-locked crops. The [reference review](../Reviews/ValidationSite03ReviewV31/references/review.json)
records a broad opening already visible on 27 May in the same location.
Cloud obscures the 3 June crop. USGS reports 99.99% scene cloud for 3 June.
The Esri catalogue value 0.9999 is a fraction. Each preview is 43 × 43 with
15 m pan-sharpened sampling and 30 m source multispectral bands. The preview
supports the existence of the opening by 27 May and leaves its exact
boundaries, earlier forest class and onset unresolved. The two small source
previews are local. The report contains the exact URLs and hashes.

The 23 May SPOT-7 record covers the crop. Its dated pixels and clarity remain
unverified because archived previews require authenticated access. The
Wayback leads use a 13 September 2012 capture. Neither resolves the onset.

All 394 existing tests passed under the guard. The
[final integrity record](../Reviews/ValidationSite03ReviewV31/final_integrity.json)
binds the feature audit, exact packet replay, class dispositions, reference
access receipts and tests. There is no newly approved validation case or
first-visible deforestation interval. For a later search, use an unexposed
group with credible earlier forest evidence and a homogeneous control.
