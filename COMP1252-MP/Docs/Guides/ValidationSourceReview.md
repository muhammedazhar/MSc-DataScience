# Antigravity review, source repair and quality screening

The review found an incomplete QA-input contract in the downloaded scenes.
Both scenes are now repaired and pass the actual frozen QA prerequisites.
Screening works, but **both fixed pairs fail the cloud gate in their pre-image**.
No validation labels, clearing-date interval or accuracy were produced.

## Review findings and correction

The seven-band inventory check used by V9 is weaker than the frozen pixel-QA
contract. Four additional cloud bands and 44 source quality GMLs per scene were
missing. The archived downloader also trusted arbitrary nonempty existing files
and wrote into final source paths before the pair had been validated. Historical
catalogue/acquisition summaries overstated mixed-baseline compatibility and size
precision. See the [review](../Reviews/AntigravityReview/2026-11-18-review.md).

A separate source-repair path fetched 96 source-pinned assets (74,541,437 bytes,
71.1 MiB), enforcing exact size, source MD5/ETag, If-Match and SHA receipts.
All 20 original files match their V9 SHA and provider MD5/size. Both full scene
trees were staged outside discovery, checked with actual `qa_scene_metadata`,
and published through a durable journal with rollback/recovery and original
backups retained. This is recoverable publication, not a single atomic two-scene
swap. No placeholder quality masks or frozen-source modifications were made.

## Quality result

Two NEW derived-source IDs identify the substituted 36UXB sources. Each retains
the original parent ID/hash, grid, dates, anchor and temporal label IDs, group
and split. Original 36UXA candidates and their coverage blockers stay unchanged.
Full feature footprints still avoid all frozen train/test and 13 training grids.

| Event suffix | Anchor-specific target / interior pixels | Full target context: pre / post flagged pixels | Outcome |
| --- | --- | --- | --- |
| `71d8da17cbbfd9babdee` | 55 / 8 | 176/176 / 4/176 | Excluded |
| `ce1c1df8d02be33bb3c8` | 92 / 25 | 274/274 / 0/274 | Excluded |

The 26 July 2018 frame flags all 4096 core pixels at both grids; every target and
supervised interior is cloud/buffer-flagged. On 31 July the target interiors and
their contexts have zero flags; the first event's FULL target margin retains
four shadow-risk pixels. No ROI is shrunk to hide those flags. Clear post target
ROIs cannot establish a forest-before/clearing-after transition by themselves;
the post cores still contain 64/88 shadow-risk pixels.

Four frozen cloud-QA executions made 44 bounded fused-band requests. No forest
model/Torch was loaded, no imagery was displayed, and no class labels were
assigned. Cached RGB/NDVI finiteness is checked; all 27-channel eligibility is
explicitly unestablished. The context margin belongs to the frozen tiny CNN.

Verification replays four bounded safe caches and two six-plane QA TIFFs,
reconstructing masks and anchor-specific contexts and rechecking all 116 source
assets and exact derived lineage. It shares lineage/QA/geometry helpers; it does
not independently predict clouds or certify visual class truth. Original source
rasters are not decoded again for this replay.

## Preservation, resources and continuation

All 341 tests pass (324 previous + 8 source-repair + 9 derived-screen cases).
Source repair:43.19 s/58.36 MiB; QA:18.40 s/265.44 MiB; replay:3.31 s/144.88 MiB;
full tests:7.94 s/298.27 MiB. All were sequential nice +10 CPU workers, threads 1,
no GPU, under the unchanged 512 MiB RSS and 1.5 GiB available-memory reserve.
Minimum available memory across these jobs was 2.69 GiB. Source worker measured
free disk 28.56 GiB before and 28.53 GiB after; filesystem free space is dynamic.

V10 evidence: `Docs/Reviews/ValidationSourceRepairV10/`. V11 protocol,
`screen/`, `verification/` and byte-identical bounded cache copies:
`Docs/Reviews/ValidationSourceScreenV11/`. Full source data remain local.
The current pair is retained as excluded, without automatic replacement.

Next proposed task: a bounded metadata-only search of earlier pre-image dates
within 30 days before 31 July 2018. Keep the cloudy 26 July outcome explicit and fix
any additional source roster before acquisition/QA. Later persistence, complete
27-channel checks and blind class review would still be needed before labels.
Both events remain one geographic group, not two independent validation locations.
