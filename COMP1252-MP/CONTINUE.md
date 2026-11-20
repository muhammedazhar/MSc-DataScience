# Continue this project with any coding agent

Work on **`codex/curated-deforestation-pilot`**. Read `AGENTS.md`, check Git
status, then the latest handoff before edits. Commit every completed verified
logical change with the handoff updated and push to the SAME branch. This entry
point works for Codex, Gemini Antigravity or another agent without chat history.

Current V20 Area B persistence check:
**`Docs/Handoffs/2026-11-20-august-persistence.md`**.
Completed V19 visual endpoint review:
**`Docs/Handoffs/2026-11-20-july-visual-review.md`**.
Completed V18 remaining July pair QA:
**`Docs/Handoffs/2026-11-20-remaining-pair-qa.md`**.
Completed V17 source acquisition:
**`Docs/Handoffs/2026-11-20-remaining-source-acquisition.md`**.
Completed V16 metadata preflight:
**`Docs/Handoffs/2026-11-20-remaining-source-metadata.md`**.
Completed V15 earlier-pair QA state:
**`Docs/Handoffs/2026-11-19-earlier-pair-qa.md`**.
Completed V14 earlier-source acquisition state:
**`Docs/Handoffs/2026-11-18-earlier-source-acquisition.md`**.
Completed V13 earlier-source metadata state:
**`Docs/Handoffs/2026-11-18-earlier-source-metadata.md`**.
Completed V12 metadata-only catalogue state:
**`Docs/Handoffs/2026-11-18-earlier-preimage-catalogue.md`**.
Completed review/repair/QA state:
**`Docs/Handoffs/2026-11-18-validation-source-review.md`**.
Earlier handoffs (historical claims qualified by the review):
`2026-11-17-validation-ingestion.md`, `2026-11-17-validation-acquisition.md`,
`2026-11-17-validation-catalogue.md`, `2026-11-17-validation-coverage.md`,
`2026-11-17-validation-unscreened.md`, `2026-11-17-validation-manual.md`,
`2026-11-17-validation-context.md`, and linked 16 November training/validation and
15 November dataset/provenance handoffs under `Docs/Handoffs/`.

Project root:
`/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Git root is its parent MSc-DataScience directory. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.

The project locates forest loss and aims to estimate when it first becomes
visible in Sentinel-2 images. Annotation image dates are not exact felling dates.
The partial-label pipeline and tiny CNN fit all 13 reviewed TRAINING regions;
this establishes neither unseen-location accuracy nor complete-mask correctness.
The frozen final checkpoint remains 1664 at 0.5; no retraining or tuning is needed.

V1 rejected all five reviewed validation cases. V2's broad support audit retained
an empty fresh shortlist. V3 cleared one target's input context but found no
automatic control; V4 manual review rejected it for ambiguous class evidence.
V5 retained all 606 unscreened pairs/630 anchors with coverage/provenance blockers.
V6 confirmed two fresh supported events lay north of all 50 original local scenes.
Antigravity V7–V9 located and acquired two covering 36UXB L1C/N0206 scenes, but
only seven feature bands: the actual frozen QA prerequisites remained incomplete.
Its broad baseline/compatibility summaries and ingestion reuse/publication
assurances were qualified by the 18 November review; preserve archived receipts.

The completed V10 correction added exactly 96 provider-pinned assets (four cloud
bands and 44 quality GMLs per scene; 71.1 MiB), verified original 20 SHA/source-MD5/size,
staged outside discovery and published using a recoverable transaction. Both
scenes now pass actual `qa_scene_metadata` with no issues. Original backups and
persistent transaction journal are retained. Do not rerun archived V8/V9 ingestion.

V11 screened TWO explicit derived pairs, preserving parent grids, annotations,
label IDs, geographic group and lineage. Original 36UXA pairs remain blocked;
new 36UXB source IDs do not impersonate frozen candidates. Both derived pairs are
**excluded by automatic QA**: the 26 July 2018 pre-image flags all target/context
pixels as cloud/buffer.31 July is cleaner but cannot replace a clear before image.
Four cache observations, two six-plane QA masks and all numeric context outcomes
were replay-verified. No visual images were opened, labels approved or date/
accuracy estimated. This phase does NOT prove all 27 feature channels finite.

V12 retained 18 catalogue products (12 N0206/6 N0500) across 12 dates and both
known-cloudy 26 July variants. Five nominal earlier metadata follow-up dates match
the fixed tile/orbit/baseline/footprint rules: 1, 6, 11, 16 and 21 July 2018. Their pixels
and QA remain unexamined; reader compatibility is unverified and no roster was
selected. Two raw pages (18 items + empty terminal) and all variants are backed up.

V13 fixed the metadata roster to 21 then 16 July by date alone. Genuine L1C
product/tile XML and complete 58-file listings pass for both. The STAC records'
L2A asset hrefs were preserved as a catalogue anomaly and never fetched; use the
explicit verified L1C provider manifest. No imagery/GML content was downloaded,
source published, QA run or label approved. Total future payload 1,050,527,541 bytes.

V14 acquired/verified/published both sources: 116 genuine files, 1,050,527,541 bytes.
Actual frozen QA metadata passes with no issues for both; 54 local L1C product
folders now exist. Transaction is COMPLETE; no partials/retries/failures or source
replacements. Prior sources and cloudy outcomes are unchanged. No pixels decoded,
models run, new pairs/targets created or labels approved; clarity is unknown.

V15 derived all four fixed earlier-date pairs with new IDs, 10/15-day intervals
and recalculated cumulative targets before pixel work. All four are automatically
QA-excluded: both 21/16 July pre-images flag every target/context pixel cloud/buffer.
All outcomes, four QA masks and six safe cache snapshots were replay-verified;
old 26 July failures/exposure/source hashes remain intact. No labels, manual
images, forest models, full 27-channel proof, date estimate or accuracy.

V16 completed metadata checks for the three remaining nominal V12 dates
(11, 6, and 1 July 2018). All three passed L1C XML compatibility and had 58
required assets listed. The combined listed payload is 1,664,246,549 bytes.
The conservative staging estimate fit the available disk with the 2 GiB reserve.
V17 acquired all three scenes and verified 174 files against their provider
pins and saved SHA-256 checksums. All three passed the frozen local metadata
checks and are published under `Datasets/Sentinel-2/`. Download workers peaked
at 66.94 MiB RSS. The six earlier QA exclusions remain recorded.
V18 prepared and screened six pairs with cumulative references. Both 6 July
to 31 July pairs passed automatic QA and await visual review. Both 1 July
and both 11 July pairs were excluded for cloud or its buffer. Replay verified
all six masks and the numeric results. QA peaked at 370.62 MiB.
**391 tests pass.** No worker is active. See the V18 handoff for pair IDs,
context flags, and verification.
Both events share ONE geographic group and are not independent locations.

V19 visually reviewed both 6–31 July pairs from the four verified V18 caches.
Area A shows plausible expansion near a narrow older strip and four flagged
context cells. Area B supports a provisional optical bracket of (6, 31] July
for the lower expansion. Earlier pale ground leaves the first-visible
deforestation date unresolved for both targets. The shared group is now
visually exposed. Both assessments, six views and their exact replay are
archived. The next bounded task is a later-observation metadata inventory for
persistence. Stable-forest control review and full 27-channel eligibility
remain requirements for label approval.

Use one nice +10 CPU worker, native/Torch threads 1, no GPU, unchanged
`validate_sample.run_guarded`, 512 MiB RSS and 1.5 GiB available-system-memory
reserve, 300 s maximum. Actual latest peaks: source repair 58.36 MiB, QA 265.44 MiB,
replay 144.88 MiB, V12 inventory ~60 MiB, V13 metadata 45.56 MiB, V14 acquisition
65.53 MiB, publication 51.39 MiB, V15 QA 360.02 MiB, replay 179.27 MiB, 380 tests
333.52 / 334.0 MiB. Preserve masks
0 forest/1 clearing/255 unknown,
all original frozen sources and full train/test footprint barriers.

Completed output directories have `metadata.json`; never overwrite or duplicate
a completed/active worker. Source repair's persistent transaction/backups live
outside report staging and must be retained for recovery/provenance. Source data
and full cache collection remain local; code and bounded evidence are on GitHub.
