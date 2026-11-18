# Earlier source acquisition continuation

## Current checkpoint: both scenes staged/verified; publication pending

Continue on `codex/curated-deforestation-pilot`; commit each verified logical
change with this handoff updated and push to the same GitHub branch. Read
`2026-11-18-earlier-source-metadata.md` and linked V12/V11/V10/older handoffs
for exact provenance, frozen QA and prior failures. Last completion is `01bf245`.

The user approved continuing. V14 acquires exactly the two V13 pinned sources:
21 July 2018 S2A then 16 July 2018 S2B, both N0206/R064/36UXB. Each has 58 required
files; combined payload 1,050,527,541 bytes. No fallback, source substitution,
catalogue cloud ranking or extra date. Use the explicit genuine L1C provider
manifest, NOT the saved STAC hrefs that point to the L2A bucket.

Protocol: `Docs/Reviews/ValidationEarlierAcquisitionV14/protocol.json`.
Both dates are acquired and locally metadata-ready in STAGING only. Each has
58 assets with pinned source MD5/size plus SHA256; actual unchanged frozen QA
metadata prerequisites pass without issues. Nothing is published.
16 July is also acquired and metadata-ready in STAGING. Final publication
remains INCOMPLETE. Protocol `4cde3e4`
and tested/frozen implementation `e79eabe` are pushed.

Stage under persistent `Datasets/ValidationPilot/v14/source_transaction`, outside
Sentinel-2 discovery and guard report staging. Stream exact pinned objects with
If-Match/Content-Length/ETag/MD5/SHA checks. Reuse only verified complete files;
record partial-file ownership durably and remove only matching owned partials
after interruption. Ambiguous ownership or unexpected files requires inspection.
Use one exclusive lock and guarded scene-sized jobs; no automatic retries.

Before publication, both complete scenes must pass unchanged actual
`qa_scene_metadata` without placeholders. Publication adds two NEW directories
using exclusive no-replace renames and a durable inode journal. It is recoverable,
not an atomic pair swap. Never overwrite/replace old or user-owned trees.
Retain all old scenes, V10 transaction/backups and known-cloudy V11 outcomes.

Mac: unchanged `validate_sample.run_guarded`, one nice +10 CPU worker, native
threads 1, no GPU, 512 MiB RSS, 1.5 GiB available-system-memory reserve, maximum
300 s per worker (tests 120 s). Initial check found 2.84 GiB available memory and
19.6 GiB free disk; these change with other Mac activity. Recheck capacity before
each job/file, require conservative initial staging allowance plus 2 GiB disk
reserve and preserve the reserve during transfers. No active project worker.

369 tests pass (358 prior plus 11 new offline tests). Initial 368-test suite
passed; read-only review then identified the frozen helper's predictable journal
.tmp writer as an ownership risk. V14 now uses its own exclusive random temporary
writer; the extra symlink regression and full 369 suite pass. No frozen helper
was edited. Hardened suite 5.09 s, RSS 360.27 MiB high-water / 366.6 MiB sampled,
minimum 3.67 GiB available. Tests cover HTTP pin/payload failures, corrupt nonempty
reuse, partial inode/path refusal, existing user trees, exclusive rename,
publication rollback/crash resume and local QA refusal. No pixel/model jobs.
Persistent completion plus read-only verify support a crash after publication
but before guard report completion. Error/recovery history is retained.

Next: acquire the two sources separately, verify full payloads/local
prerequisites and recoverably publish. Preserve every outcome and exact evidence.
No raster decoding, cloud/forest model, new temporal pairs/targets, labels,
training or evaluation is part of V14. Local metadata readiness will not prove
clear pixels, full 27-channel eligibility or a forest-to-clearing transition.
Both events remain one group; original grids/splits/QA/model and 0/1/255 masks stay
frozen. Later earlier-pre-date pairs need distinct lineage and recalculated
cumulative targets, not V11's hardcoded dates/masks.

Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`;
Git root is its parent. Completed outputs must never be overwritten.

## Verified 21 July acquisition checkpoint

Completed `Datasets/ValidationPilot/v14/download_21/`: 142.87 s, RSS 62.84 MiB high-water / 61.2 MiB sampled, minimum 3.51 GiB available. 58 new downloads, 0 reused; free disk 20.623 to 20.066 GiB.
Evidence and exact journal-at-job-completion snapshot are under
`Docs/Reviews/ValidationEarlierAcquisitionV14/download_21/`. Its recorded journal
hash is historical; future steps change the live journal. Persistent state is
`acquiring`, scene 0 prepared, scene 1 new, no partials/errors. Completed source
files remain under transaction/staged, not Sentinel-2. Do NOT rerun this worker.
Next: one guarded download of scene index 1 into NEW `v14/download_16/`, then
publication only after both full sources pass. No pixel/model/label work.

## Verified 16 July acquisition checkpoint

Completed `Datasets/ValidationPilot/v14/download_16/`: 265.82 s, RSS 65.53 MiB high-water / 61.9 MiB sampled, minimum 4.09 GiB available. 58 new downloads, 0 reused; 490,573,427 bytes; free disk 20.060 to 19.513 GiB.
Evidence/snapshot: `Docs/Reviews/ValidationEarlierAcquisitionV14/download_16/`.
Both sources have 58 verified files; persistent state is READY, both prepared,
no partials/errors/retries. Source 21 checkpoint `cba52ed` is pushed; older
checkpoint's pending-next-step statements are historical. Do NOT rerun either
download. Next: publish mode (no scene index), NEW `v14/publication/` report.
Publisher rehashes both complete sources, checks actual frozen metadata, then
uses journaled native no-replace renames and verifies final sources. Original
sources remain untouched. Pixel/model/label work is still outside this phase.
