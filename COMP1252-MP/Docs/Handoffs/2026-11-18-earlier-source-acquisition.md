# Earlier source acquisition continuation

## Current checkpoint: V14 predeclared; implementation and acquisition pending

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
No imagery has been downloaded at this checkpoint. Implementation, tests,
actual local QA readiness and publication are INCOMPLETE.

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

358 tests passed at V13. Next: implement and test strict acquisition/recovery,
freeze hashes, acquire the two sources separately, verify full payloads/local
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
