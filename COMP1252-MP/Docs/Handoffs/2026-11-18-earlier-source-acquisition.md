# Earlier source acquisition continuation

## Current state: V14 complete; both new sources published and metadata-ready

Continue on `codex/curated-deforestation-pilot`; update this handoff, commit and
push each verified logical change to the same GitHub branch. Read
`2026-11-18-earlier-source-metadata.md` and linked V12/V11/V10/older handoffs for
provenance, frozen QA/model/split constraints and prior failures. `CONTINUE.md`
works for Codex, Gemini Antigravity or another agent without chat history.

The approved acquisition task is COMPLETE. **21 July and 16 July 2018 are both
published under Sentinel-2 with all 58 genuine required files per scene.**
116 files / 1,050,527,541 bytes (1001.86 MiB) passed exact provider size/MD5 ETag,
If-Match, Content-Length and saved SHA256 checks. Actual unchanged frozen
`qa_scene_metadata` passes for both with no issues. There are now 54 local L1C
product folders; this statement does not assert all 54 pass strict QA.
No pixel decoding, cloud/forest model, new temporal pairs/targets, manual imagery
review, labels, training or evaluation occurred. Pixel clarity remains UNKNOWN.
No worker is active. All 369 tests pass. Do NOT repeat completed downloads.

Checkpoints: `4cde3e4` predeclared protocol; `e79eabe` tested/frozen acquisition;
`cba52ed` verified 21 July staging; `98ba410` both sources verified in staging.
Source checkpoints are pushed. HTTPS receive-pack repeatedly returned HTTP 408;
HTTP/1.1 succeeded for 98ba410 but later retries still timed out. SSH to
`ssh.github.com:443` authenticated as the same GitHub user. Use an explicit SSH
URL for the remaining checkpoint push, then fetch/check origin tracking and
compare local/remote HEAD. No force push or global Git config/remote change.
The completion checkpoint contains publication evidence and final integrity;
the transport follow-up preserves the same contents and signed commit history.

## Fixed source scope and verified acquisition

Protocol: `Docs/Reviews/ValidationEarlierAcquisitionV14/protocol.json`.
Implementation: `Solutions/acquire_earlier_validation_sources.py`.
Exactly the V13 pinned roster, ordered only by date: no cloud ranking, alternate
dates, source substitution or fallback. Use genuine explicit L1C provider URLs,
NOT the L2A hrefs in the saved L1C catalogue records. V13 preserves that anomaly.

| Date | Scene ID | Verified payload |
| --- | --- | ---: |
| 2018-07-21 | `S2A_MSIL1C_20180721T083601_N0206_R064_T36UXB_20180721T105541` | 559,954,114 bytes / 534.01 MiB |
| 2018-07-16 | `S2B_MSIL1C_20180716T084009_N0206_R064_T36UXB_20180716T122805` | 490,573,427 bytes / 467.85 MiB |

Each has 11 JP2 bands B01/B02/B03/B04/B05/B08/B8A/B09/B10/B11/B12, 44 genuine
per-band NODATA/SATURA/DEFECT/TECQUA GMLs, cloud GML, product XML and tile XML.
B06/B07 are not required by the frozen feature/cloud band union. All actual
files were streamed once from anonymous pinned L1C URLs, no redirects/paid
access/retries/reused files/partial recoveries. Four XML contents also match
V13 originals. No placeholder metadata, raster or empty masks were fabricated.

Files were staged OUTSIDE scene discovery and guard report staging, with durable
per-file ownership/receipts and complete error/recovery history. Both actual
staged products passed the frozen metadata checker before publication. The
publisher rehashed both complete sources before and after native exclusive
no-replace directory renames, then rechecked actual final metadata. This adds
two NEW directories; it does not replace old/user trees or claim an atomic pair
swap. All 116 prior V11 source size/mtime receipts, both known-cloudy exclusions,
V10 complete transaction and original backups remain unchanged.

Final metadata: genuine L1C / baseline 02.06 / orbit R064 / tile 36UXB, quantification
10000, no nonzero offsets, EPSG:32636 10 m grids covering original crops and full
QA contexts. Metadata/cloud/shadow prerequisites are all usable with no issues;
actual radiometric/grid/solar fields match V13. This is local metadata readiness,
not proof of clear inputs, valid decoded pixels, full 27-channel finiteness or
forest-to-clearing class truth. Both events remain ONE geographic group
`group-96c27871459d03c08bcc`, not independent validation locations.

## Durable transaction, receipts and recovery

Persistent `Datasets/ValidationPilot/v14/source_transaction/journal.json` is
**complete**. `completion.json` was durably written before the final journal
state. Both new directories are at `Datasets/Sentinel-2/<scene_id>/`; staged
scene directories are absent. Keep this transaction/completion receipt. The
staging parent is empty; no `.download` partial remains. Event history contains
116 transfer starts, 116 verified transfers, 2 prepared scenes and 1 publication
completion, with no transfer/worker failures or rollback.

All source receipts use scene ID + `target_relpath` under the FINAL product
folder; no old staging asset path needs to be trusted. Stage metadata paths in
the download reports are HISTORICAL. Each download report's journal hash refers
to its exact saved `journal_snapshot.json`, not the later live journal. Final
publication metadata hash matches the complete live/snapshotted journal.

Completed local report outputs (ignored, never overwrite):
`Datasets/ValidationPilot/v14/{download_21,download_16,publication}/`.
Tracked byte-identical evidence under `Docs/Reviews/ValidationEarlierAcquisitionV14/`:
`protocol.json`, both `download_*` result/metadata/report/journal snapshots,
`publication/` result/metadata/report/final journal/persistent completion copy,
`checks/pre_execution_tests{,_hardened}/`, `checks/final_integrity.json`.
Guide: `Docs/Guides/EarlierSourceAcquisition.md`.
GitHub backs up code and bounded receipts; full raw imagery remains on this Mac.

If a future interrupted acquisition needs recovery, use the same protocol,
transaction and scene index with a NEW report output. Completed files are
rehashed against source MD5/size/saved SHA; only allowed partial paths with
matching recorded device/inode may be removed. Ambiguous ownership, unexpected
files or changed user trees cause refusal, never deletion/overwrite. An
interrupted owned publication may resume, and ordinary failures restore owned
published trees to staging. Native exclusive moves also refuse empty user dirs.
For complete transactions whose guard report was interrupted, read-only verify
mode can check final files and regenerate a report under a new output; it does
not redownload or republish. No such recovery was needed in this execution.

## Tests, integrity and Mac capacity

369 tests pass (358 previous + 11 acquisition tests). The initial 368 suite
passed; review identified a predictable journal temporary-file ownership risk
in a frozen imported helper. V14 uses a NEW exclusive random temporary writer,
with a regression proving foreign temporary/destination symlinks are preserved.
The hardened 369 suite passed; no implementation changed afterwards. Frozen V10
and QA helpers remain unchanged. Tests cover pin/length/MD5 failures, corrupt
nonempty reuse, partial inode/target refusal, existing user trees, exclusive
rename, publication rollback/crash resume and refusal of unusable local metadata.

| Completed guarded job | Elapsed | RSS high-water / sampled peak | Minimum available |
| --- | ---: | ---: | ---: |
| Hardened 369-test suite | 5.09 s | 360.27 / 366.6 MiB | 3.67 GiB |
| 21 July download | 142.87 s | 62.84 / 61.2 MiB | 3.51 GiB |
| 16 July download | 265.82 s | 65.53 / 61.9 MiB | 4.09 GiB |
| Rehash/local metadata/publication | 4.40 s | 51.39 / 50.8 MiB | 4.29 GiB |

All jobs used unchanged `validate_sample.run_guarded`, one nice +10 CPU worker,
native threads 1, no GPU, 512 MiB RSS and 1.5 GiB available-system-memory reserve,
300 s max (tests 120 s), sequentially. The second download approached the time
budget; preserve resumable checkpoints rather than increasing limits for slower
future transfers. The acquisition/publication workers imported no NumPy,
rasterio, Torch or cloud-model library. Full tests retain their CPU Torch setup.

Actual free disk was 20.623 GiB before source 21, 20.066 after; 20.060 before
source 16, 19.513 after; 19.510 before/after publication. Earlier pre-implementation
capacity checks varied from 19.6 to 20.6 GiB due to other Mac activity. These are
measured snapshots, not guaranteed future capacity. Conservative download-job
allowance 2.070 GiB plus 2 GiB reserve and per-file/chunk reserve checks passed.
Publication uses same-filesystem renames and requires the 2 GiB reserve, without
allocating another full payload. Recheck live disk/memory before future work.

Final integrity passed: all 51 frozen input/code hashes; both historical download
snapshots and final persistent completion; all 116 final file stat/source-pin/
receipt joins; owned final directory inodes; four actual XML hashes; 369-test
receipt; prior 116 source stats, prior QA exclusions and V10 backups; zero active
workers/models/labels. Publisher performed full actual file MD5/SHA/size checks;
the final separate audit uses unchanged stats and receipt joins for JP2/GML,
not another expensive rehash or independent pixel/QA truth. Full stats/SHAs are
saved in `checks/final_integrity.json` for subsequent source fingerprint checks.

## Next bounded task

Predeclare a separate QA-only roster for **four distinct earlier-date source
pairs**: two fixed earlier pre-dates x two existing events, with fixed 31 July
post and original grids/group/split/anchor provenance. Derive explicit new IDs;
original 36UXA parents and V11's 26 July cloudy source pairs stay unchanged.
Recalculate cumulative post-minus-pre temporal targets from frozen catalogue
geometry; do NOT reuse V11's hardcoded dates, derivation or reference masks.
Validate full original train/test footprint barriers before any source pixels.

Then run unchanged automatic QA on the full fixed roster under the same Mac
limits, retaining every outcome and prior automatic-QA exposure. Do not rank
by model predictions, shrink ROIs/margins, relax QA, try extra dates until success
or approve labels. Valid controls/later persistence/manual class evidence,
full 27-channel proof and a label lock remain outstanding. No first-visible
interval or frozen-final unseen accuracy exists. Frozen model 1664 at 0.5 and
masks 0 forest/1 clearing/255 unknown remain unchanged. Retain all source data,
V10 backups and V14 transaction/receipts.

Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`;
Git root is its parent. Check branch/status, existing outputs and active jobs.
