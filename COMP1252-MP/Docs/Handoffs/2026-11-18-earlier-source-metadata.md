# Earlier source metadata continuation

## Current state: V13 complete; both metadata preflights pass, no imagery acquired

Continue on `codex/curated-deforestation-pilot`; update this handoff, commit and
push each verified logical change to the same GitHub branch. Read
`2026-11-18-earlier-preimage-catalogue.md` and its linked V10/V11/older handoffs
for provenance and frozen constraints. Entry point `CONTINUE.md` supports Codex,
Gemini Antigravity or another agent without chat history.

The approved next task is COMPLETE. **21 July and 16 July 2018 pass genuine L1C
product/tile XML and complete required-object listing checks.** Both have 58
pinned assets, with a combined payload of 1,050,527,541 bytes (1001.86 MiB).
This is remote metadata/asset evidence, NOT local QA readiness, clear pixels,
source approval or a forest-to-clearing label. No imagery/GML content was
fetched, SAFE created, source published, pixel/model run or label approved.
No worker is active. All 358 tests pass; working results must not be rerun.

Checkpoints: `643e792` declared the fixed roster before operational requests;
`8c13e74` saved the tested/frozen implementation before execution. Both are
pushed. The completion checkpoint contains raw evidence, integrity receipt and
this handoff. Prior V12 completion `eca2725` remains preserved.

## Fixed metadata roster and catalogue anomaly

Protocol: `Docs/Reviews/ValidationEarlierMetadataV13/protocol.json`.
Implementation: `Solutions/preflight_earlier_validation_metadata.py`.
Exactly two closest V12 nominal earlier N0206/R064/36UXB dates, descending date
then item ID; no cloud ranking, fallback or source substitution:

| Date | Scene | Required payload |
| --- | --- | ---: |
| 2018-07-21 | `S2A_MSIL1C_20180721T083601_N0206_R064_T36UXB_20180721T105541` | 559,954,114 bytes / 534.01 MiB |
| 2018-07-16 | `S2B_MSIL1C_20180716T084009_N0206_R064_T36UXB_20180716T122805` | 490,573,427 bytes / 467.85 MiB |

Both saved L1C STAC records have asset hrefs pointing to `sentinel-s2-l2a`.
**Do NOT fetch those hrefs as L1C source data.** V13 preserves every conflicting
href and explicitly uses `https://sentinel-s2-l1c.s3.amazonaws.com/` paths. Actual
provider product/tile XML agrees with the declared L1C identities and saved STAC
granule/datastrip/sensing metadata. The V12 nominal inventory never established
reader/source compatibility, so its raw evidence and unknown-readiness claims
remain intact; the href anomaly is a follow-up qualification, not a rewritten
catalogue. Do not silently swap in N0500 or L2A sources.

All other V12 variants and the known-cloudy 26 July source/V11 exclusions remain
preserved. No additional date was tried. Both events still share ONE geographic
group `group-96c27871459d03c08bcc`, not independent validation locations.

## Actual metadata and required objects

Exactly eight anonymous requests: two complete S3 listings, two product XML
HEADs and four actual XML GETs; 1,077,913 bytes (1.028 MiB) fetched. Every XML
GET matches pinned size, simple MD5 ETag, If-Match and Content-Length; SHA256,
raw bytes and request/response headers are saved. A strict allowlist rejects
image/GML GETs, L2A URLs, redirects and extra sources. No retries/paid access.
Caps remain 12 requests, 20 s/request, 2 MiB/listing, 5 MiB/XML, 32 MiB total.

Both sources have genuine `Level-1C_User_Product` / `Level-1C_Tile_ID` XML,
exact PRODUCT_URI/spacecraft, Level-1C, baseline 02.06, orbit 64, quantification 10000.
No RADIO_ADD_OFFSET elements are present in either actual N0206 XML; the frozen
reader's absent-list zero-offset convention is compatible. Catalogue offset
values were not used as SAFE XML/DN offsets. Product granule/datastrip identifiers
join exactly to tile XML and saved catalogue metadata. Compact granule folder
absolute orbit and sensing second agree; image stems use product acquisition time.
STAC/tile sensing timestamps agree; product datatake times differ by minutes as
expected and are retained separately, never forced to identical timestamps.

Actual grids are EPSG:32636, 10 m north-up, bounds
`[600000,5590200,709800,5700000]`, covering both unchanged original crops AND full
1200 m QA contexts with aligned pixels. Valid mean solar angles:
21 July zenith 32.6586829282836 / azimuth 153.527770413223 degrees;
16 July zenith 31.782994517838 / azimuth 153.148596708303 degrees.
No synthetic SAFE/JP2/GML placeholders were made to call `qa_scene_metadata`.

Each full manifest has **58 required assets**: 11 JP2 bands
B01/B02/B03/B04/B05/B08/B8A/B09/B10/B11/B12; 44 NODATA/SATURA/DEFECT/TECQUA
GMLs across those bands; cloud GML; product XML; tile XML. B06/B07 are not required
by the frozen union of feature/cloud bands. All 116 entries have actual measured
provider size and simple MD5 ETag, exact URL/key and XML-derived target paths.
Non-image metadata/GML sizes satisfy the frozen 5 MiB bound. Object presence and
pins come from complete listings; unread JP2/GML content is NOT verified and
individual image/GML HTTP HEAD/readability was NOT tested. Future transfers must
check If-Match/size/MD5 and local frozen QA prerequisites before publication.

## Mac capacity, disk and validation

Measured source filesystem free space: 25,820,127,232 bytes before,
25,819,086,848 bytes (24.046 GiB) after metadata evidence collection. Combined
future payload 1,050,527,541 bytes; largest asset 120,797,363 bytes. Conservative
allowance is TWO full payload copies plus largest single-asset temporary and
1,086,316 bytes of current metadata artifacts: **2,222,938,761 bytes / 2.070 GiB**,
plus a separate 2 GiB free reserve. This passed; no reuse credit or hardcoded
source sizes were used. Measure free space again before any future download.
The allowance is a conservative future staging/publication budget, not actual
imagery space consumed during V13.

| Completed job | Elapsed | RSS high-water / sampled peak | Minimum available |
| --- | ---: | ---: | ---: |
| Full 358-test suite | 4.41 s | 361.52 / 378.6 MiB | 3.95 GiB |
| Live metadata preflight | 2.06 s | 45.56 / 43.2 MiB | 3.91 GiB |

Both use unchanged `validate_sample.run_guarded`, one nice +10 CPU worker,
native threads 1, no GPU, 512 MiB RSS, 1.5 GiB available-system-memory reserve,
300 s max (tests 120 s). No jobs overlapped; the metadata worker imported no
raster/pixel/model library. The full suite retains its existing CPU Torch setup.
Nine new tests cover actual XML identity/grid/solar/granule/image-path conflicts,
nonzero/nonfinite offsets, missing/unsupported quality objects, exact 58-asset
manifest, complete listing refusal, metadata-only allowlist, transfer pin/byte/
request failures, disk arithmetic and first-source timeout retention without
fallback. Read-only review findings were fixed before the passing suite.
No implementation changed after tests or operational execution.

## Durable evidence and verification

`Docs/Reviews/ValidationEarlierMetadataV13/` contains:
`protocol.json`, `preflight/{preflight.json,metadata.json,requests.json,report.md}`,
`preflight/source-{01,02}/{listing.xml,product.xml,tile.xml}`,
`checks/pre_execution_tests/` and `checks/final_integrity.json`.
Guide: `Docs/Guides/EarlierSourceMetadata.md`.
Completed ignored local outputs: `Datasets/ValidationPilot/v13/preflight/` and
`v13/checks/pre_execution_tests/`. Never overwrite or repeat them due to a chat/
agent change. Bounded evidence is byte-identical in Docs and backed up on GitHub;
raw imagery/full caches remain local and were not expanded in V13.

Final integrity passed: all 50 frozen input/code hashes; all four raw XML hashes,
MD5/header/pin checks; both complete listings and all 116 manifest size/ETag joins;
fixed roster/order, independent identity/orbit/scale and disk arithmetic; exact
Docs copy; 358 tests; 0 active workers/imagery/GML downloads/publications/labels.
Prior 116 V11 source size/mtime receipts, both excluded outcomes and complete V10
transaction remain unchanged. The final check reuses saved grid coverage booleans
and prior source stats; it does not claim independent raster/QA truth, unread
payload integrity or future free-space guarantees. Retain V10 journal/backups.

## Next bounded task

Predeclare acquisition of exactly these two source manifests using the V13
pins. Build/test a separate strict acquisition path; do not rerun archived V8/V9
ingestion or V10's hardcoded old-source repair worker. Stream downloads with
If-Match/exact size/MD5/SHA checks, at most one low-priority worker; use bounded
scene-sized jobs, explicit recoverable state and fresh disk/memory checks. Stage
outside Sentinel-2 discovery, validate all 58 actual files per scene with unchanged
`qa_scene_metadata`, then publish using a recoverable transaction. Preserve all
old sources/failures and both new outcomes. No pixel/model execution is needed
for acquisition/local metadata readiness.

Later earlier-pre-date pairs need distinct explicit lineage and recalculated
cumulative post-minus-pre targets; V11's hardcoded dates/masks cannot be reused.
Keep original grids, QA, frozen model 1664 at 0.5, all split/exposed-group barriers,
0 forest/1 clearing/255 unknown. Full 27-channel proof, cloud QA, later persistence,
controls and class review remain outstanding. No validation dataset, label lock,
first-visible interval or frozen-final unseen accuracy exists yet.

Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`;
Git root is its parent. Check branch/status, existing outputs and active jobs.
