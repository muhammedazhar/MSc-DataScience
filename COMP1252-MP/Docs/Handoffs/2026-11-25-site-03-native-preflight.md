# Site-03 native Level-1C preflight

V29 checks three exact observations from the completed V28 catalogue.
Continue on `codex/curated-deforestation-pilot` and push each verified
checkpoint to the same branch. Preserve the original crop and proposed
control at site-03 and keep all other V28 candidates in the archive.

| Date | Exact catalogue item | Role |
| --- | --- | --- |
| 10 May 2018 | `S2B_36UXA_20180510_0_L1C` | Earlier control observation |
| 30 May 2018 | `S2B_36UXA_20180530_0_L1C` | Before observation |
| 4 June 2018 | `S2A_36UXA_20180604_0_L1C` | After observation |

The [protocol](../Reviews/ValidationSite03PreflightV29/protocol.json) pins the
exact raw catalogue records, the crop and the unchanged downloader. The
[imported plan](../Reviews/ValidationSite03PreflightV29/catalogue_import/plan.json)
binds each of the three items to both the 64 × 64 crop and its 1,200 m QA source
margin. The raw catalogue items retain their Level-2A links. Native Level-1C
keys are derived from the compact 36UXA/R107/N0206 product identities.

Run `Solutions/download.py preflight` once per exact scene, with the imported
plan and a separate output directory. Each run may list the tile and read
its product and tile XML, using four metadata requests and one low-priority
worker. Keep one native thread, a 512 MiB RSS cap, a 1.5 GiB available-memory
reserve and the 300 second runtime cap. Check the projected native 10 m grid
against the exact crop and QA margin and verify the 58-file manifest.

Checkpoint `0e0326a` pins the local catalogue import before provider requests.
The import passed identity and footprint checks.

## Completed preflight

| Date | Required files listed | Exact listed bytes | Peak worker RSS |
| --- | ---: | ---: | ---: |
| 10 May 2018 | 58 | 511,349,457 | 63.28 MiB |
| 30 May 2018 | 58 | 526,754,551 | 65.09 MiB |
| 4 June 2018 | 58 | 524,534,456 | 64.64 MiB |
| Total | 174 | 1,562,638,464 | |

All three native XML pairs confirm tile 36UXA, orbit 107 and baseline 02.06.
The native 10 m grid aligns with and covers both the exact EPSG:32636 crop and
its full QA margin. The manifests pin 11 JP2 bands, 44 per-band quality GMLs,
one cloud GML and two XML files for each scene. The 12 native metadata requests
transferred 1,616,035 response bytes. XML contents passed the provider size,
ETag and MD5 checks. Exact replay reproduced every report artifact and all
174 file pins without network access. [Verification and byte counts](../Reviews/ValidationSite03PreflightV29/verification.json).

The volume had 60.63 GiB free at verification. The conservative staging
estimate is 3.03 GiB plus the 2 GiB reserve. The three metadata workers ran
sequentially and sampled available memory stayed at or above 3.88 GiB. The full scene files
remain pending acquisition. The preflight tells us the inputs are available
to attempt. Crop clarity, forest class and timing still need evidence.

The next phase can use the archived per-date preflight directories with
`Solutions/download.py download`, recheck capacity, and stage one scene at
a time. Screen the frozen 30 May and 4 June comparison and review the proposed
control across the 10 May, 30 May and 4 June observations before label approval.
Any later persistence observation is a separate catalogue option in V28.
