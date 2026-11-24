# 2019 Sentinel-2 native metadata preflight

V26 checks all four candidates from the
[completed V25 handoff](2026-11-24-2019-validation-feasibility.md).
Continue on `codex/curated-deforestation-pilot`. Commit verified checkpoints
with the current handoff and push them to GitHub on the same branch.

| Site | Candidate dates in 2019 | Fixed post date | Comparison intervals |
| --- | --- | --- | --- |
| March | 3 and 8 March | 2 April | 30 and 25 days |
| May | 7 and 12 May | 6 June | 30 and 25 days |

Use exact 36UYA/R064/N0207 products and sequence 0. Preserve the original
64 × 64 crops and their 1,200 m QA margins. Derive explicit Level-1C provider
keys from the pinned product identities. Three of the V25 L1C records carry
Level-2A asset links. Preserve those catalogue links as evidence and read the
native product and tile XML from the L1C archive.

The [protocol](../Reviews/Validation2019PreflightV26/protocol.json) allows one
complete listing, one product XML HEAD and two XML GETs per candidate. Check
native product identity, radiometry, tile grid alignment, solar angles and
coverage. Each candidate needs 58 pinned objects for the frozen QA inputs.
Use one nice +10 CPU worker, one native thread, a 512 MiB RSS limit, a 1.5 GiB
available memory reserve and a 300 second runtime cap. The metadata requests
are anonymous and capped at 16 per run. Recheck disk for any future imagery
phase using the measured payload and a 2 GiB reserve.

Checkpoint `2797e1c` pins the scope and first worker. Offline checks verified
all raw catalogue joins, all 16 allowlisted requests, rejection of an imagery
request and parser compatibility with a local N0207 XML pair.

The [initial archive run](../Reviews/Validation2019PreflightV26/initial_attempt/preflight/preflight.json)
verified all 16 requests and saved 2,461,248 response bytes. Its strict
timestamp comparison blocked all four candidates. One native time had a
five-digit fractional second unsupported by the environment's Python parser.
The native and catalogue times also differ, by under a millisecond in May
and by 27.636840 and 622.146231 seconds on 3 and 8 March. The product URI,
granule, datastrip, generation and UTC acquisition date agree in every case.
The initial reports and raw replies replayed exactly under the first protocol.

The revised parser accepts the native timestamp precision and uses the UTC
date verified by the exact native identities. It records both time values and
their signed difference. Source bytes and crop rules are preserved. The
offline smoke identified all 58 required objects for each archived candidate
and rejected a changed UTC date or granule ID. Checkpoint `a7c8c70` pins that
correction and the initial evidence.

## Completed preflight

| Candidate | Required files listed | Listed payload |
| --- | ---: | ---: |
| 3 March 2019 | 58 | 679,536,761 bytes |
| 8 March 2019 | 58 | 610,837,722 bytes |
| 7 May 2019 | 58 | 347,466,958 bytes |
| 12 May 2019 | 58 | 570,786,606 bytes |
| Total | 232 | 2,208,628,047 bytes |

All four native L1C XML pairs confirm the required 02.07 baseline, orbit 64,
36UYA tile, quantification 10,000 and compatible absent offset lists. Each
EPSG:32636 native 10 m grid is aligned with its assigned crop and covers its
full QA margin. Solar angles pass. Each complete listing pins 11 JP2 bands,
44 band-quality masks, one cloud mask and two XML files. Only the XML file
contents were verified against the provider MD5 pins. The image and GML files
remain listed assets pending acquisition and content verification.

The guarded recheck used the 16 pinned initial receipts with zero new network
requests. Final replay reproduced the corrected metadata, all listing and XML
bytes and the manifest byte for byte. The volume had 69.28 GiB free at the
live check. The two-copy staging estimate is 4.24 GiB including temporary
space, plus the 2 GiB reserve. Peak worker RSS across V26 was 127.39 MiB.
The lowest sampled available memory was 3.38 GiB. The four guarded jobs ran
sequentially. [Preflight](../Reviews/Validation2019PreflightV26/preflight/preflight.json)
· [Replay](../Reviews/Validation2019PreflightV26/verification/verification.json)
· [Integrity](../Reviews/Validation2019PreflightV26/final_integrity.json).

The acquisition roster remains open. The next bounded phase can choose an
explicit roster from these verified manifests, recheck free disk and stage
one scene at a time. Preserve the fixed post images, the three old cloudy
dates, the seven exposed groups and the independent reference results. Local
cloud screening, visual class checks and first-visible timing are pending.
