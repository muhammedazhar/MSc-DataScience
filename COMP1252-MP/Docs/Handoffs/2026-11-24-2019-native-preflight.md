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

The scope and worker are pinned. Offline checks verified all raw catalogue
joins, all 16 allowlisted requests, rejection of an imagery request and parser
compatibility with a local N0207 XML pair. The live requests and offline
replay are pending. The acquisition roster remains open.
