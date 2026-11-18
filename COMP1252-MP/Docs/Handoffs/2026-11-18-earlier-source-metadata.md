# Earlier source metadata continuation

## Current checkpoint: V13 tested and frozen; live preflight pending

Continue on `codex/curated-deforestation-pilot`; update this handoff, commit and
push each completed verified logical change to the same branch. Read
`2026-11-18-earlier-preimage-catalogue.md` and its linked V10/V11/older handoffs
for provenance and frozen constraints. Last completed checkpoint is `eca2725`.

The user approved the next task: actual source metadata, full required asset
availability and measured download/disk budgets before imagery acquisition.
Protocol: `Docs/Reviews/ValidationEarlierMetadataV13/protocol.json`.

The fixed metadata roster is **21 July then 16 July 2018**, exactly the two
closest earlier V12 nominal N0206/R064/36UXB dates, ordered only by date/ID.
No cloud ranking, fallback or substitutions. Other V12 dates/products, the known
cloudy 26 July source and its excluded V11 outcome stay preserved.

Read-only inspection found the two saved L1C catalogue records link to a
`sentinel-s2-l2a` bucket. Those hrefs must NOT become L1C source evidence. V13
explicitly uses declared `sentinel-s2-l1c.s3.amazonaws.com` paths, cross-checked
against genuine product and tile XML. Preserve this catalogue anomaly.

Only two bounded S3 listings, two product XML HEADs and four actual XML GETs
are planned. Maximum 12 requests, no redirects/retries/paid access, 20 s/request,
2 MiB/listing, 5 MiB/XML, 32 MiB total and 300 s/worker. Listings must be complete.
Verify XML sizes, simple MD5 ETags, If-Match, Content-Length and saved SHA256.
Pin all 58 required assets per source: 11 JP2, 44 quality GML, cloud GML, 2 XML.
No image/GML content download, skeleton/placeholder SAFE or publication.

Reuse frozen pure XML grid/solar parsing helpers only. Actual identity, orbit,
baseline, scale/offsets, granule joins, sensing dates, full original crop/context
coverage and solar angles must agree. A remote metadata pass does not assert
local `qa_scene_metadata=True`, unread payload integrity, clear pixels or labels.
Disk preflight measures current free bytes and conservatively allows two full
payload copies plus the largest temporary asset, metadata bytes and 2 GiB reserve.

Use unchanged `validate_sample.run_guarded`, one nice +10 CPU worker, native
threads 1, no GPU, 512 MiB RSS, 1.5 GiB available-system-memory reserve and 300 s
maximum (tests 120 s). Never overlap guarded workers or overwrite completed output.
358 tests pass (349 previous plus nine new preflight tests), 4.41 s guarded time,
361.52 MiB RSS high-water / 378.6 MiB monitor peak, minimum 3.95 GiB available.
No operational V13 requests have run yet. Live preflight and final integrity/
evidence remain INCOMPLETE at this checkpoint. Protocol checkpoint `643e792`
is pushed. Implementation/test hashes and passing-suite receipt are frozen.

Read-only review identified per-source timeout preservation and strict image/
granule path joins. Both were fixed before tests; original frozen helpers remain
unchanged. Network allowlist, source pin failures, missing quality GMLs, complete
58-asset counts, grid/solar/identity conflicts and disk budget bounds are tested.

Next: execute the frozen metadata preflight once, preserve both outcomes and
verify raw evidence.
No earlier-pre-date derived pairs, recalculated labels, acquisition, pixel work,
predictions or training are part of this phase. Both events are still ONE group
`group-96c27871459d03c08bcc`. Preserve QA/model/splits and 0/1/255 masks.

Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`;
Git root is its parent. V10 transaction/backups must be retained.
