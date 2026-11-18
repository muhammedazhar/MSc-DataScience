# Earlier pre-image catalogue continuation

## Current checkpoint: implementation tested and frozen; query pending

Continue on `codex/curated-deforestation-pilot`; commit each verified logical
change with this handoff updated and push to the same GitHub branch. Read
`2026-11-18-validation-source-review.md` for the completed Antigravity review,
V10 source repair, V11 exclusion and linked prior provenance/frozen constraints.
The latest completed checkpoint before V12 is `b858b47`.

The user approved continuing the next bounded task. V12 is a METADATA-ONLY
catalogue inventory of earlier pre-images for the fixed 31 July 2018 post-image.
Protocol: `Docs/Reviews/ValidationEarlierCatalogueV12/protocol.json`. No query
has run, roster been selected, source imagery downloaded or pixel work started
at this checkpoint. Implementation and offline tests are complete; live query
and final integrity/evidence remain INCOMPLETE.

## Fixed scope

Query Earth Search's `sentinel-2-l1c` collection for 1 July inclusive through
31 July exclusive, using the full QA context union in EPSG:32636 transformed to
a geographic bounding box. Retain all returned variants, including other
tiles/orbits/baselines and the known-cloudy 26 July acquisition. Never summarize
a mixed catalogue using the first row. Bound to 10 pages, 100 items per requested
page, 2 MiB per response, 20 seconds per request and 300 seconds for the worker.
Save raw pages and request specifications, require complete pagination and fail
on conflicting identities or loops. Exact duplicates may coalesce only with
identical canonical content and retained occurrence provenance.

Parse each product URI's tile/orbit/baseline/date and compare available provider
metadata. Missing/conflicting identity or geometry stays explicit. Use complete
valid Polygon/MultiPolygon footprints, including holes; catalogue coverage is
not actual valid-pixel coverage. Offsets remain unknown without supplied evidence.
Nominal N0206 metadata follow-up candidates require tile36UXB, orbitR064, full
fixed footprint coverage and dates earlier than 26 July within the frozen
5–30-day forward interval. This is neither a source approval nor a selected
acquisition/review roster. Catalogue cloud percentages never select or rank.

Both events remain ONE geographic group `group-96c27871459d03c08bcc`; original
grids, split, targets, QA, labels and model are frozen. V11's 26 July product
remains QA-excluded; reprocessed same-date variants have untested own QA and are
not fresh earlier sources. No validation labels, date estimates or accuracy exist.

## Capacity and verification

Use unchanged `validate_sample.run_guarded`, one nice +10 CPU worker, native
threads 1, no GPU, 512 MiB RSS, 1.5 GiB available-system-memory reserve and 300 s
maximum (tests 120 s). Do not overlap guarded workers. Source rasters, cached
observations, models and QA are outside V12. Do not overwrite completed outputs.
349 tests pass (341 previous plus eight new offline catalogue tests). The suite
took 18.83 s, peaked at 320.53 MiB and retained at least 2.53 GiB available. An
initial read-only process check found no active guarded project worker.

Protocol checkpoint `1da24ee` is pushed. The implementation and test hashes are
frozen in the protocol, alongside the passing suite receipt. Read-only review
checked variants, pagination, holes, timestamp date semantics and unknown
offsets; asset-band offsets and malformed URI handling were corrected before
tests. The new implementation does not modify any archived V7 or frozen QA code.

Next: run one bounded metadata query using the frozen inventory. Record all outcomes, resource measurements and source integrity
before marking V12 complete. Later work requires a separately predeclared small
metadata/compatibility roster and acquisition scope before any imagery/QA.
Earlier pre-dates will also require explicit new pair lineage and recalculated
cumulative temporal targets; never reuse V11's hardcoded dates or masks.

Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`;
Git root is its parent. Completed outputs and raw source data stay local unless
bounded evidence is explicitly copied into Docs and committed.
