# Earlier validation sources: metadata inventory only

V12 found five nominal earlier pre-image dates for the fixed 31 July 2018 post:
1, 6, 11, 16 and 21 July. Their clarity and actual reader compatibility are unknown.
This is a metadata follow-up pool; no acquisition roster or source approval exists.

Start with `Docs/Handoffs/2026-11-18-earlier-preimage-catalogue.md`. The protocol
was committed before querying and the separate inventory implementation was
tested/frozen before execution. Archived V7 summaries and ingestion must not be
rerun: the 18 November review qualified their mixed-baseline/partial-QA claims.

`Solutions/inventory_earlier_validation_sources.py` queries Earth Search's
[`sentinel-2-l1c` collection](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l1c)
within the full original QA context. It saves raw pages/requests and retains
all variants. Identity/date comparisons, full geometry with holes and bounded
pagination precede nominal follow-up eligibility; cloud percentage never ranks.
Offset values stay provider evidence and reader compatibility stays unverified.

The completed query returned 18 products, two pages including the empty terminal
page, and 191,206 response bytes. 12 N0206 and 6 N0500 products remain distinct.
Both 26 July variants remain visible, with only the exact V11 source carrying
the measured QA-excluded status. This does not approve its reprocessing.

Completed output `Datasets/ValidationPilot/v12/inventory/` has `metadata.json`;
the guard rejects overwrite. Its exact bounded evidence copy is under
`Docs/Reviews/ValidationEarlierCatalogueV12/inventory/`. The final integrity
receipt and 349-test receipt are in that review's `checks/`. No imagery was
downloaded and no pixel/QA/forest model ran. One guarded nice +10 CPU worker used
about 60 MiB in 3.81 s, preserving at least 2.70 GiB available system memory.

Next, predeclare a small metadata/compatibility roster and check actual XML and
all 58 required source assets per scene with measured sizes/disk budget. Preserve
the cloudy pair, all frozen grids/QA/splits and one geographic-group identity.
Earlier dates require new temporal-label lineage before any later pixel work.
