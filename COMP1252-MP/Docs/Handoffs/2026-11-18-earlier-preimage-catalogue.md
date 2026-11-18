# Earlier pre-image catalogue continuation

## Current state: V12 metadata inventory complete; no roster selected

Continue on `codex/curated-deforestation-pilot`; commit each verified logical
change with this handoff updated and push to the same GitHub branch. Read
`2026-11-18-validation-source-review.md` for the completed Antigravity review,
V10 repair, V11 exclusion and linked earlier handoffs/frozen constraints.

The user approved continuing the next bounded task. V12 is COMPLETE: a
metadata-only catalogue inventory of earlier pre-images for the fixed
31 July 2018 post-image. It found five nominal metadata follow-up dates:
**1, 6, 11, 16 and 21 July 2018**. Their pixels and QA are unexamined. No source
imagery was downloaded, roster selected, source approved, pixel/model work run,
label approved or first-visible interval/accuracy estimated. No worker is active.

Checkpoints: `1da24ee` declared the protocol before execution; `61fd01f` saved the
tested implementation and frozen hashes before the live query. Both are pushed.
The completion checkpoint contains the raw responses, normalized inventory,
integrity receipt and this handoff. V11 checkpoint `b858b47` remains unchanged.

## Query and retained evidence

Protocol: `Docs/Reviews/ValidationEarlierCatalogueV12/protocol.json`.
Implementation: `Solutions/inventory_earlier_validation_sources.py`.
Query: Earth Search `sentinel-2-l1c`, 1 July inclusive through 31 July exclusive,
full QA context union `[662180,5603540,665480,5606840]` in EPSG:32636 transformed
to geographic bbox. No tile/orbit/cloud filter was applied in the request;
all returned products are retained. Requested page limit 100; maximum 10 pages,
2 MiB/response, 20 s/request and 300 s/worker. No image asset URL was fetched.

The provider returned **18 distinct items/products across 12 dates**: 12 N0206
and 6 N0500; 11 R064 and 7 R107, all tile 36UXB. Page 1 returned 18 items and a
next link; page 2 was the required empty terminal page. Provider matched count
18 agrees; raw responses total 191,206 bytes. Raw pages, exact POST bodies,
next-link semantics, hashes and item occurrences are preserved. There were no
duplicate identities or metadata issues. Do not infer missing N0500 variants
from the presence of other reprocessed dates; this is the returned snapshot.

Every returned footprint covers the fixed core and full QA context according
to complete valid Polygon/MultiPolygon geometry checks, including holes. This
does NOT establish actual tile-XML coverage, valid pixels or clear QA. Product
URI acquisition time and STAC tile sensing time differ by minutes; both are
retained and UTC dates agree. Tile/orbit/baseline are parsed per product, not
hardcoded from the first result. Optional MGRS triplets and provider fields are
cross-checked when present.

## Nominal metadata follow-up pool, not an acquisition roster

The predeclared rule requires 36UXB/R064/N0206, dates before the known-cloudy
26 July acquisition, the frozen 5–30-day forward interval and full catalogue
footprint coverage without identity issues. The five resulting products are:

| Date | Days before 31 July | Product URI without `.SAFE` |
| --- | ---: | --- |
| 2018-07-01 | 30 | `S2A_MSIL1C_20180701T083601_N0206_R064_T36UXB_20180701T104347` |
| 2018-07-06 | 25 | `S2B_MSIL1C_20180706T083559_N0206_R064_T36UXB_20180706T123246` |
| 2018-07-11 | 20 | `S2A_MSIL1C_20180711T083601_N0206_R064_T36UXB_20180711T122613` |
| 2018-07-16 | 15 | `S2B_MSIL1C_20180716T084009_N0206_R064_T36UXB_20180716T122805` |
| 2018-07-21 | 10 | `S2A_MSIL1C_20180721T083601_N0206_R064_T36UXB_20180721T105541` |

All other variants remain in `inventory.json` with explicit follow-up blockers.
Catalogue cloud percentages are retained but never used for selection/ranking.
The exact V11 26 July N0206 product remains QA-excluded. Its N0500 reprocessing
is retained with its OWN QA unexamined and is not a fresh earlier source.
Both 26 July products remain recorded; no failed result was erased.

Provider-supplied property/asset-band offsets are captured verbatim, including
their band indices. Absence would remain null/unknown. Raw asset metadata also
contains scales; catalogue offsets must be read with those scales and are not
proof of SAFE XML offsets, quantification, full file availability or reader
compatibility. All rows remain `reader_compatibility=unverified`,
`source_approved=False`, `roster_selected=False`.

## Verification and Mac limits

**349 tests pass** (341 previous plus eight offline inventory regressions).
They cover mixed baselines/order, variants versus exact/conflicting duplicates,
unknown/supplied offsets, malformed identities, date boundaries and known-cloudy
status, full geometry with holes/MultiPolygon, pagination merge/GET/trusted URLs,
matched counts, loops/page/byte bounds and malformed links. Read-only agent review
identified asset-band offset capture and malformed URI summary handling; both
were fixed before the passing suite and frozen operational implementation.

| Completed job | Elapsed | Peak RSS | Minimum available |
| --- | ---: | ---: | ---: |
| Full 349-test suite | 18.83 s | 320.53 MiB | 2.53 GiB |
| Live metadata inventory | 3.81 s | ~60 MiB | 2.70 GiB |

Inventory RSS high-water was 58.66 MiB; monitor peak 59.7 MiB. Both jobs used the
unchanged `validate_sample.run_guarded`: one nice +10 CPU worker, native threads 1,
no GPU, 512 MiB RSS, 1.5 GiB available-system-memory reserve, 300 s max (tests 120 s).
No guarded workers overlapped. No raster/pixel/model library was imported by
the inventory worker; full tests retain their existing CPU-only Torch setup.

Final receipt `Docs/Reviews/ValidationEarlierCatalogueV12/checks/final_integrity.json`
checks raw item/product joins, hashes, both requests, terminal pagination, matched
counts, five-date eligibility independently from simple URI/date fields, all 47
frozen input/code hashes and byte-identical Docs backup. It checks unchanged
size/mtime for all 116 prior V11 source files, both preserved excluded outcomes
and complete V10 transaction state. It does not independently re-evaluate
geometry/QA truth or decode/rehash source rasters; source hashes were verified
in V10/V11 and their stats are unchanged. All approval/download/model counters
are zero, and active guarded worker inventory is empty.

Evidence under `Docs/Reviews/ValidationEarlierCatalogueV12/`:
`protocol.json`, `inventory/{inventory.json,metadata.json,report.md,raw/}`,
`checks/pre_execution_tests/` and `checks/final_integrity.json`.
Guide: `Docs/Guides/EarlierValidationSources.md`.
Ignored completed local output: `Datasets/ValidationPilot/v12/inventory/` and
`v12/checks/pre_execution_tests/`; never overwrite or rerun these just because
an agent/chat changed. Raw imagery and the full source/cache collection remain
local; code and bounded metadata evidence are backed up on GitHub.

## Next bounded task

Predeclare a small fixed roster from these five dates for METADATA and source
compatibility checks before imagery acquisition. Verify actual provider product/
tile XML, orbit/grid/quantification/offsets, complete 58-file frozen QA asset
availability and measured byte/disk budgets. No catalogue cloud ranking or pixel
search until success. Keep the current cloudy pair and all returned variants.

Any later earlier-pre-date pair needs explicit new lineage and recalculated
cumulative post-minus-pre targets; V11's hardcoded date derivation/masks cannot
be reused. Keep exact original grids, source failures, QA policy, model 1664 at 0.5,
all split/exposed-group barriers and masks 0 forest/1 clearing/255 unknown.
Two events remain ONE geographic group `group-96c27871459d03c08bcc`; this is not
independent-location validation. Full 27-channel proof, later persistence and
valid controls/class review are still outstanding. No validation dataset/label
lock/frozen-final evaluation exists yet. V10 backups and journal must be retained.

Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`;
Git root is its parent. Check branch/status and active jobs before continuing.
