# 2019 validation site source feasibility

V25 checks the two remaining 2019 groups retained by V23. Continue on
`codex/curated-deforestation-pilot` and push verified checkpoints to that branch.
The original crop grids, post observations, positive support and geographic
split barriers stay fixed. Past failed dates remain in the record.

| Site | Annotation image date | Fixed post-image | Earlier Sentinel-2 search | Independent reference window |
| --- | --- | --- | --- | --- |
| March | 28 March 2019 | 2 April 2019 | 3–17 March | 3 March–12 May |
| May | 27 May 2019 | 6 June 2019 | 7–16 May | 7 May–11 July |

Use 36UYA/R064/N0207 metadata, original 5–30-day pair intervals and full
1,200 m QA source padding. Preserve every Earth Search variant, including
variants for 13 March, which already failed the local crop and target checks.
Reference searches use the exact 640 m crop. The Airbus public Pléiades and
SPOT queries use acquisition dates. Esri Wayback uses the first available
release on or after each of the three reference dates (start, fixed post and
end), with source capture dates reported separately. Access to a scene and
clarity over the crop require separate verification.

## Scope checkpoint

Checkpoint `0851a9b` pins the site identities, query rules, prior evidence and
implementations in the [protocol](../Reviews/Validation2019SourcesV25/protocol.json).
Keep one nice +10 CPU worker, one native thread, a 512 MiB RSS cap, a 1.5 GiB
available memory reserve and a 300 second runtime cap. Run one job at a time
through the existing guard.

## Sentinel-2 inventory

The [inventory](../Reviews/Validation2019SourcesV25/sentinel/report.md) retained
all 13 Earth Search products across four raw pages. March has two nominal
matches (3 and 8 March). The third product is the original excluded
13 March observation. May has two nominal matches (7 and 12 May) among ten
products. The other May items have a different baseline, orbit or tile.
All matches cover the entire crop and QA margin according to the catalogue.
The four raw pages and their normalized outputs replayed byte for byte without
network calls. Worker RSS peaked at 120.48 MiB and available memory stayed at
or above 3.40 GiB on this first Sentinel-2 run. Checkpoint `b064891` preserves
those results. A new guarded run bound the evidence to the corrected reference
protocol. Its raw responses and normalized results match the original across
both sites. Offline replay passed again.

## Reference configuration correction

The first guarded reference run returned four empty Airbus layer results. The
Wayback configuration request succeeded but the parser expected a separate
release date field. The live config gives each release date in its item title.
The [initial attempt](../Reviews/Validation2019SourcesV25/reference_initial_attempt/inventory/references.json)
and exact replay preserve that outcome. Checkpoint `be380b5` pins the correction.
The parser validated all 199 archived release entries and selected the six
expected crop and release combinations offline. The reference dates and
coverage rules are unchanged.

## Independent reference results

| Site | Airbus Pléiades and SPOT | Wayback layer 6 | In-window capture evidence |
| --- | --- | --- | --- |
| March | Both layers returned zero archive scenes intersecting the crop in the window | 13 March, 3 April and 15 May releases returned zero full-crop metadata records | None found |
| May | Both layers returned zero archive scenes intersecting the crop in the window | 15 May, 26 June and 17 July releases all list a GeoEye-1 capture on 26 September 2017 at 0.46 m sampling | None found. The older capture could provide historical context |

All 11 unique requests returned complete results across the corrected run and
one bounded retry. The retry recovered three timeouts. One previously successful
request timed out on retry. Both archives preserve the exact successes and
failures and both replayed byte for byte with zero network calls. Wayback
reports coverage through its `Within` query. The search covers the specified
layers and releases, with other archives and versions outside its scope.
[Reference run](../Reviews/Validation2019SourcesV25/references/references.json)
and [retry](../Reviews/Validation2019SourcesV25/reference_retry/references.json).

## State and next decision

V25 is complete. Both sites have two earlier Sentinel-2 products that meet the
catalogue rules. The checked independent archives supplied zero captures in
either reference window that cover the full crop. Any further comparison first
needs native L1C metadata and asset-list checks. Crop clarity still needs local
QA. A new acquisition roster, visual review, approved validation labels and a
first-visible interval remain pending. Preserve the seven previously exposed
groups and V24’s four automatic exclusions.

Guarded runs used one low-priority CPU worker with one native thread. The peak
worker RSS was 120.48 MiB and available memory stayed at or above 3.15 GiB.
[Integrity receipt](../Reviews/Validation2019SourcesV25/checks/final_integrity.json).
