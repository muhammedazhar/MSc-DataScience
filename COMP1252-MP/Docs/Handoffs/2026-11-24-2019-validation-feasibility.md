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
matches (3 and 8 March) among three products, including the original excluded
13 March product. May has two nominal matches (7 and 12 May) among ten
products. The other May items have a different baseline, orbit or tile.
All matches cover the entire crop and QA margin according to the catalogue.
The four raw pages and their normalized outputs replayed byte for byte without
network calls. Worker RSS peaked at 120.48 MiB and available memory stayed at
or above 3.40 GiB. This checkpoint leaves independent reference checks pending.
