# Fresh held-out site feasibility

V28 is a metadata feasibility check on `codex/curated-deforestation-pilot`.
Preserve the V27 results, all previous dispositions and the frozen training
checkpoint. The prior review used all four supported fresh groups with clear
original post crops. V28 starts from original temporal annotations in frozen
validation groups absent from the original pair inventory and prior review or
follow-up groups.

The [protocol](../Reviews/ValidationUnseenSitesV28/protocol.json) pins the
evidence and methods. The [site plan](../Reviews/ValidationUnseenSitesV28/plan/plan.json)
retains dispositions for all 514 validation events. Of 111 newly considered
temporal events in 27 groups, 37 in 17 groups passed the derived geometry gates.
The plan ranks eligible events by control geometry availability, eroded target
support and stable identifiers. It selects three different groups before live
catalogue requests. Native alignment, 1,200 m QA source coverage and frozen
train/test footprint isolation are required.

| Site | Geographic group | Annotation image date | Eroded target pixels | Proposed control |
| --- | --- | --- | ---: | --- |
| site-01 | `group-c4f441ae55f683f915fa` | 27 January 2018 | 151 | 16 × 16 pixels |
| site-02 | `group-d75dc0544bf1a0e519ad` | 28 November 2017 | 131 | 16 × 16 pixels |
| site-03 | `group-a85fa2022a25e1a72e2c` | 1 June 2018 | 115 | 16 × 16 pixels |

All use original tile 36UXA and the 64 × 64 native 10 m grid convention.
The proposed controls are unlabelled and belong to their parent groups.
Each has a full two-pixel context and avoids the buffered annotation catalogue.
`post_date` in the plan anchors Wayback release sampling to the annotation
image date. Selection of actual post-observations is part of the catalogue
assessment.

Search Earth Search L1C across the 30 days before through 45 days after each
annotation. Logical before and after options use the same tile, orbit and
baseline, cover the complete QA margin and span 5–30 days. The after image
must fall on or within ten days after the annotation image date. Archive
Airbus, Wayback and Landsat 8 reference metadata over each exact crop.
Use the actual source capture date and keep any access or resolution limits.
Each phase uses one low-priority CPU worker, one native thread, a 512 MiB
RSS cap, 1.5 GiB available-memory reserve and the 300 second runtime limit.
Planning passed at 162.47 MiB peak RSS. The second guarded run reproduced the
plan byte for byte. Checkpoint `e11ce88` froze the plan before live requests.

## Sentinel-2 metadata

| Site | Catalogue records | Pair options | Options with an earlier control-pair date | First-ranked dates |
| --- | ---: | ---: | ---: | --- |
| site-01 | 51 | 4 | 2 | 12 January–6 February 2018 |
| site-02 | 10 | 2 | 0 | 16 November–6 December 2017 |
| site-03 | 42 | 18 | 16 | 2 May–1 June 2018 |

Every pair option has a compatible later catalogue observation within the
search window. Site-03 also has a five-day option from 30 May to 4 June 2018
on orbit 107, with an earlier 10 May observation for a possible 20-day
stable-only control pair ending 30 May. Scene cloud metadata is 0% and
0.3764% for the five-day pair and 0% for 10 May.
That figure describes each scene. Whole-crop and context clarity require QA.
All 103 catalogue items across seven raw pages replayed byte for byte. The
Sentinel-2 run and replay peaked at 104.94 and 93.84 MiB.
[Catalogue evidence](../Reviews/ValidationUnseenSitesV28/sentinel/sentinel.json)
· [Replay](../Reviews/ValidationUnseenSitesV28/sentinel_verification/verification.json).

## Independent reference results

| Site | Airbus archive leads | Sampled Wayback capture | Coarse Landsat 8 records |
| --- | --- | --- | ---: |
| site-01 | No scene returned in the search window | 20 September 2011 | 13 |
| site-02 | SPOT-7, 22 November 2017, partial crop coverage | 9 June 2010 | 5 |
| site-03 | SPOT-7, 23 May 2018, full crop coverage | 13 September 2012 | 10 |

All 19 unique reference queries completed across the first run and one bounded retry.
The retry recovered both Airbus timeouts for site-02 and reused every other
response. Both runs replayed byte for byte. Wayback lists 0.5 m historical
captures for all three sites. Its source captures precede the search windows.
The SPOT-7 entry for site-03 is an archive lead. Scene sampling, image access
and crop clarity remain unresolved. Landsat has 15 m panchromatic and 30 m
multispectral sampling. The 3 June 2018 Landsat record covers the full site-03
crop according to the provider and falls inside the five-day comparison.
[Initial reference inventory](../Reviews/ValidationUnseenSitesV28/references/references.json)
· [Completed reference inventory](../Reviews/ValidationUnseenSitesV28/reference_retry/references.json)
· [USGS Landsat 8 sampling](https://www.usgs.gov/landsat-missions/landsat-8).

## Feasibility outcome

Site-03 is the first candidate for native Level-1C metadata and asset-list
preflight. The 30 May and 4 June comparison combines a short interval, low
scene-level cloud, a same-orbit control opportunity on 10 and 30 May, and
complementary dated reference leads before and during the comparison. A later
same-orbit follow-up is available in the catalogue from 9 June. The 2 May to
1 June pair is the first-ranked option under the archived scene-cloud rule
and leaves a longer timing interval. The catalogue retains all 24 pair options
and the other two sites as alternatives.

The shortlist is complete for this metadata phase. Scene transfer, crop QA,
control classification and source access checks require later work. No labels
are approved and no imagery acquisition roster is locked. The one-worker
guard peaked at 162.47 MiB during planning, and available memory stayed at or
above 3.39 GiB. [Integrity checks](../Reviews/ValidationUnseenSitesV28/checks/final_integrity.json).
