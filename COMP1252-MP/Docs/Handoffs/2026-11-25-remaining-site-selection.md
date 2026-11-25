# Remaining V28 site selection

V32 is complete. Site-01 is the nominated candidate for a bounded
reference review. Neither site has enough evidence to proceed to full
scene acquisition.

Continue on `codex/curated-deforestation-pilot` and push verified
checkpoints to the same branch. The
[V32 protocol](../Reviews/ValidationRemainingSitesV32/protocol.json)
pins the two remaining V28 sites, their six candidate pairs and the V31
visual-exposure history. Preserve each parent's exact crop, target, control,
full QA margin and frozen geographic split.

Compare independent reference timing, coverage and access, then the
geometry and known limitations of each control. Record all outcomes and
the strongest fixed follow-up candidate. Scene cloud comes from catalogue
metadata. Crop clarity and forest class require separate review.

## Reference access and original controls

The access reports for [site-01](../Reviews/ValidationRemainingSitesV32/site-01-access.json)
and [site-02](../Reviews/ValidationRemainingSitesV32/site-02-access.json) retain
the dated footprints, source cloud units and metadata-only access probes.
The [map check](../Reviews/ValidationRemainingSitesV32/control-context.json)
found a currently mapped path crossing the original site-01 square and a
primary road crossing the site-02 square. The site-02 road alignment was
already mapped before the comparison. The site-01 path's historical
presence is unverified. Retain both original control dispositions.

The [control search protocol](../Reviews/ValidationRemainingSitesV32/control_search_protocol.json)
pins a search for an alternative within each original crop. Use the existing
catalogue geometry gates and a 20 m clearance from mapped roads and from
areas outside mapped woodland, for the entire two-pixel context.
Current map geometry can nominate a review square. Forest class, historical
homogeneity and crop clarity require separate evidence.

## Fixed follow-up candidate

The [comparison](../Reviews/ValidationRemainingSitesV32/comparison.json)
retains all six V28 options. The nominated site-01 pair is 12 January to
6 February 2018, a 25-day comparison on tile 36UXA, orbit 64 and baseline
02.06. Earth Search IDs are `S2A_36UXA_20180112_0_L1C` and
`S2B_36UXA_20180206_0_L1C`. Both footprints cover the crop and the full
1,200 m QA margin. Scene cloud is 59.4713% before and 75.2523% after.
This option has compatible later catalogue observations. V28 found zero
earlier same-orbit stable-only dates for this before image.

The accessible Tier 1 Landsat before lead was acquired about 18 minutes
before the Sentinel-2 scene on 12 January and covers the exact crop.
Its reported scene cloud is 56.25%. The later 20 February Landsat lead
has 71.96% scene cloud. Public metadata resolves both locked 43 × 43
crop requests, with 15 m pan and 30 m multispectral source sampling.
Both crop dispositions and the earlier forest class remain unknown.
[USGS before item](https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l1/items/LC08_L1TP_177026_20180112_20200902_02_T1)
· [USGS later item](https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l1/items/LC08_L1TP_178025_20180220_20200902_02_T1).

The fixed site-01 alternative is a 16 × 16 square at zero-based row 44,
column 41 on the original crop. Its bounds are
`[700150, 5499120, 700310, 5499280]` in EPSG:32636. The entire 400-pixel
context is 76.225 m from the nearest mapped road and 184.832 m inside the
mapped woodland boundary. The current woodland geometry cannot establish
forest in 2018. The original control and its disposition remain archived.

Site-02 remains a reserve. Its mapped alternative is also 16 × 16, at row
5, column 45 in its original grid. The enclosing mapped woodland geometry
is traceable to before the comparison. Its best covering Landsat before
scene has 75.98% scene cloud. The 22 November SPOT-7 observation is within
both candidate windows, has 99.764% scene cloud and covers very little of
the original control. The sampled Wayback releases use older captures.
[USGS site-02 before item](https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l1/items/LC08_L1TP_178025_20171116_20200902_02_T1).

## Verification and next evidence

The guarded V28 geometry plan reproduced byte for byte. Offline replay
verified 103 Sentinel-2 catalogue items and 19 reference queries. The
alternate control search reproduced byte for byte and peaked at 139.81
MiB RSS with one low-priority CPU worker. All 394 existing tests passed.
The [integrity record](../Reviews/ValidationRemainingSitesV32/final_integrity.json)
binds the evidence and plan. V32 opened zero satellite pixels and approved
zero labels.

The next evidence is the two bounded dated Landsat views for site-01.
Check the clarity and appearance of the target and fixed control on each
date and record all limitations. Native Level-1C preflight remains a
separate acquisition decision.
