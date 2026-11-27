# Stage 1 follow-up on the site-15 reference pair

Continue on `codex/curated-deforestation-pilot` and push verified
checkpoints to that branch. The [V35 result](../Reviews/ValidationStage1SearchV35/result.json)
retains six reviewed cases and eight untouched groups. The
[site-15 follow-up](../Reviews/ValidationStage1SearchV35/wave-03.json)
selects the strongest remaining pair by maximum Tier 1 scene cloud.
It fixes the original target and crop and uses current vector data to
screen the full control context before viewing imagery.

The independent Landsat dates are 16 November 2017 and 10 January 2018.
Scene cloud is 0.61% and 32.81%. Assess the actual crop, because scene
cloud and a mapped control do not establish visibility or forest class.
Run the existing bounded source checks with one low-priority worker.
Archive both dated decisions, control and original geometry, local
preview hashes and offline replay. Stage 2 remains pending while
the earlier forest, later opening and whole-control evidence are assessed.

The [Sentinel-2 replay](../Reviews/ValidationStage1SearchV35/sentinel_wave03_replay/verification.json)
verified 10 catalogue items and two compatible options. The top option
is 16 November–6 December 2017, with catalogue cloud of 71.0987% and
68.0326%. The later Landsat scene follows that Sentinel-2 endpoint by
35 days. It can corroborate later appearance.

The [map check](../Reviews/ValidationStage1SearchV35/controls_wave03/control_and_pair_options.json)
retains the original control disposition and selects the alternate
16 × 16 square at row 2, column 44. The full context is 24.356 m from
the nearest mapped road. The [locked source receipts](../Reviews/ValidationStage1SearchV35/reference_access_wave03.json)
confirm both exact 43 × 43 Landsat extents. The
[candidate protocol](../Reviews/ValidationStage1SearchV35/candidate_protocols/site-15.json)
fixes the sources and geometry. The two guarded previews total 8,375
bytes. The
[packet](../Reviews/ValidationStage1SearchV35/candidates/site-15/packet.json)
and [replay](../Reviews/ValidationStage1SearchV35/candidates/site-15/verification.json)
verify the exact source crops and both review boards. Peak worker RSS
was 119.34 MiB for preparation and 129.61 MiB for replay.

Both reviewers found a pale patch reaching into the target on both
dates and no coherent later opening. The alternate control straddles
a persistent green-to-mauve boundary. Earlier forest and stable forest
cover across the whole square remain unresolved. The
[site-15 disposition](../Reviews/ValidationStage1SearchV35/candidates/site-15/review.json)
keeps the case pending and adds its group to the exposure history.

## Site-09 annual radar control check

Site-09 remains the strongest target lead. The
[radar metadata](../Reviews/ValidationStage1SearchV35/candidates/site-09/radar-metadata-v36.json)
identifies three public JAXA PALSAR-2 forest-map items, from 2017, 2018
and 2019. The
[protocol](../Reviews/ValidationStage1SearchV35/candidates/site-09/radar-protocol-v36.json)
fixes a 14 × 9 cell native window and the existing control and context
polygons. Compare area-weighted published classes over the exact areas.
Annual class maps can corroborate forest or a mixed pattern. Their
sampling and composite dates leave canopy at specific dates uncertain.
The three crops total 1,476 bytes and replayed offline. The
[class summaries](../Reviews/ValidationStage1SearchV35/candidates/site-09/radar-control-v36.json)
show non-forest in 11.8230%, 16.3065% and 13.4386% of the fixed square
for 2017, 2018 and 2019. The associated full contexts show the same
mixed classification. This supports the existing control ambiguity.
Keep the fixed square and its disposition. A separate geometry search
will check for any alternative within the same frozen crop. Satellite
confirmation would still be needed for a new candidate. [JAXA product](https://www.eorc.jaxa.jp/ALOS/en/dataset/fnf_e.htm)
· [Planetary Computer collection](https://planetarycomputer.microsoft.com/api/stac/v1/collections/alos-fnf-mosaic).
