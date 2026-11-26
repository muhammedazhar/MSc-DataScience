# Stage 1 held-out candidate search

Continue on `codex/curated-deforestation-pilot` and push verified checkpoints
on that branch. The user requested the four remaining stages one at a time.
This stage searches for a usable case in the unreviewed V28 groups.

The [protocol](../Reviews/ValidationStage1SearchV35/protocol.json) excludes
all ten groups in the V34 visual-exposure history and retains the V28
original-pair and prior-QA exclusions. Rebuild all 29 eligible crops in
14 groups with the frozen 64 × 64 grid, 1,200 m QA margin, positive support,
control context and geographic split. Use independent Landsat metadata
across every crop, then check compatible Sentinel-2 catalogue options
and the mapped control for the strongest prospects.

For pixel review, freeze the exact sources, target and control in advance.
Use scene-locked small dated crops to judge earlier forest, later opening
and plausible stable cover. Prefer an independently dated, accessible
before image with good crop clarity. Landsat scene cloud and map geometry
guide the search. The visual evidence determines whether a case advances.

Use one nice +10 CPU worker with one native thread, a 512 MiB RSS cap and
a 1.5 GiB available-memory reserve. Save raw small previews locally and
archive receipts, replay, and every reviewed disposition. Native
Sentinel-2 preflight and acquisition belong to stage 2.

## Completed roster and reference inventory

The [roster](../Reviews/ValidationStage1SearchV35/roster/roster.json)
reproduces the V28 plan and retains all 29 eligible events across 14
unreviewed groups. Every crop has a geometry-only control opportunity.
All 29 exact-crop Landsat catalogue requests completed. The
[reference inventory](../Reviews/ValidationStage1SearchV35/references/landsat.json)
retains the source scenes and chronology. Its
[offline replay](../Reviews/ValidationStage1SearchV35/reference_replay/replay.json)
reproduced every response without a network request. Roster reconstruction,
lookup and replay peaked at 165.52, 46.11 and 34.91 MiB RSS respectively.

## First three prospects

The [first batch](../Reviews/ValidationStage1SearchV35/wave-01.json) fixes
two dated reference leads each for sites 06, 09 and 07. All three
[Sentinel-2 searches](../Reviews/ValidationStage1SearchV35/sentinel/sentinel.json)
completed and replayed. They yielded 98 items and 54 same-orbit,
same-baseline pair options. Each site's top pair has 0% reported
scene cloud on both dates. The archived
[control checks](../Reviews/ValidationStage1SearchV35/controls/control_and_pair_options.json)
replayed all three current map snapshots. The original site-07 square
stays fixed. Use the mapped alternatives at sites 06 and 09.

Review site-07 first. It has a full 16 × 16 control, growing-season
references on 19 August and 20 September, and a 25-day Sentinel-2
candidate from 18 August to 12 September 2017. The independent before
reference is one day after the Sentinel-2 before image. The independent
later reference is eight days after its later image. The sighted
transition window will depend on the images actually reviewed.
The [access checks](../Reviews/ValidationStage1SearchV35/reference_access_wave01.json)
confirm all six Tier 1 Landsat records and exact locked crop routes.
Site-07's guarded packet retrieved two crops totaling 7,937 bytes and
reproduced both boards byte for byte offline. Its
[packet receipts](../Reviews/ValidationStage1SearchV35/candidates/site-07/packet.json)
and [replay](../Reviews/ValidationStage1SearchV35/candidates/site-07/verification.json)
bind the exact files and masks. Peak RSS was 146.28 MiB for preparation
and 137.50 MiB for replay.

On site-07, two reviewers saw a pale patch over much of the target on
19 August. The same footprint is brighter on 20 September, so earlier
continuous forest remains unverified. The fixed control is mostly pale
and north of a sharp landscape boundary. Neither date establishes stable
canopy in the square. The [disposition](../Reviews/ValidationStage1SearchV35/candidates/site-07/review.json)
keeps site-07 pending.

Site-09 uses the already pinned 27 May and 30 July 2018 reference crops
and the mapped 16 × 16 alternate control. Two crops totaling 8,293 bytes
were retrieved. The
[packet](../Reviews/ValidationStage1SearchV35/candidates/site-09/packet.json)
and [offline replay](../Reviews/ValidationStage1SearchV35/candidates/site-09/verification.json)
verify the fixed imagery and both boards. Both reviewers saw a later pale,
L-shaped opening over much of the earlier dark strip. A pale patch was
already visible beside the western notch, and some target margins appear
mixed. The fixed control is broadly stable but has a persistent tonal
division. Whole-control forest remains unresolved. The
[site-09 disposition](../Reviews/ValidationStage1SearchV35/candidates/site-09/review.json)
retains it as a target prospect pending control evidence.

Site-06 uses the pinned 9 April and 11 May 2018 reference crops and
the full context of its mapped 8 × 8 alternate control. The
[packet](../Reviews/ValidationStage1SearchV35/candidates/site-06/packet.json)
binds 8,700 bytes of raw previews and the exact
[offline board replay](../Reviews/ValidationStage1SearchV35/candidates/site-06/verification.json).
Both reviewers saw the pale diagonal opening already present over much
of the target on 9 April. The later opening is brighter and more
continuous, and the surrounding colours also change. The small control
remains relatively dark, with forest class unresolved at Landsat
sampling. The [disposition](../Reviews/ValidationStage1SearchV35/candidates/site-06/review.json)
keeps site-06 pending.

## Second batch

The [second batch](../Reviews/ValidationStage1SearchV35/wave-02.json)
retains sites 17, 21 and 27 in three different groups outside the first
batch. Favor the first two for same-path, low-cloud references and keep
site-27 as a leaf-on reserve. Recheck each exact crop's dates, map
control and Landsat access before opening a preview. The
[Sentinel-2 replay](../Reviews/ValidationStage1SearchV35/sentinel_wave02_replay/verification.json)
verified 113 items and 42 compatible options. The
[map checks](../Reviews/ValidationStage1SearchV35/controls_wave02/control_and_pair_options.json)
retain the original site-21 and site-27 squares and a 16 × 16 mapped
alternative at site-17. The site-17 and site-27 full contexts lie within
mapped woodland. All three current map snapshots replayed. The
[source access checks](../Reviews/ValidationStage1SearchV35/reference_access_wave02.json)
verified all six exact 43 × 43 Landsat locks and USGS source identities.
The three candidate protocols freeze the second-batch geometry and dates.
Site-17 is first for visual review. Its two crops total 7,929 bytes.
The [packet](../Reviews/ValidationStage1SearchV35/candidates/site-17/packet.json)
and [offline replay](../Reviews/ValidationStage1SearchV35/candidates/site-17/verification.json)
bind the 9 April and 11 May views. A bright strip broadly follows the
target in May. Muted earlier cover and strong changes in surrounding
colour leave earlier forest and control class unresolved. The
[site-17 review](../Reviews/ValidationStage1SearchV35/candidates/site-17/review.json)
keeps the case pending.

Site-21's [packet](../Reviews/ValidationStage1SearchV35/candidates/site-21/packet.json)
and [replay](../Reviews/ValidationStage1SearchV35/candidates/site-21/verification.json)
bind the two fixed views and their 8,453 bytes of source previews. The
brightest May area is largely inside the earlier pink footprint. Some
brightness reaches the candidate edge. Full-target forest conversion and
stable forest class in the mixed control remain unresolved. The
[site-21 review](../Reviews/ValidationStage1SearchV35/candidates/site-21/review.json)
keeps the case pending.

The leaf-on site-27 [packet](../Reviews/ValidationStage1SearchV35/candidates/site-27/packet.json)
and [offline replay](../Reviews/ValidationStage1SearchV35/candidates/site-27/verification.json)
bind 7,975 bytes of dated previews. The main opening is visible in both
images. The tiny target sits on its shoulder where alignment and mixed
pixels confound possible edge expansion. The control looks vegetated in
September, with canopy obscured or uncertain in August. The
[site-27 review](../Reviews/ValidationStage1SearchV35/candidates/site-27/review.json)
keeps the case pending. Neither batch has advanced a candidate. Check
the availability of finer dated control evidence for site-09 before
closing the current checkpoint.
