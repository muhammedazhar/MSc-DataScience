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
The stage-one visual disposition is pending.
