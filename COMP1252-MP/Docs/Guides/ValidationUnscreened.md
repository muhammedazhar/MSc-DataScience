# Unscreened validation inventory

The completed inventory accounts for all **606 unscreened pairs and 630
anchor-pairs**, covering 67 events in 9 geographic groups. All pairs already
have blocking flags; absent published QA is not their only unresolved gate.
No satellite images, QA masks, caches or models were read or run.

| Original blocker | Pairs |
| --- | ---: |
| Crop crosses tile edge | 347 |
| Post scene does not cover crop | 347 |
| Unverified temporal label in crop | 259 |
| Baseline-only provenance | 79 |

Reasons overlap. The two coverage reasons occur on the same 347 pairs; the
provenance/label reasons overlap too. Counts must not be summed as distinct cases.
The original pair statuses and reasons are preserved, including multi-anchor pairs.

The target is each anchor intersected with the pair's cumulative post-date
catalogue mask minus the cumulative pre-date mask. A one-pixel 3 x 3 erosion
measures its conservative interior; geometry alone cannot establish image
availability, QA, forest/clearing truth or a timing label.

- 37 anchor-pairs from 6 events in 4 groups retain at least four interior pixels.
- 25 of those have baseline-only anchor provenance; geometry cannot authorize
  temporal labels from them.
- The other 12 are temporal anchors from two events in one fresh group,
  `group-96c27871459d03c08bcc`; all have source-coverage/tile-edge blocks.
- All 630 feature footprints are isolated from frozen train/test footprints.
- 566 anchor-pairs have a later metadata-usable observation opportunity.
- 203 anchor-pairs belong to previously inspected groups and remain excluded.
- **Zero passes all metadata/geometry gates.** Quality stays unknown (`null`
  counts), no QA roster is selected, and no labels or accuracy are produced.

Verification reconstructs catalogue projection, cumulative temporal masks and
erosion independently using Rasterio and NumPy. Metadata joins/gates and summary
helpers are shared; this is independent geometry verification, not an independent
class/QA certification. Every record, reason, mask hash and summary matches.
All 305 tests pass, including six new regressions for missing/duplicate anchors,
blocked records, temporal overlap, erosion and preserved metadata exclusions.

The inventory took 4.35 s at 266.77 MiB; verification took 3.90 s at 236.50 MiB.
Both used one nice +10 CPU worker, threads one, no GPU, the unchanged 512 MiB
RSS limit and 1.5 GiB available-memory reserve. Available memory stayed at least
3.24 GiB. No further processing is needed to inspect the saved result.

Evidence is under `Docs/Reviews/ValidationUnscreenedV5/`: predeclared protocol,
complete `inventory/`, `verification/`, tests and `planning_counts.json`.
The previous V1–V4 results and frozen sources remain unchanged. This inventory
retains all six previously inspected groups, including the rejected V4 group.

Next proposed task: a metadata-only coverage feasibility check for the two
fresh temporal events. Compare their exact grids against available scene/tile
footprints and identify whether suitable coverage already exists or additional
source products would be required. Keep their current pairs blocked; do not
start QA, shift grids or relax provenance merely to obtain validation examples.
See `Docs/Handoffs/2026-11-17-validation-unscreened.md` for continuation.

That subsequent [coverage check](ValidationCoverage.md) is now complete:
neither crop intersects any of the 50 local scene footprints. The original
blocks remain, and exact requirements for a future source search are saved.
