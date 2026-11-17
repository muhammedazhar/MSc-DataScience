# Coverage feasibility for two validation events

**Neither fixed crop is covered by any of the 50 local Sentinel-2 scenes.**
The crops are wholly north of every source rectangle, with no positive-area
intersection. The original coverage blocks are correct and remain unchanged.

| Event suffix | Original crop bounds, EPSG:32636 | Gap north of nearest scene |
| --- | --- | ---: |
| `71d8da17cbbfd9babdee` | `[663640,5605000,664280,5605640]` | 4, 960 m |
| `ce1c1df8d02be33bb3c8` | `[663380,5604740,664020,5605380]` | 4, 700 m |

All local scene rectangles end at northing **5, 600, 040 m**:

| Tile | Scenes | Bounds, EPSG:32636 |
| --- | ---: | --- |
| 36 UXA | 12 | `[600000,5490240,709800,5600040]` |
| 36 UYA | 38 | `[699960,5490240,809760,5600040]` |

The check freshly parsed all 50 tile XML files, verified 50 product XML hashes,
and matched the frozen inventory. Directory discovery found exactly those 50
SAFE sources and no additional unindexed scenes in `Datasets/Sentinel-2/`.
The 100 event/scene comparisons include every local date and tile. Bounding-box
containment also agrees with native pixel-window containment; crop row starts
are negative in these scenes. Both pre and post images miss the crops.

Each event has six original blocked pairs: pre 2018-07-26, with posts 2018-07-31,
08-05, 08-10, 08-15, 08-20 and 08-25. Changing dates among the existing sources
cannot fix this spatial gap. The metadata-only review opened no satellite
pixels, imagery, QA masks or caches, and ran no cloud/forest model.

Additional correctly covering imagery, or a separately verified correction of
source/georeferencing information, is required before QA is useful here. The
fixed crops cannot be shifted into existing scenes to bypass the block.
The current cloud screen also requires a 1, 200 m source margin around each crop.
Their combined required context bounds are `[662180,5603540,665480,5606840]`
in EPSG:32636. Exact grids, dates and context bounds are saved in
`Docs/Reviews/ValidationCoverageV6/source_requirements.json`.

Next proposed task: a bounded catalogue metadata search for compatible source
products covering these fixed grids and dates, retaining processing baseline,
reader/QA compatibility and download sizes before any acquisition. No catalogue
availability, product suitability or download was established by this check.
Both events share one geographic group and cannot count as two independent
validation locations. Footprint coverage alone would not approve class labels.

**311 tests pass.** The metadata worker used **41.31 MiB in 0.51 s**, with 4.64 GiB
minimum available memory, one nice+10 CPU worker, threads 1, no GPU and the
unchanged 512 MiB/1.5 GiB memory guard. Full suite used 358.20 MiB sampled peak in 4.07 s.
Source/label/grid/model files remain frozen. No validation labels or accuracy exist.

Evidence: `Docs/Reviews/ValidationCoverageV6/`; continuation:
`Docs/Handoffs/2026-11-17-validation-coverage.md`.
