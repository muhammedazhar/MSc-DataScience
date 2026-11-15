# Reviewed partial-supervision training pilot

Assessed **28** candidate cases: **13** accepted and **15** rejected.
Supervised pixels: **151** clearing interiors and **2640** stable-forest controls; **50457** ignored.

Some candidates were rejected by geometry or quality gates before visual inspection. The assessment count does not imply that every rejected image was visually reviewed.

Accepted cases use unique training geographic groups and unchanged frozen 64-by-64 grids. No accepted core intersects a held-out conservative feature footprint.

## Accepted examples

| Event | Tile | Export pair | Observed interval | Width | Positive / stable pixels | Evidence |
| --- | --- | --- | --- | ---: | ---: | --- |
| `event-e691d44c335d87649310` | 36UYA | 2019-06-11 → 2019-06-26 | (2019-06-11, 2019-06-26] | 15 days | 25 / 256 | [timeline](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/review_parts/00002/event-e691d44c335d87649310/timeline.png>) · [pair](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/review_parts/00002/event-e691d44c335d87649310/pair.png>) |
| `event-263e9027a6142dc9fc84` | 36UYA | 2019-08-15 → 2019-08-25 | (2019-08-15, 2019-08-25] | 10 days | 13 / 256 | [timeline](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/review_parts/00004/event-263e9027a6142dc9fc84/timeline.png>) · [pair](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/review_parts/00004/event-263e9027a6142dc9fc84/pair.png>) |
| `event-8628872c64f4cabf6e67` | 36UXA | 2018-08-10 → 2018-08-20 | (2018-08-10, 2018-08-20] | 10 days | 6 / 256 | [timeline](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00000/event-8628872c64f4cabf6e67/timeline.png>) · [pair](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00000/event-8628872c64f4cabf6e67/pair.png>) |
| `event-b9d89f91b116d8d90fac` | 36UYA | 2019-06-26 → 2019-07-21 | (2019-07-01, 2019-07-21] | 20 days | 16 / 64 | [timeline](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00000/event-b9d89f91b116d8d90fac/timeline.png>) · [pair](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00000/event-b9d89f91b116d8d90fac/pair.png>) |
| `event-e59c75f02fe23bb9a1ae` | 36UYA | 2019-08-25 → 2019-08-30 | (2019-08-25, 2019-08-30] | 5 days | 5 / 256 | [timeline](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00000/event-e59c75f02fe23bb9a1ae/timeline.png>) · [pair](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00000/event-e59c75f02fe23bb9a1ae/pair.png>) |
| `event-e6224bcd39746b840ea4` | 36UXA | 2018-08-05 → 2018-08-10 | (2018-08-05, 2018-08-10] | 5 days | 6 / 256 | [timeline](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00000/event-e6224bcd39746b840ea4/timeline.png>) · [pair](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00000/event-e6224bcd39746b840ea4/pair.png>) |
| `event-073d95e8a1c667845607` | 36UXA | 2018-07-31 → 2018-08-05 | (2018-07-31, 2018-08-05] | 5 days | 12 / 64 | [timeline](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00001/event-073d95e8a1c667845607/timeline.png>) · [pair](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00001/event-073d95e8a1c667845607/pair.png>) |
| `event-4fa26438d5872a8480af` | 36UXA | 2018-07-31 → 2018-08-05 | (2018-07-31, 2018-08-05] | 5 days | 6 / 256 | [timeline](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00001/event-4fa26438d5872a8480af/timeline.png>) · [pair](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00001/event-4fa26438d5872a8480af/pair.png>) |
| `event-a72e10d66054921e60b2` | 36UXA | 2018-07-31 → 2018-08-05 | (2018-07-31, 2018-08-05] | 5 days | 15 / 256 | [timeline](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00001/event-a72e10d66054921e60b2/timeline.png>) · [pair](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00001/event-a72e10d66054921e60b2/pair.png>) |
| `event-f7e5f043776b24cb7dfa` | 36UXA | 2018-08-10 → 2018-08-20 | (2018-08-10, 2018-08-20] | 10 days | 10 / 144 | [timeline](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00001/event-f7e5f043776b24cb7dfa/timeline.png>) · [pair](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00001/event-f7e5f043776b24cb7dfa/pair.png>) |
| `event-0012effe85aa26b69541` | 36UXA | 2018-07-31 → 2018-08-05 | (2018-07-31, 2018-08-05] | 5 days | 15 / 256 | [timeline](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00002/event-0012effe85aa26b69541/timeline.png>) · [pair](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00002/event-0012effe85aa26b69541/pair.png>) |
| `event-65fa6d8d7a0c82498340` | 36UXA | 2018-07-31 → 2018-08-10 | (2018-07-31, 2018-08-10] | 10 days | 18 / 64 | [timeline](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00002/event-65fa6d8d7a0c82498340/timeline.png>) · [pair](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00002/event-65fa6d8d7a0c82498340/pair.png>) |
| `event-e932f0d28b2819f8368d` | 36UXA | 2018-08-05 → 2018-08-10 | (2018-08-05, 2018-08-10] | 5 days | 4 / 256 | [timeline](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00002/event-e932f0d28b2819f8368d/timeline.png>) · [pair](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/replacement_parts/00002/event-e932f0d28b2819f8368d/pair.png>) |

## Rejected candidates

| Event | Reason |
| --- | --- |
| `event-970414764804f89e60a9` | Rejected before label approval: only 0 pixels survive the one-pixel boundary margin; minimum pilot support is four pixels. No target or control states inferred from the unreviewed preview. |
| `event-d25898e9d88aacf3b5c7` | Rejected before label approval: only 0 pixels survive the one-pixel boundary margin; minimum pilot support is four pixels. No target or control states inferred from the unreviewed preview. |
| `event-dba65b972460aca5c126` | Rejected before label approval: only 0 pixels survive the one-pixel boundary margin; minimum pilot support is four pixels. No target or control states inferred from the unreviewed preview. |
| `event-fa401753669ab81027f4` | Rejected before label approval: only 1 pixels survive the one-pixel boundary margin; minimum pilot support is four pixels. No target or control states inferred from the unreviewed preview. |
| `event-0e2f7f165cfdf1d9ed7c` | Rejected before label approval: only 0 pixels survive the one-pixel boundary margin; minimum pilot support is four pixels. No target or control states inferred from the unreviewed preview. |
| `event-133d3d11c615a3c3bc26` | Inspected timeline but no fully quality-clear later target observation supports persistence; retain for further review, outside this pilot. |
| `event-69f390dfc9d176f50f1d` | Rejected before label approval: only 0 pixels survive the one-pixel boundary margin; minimum pilot support is four pixels. No target or control states inferred from the unreviewed preview. |
| `event-fb9c705bda551c5fd13e` | Rejected before label approval: only 0 pixels survive the one-pixel boundary margin; minimum pilot support is four pixels. No target or control states inferred from the unreviewed preview. |
| `event-09aec2bd330378e33f72` | Rejected before label approval: only 0 pixels survive the one-pixel boundary margin; minimum pilot support is four pixels. No target or control states inferred from the unreviewed preview. |
| `event-9d169a84581b6a4c50a5` | Rejected before label approval: only 0 pixels survive the one-pixel boundary margin; minimum pilot support is four pixels. No target or control states inferred from the unreviewed preview. |
| `event-aaff832c34619704f3a4` | Rejected before label approval: only 3 pixels survive the one-pixel boundary margin; minimum pilot support is four pixels. No target or control states inferred from the unreviewed preview. |
| `event-01dc179dcd8513caffaa` | Rejected before label approval: only 0 pixels survive the one-pixel boundary margin; minimum pilot support is four pixels. No target or control states inferred from the unreviewed preview. |
| `event-25888befa0297de3917f` | Rejected: post-image is the last available local acquisition, so later persistence cannot be independently observed. |
| `event-4027aba1f531addd4d1b` | Rejected before label approval: only 1 pixels survive the one-pixel boundary margin; minimum pilot support is four pixels. No target or control states inferred from the unreviewed preview. |
| `event-e48cbd49ed33587595b7` | Rejected before label approval: only 0 pixels survive the one-pixel boundary margin; minimum pilot support is four pixels. No target or control states inferred from the unreviewed preview. |

## Complete evidence

[All 28 candidate assessments and per-observation decisions](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/dataset/review_decisions.jsonl>) retain the complete review statements, rejection reasons, artifact hashes, visibility bounds and control evidence.

[Training manifest](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Datasets/CuratedPilot/v1/dataset/training_manifest.jsonl>) links each accepted example to its exported arrays and supervision counts.

## Loading and interpretation

`input_features.tif` has 27 float32 channels: pre nine, post nine, and post-minus-pre nine. Reflectance bands are scaled independently of the signed NDVI/NDMI indices; signed differences are not clipped.

`supervision_mask.tif` uses **1** for explicitly reviewed clearing interiors, **0** for reviewed stable-forest ROIs, and **255** everywhere else. `valid_mask.tif` is exactly `supervision_mask != 255`. The loader and loss must ignore 255; the existing binary trainer is incompatible until that behavior is implemented.

Controls remain inside the same original training cores. They are inspected ROI/interval controls, not independent held-out evaluation examples. Windowed first-visible evidence is not an exact felling date or a claim about observations outside the recorded local timeline.

This export performs no model training and grants no model-performance claim.
