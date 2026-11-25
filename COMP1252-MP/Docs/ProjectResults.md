# Sentinel-2 forest-clearing pilot results

## Objective and data

The project investigates whether paired Sentinel-2 observations can locate
forest clearing and bound when the change first becomes visible. The timing
target is an observation interval. A confirmed interval starts after the last
clear observation showing forest and ends at the first clear observation
showing clearing. Annotation image dates identify observations used by the
source dataset. Cloud and gaps between acquisitions can widen the interval or
leave one end unknown.

The Ukrainian forest annotations come from Isaienkov et al. The provenance
audit matched all 5,696 local rows to the authors’ files and consolidated them
into 3,911 distinct events while preserving aliases and date conflicts. It
identified temporal provenance for 1,799 events and baseline-only provenance
for 2,112 events requiring temporal-label review. A seeded assignment of 601
geographic groups produced 416 training, 91 validation and 94 test groups.
The frozen crop and annotation footprints had zero cross-split overlaps.
[Original annotation snapshots and citation](../Datasets/AnnotationSources/README.md) ·
[Annotation and split audit](../Datasets/AnnotationAudit/v1/audit_report.md)

## Acquisition and processing

1. `Solutions/download.py` searches WGS84 GeoJSON bounding boxes, verifies the
   exact native Level-1C scene and lists its required assets before acquisition.
   The downloader streams one scene at a time and verifies file sizes and
   checksums before publishing its SAFE directory. The worked example uses
   the original Kerala plotting file. The Ukraine acquisition uses archived
   native manifests tied to the two validation crops. [Downloader guide](../Datasets/README.md)
2. Candidate pairs use fixed 64 × 64 crops at 10 m and intervals of 5–30 days.
   The temporal target uses cumulative loss through the post-observation date
   with cumulative loss through the pre-observation date removed. The QA checks
   clouds, buffers, shadows, haze and invalid data. Clearing labels come from
   reviewed eroded interiors. Stable-forest labels come from reviewed control
   regions, and uncertain pixels retain the ignore value `255`.
   [Annotation audit](Guides/AnnotationAudit.md) · [Pixel QA](Guides/PixelQuality.md)
3. Each model input has 27 channels. Seven reflectance bands, NDVI and NDMI are
   included for each observation, along with nine signed post-minus-pre
   channels. The reviewed-pixel objective uses one dataset-level positive
   class weight and ignores unreviewed pixels during training and scoring.
   [Reviewed training dataset](Guides/CuratedPilot.md) · [Training protocol](Guides/ReviewedPixelPilot.md)

## Training diagnostic

The pilot assessed 28 candidate cases and retained 13 reviewed pairs from 13
training groups. Their labelled support comprises 151 clearing-interior pixels
and 2,640 stable-forest pixels. Another 50,457 pixels are ignored. Visual review
and persistence evidence produced local observed intervals 5–20 days wide
for the retained training examples. The selection favors clear crops with
sufficient interior support and recognizable transitions.
[Dataset review and selection](Guides/CuratedPilot.md)

The predeclared 2,545-parameter CNN completed 1,664 training updates. At the
fixed threshold of 0.5, it correctly classified all 2,791 reviewed training
pixels. The pooled counts were TP 151, FP 0, FN 0 and TN 2,640, with Dice 1.0.
The archived verifier reconstructed the counts independently from the saved
logits. These measurements establish training fit on the selected regions.
Accuracy on unseen locations, calibration, clearing boundaries and total
loss area remain unmeasured. The closeout independently reproduced the original
counts at the same checkpoint and threshold. [Training replay](Reviews/Validation2019ScreenV27/checks/training_verification/report.md)

## Held-out validation evidence

| Stage | Outcome |
| --- | --- |
| Initial bounded manual review | Zero approvals in five cases. Four fell below the four-interior-pixel gate and one had ambiguous earlier cover. [Review](Guides/ValidationPilot.md) |
| Broader support audit | Of 1,146 original validation candidate pairs, 540 had published QA and 606 lacked it. The audit retained every record and produced an empty fresh shortlist under the fixed rules. [Support audit](Guides/ValidationSupportPool.md) · [Unscreened inventory](Guides/ValidationUnscreened.md) |
| Additional 2018 earlier observations | In the V24 two-site check, all four comparisons were excluded because every pixel in each earlier crop was cloud-flagged or in the cloud buffer. The fixed post crops had zero QA flags. [V24 screening and replay](Handoffs/2026-11-24-fresh-validation-qa.md) |
| 2019 archive search and preflight | Four exact native products covered two original crops and their QA margins. The independent Airbus and sampled Esri Wayback checks yielded zero full-crop captures dated inside the two reference windows. [Archive search](Handoffs/2026-11-24-2019-validation-feasibility.md) · [Native preflight](Handoffs/2026-11-24-2019-native-preflight.md) |

### 2019 acquisition and screening

The guarded downloader acquired all four verified Level-1C scenes, totaling
232 files and 2,208,628,047 bytes. Each comparison used the assigned original
crop and post observation, with checked geographic split isolation and a
recomputed cumulative loss reference. [Publication and lineage](Handoffs/2026-11-24-2019-validation-closeout.md)

| Earlier observation | Fixed post observation | Earlier crop QA flags | Earlier target-context QA flags | Disposition |
| --- | --- | ---: | ---: | --- |
| 3 March 2019 | 2 April 2019 | 1,876 / 4,096 | 146 / 392 | Suspected haze. Required contexts failed the review gate |
| 8 March 2019 | 2 April 2019 | 4,096 / 4,096 | 392 / 392 | Excluded for cloud or its buffer |
| 7 May 2019 | 6 June 2019 | 4,096 / 4,096 | 137 / 137 | Excluded for cloud or its buffer |
| 12 May 2019 | 6 June 2019 | 4,096 / 4,096 | 137 / 137 | Excluded for cloud or its buffer |

Both post crops and their target contexts had zero QA flags. On 3 March,
suspected haze also flagged 103 of 244 pixels in the eroded interior's required
context. The base screen classified that pair as needing visual review, and
none of the four passed the stricter review-candidate gate. Offline replay
matched all four QA masks, numeric outcomes and eight endpoint cache bindings.
No new held-out labels, first-visible intervals or accuracy estimates were
produced. [Screening](Reviews/Validation2019ScreenV27/screen/screen.json) ·
[Replay and integrity](Reviews/Validation2019ScreenV27/checks/final_integrity.json)

### Site-03 short-interval candidate

The 30 May–4 June 2018 pair passed automated QA and the 27-channel input audit.
Two model-blind reviewers observed most of the opening already visible
on 30 May. The fixed control had a persistent mixed edge and pale strip.
Independent Landsat imagery shows the broad opening by 27 May. The 3 June
Landsat crop is cloud-obscured. Site-03 adds no approved validation labels
or first-visible deforestation interval.
[Visual review](Reviews/ValidationSite03ReviewV31/visual_review.md) ·
[Reference review](Reviews/ValidationSite03ReviewV31/references/review.json)

### Site-01 dated reference candidate

The site-01 check used two bounded Landsat crops. Pale veiling and
diffuse bright bands left the earlier forest, later opening and control
unresolved. Site-01 stayed pending before Sentinel-2 preflight.
[Site-01 reference review](Reviews/ValidationSite01ReferenceV33/visual_review.md)

## Observational case study

Area B provides the strongest observation in the supplementary 2018 reviews.
The broad lower opening expanded after 6 July and by 31 July 2018 and remained
visible on 10 August.
Landsat independently shows a broad opening by 30 July. The 5 July Landsat
observation is obscured at the lower target. Earlier WorldView imagery shows
historical tree crowns, and the 2020 observation shows a matching later
opening. Mixed and pale pixels were already visible in part of the target on
6 July. The provisional Sentinel-2 bracket **(6 July, 31 July] 2018** applies
to the lower expansion. The onset of the original clearing and the exact
forest state of the lower target on 6 July remain unresolved.
[July visual review](Reviews/ValidationJulyVisualV19/visual_review.md) ·
[August persistence](Reviews/ValidationPersistenceV20/visual_review.md) ·
[Independent reference review](Reviews/ValidationAreaBReferenceV22/visual_review.md)

This is one previously inspected geographic group. The July input audit found
all 27 channels finite across the 4,096-pixel crop and retained three
provisional control squares. Independent imagery strengthened the forest
interpretation of the first two controls. Later open ground across the third
weakens its stability across years. Whole-control forest labels still need
evidence for the comparison dates. [Area B input and control audit](Reviews/ValidationAreaBInputsV21/visual_review.md) ·
[Reference control dispositions](Reviews/ValidationAreaBReferenceV22/visual_review.md)

## Conclusion and reproduction

The pipeline delivers reproducible acquisition, crop preparation, QA and a
verified fit on reviewed training pixels. The supplementary evidence supports
the Area B lower-opening expansion interval. The held-out measurement remains
open because the review produced zero approved validation labels. An evaluation
would require independent clear pairs, finite model inputs across each required
context, verified forest-to-clearing and stable-forest labels, and a locked
label set. The frozen 1,664-update checkpoint and 0.5 threshold are available
for that evaluation. False-alarm estimates across unchanged locations require
stable or no-change cases. [Validation procedure](Guides/ValidationPilot.md) ·
[Checkpoint and protocol](Guides/ReviewedPixelPilot.md)

The branch preserves the reviewed training exports, fixed splits, protocols,
saved weights, verification results and bounded quality evidence. The larger
source scenes and caches are local. The training verification can be repeated
from the tracked artifacts using the project Python environment and a new
output directory. From the project root, run one low-priority worker with one
native thread under the existing 512 MiB RSS cap and 1.5 GiB available-memory
reserve. The closeout passed all 394 tests.
[Environment setup](Guides/EnvironmentSetup.md) · [Continuation and replay commands](../CONTINUE.md)

```bash
python -B Solutions/verify_pixel_pilot.py \
  --run Docs/Reviews/PixelPilotV1/run \
  --output Datasets/ValidationPilot/project-results-verification
```
