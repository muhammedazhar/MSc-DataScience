# Completed prediction-blind visual review

**Disposition: rejected.** The required whole-target forest-to-clearing
transition and a defensible whole-ROI forest control are not established.
No labels, observed change interval or validation predictions were produced.

The exact packet and control roster were committed at `4280811` and pushed
before inspection. All five image artifacts were opened: [timeline 1](preparation/timeline-01.png),
[timeline 2](preparation/timeline-02.png), [timeline 3](preparation/timeline-03.png),
[endpoint pair](preparation/pair.png) and [target detail](preparation/target-detail.png).
Each chronological page includes plain RGB, separate target/control/context
overlays and frozen QA. The detail is nearest-neighbour enlargement of original
10 m pixels, adding no information. Source/date/pixel integrity was independently
verified before inspection. No deforestation predictions were read.

## Visual finding

The target neighbourhood contains a pale tan/grey vertical feature with adjoining
dark or mixed pixels in the April 2 pre image. A substantially similar feature
is present on April 27. March 13 also contains a bright/mixed feature there.
Later surroundings become greener, but these views do not establish that the
entire irregular target was forest before and newly cleared after. Clear May 17
and June 6 views do not resolve that missing pre-forest evidence. This finding
retains uncertainty; it does not prove the location never experienced forest loss
or assign the feature an exact land-cover class or clearing date.

C1 is the strongest plausible forest-control region, but dark/smooth early
10 m RGB does not certify homogeneous forest across all 256 pixels at both
endpoints. C2 has an upper bright linear margin and C3 has boundary/mixed-texture
ambiguity. None is assigned a definite nonforest diagnosis. There is no redraw,
smaller ROI, replacement square, alternate pair or new case after inspection.

A separate assistant critique viewed the same five committed images and reached
the same conservative finding: target states remain uncertain on clear-context
dates, C1 is plausible but uncertified forest, and C2/C3 have additional class
ambiguity. Both are assistant assessments, not independent expert ground truth.

## Every date and full input context

Counts below are unusable pixels in the full radius 2 input contexts, not just
the ROI. Target context has 136 pixels; each control context has 400. On zero-flag
dates the target/control states remain **uncertain**. Flagged required contexts
receive **obscured** abstentions; that term does not assert every ROI pixel is
literally cloud-covered. Full notes and all 36 control-date statements are in
[reviews.json](reviews.json), bound to exact protocol, metadata and artifacts.

| Date | Target flags | C1 flags | C2 flags | C3 flags |
| --- | ---: | ---: | ---: | ---: |
| 2019-03-13 | 0 | 32 | 8 | 4 |
| 2019-03-18 | 136 | 400 | 400 | 400 |
| 2019-03-28 | 136 | 400 | 400 | 400 |
| 2019-04-02 | 0 | 0 | 0 | 0 |
| 2019-04-07 | 136 | 400 | 400 | 400 |
| 2019-04-12 | 10 | 112 | 120 | 4 |
| 2019-04-17 | 132 | 392 | 368 | 340 |
| 2019-04-22 | 136 | 0 | 208 | 336 |
| 2019-04-27 | 0 | 0 | 0 | 0 |
| 2019-05-17 | 0 | 0 | 12 | 296 |
| 2019-06-01 | 136 | 400 | 400 | 400 |
| 2019-06-06 | 0 | 0 | 0 | 0 |

April 12 illustrates why a clean supervised interior cannot bypass context QA:
its four interior pixels have clear full context, while the larger target's
required context has ten flags. April 22 is the annotation image date but has
136/136 target-context flags; it does not certify onset or exact felling time.

The [final independent verification](review_verification/report.md) reconstructs
the recorded rejection, every control disposition and absence of a date interval.
It checks sources, complete coverage, saved masks, actual cache flags, exact
plain/QA pixels and review bindings; it does not independently certify visual
class truth. Labels remain unapproved, including all potential negative pixels.
