# Small held-out validation review

The first bounded validation review produced **zero approved examples**.
Four candidates lack the required clearing interior after rasterization and
one cannot establish an unambiguous forested pre-image. No validation dataset,
label lock, model prediction or accuracy estimate was created.

This is a label-readiness result. It does not assess model performance and does
not establish that the complete validation split is unusable.

## Selection and blind review

The protocol and metadata-only roster were saved before image inspection in
`Docs/Reviews/ValidationPilotV1/`. Selection used frozen validation geographic
groups, clean endpoint QA and geometry search aids. It used no model predictions
or NDVI contrast ranking. All full 64x64 feature footprints were checked against
every train/test conservative footprint and the 13 actual training grids.

The bounded roster contained three primaries and two distinct-group alternates.
There was no eligible second Summer group under the same rules. All five
complete local timelines were opened, including cloudy observations; the Spring
primary's detailed before/after pair was also inspected. Review labels remained
unknown when evidence or minimum support was insufficient.

| Candidate suffix | Stratum | Role | Interior pixels | Outcome |
| --- | --- | --- | ---: | --- |
| 9993 | UXA | Primary | 1 | Below minimum four |
| 4ee5 | UYA Summer | Primary | 0 | Below minimum four |
| 4a343 | UYA Spring | Primary | 10 | Forested pre-image not established confidently |
| 9495 | UXA | Alternate | 0 | Below minimum four |
| 3304 | UYA Spring | Alternate | 0 | Below minimum four |

Candidate 4a343 lies in a road and disturbed-ground mosaic. Colour change and
an annotation image date do not establish a forest-to-clearing transition.
The inspection is assistant review, not independent expert certification.
Every prepared observation retains an explicit uncertain state; none of these
rejections grants clearing, control or stable-only evaluation labels.

## Why the geometry search aid was insufficient

The subsequent geometry-only audit reconstructed all **33 metadata-eligible
anchor-pairs from 11 events in six geographic groups**. It used frozen vector
annotations and grids, reading no satellite pixels and making no new image
selections or label approvals.

| Stratum | Audited anchor-pairs | Interior >=4 | Events with any interior >=4 |
| --- | ---: | ---: | ---: |
| UXA | 18 | 0 | 0 |
| UYA Summer | 6 | 0 | 0 |
| UYA Spring | 9 | 6 | 2 |

**27/33 anchor-pairs fail the interior-support gate.** Six pairs pass, but they
represent just two events in two groups. One is the already rejected ambiguous
Spring primary; the other has not been visually reviewed or approved. Passing
this geometric gate does not establish forest/clearing labels or temporal truth.

Area and bounding-box dimensions do not measure interior width. A thin polygon
and a square can have equal area, while only the square retains pixels after
the one-pixel boundary margin. Tests reproduce this with equal 1,600m2 shapes.
The anchor's own interior must also be counted separately from neighbouring
polygons in the combined temporal mask.

The audited pool inherits the original clear-image, geometry and top-candidate
filters. These counts do not describe all 514 validation events or all 91 frozen
validation groups. Source collection names do not establish seasonal coverage.

## Evidence, verification and Mac limits

All evidence is under `Docs/Reviews/ValidationPilotV1/`:

- `protocol.json`, `selection/`: predeclared criteria, roster and source hashes.
- `prepared_cases/`, `preparation_receipts/`, `reviews.json`: all five cases,
  previews, 35 hash-bound TIFFs, measured worker receipts and rejection reasons.
- [review verification report](../Reviews/ValidationPilotV1/review_verification/report.md):
  all five decisions and 53 timeline observations/caches checked independently.
- [geometry audit report](../Reviews/ValidationPilotV1/geometry_audit/report.md):
  exact raster support across the eligible pool.
- `checks/`: test logs and an actual rejected-roster export refusal check.

**250 tests pass.** The exporter refuses an all-rejected roster before publishing
an empty dataset. Inference also requires committed, hash-bound approved labels,
reviews and a label-lock file; none exists for this roster. Export and evaluation
helpers are implemented and unit-tested, but no real approved validation export
or inference has been executed.

Preparation used one low-priority CPU worker at a time: five workers totalled
63.99 seconds and peaked at 380.30MiB. Available system memory stayed at or above
2.63GiB. Independent review verification used 179.73MiB in 2.79s; geometry audit
used 165.14MiB in 2.55s. All retained the unchanged 512MiB worker limit, 1.5GiB
available-memory reserve, one native thread and no GPU. No training was repeated.

## Next bounded task

The subsequent [broader support audit](ValidationSupportPool.md) is complete:
all 540 screened validation pairs were audited and independently checked. Its
saved empty fresh-review shortlist and diagnostic gate counts now determine
the next task; do not repeat this first roster or its processing.

Use the completed broader audit's diagnostic evidence for a separately
predeclared quality/context review protocol. Keep the four-pixel minimum and
geographical separation intact. Preserve every rejected case and document
selection bias. Temporal ambiguity needs explicit review or expert adjudication,
not an assumed annotation date or automatic boundary labels.

Only after sufficient reviews are approved and committed should the frozen
final-1664 model be evaluated at threshold .5. Keep test groups untouched and
report partial-ROI metrics without complete-mask, area or exact-date claims.
