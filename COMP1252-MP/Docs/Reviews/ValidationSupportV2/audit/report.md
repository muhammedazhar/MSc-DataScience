# Broader validation support audit and fixed shortlist

Broader frozen validation QA geometry/support audit and fixed prediction-blind review shortlist; no label approval or accuracy measurement.

Audited 540 anchor-pairs from 540 QA-screened validation pairs; retained 606 unscreened candidates separately.
Geometry support >=4 pixels: 100 anchor-pairs; fresh-review eligible: 0.

| Source | Audited anchor-pairs | Events | Interior >=4 |
| --- | ---: | ---: | ---: |
| 36UXA_time-dependent | 273 | 27 | 17 |
| 36UYA_Summer_time-dependent | 57 | 7 | 16 |
| 36UYA_Spring_time-dependent | 210 | 19 | 67 |

## Locked shortlist

| Role | Source | Event | Pair | Interior pixels |
| --- | --- | --- | --- | ---: |

Missing slots: 6. No fallback or approval was added.

Counts distinguish raw anchors, temporal clipping and eroded targets. Final targets were reconstructed per pair, never cached by event alone.
Prior reviewed events and entire groups are excluded from selection but retained in the audit.
All exclusions are in audit.json; reasons can overlap. No area/bbox/top-three, NDVI or model-score selection.
A later usable-metadata scene is only a review opportunity; no forest/clearing/persistence labels or model accuracy follow.
