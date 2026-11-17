# Target and control input-context quality audit

**Numeric eligibility only: no forest/clearing labels or model accuracy.**

Candidate `event-bf3d28f31dcf19938962` / `pair-430ec0ecae32ce98efcb`.
Eligible for subsequent visual review: **False**.

| Region | ROI pixels | Context pixels | Full context inside grid | Clear and finite context |
| --- | ---: | ---: | --- | --- |
| target | 36 | 136 | True | True |
| positive_interior | 4 | 36 | True | True |
| control | 0 | 0 | False | False |

Full target and supervised-interior contexts are both required. No smaller ROI/halo fallback.
A proposed forest control still needs explicit visual and temporal review. Missing controls remain a valid negative result.
All original full-crop flags and V1/V2 decisions remain unchanged. Context radius two is bound to the frozen tiny CNN only.
All 27 source features were checked in the one-case worker. Independent verification uses saved arrays/source fingerprints without repeating raw band extraction.
No model inference, training, visual preview or label approval was performed.
