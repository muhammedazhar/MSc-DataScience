# Independently verified input-context audit

Verified `event-bf3d28f31dcf19938962` / `pair-430ec0ecae32ce98efcb`. Eligible for visual review: **False**.
**No labels, training, model prediction or accuracy estimate were produced.**

| Region | ROI pixels | Context pixels | Full context inside | Clear/finite context |
| --- | ---: | ---: | --- | --- |
| target | 36 | 136 | True | True |
| positive_interior | 4 | 36 | True | True |
| control | 0 | 0 | False | False |

Independent NumPy mask dilation and prefix-sum control search reconstruct the producer decisions.
Factory AST, source/caches, all saved feature channels, source fingerprints and original QA flags are bound.
Raw source bands were read by the audit worker; this verification did not re-extract them.

Numeric QA and NDVI proposals do not approve forest stability or temporal loss labels.
Source feature values were not independently re-extracted from satellite bands.
The two-pixel context contract belongs only to the exact frozen tiny CNN.

## Report-only control gate diagnostics

These counts describe every square already covered by the fixed search. Marginals overlap; they are not cumulative filtering totals.
Only explicitly named composite columns count gates together. No threshold, search order, proposal or approval changed.

| ROI width | Inbounds | QA/finite context | QA/finite + exclusions | Pre NDVI >.6 | Post NDVI >.6 | Both >.6 | Delta <.1 | Safe context + both NDVI | Safe context + delta | All gates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2025 | 1809 | 732 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12 | 2401 | 2185 | 1360 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 2809 | 2595 | 1990 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
