# Fresh held-out site feasibility

V28 is a metadata feasibility check on `codex/curated-deforestation-pilot`.
Preserve the V27 results, all previous dispositions and the frozen training
checkpoint. The prior review used all four supported fresh groups with clear
original post crops. V28 starts from original temporal annotations in frozen
validation groups absent from the original pair inventory and prior review or
follow-up groups.

The [protocol](../Reviews/ValidationUnseenSitesV28/protocol.json) pins the
evidence and methods. The [site plan](../Reviews/ValidationUnseenSitesV28/plan/plan.json)
retains dispositions for all 514 validation events. Of 111 newly considered
temporal events in 27 groups, 37 in 17 groups passed the derived geometry gates.
The plan ranks eligible events by control geometry availability, eroded target
support and stable identifiers. It selects three different groups before live
catalogue requests. Native alignment, 1,200 m QA source coverage and frozen
train/test footprint isolation are required.

| Site | Geographic group | Annotation image date | Eroded target pixels | Proposed control |
| --- | --- | --- | ---: | --- |
| site-01 | `group-c4f441ae55f683f915fa` | 27 January 2018 | 151 | 16 × 16 pixels |
| site-02 | `group-d75dc0544bf1a0e519ad` | 28 November 2017 | 131 | 16 × 16 pixels |
| site-03 | `group-a85fa2022a25e1a72e2c` | 1 June 2018 | 115 | 16 × 16 pixels |

All use original tile 36UXA and the 64 × 64 native 10 m grid convention.
The proposed controls are unlabelled and belong to their parent groups.
Each has a full two-pixel context and avoids the buffered annotation catalogue.
`post_date` in the plan anchors Wayback release sampling to the annotation
image date. Selection of actual post-observations is part of the catalogue
assessment.

Search Earth Search L1C across the 30 days before through 45 days after each
annotation. Logical before and after options use the same tile, orbit and
baseline, cover the complete QA margin and span 5–30 days. The after image
must fall on or within ten days after the annotation image date. Archive
Airbus, Wayback and Landsat 8 reference metadata over each exact crop.
Use the actual source capture date and keep any access or resolution limits.
Each phase uses one low-priority CPU worker, one native thread, a 512 MiB
RSS cap, 1.5 GiB available-memory reserve and the 300 second runtime limit.
Planning passed at 162.47 MiB peak RSS. The second guarded run reproduced the
plan byte for byte. Live catalogue and reference checks remain pending.
