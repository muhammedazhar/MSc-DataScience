# Area B independent reference review

V22 is complete as of 23 November 2026. Historical imagery supports earlier
forest and later clearing in Area B. Landsat independently shows the broad
opening by 30 July 2018. Its 5 July image is obscured at the lower target.
Exact near-date forest class and class approval remain unresolved. Continue on
`codex/curated-deforestation-pilot` and push verified checkpoints to that branch.

Use Area B's exact V21 crop and spatial support. Capture date, source,
ground sampling and coverage must be traceable before a reference can support
a disposition. Keep the earlier optical expansion bracket and 10 August
persistence, and preserve the original target and all three controls.

Checkpoint `6e39c0c` pins the interpretation rule and review support. The
metadata checks confirm that three 2018 Esri releases use the same 14 June
2010 WorldView-2 image. The 8 June 2022 release uses 31 August 2020 WorldView-3
imagery. Both have 0.5 m source sampling and cover the crop. The selected
Landsat 8 observations are 5 and 30 July and 6 and 15 August 2018. They provide
15 m panchromatic detail and 30 m multispectral context. The site-specific
OpenAerialMap search returned no 2018 records.

A Google Earth web visit displayed a 6 November 2018 selection immediately
after the 14 June 2010 selection. The November view attributes imagery to
Maxar and visibly mixes older imagery with the November source. Its opening
is later corroboration. The view does not date all three control squares.

Checkpoint `be14d2f` freezes the dated sources and the live comparison before
its six fixed views were opened. Both WorldView images show canopy across
control 1 and most clearly support its candidacy. Control 2 also looks wooded
at both dates, with the later western opening nearer its lower-left corner.
That opening crosses control 3 by 2020. This weakens its suitability across
years while leaving its class during the 2018 window unresolved. Preserve all
three fixed squares and their separate V21 and V22 dispositions.

`Docs/Reviews/ValidationAreaBReferenceV22/visual_review.md` contains all visual
observations, source links and the timing limitations. `reviews.json` retains
all six live layer dispositions, the Google Earth mosaic caveat, the original
Sentinel-2 bracket and persistence, and the 30 July coarse corroboration.
`checks/final_integrity.json` binds source responses, map, review and worker
receipts. Metadata, outlines and display instructions are archived. The map
loads source imagery live from the providers.

Preparation and byte-identical replay peaked at 144.62 and 146.42 MiB. All 391
guarded tests passed with 344.81 MiB worker RSS high-water. No job remains active.
Keep one low-priority worker, one native thread, a 512 MiB cap and 1.5 GiB memory
reserve. Dated finer reference evidence from around July and August 2018 would
be needed to resolve the remaining class uncertainty. Preserve the original
approval and label-lock gates and the exposed validation group.
