# Site-03 native Level-1C preflight

V29 checks three exact observations from the completed V28 catalogue.
Continue on `codex/curated-deforestation-pilot` and push each verified
checkpoint to the same branch. Preserve the original crop and proposed
control at site-03 and keep all other V28 candidates in the archive.

| Date | Exact catalogue item | Role |
| --- | --- | --- |
| 10 May 2018 | `S2B_36UXA_20180510_0_L1C` | Earlier control observation |
| 30 May 2018 | `S2B_36UXA_20180530_0_L1C` | Before observation |
| 4 June 2018 | `S2A_36UXA_20180604_0_L1C` | After observation |

The [protocol](../Reviews/ValidationSite03PreflightV29/protocol.json) pins the
exact raw catalogue records, the crop and the unchanged downloader. The
[imported plan](../Reviews/ValidationSite03PreflightV29/catalogue_import/plan.json)
binds each of the three items to both the 64 × 64 crop and its 1,200 m QA source
margin. The raw catalogue items retain their Level-2A links. Native Level-1C
keys are derived from the compact 36UXA/R107/N0206 product identities.

Run `Solutions/download.py preflight` once per exact scene, with the imported
plan and a separate output directory. Each run may list the tile and read
its product and tile XML, using four metadata requests and one low-priority
worker. Keep one native thread, a 512 MiB RSS cap, a 1.5 GiB available-memory
reserve and the 300 second runtime cap. Check the projected native 10 m grid
against the exact crop and QA margin and verify the 58-file manifest.

The local catalogue import passed identity and footprint checks. The native
requests, final size calculation and offline replay are pending.
