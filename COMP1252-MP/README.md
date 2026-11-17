# COMP1252 - Master's Project

For continuation with Codex, Gemini Antigravity or another coding agent, start
with [CONTINUE.md](./CONTINUE.md) on `codex/curated-deforestation-pilot`.

Read thee [Contribution Guidelines](./Docs/Guides/Contributions.md) to know how to contribute to the project. To run this project onto your local machine, go to the [Environment Setup](./Docs/Guides/EnvironmentSetup.md) page to setup the environment for the project.

The [single-sample validation guide](./Docs/Guides/SingleSampleValidation.md)
documents the corrected, resource-limited image-pair example and its inspection.
The [annotation audit guide](./Docs/Guides/AnnotationAudit.md) describes verified
source provenance, duplicate handling, geographic splits and candidate manifests.
The [pixel-quality guide](./Docs/Guides/PixelQuality.md) documents bounded cloud,
shadow-risk and invalid-pixel checks, resumable batches and review requirements.
The [reviewed pilot guide](./Docs/Guides/CuratedPilot.md) describes the 13 accepted
training pairs, stable-forest control regions, observed date intervals and
partial labels that require an ignore-aware loader and loss.
The [pilot training guide](./Docs/Guides/PilotBaseline.md) covers the implemented
loader, masked objectives and bounded CPU smoke run, including the short run's
all-positive predictions.
The [overfit diagnostic](./Docs/Guides/OverfitDiagnostic.md) confirms that the
same training path can fit one reviewed crop, with controlled weight/optimizer
comparisons and independent reconstruction of the saved results.
The [reviewed-pixel training guide](./Docs/Guides/ReviewedPixelPilot.md) documents
the corrected dataset-wide loss weighting and completed bounded fit of all 13
reviewed training examples. The [validation review guide](./Docs/Guides/ValidationPilot.md)
records the first bounded held-out review: all five candidates were rejected,
with independent evidence checks and a geometry-support audit. No validation
accuracy has been measured; usable reviewed validation labels remain needed.
The [broader validation support audit](./Docs/Guides/ValidationSupportPool.md)
accounts for all 540 screened validation pairs and explains why the current
rules yield no fresh review shortlist. Its independent check and all exclusion
records are saved without opening new imagery or measuring model accuracy.
The [local context quality audit](./Docs/Guides/ValidationContext.md) checks the
one fixed fresh candidate's actual model inputs. Its target contexts pass, but
the automatic forest-control proxy finds no complete qualifying square; visual
control/temporal evidence and approved validation labels are still needed.
The separate [manual timeline review](./Docs/Guides/ValidationManual.md) is now
complete: all 12 dates and three fixed controls were inspected, but the target
transition and forest-control classes remain uncertain. The rejected case
supplies no validation labels or first-visible interval; all 299 tests pass.
The [unscreened-pair inventory](./Docs/Guides/ValidationUnscreened.md) accounts
for the remaining 606 pairs/630 anchors. All retain original coverage or
provenance blocks; zero passes all metadata gates. All 305 tests pass.
The [coverage feasibility check](./Docs/Guides/ValidationCoverage.md) confirms
that neither of the two fresh supported events intersects any local Sentinel-2
scene. Exact source requirements are saved; all 311 tests pass.
The [source catalogue audit](./Docs/Guides/ValidationCatalogue.md) identifies
covering Sentinel-2 tile 36UXB on relative orbit R064 across all 7 original pair dates,
documents CDSE baseline 05.00 offset incompatibility and download size (4.69 GiB for 7 scenes);
all 316 tests pass.
The [source acquisition and reader adaptation guide](./Docs/Guides/ValidationAcquisition.md)
establishes direct reader compatibility and SAFE assembly for the minimal pair (2018-07-26 and 2018-07-31)
via AWS open data Baseline 02.06 (870.07 MiB combined for 7 required bands); all 319 tests pass.

## Structure of the repository

```bash
.
├── Datasets
│   ├── GoogleEarthEngine
│   ├── NASA-Earthdata
│   │   ├── PLOT1
│   │   └── PLOT2
│   ├── README.md
│   ├── USGS-EarthExplorer
│   └── bbox.geojson
├── Docs
│   ├── Diagrams
│   │   └── xarray-boxes-2.png
│   ├── Guides
│   │   ├── Contributions.md
│   │   ├── EnvironmentSetup.md
│   │   └── GPG-KeyDiagnosing.md
│   ├── References
│   │   └── Links.md
│   ├── Templates
│   │   └── .env
│   └── requirements.txt
├── Keys
│   └── service_account.json
├── LICENSE.md
├── README.md
├── Solutions
│   ├── .env
│   ├── download.py
│   ├── env_check.py
│   └── kernel_check.py
└── Testing
    ├── Notebook.ipynb
    ├── Tasks.md
    ├── Testing.ipynb
    ├── gee.py
    └── usgs-m2m.py

15 directories, 21 files
```

## `earthaccess` package docs

>[earthaccess](https://earthaccess.readthedocs.io/en/latest/)

### Understanding the `count` Parameter

In the `earthaccess.search_data()` function, the count parameter specifies the maximum number of results to return from the search query. Here’s a quick rundown of how it works:

`count` Parameter: This determines the number of files or data products that are retrieved by the search. Setting `count=10` means you requested up to 10 files matching your search criteria.

### Implications of Different `count` Values

**Low Values**: Retrieve fewer files. Useful if you only need a small sample or if you want to limit the amount of data processed.

**High Values**: Retrieve more files. Useful for getting a comprehensive dataset but might lead to larger downloads and more data to process.
