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
