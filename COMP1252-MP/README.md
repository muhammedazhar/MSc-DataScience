# COMP1252 - Master's Project

Read thee [Contribution Guidelines](./Docs/Guides/Contributions.md) to know how to contribute to the project. To run this project onto your local machine, go to the [Environment Setup](./Docs/Guides/EnvironmentSetup.md) page to setup the environment for the project.

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
