# Original author annotation snapshots

These four small GeoJSON files were retrieved on 14 November 2026 from the
[authors' public data folder](https://drive.google.com/drive/folders/1GJwDQ7SqASyPlusjbYWHLbrpNg_wAtuP),
linked from their [research repository](https://github.com/QuantuMobileSoftware/forest_change_detection#dataset).
They support the provenance audit of the existing local annotation file.

`sources.json` records the original file URLs, roles, sizes and SHA-256 hashes.
The files are retained as retrieved; cleaning occurs in a separate output
catalogue. A fresh audit verifies their checksums and works offline.

- `36UXA_time-dependent.geojson`: 1,404 source records; the collection declares
  its single tile because its records have no tile-ID field.
- `36UYA_Spring_time-dependent.geojson`: 278 records, including five with null
  tile IDs. The audit preserves those five as source exclusions.
- `36UYA_Summer_time-dependent.geojson`: 123 records.
- `36UYA_36UXA_baseline.geojson`: 2,318 records, including 85 for tile 37UCR
  outside the local study's two-tile scope.

These are observation-date annotations. Baseline membership alone does not
establish a reliable first-visible-change date.

Please cite the authors when using their data:

Isaienkov, K., Yushchuk, M., Khramtsov, V., and Seliverstov, O. (2021).
*Deep Learning for Regular Change Detection in Ukrainian Forest Ecosystem With
Sentinel-2*. IEEE JSTARS, 14, 364–376.
[DOI: 10.1109/JSTARS.2020.3034186](https://doi.org/10.1109/JSTARS.2020.3034186).
