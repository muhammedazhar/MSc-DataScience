# Download Sentinel-2 scenes by bounding box

[`Solutions/download.py`](../Solutions/download.py) reads plot polygons from
[`BoundingBox/bbox.geojson`](./BoundingBox/bbox.geojson). It uses each polygon's
bounding box to select a covering Sentinel-2 Level-1C scene. The bundled plots
are in Kerala. To search another area, use `plan --bbox-file` with a WGS84
GeoJSON FeatureCollection. Give each Polygon or MultiPolygon a unique
`properties.code`. An optional `properties.status` is kept in the report.

[`BoundingBox/search.json`](./BoundingBox/search.json) sets inclusive UTC dates,
a candidate count per plot and a scene-level cloud ceiling. The plan command
can override these with `--start`, `--end`, `--count` and `--cloud-cover`.
The Sentinel-2 collection replaces the old NASA ASTER dataset in this config.
The current QA reader requires the native Level-1C layout with GML quality
masks and zero offsets, so the search targets products below baseline 04.00.
The earliest Sentinel-2 acquisitions begin in 2015. A candidate's catalogue
cloud percentage describes the scene. Crop-level cloud QA happens later.

Run these commands from the project root with the project Python environment.
Each `--output` names a new report directory. The example uses a catalogue item
returned for the original `PLOT1` in the documented search window.

```bash
python Solutions/download.py plan --plot PLOT1 \
  --start 2019-01-01 --end 2019-02-28 --count 3 \
  --output Datasets/ValidationPilot/plot1-plan

python Solutions/download.py preflight --plan Datasets/ValidationPilot/plot1-plan \
  --scene S2B_43PFM_20190228_0_L1C \
  --output Datasets/ValidationPilot/plot1-preflight

python Solutions/download.py download --preflight Datasets/ValidationPilot/plot1-preflight \
  --output Datasets/ValidationPilot/plot1-transfer
```

Repeat `--plot` to search a group of plots. `--all-plots` includes the full file
in the plan and records any plots left partial when the bounded catalogue
request budget is reached. Run smaller groups for the remaining plots. Use a
candidate's exact ID from the plan or the terminal output. Preflight checks
native XML, full plot coverage and the 58 required files and reports the
listed size before transfer.

Download streams one scene at a time. It verifies provider size and MD5 pins,
retains SHA-256 receipts, checks the local SAFE metadata and publishes only
verified files under `Datasets/Sentinel-2/`. One low-priority worker runs with
one native thread and a 2 GiB free-disk reserve for transfers. A rerun with the
same preflight and a new report directory reuses verified staged files or
verifies an already published product. Keep the staging journal for resumption.
The bounding box selects coverage. The saved files are native scene tiles.
