# Bounding-box Sentinel-2 downloader

The original `Solutions/download.py` now drives the project’s native
Sentinel-2 acquisition workflow. Continue on `codex/curated-deforestation-pilot`
and push verified checkpoints to that branch.

The original GeoJSON contains 104 Kerala plots. `plan` keeps each plot code,
status and geographic bounds. `search.json` retains its date window, candidate
count and cloud ceiling, with Sentinel-2 Level-1C replacing NASA ASTER. Other
areas can use their own WGS84 GeoJSON. The workflow targets native pre-04.00
products compatible with the frozen GML quality-mask reader. See the
[dataset guide](../../Datasets/README.md) for the commands and options.

`plan` archives the bounded catalogue search and deduplicates acquisitions.
`preflight` identifies one exact candidate, verifies native product and tile
XML, checks full plot coverage, and pins the sizes and MD5 ETags for the 58
required files. It records catalogue and native sensing times separately.
`download` uses the existing guarded transfer and durable transaction helpers.
It checks disk space, streams one scene, and verifies the local QA metadata
before publishing under `Datasets/Sentinel-2/`. An interrupted transfer can
resume with the same preflight and a new report directory. Previous QA and
validation evidence remain frozen.

The live bounded smoke used the original `PLOT1`, 1 January through 28 February
2019, three candidates and the configured cloud ceiling. One catalogue request
returned three covering IDs. Preflight of `S2B_43PFM_20190228_0_L1C` passed with
58 pinned assets and a listed payload of 425,082,466 bytes. The live plan and
preflight peaked at 69.41 and 64.62 MiB worker RSS. Their raw responses and
guard receipts are under `Datasets/ValidationPilot/bbox-downloader-smoke/`
in the local workspace.

All 394 tests passed under the guard. The three new regression tests cover
exact plot and scene binding, verified 58-file publication with tiny fixture
payloads, repeat runs with zero transfers, and resumption after interruption.
The full test worker peaked at 327.75 MiB RSS. The live smoke fetched metadata
only. The documented PLOT1 preflight is available for a future scene download.
