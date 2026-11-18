# Earlier source metadata preflight

V13 checked exactly 21 July then 16 July 2018, the two closest earlier V12 nominal
dates selected by date alone. Both pass actual L1C XML and required-object
listing checks; no imagery was downloaded or source approved for QA/labels.

Read `Docs/Handoffs/2026-11-18-earlier-source-metadata.md` for continuation.
The protocol was declared at 643e792; implementation/tests frozen at 8c13e74
before the eight operational metadata requests. All 358 tests pass.

The saved L1C STAC asset links point to a sentinel-s2-l2a bucket. The preflight
preserves this mismatch and uses explicit L1C source URLs, never those hrefs.
The provider's [L1C layout documentation](https://roda.sentinel-hub.com/sentinel-s2-l1c/readme.html)
describes tile data and product metadata under separate paths. Actual raw product/
tile XML is the identity/grid/scale evidence; a collection label or catalogue
offset alone cannot establish local frozen-reader readiness.

`Solutions/preflight_earlier_validation_metadata.py` saves two complete listings
and four original XML files, verifies If-Match/size/ETag/MD5/SHA, joins exact
product/tile/catalogue identity and builds complete 58-file manifests. It checks
the unchanged crops/full QA contexts, solar angles and measured conservative
disk budget. It never creates a SAFE skeleton or requests JP2/GML content.

Evidence: `Docs/Reviews/ValidationEarlierMetadataV13/preflight/` and `checks/`.
Completed local output `Datasets/ValidationPilot/v13/preflight/` must not be
rerun/overwritten. Payload totals 1,050,527,541 bytes (~1.05 GB); conservative
future staging allowance 2.07 GiB plus 2 GiB reserve, versus measured 24.05 GiB free.
Recheck live capacity before downloads. Metadata worker used 45.56 MiB in 2.06 s,
with at least 3.91 GiB available, one guarded nice +10 CPU worker and no GPU.

Next: separately predeclare strict acquisition of these 116 pinned files into
staging outside source discovery, verify unread payloads and actual local frozen
QA prerequisites, then recoverably publish. Pixel QA/new derived temporal targets
come later. Keep all old sources, cloudy outcomes, variants, grids and barriers.
