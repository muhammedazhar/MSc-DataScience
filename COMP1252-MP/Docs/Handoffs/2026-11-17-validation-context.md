# Validation context quality audit continuation

## Current state: audit and independent verification complete; no control proposal

Continue on **`codex/curated-deforestation-pilot`**. Commit each completed,
verified logical change with this handoff updated, then push to the SAME GitHub
branch. Root `CONTINUE.md` is the tool-neutral entry point for Codex, Gemini
Antigravity or another agent. No previous chat or running worker needs recovering.

The approved one-candidate context audit is DONE. The full target and supervised
interior, each with a two-pixel input margin, pass QA and finite-feature checks
at BOTH dates. No forest-control square meets the fixed automatic NDVI rules.
Overall numeric eligibility remains **false**. No new image review, labels,
training, validation predictions, dataset or accuracy score was produced.
The missing control does not establish that forest is absent.

Checkpoints: `e2de044` predeclared protocol; `6c00a8b` tested worker and frozen
implementation; `f3bde46` completed raw audit and retained negative control outcome.
The following completion commit contains independent verification, diagnostic
counts, the final 278-test receipt and this complete handoff. Completed checkpoints
are pushed to origin on the same branch. Previous broader audit: `41fb657`.

## Fixed candidate, context contract and results

Protocol: `Docs/Reviews/ValidationContextV3/protocol.json`, committed before
processing. It binds 19 source inputs, original immutable implementations,
twelve audit implementation files, the exact model factory and capacity limits.
The unique V2 record whose reasons are exactly `endpoint_crop_qa_flags` is fixed:

- Event: `event-bf3d28f31dcf19938962`.
- Pair: `pair-430ec0ecae32ce98efcb`.
- Fresh validation group: `group-45f0c39e787ca8622918`.
- Stratum: `36UYA_Spring_time-dependent`.
- Pre/post images: 2019-04-02 / 2019-04-27; annotation image date: 2019-04-22.
- Original 64x64, 10m grid, EPSG:32636;
  transform `[10, 0, 771700, 0, -10, 5537480]`.

The frozen tiny CNN has two 3x3 convolutions and a 1x1 output: a 5x5 receptive
field, radius 2 square input margin, 2,545 parameters and no global operations.
An exact factory AST/hash check and synthetic gradient test establish this
architecture contract without trained weights or real validation inputs. Do not
transfer it to a larger/global-context model.

| Region | ROI pixels | Context pixels | Full context inside | QA pre / post | Nonfinite/invalid | Result |
| --- | ---: | ---: | --- | --- | ---: | --- |
| Full target | 36 | 136 | Yes | 0 / 0 | 0 | Pass |
| Eroded supervised interior | 4 | 36 | Yes | 0 / 0 | 0 | Pass |
| Proposed control | None | None | Not applicable | Not applicable | Not applicable | No proposal |

Both target contexts are clear in all five frozen components: cloud, cloud buffer,
shadow risk, haze risk and invalid. All 27 actual input channels are finite there.
No ROI or margin was shrunk, clipped or filtered to force a pass. Original whole
crop flags are retained unchanged: 56 shadow-risk pixels before, zero after;
other components are zero. Those flagged pixels are outside both target contexts.
Full feature footprints remain isolated from frozen train/test footprints and all
13 actual training grids. Later metadata-usable observations on May 17, June 1
and June 6 are only potential review opportunities, not persistence evidence.

## Why the automatic control search is negative

The fixed search checks 16x16, 12x12 and 8x8 ROIs, largest first then ascending
row/column. Every ROI pixel requires NDVI >0.6 at both endpoints and absolute
endpoint change <0.1. Every full radius-2 context must be inside the grid,
QA-clear and finite, and avoid the full-catalogue 20m buffered exclusion plus
full target context. There is no best-NDVI/model ranking or fallback.

| ROI width | Inbounds squares | QA/finite context | Also outside exclusions | All gates |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 2025 | 1809 | 732 | 0 |
| 12 | 2401 | 2185 | 1360 | 0 |
| 8 | 2809 | 2595 | 1990 | 0 |

Independent report-only diagnostics also find zero complete squares at each size
satisfying all ROI pixels' pre-date greenness, post-date greenness or stable-change
threshold. These are overlapping window counts, not distinct sites or independent
samples. They do not mean every pixel has low NDVI or every safe region is unstable.
NDVI is an automatic search proxy, not a forest/stability label. Thresholds, search
order, original proposal and negative approval state remain unchanged.

## Independent verification, tests and resource receipts

Independent verification **PASSED**. It reconstructs target geometry against the
published sixth QA plane, erosion with nine NumPy slices, square margins with
25 shifts, full-catalogue exclusions, both-date QA and the deterministic control
search with prefix sums. It checks safe NPZ schema and array/artifact hashes,
all saved feature channels, index/difference identities, cached RGB/NDVI consistency,
architecture/source bindings and before/after source fingerprints. The raw worker
uses separate geometry/erosion/context/search helpers.

The audit extracted actual model features with fourteen bounded source-band
window requests for this single pair. The verifier did NOT re-extract satellite
bands; source values remain worker-extracted evidence cross-checked against cache
fields, internal feature relationships and source fingerprints. Neither audit nor
verification imports Torch or executes a cloud/deforestation model. No new imagery
was visually inspected and all labels remain unapproved.

**278 tests pass**, with no failures, errors or skips: prior 263 plus 9 context/
protocol cases, one actual-factory receptive-field gradient test and 5 independent
verifier cases. Full suite completed in a 5.85s guarded worker, peak sampled RSS
299.50MiB, minimum 2.84GiB available. Its Torch runtime reports
`2.6.0.dev20241112` while distribution metadata says `2.5.1`; this existing
discrepancy remains documented, not repaired or reinstalled.

| Completed job | Elapsed | Peak worker RSS | Minimum available |
| --- | ---: | ---: | ---: |
| One-pair context/feature audit | 6.56s | 205.62MiB | 3.03GiB |
| Independent verification | 2.53s | 130.52MiB | 2.75GiB |
| Full 278-test suite | 5.85s | 299.50MiB | 2.84GiB |

All workers were sequential nice +10 CPU workers, native threads 1, no GPU,
unchanged 512MiB RSS and 1.5GiB available-memory reserve guard. Audit/verification
maximum runtime was 300s; the test suite used 120s. No pixel/model job remains
active. Do not repeat these completed jobs without a new failure, source change
or specific recheck need.

## Durable evidence and frozen inputs

Authoritative tracked evidence: `Docs/Reviews/ValidationContextV3/`:

- `protocol.json`: predeclared candidate, sources, architecture and fixed rules.
- `audit/{context_arrays.npz,metadata.json,report.md}`: safe numeric evidence,
  per-region results, unchanged full-crop flags and bounded feature-read receipt.
- `verification/{metadata.json,report.md}`: independent reconstruction and
  report-only control gate diagnostics.
- `checks/`: focused, architecture and final full-suite receipts and logs.

`checks/final_integrity.json` confirms all 19 source inputs, 27 immutable
sources, 12 frozen worker implementations and 116 source-file size/mtime
fingerprints still match. Archived audit/verification/test outputs match the
completed local copies, with no active context processing worker.

New code: `Solutions/audit_validation_context.py`,
`Solutions/validation_context_worker.py`, `Solutions/verify_validation_context.py`.
Tests: `Tests/test_validation_context.py`, `Tests/test_validation_context_architecture.py`,
`Tests/test_validation_context_verifier.py`. Guide: `Docs/Guides/ValidationContext.md`.
Ignored local working copies are under `Datasets/ValidationPilot/v3/`.
GitHub contains code and bounded evidence; full source imagery, caches and audit
inputs remain on this Mac and are needed for source-aware reproduction.

Do not edit original frozen QA/model/objective/curation/validation sources or the
twelve implementations frozen by this completed protocol. QA policy:
`fe2472ec030f466bebbb9dc2cb033ed6175de489bcf022f650081dd8e9347560`.
The full V1 rejected roster and V2 empty shortlist are unchanged. Preserve masks
0 reviewed forest, 1 reviewed clearing, 255 unknown, and all split barriers.

Prior provenance and constraints:
`Docs/Handoffs/2026-11-16-validation-support-pool.md`,
`Docs/Handoffs/2026-11-16-validation-pilot.md`, and their linked training/curation
handoffs. Annotation image dates are not exact felling dates; the project aims
to locate loss and estimate the interval when it first becomes visible.

## Next bounded task (not executed)

Prepare a SEPARATE, predeclared, prediction-blind manual image/timeline review
of this target and potential forest controls. V3 is complete and negative; it
does not authorize visual labels under its automatic eligibility rule. A new
review protocol must explicitly distinguish observed forest stability from the
failed all-pixel NDVI proxy, preserve suitable minimum reviewed area, full clear/
finite input contexts, catalogue exclusions, split separation and uncertainty,
and require temporal forest-to-clearing evidence and later persistence.
Do not silently lower V3 thresholds, force a control or reuse a rejected group.
If a defensible control or target transition cannot be established, retain rejection.

No manual review preparation, new image inspection or validation label lock has
been performed in V3. Record a new protocol and all resulting review evidence
separately before considering label approval. Only committed approved labels,
independent dataset verification and a label lock may precede evaluation of the
frozen final 1664 checkpoint at 0.5. No tuning, retraining or test-group exposure.
First-visible-date inference remains later work.

Project root:
`/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Git root is its parent. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Check branch/status, active Python commands and completed output directories
before any new worker. Use one low-priority worker, threads 1, no GPU, unchanged
resource guard; never overwrite completed outputs or duplicate an active job.
