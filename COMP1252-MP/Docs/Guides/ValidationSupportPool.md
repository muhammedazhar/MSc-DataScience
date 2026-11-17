# Broader validation support audit

The broader audit is complete, independently verified, and leaves an **empty
fresh-review shortlist** under its predeclared rules. No new imagery was opened,
no labels were approved and no model accuracy was measured.

The audit inspected all completed validation QA pairs, removing the previous
area/bounding-dimension and top-three proposal restrictions. It also excluded
all five previously reviewed geographic groups from selection to preserve fresh
review. Those groups and every rejected record remain in the audit.

## Coverage and result

| Stage | Pairs / anchor-pairs | Events | Groups |
| --- | ---: | ---: | ---: |
| Completed validation QA audited | 540 | 53 | 14 |
| Actual eroded interior >=4 pixels | 100 | 10 | — |
| All quality, follow-up and separation gates before history exclusion | 8 | 2 | 2 |
| All gates including fresh review history | 0 | 0 | 0 |

All 1,146 frozen validation candidate pairs are accounted for. The 606 without
completed published QA are retained separately, with no invented support counts
or clearance claims. The 540 screened pairs contain 229 automatic passes, 248
quality exclusions and 63 visual-review cases. Every anchor is counted; current
pairs happen to have one anchor each, and tests protect multi-anchor handling.

The fixed shortlist retains all six missing primary/alternate slots explicitly.
An empty shortlist was permitted before the audit; no rule was relaxed to fill it.

## What was measured

Each original 64x64, 10m grid retains its geographic identity. The audit burns
each anchor, reconstructs its pair-specific temporal reference and applies one
pixel of 3x3 erosion. It records raw anchor pixels/interior, temporal union,
anchor target, clipped pixels and final interior separately, with mask hashes.
Earlier catalogue loss can change a temporal target across date pairs, so final
targets are not cached using only event/grid identity.

34 pairs had some temporal clipping. None crossed from an adequate raw interior
to an inadequate target interior at the four-pixel cutoff in this pool. The
actual insufficient-interior count is 440 pairs. This distinction avoids
attributing every small target to either polygon width or temporal clipping.

Selection additionally requires an automatic QA pass, zero endpoint crop and
target QA, zero actual interior QA, complete feature-footprint separation from
all train/test conservative footprints and the 13 actual training grids, and
a later metadata-usable same-year scene within annotation date +45 days. Such
a later scene is a review opportunity, not proof of clear persistence.

Ranking uses group/event identity, exact annotation-date post preference,
interval length and pair ID. All source primaries precede alternates and groups
cannot repeat. No NDVI, spectral contrast or model predictions influence ranking.
Area/bounding-size filters and prior top-candidate ranks are not used.

## Exclusions and diagnostic checks

| Exclusion reason | Anchor-pairs |
| --- | ---: |
| Interior below four pixels | 440 |
| Endpoint crop quality flags | 339 |
| Not an automatic quality pass | 311 |
| Endpoint target quality flags | 270 |
| Actual interior quality flags | 117 |
| Previously reviewed group | 312 |
| Previously rejected event | 55 |
| No later usable-metadata observation | 17 |

Reasons overlap and must not be added as if they were distinct populations.
Two diagnostic combinations were computed without changing the protocol or
selection: omitting only the whole-crop-zero-QA gate leaves one pair/event/group;
omitting only the previous-group exclusion leaves three pairs from one event
in one group. These are planning diagnostics, not approved candidates or labels.

The subsequent [one-case context audit](ValidationContext.md) is complete:
target contexts pass, while the automatic control proxy returns no proposal.
Its independent evidence and manual-review next step are now documented there.
That separate audit requires clear QA at both dates over each ROI and the exact
tiny CNN's two-pixel input margin. This architecture-specific result does not
change the broader audit's original rules or transfer to larger models. Visual
temporal review and committed labels still precede any inference.

## Evidence and verification

Evidence is tracked under `Docs/Reviews/ValidationSupportV2/`:

- `protocol.json`: rules, input/source hashes and resource bounds committed first.
- `audit/`: all 540 reconstructed records, 606 unscreened dispositions, complete
  exclusion reasons, fixed empty selection, source hashes and resource receipts.
- [independent report](../Reviews/ValidationSupportV2/verification/report.md):
  geometry, masks, gates, history, all counts and deterministic roster verified.
- `checks/`: focused and complete test logs.

The worker reconstructs temporal masks from catalogue dates and erodes them with
SciPy. The verifier instead intersects independently burned anchors with the
published sixth QA plane and erodes with nine NumPy slices. It checks exact
mask hashes, every source join, complete coverage and all selection fields.
Neither process imports Torch or reads source satellite JP2 pixels. Frozen QA
masks are read; cloud/QA models are not rerun.

**263 tests pass.** The audit used 198.77MiB in 4.59s, and independent verification
used 284.67MiB in 2.04s. Both were sequential nice+10 CPU workers, native threads
one, no GPU, under the unchanged 512MiB RSS and 1.5GiB available-memory reserve.
Measured available memory remained at least 2.55GiB. Original models, QA policy,
splits and prior review artifacts remain unchanged.

Rechecks require the original local audit/QA inputs. Use a new output directory
and the existing guarded `verify_validation_support_pool.py --run
Docs/Reviews/ValidationSupportV2/audit --output <new-directory>` command. There
is no need to rerun completed processing to continue from `CONTINUE.md`.
