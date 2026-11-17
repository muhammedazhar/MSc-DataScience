# Target and forest-control context quality audit

The fixed candidate's **target and input context pass quality checks at both
dates**. The automatic forest-control search produces **no proposal**, so the
case does not pass the complete numeric protocol. No labels, model predictions
or validation accuracy were produced.

Candidate: `event-bf3d28f31dcf19938962`, pair `pair-430ec0ecae32ce98efcb`, validation
group `group-45f0c39e787ca8622918`. Images are dated 2019-04-02 and 2019-04-27;
the annotation image date is 2019-04-22. It is the unique fresh V2 candidate
excluded only by whole-crop QA, fixed before this separate V3 audit.

## Context contract and result

The current tiny CNN has two 3x3 convolutions and a 1x1 output. A reviewed output
pixel depends on a 5x5 input region: a two-pixel square, or Chebyshev, margin.
The actual factory is statically checked, source-hash bound and tested using
synthetic input gradients with nonzero paths inside that 5x5 region and zero
gradients outside. No trained weights or real validation inputs were used in
that architecture test. Larger/global-context models need a different contract.

This audit conservatively requires both the full anchor-temporal target plus
margin and its one-pixel-eroded supervised interior plus margin. Neither ROI
is shrunk, and missing boundary context cannot be silently clipped to pass.

| Region | ROI pixels | Context pixels | QA flags pre / post | Invalid/nonfinite context | Result |
| --- | ---: | ---: | --- | ---: | --- |
| Full target | 36 | 136 | 0 / 0 | 0 | Pass |
| Supervised interior | 4 | 36 | 0 / 0 | 0 | Pass |
| Proposed control | None | None | Not applicable | Not applicable | No proposal |

All 27 original pre/post/difference feature channels are finite on these target
contexts. They were extracted with fourteen bounded feature-band window requests,
preserving their scales, signs, grid and source fingerprints. Cached RGB/NDVI
were cross-checked against those channels; cache fields alone do not prove all
model inputs valid.

The original full-crop QA remains unchanged: **56 shadow-risk pixels before,
zero after**, with other QA components zero. These pixels are outside both
target contexts. This clears the local numeric quality concern for this exact
architecture; it does not certify visual forest/clearing truth or the entire image.

## Why no control was proposed

The predeclared search checked 16x16, 12x12 and 8x8 squares, largest first then
ascending row/column. Each full context had to be inside the grid, clear at both
dates, finite in all input channels, and outside all catalogue footprints with
a 20m margin plus the target context. ROI pixels additionally required NDVI>.6
at both endpoints and absolute endpoint NDVI change<.1. These are numerical
search aids, not ground-truth forest labels.

| Square size | Inbounds windows | QA/finite context | Also outside exclusions | All NDVI/proposal rules |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 2025 | 1809 | 732 | 0 |
| 12 | 2401 | 2185 | 1360 | 0 |
| 8 | 2809 | 2595 | 1990 | 0 |

The report-only diagnostics also show zero windows in each size satisfying
every ROI pixel's pre-date NDVI threshold, post-date threshold or stable-change
threshold. This does **not** mean every pixel has low NDVI, that forest is absent,
or that every safe region is unstable. It means no complete square met those
strict automatic rules. Windows overlap and are not independent control sites.
No threshold, search order, proposal or approval was changed after the result.

## Verification, preservation and capacity

The independent verifier reconstructed target geometry, erosion, square margins,
both-date cached QA, all saved feature checks, catalogue exclusion and the
first-match control search using separate NumPy mask shifts and prefix sums.
It confirmed the complete no-control result and all source/architecture bindings.
It did not re-extract raw satellite bands; those values remain worker-extracted
evidence checked against source fingerprints, index/difference relationships
and cached RGB/NDVI. Both processes imported no Torch and executed no cloud or
deforestation model. The original V1/V2 protocols, masks and empty shortlist remain
unchanged. All labels remain unapproved.

**278 tests pass.** The audit used **205.62MiB in 6.56s**, and verification used
**130.52MiB in 2.53s**. Each used one low-priority CPU worker, one native thread,
no GPU, the unchanged 512MiB RSS limit and 1.5GiB available-memory reserve.
Their sampled minimum available memory was 2.75GiB. The full test worker used
299.50MiB peak sampled RSS in 5.85s, with 2.84GiB minimum available memory.

Durable evidence is under `Docs/Reviews/ValidationContextV3/`: predeclared
`protocol.json`, raw `audit/` safe NPZ/metadata/report, independent
[verification report](../Reviews/ValidationContextV3/verification/report.md),
and `checks/` test logs. `CONTINUE.md` links the complete cross-agent handoff.

## Next task

The subsequent [manual timeline review](ValidationManual.md) is now complete
and independently verified. It rejects this case: the target's prior forest
state and whole-ROI forest controls remain ambiguous. No labels or timing
interval follow. The completed numeric protocol and its no-control result are
unchanged. The paragraph below records the separate review method proposed here.

Prepare a separately predeclared, prediction-blind image timeline for manual
assessment of the target and possible forest controls. The current numeric
protocol is complete and remains negative; do not silently retune its NDVI
thresholds or claim an approved control. A new manual-review protocol should
distinguish observed forest stability from the automatic spectral proxy, retain
uncertainty, require suitable reviewed area and persistence, and preserve clear,
finite input contexts and catalogue/split separation. If no defensible forest
control or forest-to-clearing transition can be established, retain rejection.

Commit explicit review evidence and approved labels before any model inference.
No temporal onset or independent accuracy claim follows from this QA result.
