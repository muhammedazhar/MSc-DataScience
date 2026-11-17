# Manual target and forest-control timeline review

This separate V4 review asks whether images support a forest-to-clearing
transition at the fixed target and stable forest in a nearby control. The
completed [V3 context audit](ValidationContext.md) showed clear target inputs
but found no control meeting its automatic NDVI rules. V3 remains negative.

Current status: all12dates are prepared and independently verified. The exact
three-control roster and artifacts are committed before image inspection.
No visual verdict, labels or validation accuracy exist at this checkpoint.

## What is fixed before inspection

The sole case is `event-bf3d28f31dcf19938962`, pair
`pair-430ec0ecae32ce98efcb`, validation group `group-45f0c39e787ca8622918`.
The original images are 2019-04-02 and 2019-04-27; the annotation image date is
2019-04-22. The unchanged 64x64 grid has 10m pixels, a 36-pixel target and
four-pixel eroded supervised interior. Both target regions' two-pixel input
margins are quality-clear and finite in all 27 endpoint model features.

The complete local same-year window is 2019-03-08 through 2019-06-06. All
12 available scenes remain, including cloudy and uncertain dates. Eight
quality caches already exist; four missing observations are prepared using
the frozen cloud/quality policy under the Mac resource guard. That screening
does not predict deforestation. Endpoint features are reused from V3.
Preparing four missing QA observations does read 44 bounded source-band windows,
including 28 per-date feature requests for display/validity, and executes the
frozen cloud classifier four times. It does not re-extract the endpoint
27-channel feature stack or run the deforestation model.

Before viewing images, a deterministic search locks at most three square
controls: sizes 16, 12 and 8, largest first, then ascending row and column.
Every full two-pixel margin must be inside the grid, clear and finite at both
endpoints, and outside all catalogue geometries buffered by 20m, the target
context, and earlier control contexts. No NDVI threshold, image class or model
prediction enters this roster. The roster and artifacts are committed before
inspection; candidates cannot later be shrunk, redrawn or replaced.

## What the images must establish

Reviewers inspect plain RGB separately from exact target/control/context
overlays and quality masks. Display uses a fixed 0–0.30 reflectance range and
nearest-neighbour enlargement of the original pixels. All dates and controls
receive explicit states and notes, including obscured or uncertain evidence.

The target must unambiguously show forest before, clearing after, and clearing
on a later clear date. Later clear forest contradicts persistent loss. Seasonal
colour, fields, disturbed ground, roads, mixed pixels or ambiguous boundaries
do not establish deforestation. The whole target is judged; its four interior
pixels cannot rescue an unclear surrounding reference.

A control must show forest across its entire ROI at both endpoints and at
least one later clear date, giving at least three distinct dates, with no
clear nonforest/change contradiction. Every definite target/control claim
requires clear QA and finite RGB/NDVI across its full input margin. All 27
feature channels are proved finite for the fixed endpoint pair; later images
provide supporting visual evidence rather than new model samples.

After every control and date is reviewed, the first qualifying locked control
is retained. If the target or all controls lack defensible evidence, the case
is rejected and supplies no positive or negative validation labels.

## How to interpret the result

This is assistant image assessment in one selected validation group. It is
not independent expert annotation or representative sampling. Selecting a
visually qualifying control from a small quality-filtered roster also conditions
the sample on class appearance; it cannot establish population or clearing-area
accuracy. An independent verifier checks recorded sources, masks, quality,
complete coverage and decision rules, rather than certifying visual truth.

If supported, the last clear forest image before the first clear clearing
image gives an observed interval `(last forest, first visible clearing]`.
Intervening uncertain or obscured dates remain explicit. This is neither an
exact felling date nor a claim about earlier images outside this local window.

V4 prepares and reviews evidence only. Exported validation labels, dataset
verification, a committed label lock and frozen-checkpoint evaluation are
separate subsequent work. Test groups remain untouched.

Evidence: `Docs/Reviews/ValidationManualV4/`. Protocol:
`Docs/Reviews/ValidationManualV4/protocol.json`. Continuation:
`Docs/Handoffs/2026-11-17-validation-manual.md`. All processing uses one nice+10
CPU worker, native threads one, no GPU, the unchanged 512MiB RSS guard,
1.5GiB available-memory reserve and 300s maximum per worker.
