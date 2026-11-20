# Area B endpoint inputs and control review

V21 is complete. All 27 July model-input channels pass the Area B feature
audit. Three fixed control squares pass QA and appear stable and forest-like
on 6 July, 31 July and 10 August. Forest class remains provisional. Work on
`codex/curated-deforestation-pilot` and push verified checkpoints to that branch.

The fixed case is Area B, `derived-remaining-e61a8a8e623222b69efb`, one existing
exposed validation group. The 64 × 64 grid, 92-pixel target, 25-pixel eroded
interior and their full two-pixel contexts are inherited from V18–V20. Use the
frozen 6 July and 31 July source inputs, validation split and architecture.
The 10 August cache provides a third observation for control review.

The control rule reuses V4. Choose up to three whole squares at 16, 12 or 8
pixels per side, largest first and then ascending row and column. Require
each complete context to be clear and finite in the July data and disjoint
from other contexts, the target context and the 20 m all-catalogue exclusion.
Check 10 August on those fixed squares. Retain every outcome and class
uncertainty. First-visible deforestation timing and label approval remain open.

Checkpoint `1a89f14` pins the protocol and runners. All 391 existing tests
passed under the resource guard. The feature audit read 14 bounded band
windows and confirmed all 27 channels finite in all 4,096 crop pixels.
Both the 274-pixel full target context and the 104-pixel interior context
pass endpoint QA and finiteness. The three fixed 16 × 16 control squares
start at zero-based (row, column) (2, 2), (22, 2) and (42, 2). Each 400-pixel
context is QA-clear on all three dates. Independent roster replay and exact
view reproduction passed using the saved arrays. Feature audit and replay
peaked at 174.95 and 159.86 MiB RSS respectively. Checkpoint `552153b` saves
the tensor, frozen control roster and all eight boards before inspection.

Both reviewers opened the plain boards before the outlines. All three fixed
squares have continuous dark cover and show no coherent opening or advancing
patch. The third square is the smoothest and has the weakest visible texture
to identify the class. `visual_review.md` and `reviews.json` record each
provisional control candidate, the three dates and limitations, under
`Docs/Reviews/ValidationAreaBInputsV21/`. Final integrity verifies the
archived evidence, resources, source and replay bindings. No workers are active.

Preserve the V19 expansion bracket and V20 10 August persistence observation.
An independent class review should corroborate the earlier cover of the lower
expansion and at least one complete stable-forest square before label approval.
The present case is an already visually exposed location. Keep the existing
split barriers and approval/label-lock gates for any model evaluation.

Keep one nice +10 CPU worker,
one native thread, no GPU, the existing 512 MiB cap and 1.5 GiB reserve.
