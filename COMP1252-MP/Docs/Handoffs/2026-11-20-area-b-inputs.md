# Area B endpoint inputs and control review

V21 follows the July comparison and clear 10 August persistence observation.
It audits the 27 frozen July model-input channels and fixes the control
candidate locations before their detailed visual inspection. Work on
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
peaked at 174.95 and 159.86 MiB RSS respectively. Control class disposition
is pending visual inspection of the frozen packet.

Keep one nice +10 CPU worker,
one native thread, no GPU, the existing 512 MiB cap and 1.5 GiB reserve.
