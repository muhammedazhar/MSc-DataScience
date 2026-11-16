# Reviewed-pixel pilot continuation

## Current state: approved work in progress

Branch: `codex/curated-deforestation-pilot`. The previous overfit milestone is
complete at `23b2b66`; do not repeat its three runs. This task aligns training
with the dataset-wide reviewed-pixel objective and then executes ONE fixed
all-13 training diagnostic. Read `CONTINUE.md` and `AGENTS.md`; commit every
verified logical change on this branch with this handoff updated.

The predeclared protocol is `Docs/Reviews/PixelPilotV1/experiment_plan.json`.
Use global positive weight 2640/151. For each single-image step multiply mean
reviewed BCE by `13 * reviewed_pixels_in_image / 2791`. At fixed parameters,
the mean scaled loss and gradient across all images equal the concatenated
reviewed-pixel loss and gradient. Sequential updates do not equal one batch
update. The original modules remain unchanged to preserve archived source
hashes; new code and equivalence tests go in separate modules.

Training has NOT started. Planned: same 2545-parameter CPU CNN, seed 42, SGD
rate .05, fixed manifest order, 128 cycles / 1664 updates, checkpoints
0/26/416/832/1664, fixed threshold .5 and no early stopping. Primary final fit
requires precision/recall/specificity/Dice >=.95 on every example AND aggregate.
A finite no-fit result is valid evidence. Do not extend training until it fits.
No model selection, new labels, threshold tuning or held-out evaluation.

## Limits and preservation

One low-priority CPU worker, native/Torch threads one, no GPU, DataLoader workers
zero. Existing unchanged guard: 512 MiB RSS, 1.5 GiB available-memory reserve,
300 seconds maximum per worker. Cache only the 13 exported 64x64 examples
(roughly 6 MiB), never source JP2 scenes. Check active workers before starting.
Labels 0=reviewed forest, 1=reviewed clearing, 255=unknown/ignore. Preserve exact
loss/metric masking and prove ignored logit gradients zero at every update.

Project root: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Git root is its parent. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Dataset: `Datasets/CuratedPilot/v1/dataset` (tracked, immutable).
Earlier complete state and frozen QA constraints:
`Docs/Handoffs/2026-11-16-overfit-diagnostic.md` and linked prior handoffs.
Keep all old QA/model/objective source files unchanged; do not reinstall Torch
(the documented runtime/distribution version discrepancy persists).

## Remaining work

1. Implement and guard-test the pixel objective, including shared-parameter
   gradient equivalence for unequal reviewed counts and ignored-value invariance.
2. Implement bounded worker and independent logits verifier; bind plan, sources,
   arrays, steps and final weight replay. Freeze/commit code before running.
3. Run the single predeclared diagnostic only after tests pass. Save all results,
   including no-fit outcomes, inspect previews and verify saved logits numerically.
4. Update this handoff, guide and `CONTINUE.md`, run appropriate checks and commit.

Next milestone after this task: geographically separate reviewed validation,
before claiming detection performance. First-visible-date inference remains later.
