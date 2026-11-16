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

The single predeclared training run is **COMPLETE**. All 13 examples pass the
final fit gate (pooled TP151/FP0/FN0/TN2640). First passing recorded checkpoint:
416; checkpoints 0 and 26 still predict every reviewed pixel as clearing.
All 1664 fixed updates ran, with no early stopping. Worker resources: 8.87s,
313.94 MiB peak, minimum 2.60 GiB available memory. Saved weights reproduce
all 13 final logit maps exactly. Raw outputs and hash receipts are archived in
`Docs/Reviews/PixelPilotV1/run/`. **Independent numerical verification and
preview inspection remain pending at this checkpoint. Do not rerun training.**

Implementation is complete and **206 tests pass**.
Objective equivalence (8 tests) and worker skip/scope (3 tests) also passed
separately. Evidence: `Docs/Reviews/PixelPilotV1/checks/`. Full suite used
251.5 MiB peak, 11.98 seconds, minimum 2.28 GiB available memory. The plan now
freezes all ten executable source hashes before any training. Independent
NumPy verifier and four regression tests are implemented. Old source hashes
are unchanged. Guide: `Docs/Guides/ReviewedPixelPilot.md`.

Completed: same 2545-parameter CPU CNN, seed 42, SGD
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

1. DONE: implementation and 206 tests, frozen code at `8a00e23`; training complete.
2. Run `verify_pixel_pilot.py` against the committed run and inspect all four
   preview pages. Save verification results without rerunning any model.
3. Update this handoff, guide and `CONTINUE.md`, complete verification and commit.

Next milestone after this task: geographically separate reviewed validation,
before claiming detection performance. First-visible-date inference remains later.
