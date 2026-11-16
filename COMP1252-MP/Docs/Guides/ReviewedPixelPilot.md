# Training the reviewed-pixel pilot

The approved pilot has 13 independent training groups, 151 reviewed clearing
interior pixels and 2,640 reviewed stable-forest pixels. Another 50,457 pixels
are unknown (`255`) and contribute neither loss nor reported performance.
All features retain their exported 27-channel, 64x64 grids and signed values.

## Chosen training objective

The intended objective is the mean weighted binary cross entropy over ALL
reviewed pixels in the dataset. Use the single dataset-level positive class
weight `2640/151`, so the positive and negative classes carry equal total mass.
The constant-probability optimum of this objective is .5. Class weighting alone
does not guarantee calibrated probabilities or unbiased forest-loss area.

Let `M=13`, `N=2791`, and `n_i` be the reviewed pixel count in image `i`. The
existing masked loss is a within-image mean. For each uniformly represented
single-image step, use:

```
step_loss_i = (M * n_i / N) * mean_reviewed_weighted_BCE_i
```

At fixed model parameters, the arithmetic mean of these 13 losses equals the
dataset-wide pixel mean, as does its shared-parameter gradient. The scaling
ranges from .3540 to 1.3088 in this pilot. The eight objective tests prove loss
AND shared-parameter gradient equality using unequal counts, an independent
concatenated reference and nonfinite ignored logits. The old equal-image mean
is tested as a counterexample. Sequential SGD changes parameters between
images and is not equivalent to one full-batch update.

This is a sampling correction, separate from the class weight. Do not replace
the global class weight with image-local weights or divide training loss by
image-local class-weight mass. Report pooled weighted BCE and separately its
mean-class-weight-normalized value for readability. The normalization of the
reported score does not change the optimized objective.

New `pilot_pixel_objective.py`, `train_reviewed_pilot.py` and
`reviewed_pilot_worker.py` implement this path. Earlier code remains unchanged
so archived smoke and single-image diagnostic hashes stay valid.

## Fixed diagnostic protocol

`Docs/Reviews/PixelPilotV1/experiment_plan.json` was committed before training.
It binds the dataset, exact manifest order and all exported artifact hashes.
The executable sources are frozen in the plan before the model run.

The same 2,545-parameter CPU CNN starts fresh with seed 42. SGD uses rate .05,
zero momentum and zero weight decay. Run exactly 128 complete manifest-order
cycles, or 1,664 updates; save checkpoints 0/26/416/832/1664. There is no random
augmentation, early stopping, threshold tuning or search until success.
The threshold remains .5. Primary final fit requires precision, recall,
specificity and Dice >=.95 on each example AND pooled reviewed support, with
both predicted classes present. A finite completed no-fit outcome is evidence,
not an excuse to discard results or extend the run.

Every step checks finite gradients/state and exactly zero ignored-logit
gradients. An all-ignored safety exercise checks that forward and optimizer
execution are skipped. Safe final NPZ weights must replay every final logit
exactly. Inputs, plan and executable sources are hashed before and after.
All 13 small samples are cached; raw JP2 scenes are never read.

An independent guarded NumPy verifier reconstructs all saved checkpoint losses,
counts, per-example gates and pooled gates from logits, checks all step
receipts and source TIFFs, and imports no Torch. Gradient execution and weight
forward replay remain worker receipts, not independently repeated operations.

## Mac limits and reproduction

Use one low-priority CPU worker and native/Torch threads one, no GPU, no
DataLoader subprocesses. The unchanged guard enforces 512 MiB RSS, 1.5 GiB
available memory and a 300-second maximum. Check existing completed output
before starting a worker; outputs are atomic and never overwritten.

```bash
PILOT_PYTHON=/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python
"$PILOT_PYTHON" -B Solutions/train_reviewed_pilot.py --output Datasets/CuratedPilot/v1/pixel_pilot/run
"$PILOT_PYTHON" -B Solutions/verify_pixel_pilot.py --run Docs/Reviews/PixelPilotV1/run --output Datasets/CuratedPilot/v1/pixel_pilot/verification-recheck
```

The documented Torch runtime/distribution discrepancy persists; both identities
are recorded. Do not claim portability to another runtime without checking it.

## Interpretation and next work

This diagnostic uses reviewed TRAINING pixels only. A fitting result demonstrates
training capability, not unseen-location performance or correctness on ignored
pixels. The longer exposure and new normalization differ from the old smoke;
this is not an isolated empirical comparison of normalization effects.

Geographically separate, explicitly reviewed validation is needed before model
selection or performance claims. Keep test locations untouched. Annotation
image dates are observation dates; first-visible-date inference remains a
separate later task and cannot establish exact felling dates.

Current state and completed evidence: `Docs/Handoffs/2026-11-16-pixel-pilot.md`.
