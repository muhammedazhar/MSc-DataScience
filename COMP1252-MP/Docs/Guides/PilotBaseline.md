# Training with reviewed pilot labels

The new training path consumes the completed 13-example pilot directly. The
legacy model scripts remain separate because they do not support its partial
labels. Work stays on `codex/curated-deforestation-pilot`, with a commit and
handoff update after each verified change.

## Loader, loss and metrics

`Solutions/pilot_dataset.py` provides `ReviewedPilotDataset`. Each sample has
float32 `features` shaped `(27, 64, 64)`, uint8 `labels` shaped `(64, 64)`, a bool
`valid` mask and its event/pair/group IDs. It verifies approved training-only
manifest records, artifact hashes, channel order, geographic grids and label
counts. Existing finite features are preserved without normalization, resizing,
clipping or histogram matching. Non-finite inputs can be zeroed only at ignored
pixels, with the replacement count reported; supervised non-finite values fail.

`Solutions/pilot_objectives.py` supplies `masked_bce_with_logits` and
`ConfusionCounts`. Targets are selected **before** BCE is computed, so the
reviewed target values are only 0 or 1. Pixels labelled 255 never contribute
to loss or metrics. All-ignored batches return no loss and require skipping the
optimizer step. Precision/recall/Dice/IoU/accuracy are calculated from aggregated
reviewed-pixel counts; undefined ratios remain null.

The optional positive weight multiplies the positive BCE term; its mean still
uses the number of reviewed pixels. This follows the [PyTorch 2.5 BCE-with-logits
contract](https://pytorch.org/docs/2.5/generated/torch.nn.functional.binary_cross_entropy_with_logits.html).
The mask and scalar-weight checks are enforced by the new code.

The real-dataset audit confirmed **13 examples, 151 positive pixels, 2,640
stable-forest pixels and 50,457 ignored pixels**, with zero non-finite input
replacements. The new 12 loader and 17 objective tests pass, including ignored
prediction/gradient invariance. The full foundation suite passed **165 tests**.

## Bounded checks

`Solutions/run_pilot_checks.py` runs a CPU import probe, test discovery or the
real dataset audit in a fresh, guarded worker. It sets one native thread and
one Torch intra/inter-op thread before tensor work. The thread settings follow
the [intra-op](https://pytorch.org/docs/2.5/generated/torch.set_num_threads.html)
and [inter-op](https://pytorch.org/docs/2.5/generated/torch.set_num_interop_threads.html)
requirements. It uses no GPU, child data workers or legacy model imports.

```bash
PILOT_PYTHON=/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python
"$PILOT_PYTHON" -B Solutions/run_pilot_checks.py --mode tests \
  --output Datasets/CuratedPilot/v1/training_checks/tests-recheck
"$PILOT_PYTHON" -B Solutions/run_pilot_checks.py --mode audit \
  --output Datasets/CuratedPilot/v1/training_checks/audit-recheck
```

Use a new output path; completed outputs are not overwritten. Limits remain a
monitored 512 MiB worker budget, 1.5 GiB available-system-memory reserve, one
low-priority CPU worker and at most 300 seconds. The initial import probe used
about 183 MiB sampled RSS. The foundation test worker peaked around 251 MiB.
Receipts and logs are tracked under `Docs/Reviews/PilotBaselineV1/checks/`.

One existing watchdog test needed an isolated `sys.argv` fixture when nested
inside this guarded test runner: the outer worker flag otherwise confused the
test's mock-command assertion. The production watchdog and frozen QA policy
files were preserved.

## Current milestone

The loader, masked objective, metrics and guarded checks are complete. The
small CPU model smoke run is the next part of this milestone. Its purpose is
finite gradients and actual updates through the correct data path. These 13
training groups cannot provide an independent test score or validate prediction
of first-visible dates. An all-negative mask already scores about 94.59%
reviewed-pixel accuracy, so accuracy alone is insufficient.
