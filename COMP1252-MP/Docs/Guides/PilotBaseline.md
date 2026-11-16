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
uses the number of reviewed pixels. This follows the [PyTorch 2.6 BCE-with-logits
contract](https://pytorch.org/docs/2.6/generated/torch.nn.functional.binary_cross_entropy_with_logits.html).
The mask and scalar-weight checks are enforced by the new code.

The real-dataset audit confirmed **13 examples, 151 positive pixels, 2,640
stable-forest pixels and 50,457 ignored pixels**, with zero non-finite input
replacements. The new 12 loader and 17 objective tests pass, including ignored
prediction/gradient invariance. The full foundation suite passed **165 tests**.

## Bounded checks

`Solutions/run_pilot_checks.py` runs a CPU import probe, test discovery or the
real dataset audit in a fresh, guarded worker. It sets one native thread and
one Torch intra/inter-op thread before tensor work. The thread settings follow
the [intra-op](https://pytorch.org/docs/2.6/generated/torch.set_num_threads.html)
and [inter-op](https://pytorch.org/docs/2.6/generated/torch.set_num_interop_threads.html)
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

## Completed CPU smoke run

`Solutions/train_pilot_baseline.py` constructs a fresh 2,545-parameter CNN with
27→8→8→1 convolution channels. It runs two fixed passes (26 attempted batches)
over the training manifest, batch size one and no data-worker processes. SGD
uses learning rate 0.05 and a positive BCE weight of 2,640/151 = 17.48344.
No old forest checkpoint, torchvision, augmentation or GPU is used.

The actual all-ignored path clears stale gradients and skips **both forward
and optimizer execution**. Tests exercise this with existing momentum and
weight decay. Finite reviewed loss, finite gradients and nonzero parameter
changes are required for every supervised step. Evaluation aggregates counts
and loss sums over reviewed pixels before computing ratios.

The completed run passed all 26 updates. Saved NPZ weights reproduced the
predictions exactly. A separate fresh worker with the same seed reproduced
the full step log, final weights, predictions and metrics. Dataset fingerprints
and counts stayed unchanged. The full suite now passes **173 tests**.

| Training diagnostic | Initial | Final |
| --- | ---: | ---: |
| Weighted BCE | 1.312841 | 1.273797 |
| Unweighted BCE | 0.746825 | 0.815136 |
| True positives | 151 | 151 |
| False positives | 2,640 | 2,640 |
| False negatives / true negatives | 0 / 0 | 0 / 0 |
| Dice | 0.102651 | 0.102651 |

**At threshold 0.5 the short smoke model predicts every reviewed pixel as
clearing.** It verifies pipeline execution but has not learned useful
discrimination. A lower weighted objective is not evidence of better
segmentation. An all-negative mask already scores about 94.59% reviewed-pixel
accuracy, so accuracy alone is insufficient. Every score above uses training
data; there is no independent test or first-visible-date prediction result.

The main run took **3.05 seconds**, reaching **304.88 MiB** high-water memory.
The reproducibility worker took **3.31 seconds**, reaching **311.69 MiB**.
Both retained at least **3.66 GiB** sampled available memory. The monitored
worker limit stayed 512 MiB and system reserve 1.5 GiB throughout.

Run a new bounded check with:

```bash
"$PILOT_PYTHON" -B Solutions/train_pilot_baseline.py \
  --output Datasets/CuratedPilot/v1/baseline-smoke-recheck
```

CLI limits attempted steps to 1–78 (default 26), a CPU-only fresh model, and
at most 300 seconds. Completed outputs are not overwritten. Stable snapshots
are tracked under `Docs/Reviews/PilotBaselineV1/run/` and `replay/`, with the
cross-process comparison in `reproducibility.json`. Each run includes a report,
actual prediction preview, probabilities, safe NPZ weights and per-step logs.

### Runtime identity

The imported runtime is **PyTorch 2.6.0.dev20241112**, git
`555bc8349d67a8f5ccd5325d1a7f46c3619b1769`, while its installed distribution
metadata reports **2.5.1**. This existing environment discrepancy was preserved,
recorded explicitly and tested; no package was reinstalled. Metadata retains
both versions and the actual module path. The documented 2.6 API contracts
were checked, and the tests/run receipts establish behaviour of the actual
nightly runtime. A later environment cleanup should resolve this before
claiming reproducibility across machines or stable Torch releases.

## Next training work

The [bounded single-example diagnostic](OverfitDiagnostic.md) is now complete:
all three predeclared arms fitted the reviewed regions of that crop. The
[reviewed-pixel objective and bounded all-13 fit](ReviewedPixelPilot.md) are
also complete, with independent numerical verification. Next curate
geographically separate reviewed validation examples. Do not tune thresholds on held-out imagery or
treat the current 13 training groups as a validation/test set. The original
binary training scripts remain incompatible with 255 labels; the new pilot
path supplies the required masking.
