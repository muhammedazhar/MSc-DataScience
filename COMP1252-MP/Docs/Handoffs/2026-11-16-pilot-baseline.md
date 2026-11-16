# Pilot baseline continuation

Branch: `codex/curated-deforestation-pilot`. Save each completed, verified
logical change as a commit on this branch and update this handoff. The previous
dataset/provenance handoff is `2026-11-15-curated-pilot.md` in this directory.

## Current state: training foundation complete, smoke run pending

The user authorized implementing the ignore-aware loader, loss/metrics and a
bounded CPU baseline smoke run. Current completed code:

- `Solutions/pilot_dataset.py`: strict lazy TIFF loader with unchanged float32
  27-channel inputs, uint8 0/1/255 labels, exact bool validity, hashes, identities
  and frozen-grid validation.
- `Solutions/pilot_objectives.py`: reviewed-pixel-only BCE and aggregated
  confusion counts. Ignored predictions cannot influence loss/metrics/gradients.
  All-ignored batches return None for explicit optimizer skipping.
- `Solutions/run_pilot_checks.py`: fresh guarded probe/tests/dataset audit,
  explicit CPU/thread controls. Parent remains lightweight.
- New tests: 12 loader and 17 objective tests. Full foundation suite: **165
  passed**, 0.665s tests / 2.29s guarded worker, 250.92 MiB high-water memory.
- Actual dataset audit: 13 examples, 151 positive, 2,640 stable, 50,457 ignored,
  zero non-finite replacements. No source datasets or frozen QA files changed.

The existing guard test now isolates sys.argv to avoid confusing the external
guarded test runner's worker flag with its mocked inner worker flag. This is a
test fixture fix; the frozen production resource guard is unchanged.

Recorded checks/logs: `Docs/Reviews/PilotBaselineV1/checks/`.
Usage/data semantics: `Docs/Guides/PilotBaseline.md`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Project root: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.

## Remaining work

1. Finish NEW `Solutions/train_pilot_baseline.py`: fresh tiny 27→8→8→1 CPU CNN,
   batch1/workers0, fixed seed/order, 26 attempted training steps maximum by
   default. No old forest checkpoint, torchvision, GPU or legacy helper imports.
2. `Tests/test_train_pilot_baseline.py` is assigned to the baseline review agent;
   confirm whether its file is complete before running tests. It should check
   actual finite updates, ignored-logit gradients, all-ignored skip BEFORE
   forward/optimizer step (including stale grads, momentum and weight decay),
   malformed supervision and non-finite backward gradients.
3. Root planned APIs: `make_model()`, `CountingOptimizer(params, lr=.05,
   momentum=0., weight_decay=0.)`, and
   `train_step(model, optimizer, sample, pos_weight=None)`. Skip result must have
   updated=False, loss=None, reviewed_pixels=0, gradient_norm=0, delta=0.
4. After focused tests, run one guarded CPU smoke worker under the unchanged
   512 MiB/1.5 GiB reserve/one-thread/nice10/<=300s limits. Require finite
   loss/gradients, actual parameter changes and exact data counts; do not require
   accuracy improvement as the pipeline safety gate.
5. Evaluate the same training examples once before and after. Aggregate counts
   and weighted loss sums over reviewed pixels before ratios. Record positive
   support, false alarms, precision/recall/Dice/IoU and the all-negative accuracy
   baseline (~94.59%). Label every score as training diagnostics, not held-out
   accuracy or onset-date performance.
6. Save manifest/source hashes, seed/order/visited IDs, step/update/skip counts,
   parameter checksums/deltas, actual resource measurements, predictions and a
   small safe weights artifact; inspect actual predictions in previews. Check
   deterministic replay and unchanged data. Save final receipts and commits.

## Constraints to preserve

Use one pixel/model worker, native/Torch intra/inter-op threads1, CPU only.
The import-only Torch probe passed (~183MiB sampled); Torch2.5.1 is installed.
Keep `validate_sample.run_guarded` and the five QA-policy files unchanged.
The dataset uses 255 for uncertain pixels: no binary-mask coercion, additional
1/10000 scaling, difference clipping, resizing or histogram matching.
Ignored-logit gradients must be zero; neighboring input pixels can influence
reviewed logits through convolutional context. Controls are training ROIs, not
independent negative test pairs. Stable controls and observed intervals remain
bounded by inspected data; the smoke run does not establish felling dates.
