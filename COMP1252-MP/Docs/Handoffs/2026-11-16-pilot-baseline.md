# Pilot baseline continuation

Branch: `codex/curated-deforestation-pilot`. Commit each completed, verified
logical change on this branch and update this handoff. Earlier data/provenance
state is in `2026-11-15-curated-pilot.md` in this directory.

## Current state: requested pipeline and smoke milestone complete

The user authorized an ignore-aware loader, loss/metrics and a small bounded
CPU baseline run. The foundation is saved in commit `07e0a6d`. The completion
commit adds the tiny trainer, actual smoke results, deterministic replay and
updated documentation. This is a pipeline milestone; the model is not useful
for detection yet (see findings below).

Completed code:

- `Solutions/pilot_dataset.py`: strict lazy reader of tracked 27x64x64 float32
  feature TIFFs and 0/1/255 uint8 labels. Exact bool validity, approved train
  identities, hashes, grids, channels and counts are checked. No resizing,
  extra scaling, clipping, histogram matching or augmentation. Only non-finite
  ignored inputs may be zeroed in memory, with counts; actual data needed none.
- `Solutions/pilot_objectives.py`: BCE selects reviewed logits/targets before
  computation, with optional scalar positive weighting. Count-based metrics
  aggregate reviewed pixels before ratios. Undefined metrics remain null.
  Ignored-logit gradients are exactly zero; all-ignored loss is None.
- `Solutions/run_pilot_checks.py`: fresh guarded CPU probe, unit tests or real
  dataset audit; Torch/native threads1 and lightweight parent process.
- `Solutions/train_pilot_baseline.py`: fresh tiny CPU CNN (27→8→8→1; 2,545
  parameters), counting SGD, guarded attempted-step limit, all-ignored skip
  before forward/step, finite updates, training-only evaluation, safe NPZ
  weights and actual probability preview. No old model/torchvision/GPU import.

Verification:

- Full suite **173 passed** in 1.218s; guarded test worker 2.8s / 303.88 MiB.
  New tests cover 12 loader cases, 17 loss/metrics cases and 8 training cases.
- Real audit: 13 examples, 151 positive, 2,640 stable and 50,457 ignored pixels;
  zero non-finite replacements. Dataset fingerprints/counts unchanged.
- All-ignored input skipped both model forward and optimizer step. Tests also
  warm momentum/weight decay and verify stale gradients are cleared.
- Two separate final workers each performed 26 finite, nonzero SGD updates,
  seed 42, batch size 1 / workers 0, deterministic manifest order, learning rate .05,
  positive weight 2,640/151 = 17.48344. No held-out data or threshold tuning.
- Saved weights reproduce final probabilities. Cross-process weights,
  probabilities, step log, metrics, parameter hashes and source hashes match
  exactly. Tracked `reproducibility.json` records the comparison.
- Main worker 3.05 s / 304.88 MiB high-water; replay 3.31 s / 311.69 MiB. Their
  minimum sampled available memory was 3.66 GiB. Initial earlier receipt probes
  peaked about 309 MiB; all stayed below 512 MiB, reserve 1.5 GiB, one CPU thread.
- Independent agent recomputed support/confusion/loss from saved probabilities
  and reviewed TIFFs and checked source/artifact hashes (no Torch/model/JP2).

## Actual learning findings — do not present as detection performance

Initial/final weighted BCE: 1.312841 → 1.273797. Unweighted BCE:
.746825 → .815136. **The model predicts every reviewed pixel as clearing at
threshold 0.5:** TP 151, FP 2,640, FN 0, TN 0; Dice .102651, IoU .054102. Probabilities
range about .55681–.60764. Thus finite learning updates/integration passed, but
useful classification has not emerged. Do not imply a successful deforestation
model, test accuracy or date-inference result. All-negative reviewed-pixel
accuracy is already 94.59%, which explains why accuracy alone is misleading.

The imported runtime is `2.6.0.dev20241112`, Torch git
`555bc8349d67a8f5ccd5325d1a7f46c3619b1769`, while importlib distribution metadata
says 2.5.1. Receipts retain both versions and module path; the existing environment
was not reinstalled. Official 2.6 APIs were checked and actual runtime tests pass.
Resolve this package/runtime discrepancy during later environment cleanup
before claiming portability across machines/stable releases.

## Durable state and commands

Tracked results live under `Docs/Reviews/PilotBaselineV1/`:

- `run/`: main metadata/report, steps, per-example diagnostics, preview PNG,
  predictions.npz and model_weights.npz. NPZ loading uses allow_pickle=False.
- `replay/`: independent worker results with the same numerical outcome.
- `reproducibility.json`: exact cross-process comparison and hashes.
- `checks/`: import probe, loader/objective/trainer/foundation/final test logs,
  plus the real-dataset audit and measured resources.

Guide: `Docs/Guides/PilotBaseline.md`. Dataset: tracked
`Datasets/CuratedPilot/v1/dataset/`. Original local run outputs are under
`Datasets/CuratedPilot/v1/baseline_smoke` and `baseline_smoke_replay` (ignored;
tracked snapshots above preserve them). Earlier development runs were retained
under local training_checks/initial_smoke*; final proof uses current source.
No raw imagery, QA caches, original forest models or frozen QA files changed.

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.

```bash
# Always use a NEW output directory for guarded checks/runs.
/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python -B Solutions/run_pilot_checks.py --mode tests --output Datasets/CuratedPilot/v1/training_checks/tests-recheck

/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python -B Solutions/train_pilot_baseline.py --output Datasets/CuratedPilot/v1/smoke-recheck
```

The default run is 26 attempted batches; CLI permits 1–78, fresh CPU model,
maximum 300 seconds. Resource limits remain 512 MiB RSS, 1.5 GiB available reserve,
native/Torch threads 1, nice +10, DataLoader workers 0. Do not run multiple pixel/model
workers concurrently. The original resource guard's five QA-policy files remain
frozen. A test-only sys.argv isolation fix permits its mocked watchdog test to
run inside the new guarded test CLI without confusing nested worker flags.

## Next work

The requested milestone is complete. The next recommended task is a bounded
single-example overfit diagnostic to investigate all-positive collapse, then a
larger reviewed training set with geographically separate validation examples.
Do not silently extend this smoke to a large fit or tune on the test split.

Retain 255-ignore semantics in every loss/metric/augmentation. Reviewed forest
controls remain training ROIs, not independent negative test pairs. Neighboring
input pixels legitimately influence reviewed logits through convolutional
context; it is ignored LOGIT gradients that must stay zero. The original binary
trainer remains incompatible and unmodified. This segmentation smoke does not
predict first-visible dates; those remain observed annotation/evidence bounds.
