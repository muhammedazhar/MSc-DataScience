# Reviewed-pixel pilot continuation

## Current state: authorized milestone complete

Branch: `codex/curated-deforestation-pilot`. Keep each completed, verified
logical change committed on this same branch with an updated handoff. Root
`CONTINUE.md` is the tool-neutral entry point for Codex, Gemini Antigravity or
another agent; no chat history or Codex memory is needed.

Checkpoints: `0c0461b` predeclared all-13 experiment; `8a00e23` tested objective,
worker and verifier with frozen sources; `35aa1f5` completed raw training run.
The following completion commit contains independent verification, visual
review, guide updates and this final handoff. No model job remains active.
**Do not repeat completed training or run a larger fit to continue this task.**

## What changed and what was learned

The old step averaged each image's reviewed BCE equally, while evaluation
aggregated reviewed pixels. New modules target the dataset-wide reviewed-pixel
objective. Global positive weight remains `2640/151`; each single-image mean
loss is multiplied by `13 * reviewed_pixels_in_image / 2791`. At fixed model
parameters, averaging these image losses AND their gradients equals the full
reviewed-pixel loss/gradient. Sequential updates are not one full-batch update.
The eight objective tests prove this using unequal support, a shared parameter
and an independent concatenated reference, including nonfinite ignored logits.
All old model/objective/QA source files remain unchanged, preserving archives.

The one predeclared run used the same fresh 2545-parameter CNN, seed 42, SGD rate
.05, no momentum/weight decay, fixed manifest order, no augmentation, threshold
.5, and 128 complete cycles / 1664 updates. It never stopped early or searched
until success. Every checkpoint and outcome is retained:

| Attempt | Complete cycles | Pooled Dice | Cases passing final-style gate |
| ---: | ---: | ---: | ---: |
| 0 | 0 | .102651 | 0/13 |
| 26 | 2 | .102651 | 0/13 |
| 416 | 32 | 1.0 | 13/13 |
| 832 | 64 | 1.0 | 13/13 |
| 1664 | 128 | 1.0 | 13/13 |

Final TP151 / FP0 / FN0 / TN2640. Precision/recall/specificity/Dice are 1.0
on each example AND pooled support; the predeclared minimum was .95 with both
predicted classes. Final weighted BCE approx .000318, normalized BCE .000168.
416 is the first PASSING RECORDED checkpoint, not the exact first fitting
update. At 26 the corrected objective still predicted every reviewed pixel as
clearing. Longer exposure and changed normalization differ from the old smoke;
this does not isolate the empirical cause of its outputs.

This demonstrates that the tiny CNN and masked pipeline can learn all 13
reviewed TRAINING regions. It proves no unseen-location accuracy, complete-mask
correctness, clearing boundaries/area, probability calibration or timing skill.
The 50457 ignored pixels remain unknown. The pilot contains 13 distinct
geographic training groups, not a statistically independent accuracy sample.

## Verification, resources and durable evidence

**206 tests pass**: eight new objective tests, three worker safety/scope tests,
four independent verifier regressions plus the prior 191. Full suite took
3.708s inside an 11.98s guarded worker, peaking at 251.5 MiB with minimum 2.28 GiB
available. The all-ignored new step preserves forward/RNG/BatchNorm/established
optimizer state and malformed validity fails before forward/optimizer execution.

Training ran 8.87s, peak 313.94 MiB, minimum 2.60 GiB available. Every one of 1664
steps had finite state/gradients and exactly zero ignored-logit gradients.
No stationary parameter deltas occurred. Safe NPZ final weights reproduced
all 13 final logit maps exactly inside the worker. Inputs, artifacts, plan and
all ten executable source hashes matched before/after training.

Independent NumPy verification PASSED in .51s at 78.66 MiB. It reconstructed all 65
example checkpoints and 5 pooled summaries from logits using float64 logaddexp,
logits>=0 counts, exact support and unchanged gates. It checked all source TIFF
headers/grids/arrays/hashes, safe NPZs, 1664 step receipts, final parameter hash,
plan/source identities and resource limits. It imported no Torch and executed
no model. Gradient execution and saved-weight forward replay remain worker
receipts, not independently re-executed operations.

All four preview pages were opened: all 13 examples, reviewed labels and actual
probabilities are readable. Reviewed clearings appear in high-probability
regions and reviewed forest controls in low-probability regions. Predictions
extend beyond the reviewed ROIs; those pixels are unvalidated. No new labels,
checkpoint choice or threshold change came from viewing predictions.

Tracked under `Docs/Reviews/PixelPilotV1/`:

- `experiment_plan.json`: predeclared protocol/data and frozen source hashes.
- `run/`: five safe checkpoint NPZs, final safe weights, 1664 step records,
  source-bound metadata, four previews and raw report.
- `verification/`: independently reconstructed report and metadata.
- `checks/{objectives,worker,pre_execution_tests}/`: guarded test receipts/logs.
- `preview_review.md`: explicit limits and record of all-page visual inspection.

Guide: `Docs/Guides/ReviewedPixelPilot.md`. New code:
`pilot_pixel_objective.py`, `train_reviewed_pilot.py`,
`reviewed_pilot_worker.py`, `verify_pixel_pilot.py`.
Local working outputs are ignored under `Datasets/CuratedPilot/v1/pixel_pilot/`.
Committed evidence and the tracked dataset suffice for numerical verification;
plan content hashes do not bind the original absolute worktree location.

## Mac limits, source preservation and commands

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Git root is its parent. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Dataset: `Datasets/CuratedPilot/v1/dataset` (immutable tracked exports).
One low-priority CPU worker, native/Torch threads 1, no GPU, DataLoader workers 0.
Use the unchanged guard: 512 MiB RSS, 1.5 GiB available-memory reserve, 300s maximum.
Cache only the 13 exported 64x64 examples (~6 MiB), never raw JP2 scenes. Check
active workers before starting; never duplicate a run or overwrite outputs.

Labels: 0 reviewed forest, 1 reviewed clearing, 255 ignore. Preserve exact masking.
Frozen QA policy:
`fe2472ec030f466bebbb9dc2cb033ed6175de489bcf022f650081dd8e9347560`.
All earlier QA/model/objective files remain unchanged. Actual Torch runtime is
2.6.0.dev20241112 (git 555bc834...), distribution metadata 2.5.1; both are recorded.
Do not reinstall the environment as part of this task or claim cross-runtime
reproducibility. Prior provenance/frozen-source details are in the linked
`2026-11-16-overfit-diagnostic.md`, `2026-11-16-pilot-baseline.md` and
`2026-11-15-curated-pilot.md` handoffs.

If a verification recheck is specifically needed, use a NEW output directory:

```bash
PILOT_PYTHON=/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python
"$PILOT_PYTHON" -B Solutions/verify_pixel_pilot.py --run Docs/Reviews/PixelPilotV1/run --output Datasets/CuratedPilot/v1/pixel_pilot/verification-recheck
```

All tests, the training run, independent verification and visual review are
DONE. Do not rerun models merely because an agent/chat changed.

## Proposed next task — not executed

Prepare a SMALL reviewed validation set from the original frozen VALIDATION
geographic groups, selected with existing quality/temporal criteria BEFORE
viewing model predictions. Check full feature footprints against all 13 training
grids, retain rejections/alternates/cloud uncertainty, and leave test groups
untouched. Include stable/no-change pairs for false-alarm assessment; reviewing
known clearings alone cannot establish behaviour across unchanged locations.
Use explicit visual/temporal review evidence and retain unknown 255 pixels.

The existing loader and export intentionally enforce TRAIN-only data. Build an
explicit separate validation path; do not silently relax those checks or edit
source files bound into completed experiment receipts. Freeze validation labels,
manifest and review hashes before inference. Then evaluate the CURRENT FINAL 1664
checkpoint at threshold .5 without retraining, threshold tuning or best-checkpoint
selection. Partial ROIs support reviewed-region metrics only; stronger boundary
or area claims require denser reference annotation. First-visible-date inference
and larger model training remain later work.
