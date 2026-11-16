# Single-example overfit diagnostic

## Current state: predeclared experiment, implementation pending

Branch: `codex/curated-deforestation-pilot`. The user authorized the next task
after the pipeline smoke and requested tool-neutral continuity for Gemini
Antigravity if the ChatGPT usage window expires. Start with root `CONTINUE.md`.
Commit verified logical changes on this branch and update this file.

The prior milestone is complete in `9e3f110` (foundation `07e0a6d`): 173 tests,
13-example reviewed dataset, correct 255-ignore loss/metrics, deterministic
26-step tiny CPU smoke. It still predicts every reviewed pixel as clearing at
threshold 0.5. Full source, reports and weights are committed. Read the linked
prior handoffs and `Docs/Guides/PilotBaseline.md` for detailed context.

## Approved current scope and frozen plan

`Docs/Reviews/OverfitDiagnosticV1/experiment_plan.json` was written before any
new diagnostic model execution. It binds the FIRST manifest record:
`event-e691d44c335d87649310`, 25 reviewed positive / 256 forest / 3,815 ignored
pixels. Do not select a different example after seeing results.

Three sequential, fresh CPU workers will use identical initialization (seed42),
the same 2,545-parameter CNN and cached tensors, 512 attempted updates each:

1. SGD .05, global positive weight17.48344.
2. SGD .05, local positive weight10.24 (only weight differs).
3. Adam .01, local weight10.24 (optimizer AND rate differ; not an isolated
   optimizer comparison).

Checkpoints are 0/26/128/256/512, with no early stopping. The primary fit gate
is at512: reviewed precision, recall, specificity and Dice >=.95 at frozen
threshold.5, both predicted classes, exact ground-truth support. No shared
raw-BCE cutoff; record raw and weight-mass-normalized losses. A finite completed
arm with no adequate fit is a valid negative result, not a processing failure.
Record finite stationary/zero parameter deltas after convergence rather than
aborting solely because every single step is not nonzero.

## Analytic finding to test, not overstate

The previous trainer averages each image's mean reviewed-pixel loss equally;
evaluation aggregates reviewed pixels. With the global class weight, the
equal-image constant-probability optimum is0.59131, versus0.5 for the pixel
objective. On the selected case alone, global weight gives constant optimum
0.63064 and local weight gives0.5. This suggests a positive prior bias but is
not proof of the empirical cause. Prior evaluated BCE1.273797 beats the global
constant minimum1.311292, so all-positive threshold outputs do not prove that
probabilities contain no class signal. Record reviewed positive/negative
probability distributions and training-only ranking AUC without threshold tuning.

## Remaining implementation and checks

- Root will add NEW `Solutions/diagnose_pilot_overfit.py`, reusing existing
  `make_model`, loader, masked BCE and count metrics. Keep prior model helper
  and frozen QA source unchanged. Parent stays stdlib; child configures CPU
  before Torch work and uses the existing guard (512MiB/1.5GiB/nice10/threads1/
  noGPU/max300s). One arm per worker, run sequentially.
- The overfit_test_review agent owns NEW `Tests/test_pilot_overfit.py` (pure
  helper tests, AST-only until root serial execution). Check its state before
  importing; a usage-limited agent may not have finished the file.
- Pure helper APIs planned: constant_optimum(P,N,w), fit_assessment(metrics,P,N,
  minimum=.95), checkpoint_steps(limit), select_record(records,event_id=None),
  probability_diagnostics(probabilities,labels). Fit derives from counts, not
  caller-provided scores; trivial all-positive/all-negative outcomes must fail.
- Run focused/full tests under `run_pilot_checks.py`; then execute the three
  predeclared arms. Every run must retain source/plan/data hashes, exact one
  selected ID, step/checkpoint logs, counts, probabilities, safe NPZ weights,
  finite-gradient/update evidence and measured resource use. Verify reload and
  same initialization across arms, ignored-gradient safety and unchanged inputs.
- Independently recompute fit/confusion/AUC from saved arrays; inspect actual
  prediction previews. Save small results under tracked
  `Docs/Reviews/OverfitDiagnosticV1/`; local staging/run folders may remain under
  ignored `Datasets/CuratedPilot/v1/overfit_diagnostic/`. Save findings honestly,
  including any exhausted/no-fit arm; do not search settings until it passes.

## Continuity essentials

No diagnostic model run has started at this checkpoint. Dataset and model
sources are local/committed; there is no need for a new download or API account.
The exact Torch runtime is2.6.0.dev20241112 (git555bc834...), although distribution
metadata says2.5.1; preserve both identities in receipts and avoid environment
reinstallation during this diagnostic. Official2.6 contracts were checked in
the preceding task. Do not claim portability or generalization from this check.

This task does not authorize a large fit, held-out tuning, new annotations or
first-visible-date prediction. The user's Mac capacity limits remain in force.
