# Single-example overfit diagnostic continuation

## Current state: authorized diagnostic complete

Branch: `codex/curated-deforestation-pilot`. Commit each verified logical change
on this same branch and update this handoff. Root `CONTINUE.md` is the
self-contained entry point for Codex, Gemini Antigravity or another agent.
The user explicitly requested cross-agent continuity if the ChatGPT usage
window expires. No chat history or Codex-specific memory is needed.

Milestone checkpoints: `5aafc56` predeclared plan + tool-neutral continuation;
`74338f8` tested gates/worker; `c00f63e` all three completed raw experiment arms.
The following completion commit contains the independent summary/verifier,
191-test proof, guide and this final handoff. No model job remains active and
there is no need to repeat the completed experiment.

## What was done and learned

The first manifest case was fixed BEFORE observing outcomes:
`event-e691d44c335d87649310`, pair `pair-9e46432dba83dd68c052`, group
`group-8dea78ccfd15060ff7f3`. It has 25 clearing-interior / 256 forest-control /
3,815 ignored pixels in its original 64x64, 27-channel grid. The CNN and partial
0/1/255 supervision were preserved. No held-out imagery, new annotations,
threshold tuning, raw JP2 rereads or legacy forest checkpoint was used.

Each arm used identical seed 42 initial weights/tensors and 512 fixed updates,
with checkpoints 0/26/128/256/512 and no early stopping:

| Arm | Optimizer / rate | Positive weight | First passing recorded checkpoint | Final TP / FP / FN / TN |
| --- | --- | ---: | ---: | --- |
| global_weight_sgd | SGD / .05 | 17.48344 | 128 | 25 / 0 / 0 / 256 |
| local_weight_sgd | SGD / .05 | 10.24 | 128 | 25 / 0 / 0 / 256 |
| local_weight_adam | Adam / .01 | 10.24 | 26 | 25 / 0 / 0 / 256 |

All arms passed final reviewed precision/recall/specificity/Dice>=.95 (actual
all 1.0). These are **ONE-crop training fits**, NOT independent accuracy or
predictions of first-visible dates. Exact first fitting steps are unknown
between the recorded checkpoints. The Adam variant also changes learning rate,
so it is not a causal optimizer-only comparison.

At 26 attempts both SGD arms were still all-positive at threshold 0.5, while their
reviewed ranking AUC was already 1.0. Global-SGD mean probabilities were .75255
(clearing) versus .58502 (forest); local-SGD .54322 versus .50537. By 128 both
separated the reviewed pixels. Thus the model/masked path is capable of learning
this crop; an all-positive early threshold result does not imply no class
signal. The old 13-case smoke saw each crop only twice, so this is consistent
with exposure/optimization limitations but does not prove its exact cause.

Analytic loss audit: the previous trainer weights each image's mean BCE equally,
while reporting pixel-aggregated BCE. With its global pixel-derived class weight,
the equal-image constant optimum is .59131 versus .5 for pixel aggregation.
For this selected case, global weight gives .63064 versus .5 with local weight.
This is a real objective/prior distinction, not proof of a masking bug, and it
did not prevent eventual fitting in either SGD arm. Raw BCE scales differ;
mean-weight-normalized loss is also retained, with no common raw-loss cutoff.

## Verification, resources and files

- **191 tests pass** (1.381s tests, 3.56s guarded worker). Fourteen new tests cover
  analytic controls, exact case selection, strict nontrivial gates and ignored
  probability/AUC invariance. Four independent-verifier regressions cover
  numerical reconstruction, retained no-fit outcomes and unsafe resources.
- New `Solutions/summarize_pilot_overfit.py` independently reconstructed all
  15 checkpoints from saved logits using NumPy float64 logaddexp, no Torch or
  model execution. It checked all three plans/configurations, common initial
  weights/predictions, source tensor bytes/TIFF headers/hashes, exact class
  support, metrics/AUC/fit gates, 1,536 step receipts, final parameter arrays,
  and resource limits. All checks passed in .51s at 69.84 MiB.
- Gradient execution, zero ignored-logit gradients every step, all-ignored
  no-forward/no-optimizer safety and exact saved-weight forward replay are
  worker-validated receipts, not re-executed by the independent verifier.
- No stationary parameter deltas occurred; finite zero deltas would have been
  retained as stalls rather than converting a valid fit into processing error.
- Three model workers ran sequentially for 3.06/2.55/2.80s (sum 8.41s), peaking
  293.81/316.75/308.66MiB. Sampled available memory during training stayed above
  3.49GiB. The limits remained 512 MiB worker RSS, 1.5 GiB reserve, one CPU thread,
  nice +10, no GPU, DataLoader workers 0, max 300s per arm.

Tracked durable evidence under `Docs/Reviews/OverfitDiagnosticV1/`:

- `experiment_plan.json`: declaration made before any diagnostic model run.
- `runs/<arm>/`: all checkpoint logits/probabilities/labels/validity in safe NPZ,
  final weights,512 step receipts, preview, source-bound metadata and report.
- `summary/{metadata.json,report.md}`: independently verified complete results.
- `checks/`:14 helper tests,187 pre-execution tests,4 summary tests and191 final tests,
  with exact console logs and measured guard resources.

Guide: `Docs/Guides/OverfitDiagnostic.md`. Code:
`diagnose_pilot_overfit.py` pure gates + guarded CLI;
`pilot_overfit_worker.py` model worker;
`summarize_pilot_overfit.py` independent numerical verifier/summary.
Prior dataset/pipeline handoffs remain linked in `CONTINUE.md`.

Original local runs are under ignored
`Datasets/CuratedPilot/v1/overfit_diagnostic/`. The committed snapshots above are
sufficient for verification on another worktree; the verifier binds plan
CONTENT hashes, not the original absolute plan path. The selected feature TIFFs
are already tracked in `Datasets/CuratedPilot/v1/dataset/`, so no source imagery
or new download is needed for this check. Never modify frozen QA files or
original dataset arrays. The narrow .gitattributes rule preserves byte-hashed
raw report formatting while tolerating its existing final blank line; all
other whitespace checks remain active.

Actual Torch runtime remains 2.6.0.dev20241112 (git 555bc834...), distribution
metadata 2.5.1. Both identities and module path are recorded. Environment
reinstallation was outside this diagnostic; do not claim cross-machine
reproducibility without resolving/documenting that discrepancy.

## Commands for any agent

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Git root: its parent MSc-DataScience directory. Check branch/status before edits.
Python: `/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.

```bash
PILOT_PYTHON=/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python
# Verify committed outcomes only; NEW output directory required.
"$PILOT_PYTHON" -B Solutions/summarize_pilot_overfit.py --runs-root Docs/Reviews/OverfitDiagnosticV1/runs --output Datasets/CuratedPilot/v1/overfit_diagnostic/summary-recheck
"$PILOT_PYTHON" -B Solutions/run_pilot_checks.py --mode tests --output Datasets/CuratedPilot/v1/overfit_diagnostic/checks/tests-recheck
```

All three arms are DONE; do not rerun unless explicitly required. If a future
agent is interrupted mid-experiment, inspect active Python commands and final
metadata.json outputs before resuming. Keep one pixel/model worker at a time.

## Proposed next task — not executed

Choose and document the intended objective before extending bounded training
to all 13 reviewed examples. For a pixel-aggregated objective with uniform
image sampling, multiplying each image's mean reviewed loss by
`N_images * n_reviewed_image / total_reviewed_pixels` gives the correct expected
pixel-mean gradient. If equal-image weighting is intentional, derive its class
balance consistently instead. Prove the chosen normalization with an analytic
loss/gradient equivalence test; do not silently compare different objectives.

Then predeclare a bounded 13-example training diagnostic. Develop geographically
separate reviewed validation examples before model selection/accuracy claims.
Do not use the test split for tuning, treat unknown 255 pixels as negative, or
present one-crop memorization as generalization. Temporal date inference and
larger model fitting remain later tasks. The original binary trainer is still
unmodified/incompatible with partial labels; use the new masked path.
