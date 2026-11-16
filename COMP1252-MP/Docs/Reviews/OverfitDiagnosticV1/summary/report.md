# Verified single-example overfit diagnostic

Verified all **3** completed arms and retained every fit outcome; **3** pass the predeclared final-512 fit gate.
**This is one-crop training memorization, not held-out accuracy, deployment readiness or first-visible-date inference.**

Selected event: `event-e691d44c335d87649310`; reviewed support **25 clearing / 256 forest / 3,815 ignored pixels**. All arms share identical initial tensors, weights and predictions.

| Arm | Positive weight | Optimizer / rate | Final fit | Dice | Precision | Recall | Specificity | Normalized BCE | Training AUC | First passing checkpoint |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `global_weight_sgd` | 17.483444 | SGD / 0.05 | True | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000419 | 1.000000 | 128 |
| `local_weight_sgd` | 10.240000 | SGD / 0.05 | True | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000600 | 1.000000 | 128 |
| `local_weight_adam` | 10.240000 | Adam / 0.01 | True | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000014 | 1.000000 | 26 |

The primary result is the fixed final checkpoint. Earlier passing checkpoints are descriptive; no arm stopped early or substituted a best checkpoint.

## Predeclared checkpoint curves

| Arm | Attempt | Weighted BCE | Normalized BCE | Dice | Precision | Recall | Specificity | Training AUC | Fit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `global_weight_sgd` | 0 | 1.672901 | 0.678249 | 0.163399 | 0.088968 | 1.000000 | 0.000000 | 0.971406 | False |
| `global_weight_sgd` | 26 | 1.243983 | 0.504352 | 0.163399 | 0.088968 | 1.000000 | 0.000000 | 1.000000 | False |
| `global_weight_sgd` | 128 | 0.008631 | 0.003499 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| `global_weight_sgd` | 256 | 0.002726 | 0.001105 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| `global_weight_sgd` | 512 | 0.001033 | 0.000419 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| `local_weight_sgd` | 0 | 1.263802 | 0.693610 | 0.163399 | 0.088968 | 1.000000 | 0.000000 | 0.971406 | False |
| `local_weight_sgd` | 26 | 1.197290 | 0.657106 | 0.163399 | 0.088968 | 1.000000 | 0.000000 | 1.000000 | False |
| `local_weight_sgd` | 128 | 0.010341 | 0.005675 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| `local_weight_sgd` | 256 | 0.002988 | 0.001640 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| `local_weight_sgd` | 512 | 0.001092 | 0.000600 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| `local_weight_adam` | 0 | 1.263802 | 0.693610 | 0.163399 | 0.088968 | 1.000000 | 0.000000 | 0.971406 | False |
| `local_weight_adam` | 26 | 0.004005 | 0.002198 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| `local_weight_adam` | 128 | 0.000072 | 0.000039 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| `local_weight_adam` | 256 | 0.000046 | 0.000025 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| `local_weight_adam` | 512 | 0.000026 | 0.000014 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |

## Evidence and resources

- `global_weight_sgd`: [actual probability preview](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Docs/Reviews/OverfitDiagnosticV1/runs/global_weight_sgd/preview.png>), [worker metadata](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Docs/Reviews/OverfitDiagnosticV1/runs/global_weight_sgd/metadata.json>); peak 293.81 MiB, 3.06s, 0 stationary parameter attempts.
- `local_weight_sgd`: [actual probability preview](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Docs/Reviews/OverfitDiagnosticV1/runs/local_weight_sgd/preview.png>), [worker metadata](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Docs/Reviews/OverfitDiagnosticV1/runs/local_weight_sgd/metadata.json>); peak 316.75 MiB, 2.55s, 0 stationary parameter attempts.
- `local_weight_adam`: [actual probability preview](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Docs/Reviews/OverfitDiagnosticV1/runs/local_weight_adam/preview.png>), [worker metadata](</Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP/Docs/Reviews/OverfitDiagnosticV1/runs/local_weight_adam/metadata.json>); peak 308.66 MiB, 2.80s, 0 stationary parameter attempts.

## Interpretation

The selected-case constant-predictor optimum is 0.63064 under the global weight and 0.50000 under the local weight. The earlier full-pilot equal-image objective has optimum 0.59131, whereas its pixel-aggregated counterpart has optimum 0.5. This analytic positive-prior bias is context, not an empirical proof of the cause of the earlier all-positive outputs.
The two SGD arms change only class weight for this fixed example. The Adam arm changes optimizer **and learning rate**, so an observed difference is not an isolated optimizer effect. Checkpoint exposure on one case does not directly reproduce the earlier 13-case smoke.
BCE is independently recomputed from saved logits with stable float64 logaddexp. Counts use logits >=0, avoiding probability-rounding threshold ambiguity. Normalized BCE divides raw mean weighted BCE by its mean weight mass; no common raw-BCE cutoff is used.
All 512 step receipts, zero ignored-logit gradients and allowed stationary deltas are retained. Gradient execution and Torch saved-weight replay remain validated worker receipts; this summary executes no model and reads no JP2 pixels.
Unreviewed pixels remain ignored. Passing memorization proves neither generalization nor precise clearing dates. A finite completed no-fit arm is a valid negative result and is not removed.
