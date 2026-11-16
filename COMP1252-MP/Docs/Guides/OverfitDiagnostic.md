# Single-example learning diagnostic

The tiny CNN and masked training path **can fit the reviewed regions of the
fixed training crop**. All three predeclared configurations finished 512 update
attempts and achieved TP 25, FP 0, FN 0, TN 256 at the final checkpoint.
This is a memorization check on one crop, not independent detection accuracy.
It does not certify predictions on the 3,815 ignored pixels or infer dates.

The plan was committed before execution in `5aafc56`, implementation in
`74338f8`, and raw results in `c00f63e`. Source images, annotations, masks,
geographic splits, previous model code and frozen QA files were preserved.

## Fixed protocol

The first accepted manifest record was selected before observing results:
`event-e691d44c335d87649310`, 25 clearing-interior pixels and a 256-pixel forest
control. Every arm used its exact same cached 27-channel 64x64 input and
0/1/255 labels, the same fresh 2,545-parameter CNN, seed 42 and initial weights.
No different example was substituted and no threshold was tuned.

| Arm | Optimizer / rate | Positive weight | First passing recorded checkpoint | Final Dice / precision / recall / specificity |
| --- | --- | ---: | ---: | --- |
| Global weight | SGD / 0.05 | 17.48344 | 128 | 1 / 1 / 1 / 1 |
| Local weight | SGD / 0.05 | 10.24 | 128 | 1 / 1 / 1 / 1 |
| Local optimization variant | Adam / 0.01 | 10.24 | 26 | 1 / 1 / 1 / 1 |

Only checkpoints 0, 26, 128, 256 and 512 were measured. Thus the table gives
the first **recorded** passing point, not the exact update where fitting first
became adequate. All arms continued to 512; the primary decision uses the final
checkpoint, not the best earlier checkpoint.

The frozen fit gate requires reviewed precision, recall, specificity and Dice
each >=0.95 at probability threshold 0.5, exact class support and both predicted
classes. All-positive and all-negative masks cannot pass. An arm that completes
with finite state but misses the fit gate remains a valid negative diagnostic
outcome; it is not discarded as a processing failure.

The two SGD arms change only the weight choice on this case. The Adam variant
changes optimizer **and learning rate**; its earlier recorded fit cannot be
attributed solely to the optimizer. The CPU for-loop optimizer settings avoid
extra tensor-list intermediates; see the [PyTorch Adam parameter documentation](https://pytorch.org/docs/2.6/generated/torch.optim.Adam.html).

## What the result explains

At checkpoint 26, both SGD arms still predicted every reviewed pixel positive,
yet their reviewed positive-versus-negative ranking AUC was already 1.0:

| SGD arm at 26 | Mean clearing probability | Mean forest probability | Dice at 0.5 |
| --- | ---: | ---: | ---: |
| Global weight | 0.75255 | 0.58502 | 0.16340 |
| Local weight | 0.54322 | 0.50537 | 0.16340 |

There was class ranking signal before the fixed threshold separated the two
classes. At checkpoint 128 both SGD arms separated the reviewed regions
perfectly. This rules out an inability to learn this particular crop with the
current inputs, masked BCE and CNN. It is consistent with insufficient exposure
and optimization in the short smoke; it does not prove the cause of its
earlier 13-case behaviour. The old smoke saw each crop only twice, while this
diagnostic repeatedly fits one crop.

The analytic loss audit also found a reduction distinction. The old trainer
averages each image's mean reviewed-pixel loss equally, while evaluation
aggregates pixels. With the global pixel-derived class weight, the constant
probability optimum is **0.59131** for that equal-image objective and **0.5** for
the pixel objective. For the selected case alone, the global weight gives
optimum **0.63064** and the local balanced weight gives **0.5**. This indicates a
positive prior bias; it does not make eventual separation impossible, as both
SGD arms demonstrate. Choose and document the intended reduction before a
larger training comparison.

Raw weighted BCE has a different scale under the two class weights. The report
also divides it by mean weight mass `(w*P+N)/(P+N)`. This normalization aids scale
inspection but does not make the relative-class objectives identical. There was
no shared raw-loss cutoff or threshold search.

## Verification and resources

`Solutions/summarize_pilot_overfit.py` independently reconstructed all **15**
saved checkpoints using NumPy, with no Torch import or model execution. It
checked source TIFF grids/hashes, the exact selected tensor bytes, logits >=0
confusion counts, stable float64 BCE, probability distributions/AUC, fit gates,
all 1,536 step receipts and final weight-array hashes. Initial states and
initial predictions were identical across arms. Gradient execution and the
exact saved-weight forward replay are worker receipts, not re-executed claims
of the independent verifier.

All ignored-logit gradients were zero on every update. All-ignored input skipped
forward and optimizer execution. Finite stationary deltas would have been
recorded without converting a fit into a process failure; no stationary updates
occurred in these runs. Selected tensors and source files stayed unchanged.

The complete suite passes **191 tests**, including 14 new analytic/fit-gate
tests and four independent-reconstruction/resource tests. The fine-grained
`.gitattributes` rule preserves the original formatting of byte-hashed raw arm
reports; it tolerates their final blank line without altering experimental
artifacts or other whitespace checks.

| Worker | Elapsed | Peak observed memory |
| --- | ---: | ---: |
| Global-weight SGD | 3.06 s | 293.81 MiB |
| Local-weight SGD | 2.55 s | 316.75 MiB |
| Local-weight Adam | 2.80 s | 308.66 MiB |
| Independent verification | 0.51 s | 69.84 MiB |

Training was sequential and used one low-priority CPU worker, native/Torch
threads one, no GPU and no data-worker processes. The monitored limits stayed
512 MiB RSS, 1.5 GiB available memory and 300 seconds per arm. Training workers
retained at least 3.49 GiB available memory. Actual Torch runtime/distribution
identities remain recorded, including the existing nightly-versus-package
metadata discrepancy described in the baseline guide.

## Saved state and reproduction

All plans, source-bound results, previews, safe NPZ arrays/weights, test logs and
independent summary are tracked under `Docs/Reviews/OverfitDiagnosticV1/`.
Root `CONTINUE.md` and the current handoff are usable by Codex, Gemini
Antigravity or another agent without chat history.

To re-verify the committed results without training or source JP2 files:

```bash
PILOT_PYTHON=/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python
"$PILOT_PYTHON" -B Solutions/summarize_pilot_overfit.py \
  --runs-root Docs/Reviews/OverfitDiagnosticV1/runs \
  --output Datasets/CuratedPilot/v1/overfit_diagnostic/summary-recheck
```

Use a fresh output directory. If explicitly rerunning the experiment, use
`Solutions/diagnose_pilot_overfit.py --arm <arm-id> --output <new-directory>`
sequentially with the committed plan; completed outputs are not overwritten.
There is no need to rerun the arms to continue the project.

## Next work

Define the intended reviewed-pixel versus equal-image objective, then extend a
bounded training diagnostic to all 13 examples. Develop a geographically
separate reviewed validation set before model selection or accuracy claims.
Do not treat this one-crop fit, ignored regions, or training controls as a test
set. Temporal first-visible-date inference remains a later task.
