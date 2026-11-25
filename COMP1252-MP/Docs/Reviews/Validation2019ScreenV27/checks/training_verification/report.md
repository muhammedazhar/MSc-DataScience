# Verified all-13 reviewed-pixel training diagnostic

**Training diagnostics only. No held-out accuracy or temporal inference.**

Independent verification: **passed**. Primary final fit: **True**.
Every predeclared checkpoint and finite no-fit outcome is retained.

| Attempt | Weighted BCE | Normalized BCE | Dice | Precision | Recall | Specificity | Cases passing / 13 | Overall fit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1.312841 | 0.693966 | 0.102651 | 0.054102 | 1.000000 | 0.000000 | 0 | False |
| 26 | 1.268606 | 0.670583 | 0.102651 | 0.054102 | 1.000000 | 0.000000 | 0 | False |
| 416 | 0.001754 | 0.000927 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 13 | True |
| 832 | 0.000727 | 0.000384 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 13 | True |
| 1664 | 0.000318 | 0.000168 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 13 | True |

Final per-example results:

| Event | TP / FP / FN / TN | Dice | Precision | Recall | Specificity | Training AUC | Fit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| event-e691d44c335d87649310 | 25 / 0 / 0 / 256 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| event-263e9027a6142dc9fc84 | 13 / 0 / 0 / 256 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| event-8628872c64f4cabf6e67 | 6 / 0 / 0 / 256 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| event-b9d89f91b116d8d90fac | 16 / 0 / 0 / 64 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| event-e59c75f02fe23bb9a1ae | 5 / 0 / 0 / 256 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| event-e6224bcd39746b840ea4 | 6 / 0 / 0 / 256 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| event-073d95e8a1c667845607 | 12 / 0 / 0 / 64 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| event-4fa26438d5872a8480af | 6 / 0 / 0 / 256 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| event-a72e10d66054921e60b2 | 15 / 0 / 0 / 256 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| event-f7e5f043776b24cb7dfa | 10 / 0 / 0 / 144 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| event-0012effe85aa26b69541 | 15 / 0 / 0 / 256 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| event-65fa6d8d7a0c82498340 | 18 / 0 / 0 / 64 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |
| event-e932f0d28b2819f8368d | 4 / 0 / 0 / 256 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | True |

Training worker: 313.94 MiB peak, 8.87s; minimum available memory 2.60 GiB.
The verifier reconstructs losses from logits with NumPy float64 logaddexp and uses logits >=0 for counts.
Per-image step scale is 13 times the reviewed-pixel count divided by 2791. At fixed parameters, averaging scaled image losses/gradients yields the full reviewed-pixel objective. Sequential SGD updates are not equivalent to one full-batch update.

All scores use reviewed training regions, not unseen locations or ignored pixels.
Gradient execution and saved-weight forward replay are worker receipts.
Longer exposure and changed normalization do not isolate the old smoke failure cause.
No first-visible-date inference or threshold selection was performed.
