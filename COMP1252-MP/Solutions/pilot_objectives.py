"""Loss and count-based diagnostics for reviewed 0/1/255 pilot supervision.

Ignored pixels are selected out before any loss, thresholding or metric
calculation. An entirely ignored batch has no loss and must skip the optimizer
step, including weight decay and optimizer-state updates.
"""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


IGNORE_VALUE = 255


def _reviewed(logits, labels, valid=None):
    """Validate binary supervision and align only the explicit single channel.

    Labels are HxW or BxHxW. Logits have the same shape, or Bx1xHxW.
    A singleton batch dimension may also accompany unbatched HxW labels.
    No other dimension broadcasting or implicit device movement is allowed.
    """
    if not isinstance(logits, torch.Tensor) or not torch.is_floating_point(logits):
        raise ValueError("Logits must be a floating-point tensor")
    if (not isinstance(labels, torch.Tensor) or labels.layout != torch.strided
            or labels.is_complex() or labels.is_quantized or labels.dtype == torch.bool):
        raise ValueError("Labels must be numeric 0/1/255 tensors, not boolean masks")
    if logits.layout != torch.strided or labels.ndim not in (2, 3) or not all(labels.shape):
        raise ValueError("Labels must have nonempty HxW or BxHxW dimensions")
    if logits.device != labels.device:
        raise ValueError("Logits and labels must be on the same device")
    aligned = logits
    if aligned.ndim == 4:
        if aligned.shape[1] != 1:
            raise ValueError("Binary logits require exactly one channel")
        aligned = aligned[:, 0]
    if labels.ndim == 2 and aligned.ndim == 3 and aligned.shape[0] == 1:
        aligned = aligned[0]
    if aligned.shape != labels.shape:
        raise ValueError("Logits and labels must have matching spatial/batch shapes")
    if not bool(((labels == 0) | (labels == 1) | (labels == IGNORE_VALUE)).all().item()):
        raise ValueError("Labels may contain only 0, 1 and 255")
    mask = labels != IGNORE_VALUE
    if valid is not None:
        if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
                or valid.layout != torch.strided or valid.device != labels.device
                or valid.shape != labels.shape):
            raise ValueError("Validity must be a matching boolean tensor on the labels' device")
        if not torch.equal(valid, mask):
            raise ValueError("Validity must equal labels != 255 exactly")
    if not bool(torch.isfinite(aligned[mask]).all().item()):
        raise ValueError("Reviewed logits must be finite")
    return aligned, labels, mask


def _positive_weight(value, logits):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Positive weight must be a finite positive scalar")
    try:
        # Preserve a Python scalar's precision in the actual logit dtype rather
        # than first rounding it through torch's default float32 dtype.
        dtype = logits.dtype if isinstance(value, (int, float)) else None
        weight = torch.as_tensor(value, device=logits.device, dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Positive weight must be a finite positive scalar") from error
    if (weight.numel() != 1 or weight.is_complex() or weight.dtype == torch.bool
            or not bool(torch.isfinite(weight).all().item()) or float(weight.item()) <= 0):
        raise ValueError("Positive weight must be a finite positive scalar")
    weight = weight.detach().to(dtype=logits.dtype).reshape(())
    if not bool(torch.isfinite(weight).item()) or float(weight.item()) <= 0:
        raise ValueError("Positive weight is outside the logit dtype's finite positive range")
    return weight


def masked_bce_with_logits(logits, labels, valid=None, pos_weight=None):
    """Mean BCE over reviewed pixels, or None for a completely ignored batch.

    An optional scalar ``pos_weight`` weights the positive BCE term; reduction
    remains the mean over the reviewed-pixel count, matching PyTorch's BCE
    semantics. Ignored logits may be arbitrary, including NaN/inf, because they
    never enter BCE. Their gradient is exactly zero when a loss exists.
    """
    aligned, labels, mask = _reviewed(logits, labels, valid)
    weight = _positive_weight(pos_weight, aligned)
    if not bool(mask.any().item()):
        return None
    selected_logits = aligned[mask]
    selected_targets = labels[mask].to(dtype=selected_logits.dtype)
    return F.binary_cross_entropy_with_logits(
        selected_logits, selected_targets, pos_weight=weight, reduction="mean")


def _ratio(numerator, denominator):
    return numerator/denominator if denominator else None


@dataclass
class ConfusionCounts:
    """Accumulate reviewed pixels before deriving segmentation diagnostics.

    ``threshold`` is a probability strictly between zero and one. Predictions
    at the threshold are positive; the default therefore uses logits >= 0.
    Undefined ratios are None, including Dice/IoU when neither predictions nor
    labels contain a positive pixel. These are development diagnostics only.
    """

    threshold: float = .5
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    updates: int = 0
    empty_updates: int = 0
    ignored_pixels: int = 0

    def __post_init__(self):
        if (isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float))
                or not math.isfinite(self.threshold) or not 0 < self.threshold < 1):
            raise ValueError("Probability threshold must be finite and between zero and one")
        for field in ("tp", "fp", "fn", "tn", "updates", "empty_updates", "ignored_pixels"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError("Confusion counts must be nonnegative integers")

    @property
    def supervised_pixels(self):
        return self.tp+self.fp+self.fn+self.tn

    def update(self, logits, labels, valid=None):
        """Add one batch's counts and return its reviewed-pixel count."""
        aligned, labels, mask = _reviewed(logits, labels, valid)
        count = int(mask.sum().item())
        self.updates += 1
        self.ignored_pixels += int(mask.numel())-count
        if not count:
            self.empty_updates += 1
            return 0
        threshold_logit = math.log(self.threshold/(1-self.threshold))
        predicted = aligned.detach()[mask] >= threshold_logit
        actual = labels.detach()[mask] == 1
        self.tp += int((predicted & actual).sum().item())
        self.fp += int((predicted & ~actual).sum().item())
        self.fn += int((~predicted & actual).sum().item())
        self.tn += int((~predicted & ~actual).sum().item())
        return count

    def summary(self):
        precision = _ratio(self.tp, self.tp+self.fp)
        recall = _ratio(self.tp, self.tp+self.fn)
        specificity = _ratio(self.tn, self.tn+self.fp)
        overlap = _ratio(2*self.tp, 2*self.tp+self.fp+self.fn)
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "supervised_pixels": self.supervised_pixels,
            "positive_pixels": self.tp+self.fn, "stable_pixels": self.tn+self.fp,
            "ignored_pixels": self.ignored_pixels,
            "updates": self.updates, "empty_updates": self.empty_updates,
            "probability_threshold": self.threshold,
            "precision": precision, "recall": recall, "specificity": specificity,
            "f1": overlap, "dice": overlap,
            "iou": _ratio(self.tp, self.tp+self.fp+self.fn),
            "accuracy": _ratio(self.tp+self.tn, self.supervised_pixels),
            "balanced_accuracy": (recall+specificity)/2
            if recall is not None and specificity is not None else None,
        }
