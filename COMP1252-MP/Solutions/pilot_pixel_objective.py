"""Pixel-aggregate training objective for uniform sampling of reviewed images.

The existing masked BCE is a mean within one sampled image. Multiplying it by
``image_count * reviewed_pixels / total_reviewed_pixels`` makes the mean of
all image losses equal the dataset's reviewed-pixel mean, at fixed parameters.
This identity also holds for gradients. Sequential optimizer updates are not
equivalent to a single full-dataset update, because their parameters change.

Class weighting is independent of this sampling correction: callers must pass
the same dataset-level positive weight for every image. Ignored pixels retain
the original masked objective's semantics, including its all-ignored skip.
"""

import math


def pixel_step_scale(reviewed_pixels, image_count, total_reviewed_pixels):
    """Return the uniform-image sampling correction for one image's mean loss.

    Counts are exact Python integers; booleans and coerced floats are rejected.
    The current image may be all ignored, but the full dataset must contain at
    least one image and one reviewed pixel. No image can exceed its dataset's
    total reviewed support.
    """
    for value, name, minimum in ((reviewed_pixels, 'reviewed_pixels', 0),
                                 (image_count, 'image_count', 1),
                                 (total_reviewed_pixels, 'total_reviewed_pixels', 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if reviewed_pixels > total_reviewed_pixels:
        raise ValueError('Image reviewed support exceeds total_reviewed_pixels')
    try:
        scale = image_count*reviewed_pixels/total_reviewed_pixels
    except OverflowError as error:
        raise ValueError('Pixel sampling scale must be finite') from error
    if not math.isfinite(scale) or (reviewed_pixels > 0 and scale <= 0):
        raise ValueError('Pixel sampling scale must be finite and positive for reviewed input')
    return scale


def pixel_aggregate_step_loss(logits, labels, valid=None, pos_weight=None, *,
                              image_count, total_reviewed_pixels):
    """Scale one sampled image's masked mean BCE to target a pixel mean.

    ``image_count`` includes every uniformly sampled image, even any images
    with no reviewed pixels. This function does not choose class weights or
    normalize by weighted class mass. As in the original objective, an entirely
    ignored image returns ``None`` and must skip model/optimizer execution in
    the caller. Labels must be HxW or 1xHxW: correcting a multi-image batch as
    though it were one sampled image would change the objective. Other shape
    validation and ignored-logit gradient behavior are preserved.
    """
    from pilot_objectives import _reviewed, masked_bce_with_logits

    # Validate global counts even for an all-ignored input. The original helper
    # validates labels, shapes and the exact validity mask before counting them.
    pixel_step_scale(0, image_count, total_reviewed_pixels)
    _, _, reviewed = _reviewed(logits, labels, valid)
    if labels.ndim == 3 and labels.shape[0] != 1:
        raise ValueError('Pixel step loss requires exactly one sampled image')
    count = int(reviewed.sum().item())
    scale = pixel_step_scale(count, image_count, total_reviewed_pixels)
    loss = masked_bce_with_logits(logits, labels, valid, pos_weight)
    return None if loss is None else loss*scale
