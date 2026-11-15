"""Small-array cloud and suspected-shadow quality screening.

The cloud probability is supplied by the caller; this module neither imports
nor runs a model. Cloud projection intersected with dark NIR follows the general
approach in Google's official s2cloudless tutorial:
https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless

The tutorial's dark-NIR example uses surface reflectance, whereas this project's
inputs are L1C. Both the dark-NIR and simple blue-excess haze rules below are
unvalidated review screens, not cloud/shadow truth or training approval.

Azimuth is clockwise from north. For a north-up raster, a shadow opposite the
sun has (column, row) direction (-sin(azimuth), +cos(azimuth)). Context padding
is the caller's responsibility; objects outside this bounded context cannot
contribute projected shadows.
"""

import math

import numpy as np
from scipy.ndimage import binary_dilation


MASK_NAMES = ("cloud", "cloud_buffer", "shadow_risk", "haze_risk", "invalid")


def _finite_scalar(value, name, minimum=None, maximum=None):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _shift_without_wrap(mask, row_shift, column_shift):
    """Translate true pixels, dropping anything moved beyond the array edges."""
    height, width = mask.shape
    shifted = np.zeros_like(mask, dtype=bool)
    if abs(row_shift) >= height or abs(column_shift) >= width:
        return shifted
    source_rows = slice(max(0, -row_shift), min(height, height-row_shift))
    target_rows = slice(max(0, row_shift), min(height, height+row_shift))
    source_columns = slice(max(0, -column_shift), min(width, width-column_shift))
    target_columns = slice(max(0, column_shift), min(width, width+column_shift))
    shifted[target_rows, target_columns] = mask[source_rows, source_columns]
    return shifted


def assess_masks(
    probability, product_cloud, invalid, nir, blue, red, solar_azimuth_deg,
    resolution_m=20, cloud_threshold=.4, cloud_buffer_m=60,
    shadow_distance_m=1000, dark_nir_threshold=.15,
    haze_blue_threshold=.15, haze_blue_minus_red_threshold=.03,
):
    """Return five boolean masks on a padded, north-up context grid.

    ``nir``, ``blue`` and ``red`` are reflectances, not raw DNs; probability is
    on a [0, 1] scale. Cloud combines probability >= threshold with product clouds.
    Cloud buffering uses a physical disk, including the cloud itself. Shadow
    projection steps by one raster pixel in distance and rounds shifts to the
    nearest integer pixel; the final step may conservatively extend by less
    than one pixel beyond the requested distance. Shadows exclude buffered
    clouds and require dark, valid NIR. Haze risk requires blue > .15 and
    blue-red > .03. Neither risk flag establishes a physical cause.

    Non-finite spectral inputs and non-finite/out-of-range probabilities are
    invalid. Invalid locations never become cloud-free observations by filling
    input values. Output masks are separate so callers can retain reasons.
    """
    probability = np.asarray(probability, dtype=float)
    inputs = {
        "product_cloud": np.asarray(product_cloud), "invalid": np.asarray(invalid),
        "nir": np.asarray(nir, dtype=float), "blue": np.asarray(blue, dtype=float),
        "red": np.asarray(red, dtype=float),
    }
    if probability.ndim != 2 or min(probability.shape, default=0) == 0:
        raise ValueError("probability must be a nonempty two-dimensional array")
    if any(values.shape != probability.shape for values in inputs.values()):
        raise ValueError("All context arrays must have the same two-dimensional shape")

    resolution_m = _finite_scalar(resolution_m, "resolution_m", minimum=0)
    if resolution_m == 0:
        raise ValueError("resolution_m must be positive")
    threshold = _finite_scalar(cloud_threshold, "cloud_threshold", 0, 1)
    buffer_m = _finite_scalar(cloud_buffer_m, "cloud_buffer_m", 0)
    distance_m = _finite_scalar(shadow_distance_m, "shadow_distance_m", 0)
    dark_threshold = _finite_scalar(dark_nir_threshold, "dark_nir_threshold", 0)
    haze_blue = _finite_scalar(haze_blue_threshold, "haze_blue_threshold", 0)
    haze_difference = _finite_scalar(haze_blue_minus_red_threshold, "haze_blue_minus_red_threshold", 0)
    azimuth = _finite_scalar(solar_azimuth_deg, "solar_azimuth_deg", 0, 360)

    invalid_mask = inputs["invalid"].astype(bool).copy()
    invalid_mask |= ~np.isfinite(probability) | (probability < 0) | (probability > 1)
    for name in ("nir", "blue", "red"):
        invalid_mask |= ~np.isfinite(inputs[name])
    product = inputs["product_cloud"]
    if np.issubdtype(product.dtype, np.number):
        invalid_mask |= ~np.isfinite(product)
    cloud = ((probability >= threshold) | product.astype(bool)) & ~invalid_mask

    radius = buffer_m/resolution_m
    max_radius = math.ceil(math.hypot(*cloud.shape))
    # A disk larger than the image diagonal cannot reveal additional pixels;
    # clamping avoids allocating enormous kernels for accidental parameters.
    radius = min(radius, max_radius)
    if radius and cloud.any():
        extent = math.ceil(radius)
        rows, columns = np.ogrid[-extent:extent+1, -extent:extent+1]
        disk = rows*rows + columns*columns <= radius*radius
        buffered = binary_dilation(cloud, structure=disk)
    else:
        buffered = cloud.copy()
    buffered &= ~invalid_mask

    projected = np.zeros_like(cloud)
    maximum_steps = math.ceil(math.hypot(*cloud.shape))+1
    requested_steps = distance_m/resolution_m
    steps = maximum_steps if not math.isfinite(requested_steps) else min(math.ceil(requested_steps), maximum_steps)
    radians = math.radians(azimuth)
    shifts = set()
    if cloud.any():
        for step in range(1, steps+1):
            row_shift = round(math.cos(radians)*step)
            column_shift = round(-math.sin(radians)*step)
            if abs(row_shift) >= cloud.shape[0] or abs(column_shift) >= cloud.shape[1]:
                break
            if (row_shift, column_shift) not in shifts:
                projected |= _shift_without_wrap(cloud, row_shift, column_shift)
                shifts.add((row_shift, column_shift))
    shadow = projected & (inputs["nir"] < dark_threshold) & ~buffered & ~invalid_mask
    haze = (inputs["blue"] > haze_blue) & ((inputs["blue"]-inputs["red"]) > haze_difference) & ~invalid_mask

    return {"cloud": cloud, "cloud_buffer": buffered, "shadow_risk": shadow,
            "haze_risk": haze, "invalid": invalid_mask}


def region_metrics(masks, region):
    """Count quality flags inside a region with explicit, shared denominators.

    Every fraction uses all region pixels as its denominator. Fractions are
    ``None`` for an empty region. ``valid_pixels`` excludes invalid values;
    ``usable_pixels`` additionally excludes all quality/risk masks. No metric
    constitutes training approval, including a zero unusable fraction.
    """
    region = np.asarray(region, dtype=bool)
    if region.ndim != 2:
        raise ValueError("region must be two-dimensional")
    if any(name not in masks for name in MASK_NAMES):
        raise ValueError("All five quality masks are required")
    arrays = {name: np.asarray(masks[name], dtype=bool) for name in MASK_NAMES}
    if any(array.shape != region.shape for array in arrays.values()):
        raise ValueError("Quality masks and region must share the same shape")
    denominator = int(region.sum())
    counts = {name: int((mask & region).sum()) for name, mask in arrays.items()}
    union = np.logical_or.reduce(tuple(arrays.values()))
    unusable = int((union & region).sum())
    result = {"region_pixels": denominator, "denominator_pixels": denominator,
              "valid_pixels": denominator-counts["invalid"],
              "usable_pixels": denominator-unusable, "unusable_pixels": unusable}
    result.update({f"{name}_pixels": count for name, count in counts.items()})
    for name in (*MASK_NAMES, "valid", "usable", "unusable"):
        result[f"{name}_fraction"] = result[f"{name}_pixels"]/denominator if denominator else None
    return result
