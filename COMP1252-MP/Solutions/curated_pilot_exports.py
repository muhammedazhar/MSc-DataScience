#!/usr/bin/env python3
"""Export explicitly reviewed, partially supervised training-pilot examples.

This module is called by the pilot's guarded single worker. It never changes
the frozen annotation/quality products or reads complete satellite bands.
"""

from datetime import date
from functools import lru_cache
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANDS = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")
BASE_CHANNELS = (*BANDS, "NDVI", "NDMI")
CHANNELS = tuple(f"{period}_{name}" for period in ("pre", "post", "difference") for name in BASE_CHANNELS)


def _digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _mask(value, name):
    import numpy as np

    array = np.asarray(value)
    if array.dtype != np.dtype(bool) or array.ndim != 2 or not all(array.shape):
        raise ValueError(f"{name} must be a nonempty two-dimensional boolean mask")
    return array


def stack_pair(pre9, post9):
    """Return float32 pre9/post9/(post-pre)9 without clipping signed values."""
    import numpy as np

    pre, post = np.asarray(pre9), np.asarray(post9)
    if (pre.ndim != 3 or pre.shape[0] != 9 or not all(pre.shape)
            or post.shape != pre.shape
            or pre.dtype.kind != "f" or post.dtype.kind != "f"):
        raise ValueError("Pair features must be equally shaped floating arrays with nine channels")
    if np.isinf(pre).any() or np.isinf(post).any():
        raise ValueError("Infinite feature values are unsupported")
    pre, post = pre.astype("float32"), post.astype("float32")
    if np.isinf(pre).any() or np.isinf(post).any():
        raise ValueError("Feature values exceed float32 range")
    with np.errstate(invalid="ignore", over="ignore"):
        difference = post-pre
    if np.isinf(difference).any():
        raise ValueError("Feature differences exceed float32 range")
    return np.concatenate((pre, post, difference), axis=0)


def build_supervision(positive_bool, stable_bool, qa_unusable_bool):
    """Use 1/0 only in disjoint reviewed ROIs; every other pixel is 255.

    Review ROIs crossing unusable pixels are rejected rather than silently
    changing the accepted review extent. The caller must explicitly fix them.
    """
    import numpy as np

    positive = _mask(positive_bool, "positive")
    stable = _mask(stable_bool, "stable")
    unusable = _mask(qa_unusable_bool, "QA unusable")
    if stable.shape != positive.shape or unusable.shape != positive.shape:
        raise ValueError("Supervision and QA masks must have the same shape")
    if (positive & stable).any():
        raise ValueError("Positive and stable review ROIs overlap")
    if ((positive | stable) & unusable).any():
        raise ValueError("Reviewed supervision intersects unusable QA pixels")
    labels = np.full(positive.shape, 255, dtype="uint8")
    labels[stable] = 0
    labels[positive] = 1
    return labels


def _validate_pair(pair):
    from affine import Affine

    if pair.get("split") != "train":
        raise ValueError("This reviewed pilot exports training-split examples only")
    if not pair.get("pair_id") or not pair.get("group_id") or not pair.get("anchor_event_ids"):
        raise ValueError("Frozen pair/event/group identities are required")
    before, after = date.fromisoformat(pair["pre_date"]), date.fromisoformat(pair["post_date"])
    if not 5 <= (after-before).days <= 30:
        raise ValueError("The export pair must retain its frozen 5–30 day interval")
    grid = pair["grid"]
    if (grid["width"], grid["height"]) != (64, 64):
        raise ValueError("The pilot requires its frozen 64 by 64 grid")
    affine = Affine(*grid["transform"])
    if affine.a != 10 or affine.e != -10 or affine.b != 0 or affine.d != 0:
        raise ValueError("The pilot requires its frozen north-up 10 m grid")
    expected_bounds = [affine.c, affine.f-640, affine.c+640, affine.f]
    if list(grid["bounds"]) != expected_bounds:
        raise ValueError("Grid bounds and transform disagree")
    return {**grid, "transform": affine}


@lru_cache(maxsize=1)
def _frozen_summary():
    path = ROOT/"Datasets/PixelQA/v1/summary/metadata.json"
    summary = json.loads(path.read_text())
    if not all(summary["checks"].values()) or summary["processing_errors"]:
        raise ValueError("The source quality summary must be complete and error-free")
    return summary


def _pair_metadata(pair):
    from pixelqa_scene_metadata import qa_scene_metadata

    _validate_pair(pair)
    output = []
    for side in ("pre", "post"):
        scene_path = ROOT/pair[side+"_path"]
        metadata = qa_scene_metadata(scene_path, ROOT)
        if not metadata["metadata_usable"]:
            raise ValueError("Source QA prerequisites are incomplete")
        if metadata["scene_id"] != pair[side+"_scene_id"]:
            raise ValueError("The source scene disagrees with the frozen pair")
        if metadata["feature_bands"] != list(BANDS):
            raise ValueError("Feature band order differs from the reviewed pilot")
        output.append((scene_path, metadata))
    return output


def _pair_source_statistics(pair, metadata=None):
    metadata = metadata or _pair_metadata(pair)
    frozen = _frozen_summary()["source_file_stats"]
    output = {}
    for scene_path, meta in metadata:
        paths = set(meta["cloud_band_paths"].values()) | set(meta["feature_band_paths"].values())
        paths.add(meta["quality_paths"]["cloud"])
        for kind in ("nodata", "satura", "defect", "tecqua"):
            paths.update(meta["quality_paths"][kind])
        paths.update(str(p.relative_to(ROOT)) for p in scene_path.glob("GRANULE/*/MTD_TL.xml"))
        paths.add(str((scene_path/"MTD_MSIL1C.xml").relative_to(ROOT)))
        for path in sorted(paths):
            stat = (ROOT/path).stat()
            actual = [stat.st_size, stat.st_mtime_ns]
            if path not in frozen or actual != frozen[path]:
                raise ValueError("Source statistics changed after frozen quality screening")
            output[path] = actual
    return output


def read_pair_features(pair):
    """Read seven bounded band windows/date, preserving conservative validity."""
    import numpy as np
    import rasterio
    from assess_pixel_quality import read_fused_band
    from validate_sample import compute_features

    metadata = _pair_metadata(pair)
    before_stats = _pair_source_statistics(pair, metadata)
    features, valid = [], np.ones((64, 64), dtype=bool)
    with rasterio.Env(GDAL_CACHEMAX=32*2**20, GDAL_NUM_THREADS="1"):
        for _, meta in metadata:
            raw = {}
            scene_valid = np.ones((64, 64), dtype=bool)
            for band in BANDS:
                _, _, values, band_valid = read_fused_band(ROOT/meta["feature_band_paths"][band], pair["grid"], True)
                raw[band] = values
                scene_valid &= band_valid
            values = compute_features(raw, meta["quantification_value"])
            scene_valid &= np.isfinite(values).all(axis=0)
            features.append(values)
            valid &= scene_valid
    if before_stats != _pair_source_statistics(pair, metadata):
        raise ValueError("Source files changed during feature extraction")
    for values in features:
        values[:, ~valid] = np.nan
    return features[0], features[1], valid


def _published_qa(pair):
    """Read the small frozen QA mask rather than trusting a caller's override."""
    import numpy as np
    import rasterio

    wanted = pair["pair_id"]
    index = ROOT/"Datasets/PixelQA/v1/review/quality_result_index.jsonl"
    with index.open() as stream:
        row = next((record for record in map(json.loads, stream) if record["pair_id"] == wanted), None)
    if row is None:
        raise ValueError("The pair is absent from the frozen quality review index")
    for name in ("group_id", "split", "tile", "pre_date", "post_date", "anchor_event_ids"):
        if row[name] != pair[name]:
            raise ValueError("Frozen quality identities disagree with the export pair")
    if row["config_id"] != _frozen_summary()["config_id"]:
        raise ValueError("The quality policy changed")
    grid = _validate_pair(pair)
    with rasterio.open(ROOT/row["batch_path"]/row["qa_mask_file"]) as source:
        if (source.shape != (64, 64) or source.count != 6 or source.transform != grid["transform"]
                or source.crs != rasterio.crs.CRS.from_user_input(grid["crs"])):
            raise ValueError("Published QA mask has a different geographic grid")
        masks = source.read()
    if not np.isin(masks, (0, 1)).all():
        raise ValueError("Published QA mask is not binary")
    return np.any(masks[:5].astype(bool), axis=0), masks[5].astype(bool)


@lru_cache(maxsize=1)
def _projected_catalogue(crs):
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform

    path = ROOT/"Datasets/AnnotationAudit/v1/annotations_clean.geojson"
    if _digest(path) != _frozen_summary()["annotation_catalogue_sha256"]:
        raise ValueError("The annotation catalogue changed after quality screening")
    project = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    return tuple(transform(project, shape(feature["geometry"]))
                 for feature in json.loads(path.read_text())["features"])


def _catalogue_loss_mask(pair):
    """Exclude all known loss dates and baseline footprints from controls."""
    import math
    from shapely import make_valid
    from shapely.geometry import box
    from validate_sample import burn

    if _digest(ROOT/"Datasets/AnnotationAudit/v1/annotations_clean.geojson") != _frozen_summary()["annotation_catalogue_sha256"]:
        raise ValueError("The annotation catalogue changed after quality screening")
    grid = _validate_pair(pair)
    footprint = box(*grid["bounds"])
    geometries = []
    for geometry in _projected_catalogue(grid["crs"]):
        bounds = geometry.bounds
        if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
            raise ValueError("Catalogue geometry has no finite projected bounds")
        # Projection can introduce tiny invalidities into otherwise valid
        # source geometry. Avoid distance operations on an invalid polygon;
        # conservatively prefilter its finite bounding rectangle instead.
        if box(*bounds).distance(footprint) > 10:
            continue
        repaired = make_valid(geometry) if not geometry.is_valid else geometry
        buffered = repaired.buffer(10)
        if not buffered.is_valid:
            buffered = make_valid(buffered)
        geometries.append(buffered)
    return burn(geometries, grid, all_touched=True).astype(bool)


def export_example(output_dir, pair, positive_bool, stable_bool, qa_unusable_bool, decision):
    """Export one accepted partial-label example; output must not already exist."""
    import numpy as np
    import rasterio
    from validate_sample import write_tif

    grid = _validate_pair(pair)
    if (decision.get("visual_review") != "accepted"
            or decision.get("temporal_label_review") != "accepted"):
        raise ValueError("Explicit accepted visual and temporal-label reviews are required")
    policy = _frozen_summary()["config_id"]
    if decision.get("qa_policy_id", decision.get("config_id")) != policy:
        raise ValueError("The reviewed decision must reference the frozen QA policy")
    positive, stable = _mask(positive_bool, "positive"), _mask(stable_bool, "stable")
    unusable = _mask(qa_unusable_bool, "QA unusable")
    labels = build_supervision(positive, stable, unusable)
    if labels.shape != (64, 64) or not positive.any() or not stable.any():
        raise ValueError("The pilot needs at least one positive and stable pixel on its frozen grid")
    frozen_unusable, reference = _published_qa(pair)
    if not np.array_equal(unusable, frozen_unusable):
        raise ValueError("Caller QA differs from the frozen published masks")
    if (positive & ~reference).any():
        raise ValueError("Reviewed positives extend beyond the frozen temporal reference target")
    if (stable & reference).any():
        raise ValueError("Stable supervision intersects the interval reference loss")
    if (stable & _catalogue_loss_mask(pair)).any():
        raise ValueError("Stable supervision intersects known loss or its 10 m margin")
    stats = _pair_source_statistics(pair)
    pre, post, feature_valid = read_pair_features(pair)
    features = stack_pair(pre, post)
    supervised = labels != 255
    if (not feature_valid[supervised].all()
            or not np.isfinite(features[:, supervised]).all()):
        raise ValueError("Reviewed supervision contains missing or invalid source features")
    if stats != _pair_source_statistics(pair):
        raise ValueError("Source files changed during export")
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=False)
    write_tif(path/"input_features.tif", features, grid, float("nan"), CHANNELS)
    write_tif(path/"supervision_mask.tif", labels, grid, 255, ("reviewed_partial_supervision",))
    write_tif(path/"valid_mask.tif", supervised.astype("uint8"), grid, None, ("supervised_pixels",))
    expected = {"input_features.tif": features, "supervision_mask.tif": labels[None],
                "valid_mask.tif": supervised.astype("uint8")[None]}
    for name, values in expected.items():
        with rasterio.open(path/name) as source:
            actual = source.read()
            if (source.transform != grid["transform"] or source.shape != (64, 64)
                    or source.crs != rasterio.crs.CRS.from_user_input(grid["crs"])
                    or not np.array_equal(actual, values, equal_nan=True)):
                raise ValueError("Export round-trip changed the data or geographic grid")
    metadata = {
        "schema_version": 1, "pair_id": pair["pair_id"], "anchor_event_ids": pair["anchor_event_ids"],
        "group_id": pair["group_id"], "split": pair["split"], "tile": pair["tile"],
        "pre_date": pair["pre_date"], "post_date": pair["post_date"], "grid": pair["grid"],
        "channels": list(CHANNELS), "difference_direction": "post minus pre, without clipping",
        "scaling": "Seven reflectance bands divided by product quantification; signed NDVI and NDMI remain indices",
        "label_values": {"0": "visually reviewed stable-forest ROI", "1": "reviewed loss interior", "255": "ignored/unreviewed"},
        "valid_mask_semantics": "1 exactly where supervision is 0 or 1; not a declaration that all other pixels are invalid",
        "supervised_positive_pixels": int(positive.sum()), "supervised_stable_pixels": int(stable.sum()),
        "ignored_pixels": int((labels == 255).sum()), "pixel_area_m2": 100,
        "qa_policy_id": policy, "decision": decision, "decision_sha256": _object_digest(decision),
        "frozen_pair_sha256": _object_digest(pair), "source_file_stats": stats,
        "source_summary_sha256": _digest(ROOT/"Datasets/PixelQA/v1/summary/metadata.json"),
        "implementation_sha256": _digest(__file__),
        "output_sha256": {name: _digest(path/name) for name in expected},
        "training_approved": True,
        "approval_scope": "Small partially supervised training pilot; no held-out evaluation or exact onset-date certification",
        "checks": {"train_split_only": True, "all_grids_equal": True, "round_trip_arrays_equal": True,
                   "supervision_rois_disjoint": True, "supervised_features_finite": True,
                   "source_statistics_unchanged": True, "frozen_qa_exact": True,
                   "positives_within_temporal_reference": True, "stable_avoids_all_known_loss_plus_10m": True,
                   "unreviewed_pixels_ignored": True},
    }
    (path/"metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    return metadata
