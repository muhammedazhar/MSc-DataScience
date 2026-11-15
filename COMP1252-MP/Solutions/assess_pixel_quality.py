#!/usr/bin/env python3
"""Screen frozen crop candidates sequentially with bounded CPU and memory.

The packaged cloud classifier is used only for image-quality screening. Forest
models are not loaded or trained. Shadow/haze screens remain review flags.
"""

import argparse
from collections import Counter, OrderedDict, defaultdict
from datetime import date
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "schema_version": 1, "policy_version": "l1c-quality-screen-v1",
    "core_pixels": 64, "core_resolution_m": 10, "context_resolution_m": 20,
    "context_padding_m": 1200, "cloud_probability_threshold": .4,
    "cloud_resampling": "Mean of 2x2 samples from the aligned 10 m bilinear grid",
    "cloud_buffer_m": 60, "shadow_distance_m": 1000, "dark_nir_threshold": .15,
    "haze_blue_threshold": .15, "haze_blue_minus_red_threshold": .03,
    "crop_obscured_review_fraction": .10,
    "threshold_development": "Frozen before held-out imagery; defaults and conservative engineering review rules",
    "shadow_haze_status": "Unvalidated L1C review heuristics, not ground truth",
    "training_approved": False,
}
MASK_NAMES = ("cloud", "cloud_buffer", "shadow_risk", "haze_risk", "invalid")


def json_read(path):
    return json.loads(Path(path).read_text())


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def model_identity():
    spec = importlib.util.find_spec("s2cloudless")
    if spec is None or spec.origin is None:
        raise RuntimeError("s2cloudless is required; no clear-sky fallback is allowed")
    path = Path(spec.origin).parent / "models/pixel_s2_cloud_detector_lightGBM_v0.1.txt"
    return {"s2cloudless_version": importlib.metadata.version("s2cloudless"),
            "lightgbm_version": importlib.metadata.version("lightgbm"),
            "model_sha256": digest(path), "model_path": str(path),
            "runtime_versions": {name: importlib.metadata.version(name) for name in ("rasterio", "numpy", "scipy", "shapely", "pyproj", "affine")},
            "algorithm_sources_sha256": {name: digest(ROOT/"Solutions"/name) for name in
                ("assess_pixel_quality.py", "pixelqa_masks.py", "pixelqa_scene_metadata.py", "validate_sample.py", "annotation_scene_inventory.py")}}


def candidates(audit, include_blocked=False):
    with (Path(audit)/"candidate_pairs.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            if include_blocked or record["status"] == "candidate_for_pixel_review":
                yield record


def pair_order(pair):
    return (pair["split"], pair["group_id"], tuple(pair["grid"]["bounds"]),
            pair["pre_scene_id"], pair["post_scene_id"], pair["pair_id"])


def policy_id(identity=None):
    identity = identity or model_identity()
    return object_hash({"config": CONFIG, "model": {k: v for k, v in identity.items() if k != "model_path"}})


def choose_pilot(records, per_cell=4, seed=42):
    """Fixed, group-diverse sample; scene cloud percentage is not a selector."""
    best = {}
    for pair in records:
        source = "summer" if any("Summer" in s for s in pair.get("temporal_source_ids", [])) else "spring" if any("Spring" in s for s in pair.get("temporal_source_ids", [])) else "other"
        key = (pair["split"], pair["tile"], pair["group_id"], pair["interval_days"] <= 10, source)
        score = object_hash([seed, pair["pair_id"]])
        if key not in best or score < best[key][0]:
            best[key] = (score, pair)
    cells = defaultdict(list)
    for score, pair in best.values():
        cells[(pair["split"], pair["tile"])].append(pair)
    selected, used_groups = [], set()
    for cell in sorted(cells):
        pool = sorted(cells[cell], key=lambda p: object_hash([seed, p["pair_id"]]))
        # Alternating interval ranges exercises short and longer pairs without
        # looking at their quality outcomes or any forest-model predictions.
        for turn in range(per_cell):
            available = [p for p in pool if p["group_id"] not in used_groups]
            desired_season = "Spring" if turn % 2 == 0 else "Summer"
            seasonal = [p for p in available if any(desired_season in s for s in p.get("temporal_source_ids", []))]
            available = seasonal or available
            preferred = [p for p in available
                         if (p["interval_days"] <= 10) == (turn % 2 == 0)]
            preferred = preferred or available
            if not preferred:
                raise ValueError(f"Insufficient distinct geographic groups for pilot cell {cell}")
            pair = preferred[0]
            selected.append(pair); used_groups.add(pair["group_id"])
    return sorted(selected, key=pair_order)


def batch_records(audit, batch_index, batch_size):
    ordered = sorted((pair_order(p), p["pair_id"]) for p in candidates(audit))
    selected_ids = [identifier for _, identifier in ordered[batch_index*batch_size:(batch_index+1)*batch_size]]
    wanted = set(selected_ids)
    selected = {p["pair_id"]: p for p in candidates(audit) if p["pair_id"] in wanted}
    return [selected[key] for key in selected_ids], len(ordered)


def selected_records(args):
    if args.mode != "pilot":
        return batch_records(args.audit, args.batch_index, args.batch_size)
    records = choose_pilot(candidates(args.audit), seed=args.seed)
    aliases = {r["plot"]: r for r in map(json.loads, (args.audit/"sampled_plot_aliases.jsonl").read_text().splitlines())}
    controls = {"PLOT-00001": ("2018-07-26", "2018-07-31"),
                "PLOT-00004": ("2018-07-26", "2018-07-31"),
                "PLOT-00026": ("2018-07-31", "2018-08-05")}
    for plot, dates in controls.items():
        matches = [p for p in candidates(args.audit, True) if aliases[plot]["event_id"] in p["anchor_event_ids"]
                   and (p["pre_date"], p["post_date"]) == dates]
        if len(matches) != 1:
            raise ValueError(f"Control {plot} cannot be traced to one frozen pair")
        records.append({**matches[0], "control_plot": plot})
    return records, sum(1 for _ in candidates(args.audit))


def selection_hash(records):
    return object_hash([(p["pair_id"], p.get("control_plot")) for p in records])


def validate_source_stats(metadata):
    for path, stats in metadata.get("source_file_stats", {}).items():
        stat = (ROOT/path).stat()
        if [stat.st_size, stat.st_mtime_ns] != stats:
            raise ValueError("A source file changed since quality screening")


def context_grid(core):
    from affine import Affine

    padding = CONFIG["context_padding_m"]
    width = int(core["width"]*10/20 + 2*padding/20)
    west, _, _, north = core["bounds"]
    return {"crs": core["crs"], "width": width, "height": width,
            "transform": Affine(20, 0, west-padding, 0, -20, north+padding)}


def core_grid(raw):
    from affine import Affine

    return {**raw, "transform": Affine(*raw["transform"])}


def crop_context(array):
    """Nearest-neighbour mask/probability expansion from exact 20 m lattice."""
    import numpy as np

    halo = CONFIG["context_padding_m"]//20
    core = array[halo:halo+32, halo:halo+32]
    return np.repeat(np.repeat(core, 2, axis=0), 2, axis=1)


def read_fused_band(path, raw_grid, include_feature):
    """Decode each band once for its halo and core, keeping one file open."""
    import numpy as np
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.windows import Window, from_bounds

    core = core_grid(raw_grid)
    halo = CONFIG["context_padding_m"]//10
    width = 64+2*halo
    west, _, _, north = raw_grid["bounds"]
    transform = Affine(10, 0, west-halo*10, 0, -10, north+halo*10)
    with rasterio.open(path) as source:
        if source.crs != core["crs"]:
            raise ValueError("QA band CRS differs from the frozen crop plan")
        requested = from_bounds(west-halo*10, north-(64+halo)*10,
                                west+(64+halo)*10, north+halo*10, source.transform)
        # Read a small native window with a one-source-pixel interpolation halo.
        # Decoding once avoids repeatedly reopening the same JPEG2000 band.
        import math
        col, row = math.floor(requested.col_off)-1, math.floor(requested.row_off)-1
        window = Window(col, row, math.ceil(requested.col_off+requested.width)-col+1,
                        math.ceil(requested.row_off+requested.height)-row+1)
        if window.width > 2048 or window.height > 2048:
            raise ValueError("Native QA window exceeds its safety limit")
        native = source.read(1, window=window, boundless=True, fill_value=0)
        native_transform = source.window_transform(window)
        native_crs = source.crs
    values = np.full((width, width), np.nan, dtype="float32")
    native_valid = (np.isfinite(native) & (native > 0) & (native < 65535)).astype("uint8")
    valid_values = np.zeros((width, width), dtype="uint8")
    reproject(native, values, src_transform=native_transform, src_crs=native_crs,
              dst_transform=transform, dst_crs=core["crs"], src_nodata=0, dst_nodata=float("nan"),
              resampling=Resampling.bilinear, num_threads=1, warp_mem_limit=16)
    # Invalid categories must not be hidden by spectral interpolation. Zero is
    # a real invalid category here, not a source nodata value to skip.
    reproject(native_valid, valid_values, src_transform=native_transform, src_crs=native_crs,
              dst_transform=transform, dst_crs=core["crs"], src_nodata=None, dst_nodata=0,
              resampling=Resampling.min, num_threads=1, warp_mem_limit=16)
    valid = valid_values.astype(bool) & np.isfinite(values)
    core_values = values[halo:halo+64, halo:halo+64].copy()
    core_valid = valid[halo:halo+64, halo:halo+64].copy()
    # Every cloud-model input has an explicit validity mask; partial 2x2 blocks
    # remain unknown rather than filling from valid neighbouring pixels.
    coarse = values.reshape(width//2, 2, width//2, 2).mean(axis=(1, 3))
    coarse_valid = valid.reshape(width//2, 2, width//2, 2).all(axis=(1, 3))
    return coarse, coarse_valid, core_values, core_valid


@lru_cache(maxsize=96)
def qa_geometries(path, crs):
    from validate_sample import quality_geometries

    if Path(path).stat().st_size > 5*2**20:
        raise ValueError("QA geometry file exceeds its bounded metadata size")
    return quality_geometries(Path(path), crs)[0]


def burn_paths(paths, grid):
    from shapely.geometry import box
    from rasterio.transform import array_bounds
    from validate_sample import burn

    footprint = box(*array_bounds(grid["height"], grid["width"], grid["transform"]))
    relevant = []
    for path in paths:
        for geometry in qa_geometries(str(ROOT/path), str(grid["crs"])):
            if geometry.intersects(footprint):
                relevant.append(geometry.intersection(footprint))
    return burn(relevant, grid, all_touched=True).astype(bool)


class ObservationReader:
    """Disk-cached, small-window observations; memory retains at most 12 cores."""

    def __init__(self, cache_root, identity):
        import cv2
        from s2cloudless import S2PixelCloudDetector

        cv2.setNumThreads(1)
        self.identity = identity
        self.config_id = policy_id(identity)
        self.cache_root = Path(cache_root)/self.config_id
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.detector = S2PixelCloudDetector(threshold=.4, all_bands=False, average_over=None,
                                             dilation_size=None, model_filename=identity["model_path"])
        self.scenes, self.memory = {}, OrderedDict()
        self.hits, self.misses = 0, 0

    def metadata(self, scene_path):
        from pixelqa_scene_metadata import qa_scene_metadata

        if scene_path not in self.scenes:
            meta = qa_scene_metadata(ROOT/scene_path, ROOT)
            if not meta["metadata_usable"]:
                raise ValueError("Incomplete QA prerequisites: "+",".join(meta["issues"]))
            raster_paths = sorted(set(meta["cloud_band_paths"].values()) | set(meta["feature_band_paths"].values()))
            raster_stats = {path: [Path(ROOT/path).stat().st_size, Path(ROOT/path).stat().st_mtime_ns] for path in raster_paths}
            quality_paths = [meta["quality_paths"]["cloud"]] + [path for kind in ("nodata", "satura", "defect", "tecqua") for path in meta["quality_paths"][kind]]
            meta["cache_source_id"] = object_hash({"metadata_uid": meta["metadata_uid"], "raster_stats": raster_stats,
                                                   "quality_hashes": {path: digest(ROOT/path) for path in quality_paths}})
            other = [path for path in (ROOT/scene_path).glob("GRANULE/*/MTD_TL.xml")]
            other += [ROOT/scene_path/"MTD_MSIL1C.xml"]
            meta["source_file_stats"] = {relative(path): [path.stat().st_size, path.stat().st_mtime_ns]
                                         for path in [*(ROOT/p for p in raster_paths+quality_paths), *other]}
            self.scenes[scene_path] = meta
        return self.scenes[scene_path]

    def observation(self, scene_path, raw_grid):
        import numpy as np

        meta = self.metadata(scene_path)
        key = object_hash({"source": meta["cache_source_id"], "grid": raw_grid, "config_id": self.config_id})
        if key in self.memory:
            self.memory.move_to_end(key); self.hits += 1
            return self.memory[key]
        path = self.cache_root/(key+".npz")
        if path.exists():
            if path.stat().st_size > 2*2**20:
                raise ValueError("Unexpectedly large quality cache file")
            with np.load(path, allow_pickle=False) as data:
                record = {name: data[name] for name in data.files if name != "metadata"}
                record["metadata"] = json.loads(str(data["metadata"]))
            if record["metadata"]["cache_key"] != key or any(record[name].shape != (64, 64) for name in MASK_NAMES):
                raise ValueError("Quality cache identity or mask shape mismatch")
            self.hits += 1
        else:
            self.misses += 1
            record = self.extract(meta, raw_grid)
            record["metadata"].update(cache_key=key, config_id=self.config_id, cache_path=relative(path))
            handle, temporary = tempfile.mkstemp(prefix=".qa-", suffix=".npz", dir=self.cache_root)
            os.close(handle)
            try:
                np.savez_compressed(temporary, **{k: v for k, v in record.items() if k != "metadata"},
                                    metadata=np.array(json.dumps(record["metadata"], allow_nan=False)))
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        self.memory[key] = record
        while len(self.memory) > 12:
            self.memory.popitem(last=False)
        return record

    def extract(self, meta, raw_grid):
        import numpy as np
        from pixelqa_masks import assess_masks
        from validate_sample import compute_features

        started = time.monotonic()
        grid = core_grid(raw_grid); context = context_grid(raw_grid)
        scale = meta["quantification_value"]
        raw, feature_valid = {}, np.ones((64, 64), dtype=bool)
        cloud_inputs, context_valid = {}, np.ones((context["height"], context["width"]), dtype=bool)
        paths = {**meta["cloud_band_paths"], **meta["feature_band_paths"]}
        for band, path in paths.items():
            coarse, coarse_valid, values, valid = read_fused_band(ROOT/path, raw_grid, band in meta["feature_bands"])
            if band in meta["cloud_bands"]:
                cloud_inputs[band] = coarse/scale
                context_valid &= coarse_valid
            if band in meta["feature_bands"]:
                raw[band] = values
                feature_valid &= valid
        features = compute_features(raw, scale)
        read_finished = time.monotonic()
        # Green is needed only for RGB, already read at full 10 m resolution.
        product = burn_paths([meta["quality_paths"]["cloud"]], context)
        invalid_paths = [path for kind in ("nodata", "satura", "defect", "tecqua")
                         for path in meta["quality_paths"][kind]]
        context_invalid = ~context_valid | burn_paths(invalid_paths, context)
        core_invalid = ~feature_valid | burn_paths(invalid_paths, grid)
        qa_finished = time.monotonic()
        model_input = np.stack([cloud_inputs[band] for band in meta["cloud_bands"]], axis=-1)
        model_input[context_invalid] = 0
        probability = self.detector.get_cloud_probability_maps(model_input[None], num_threads=1)[0]
        model_finished = time.monotonic()
        probability[context_invalid] = np.nan
        masks = assess_masks(probability, product, context_invalid,
                             cloud_inputs["B08"], cloud_inputs["B02"], cloud_inputs["B04"],
                             meta["solar_azimuth_deg"], resolution_m=20,
                             cloud_threshold=CONFIG["cloud_probability_threshold"],
                             cloud_buffer_m=CONFIG["cloud_buffer_m"], shadow_distance_m=CONFIG["shadow_distance_m"],
                             dark_nir_threshold=CONFIG["dark_nir_threshold"],
                             haze_blue_threshold=CONFIG["haze_blue_threshold"],
                             haze_blue_minus_red_threshold=CONFIG["haze_blue_minus_red_threshold"])
        cropped = {name: crop_context(array).astype(bool) for name, array in masks.items()}
        cropped["invalid"] |= core_invalid
        # Unavailable padded input creates uncertainty about off-crop clouds.
        # This is recorded separately from confirmed invalid core pixels.
        record = {**cropped, "probability": crop_context(probability).astype("float32"),
                  "product_cloud": crop_context(product).astype(bool),
                  "rgb": features[[2, 1, 0]], "ndvi": features[7],
                  "metadata": {"scene_id": meta["scene_id"], "metadata_uid": meta["metadata_uid"],
                               "source_id": meta["cache_source_id"], "grid": raw_grid,
                               "context_invalid_fraction": float(context_invalid.mean()),
                               "context_incomplete": bool(context_invalid.any()),
                               "solar_azimuth_deg": meta["solar_azimuth_deg"],
                               "solar_zenith_deg": meta["solar_zenith_deg"],
                               "timings_seconds": {"read": round(read_finished-started, 4),
                                                   "product_qa": round(qa_finished-read_finished, 4),
                                                   "cloud_model": round(model_finished-qa_finished, 4),
                                                   "total": round(time.monotonic()-started, 4)}}}
        return record


def label_target(pair, events):
    from validate_sample import temporal_masks

    records = []
    for key in pair["label_event_ids_by_post_date"]:
        event = events[key]
        if not event["properties"]["temporal_source_ids"]:
            raise ValueError("A baseline-only event entered temporal target construction")
        records.append((date.fromisoformat(event["properties"]["observation_date"]), event["geometry"]))
    return temporal_masks(records, date.fromisoformat(pair["pre_date"]), date.fromisoformat(pair["post_date"]), core_grid(pair["grid"]))[2]


def classify_quality(pre, post, target, context_incomplete):
    """Quality screening never approves the temporal training label itself."""
    import numpy as np
    from pixelqa_masks import region_metrics

    masks = {name: pre[name] | post[name] for name in MASK_NAMES}
    full = region_metrics(masks, np.ones(target.shape, dtype=bool))
    change = region_metrics(masks, target.astype(bool))
    reasons = []
    if not target.any():
        return "needs_label_review", ["empty_rasterized_new_loss_target"], masks, full, change
    if change["invalid_pixels"]:
        reasons.append("invalid_target_pixels")
    if change["cloud_buffer_pixels"]:
        reasons.append("cloud_or_buffer_on_target")
    if change["shadow_risk_pixels"]:
        reasons.append("suspected_shadow_on_target")
    if change["haze_risk_pixels"]:
        reasons.append("suspected_haze_on_target")
    if full["unusable_fraction"] > CONFIG["crop_obscured_review_fraction"]:
        reasons.append("crop_quality_flags_exceed_10_percent")
    if context_incomplete:
        reasons.append("incomplete_off_crop_context")
    if any(r in reasons for r in ("invalid_target_pixels", "cloud_or_buffer_on_target")):
        status = "excluded_by_automatic_quality_screen"
    elif reasons:
        status = "needs_visual_review"
    else:
        status = "passed_automatic_screen_pending_visual_review"
    return status, reasons, masks, full, change


def render_pair(path, pre, post, target, title):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    canvas = Image.new("RGB", (1000, 755), "#101822")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((18, 12), title, fill="white", font=font)
    images = []
    for observation in (pre, post):
        rgb = np.moveaxis(observation["rgb"], 0, -1)
        images.append(Image.fromarray((np.clip(np.nan_to_num(rgb)/.30, 0, 1)*255).astype("uint8")))
    labels = ["Before", "After", "After + reference new-loss target",
              "Before + quality flags", "After + quality flags", "Target + combined quality flags"]
    colors = {"cloud_buffer": (255, 170, 35), "cloud": (255, 255, 255),
              "shadow_risk": (180, 70, 255), "haze_risk": (40, 210, 255), "invalid": (255, 30, 70)}
    for index in range(6):
        image = images[0 if index in (0, 3) else 1].resize((310, 310), Image.Resampling.NEAREST)
        if index >= 3:
            masks = pre if index == 3 else post if index == 4 else {name: pre[name] | post[name] for name in MASK_NAMES}
            for name, color in colors.items():
                mask = Image.fromarray(masks[name].astype("uint8")*255).resize(image.size, Image.Resampling.NEAREST)
                image = Image.composite(Image.blend(image, Image.new("RGB", image.size, color), .55), image, mask)
        if index in (2, 5):
            mask = Image.fromarray(target.astype("uint8")*255).resize(image.size, Image.Resampling.NEAREST)
            edge = np.asarray(mask) > np.asarray(mask.filter(ImageFilter.MinFilter(3)))
            values = np.asarray(image).copy(); values[edge] = [255, 70, 70]; image = Image.fromarray(values)
        x, y = 18+(index % 3)*328, 70+(index//3)*340
        draw.text((x, y-24), labels[index], fill="white", font=small)
        canvas.paste(image, (x, y))
    draw.text((18, 728), "640 m crops | White/orange: cloud/buffer | Purple: suspected shadow | Cyan: haze risk | Red: invalid/target outline", fill="white", font=small)
    canvas.save(path)


def assess_pair(pair, reader, events, output, preview=False):
    import numpy as np
    from pixelqa_masks import region_metrics
    from validate_sample import write_tif

    base = {"pair_id": pair["pair_id"], "group_id": pair["group_id"], "split": pair["split"], "tile": pair["tile"],
            "pre_date": pair["pre_date"], "post_date": pair["post_date"], "anchor_event_ids": pair["anchor_event_ids"],
            "training_approved": False, "visual_review": "unreviewed", "config_id": reader.config_id}
    try:
        target = label_target(pair, events)
        pre = reader.observation(pair["pre_path"], pair["grid"])
        post = reader.observation(pair["post_path"], pair["grid"])
        context_incomplete = pre["metadata"]["context_incomplete"] or post["metadata"]["context_incomplete"]
        status, reasons, masks, crop_metrics, target_metrics = classify_quality(pre, post, target, context_incomplete)
        mask_path = output / "masks" / (pair["pair_id"]+".tif")
        mask_path.parent.mkdir(exist_ok=True)
        write_tif(mask_path, np.stack([*(masks[name].astype("uint8") for name in MASK_NAMES), target]),
                  core_grid(pair["grid"]), None, (*MASK_NAMES, "reference_new_loss"))
        if preview:
            preview_path = output / "previews" / (pair["pair_id"]+".png")
            preview_path.parent.mkdir(exist_ok=True)
            render_pair(preview_path, pre, post, target, pair["split"]+" | "+pair["pair_id"]+" | "+pair["pre_date"]+" -> "+pair["post_date"])
        else:
            preview_path = None
        per_date = {}
        for side, record in (("pre", pre), ("post", post)):
            indices = record["ndvi"][target.astype(bool) & ~record["invalid"]]
            per_date[side] = {"crop": region_metrics(record, np.ones((64, 64), dtype=bool)),
                              "target": region_metrics(record, target.astype(bool)),
                              "target_median_ndvi": float(np.nanmedian(indices)) if indices.size else None,
                              "product_cloud_pixels": int(record["product_cloud"].sum()),
                              "observation": record["metadata"]}
        return {**base, "quality_status": status, "quality_reasons": reasons,
                "crop_metrics": crop_metrics, "target_metrics": target_metrics,
                "per_date": per_date, "qa_mask_file": "masks/"+mask_path.name,
                "preview_file": "previews/"+preview_path.name if preview_path else None}
    except Exception as error:
        return {**base, "quality_status": "processing_error", "quality_reasons": [type(error).__name__+": "+str(error)],
                "qa_mask_file": None, "preview_file": None}


def load_events(audit):
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform

    project = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True).transform
    return {f["id"]: {"properties": {k: f["properties"][k] for k in ("temporal_source_ids", "observation_date")},
                       "geometry": transform(project, shape(f["geometry"]))}
            for f in json_read(Path(audit)/"annotations_clean.geojson")["features"]}


def build_batch(args, output):
    import rasterio

    identity = model_identity()
    reader = ObservationReader(args.cache_root, identity)
    records, total = selected_records(args)
    if not records:
        raise ValueError("The selected batch is empty")
    events = load_events(args.audit)
    json_write(output/"config.json", {"config": CONFIG, "identity": identity, "config_id": reader.config_id})
    with (output/"selection.jsonl").open("w") as selected, (output/"results.jsonl").open("w") as results:
        counts, control_counts = Counter(), Counter()
        for index, pair in enumerate(records):
            selected.write(json.dumps(pair, separators=(",", ":"))+"\n")
            with rasterio.Env(GDAL_CACHEMAX=32*2**20, GDAL_NUM_THREADS="1"):
                result = assess_pair(pair, reader, events, output, preview=args.mode == "pilot")
            if "control_plot" in pair:
                result["control_plot"] = pair["control_plot"]
            results.write(json.dumps(result, separators=(",", ":"), allow_nan=False)+"\n"); results.flush()
            (control_counts if "control_plot" in result else counts)[result["quality_status"]] += 1
            if index % 16 == 0 or index+1 == len(records):
                print(f"[quality] {index+1}/{len(records)} pairs; fresh observations={reader.misses}; cached={reader.hits}", flush=True)
    json_write(output/"metadata.json", {
        "mode": args.mode, "batch_index": args.batch_index, "batch_size": args.batch_size,
        "pair_count": len(records), "candidate_pool_count": total,
        "status_counts": dict(counts), "control_status_counts": dict(control_counts),
        "representative_pair_count": len(records)-sum(control_counts.values()),
        "new_observations": reader.misses, "cache_hits": reader.hits,
        "config_id": reader.config_id, "audit_metadata_sha256": digest(args.audit/"metadata.json"),
        "candidate_manifest_sha256": digest(args.audit/"candidate_pairs.jsonl"),
        "annotation_catalogue_sha256": digest(args.audit/"annotations_clean.geojson"),
        "selection_hash": selection_hash(records), "seed": args.seed,
        "source_file_stats": {path: stats for meta in reader.scenes.values() for path, stats in meta["source_file_stats"].items()},
        "training_approved": False,
        "limitations": ["Automatic quality screening is not training-label approval.",
                        "Shadow and haze masks are unvalidated conservative review heuristics for these L1C inputs.",
                        "Off-crop shadow searches stop at 1 km; incomplete context is explicitly flagged.",
                        "Cloud-model false positives can include bright surface features."]})


def summarize(args, output):
    batches = sorted((args.output_base/"batches").glob("*/metadata.json"))
    if not batches:
        raise ValueError("No completed quality batches found")
    config_ids, ids, counts, split_counts, errors = set(), set(), Counter(), defaultdict(Counter), []
    resource_runs, source_stats = [], {}
    expected = {p["pair_id"]: p for p in candidates(args.audit)}
    expected_policy = policy_id()
    with (output/"quality_manifest.jsonl").open("w") as stream:
        for metadata_path in batches:
            metadata = json_read(metadata_path); resource_runs.append(metadata["resources"])
            validate_source_stats(metadata)
            for path, stats in metadata["source_file_stats"].items():
                if path in source_stats and source_stats[path] != stats:
                    raise ValueError("A source file changed between quality batches")
                source_stats[path] = stats
            if (metadata["audit_metadata_sha256"] != digest(args.audit/"metadata.json")
                    or metadata["candidate_manifest_sha256"] != digest(args.audit/"candidate_pairs.jsonl")
                    or metadata["annotation_catalogue_sha256"] != digest(args.audit/"annotations_clean.geojson")
                    or metadata["config_id"] != expected_policy):
                raise ValueError("A quality batch has a stale audit, candidate manifest or policy")
            config_ids.add(metadata["config_id"])
            for line in (metadata_path.parent/"results.jsonl").open():
                row = json.loads(line)
                if row["pair_id"] in ids:
                    raise ValueError("Duplicate pair in published quality batches")
                frozen = expected[row["pair_id"]]
                if any(row[key] != frozen[key] for key in ("split", "group_id", "pre_date", "post_date", "tile", "anchor_event_ids")):
                    raise ValueError("Quality result identities disagree with the frozen manifest")
                if row["training_approved"] or row["config_id"] != expected_policy:
                    raise ValueError("Quality result has unexpected approval or policy")
                ids.add(row["pair_id"]); counts[row["quality_status"]] += 1
                split_counts[row["split"]][row["quality_status"]] += 1
                row["batch_path"] = relative(metadata_path.parent)
                if row["quality_status"] == "processing_error":
                    errors.append({"pair_id": row["pair_id"], "reasons": row["quality_reasons"]})
                stream.write(json.dumps(row, separators=(",", ":"))+"\n")
    expected_ids = set(expected)
    if ids != expected_ids or len(config_ids) != 1:
        raise ValueError("Quality batches do not exactly cover the frozen candidate pool with one policy")
    summary = {"pair_count": len(ids), "status_counts": dict(counts),
               "status_counts_by_split": {s: dict(v) for s, v in split_counts.items()},
               "config_id": next(iter(config_ids)), "processing_errors": errors,
               "audit_metadata_sha256": digest(args.audit/"metadata.json"),
               "candidate_manifest_sha256": digest(args.audit/"candidate_pairs.jsonl"),
               "annotation_catalogue_sha256": digest(args.audit/"annotations_clean.geojson"),
               "source_file_stats": source_stats,
               "training_approved": False, "completed_batches": len(batches),
               "worker_high_water_rss_mib_max": max(r["worker_high_water_rss_mib"] for r in resource_runs),
               "worker_elapsed_seconds_total": round(sum(r["elapsed_seconds"] for r in resource_runs), 2),
               "checks": {"exact_candidate_coverage": True, "no_duplicate_pairs": True,
                          "single_frozen_quality_policy": True, "splits_preserved": True}}
    json_write(output/"metadata.json", summary)
    lines = ["# Per-crop quality screening", "", f"Attempted **{len(ids):,}** frozen candidate pairs in **{len(batches)}** sequential batches; **{len(ids)-len(errors):,}** completed screening and **{len(errors)}** have processing errors.",
             "", "| Status | Pairs |", "| --- | ---: |"]
    lines += [f"| {status} | {count} |" for status, count in sorted(counts.items())]
    lines += ["", "Masks and results preserve pair/group/split identities. Cloud probabilities combine the packaged ten-band s2cloudless model with source cloud polygons. Suspected shadow uses padded cloud projection and dark NIR; haze uses a blue-excess diagnostic. Both are unvalidated review screens for these L1C images.",
              "", "No candidate is automatically training-approved. Passing the screen still requires visual and temporal-label review. Product cloud polygons, cloud-model probabilities and shadow/haze uncertainty are retained separately in per-date results.",
              "", f"Peak worker high-water memory: **{summary['worker_high_water_rss_mib_max']} MiB**; total worker time: **{summary['worker_elapsed_seconds_total']} seconds**. Work was sequential, single-thread limited and low priority, with 512 MiB RSS monitoring and a 1.5 GiB available-memory reserve.", ""]
    (output/"quality_report.md").write_text("\n".join(lines))


def run_one(args, mode, index=None):
    from validate_sample import run_guarded

    argv = sys.argv[:]
    target = args.output_base/("pilot" if mode == "pilot" else "summary" if mode == "summarize" else f"batches/{index:05d}")
    if target.exists():
        metadata = json_read(target/"metadata.json")
        if (metadata.get("audit_metadata_sha256") != digest(args.audit/"metadata.json")
                or metadata.get("candidate_manifest_sha256") != digest(args.audit/"candidate_pairs.jsonl")
                or metadata.get("annotation_catalogue_sha256") != digest(args.audit/"annotations_clean.geojson")
                or metadata.get("config_id") != policy_id()):
            raise ValueError("Existing quality output has a different audit, manifest or policy")
        if mode != "summarize":
            planning_args = argparse.Namespace(**vars(args)); planning_args.mode = mode; planning_args.batch_index = index or 0
            records, _ = selected_records(planning_args)
            if (metadata.get("selection_hash") != selection_hash(records) or metadata.get("batch_size") != args.batch_size
                    or metadata.get("seed") != args.seed or metadata.get("mode") != mode):
                raise ValueError("Existing batch has a different frozen selection")
        validate_source_stats(metadata)
        return
    try:
        sys.argv = [argv[0], "--mode", mode, "--audit", str(args.audit), "--output-base", str(args.output_base),
                    "--cache-root", str(args.cache_root), "--batch-size", str(args.batch_size),
                    "--batch-index", str(index or 0), "--seed", str(args.seed),
                    "--max-seconds", str(args.max_seconds), "--max-rss-mib", str(args.max_rss_mib),
                    "--min-available-gib", str(args.min_available_gib)]
        args.output = target
        run_guarded(args, script_path=__file__, operation="pixel-quality", quiet=True)
        meta = json_read(target/"metadata.json")
        print(f"[quality] Published {relative(target)}; {meta.get('pair_count')} pairs; "
              f"{meta['resources']['elapsed_seconds']}s; {meta['resources']['worker_high_water_rss_mib']} MiB", flush=True)
    finally:
        sys.argv = argv


def main():
    from validate_sample import THREAD_VARIABLES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "batch", "all", "summarize"), default="pilot")
    parser.add_argument("--audit", type=Path, default=ROOT/"Datasets/AnnotationAudit/v1")
    parser.add_argument("--output-base", type=Path, default=ROOT/"Datasets/PixelQA/v1")
    parser.add_argument("--cache-root", type=Path, default=ROOT/"Datasets/PixelQA/cache")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--max-rss-mib", type=int, default=512)
    parser.add_argument("--min-available-gib", type=float, default=1.5)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (1 <= args.batch_size <= 128 and args.batch_index >= 0 and 0 < args.max_seconds <= 300
            and 128 <= args.max_rss_mib <= 768 and args.min_available_gib >= 1.5):
        parser.error("Use batches <=128, <=300 seconds, 128–768 MiB RSS and >=1.5 GiB memory reserve")
    for key in THREAD_VARIABLES:
        os.environ[key] = "1"
    if args.worker_output:
        os.nice(10)
        if args.mode == "summarize":
            summarize(args, args.worker_output)
        else:
            build_batch(args, args.worker_output)
        import resource
        metadata = json_read(args.worker_output/"metadata.json")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata["worker_high_water_rss_mib"] = round(peak/(2**20 if sys.platform == "darwin" else 1024), 1)
        json_write(args.worker_output/"metadata.json", metadata)
    elif args.mode == "all":
        count = sum(1 for _ in candidates(args.audit))
        for index in range((count+args.batch_size-1)//args.batch_size):
            run_one(args, "batch", index)
        run_one(args, "summarize")
    else:
        run_one(args, args.mode, args.batch_index)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileExistsError) as error:
        print(f"[quality] {error}", file=sys.stderr)
        sys.exit(1)
