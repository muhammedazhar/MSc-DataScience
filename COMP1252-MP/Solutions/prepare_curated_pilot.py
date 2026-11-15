#!/usr/bin/env python3
"""Prepare training-only, bounded image timelines for explicit pilot review.

Numerical ROI proposals are search aids. No output of this program grants a
visual, temporal, control-label or training approval.
"""

import argparse
from datetime import date, timedelta
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
WINDOW_DAYS = 45
MAX_TIMELINE_SCENES = 40


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def records(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def select_timeline(scenes, tile, observed, window_days=WINDOW_DAYS):
    """Include every local observation in the declared same-year window."""
    when = date.fromisoformat(observed)
    start, end = when-timedelta(days=window_days), when+timedelta(days=window_days)
    selected = [scene for scene in scenes if scene["tile"] == tile
                and start <= date.fromisoformat(scene["acquisition_date"]) <= end
                and date.fromisoformat(scene["acquisition_date"]).year == when.year]
    selected.sort(key=lambda scene: (scene["acquisition_date"], scene["scene_id"]))
    if len(selected) > MAX_TIMELINE_SCENES:
        raise ValueError("Timeline exceeds the explicit 40-scene review bound")
    return selected, {"start": start.isoformat(), "end": end.isoformat(),
                      "same_year_only": when.year, "window_days_each_side": window_days,
                      "interpretation": "Local windowed observation evidence; not a global first-clearing claim"}


def unusable(observation):
    import numpy as np
    from assess_pixel_quality import MASK_NAMES

    return np.logical_or.reduce([observation[name] for name in MASK_NAMES])


def propose_control(pre, post, excluded, sizes=(16, 12, 8)):
    """Propose a wholly clear, unannotated green ROI; visual truth is required."""
    import numpy as np

    if excluded.shape != (64, 64):
        raise ValueError("Control search must stay inside the frozen 64-pixel core")
    before, after = pre["ndvi"], post["ndvi"]
    allowed = (~excluded & ~unusable(pre) & ~unusable(post)
               & np.isfinite(before) & np.isfinite(after)
               & (before > .6) & (after > .6) & (np.abs(after-before) < .10))
    floor = np.minimum(before, after)
    for size in sizes:
        best = None
        for row in range(65-size):
            for col in range(65-size):
                view = (slice(row, row+size), slice(col, col+size))
                if not allowed[view].all():
                    continue
                score = (float(floor[view].min()), float(floor[view].mean()), -row, -col)
                if best is None or score > best[0]:
                    best = (score, row, col)
        if best is not None:
            _, row, col = best
            mask = np.zeros((64, 64), dtype=bool)
            mask[row:row+size, col:col+size] = True
            return mask, {"proposal_found": True, "row": row, "col": col,
                          "height": size, "width": size, "pixels": int(mask.sum()),
                          "minimum_endpoint_ndvi": best[0][0],
                          "mean_minimum_endpoint_ndvi": best[0][1],
                          "search_rule": "No catalogue geometry within 20 m; clear endpoints; NDVI > .6 both and absolute change < .10",
                          "control_label_review": "unreviewed", "training_approved": False}
    return np.zeros((64, 64), dtype=bool), {
        "proposal_found": False, "pixels": 0,
        "reason": "No complete 16x16, 12x12 or 8x8 region passed the numerical search aid",
        "control_label_review": "unreviewed", "training_approved": False}


def region_summary(observation, region):
    import numpy as np
    from pixelqa_masks import region_metrics

    metrics = region_metrics(observation, region)
    values = observation["ndvi"][region & ~unusable(observation)]
    values = values[np.isfinite(values)]
    metrics["usable_median_ndvi"] = float(np.median(values)) if values.size else None
    return metrics


def render_timeline(path, items, target, control, title):
    """A readable fixed-stretch contact sheet; no source pixels are altered."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    columns, panel, gap, top = 4, 200, 12, 64
    stride = panel+44
    rows = max(1, math.ceil(len(items)/columns))
    height = top+rows*stride+30
    if height > 2600:
        raise ValueError("Contact sheet exceeds the bounded render size")
    canvas = Image.new("RGB", (columns*(panel+gap)+gap, height), "#101822")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((12, 8), title, fill="white", font=font)
    draw.text((12, 32), "Red: event reference | Green: proposed stable ROI | RGB fixed reflectance 0-.30", fill="white", font=small)
    for index, item in enumerate(items):
        x, y = gap+(index % columns)*(panel+gap), top+(index//columns)*stride
        record, metadata = item["observation"], item["metadata"]
        label = metadata["date"]
        if record is None:
            image = Image.new("RGB", (panel, panel), "#555555")
            second = "unavailable: metadata"
        else:
            values = np.moveaxis(record["rgb"], 0, -1)
            image = Image.fromarray((np.clip(np.nan_to_num(values)/.30, 0, 1)*255).astype("uint8"))
            image = image.resize((panel, panel), Image.Resampling.NEAREST)
            edge_source = Image.fromarray(target.astype("uint8")*255).resize(image.size, Image.Resampling.NEAREST)
            edge = np.asarray(edge_source) > np.asarray(edge_source.filter(ImageFilter.MinFilter(3)))
            rgb = np.asarray(image).copy(); rgb[edge] = [255, 65, 65]
            image = Image.fromarray(rgb)
            if control.any():
                yy, xx = np.nonzero(control)
                box = (int(xx.min()*panel/64), int(yy.min()*panel/64),
                       int((xx.max()+1)*panel/64)-1, int((yy.max()+1)*panel/64)-1)
                ImageDraw.Draw(image).rectangle(box, outline=(80, 255, 90), width=2)
            fraction = metadata["event_metrics"]["unusable_fraction"]
            second = f"event QA {fraction:.0%}" if fraction is not None else "empty event"
            if control.any():
                fraction = metadata["control_metrics"]["unusable_fraction"]
                second += f" | ROI QA {fraction:.0%}"
        draw.text((x, y), label, fill="white", font=font)
        draw.text((x, y+21), second, fill="white", font=small)
        canvas.paste(image, (x, y+43))
    draw.text((12, height-23), "Inspection only. Atmospheric screens and NDVI are not labels or onset-date evidence by themselves.", fill="white", font=small)
    canvas.save(path)


def prepare_case(entry, pair, event, scenes, events, reader, output):
    import numpy as np
    from scipy.ndimage import binary_erosion
    from shapely.geometry import box
    from assess_pixel_quality import core_grid, label_target, render_pair
    from validate_sample import burn, write_tif

    case = output/entry["event_id"]
    case.mkdir()
    grid = core_grid(pair["grid"])
    anchor = burn([event["geometry"]], grid).astype(bool)
    target = anchor & label_target(pair, events).astype(bool)
    interior = binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
    footprint = box(*pair["grid"]["bounds"])
    excluded_geometries = [value["geometry"].buffer(20) for value in events.values()
                           if value["geometry"].distance(footprint) <= 20]
    excluded = burn(excluded_geometries, grid, all_touched=True).astype(bool)
    pre = reader.observation(pair["pre_path"], pair["grid"])
    post = reader.observation(pair["post_path"], pair["grid"])
    control, control_metadata = propose_control(pre, post, excluded)
    selected, window = select_timeline(scenes, pair["tile"], event["properties"]["observation_date"])
    items, timeline = [], []
    for scene in selected:
        record = reader.observation(scene["path"], pair["grid"]) if scene["metadata_usable"] else None
        metadata = {"date": scene["acquisition_date"], "scene_id": scene["scene_id"],
                    "scene_path": scene["path"], "metadata_usable": scene["metadata_usable"],
                    "metadata_issues": scene.get("issues", []),
                    "visual_event_state": "unreviewed", "visual_control_state": "unreviewed",
                    "training_approved": False}
        if record is not None:
            metadata.update(observation=record["metadata"],
                            event_metrics=region_summary(record, target),
                            control_metrics=region_summary(record, control),
                            crop_metrics=region_summary(record, np.ones((64, 64), dtype=bool)))
        else:
            metadata.update(event_metrics=None, control_metrics=None, crop_metrics=None)
        timeline.append(metadata)
        items.append({"observation": record, "metadata": metadata})
    if not {pair["pre_scene_id"], pair["post_scene_id"]}.issubset({row["scene_id"] for row in timeline}):
        raise ValueError("Primary endpoints are missing from the complete declared timeline")
    render_pair(case/"pair.png", pre, post, target,
                "train | "+entry["event_id"]+" | "+pair["pre_date"]+" -> "+pair["post_date"])
    render_timeline(case/"timeline.png", items, target, control,
                    entry["event_id"]+" | "+pair["tile"]+" | "+entry.get("stratum", "train"))
    for name, values in (("target_mask", target), ("positive_interior", interior), ("proposed_control", control)):
        write_tif(case/(name+".tif"), values.astype("uint8"), grid, None, (name,))
    metadata = {
        "schema_version": 1, "event_id": entry["event_id"], "pair_id": pair["pair_id"],
        "group_id": pair["group_id"], "split": "train", "stratum": entry.get("stratum"),
        "tile": pair["tile"], "annotation_observation_date": event["properties"]["observation_date"],
        "selection": entry, "pair": pair, "config_id": reader.config_id,
        "target_pixels": int(target.sum()), "positive_interior_pixels": int(interior.sum()),
        "interior_rule": "One-pixel 3x3 binary erosion of anchor-specific new-loss reference",
        "empty_interior_action": "Explicit manual pixel selection or rejection required; no boundary fallback" if not interior.any() else None,
        "control_proposal": control_metadata,
        "catalogue_exclusion_rule": "Every cleaned catalogue geometry, all dates and provenance classes, buffered 20 metres and burned all-touched",
        "timeline_window": window, "timeline": timeline,
        "primary_endpoint_cache_paths": {"pre": pre["metadata"]["cache_path"], "post": post["metadata"]["cache_path"]},
        "visual_review": "unreviewed", "temporal_label_review": "unreviewed",
        "control_label_review": "unreviewed", "training_approved": False,
        "limitations": ["Strong-signal, clear-image training pilot is not a representative benchmark.",
                        "Numerical control proposal is not verified unchanged forest.",
                        "Timeline is limited to local same-year observations within the declared 45-day window.",
                        "Annotation image date is not exact felling date; explicit visible-state review is required."]}
    write_json(case/"case.json", metadata)
    return {"event_id": entry["event_id"], "pair_id": pair["pair_id"], "case_directory": case.name,
            "timeline_scene_count": len(timeline), "target_pixels": int(target.sum()),
            "positive_interior_pixels": int(interior.sum()), "control_proposal_pixels": int(control.sum()),
            "training_approved": False}


def prepare(args, output):
    import rasterio
    from assess_pixel_quality import ObservationReader, digest, load_events, model_identity, validate_source_stats

    raw = read_json(args.selection)
    entries = raw.get("candidates") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise ValueError("Selection must be a nonempty list or candidates list")
    pair_ids = [entry["pair_id"] for entry in entries]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Duplicate selected pairs")
    wanted = set(pair_ids)
    pairs = {row["pair_id"]: row for row in records(args.audit/"candidate_pairs.jsonl") if row["pair_id"] in wanted}
    if len(pairs) != len(entries):
        raise ValueError("Selection contains an unknown frozen pair")
    groups = set()
    for entry in entries:
        pair = pairs[entry["pair_id"]]
        if pair["split"] != "train" or entry["event_id"] not in pair["anchor_event_ids"]:
            raise ValueError("Selection is not an exact training anchor/pair join")
        if pair["group_id"] in groups:
            raise ValueError("Selection reuses a geographic group")
        groups.add(pair["group_id"])
    start = args.batch_index*args.batch_size
    selected = entries[start:start+args.batch_size]
    if not selected:
        raise ValueError("Selected batch is empty")
    summary = read_json(args.quality_root/"summary/metadata.json")
    validate_source_stats(summary)
    reader = ObservationReader(args.cache_root, model_identity())
    if reader.config_id != summary["config_id"]:
        raise ValueError("Frozen quality policy has changed")
    events = load_events(args.audit)
    scenes = list(records(args.audit/"scene_inventory.jsonl"))
    results = []
    with rasterio.Env(GDAL_CACHEMAX=32*2**20, GDAL_NUM_THREADS="1"):
        for index, entry in enumerate(selected):
            result = prepare_case(entry, pairs[entry["pair_id"]], events[entry["event_id"]], scenes, events, reader, output)
            results.append(result)
            print(f"[pilot-review] {index+1}/{len(selected)} cases; observations fresh={reader.misses}, cached={reader.hits}", flush=True)
    validate_source_stats(summary)
    validate_source_stats({"source_file_stats": {key: value for scene in reader.scenes.values()
                                                 for key, value in scene["source_file_stats"].items()}})
    write_json(output/"metadata.json", {
        "schema_version": 1, "mode": "prepare_review", "batch_index": args.batch_index,
        "batch_size": args.batch_size, "selection_count": len(entries), "case_count": len(results),
        "selection_sha256": digest(args.selection), "audit_metadata_sha256": digest(args.audit/"metadata.json"),
        "candidate_manifest_sha256": digest(args.audit/"candidate_pairs.jsonl"),
        "annotation_catalogue_sha256": digest(args.audit/"annotations_clean.geojson"),
        "implementation_sha256": digest(__file__), "config_id": reader.config_id,
        "new_observations": reader.misses, "cache_hits": reader.hits, "cases": results,
        "source_file_stats": {key: value for scene in reader.scenes.values() for key, value in scene["source_file_stats"].items()},
        "training_approved": False})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=ROOT/"Datasets/AnnotationAudit/v1")
    parser.add_argument("--quality-root", type=Path, default=ROOT/"Datasets/PixelQA/v1")
    parser.add_argument("--cache-root", type=Path, default=ROOT/"Datasets/PixelQA/cache")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rss-mib", type=int, default=512)
    parser.add_argument("--min-available-gib", type=float, default=1.5)
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (0 <= args.batch_index and 1 <= args.batch_size <= 4 and 128 <= args.max_rss_mib <= 768
            and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error("Use nonnegative batch index, 1–4 cases, 128–768 MiB RSS, >=1.5 GiB reserve and <=300 seconds")
    for key in THREAD_VARIABLES:
        os.environ[key] = "1"
    if args.worker_output:
        os.nice(10)
        prepare(args, args.worker_output)
        import resource
        path = args.worker_output/"metadata.json"
        metadata = read_json(path)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata["worker_high_water_rss_mib"] = round(peak/(2**20 if sys.platform == "darwin" else 1024), 1)
        write_json(path, metadata)
    else:
        run_guarded(args, script_path=__file__, operation="pilot-review")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"[pilot-review] {error}", file=sys.stderr)
        sys.exit(1)
