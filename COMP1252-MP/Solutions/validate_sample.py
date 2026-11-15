#!/usr/bin/env python3
"""Build one georeferenced image pair under explicit Mac resource limits.

This deliberately does not import the existing helper/training modules, read
whole Sentinel tiles, train a model, or modify the original datasets.
"""

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
BANDS = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")
CHANNELS = (*BANDS, "NDVI", "NDMI")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "GDAL_NUM_THREADS",
    "OPJ_NUM_THREADS",
)
GML = {"gml": "http://www.opengis.net/gml/3.2"}


def read_json(path):
    return json.loads(Path(path).read_text())


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def digest(path):
    # Annotation files are small; satellite image files are never hashed/read whole.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_scene(sentinel_dir, tile, observation_date):
    stamp = observation_date.strftime("%Y%m%d")
    matches = sorted(
        safe for product in Path(sentinel_dir).glob(f"S2*_MSIL1C_{stamp}T*_T{tile}_*")
        for safe in product.glob("*.SAFE")
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one L1C scene for {tile}/{stamp}, found {len(matches)}")
    return matches[0]


def band_path(scene, band):
    matches = list(scene.glob(f"GRANULE/*/IMG_DATA/*_{band}.jp2"))
    if len(matches) != 1:
        raise ValueError(f"Expected one {band} band in {scene.name}")
    return matches[0]


def scene_scale(scene):
    root = ET.parse(scene / "MTD_MSIL1C.xml").getroot()
    values = [float(e.text) for e in root.iter()
              if e.tag.split("}")[-1] == "QUANTIFICATION_VALUE"]
    offsets = [float(e.text) for e in root.iter()
               if e.tag.split("}")[-1] == "RADIO_ADD_OFFSET"]
    if len(values) != 1 or values[0] <= 0 or any(offsets):
        raise ValueError("This validator supports positive quantification with zero L1C offsets only")
    return values[0]


def polygonal(geometry):
    """Repair invalid QA geometry, retaining polygonal components only."""
    from shapely import make_valid
    from shapely.ops import unary_union

    geometry = make_valid(geometry) if not geometry.is_valid else geometry
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return geometry
    pieces = [polygonal(g) for g in getattr(geometry, "geoms", [])]
    return unary_union([g for g in pieces if not g.is_empty])


def quality_geometries(path, target_crs):
    """Read a small Sentinel GML QA file, preserving polygon holes and CRS."""
    from pyproj import CRS, Transformer
    from shapely.geometry import Polygon
    from shapely.ops import transform

    root = ET.parse(path).getroot()
    nodes = root.findall(".//gml:Polygon", GML)
    # Sentinel explicitly represents an empty QA mask without an Envelope/CRS.
    # A nonempty but unsupported geometry must not silently become a clear mask.
    if not nodes:
        if any(e.tag.split("}")[-1] in ("MaskFeature", "Polygon", "MultiSurface", "Surface") for e in root.iter()):
            raise ValueError(f"Unsupported nonempty QA geometry: {path}")
        return [], 0
    names = {e.attrib["srsName"] for e in root.iter() if "srsName" in e.attrib}
    if len(names) != 1:
        raise ValueError(f"Missing or ambiguous QA CRS: {path}")
    source_crs = CRS.from_user_input(next(iter(names)))
    project = Transformer.from_crs(source_crs, target_crs, always_xy=True).transform
    output, repaired = [], 0

    def ring(element):
        if element is None:
            raise ValueError(f"GML polygon has no coordinate ring: {path}")
        dimension = int(element.attrib.get("srsDimension", "2"))
        values = [float(v) for v in element.text.split()]
        if dimension not in (2, 3) or len(values) % dimension:
            raise ValueError(f"Unsupported GML coordinates: {path}")
        # Some source QA rings include altitude as their third ordinate.
        return list(zip(values[::dimension], values[1::dimension]))

    for node in nodes:
        shell = ring(node.find("gml:exterior/gml:LinearRing/gml:posList", GML))
        holes = [ring(e) for e in node.findall("gml:interior/gml:LinearRing/gml:posList", GML)]
        geometry = Polygon(shell, holes)
        repaired += int(not geometry.is_valid)
        geometry = polygonal(geometry)
        if not geometry.is_empty:
            output.append(transform(project, geometry))
    return output, repaired


def burn(geometries, grid, all_touched=False):
    import numpy as np
    from rasterio.features import rasterize

    if not geometries:
        return np.zeros((grid["height"], grid["width"]), dtype="uint8")
    return rasterize(
        [(g, 1) for g in geometries], out_shape=(grid["height"], grid["width"]),
        transform=grid["transform"], fill=0, dtype="uint8", all_touched=all_touched,
    )


def scene_quality(scene, grid):
    """Conservatively exclude cloud and supplied per-band quality polygons."""
    from rasterio.transform import array_bounds
    from shapely.geometry import box

    footprint = box(*array_bounds(grid["height"], grid["width"], grid["transform"]))
    clouds = list(scene.glob("GRANULE/*/QI_DATA/MSK_CLOUDS_B00.gml"))
    if len(clouds) != 1:
        raise ValueError(f"Cloud QA is required for {scene.name}")
    paths = clouds + [p for band in BANDS for kind in ("NODATA", "SATURA", "DEFECT", "TECQUA")
                      for p in scene.glob(f"GRANULE/*/QI_DATA/MSK_{kind}_{band}.gml")]
    bad, cloud, repair_count = [], [], 0
    for path in paths:
        geometries, repaired = quality_geometries(path, grid["crs"])
        repair_count += repaired
        relevant = [g.intersection(footprint) for g in geometries if g.intersects(footprint)]
        bad.extend(relevant)
        if path in clouds:
            cloud.extend(relevant)
    return burn(bad, grid, all_touched=True).astype(bool), {
        "cloud_pixels": int(burn(cloud, grid, all_touched=True).sum()),
        "quality_files_checked": len(paths), "repaired_quality_geometries": repair_count,
        "cloud_source": relative(clouds[0]),
        "limitations": "Product QA polygons do not guarantee absence of cloud shadow, haze or annotation errors.",
    }


def read_band_on_grid(path, grid):
    """The virtual raster is only the requested crop, never the full tile."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    with rasterio.open(path) as source:
        with WarpedVRT(
            source, crs=grid["crs"], transform=grid["transform"],
            width=grid["width"], height=grid["height"],
            src_nodata=0, nodata=float("nan"), dtype="float32",
            resampling=Resampling.bilinear, warp_mem_limit=16,
            warp_extras={"NUM_THREADS": "1"},
        ) as crop:
            data = crop.read(1)
        # Sample saturation separately with conservative max resampling so that
        # interpolation cannot turn a saturated DN into a plausible reflectance.
        with WarpedVRT(
            source, crs=grid["crs"], transform=grid["transform"],
            width=grid["width"], height=grid["height"],
            src_nodata=0, nodata=0, dtype="float32", resampling=Resampling.max,
            warp_mem_limit=16, warp_extras={"NUM_THREADS": "1"},
        ) as crop:
            saturated = crop.read(1) >= 65535
    valid = np.isfinite(data) & (data > 0) & ~saturated
    return data, valid


def compute_features(raw_bands, quantification):
    import numpy as np

    bands = {name: np.asarray(raw_bands[name], dtype="float32") for name in BANDS}

    def index(a, b):
        denominator = a + b
        return np.divide(a - b, denominator, out=np.zeros_like(a),
                         where=np.isfinite(denominator) & (denominator != 0))

    ndvi = index(bands["B08"], bands["B04"])
    ndmi = index(bands["B8A"], bands["B11"])
    return np.stack([*(bands[b] / quantification for b in BANDS), ndvi, ndmi])


def temporal_masks(records, pre_date, post_date, grid):
    """New observed loss is the later cumulative footprint minus the earlier."""
    import numpy as np

    if not pre_date < post_date:
        raise ValueError("The post observation must follow the pre observation")
    before = burn([g for d, g in records if d <= pre_date], grid)
    after = burn([g for d, g in records if d <= post_date], grid)
    change = ((after > 0) & (before == 0)).astype(np.uint8)
    return before, after, change


def write_tif(path, values, grid, nodata, descriptions):
    import rasterio

    data = values[None] if values.ndim == 2 else values
    with rasterio.open(
        path, "w", driver="GTiff", crs=grid["crs"], transform=grid["transform"],
        width=grid["width"], height=grid["height"], count=data.shape[0],
        dtype=data.dtype, nodata=nodata, compress="deflate", num_threads=1,
    ) as output:
        output.write(data)
        output.descriptions = tuple(descriptions)


def create_preview(path, before, after, change, selected_mask, valid, title, dates):
    """Render shared-stretch RGB comparisons at full crop and annotation zoom."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    rgb = [np.moveaxis(a[[2, 1, 0]], 0, -1) for a in (before, after)]
    pixels = np.concatenate([a[valid] for a in rgb])
    low, high = np.percentile(pixels, (2, 98), axis=0)
    rgb = [Image.fromarray((np.clip((np.nan_to_num(a) - low) / np.maximum(high-low, 1e-6), 0, 1)
                           * 255).astype("uint8")) for a in rgb]
    mask_image = Image.fromarray((change * 255).astype("uint8"))
    ys, xs = np.where(selected_mask)
    cx, cy = int((xs.min()+xs.max())/2), int((ys.min()+ys.max())/2)
    side = min(64, before.shape[1]//2, before.shape[2]//2)
    x = max(0, min(cx-side//2, before.shape[2]-side)); y = max(0, min(cy-side//2, before.shape[1]-side))
    zoom = (x, y, x+side, y+side)
    canvas = Image.new("RGB", (1280, 1035), "#101822")
    draw = ImageDraw.Draw(canvas)
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    try:
        font, small, heading = (ImageFont.truetype(font_path, s) for s in (21, 17, 28))
    except OSError:
        font = small = heading = ImageFont.load_default()
    draw.text((25, 20), title, font=heading, fill="white")
    draw.text((25, 60), "Reference annotations, not model predictions | shared RGB contrast | north up", font=small, fill="#b7c5d6")
    headers = [f"Before: {dates[0]}", f"After: {dates[1]}", "After + new-loss annotation"]
    for row in range(2):
        for col, picture in enumerate([*rgb, rgb[1]]):
            px, py = 25+col*420, 118+row*440
            display = picture if row == 0 else picture.crop(zoom)
            display = display.resize((400, 400), Image.Resampling.NEAREST)
            if col == 2:
                label = mask_image if row == 0 else mask_image.crop(zoom)
                label = label.resize((400, 400), Image.Resampling.NEAREST)
                display = Image.composite(Image.blend(display, Image.new("RGB", display.size, "#ff3535"), .25), display, label)
                edge = np.asarray(label) > np.asarray(label.filter(ImageFilter.MinFilter(3)))
                a = np.asarray(display).copy(); a[edge] = [255, 255, 255]; display = Image.fromarray(a)
            draw.text((px, py-29), headers[col] if row == 0 else f"Detail: {side*10} m x {side*10} m", font=font, fill="white")
            canvas.paste(display, (px, py))
    draw.text((25, 998), f"Top: {before.shape[1]*10/1000:g} km crop | Bottom: 10 m pixels enlarged | White/red = new annotated loss", font=small, fill="#b7c5d6")
    canvas.save(path)


def build_sample(args, output):
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import array_bounds
    from rasterio.windows import Window
    from shapely.geometry import box, mapping, shape
    from shapely.ops import transform

    selected = [f for f in read_json(args.samples)["features"]
                if f["properties"].get("name") == args.plot]
    if len(selected) != 1:
        raise ValueError("The selected plot must identify exactly one annotation")
    selected = selected[0]
    tile = selected["properties"]["tile"]
    pre_date, post_date = date.fromisoformat(args.pre_date), date.fromisoformat(args.post_date)
    annotation_date = date.fromisoformat(selected["properties"]["img_date"][:10])
    if not (5 <= (post_date-pre_date).days <= 30 and pre_date < annotation_date <= post_date):
        raise ValueError("Use a 5–30 day pair bracketing the selected annotation observation")
    scenes = [find_scene(args.sentinel_dir, tile, d) for d in (pre_date, post_date)]
    with rasterio.open(band_path(scenes[0], "B02")) as reference:
        if not reference.crs.is_projected or reference.res != (10., 10.):
            raise ValueError("This sample requires a projected 10 m reference grid")
        project = Transformer.from_crs("EPSG:4326", reference.crs, always_xy=True).transform
        selected_geometry = transform(project, shape(selected["geometry"]))
        if not selected_geometry.is_valid:
            raise ValueError("The selected annotation is invalid; review it before extraction")
        cx, cy = selected_geometry.centroid.coords[0]
        row, col = reference.index(cx, cy)
        # Align the window origin to the 20 m band lattice as well as the 10 m grid.
        row0, col0 = ((v-args.patch_size//2)//2*2 for v in (row, col))
        window = Window(col0, row0, args.patch_size, args.patch_size)
        if min(row0, col0) < 0 or row0+args.patch_size > reference.height or col0+args.patch_size > reference.width:
            raise ValueError("The crop crosses the reference tile edge")
        grid = dict(crs=reference.crs, transform=reference.window_transform(window),
                    width=args.patch_size, height=args.patch_size)
    footprint = box(*array_bounds(grid["height"], grid["width"], grid["transform"]))
    if not footprint.covers(selected_geometry):
        raise ValueError("The selected annotation does not fit in this crop")

    records, audit, seen = [], [], set()
    source = read_json(args.annotations)
    for i, feature in enumerate(source["features"]):
        props = feature["properties"]
        if props.get("tile", props.get("tileID")) != tile:
            continue
        observed = date.fromisoformat(props["img_date"][:10])
        if observed > post_date:
            continue
        geometry = transform(project, shape(feature["geometry"]))
        if not box(*geometry.bounds).intersects(footprint):
            continue
        if not geometry.is_valid:
            raise ValueError(f"Invalid source annotation at feature {i}; review it explicitly")
        if not geometry.intersects(footprint):
            continue
        key = (observed.isoformat(), geometry.normalize().wkb)
        if key in seen:
            continue
        seen.add(key)
        records.append((observed, geometry))
        audit.append({"type": "Feature", "geometry": mapping(shape(feature["geometry"])),
                      "properties": {"source_feature_index": i, "img_date": observed.isoformat(),
                                     "tile": tile, "within_pair_interval": pre_date < observed <= post_date}})
    if not any(d == annotation_date and g.hausdorff_distance(selected_geometry) <= .001
               and g.symmetric_difference(selected_geometry).area <= .01 for d, g in records):
        raise ValueError("The selected event cannot be traced to the full annotation source")
    before_mask, after_mask, change = temporal_masks(records, pre_date, post_date, grid)
    selected_mask = burn([selected_geometry], grid)
    if not selected_mask.any() or not (change & selected_mask).any():
        raise ValueError("The selected polygon must contain rasterized new-loss pixels")
    valid, features, qa = np.ones(change.shape, dtype=bool), [], []
    for scene in scenes:
        bad, quality = scene_quality(scene, grid)
        if bad[selected_mask.astype(bool)].any():
            raise ValueError(f"Selected annotation is cloud-covered or QA-invalid in {scene.name}")
        print(f"[sample] Reading seven small windows: {scene.name}", flush=True)
        raw = {}
        scene_valid = ~bad
        for band in BANDS:
            raw[band], band_valid = read_band_on_grid(band_path(scene, band), grid)
            scene_valid &= band_valid
        values = compute_features(raw, scene_scale(scene))
        values[:, ~scene_valid] = np.nan
        features.append(values)
        valid &= scene_valid
        quality["invalid_pixels"] = int((~scene_valid).sum())
        qa.append(quality)
    if not valid[selected_mask.astype(bool)].all() or not change.any():
        raise ValueError("The selected change has missing pixels or no rasterized target")
    for values in features:
        values[:, ~valid] = np.nan
    target = change.copy(); target[~valid] = 255
    for label, values in zip(("pre", "post"), features):
        write_tif(output / f"{label}_features.tif", values, grid, float("nan"), CHANNELS)
    for name, values in [("pre_cumulative_loss", before_mask), ("post_cumulative_loss", after_mask),
                         ("change_mask", target), ("valid_pixels", valid.astype("uint8"))]:
        write_tif(output / f"{name}.tif", values, grid, 255 if name == "change_mask" else None, (name,))
    (output / "annotations.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": audit}, indent=2)+"\n")
    create_preview(output / "comparison.png", *features, change, selected_mask, valid,
                   args.plot+" | aligned satellite pair", (args.pre_date, args.post_date))
    # Round-trip validation checks the deliverables rather than trusting write parameters.
    for path in output.glob("*.tif"):
        with rasterio.open(path) as src:
            if src.crs != grid["crs"] or src.transform != grid["transform"] or src.shape != change.shape:
                raise AssertionError(f"Output grid mismatch: {path}")
    metadata = {
        "plot": args.plot, "tile": tile, "pre_date": args.pre_date, "post_date": args.post_date,
        "interval_days": (post_date-pre_date).days, "annotation_observation_date": annotation_date.isoformat(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid": {"crs": str(grid["crs"]), "transform": list(grid["transform"])[:6],
                 "width": grid["width"], "height": grid["height"], "resolution_m": [10, 10],
                 "bounds": list(footprint.bounds)},
        "sources": {"annotations": relative(args.annotations), "annotations_sha256": digest(args.annotations),
                    "samples": relative(args.samples), "samples_sha256": digest(args.samples),
                    "pre_scene": relative(scenes[0]), "post_scene": relative(scenes[1]),
                    "dataset": "https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine",
                    "paper_doi": "10.1109/JSTARS.2020.3034186"},
        "channels": list(CHANNELS), "scaling": "Seven reflectance bands / quantification; indices remain signed [-1,1]",
        "resampling": "Bilinear onto the same 10 m grid; labels use pixel centres; QA uses all touched pixels",
        "target": "Later cumulative observed-loss mask minus earlier cumulative mask; 0=no new annotated loss,1=new loss,255=invalid",
        "annotation_provenance": "Selected polygon matched to local full source; original baseline/temporal subset membership remains unverified",
        "selected_source_match_tolerance": "<=1 mm Hausdorff distance and <=0.01 square metres symmetric difference (projection roundoff)",
        "annotation_dates_are": "Image observation dates, not exact felling dates",
        "intersecting_unique_annotations_by_post_date": len(records),
        "new_annotations_in_interval": sum(pre_date < d <= post_date for d, _ in records),
        "selected_polygon_area_m2": selected_geometry.area,
        "selected_polygon_pixels": int(selected_mask.sum()),
        "selected_new_loss_pixels": int((change & selected_mask)[valid].sum()),
        "change_pixels": int(change[valid].sum()), "rasterized_change_area_m2": int(change[valid].sum())*100,
        "valid_pixels": int(valid.sum()), "total_pixels": int(valid.size), "quality": qa,
        "feature_ranges": {label: {name: [float(np.nanmin(a)), float(np.nanmax(a))]
                                    for name, a in zip(CHANNELS, values)}
                           for label, values in zip(("pre", "post"), features)},
        "checks": {"output_grids_equal": True, "selected_annotation_traceable": True,
                   "selected_annotation_valid_both_dates": True, "change_target_nonempty": True,
                   "visual_review": "pending"},
        "limitations": ["0 means no new annotated loss, not proof of unchanged forest.",
                        "Annotation observation dates and product QA have uncertainties.",
                        "This single sample does not establish model performance or exact event timing."],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print(f"[sample] Grid checks passed; {metadata['change_pixels']} change pixels", flush=True)


def run_guarded(args, script_path=None, operation="sample", gdal_cache_mib=32, warp_memory_mib=16, quiet=False):
    import psutil

    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists; choose a new --output: {destination}")
    before = psutil.virtual_memory()
    reserve = args.min_available_gib * 2**30
    if before.available < reserve + 256*2**20:
        raise RuntimeError("Insufficient available memory for even the guarded sample")
    if shutil.disk_usage(destination.parent if destination.parent.exists() else ROOT).free < 256*2**20:
        raise RuntimeError("At least 256 MiB of disk space is required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{operation}-", dir=destination.parent))
    env = os.environ.copy()
    env.update({key: "1" for key in THREAD_VARIABLES})
    env.update(PYTHONDONTWRITEBYTECODE="1", GDAL_CACHEMAX="32")
    command = [sys.executable, "-B", str(Path(script_path or __file__).resolve()), *sys.argv[1:], "--worker-output", str(stage)]
    worker, started, peak_rss, minimum_available = None, time.monotonic(), 0, before.available
    try:
        worker = subprocess.Popen(command, env=env)
        process = psutil.Process(worker.pid)
        while worker.poll() is None:
            available = psutil.virtual_memory().available
            minimum_available = min(minimum_available, available)
            try:
                rss = process.memory_info().rss
            except psutil.NoSuchProcess:
                break
            peak_rss = max(peak_rss, rss)
            if rss > args.max_rss_mib*2**20:
                raise RuntimeError("Worker exceeded its memory budget; stopped without publishing output")
            if available < reserve:
                raise RuntimeError("Available Mac memory fell below the reserve; stopped processing")
            if time.monotonic()-started > args.max_seconds:
                raise RuntimeError("Sample exceeded its runtime budget; stopped processing")
            time.sleep(.25)
        if worker.wait() != 0:
            raise RuntimeError(f"{operation.capitalize()} failed; staging output was discarded")
        metadata = read_json(stage / "metadata.json")
        if metadata["worker_high_water_rss_mib"] > args.max_rss_mib:
            raise RuntimeError("Worker high-water memory exceeded its budget; output was not published")
        metadata["resources"] = {
            "logical_cpu_count": psutil.cpu_count(), "machine_ram_gib": round(before.total/2**30, 2),
            "worker_peak_rss_mib_sampled": round(peak_rss/2**20, 1),
            "worker_high_water_rss_mib": metadata.pop("worker_high_water_rss_mib"),
            "minimum_available_gib_sampled": round(minimum_available/2**30, 2),
            "elapsed_seconds": round(time.monotonic()-started, 2),
            "worker_limit_mib": args.max_rss_mib, "minimum_available_gib_limit": args.min_available_gib,
            "runtime_limit_seconds": args.max_seconds, "workers": 1, "native_thread_limit": 1,
            "nice_increment": 10, "gdal_cache_mib": gdal_cache_mib, "warp_memory_mib": warp_memory_mib,
            "gpu_used": False, "monitor_interval_seconds": .25,
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
        if destination.exists():
            raise FileExistsError(f"Output appeared during processing: {destination}")
        stage.rename(destination)
        if not quiet:
            print(json.dumps({"output": str(destination), "resources": metadata["resources"]}, indent=2))
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker.kill(); worker.wait()
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", default="PLOT-00026")
    parser.add_argument("--pre-date", default="2018-07-31")
    parser.add_argument("--post-date", default="2018-08-05")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--sentinel-dir", type=Path, default=ROOT / "Datasets/Sentinel-2")
    parser.add_argument("--samples", type=Path, default=ROOT / "Datasets/Testing/Samples/sampled_events_20250129_155343.geojson")
    parser.add_argument("--annotations", type=Path, default=ROOT / "Datasets/BoundingBox/deforestation.geojson")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-rss-mib", type=int, default=512)
    parser.add_argument("--min-available-gib", type=float, default=1.5)
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 64 <= args.patch_size <= 256 or args.patch_size % 32:
        parser.error("--patch-size must be a multiple of 32 from 64 to 256")
    if not (128 <= args.max_rss_mib <= 768 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error("Use 128–768 MiB RSS, >=1.5 GiB available reserve and <=300 seconds")
    args.output = args.output or ROOT / "Datasets/Validation" / f"{args.plot}_{args.pre_date}_{args.post_date}_{args.patch_size}px"
    for key in THREAD_VARIABLES:
        os.environ[key] = "1"
    if args.worker_output:
        os.nice(10)
        import rasterio
        with rasterio.Env(GDAL_CACHEMAX=32*2**20, GDAL_NUM_THREADS="1"):
            build_sample(args, args.worker_output)
        import resource
        metadata = read_json(args.worker_output / "metadata.json")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata["worker_high_water_rss_mib"] = round(peak/(2**20 if sys.platform == "darwin" else 1024), 1)
        (args.worker_output / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    else:
        run_guarded(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileExistsError) as error:
        print(f"[sample] {error}", file=sys.stderr)
        sys.exit(1)
