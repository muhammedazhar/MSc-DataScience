#!/usr/bin/env python3
"""Audit annotation provenance and freeze geographic splits using metadata only.

No raster pixels, model weights, or training code are loaded. Original files are
preserved. JSONL manifests describe candidates requiring later pixel inspection.
"""

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
STUDY_TILES = {"36UXA", "36UYA"}
PROJECTED_CRS = "EPSG:32636"
MAX_JSON_BYTES = 12 * 2**20


def read_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"Annotation file exceeds the 12 MiB per-file limit: {path}")
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_id(prefix, *parts):
    return prefix + hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def observation_date(value):
    if not isinstance(value, str):
        raise ValueError("Missing observation date")
    return date.fromisoformat(value[:10].replace("/", "-")).isoformat()


def canonical_geometry(value):
    """Exact 2D identity: ring order and single-part wrapper are immaterial."""
    from shapely import force_2d, normalize
    from shapely.geometry import shape

    if value is None:
        raise ValueError("missing_geometry")
    geometry = force_2d(shape(value) if isinstance(value, dict) else value)
    if geometry is None or not hasattr(geometry, "geom_type"):
        raise ValueError("invalid_geometry")
    if geometry.geom_type == "MultiPolygon" and len(geometry.geoms) == 1:
        geometry = geometry.geoms[0]
    return normalize(geometry)


def feature_collection(path):
    content = read_json(path)
    if content.get("type") != "FeatureCollection":
        raise ValueError(f"Expected a GeoJSON FeatureCollection: {path}")
    crs = content.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    if crs not in ("EPSG:4326", "urn:ogc:def:crs:OGC:1.3:CRS84", "urn:ogc:def:crs:OGC::CRS84"):
        raise ValueError(f"Unrecognized annotation CRS {crs}: {path}")
    return content["features"]


def reference_index(registry_path):
    """Index verified originals; missing tile IDs remain explicit exclusions."""
    registry = read_json(registry_path)
    lookup, records, source_summaries = defaultdict(list), [], []
    for source in registry["sources"]:
        path = Path(registry_path).parent / source["filename"]
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {path}")
        features = feature_collection(path)
        dates, tile_counts = [], Counter()
        for index, feature in enumerate(features):
            properties = feature.get("properties") or {}
            entry = {"source_id": source["source_id"], "kind": source["kind"],
                     "source_feature_index": index, "source_url": source["source_url"],
                     "tile": None, "observation_date": None, "issues": []}
            try:
                entry["observation_date"] = observation_date(properties.get("img_date"))
                dates.append(entry["observation_date"])
                # UXA temporal records have no tile field; the original named
                # single-tile collection supplies its declared scope. Null IDs
                # in other sources are not filled from filename or geography.
                if "tileID" not in properties and source.get("declared_tile"):
                    entry["tile"] = source["declared_tile"]
                    entry["tile_derivation"] = "declared_single_tile_source"
                else:
                    matches = re.findall(r"\d{2}[A-Z]{3}", properties.get("tileID") or "")
                    entry["tile"] = matches[-1] if matches else None
                    entry["tile_derivation"] = "source_tileID"
                geometry = canonical_geometry(feature["geometry"])
                if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in ("Polygon", "MultiPolygon"):
                    raise ValueError("invalid_geometry")
                entry["geometry_key"] = hashlib.sha256(geometry.wkb).hexdigest()
                entry["_geometry"] = geometry
                if entry["tile"] is None:
                    entry["issues"].append("missing_original_tile_id")
                else:
                    key = (entry["tile"], entry["observation_date"], entry["geometry_key"])
                    lookup[key].append(entry)
                    tile_counts[entry["tile"]] += 1
            except (KeyError, TypeError, ValueError) as error:
                entry["issues"].append(f"unusable_reference:{error}")
            records.append(entry)
        source_summaries.append({**source, "feature_count": len(features),
                                 "date_min": min(dates) if dates else None,
                                 "date_max": max(dates) if dates else None,
                                 "tile_counts": dict(sorted(tile_counts.items()))})
    return lookup, records, source_summaries


def clean_events(features, reference_index=None):
    from pyproj import Transformer
    from shapely.ops import transform

    project = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True).transform
    by_key, issues = {}, []
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            issues.append({"source_feature_index": index, "reason": "invalid_feature", "properties": {}})
            continue
        properties = feature.get("properties") or {}
        try:
            tile = properties.get("tile", properties.get("tileID"))
            if tile not in STUDY_TILES:
                raise ValueError("unsupported_or_missing_tile")
            observed = observation_date(properties.get("img_date"))
            geometry = canonical_geometry(feature["geometry"])
            if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in ("Polygon", "MultiPolygon"):
                raise ValueError("invalid_or_unsupported_geometry")
            west, south, east, north = geometry.bounds
            if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
                raise ValueError("coordinates_outside_wgs84")
            geometry_hash = hashlib.sha256(geometry.wkb).hexdigest()
            key = (tile, observed, geometry_hash)
            if key in by_key:
                by_key[key]["source_indices"].append(index)
                continue
            projected = transform(project, geometry)
            members = [{k: v for k, v in m.items() if not k.startswith("_")}
                       for m in (reference_index or {}).get(key, [])]
            flags = []
            if not members:
                flags.append("unmatched_reference_provenance")
            temporal_sources = sorted({m["source_id"] for m in members if m["kind"] == "temporal"})
            if members and not temporal_sources:
                flags.append("baseline_only_provenance")
            by_key[key] = {
                "event_id": stable_id("event-", tile, observed, geometry_hash),
                "location_id": "location-" + geometry_hash[:20], "tile": tile,
                "observation_date": observed, "geometry": geometry,
                "projected_geometry": projected, "source_indices": [index],
                "source_memberships": members,
                "temporal_source_ids": temporal_sources, "flags": flags,
                "area_m2": projected.area,
                "width_m": projected.bounds[2]-projected.bounds[0],
                "height_m": projected.bounds[3]-projected.bounds[1],
            }
        except (KeyError, TypeError, ValueError) as error:
            issues.append({"source_feature_index": index, "reason": str(error), "properties": properties})
    events = sorted(by_key.values(), key=lambda e: e["event_id"])
    locations = defaultdict(list)
    for event in events:
        locations[event["location_id"]].append(event)
    for same_location in locations.values():
        if len({e["observation_date"] for e in same_location}) > 1:
            for event in same_location:
                event["flags"].append("conflicting_observation_dates")
    return events, issues


def assign_spatial_splits(events, patch_size=64, seed=42):
    """Connected components prevent overlap of full labels and snapped crops."""
    from shapely.geometry import box
    from shapely.strtree import STRtree

    parent = list(range(len(events)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def merge(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    # The actual window is snapped down to an even 10 m pixel offset. A 20 m
    # margin around the centred crop bounds covers that displacement.
    half = patch_size*10/2 + 20
    footprints = []
    for event in events:
        geometry = event["projected_geometry"]
        if (event["width_m"] > patch_size*10 or event["height_m"] > patch_size*10) and "annotation_requires_tiling" not in event["flags"]:
            event["flags"].append("annotation_requires_tiling")
        cx, cy = geometry.centroid.coords[0]
        footprint = geometry.union(box(cx-half, cy-half, cx+half, cy+half)).envelope
        event["group_footprint_bounds"] = list(footprint.bounds)
        footprints.append(footprint)
    tree = STRtree(footprints)
    for i, footprint in enumerate(footprints):
        for j in tree.query(footprint, predicate="intersects"):
            merge(i, int(j))
    components = defaultdict(list)
    for i, event in enumerate(events):
        components[find(i)].append(event)
    groups = []
    for members in components.values():
        locations = sorted({e["location_id"] for e in members})
        group_id = stable_id("group-", *locations)
        score = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16], 16)/2**64
        split = "train" if score < .7 else "validation" if score < .85 else "test"
        for event in members:
            event["group_id"], event["split"] = group_id, split
        groups.append({"group_id": group_id, "split": split,
                       "event_ids": sorted(e["event_id"] for e in members),
                       "location_ids": locations, "event_count": len(members),
                       "tiles": sorted({e["tile"] for e in members})})
    # Independently verify the published invariant, including touching envelopes.
    for i, footprint in enumerate(footprints):
        for j in tree.query(footprint, predicate="intersects"):
            if events[i]["split"] != events[int(j)]["split"]:
                raise AssertionError("Overlapping label/crop footprints cross splits")
    return sorted(groups, key=lambda g: g["group_id"])


def crop_grid(event, scene, patch_size):
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform

    grid = scene.get("grid_10m")
    if not grid or not scene.get("crs"):
        return None, ["missing_scene_grid"]
    if scene["crs"] == PROJECTED_CRS:
        geometry = event["projected_geometry"]
    else:
        project = Transformer.from_crs("EPSG:4326", scene["crs"], always_xy=True).transform
        geometry = transform(project, event["geometry"])
    cx, cy = geometry.centroid.coords[0]
    col = math.floor((cx-grid["ulx"])/10)
    row = math.floor((grid["uly"]-cy)/10)
    col0, row0 = ((v-patch_size//2)//2*2 for v in (col, row))
    west, north = grid["ulx"]+col0*10, grid["uly"]-row0*10
    bounds = [west, north-patch_size*10, west+patch_size*10, north]
    reasons = []
    if min(col0, row0) < 0 or col0+patch_size > grid["width"] or row0+patch_size > grid["height"]:
        reasons.append("crop_crosses_tile_edge")
    if not box(*bounds).covers(geometry):
        reasons.append("annotation_requires_tiling")
    return {"crs": scene["crs"], "transform": [10., 0., west, 0., -10., north],
            "width": patch_size, "height": patch_size, "bounds": bounds}, reasons


def enumerate_candidates(events, scenes, patch_size=64, source_exclusions=None):
    """Enumerate calendar-valid possibilities, never call them training-approved."""
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform
    from shapely.strtree import STRtree

    by_tile = defaultdict(list)
    for scene in scenes:
        if scene.get("tile") and scene.get("acquisition_date"):
            by_tile[scene["tile"]].append(scene)
    scene_pairs = {}
    for tile, records in by_tile.items():
        ordered = sorted(records, key=lambda s: (s["acquisition_date"], s["scene_id"]))
        scene_pairs[tile] = [(a, b) for i, a in enumerate(ordered) for b in ordered[i+1:]
                             if 5 <= (date.fromisoformat(b["acquisition_date"])-date.fromisoformat(a["acquisition_date"])).days <= 30]
    geometries = [e["projected_geometry"] for e in events]
    tree = STRtree(geometries)
    unresolved_footprints = [r["_projected_geometry"] for r in (source_exclusions or [])
                             if r.get("_projected_geometry") is not None]
    unresolved_tree = STRtree(unresolved_footprints)
    pairs, manifest = {}, []
    for event in events:
        observed = event["observation_date"]
        records = by_tile[event["tile"]]
        reasons = list(event["flags"])
        if not any(s["acquisition_date"] < observed for s in records):
            reasons.append("missing_pre_observation")
        if not any(s["acquisition_date"] >= observed for s in records):
            reasons.append("missing_post_observation")
        pair_ids = []
        for pre, post in scene_pairs.get(event["tile"], []):
            if not pre["acquisition_date"] < observed <= post["acquisition_date"]:
                continue
            grid, crop_reasons = crop_grid(event, pre, patch_size)
            blocking = list(reasons) + crop_reasons
            if not pre.get("metadata_usable") or not post.get("metadata_usable"):
                blocking.append("scene_metadata_incomplete")
            if grid:
                # Transform the actual crop, including cross-tile acquisitions,
                # into the common grouping CRS for conflict and label queries.
                footprint = box(*grid["bounds"])
                if grid["crs"] != PROJECTED_CRS:
                    footprint = transform(Transformer.from_crs(grid["crs"], PROJECTED_CRS, always_xy=True).transform, footprint)
                relevant = [events[int(j)] for j in tree.query(footprint, predicate="intersects")]
                if any(e["group_id"] != event["group_id"] for e in relevant):
                    raise AssertionError("A crop includes labels assigned to another spatial group")
                ambiguous = sorted(e["event_id"] for e in relevant if "conflicting_observation_dates" in e["flags"])
                if ambiguous:
                    blocking.append("conflicting_annotation_in_crop")
                if len(unresolved_tree.query(footprint, predicate="intersects")):
                    blocking.append("unresolved_original_annotation_in_crop")
                if any(not e["temporal_source_ids"] and pre["acquisition_date"] < e["observation_date"] <= post["acquisition_date"]
                       for e in relevant):
                    blocking.append("unverified_temporal_label_in_crop")
                post_grid = post.get("grid_10m")
                if not post_grid or post.get("crs") != grid["crs"] or not box(*post_grid["bounds"]).covers(box(*grid["bounds"])):
                    blocking.append("post_scene_does_not_cover_crop")
                bounds_key = json.dumps(grid["bounds"], separators=(",", ":"))
            else:
                relevant, ambiguous = [], []
                bounds_key = event["location_id"]
            pair_id = stable_id("pair-", pre["scene_id"], post["scene_id"], str(patch_size), bounds_key)
            pair_ids.append(pair_id)
            if pair_id not in pairs:
                pairs[pair_id] = {
                    "pair_id": pair_id, "group_id": event["group_id"], "split": event["split"],
                    "tile": event["tile"], "pre_date": pre["acquisition_date"], "post_date": post["acquisition_date"],
                    "interval_days": (date.fromisoformat(post["acquisition_date"])-date.fromisoformat(pre["acquisition_date"])).days,
                    "pre_scene_id": pre["scene_id"], "post_scene_id": post["scene_id"],
                    "pre_path": pre["path"], "post_path": post["path"], "grid": grid,
                    "anchor_event_ids": [], "blocking_reasons": [],
                    "label_event_ids_by_post_date": sorted(e["event_id"] for e in relevant if e["temporal_source_ids"] and e["observation_date"] <= post["acquisition_date"]),
                    "context_event_ids": sorted(e["event_id"] for e in relevant),
                    "conflicting_label_event_ids": ambiguous,
                    "temporal_source_ids": [], "status": "candidate_for_pixel_review",
                    "pixel_quality": "unreviewed", "visual_review": "unreviewed",
                    "training_approved": False,
                    "label_interpretation": "New annotated loss between observations; zero is not verified no-change",
                }
            pair = pairs[pair_id]
            pair["anchor_event_ids"].append(event["event_id"])
            pair["blocking_reasons"] = sorted(set(pair["blocking_reasons"] + blocking))
            pair["temporal_source_ids"] = sorted(set(pair["temporal_source_ids"] + event["temporal_source_ids"]))
            if pair["blocking_reasons"]:
                pair["status"] = "blocked_for_temporal_training"
        if not pair_ids:
            reasons.append("no_5_to_30_day_bracketing_pair")
        manifest.append({"event_id": event["event_id"], "location_id": event["location_id"],
                         "tile": event["tile"], "observation_date": observed,
                         "group_id": event["group_id"], "split": event["split"],
                         "source_indices": event["source_indices"],
                         "temporal_source_ids": event["temporal_source_ids"],
                         "has_scene_on_annotation_date": any(s["acquisition_date"] == observed for s in records),
                         "available_pre_scene_dates_30d": sorted({s["acquisition_date"] for s in records
                             if 0 < (date.fromisoformat(observed)-date.fromisoformat(s["acquisition_date"])).days <= 30}),
                         "available_post_scene_dates_30d": sorted({s["acquisition_date"] for s in records
                             if 0 <= (date.fromisoformat(s["acquisition_date"])-date.fromisoformat(observed)).days <= 30}),
                         "candidate_pair_ids": sorted(set(pair_ids)),
                         "candidate_pair_count": len(set(pair_ids)),
                         "review_reasons": sorted(set(reasons)), "training_approved": False})
    for item in manifest:
        linked = [pairs[key] for key in item["candidate_pair_ids"]]
        item["pairs_for_pixel_review"] = sum(p["status"] == "candidate_for_pixel_review" for p in linked)
        item["blocked_pair_count"] = len(linked)-item["pairs_for_pixel_review"]
        item["pair_blocking_reasons"] = sorted({reason for p in linked for reason in p["blocking_reasons"]})
    return sorted(pairs.values(), key=lambda p: p["pair_id"]), manifest


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False)+"\n")


def write_jsonl(path, records):
    with Path(path).open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False)+"\n")


def public_event(event):
    return {k: v for k, v in event.items() if k not in ("geometry", "projected_geometry")}


def sampled_plot_aliases(path, events):
    """Trace existing PLOT names through sub-millimetre projection roundoff."""
    from pyproj import Transformer
    from shapely.ops import transform
    from shapely.strtree import STRtree

    project = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True).transform
    tree = STRtree([e["projected_geometry"] for e in events])
    aliases = []
    for index, feature in enumerate(feature_collection(path)):
        properties = feature.get("properties") or {}
        geometry = transform(project, canonical_geometry(feature.get("geometry")))
        observed = observation_date(properties.get("img_date"))
        matches = [events[int(j)] for j in tree.query(geometry.buffer(.001), predicate="intersects")
                   if events[int(j)]["tile"] == properties.get("tile")
                   and events[int(j)]["observation_date"] == observed
                   and roundoff_match(geometry, events[int(j)]["projected_geometry"])]
        alias = {"plot": properties.get("name"), "sample_feature_index": index,
                 "tile": properties.get("tile"), "observation_date": observed,
                 "matching": "Mutual coverage within 1 mm and <=0.01 m2 symmetric difference after projection",
                 "status": "matched" if len(matches) == 1 else "unmatched_or_ambiguous",
                 "event_id": matches[0]["event_id"] if len(matches) == 1 else None,
                 "group_id": matches[0]["group_id"] if len(matches) == 1 else None,
                 "split": matches[0]["split"] if len(matches) == 1 else None}
        aliases.append(alias)
    return aliases


def roundoff_match(a, b):
    """Tolerate reprojection noise without GEOS Hausdorff floating exceptions."""
    if a.symmetric_difference(b).area > .01:
        return False
    return a.buffer(.001).covers(b) and b.buffer(.001).covers(a)


def relative_path(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def build_audit(args, output):
    from shapely.geometry import mapping
    from annotation_scene_inventory import inventory_scenes

    lookup, originals, sources = reference_index(args.references)
    raw = feature_collection(args.annotations)
    events, invalid = clean_events(raw, lookup)
    print(f"[audit] {len(raw)} rows -> {len(events)} unique events", flush=True)
    groups = assign_spatial_splits(events, args.patch_size, args.seed)
    scenes = inventory_scenes(args.sentinel_dir, ROOT)
    matched_originals = {(m["source_id"], m["source_feature_index"])
                         for e in events for m in e["source_memberships"]}
    source_exclusions = []
    for record in originals:
        matched = (record["source_id"], record["source_feature_index"]) in matched_originals
        record["matched_local"] = matched
        if not matched:
            reason = "missing_original_tile_id" if record["tile"] is None else "outside_local_study_tiles" if record["tile"] not in STUDY_TILES else "original_annotation_absent_locally"
            entry = {k: v for k, v in record.items() if not k.startswith("_")}
            entry["exclusion_reason"] = reason
            if record.get("_geometry") is not None:
                entry["geometry"] = mapping(record["_geometry"])
                if reason == "missing_original_tile_id" and record["kind"] == "temporal":
                    from pyproj import Transformer
                    from shapely.ops import transform
                    entry["_projected_geometry"] = transform(Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True).transform, record["_geometry"])
            source_exclusions.append(entry)
    pairs, manifest = enumerate_candidates(events, scenes, args.patch_size, source_exclusions)
    plot_aliases = sampled_plot_aliases(args.samples, events)
    for source in sources:
        source["matched_original_records"] = sum(r["matched_local"] for r in originals if r["source_id"] == source["source_id"])
    aliases = [{"source_feature_index": index, "event_id": e["event_id"], "group_id": e["group_id"], "split": e["split"]}
               for e in events for index in e["source_indices"]]
    duplicates = [{"event_id": e["event_id"], "retained_source_index": min(e["source_indices"]),
                   "source_indices": e["source_indices"], "duplicate_rows_removed": len(e["source_indices"])-1}
                  for e in events if len(e["source_indices"]) > 1]
    conflicts = [{"event_id": e["event_id"], "location_id": e["location_id"], "tile": e["tile"],
                  "observation_date": e["observation_date"], "source_indices": e["source_indices"],
                  "source_memberships": e["source_memberships"]}
                 for e in events if "conflicting_observation_dates" in e["flags"]]
    write_json(output / "annotations_clean.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": e["event_id"], "properties": public_event(e), "geometry": mapping(e["geometry"])} for e in events]})
    write_json(output / "splits.json", {
        "schema_version": 1, "seed": args.seed, "target_group_proportions": {"train": .7, "validation": .15, "test": .15},
        "method": "Connected components of full annotation extents and conservative crop envelopes; deterministic hash allocation",
        "crs": PROJECTED_CRS, "patch_size_pixels": args.patch_size, "resolution_m": 10,
        "grid_snap_margin_m": 20, "requires_recompute_if_crop_plan_changes": True,
        "groups": groups,
    })
    for name, records in [("annotation_manifest", manifest), ("scene_inventory", scenes), ("candidate_pairs", pairs),
                          ("source_aliases", sorted(aliases, key=lambda a: a["source_feature_index"])),
                          ("duplicate_events", duplicates), ("date_conflicts", conflicts),
                          ("sampled_plot_aliases", plot_aliases),
                          ("invalid_local_records", invalid),
                          ("original_source_exclusions", [{k: v for k, v in r.items() if not k.startswith("_")} for r in source_exclusions])]:
        write_jsonl(output / f"{name}.jsonl", records)
    write_json(output / "source_provenance.json", {"sources": sources, "matching": "Exact normalized 2D geometry, observation date and tile; no coordinate rounding"})
    stats = {
        "raw_rows": len(raw), "unique_events": len(events), "duplicate_rows_removed": len(raw)-len(events)-len(invalid),
        "duplicate_groups": len(duplicates), "invalid_local_records": len(invalid),
        "local_rows_matched_to_originals": sum(len(e["source_indices"]) for e in events if e["source_memberships"]),
        "source_exclusions": len(source_exclusions),
        "source_exclusion_reasons": dict(Counter(r["exclusion_reason"] for r in source_exclusions)),
        "multi_source_events": sum(len({m["source_id"] for m in e["source_memberships"]}) > 1 for e in events),
        "baseline_only_events": sum("baseline_only_provenance" in e["flags"] for e in events),
        "events_with_temporal_provenance": sum(bool(e["temporal_source_ids"]) for e in events),
        "conflicting_locations": len({e["location_id"] for e in conflicts}), "conflicting_events": len(conflicts),
        "events_wider_or_taller_than_crop": sum(e["width_m"] > args.patch_size*10 or e["height_m"] > args.patch_size*10 for e in events),
        "geographic_groups": len(groups), "largest_group_events": max((g["event_count"] for g in groups), default=0),
        "group_split_counts": dict(Counter(g["split"] for g in groups)),
        "event_split_counts": dict(Counter(e["split"] for e in events)),
        "scenes": len(scenes), "metadata_usable_scenes": sum(s["metadata_usable"] for s in scenes),
        "candidate_pairs": len(pairs), "candidate_pair_status_counts": dict(Counter(p["status"] for p in pairs)),
        "candidate_pair_blocking_reasons": dict(Counter(reason for p in pairs for reason in p["blocking_reasons"])),
        "events_with_no_bracketing_pair": sum(not e["candidate_pair_ids"] for e in manifest),
        "training_approved_pairs": 0, "cross_split_footprint_overlaps": 0,
        "sampled_plot_aliases": len(plot_aliases),
        "sampled_plots_matched": sum(a["status"] == "matched" for a in plot_aliases),
        "pixel_review_candidates_by_split": {
            split: {"pairs": sum(p["status"] == "candidate_for_pixel_review" and p["split"] == split for p in pairs),
                    "distinct_anchor_events": len({event_id for p in pairs if p["status"] == "candidate_for_pixel_review" and p["split"] == split for event_id in p["anchor_event_ids"]}),
                    "distinct_geographic_groups": len({p["group_id"] for p in pairs if p["status"] == "candidate_for_pixel_review" and p["split"] == split})}
            for split in ("train", "validation", "test")},
    }
    metadata = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "annotations": relative_path(args.annotations), "annotations_sha256": sha256(args.annotations),
        "reference_registry": relative_path(args.references), "reference_registry_sha256": sha256(args.references),
        "sampled_plots": relative_path(args.samples), "sampled_plots_sha256": sha256(args.samples),
        "seed": args.seed, "patch_size_pixels": args.patch_size, "statistics": stats,
        "date_semantics": "Satellite annotation observation date, not exact felling date",
        "checks": {"aliases_preserve_all_valid_local_rows": len(aliases)+len(invalid) == len(raw),
                   "no_duplicate_event_ids": len({e["event_id"] for e in events}) == len(events),
                   "no_cross_split_label_or_crop_overlap": True,
                   "all_candidate_intervals_5_to_30_days": all(5 <= p["interval_days"] <= 30 for p in pairs)},
        "limitations": [
            "Metadata readiness does not establish clear pixels, trustworthy boundaries, or verified negative observations.",
            "Baseline-only events and conflicting dates are blocked from confident temporal training.",
            "Five source spring records lack tile IDs and are preserved as exclusions rather than silently restored.",
            "Spatial splits support within-region evaluation; they do not establish generalization to different countries or years.",
            "Changing crop size, crop centres or tiling strategy requires recomputing and verifying spatial groups.",
        ],
    }
    if not all(metadata["checks"].values()):
        raise AssertionError("An annotation audit invariant failed")
    write_json(output / "metadata.json", metadata)
    write_report(output / "audit_report.md", metadata, sources, conflicts)
    print(json.dumps(stats, indent=2), flush=True)


def write_report(path, metadata, sources, conflicts):
    s = metadata["statistics"]
    lines = ["# Annotation and dataset-manifest audit", "", "## Catalogue", "",
             f"- Local rows: **{s['raw_rows']:,}**.", f"- Unique events: **{s['unique_events']:,}**.",
             f"- Duplicate rows removed: **{s['duplicate_rows_removed']:,}**; every original index is preserved as an alias.",
             f"- Local rows matched exactly to original author files: **{s['local_rows_matched_to_originals']:,}**.",
             f"- Events with temporal-source provenance: **{s['events_with_temporal_provenance']:,}**.",
             f"- Baseline-only events requiring temporal-label review: **{s['baseline_only_events']:,}**.",
             "", "Exact matches ignore ring ordering, altitude and a single-part Polygon/MultiPolygon wrapper. Coordinates are not rounded.",
             "", "## Original sources", "", "| File | Role | Original rows | Matched original rows |", "| --- | --- | ---: | ---: |"]
    lines += [f"| [{x['filename']}]({x['source_url']}) | {x['kind']} | {x['feature_count']} | {x['matched_original_records']} |" for x in sources]
    lines += ["", "85 unmatched baseline records belong to tile 37UCR outside the local study. Five spring records have missing tile IDs. They are retained in the source audit and not automatically inserted into the local catalogue.",
              "", "## Date conflicts", "", "Identical footprints with different observation dates are preserved and blocked from confident temporal training:", ""]
    lines += [f"- `{e['location_id']}`: {e['observation_date']}; local rows {e['source_indices']}." for e in conflicts]
    lines += ["", "## Fixed geographic splits", "", f"**{s['geographic_groups']}** connected geographic groups; largest group: **{s['largest_group_events']} events**.",
              "", "Full annotation extents and conservative crop envelopes are grouped across dates and tile IDs. The crop is 64 pixels at 10 m, with a 20 m grid-snapping margin. Any overlap, including transitive overlap chains, keeps records together. A seeded hash assigns whole groups to nominal 70/15/15 train/validation/test splits; actual proportions need not be exact.",
              "", "| Split | Groups | Events |", "| --- | ---: | ---: |"]
    lines += [f"| {split} | {s['group_split_counts'].get(split, 0)} | {s['event_split_counts'].get(split, 0)} |" for split in ("train", "validation", "test")]
    lines += ["", "The audit verifies **zero cross-split crop/label footprint overlaps** for this fixed plan. Enlarging windows or changing crop centres/tiling requires rebuilding the plan.",
              "", "## Imagery and candidate pairs", "", f"- **{s['metadata_usable_scenes']}/{s['scenes']}** local scenes have complete supported metadata.",
              f"- **{s['candidate_pairs']}** distinct 5–30 day bracketing candidates.",
              f"- **{s['events_with_no_bracketing_pair']}** events have no local bracketing pair in that window.",
              f"- **{s['events_wider_or_taller_than_crop']}** annotations exceed the crop width/height and need a tiling decision.",
              "", "| Candidate status | Count |", "| --- | ---: |"]
    lines += [f"| {status} | {count} |" for status, count in sorted(s["candidate_pair_status_counts"].items())]
    lines += ["", "Blocked candidates remain visible with explicit reasons; they are not silently discarded. No candidate is training-approved: per-crop cloud/shadow, nodata and visual checks are the next stage. Scene cloud percentages cannot establish local visibility.",
              "", "| Split | Candidates for pixel review | Distinct anchor events | Geographic groups |",
              "| --- | ---: | ---: | ---: |"]
    lines += [f"| {split} | {counts['pairs']} | {counts['distinct_anchor_events']} | {counts['distinct_geographic_groups']} |"
              for split, counts in s["pixel_review_candidates_by_split"].items()]
    lines += ["", "Pairs sharing an event or geographic group are correlated. Evaluation coverage must be reassessed after pixel QA; the pair count is not a count of independent locations.",
              "", "## Files", "", "`annotations_clean.geojson` is the deduplicated catalogue. `source_aliases.jsonl` maps every valid original local index to its canonical event and split. `splits.json` freezes the groups. `annotation_manifest.jsonl` lists event-level image availability. `candidate_pairs.jsonl` records candidate crops and their prerequisites. `scene_inventory.jsonl` inventories metadata without reading JP2 bytes. Other JSONL files preserve duplicates, date conflicts, invalid records and source exclusions.",
              "", "## Scope and limits", ""]
    lines += [f"- {text}" for text in metadata["limitations"]]
    lines += ["", "Original imagery, annotations, existing model artifacts and prior training scripts are unchanged. Reproduction commands and resource measurements are documented in the project annotation-audit guide.", ""]
    Path(path).write_text("\n".join(lines))


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ROOT / "Datasets/BoundingBox/deforestation.geojson")
    parser.add_argument("--references", type=Path, default=ROOT / "Datasets/AnnotationSources/sources.json")
    parser.add_argument("--sentinel-dir", type=Path, default=ROOT / "Datasets/Sentinel-2")
    parser.add_argument("--samples", type=Path, default=ROOT / "Datasets/Testing/Samples/sampled_events_20250129_155343.geojson")
    parser.add_argument("--output", type=Path, default=ROOT / "Datasets/AnnotationAudit/v1")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rss-mib", type=int, default=512)
    parser.add_argument("--min-available-gib", type=float, default=1.5)
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.patch_size != 64:
        parser.error("This audit freezes the reviewed 64-pixel crop plan; changing it requires a new reviewed plan")
    if not (128 <= args.max_rss_mib <= 768 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error("Use 128–768 MiB RSS, >=1.5 GiB available reserve and <=300 seconds")
    for key in THREAD_VARIABLES:
        os.environ[key] = "1"
    if args.worker_output:
        os.nice(10)
        build_audit(args, args.worker_output)
        import resource
        metadata = read_json(args.worker_output / "metadata.json")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata["worker_high_water_rss_mib"] = round(peak/(2**20 if sys.platform == "darwin" else 1024), 1)
        write_json(args.worker_output / "metadata.json", metadata)
    else:
        run_guarded(args, script_path=__file__, operation="annotation-audit", gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileExistsError) as error:
        print(f"[audit] {error}", file=sys.stderr)
        sys.exit(1)
