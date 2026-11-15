#!/usr/bin/env python3
"""Publish explicitly reviewed, training-only partial-supervision examples.

Run after inspecting prepared timelines and recording decisions and review-time
artifact hashes. Original imagery, QA products, split plans and models stay
unchanged. A single bounded worker performs all feature extraction sequentially.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
MAX_REVIEWS = 64


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def read_json(path, limit_mib=20):
    path = Path(path)
    if path.stat().st_size > limit_mib*2**20:
        raise ValueError(f"Metadata exceeds its bounded size: {path}")
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def records(path):
    with Path(path).open("rb") as stream:
        for number in range(20_001):
            line = stream.readline(512*1024+1)
            if not line:
                return
            if number == 20_000 or len(line) > 512*1024 or not line.strip():
                raise ValueError("Input JSONL exceeds its record/line bounds")
            yield json.loads(line)


def rooted(path):
    resolved = (ROOT/str(path)).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("Input or output paths must stay inside the project")
    return resolved


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def check_stats(expected):
    for path, values in expected.items():
        stat = rooted(path).stat()
        if [stat.st_size, stat.st_mtime_ns] != values:
            raise ValueError("Prepared timeline/source files changed after screening")


def read_mask(path, grid):
    import numpy as np
    import rasterio
    from affine import Affine

    with rasterio.open(path) as source:
        if (source.shape != (64, 64) or source.count != 1 or source.dtypes != ("uint8",)
                or source.transform != Affine(*grid["transform"])
                or source.crs != rasterio.crs.CRS.from_user_input(grid["crs"])):
            raise ValueError("Prepared review ROI mask has a different grid or type")
        values = source.read(1)
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Prepared review ROI mask must be binary")
    return values.astype(bool)


def bind_review(review):
    """Bind every decision, including rejection, to the inspected source bytes."""
    case_path = rooted(review["case_file"])
    files = {"case_sha256": case_path, "timeline_png_sha256": case_path.parent/"timeline.png",
             "pair_png_sha256": case_path.parent/"pair.png"}
    optional = {"positive_interior_sha256": case_path.parent/"positive_interior.tif",
                "proposed_control_sha256": case_path.parent/"proposed_control.tif"}
    for field, path in files.items():
        if not isinstance(review.get(field), str) or review[field] != digest(path):
            raise ValueError(f"Review-time artifact hash is missing or stale: {field}")
    for field, path in optional.items():
        if field in review and review[field] != digest(path):
            raise ValueError(f"Review-time artifact hash is stale: {field}")
    case = read_json(case_path)
    if not re.fullmatch(r"event-[a-zA-Z0-9-]+", case["event_id"]):
        raise ValueError("Invalid prepared event identifier")
    if case_path.parent.name != case["event_id"]:
        raise ValueError("Prepared case directory and event ID disagree")
    batch_path = case_path.parent.parent/"metadata.json"
    batch = read_json(batch_path)
    if not any(item["event_id"] == case["event_id"] and item["pair_id"] == case["pair_id"] for item in batch["cases"]):
        raise ValueError("The case is absent from its prepared-batch manifest")
    artifact_paths = [case_path, case_path.parent/"timeline.png", case_path.parent/"pair.png",
                      case_path.parent/"positive_interior.tif", case_path.parent/"proposed_control.tif",
                      case_path.parent/"target_mask.tif", batch_path]
    return case, batch, {relative(path): digest(path) for path in artifact_paths}


def validate_frozen_case(case, batch, frozen_pair, summary, audit):
    """The full candidate dictionary and original split footprint stay fixed."""
    from shapely.geometry import box

    if case["pair"] != frozen_pair:
        raise ValueError("The prepared case no longer matches the exact frozen candidate")
    if (case["split"] != "train" or frozen_pair["split"] != "train"
            or case["group_id"] != frozen_pair["group_id"]
            or case["pair_id"] != frozen_pair["pair_id"]
            or case["event_id"] not in frozen_pair["anchor_event_ids"]):
        raise ValueError("The case is not an exact training anchor/pair join")
    if case["config_id"] != summary["config_id"] or batch["config_id"] != summary["config_id"]:
        raise ValueError("Prepared case and completed QA policies differ")
    expected = {"candidate_manifest_sha256": digest(audit/"candidate_pairs.jsonl"),
                "annotation_catalogue_sha256": digest(audit/"annotations_clean.geojson"),
                "audit_metadata_sha256": digest(audit/"metadata.json"),
                "implementation_sha256": digest(ROOT/"Solutions/prepare_curated_pilot.py")}
    if any(batch.get(name) != value for name, value in expected.items()):
        raise ValueError("Prepared batch has stale audit, candidate, catalogue or implementation provenance")
    footprint = box(*frozen_pair["grid"]["bounds"])
    catalogue = read_json(audit/"annotations_clean.geojson")
    for feature in catalogue["features"]:
        props = feature["properties"]
        if props["split"] in ("validation", "test") and footprint.intersects(box(*props["group_footprint_bounds"])):
            raise ValueError("The training core intersects a held-out conservative feature footprint")


def acceptance(validation):
    """Keep the validator's explicit boolean result separate from raw reviews."""
    if (not isinstance(validation, dict) or validation.get("decision_status") not in ("accepted", "rejected")
            or not isinstance(validation.get("training_approved"), bool)
            or validation["training_approved"] != (validation["decision_status"] == "accepted")):
        raise ValueError("Review validator must return consistent explicit approval and decision states")
    return validation["decision_status"] == "accepted"


def build(args, output):
    from curated_pilot_exports import _published_qa, export_example
    from curated_pilot_review import validate_review

    reviews_path = rooted(args.reviews)
    raw = read_json(reviews_path, limit_mib=4)
    reviews = raw.get("reviews") if isinstance(raw, dict) else None
    if (not isinstance(reviews, list) or not 1 <= len(reviews) <= MAX_REVIEWS
            or any(not isinstance(review, dict) for review in reviews)):
        raise ValueError("reviews.json must contain one to 64 explicit review objects")
    paths = [review.get("case_file") for review in reviews]
    if any(not isinstance(path, str) for path in paths) or len(paths) != len(set(paths)):
        raise ValueError("Every review must identify a unique prepared case file")
    cases = [bind_review(review) for review in reviews]
    wanted = {case["pair_id"] for case, _, _ in cases}
    audit = ROOT/"Datasets/AnnotationAudit/v1"
    frozen = {row["pair_id"]: row for row in records(audit/"candidate_pairs.jsonl") if row["pair_id"] in wanted}
    if set(frozen) != wanted:
        raise ValueError("A reviewed case refers to an unknown frozen pair")
    summary_path = ROOT/"Datasets/PixelQA/v1/summary/metadata.json"
    summary = read_json(summary_path)
    if not all(summary["checks"].values()) or summary["processing_errors"]:
        raise ValueError("The source quality assessment must be complete and error-free")
    stats, prepared_hashes, validated = {}, {}, []
    groups, events = set(), set()
    for review, (case, batch, hashes) in zip(reviews, cases):
        pair = frozen[case["pair_id"]]
        validate_frozen_case(case, batch, pair, summary, audit)
        for path, value in batch["source_file_stats"].items():
            if path in stats and stats[path] != value:
                raise ValueError("Prepared batches disagree on source-file statistics")
            stats[path] = value
        prepared_hashes.update(hashes)
        validation = validate_review(case, review)
        if acceptance(validation):
            if case["event_id"] in events or case["group_id"] in groups:
                raise ValueError("Accepted pilot cases repeat an event or geographic group")
            groups.add(case["group_id"]); events.add(case["event_id"])
        validated.append((case, pair, review, validation))
    check_stats(stats)
    (output/"examples").mkdir()
    counts = Counter(); totals = Counter(); manifest_rows = []
    with (output/"review_decisions.jsonl").open("w") as decisions_stream, (output/"training_manifest.jsonl").open("w") as manifest_stream:
        for case, pair, review, validation in validated:
            accepted = acceptance(validation)
            decision_record = {"event_id": case["event_id"], "pair_id": case["pair_id"],
                               "group_id": case["group_id"], "case_file": review["case_file"],
                               "accepted_for_partial_pilot": accepted, "review": review, "validation": validation}
            decisions_stream.write(json.dumps(decision_record, separators=(",", ":"), allow_nan=False)+"\n")
            counts["accepted" if accepted else "rejected"] += 1
            if not accepted:
                continue
            case_dir = rooted(review["case_file"]).parent
            positive = read_mask(case_dir/"positive_interior.tif", pair["grid"])
            stable = read_mask(case_dir/"proposed_control.tif", pair["grid"])
            qa, _ = _published_qa(pair)
            example_dir = output/"examples"/case["event_id"]
            decision = {**review, "validation": validation, "approval_scope": "Explicit reviewed partial pilot supervision"}
            exported = export_example(example_dir, pair, positive, stable, qa, decision)
            artifacts = {"input_features": f"examples/{case['event_id']}/input_features.tif",
                         "supervision_mask": f"examples/{case['event_id']}/supervision_mask.tif",
                         "valid_mask": f"examples/{case['event_id']}/valid_mask.tif",
                         "metadata": f"examples/{case['event_id']}/metadata.json"}
            row = {"event_id": case["event_id"], "pair_id": pair["pair_id"], "group_id": pair["group_id"],
                   "split": "train", "tile": pair["tile"], "pre_date": pair["pre_date"], "post_date": pair["post_date"],
                   "grid": pair["grid"], "artifact_paths": artifacts,
                   "supervised_positive_pixels": exported["supervised_positive_pixels"],
                   "supervised_stable_pixels": exported["supervised_stable_pixels"], "ignored_pixels": exported["ignored_pixels"],
                   "visibility_evidence": validation, "control_scope": "Reviewed ROI inside the same frozen training core and declared local timeline",
                   "source_case_file": review["case_file"], "qa_policy_id": summary["config_id"], "training_approved": True,
                   "approval_scope": "Partial supervised training pilot; no model performance or exact felling-date claim"}
            manifest_stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
            manifest_rows.append(row)
            for name in ("supervised_positive_pixels", "supervised_stable_pixels", "ignored_pixels"):
                totals[name] += exported[name]
            print(f"[curated-pilot] exported {len(manifest_rows)} reviewed examples", flush=True)
    check_stats(stats)
    for path, checksum in prepared_hashes.items():
        if digest(rooted(path)) != checksum:
            raise ValueError("Prepared review evidence changed during export")
    write_report(output/"report.md", counts, totals, validated, manifest_rows,
                 final_output=getattr(args, "output", output))
    output_hashes = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
    sources = ("build_curated_pilot.py", "curated_pilot_exports.py", "curated_pilot_review.py", "prepare_curated_pilot.py")
    metadata = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "review_count": len(reviews), "accepted_example_count": counts["accepted"], "rejected_review_count": counts["rejected"],
                "ready_for_partial_pilot_training": bool(manifest_rows),
                "accepted_geographic_group_count": len(groups), "accepted_event_count": len(events),
                "supervision_pixel_counts": dict(totals), "qa_policy_id": summary["config_id"],
                "reviews_file": relative(reviews_path), "reviews_sha256": digest(reviews_path),
                "prepared_evidence_sha256": prepared_hashes, "source_file_stats": stats,
                "quality_summary_sha256": digest(summary_path), "audit_metadata_sha256": digest(audit/"metadata.json"),
                "candidate_manifest_sha256": digest(audit/"candidate_pairs.jsonl"),
                "annotation_catalogue_sha256": digest(audit/"annotations_clean.geojson"),
                "implementation_sources_sha256": {name: digest(ROOT/"Solutions"/name) for name in sources},
                "output_sha256_excluding_this_metadata": output_hashes,
                "forest_model_training_run": False, "gpu_used": False,
                "checks": {"explicit_hash_bound_reviews_only": True, "all_rejections_retained": True,
                           "exact_frozen_pair_joins": True, "train_split_only": True,
                           "accepted_events_and_groups_unique": True, "no_held_out_feature_footprint_overlap": True,
                           "all_supervision_partial_and_reviewed": True, "source_statistics_unchanged": True},
                "limitations": ["A small strong-signal training pilot is not a representative evaluation set.",
                                "Only explicit loss-interior and stable-ROI pixels are supervised; all others are 255/ignore.",
                                "Visibility bounds depend on the finite local timeline and do not certify the physical felling date.",
                                "Stable controls establish inspected interval/ROI evidence rather than indefinite stability.",
                                "Existing binary training scripts require an ignore-255 loader/loss before consuming this dataset."]}
    write_json(output/"metadata.json", metadata)


def write_report(path, counts, totals, validated, manifest_rows=None, final_output=None):
    final_output = Path(final_output or Path(path).parent).resolve()
    manifests = {row["event_id"]: row for row in (manifest_rows or [])}

    def link(label, target):
        return f"[{label}](<{Path(target).resolve()}>)"

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    lines = ["# Reviewed partial-supervision training pilot", "",
             f"Assessed **{sum(counts.values())}** candidate cases: **{counts['accepted']}** accepted and **{counts['rejected']}** rejected.",
             f"Supervised pixels: **{totals['supervised_positive_pixels']}** clearing interiors and **{totals['supervised_stable_pixels']}** stable-forest controls; **{totals['ignored_pixels']}** ignored.",
             "", "Some candidates were rejected by geometry or quality gates before visual inspection. The assessment count does not imply that every rejected image was visually reviewed.",
             "", "Accepted cases use unique training geographic groups and unchanged frozen 64-by-64 grids. No accepted core intersects a held-out conservative feature footprint.",
             "", "## Accepted examples", "",
             "| Event | Tile | Export pair | Observed interval | Width | Positive / stable pixels | Evidence |",
             "| --- | --- | --- | --- | ---: | ---: | --- |"]
    for case, pair, review, validation in validated:
        if not acceptance(validation):
            continue
        bounds = validation["observed_change_interval"]
        interval = f"({bounds['lower_bound_exclusive']}, {bounds['upper_bound_inclusive']}]"
        exported = manifests.get(case["event_id"], {})
        positives = exported.get("supervised_positive_pixels", case["positive_interior_pixels"])
        stable = exported.get("supervised_stable_pixels", case["control_proposal"]["pixels"])
        case_dir = rooted(review["case_file"]).parent
        evidence = link("timeline", case_dir/"timeline.png")+" · "+link("pair", case_dir/"pair.png")
        lines.append(f"| `{case['event_id']}` | {pair['tile']} | {pair['pre_date']} → {pair['post_date']} | {interval} | {bounds['width_days']} days | {positives} / {stable} | {evidence} |")
    if not counts["accepted"]:
        lines += ["", "No candidate passed all approval requirements."]
    lines += ["", "## Rejected candidates", "",
              "| Event | Reason |", "| --- | --- |"]
    for case, pair, review, validation in validated:
        if acceptance(validation):
            continue
        reasons = validation.get("rejection_reasons") or [review.get("notes", "No accepted review")]
        if not isinstance(reasons, list):
            reasons = [reasons]
        lines.append(f"| `{case['event_id']}` | {cell('; '.join(str(reason) for reason in reasons))} |")
    if not counts["rejected"]:
        lines += ["", "No candidates were rejected."]
    lines += ["", "## Complete evidence", "",
              link(f"All {sum(counts.values())} candidate assessments and per-observation decisions", final_output/"review_decisions.jsonl")+" retain the complete review statements, rejection reasons, artifact hashes, visibility bounds and control evidence.",
              "", link("Training manifest", final_output/"training_manifest.jsonl")+" links each accepted example to its exported arrays and supervision counts."]
    lines += ["", "## Loading and interpretation", "",
              "`input_features.tif` has 27 float32 channels: pre nine, post nine, and post-minus-pre nine. Reflectance bands are scaled independently of the signed NDVI/NDMI indices; signed differences are not clipped.", "",
              "`supervision_mask.tif` uses **1** for explicitly reviewed clearing interiors, **0** for reviewed stable-forest ROIs, and **255** everywhere else. `valid_mask.tif` is exactly `supervision_mask != 255`. The loader and loss must ignore 255; the existing binary trainer is incompatible until that behavior is implemented.", "",
              "Controls remain inside the same original training cores. They are inspected ROI/interval controls, not independent held-out evaluation examples. Windowed first-visible evidence is not an exact felling date or a claim about observations outside the recorded local timeline.", "",
              "This export performs no model training and grants no model-performance claim.", ""]
    Path(path).write_text("\n".join(lines))


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT/"Datasets/CuratedPilot/v1/dataset")
    parser.add_argument("--max-rss-mib", type=int, default=512)
    parser.add_argument("--min-available-gib", type=float, default=1.5)
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error("Use 128–512 MiB RSS, >=1.5 GiB reserve and <=300 seconds")
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    if args.worker_output:
        os.nice(10)
        build(args, args.worker_output)
        import resource
        path = args.worker_output/"metadata.json"
        metadata = read_json(path)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata["worker_high_water_rss_mib"] = round(peak/(2**20 if sys.platform == "darwin" else 1024), 1)
        write_json(path, metadata)
    else:
        run_guarded(args, script_path=__file__, operation="curated-pilot", gdal_cache_mib=32, warp_memory_mib=16)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"[curated-pilot] {error}", file=sys.stderr)
        sys.exit(1)
