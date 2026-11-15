#!/usr/bin/env python3
"""Build an event-level review queue from completed, frozen pixel-QA results.

This is annotation-based data curation, not an inference or event-onset model.
It reads JSONL metadata only and preserves every input pair's identity/reasons.
"""

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MAX_LINE_BYTES = 512 * 1024
MAX_RECORDS = 20_000
TOP_K = 3
STATUS_ORDER = {
    "passed_automatic_screen_pending_visual_review": 0,
    "needs_visual_review": 1,
    "needs_label_review": 2,
    "excluded_by_automatic_quality_screen": 3,
    "processing_error": 4,
}
PROPOSAL_STATUSES = frozenset((
    "passed_automatic_screen_pending_visual_review", "needs_visual_review",
))


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON number is unsupported: {value}")


def json_records(path):
    """Bound each streamed line; never retain rich per-observation arrays."""
    with Path(path).open("rb") as stream:
        for index in range(MAX_RECORDS + 1):
            line = stream.readline(MAX_LINE_BYTES + 1)
            if not line:
                return
            if index == MAX_RECORDS:
                raise ValueError(f"Input exceeds the {MAX_RECORDS}-record bound: {path}")
            if len(line) > MAX_LINE_BYTES:
                raise ValueError(f"Input line exceeds the metadata size bound: {path}")
            if not line.strip():
                raise ValueError(f"Unexpected blank JSONL record: {path}:{index+1}")
            value = json.loads(line, parse_constant=_invalid_constant)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object: {path}:{index+1}")
            yield value


def iso_date(value):
    if not isinstance(value, str):
        raise ValueError("Observation dates must be ISO date strings")
    parsed = date.fromisoformat(value)
    if value != parsed.isoformat():
        raise ValueError("Observation dates must use YYYY-MM-DD")
    return parsed


def _identifier(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing or invalid {field}")
    return value


def _identifiers(values, field):
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list")
    checked = [_identifier(value, field) for value in values]
    if len(checked) != len(set(checked)):
        raise ValueError(f"Duplicate identifier in {field}")
    return checked


def load_annotation_index(path):
    events, pair_anchors = {}, defaultdict(set)
    for record in json_records(path):
        identifier = _identifier(record.get("event_id"), "event_id")
        if identifier in events:
            raise ValueError("Duplicate event in the annotation manifest")
        split = record.get("split")
        if split not in ("train", "validation", "test"):
            raise ValueError("Invalid frozen annotation split")
        candidates = _identifiers(record.get("candidate_pair_ids"), "candidate_pair_ids")
        events[identifier] = {
            "event_id": identifier,
            "group_id": _identifier(record.get("group_id"), "group_id"),
            "split": split,
            "tile": _identifier(record.get("tile"), "tile"),
            "observation_date": iso_date(record.get("observation_date")).isoformat(),
            "frozen_candidate_pair_count": len(candidates),
        }
        for pair_id in candidates:
            pair_anchors[pair_id].add(identifier)
    if not events:
        raise ValueError("The annotation manifest is empty")
    return events, pair_anchors


def compact_result(record, events, pair_anchors):
    """Validate the bidirectional join and keep ranking/trace essentials only."""
    pair_id = _identifier(record.get("pair_id"), "pair_id")
    anchors = _identifiers(record.get("anchor_event_ids"), "anchor_event_ids")
    if not anchors or pair_id not in pair_anchors:
        raise ValueError("A quality pair is absent from the frozen annotation manifest")
    if set(anchors) != pair_anchors[pair_id]:
        raise ValueError("Quality pair anchors disagree with the frozen annotation links")
    if record.get("training_approved") is not False:
        raise ValueError("Quality screening must not supply training approval")
    status = record.get("quality_status")
    if status not in STATUS_ORDER:
        raise ValueError(f"Unknown quality status: {status}")
    reasons = record.get("quality_reasons")
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise ValueError("Quality reasons must be a list of strings")
    pre, post = iso_date(record.get("pre_date")), iso_date(record.get("post_date"))
    interval = (post-pre).days
    if not 5 <= interval <= 30:
        raise ValueError("Quality pair is outside the frozen 5–30 day interval")
    for identifier in anchors:
        event = events[identifier]
        if any(record.get(field) != event[field] for field in ("group_id", "split", "tile")):
            raise ValueError("Quality pair group, split or tile disagrees with its frozen event")
        if not pre < iso_date(event["observation_date"]) <= post:
            raise ValueError("Quality pair does not bracket its annotation observation date")
    return {
        "pair_id": pair_id, "group_id": record["group_id"], "split": record["split"],
        "tile": record["tile"], "anchor_event_ids": sorted(anchors),
        "pre_date": pre.isoformat(), "post_date": post.isoformat(), "interval_days": interval,
        "quality_status": status, "quality_reasons": list(reasons),
        "config_id": _identifier(record.get("config_id"), "config_id"),
        "batch_path": record.get("batch_path"), "qa_mask_file": record.get("qa_mask_file"),
        "preview_file": record.get("preview_file"),
        "visual_review": record.get("visual_review", "unreviewed"),
        "training_approved": False,
    }


def rank_key(candidate, annotation_date):
    observed = iso_date(annotation_date)
    pre, post = iso_date(candidate["pre_date"]), iso_date(candidate["post_date"])
    if not pre < observed <= post:
        raise ValueError("A ranked pair does not bracket the annotation observation date")
    return (STATUS_ORDER[candidate["quality_status"]], post.toordinal(),
            (observed-pre).days, candidate["interval_days"], candidate["pair_id"])


def event_review_record(event, candidates):
    ordered = sorted(candidates, key=lambda row: rank_key(row, event["observation_date"]))
    statuses = Counter(row["quality_status"] for row in ordered)
    eligible = [row for row in ordered if row["quality_status"] in PROPOSAL_STATUSES]
    if eligible:
        queue_status = "pending_visual_and_temporal_review"
    elif statuses["needs_label_review"]:
        queue_status = "requires_label_review"
    elif statuses["excluded_by_automatic_quality_screen"]:
        queue_status = "requires_alternate_imagery_or_quality_review"
    else:
        queue_status = "requires_processing_recovery"
    top = [{**row, "eligible_for_review_proposal": row["quality_status"] in PROPOSAL_STATUSES}
           for row in ordered[:TOP_K]]
    return {
        **event, "queue_status": queue_status,
        "screened_candidate_count": len(ordered), "eligible_review_candidate_count": len(eligible),
        "candidate_pair_ids": [row["pair_id"] for row in ordered],
        "candidate_quality": [{"pair_id": row["pair_id"], "quality_status": row["quality_status"],
                               "quality_reasons": row["quality_reasons"]} for row in ordered],
        "quality_status_counts": dict(sorted(statuses.items())),
        "top_candidates": top,
        "proposed_pair_ids": [row["pair_id"] for row in top if row["eligible_for_review_proposal"]],
        "visual_review": "unreviewed", "temporal_label_review": "unreviewed",
        "training_approved": False,
        "date_interpretation": "Annotation-based candidate curation; not an inferred onset or exact felling date",
    }


def build_review_queue(annotation_path, quality_path, output, summary_metadata=None):
    """Preserve every screened anchor and pair, including excluded/error cases."""
    output = Path(output)
    events, pair_anchors = load_annotation_index(annotation_path)
    pairs, event_links, policy_ids = {}, defaultdict(list), set()
    with (output/"quality_result_index.jsonl").open("w") as index_stream:
        for record in json_records(quality_path):
            row = compact_result(record, events, pair_anchors)
            identifier = row["pair_id"]
            if identifier in pairs:
                raise ValueError("Duplicate quality pair in completed results")
            pairs[identifier] = row
            policy_ids.add(row["config_id"])
            for event_id in row["anchor_event_ids"]:
                event_links[event_id].append(identifier)
            index_stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
    if not pairs:
        raise ValueError("No screened quality results were supplied")
    if len(policy_ids) != 1:
        raise ValueError("Quality results combine different frozen policies")
    if summary_metadata is not None:
        checks = summary_metadata.get("checks", {})
        required = ("exact_candidate_coverage", "no_duplicate_pairs", "single_frozen_quality_policy", "splits_preserved")
        if not all(checks.get(key) is True for key in required):
            raise ValueError("The completed quality summary does not pass its publication checks")
        if (summary_metadata.get("pair_count") != len(pairs)
                or summary_metadata.get("config_id") != next(iter(policy_ids))
                or summary_metadata.get("training_approved") is not False):
            raise ValueError("Quality results disagree with their completed summary")
    queue_counts, split_counts, proposals = Counter(), defaultdict(Counter), 0
    with (output/"review_queue.jsonl").open("w") as stream:
        for event_id in sorted(event_links):
            queue = event_review_record(events[event_id], [pairs[key] for key in event_links[event_id]])
            queue_counts[queue["queue_status"]] += 1
            split_counts[queue["split"]][queue["queue_status"]] += 1
            proposals += bool(queue["proposed_pair_ids"])
            stream.write(json.dumps(queue, separators=(",", ":"), allow_nan=False)+"\n")
    metadata = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "quality_manifest_sha256": digest(quality_path), "annotation_manifest_sha256": digest(annotation_path),
        "implementation_sha256": digest(__file__), "config_id": next(iter(policy_ids)),
        "input_quality_pair_count": len(pairs), "screened_anchor_event_count": len(event_links),
        "catalogue_event_count": len(events), "catalogue_events_outside_queue": len(events)-len(event_links),
        "event_pair_link_count": sum(len(ids) for ids in event_links.values()),
        "events_with_review_proposals": proposals, "top_k": TOP_K,
        "queue_status_counts": dict(sorted(queue_counts.items())),
        "queue_status_counts_by_split": {split: dict(sorted(counts.items())) for split, counts in sorted(split_counts.items())},
        "ranking": ["quality-status priority", "earliest post observation on/after annotation date",
                    "closest preceding observation", "shortest interval", "stable pair ID"],
        "training_approved": False,
        "checks": {"all_screened_pairs_retained": True, "all_screened_anchors_retained": True,
                   "exact_annotation_links_validated": True, "frozen_splits_preserved": True,
                   "no_forest_model_or_dice_scores_used": True},
        "limitations": ["The queue curates labelled examples; it does not predict onset or exact felling dates.",
                        "Automatic quality exclusions remain available with their reasons and can require alternate imagery or review.",
                        "Events without screened candidates remain in the original annotation manifest and are counted outside this queue.",
                        "No visual, temporal-label or training approval is granted automatically."],
    }
    write_json(output/"metadata.json", metadata)
    write_report(output/"review_report.md", metadata)
    return metadata


def write_report(path, metadata):
    lines = ["# Event-level pixel-quality review queue", "",
             f"Retained **{metadata['input_quality_pair_count']:,}** screened pairs and produced one queue entry for each of **{metadata['screened_anchor_event_count']:,}** represented anchor events.",
             f"**{metadata['catalogue_events_outside_queue']:,}** catalogue events have no screened result in this input and remain in the original annotation manifest.",
             "", "| Queue status | Anchor events |", "| --- | ---: |"]
    lines += [f"| {status} | {count} |" for status, count in metadata["queue_status_counts"].items()]
    lines += ["", "Each queue entry retains all candidate IDs, statuses and reasons, plus the top three ranked candidates. The compact quality-result index preserves every input pair and links to its original QA artifacts. Automatic exclusions and processing errors are not dropped.",
              "", "Ranking uses quality status first, then the earliest post-image on or after the known annotation observation date, the closest preceding image, shortest interval, and stable pair ID. Forest-model predictions, Dice scores and other performance scores are not used.",
              "", "These are proposals for visual and temporal-label review. An event with only excluded candidates has no eligible proposal and is flagged for alternate imagery or quality review. Empty-label and processing-recovery cases remain distinct.",
              "", "This is annotation-based dataset curation, not a date-inference algorithm. The queue does not establish the first physical clearing date, pristine forest in every earlier pixel, or automatic training approval.", ""]
    Path(path).write_text("\n".join(lines))


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-manifest", type=Path, default=ROOT/"Datasets/PixelQA/v1/summary/quality_manifest.jsonl")
    parser.add_argument("--annotation-manifest", type=Path, default=ROOT/"Datasets/AnnotationAudit/v1/annotation_manifest.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT/"Datasets/PixelQA/v1/review")
    parser.add_argument("--max-rss-mib", type=int, default=512)
    parser.add_argument("--min-available-gib", type=float, default=1.5)
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 768 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error("Use 128–768 MiB RSS, >=1.5 GiB available reserve and <=300 seconds")
    for key in THREAD_VARIABLES:
        os.environ[key] = "1"
    if args.worker_output:
        os.nice(10)
        summary_path = args.quality_manifest.parent/"metadata.json"
        summary = json.loads(summary_path.read_text(), parse_constant=_invalid_constant)
        build_review_queue(args.annotation_manifest, args.quality_manifest, args.worker_output, summary)
        import resource
        metadata_path = args.worker_output/"metadata.json"
        metadata = json.loads(metadata_path.read_text())
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata["worker_high_water_rss_mib"] = round(peak/(2**20 if sys.platform == "darwin" else 1024), 1)
        write_json(metadata_path, metadata)
    else:
        run_guarded(args, script_path=__file__, operation="quality-review", gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[quality-review] {error}", file=sys.stderr)
        sys.exit(1)
