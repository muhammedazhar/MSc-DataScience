"""Validate explicit inspection decisions for the training-only curation pilot.

This module uses prepared metadata and reviewer statements only. It does not
infer forest/clearing states from NDVI, annotation dates, or model predictions.
An accepted interval describes the available, inspected local observations.
"""

from copy import deepcopy
from datetime import date
import hashlib
import json


TARGET_STATES = frozenset(("forest", "clearing_visible", "obscured", "uncertain"))
CONTROL_STATES = frozenset(("stable_forest", "obscured", "uncertain"))
QA_COMPONENTS = ("cloud", "cloud_buffer", "shadow_risk", "haze_risk", "invalid")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _string(value, name):
    _require(isinstance(value, str) and bool(value.strip()), f"Missing {name}")
    return value


def _day(value):
    _require(isinstance(value, str), "Observation dates must be ISO strings")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Observation dates must use YYYY-MM-DD") from error
    _require(parsed.isoformat() == value, "Observation dates must use YYYY-MM-DD")
    return parsed


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum,
             f"{name} must be an integer >= {minimum}")
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _prepared(case):
    _require(isinstance(case, dict), "Prepared case must be an object")
    pair = case.get("pair")
    _require(isinstance(pair, dict), "Prepared case requires its frozen pair")
    _require(pair.get("split") == "train", "The reviewed pilot is training-split only")
    for field in ("pair_id", "group_id", "pre_scene_id", "post_scene_id"):
        _string(pair.get(field), field)
    before, after = _day(pair.get("pre_date")), _day(pair.get("post_date"))
    _require(5 <= (after-before).days <= 30,
             "Primary pair dates must retain their forward 5-30 day interval")
    policy = _string(case.get("config_id"), "prepared QA policy")
    timeline = case.get("timeline")
    _require(isinstance(timeline, list) and bool(timeline), "Prepared timeline is empty")
    prepared, days = {}, []
    for row in timeline:
        _require(isinstance(row, dict), "Prepared observation must be an object")
        identifier = _string(row.get("scene_id"), "prepared scene ID")
        day = _day(row.get("date"))
        _require(identifier not in prepared, "Duplicate prepared scene ID")
        _require(day not in days, "Duplicate prepared observation date")
        prepared[identifier] = row
        days.append(day)
    _require(days == sorted(days), "Prepared timeline must be chronological")
    for side in ("pre", "post"):
        identifier = pair[side+"_scene_id"]
        _require(identifier in prepared and prepared[identifier]["date"] == pair[side+"_date"],
                 "Primary endpoint is missing or disagrees with its prepared scene/date")
    if case.get("event_id") is not None:
        _require(case["event_id"] in pair.get("anchor_event_ids", []),
                 "Prepared event is absent from the frozen pair anchors")
    return pair, policy, prepared


def _bound_review(case, review, pair, policy, prepared):
    _require(isinstance(review, dict), "Review must be an object")
    for field in ("visual_review", "temporal_label_review"):
        _require(review.get(field) in ("accepted", "rejected"),
                 f"{field} must be accepted or rejected")
    _require(review.get("qa_policy_id") == policy, "Review QA policy differs from the prepared case")
    _string(review.get("reviewer"), "reviewer")
    _string(review.get("notes"), "review notes")
    for key in ("event_id", "pair_id"):
        expected = case.get(key) if key == "event_id" else pair["pair_id"]
        if review.get(key) is not None:
            _require(expected is not None and review[key] == expected,
                     f"Reviewed {key} differs from the prepared case")
    rows = review.get("observations")
    _require(isinstance(rows, list), "Review observations must be a list")
    inspected, days = {}, []
    for row in rows:
        _require(isinstance(row, dict), "Reviewed observation must be an object")
        identifier = _string(row.get("scene_id"), "reviewed scene ID")
        day = _day(row.get("date"))
        _require(identifier not in inspected, "Duplicate reviewed scene ID")
        _require(day not in days, "Duplicate reviewed observation date")
        _require(identifier in prepared and row["date"] == prepared[identifier]["date"],
                 "Review scene/date does not match the prepared observation")
        _require(row.get("target_state") in TARGET_STATES, "Unsupported reviewed target state")
        _require(row.get("control_state") in CONTROL_STATES, "Unsupported reviewed control state")
        _string(row.get("notes"), "observation inspection notes")
        inspected[identifier] = row
        days.append(day)
    _require(set(inspected) == set(prepared), "Review must cover every prepared observation exactly")
    _require(days == sorted(days), "Reviewed observations must be chronological")
    return inspected


def _clear_claim(prepared, region_name, expected_pixels, policy):
    _require(prepared.get("metadata_usable") is True,
             "A clear visual claim needs usable source metadata")
    observation = prepared.get("observation")
    _require(isinstance(observation, dict) and observation.get("config_id") == policy,
             "A clear visual claim needs its bound frozen-policy observation cache")
    metrics = prepared.get(region_name)
    _require(isinstance(metrics, dict), "A clear visual claim has no prepared ROI quality metrics")
    count = _integer(metrics.get("region_pixels"), "ROI region_pixels", minimum=1)
    _require(count == expected_pixels, "ROI quality denominator differs from the prepared ROI")
    unusable = _integer(metrics.get("unusable_pixels"), "ROI unusable_pixels")
    _require(unusable == 0, "A clear visual claim intersects numeric QA flags")
    for name in QA_COMPONENTS:
        key = name+"_pixels"
        if key in metrics:
            _require(_integer(metrics[key], key) == 0,
                     "A clear visual claim intersects a numeric QA component")
    for key in ("valid_pixels", "usable_pixels", "denominator_pixels"):
        if key in metrics:
            _require(_integer(metrics[key], key) == count,
                     "A clear visual claim has inconsistent ROI validity/denominator")


def validate_review(case, review):
    """Return a bound, enriched decision or raise ValueError for unsafe approval.

    Required prepared keys follow ``prepare_curated_pilot``: ``pair``,
    ``config_id``, ``timeline``, ``target_pixels``, ``positive_interior_pixels``
    and ``control_proposal``. Every prepared scene/date requires an explicit
    target/control state and notes, including obscured/unavailable observations.
    Rejections preserve their evidence and never derive an onset interval.
    """
    pair, policy, prepared = _prepared(case)
    inspected = _bound_review(case, review, pair, policy, prepared)
    result = deepcopy(review)
    result.update({"pair_id": pair["pair_id"], "group_id": pair["group_id"], "split": "train",
                   "prepared_case_sha256": _hash(case), "training_approved": False,
                   "observed_change_interval": None, "controls": None,
                   "approval_scope": "Small partially supervised training pilot only"})
    if case.get("event_id") is not None:
        result["event_id"] = case["event_id"]
    if review["visual_review"] != "accepted" or review["temporal_label_review"] != "accepted":
        result["decision_status"] = "rejected"
        result["rejection_reasons"] = deepcopy(review.get("rejection_reasons", [review["notes"]]))
        return result

    target_pixels = _integer(case.get("target_pixels"), "target_pixels", minimum=1)
    interior = _integer(case.get("positive_interior_pixels"), "positive_interior_pixels", minimum=1)
    _require(interior <= target_pixels, "Positive interior exceeds its prepared target")
    proposal = case.get("control_proposal")
    _require(isinstance(proposal, dict) and proposal.get("proposal_found") is True,
             "Accepted review needs a prepared stable-control ROI")
    control_pixels = _integer(proposal.get("pixels"), "control pixels", minimum=1)
    for identifier, row in inspected.items():
        if row["target_state"] in ("forest", "clearing_visible"):
            _clear_claim(prepared[identifier], "event_metrics", target_pixels, policy)
        if row["control_state"] == "stable_forest":
            _clear_claim(prepared[identifier], "control_metrics", control_pixels, policy)

    pre, post = inspected[pair["pre_scene_id"]], inspected[pair["post_scene_id"]]
    _require(pre["target_state"] == "forest", "Primary pre-image must explicitly show forest")
    _require(post["target_state"] == "clearing_visible", "Primary post-image must explicitly show clearing")
    _require(pre["control_state"] == post["control_state"] == "stable_forest",
             "Stable-control ROI must be clear forest at both primary endpoints")

    chronological = list(inspected.values())
    clearings = [row for row in chronological if row["target_state"] == "clearing_visible"]
    first = clearings[0]
    first_day = _day(first["date"])
    later = [row for row in clearings if _day(row["date"]) > first_day]
    _require(bool(later), "Clearing needs a later clear observation confirming persistence")
    _require(not any(row["target_state"] == "forest" and _day(row["date"]) >= first_day
                     for row in chronological), "Forest after first clearing contradicts a persistent-loss interval")
    forests = [row for row in chronological
               if row["target_state"] == "forest" and _day(row["date"]) < first_day]
    _require(bool(forests), "Clearing has no prior observed forest bound")
    last = forests[-1]
    last_day = _day(last["date"])
    intermediate = [{"scene_id": row["scene_id"], "date": row["date"],
                     "target_state": row["target_state"], "notes": row["notes"]}
                    for row in chronological if last_day < _day(row["date"]) < first_day]
    _require(all(row["target_state"] in ("obscured", "uncertain") for row in intermediate),
             "Intermediate states contradict the derived observation bounds")

    stable = [row for row in chronological if row["control_state"] == "stable_forest"]
    _require(len(stable) >= 3, "Stable control needs at least three clear observed dates")
    result["observed_change_interval"] = {
        "last_observed_forest_date": last["date"], "last_observed_forest_scene_id": last["scene_id"],
        "first_observed_clearing_date": first["date"], "first_observed_clearing_scene_id": first["scene_id"],
        "lower_bound_exclusive": last["date"], "upper_bound_inclusive": first["date"],
        "interval_notation": "(last_observed_forest_date, first_observed_clearing_date]",
        "width_days": (first_day-last_day).days,
        "intervening_uncertain_or_obscured_observations": intermediate,
        "persistence_scene_ids": [row["scene_id"] for row in later],
        "interpretation": "Bounded by inspected available local observations; not an exact felling date or globally earliest sighting",
    }
    result["controls"] = {
        "stable_scene_ids": [row["scene_id"] for row in stable],
        "clear_observation_count": len(stable), "supervised_pixels": control_pixels,
        "observed_stability_start": stable[0]["date"], "observed_stability_end": stable[-1]["date"],
        "primary_endpoints_stable": True,
        "interpretation": "Visually reviewed ROI stability over inspected dates only; all other pixels remain unreviewed",
    }
    result["training_approved"] = True
    result["decision_status"] = "accepted"
    return result
