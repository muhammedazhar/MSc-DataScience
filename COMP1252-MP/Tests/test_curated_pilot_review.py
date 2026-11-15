"""Small decision-evidence fixtures; no imagery, models or raster packages."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Solutions"))
from curated_pilot_review import validate_review


def fixture():
    days = ("2019-06-01", "2019-06-06", "2019-06-11", "2019-06-16", "2019-06-21", "2019-06-26")
    states = ("forest", "forest", "obscured", "clearing_visible", "uncertain", "clearing_visible")
    timeline, inspected = [], []
    for index, (day, state) in enumerate(zip(days, states)):
        metrics = {"region_pixels": 40, "unusable_pixels": 0, "invalid_pixels": 0,
                   "valid_pixels": 40, "usable_pixels": 40, "denominator_pixels": 40}
        timeline.append({"scene_id": f"scene-{index}", "date": day, "metadata_usable": True,
                         "event_metrics": deepcopy(metrics), "control_metrics": {**metrics, "region_pixels": 64,
                         "valid_pixels": 64, "usable_pixels": 64, "denominator_pixels": 64},
                         "observation": {"config_id": "frozen-policy"}})
        inspected.append({"scene_id": f"scene-{index}", "date": day, "target_state": state,
                          "control_state": "obscured" if index == 2 else "stable_forest",
                          "notes": f"Explicit inspection of {day}."})
    case = {"event_id": "event-1", "config_id": "frozen-policy", "target_pixels": 40,
            "positive_interior_pixels": 16, "control_proposal": {"proposal_found": True, "pixels": 64},
            "pair": {"pair_id": "pair-1", "group_id": "group-1", "split": "train",
                     "anchor_event_ids": ["event-1"], "pre_scene_id": "scene-1", "post_scene_id": "scene-3",
                     "pre_date": days[1], "post_date": days[3]}, "timeline": timeline,
            "annotation_date": "2019-06-12"}
    review = {"visual_review": "accepted", "temporal_label_review": "accepted", "reviewer": "assistant",
              "qa_policy_id": "frozen-policy", "notes": "Reviewed all target and control previews.",
              "observations": inspected}
    return case, review


class ExplicitReviewTests(unittest.TestCase):
    def test_bounds_come_from_inspected_states_and_keep_obscured_dates(self):
        case, review = fixture()
        result = validate_review(case, review)
        interval = result["observed_change_interval"]
        self.assertEqual(interval["lower_bound_exclusive"], "2019-06-06")
        self.assertEqual(interval["upper_bound_inclusive"], "2019-06-16")
        self.assertEqual(interval["width_days"], 10)
        self.assertEqual(interval["persistence_scene_ids"], ["scene-5"])
        self.assertEqual(interval["intervening_uncertain_or_obscured_observations"][0]["scene_id"], "scene-2")
        self.assertEqual(result["controls"]["clear_observation_count"], 5)
        self.assertIs(result["training_approved"], True)

    def test_annotation_date_ndvi_and_existing_derived_fields_cannot_supply_bounds(self):
        case, review = fixture()
        expected = validate_review(case, review)["observed_change_interval"]
        case["annotation_date"] = "2001-01-01"
        for row in case["timeline"]:
            row["event_metrics"]["usable_median_ndvi"] = 999
        review["observed_change_interval"] = {"lower_bound_exclusive": "2001-01-01"}
        self.assertEqual(validate_review(case, review)["observed_change_interval"], expected)

    def test_an_earlier_visible_observation_tightens_the_upper_bound(self):
        case, review = fixture()
        review["observations"][2]["target_state"] = "clearing_visible"
        result = validate_review(case, review)
        self.assertEqual(result["observed_change_interval"]["upper_bound_inclusive"], "2019-06-11")
        self.assertEqual(result["observed_change_interval"]["persistence_scene_ids"], ["scene-3", "scene-5"])

    def test_exact_scene_date_coverage_rejects_missing_duplicate_unknown_and_misdated_rows(self):
        for change in ("missing", "duplicate", "unknown", "wrong_date"):
            case, review = fixture()
            if change == "missing": review["observations"].pop()
            elif change == "duplicate": review["observations"].append(deepcopy(review["observations"][0]))
            elif change == "unknown": review["observations"][0]["scene_id"] = "not-prepared"
            else: review["observations"][0]["date"] = "2019-05-31"
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_review(case, review)

    def test_reversed_pair_or_timeline_order_is_rejected(self):
        for change in ("pair", "prepared_order", "review_order"):
            case, review = fixture()
            if change == "pair":
                case["pair"]["pre_date"], case["pair"]["post_date"] = case["pair"]["post_date"], case["pair"]["pre_date"]
            elif change == "prepared_order": case["timeline"].reverse()
            else: review["observations"].reverse()
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_review(case, review)

    def test_held_out_or_wrong_policy_or_event_cannot_be_approved(self):
        for change in ("split", "review_policy", "cache_policy", "event"):
            case, review = fixture()
            if change == "split": case["pair"]["split"] = "test"
            elif change == "review_policy": review["qa_policy_id"] = "other"
            elif change == "cache_policy": case["timeline"][1]["observation"]["config_id"] = "other"
            else: review["event_id"] = "other-event"
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_review(case, review)

    def test_clear_claims_cannot_override_numeric_quality_or_missing_metadata(self):
        for change in ("target_flags", "control_flags", "metadata", "metrics", "component", "validity"):
            case, review = fixture()
            row = case["timeline"][1]
            if change == "target_flags": row["event_metrics"]["unusable_pixels"] = 1
            elif change == "control_flags": row["control_metrics"]["unusable_pixels"] = 1
            elif change == "metadata": row["metadata_usable"] = False
            elif change == "metrics": row["event_metrics"] = None
            elif change == "component": row["event_metrics"]["haze_risk_pixels"] = 1
            else: row["event_metrics"]["valid_pixels"] = 39
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_review(case, review)

    def test_obscured_observation_can_remain_explicitly_unknown_with_no_numeric_metrics(self):
        case, review = fixture()
        row = case["timeline"][2]
        row.update(metadata_usable=False, event_metrics=None, control_metrics=None, observation=None)
        result = validate_review(case, review)
        self.assertIs(result["training_approved"], True)
        self.assertEqual(result["observed_change_interval"]["width_days"], 10)

    def test_primary_endpoints_require_explicit_forest_then_clearing(self):
        for index, state in ((1, "obscured"), (3, "uncertain")):
            case, review = fixture()
            review["observations"][index]["target_state"] = state
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "Primary"):
                validate_review(case, review)

    def test_one_clearing_without_later_persistence_is_insufficient(self):
        case, review = fixture()
        review["observations"][5]["target_state"] = "uncertain"
        with self.assertRaisesRegex(ValueError, "persistence"):
            validate_review(case, review)

    def test_forest_after_clearing_is_a_contradiction(self):
        case, review = fixture()
        review["observations"][4]["target_state"] = "forest"
        with self.assertRaisesRegex(ValueError, "Forest after first clearing"):
            validate_review(case, review)

    def test_no_prior_forest_cannot_be_replaced_by_a_catalogue_date(self):
        case, review = fixture()
        review["observations"][0]["target_state"] = "clearing_visible"
        with self.assertRaisesRegex(ValueError, "Forest after first clearing"):
            validate_review(case, review)

    def test_control_requires_both_primary_endpoints_and_three_clear_dates(self):
        for change in ("endpoint", "too_few"):
            case, review = fixture()
            if change == "endpoint": review["observations"][1]["control_state"] = "uncertain"
            else:
                for index in (0, 4, 5): review["observations"][index]["control_state"] = "uncertain"
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_review(case, review)

    def test_empty_or_inconsistent_review_roi_cannot_be_approved(self):
        for change in ("positive", "control", "denominator"):
            case, review = fixture()
            if change == "positive": case["positive_interior_pixels"] = 0
            elif change == "control": case["control_proposal"]["proposal_found"] = False
            else: case["timeline"][1]["control_metrics"]["region_pixels"] = 63
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_review(case, review)

    def test_rejection_retains_reasons_without_imputing_dates_or_controls(self):
        case, review = fixture()
        review.update(visual_review="rejected", notes="Haze obscures the clearing.",
                      rejection_reasons=["suspected_haze"], training_approved=True)
        for row in review["observations"]:
            row["target_state"] = "uncertain"
            row["control_state"] = "uncertain"
        result = validate_review(case, review)
        self.assertEqual(result["rejection_reasons"], ["suspected_haze"])
        self.assertIs(result["observed_change_interval"], None)
        self.assertIs(result["controls"], None)
        self.assertIs(result["training_approved"], False)

    def test_inputs_are_not_mutated_and_decision_is_bound_to_the_prepared_case(self):
        case, review = fixture()
        originals = deepcopy((case, review))
        result = validate_review(case, review)
        self.assertEqual((case, review), originals)
        self.assertEqual(len(result["prepared_case_sha256"]), 64)
        result["observations"][0]["notes"] = "Changed returned copy"
        self.assertEqual((case, review), originals)


if __name__ == "__main__":
    unittest.main()
