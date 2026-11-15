"""Integrity regressions for binding reviews and preserving rejection evidence."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SOLUTIONS = Path(__file__).resolve().parents[1]/"Solutions"
sys.path.insert(0, str(SOLUTIONS))
spec = importlib.util.spec_from_file_location("build_curated_pilot", SOLUTIONS/"build_curated_pilot.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ReviewPublicationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.patch = mock.patch.object(builder, "ROOT", self.root)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.audit = self.root/"Datasets/AnnotationAudit/v1"
        self.audit.mkdir(parents=True)
        (self.root/"Solutions").mkdir()
        for name in ("build_curated_pilot.py", "curated_pilot_exports.py", "curated_pilot_review.py", "prepare_curated_pilot.py"):
            (self.root/"Solutions"/name).write_text("# frozen fixture source\n")
        self.pair = {"pair_id": "pair-demo", "group_id": "group-demo", "split": "train", "tile": "36UXA",
                     "anchor_event_ids": ["event-demo"], "pre_date": "2018-07-31", "post_date": "2018-08-05",
                     "pre_scene_id": "pre", "post_scene_id": "post",
                     "grid": {"crs": "EPSG:32636", "width": 64, "height": 64,
                              "transform": [10, 0, 0, 0, -10, 640], "bounds": [0, 0, 640, 640]}}
        (self.audit/"candidate_pairs.jsonl").write_text(json.dumps(self.pair)+"\n")
        (self.audit/"metadata.json").write_text('{"audit":"frozen"}\n')
        self.catalogue = {"features": [{"properties": {"split": "test", "group_footprint_bounds": [1000, 1000, 1680, 1680]}}]}
        builder.write_json(self.audit/"annotations_clean.geojson", self.catalogue)
        self.summary = {"checks": {"coverage": True}, "processing_errors": [], "config_id": "policy-demo"}
        summary_path = self.root/"Datasets/PixelQA/v1/summary/metadata.json"
        summary_path.parent.mkdir(parents=True)
        builder.write_json(summary_path, self.summary)
        self.case = {"event_id": "event-demo", "pair_id": "pair-demo", "group_id": "group-demo",
                     "split": "train", "config_id": "policy-demo", "pair": deepcopy(self.pair)}
        self.case_path = self.root/"Datasets/CuratedPilot/v1/review_parts/00000/event-demo/case.json"
        self.case_path.parent.mkdir(parents=True)
        builder.write_json(self.case_path, self.case)
        for name in ("pair.png", "timeline.png", "positive_interior.tif", "proposed_control.tif", "target_mask.tif"):
            (self.case_path.parent/name).write_bytes(b"inspection fixture "+name.encode())
        self.batch = {"cases": [{"event_id": "event-demo", "pair_id": "pair-demo"}],
                      "config_id": "policy-demo", "source_file_stats": {}}
        self.sync_batch()
        self.review = {"case_file": str(self.case_path.relative_to(self.root)),
                       "visual_review": "rejected", "temporal_label_review": "rejected",
                       "reviewer": "assistant", "notes": "Not visually reviewed: zero conservative interior",
                       "qa_policy_id": "policy-demo", "observations": [],
                       "case_sha256": builder.digest(self.case_path),
                       "timeline_png_sha256": builder.digest(self.case_path.parent/"timeline.png"),
                       "pair_png_sha256": builder.digest(self.case_path.parent/"pair.png")}

    def sync_batch(self):
        self.batch.update({"candidate_manifest_sha256": builder.digest(self.audit/"candidate_pairs.jsonl"),
                           "annotation_catalogue_sha256": builder.digest(self.audit/"annotations_clean.geojson"),
                           "audit_metadata_sha256": builder.digest(self.audit/"metadata.json"),
                           "implementation_sha256": builder.digest(self.root/"Solutions/prepare_curated_pilot.py")})
        builder.write_json(self.case_path.parent.parent/"metadata.json", self.batch)

    def test_rejected_review_still_requires_all_inspected_artifact_hashes(self):
        case, batch, hashes = builder.bind_review(self.review)
        self.assertEqual(case, self.case)
        self.assertEqual(batch, self.batch)
        self.assertEqual(len(hashes), 7)
        for name in ("case_sha256", "timeline_png_sha256", "pair_png_sha256"):
            wrong = {key: value for key, value in self.review.items() if key != name}
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "missing or stale"):
                builder.bind_review(wrong)

    def test_modified_preview_invalidates_review_instead_of_rebinding_approval(self):
        (self.case_path.parent/"timeline.png").write_bytes(b"different preview")
        with self.assertRaisesRegex(ValueError, "timeline_png_sha256"):
            builder.bind_review(self.review)

    def test_frozen_join_rejects_changed_crop_even_when_pair_id_is_unchanged(self):
        changed = deepcopy(self.case)
        changed["pair"]["grid"]["transform"][2] += 20
        with self.assertRaisesRegex(ValueError, "exact frozen candidate"):
            builder.validate_frozen_case(changed, self.batch, self.pair, self.summary, self.audit)

    def test_per_feature_held_out_envelope_overlap_blocks_training_export(self):
        builder.validate_frozen_case(self.case, self.batch, self.pair, self.summary, self.audit)
        self.catalogue["features"][0]["properties"]["group_footprint_bounds"] = [630, 100, 1310, 780]
        builder.write_json(self.audit/"annotations_clean.geojson", self.catalogue)
        self.sync_batch()
        with self.assertRaisesRegex(ValueError, "held-out conservative feature"):
            builder.validate_frozen_case(self.case, self.batch, self.pair, self.summary, self.audit)

    def test_inconsistent_validator_approval_cannot_publish(self):
        for wrong in ({"decision_status": "accepted", "training_approved": False},
                      {"decision_status": "rejected", "training_approved": True},
                      {"accepted": True}):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, "consistent explicit"):
                builder.acceptance(wrong)

    def test_rejection_only_run_preserves_evidence_and_does_not_claim_ready_dataset(self):
        reviews_path = self.root/"reviews.json"
        builder.write_json(reviews_path, {"reviews": [self.review]})
        output = self.root/"dataset"
        output.mkdir()
        validation = {**self.review, "decision_status": "rejected", "training_approved": False,
                      "observed_change_interval": None, "controls": None,
                      "rejection_reasons": [self.review["notes"]]}
        with mock.patch("curated_pilot_review.validate_review", return_value=validation), \
                mock.patch("curated_pilot_exports.export_example") as exporter:
            builder.build(SimpleNamespace(reviews=reviews_path), output)
        exporter.assert_not_called()
        metadata = builder.read_json(output/"metadata.json")
        self.assertEqual(metadata["rejected_review_count"], 1)
        self.assertEqual(metadata["accepted_example_count"], 0)
        self.assertFalse(metadata["ready_for_partial_pilot_training"])
        decision = json.loads((output/"review_decisions.jsonl").read_text())
        self.assertEqual(decision["review"], self.review)
        self.assertIsNone(decision["validation"]["observed_change_interval"])
        self.assertEqual((output/"training_manifest.jsonl").read_text(), "")

    def test_report_summarizes_evidence_and_links_to_final_output_not_temporary_stage(self):
        case = {**self.case, "positive_interior_pixels": 7, "control_proposal": {"pixels": 16}}
        review = {**self.review, "visual_review": "accepted", "temporal_label_review": "accepted"}
        validation = {"decision_status": "accepted", "training_approved": True,
                      "observed_change_interval": {"lower_bound_exclusive": "2018-07-31",
                                                   "upper_bound_inclusive": "2018-08-05", "width_days": 5}}
        stage = self.root/".temporary-stage"
        stage.mkdir()
        final = self.root/"final-dataset"
        builder.write_report(stage/"report.md", {"accepted": 1, "rejected": 0},
                             {"supervised_positive_pixels": 7, "supervised_stable_pixels": 16, "ignored_pixels": 4073},
                             [(case, self.pair, review, validation)], final_output=final)
        report = (stage/"report.md").read_text()
        self.assertIn("Assessed **1** candidate cases", report)
        self.assertIn("2018-07-31 → 2018-08-05", report)
        self.assertIn("7 / 16", report)
        self.assertIn(str(final.resolve()/"review_decisions.jsonl"), report)
        self.assertIn(str(self.case_path.parent.resolve()/"timeline.png"), report)
        self.assertNotIn(str(stage.resolve()/"review_decisions.jsonl"), report)
        self.assertNotIn("```json", report)


if __name__ == "__main__":
    unittest.main()
