"""Tiny metadata-only tests; no raster/model packages or source imagery."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Solutions"))
import build_quality_review_queue as queue


PASS = "passed_automatic_screen_pending_visual_review"
REVIEW = "needs_visual_review"
EXCLUDED = "excluded_by_automatic_quality_screen"


def event(identifier="event-1", candidates=None, observed="2018-08-05"):
    return {"event_id": identifier, "group_id": "group-1", "split": "train", "tile": "36UXA",
            "observation_date": observed, "candidate_pair_ids": candidates or []}


def result(identifier, status=PASS, pre="2018-07-31", post="2018-08-05", anchors=None):
    return {"pair_id": identifier, "group_id": "group-1", "split": "train", "tile": "36UXA",
            "anchor_event_ids": anchors or ["event-1"], "pre_date": pre, "post_date": post,
            "quality_status": status, "quality_reasons": [f"reason:{identifier}"] if status != PASS else [],
            "config_id": "frozen-policy", "training_approved": False,
            "batch_path": "fixture-batch", "qa_mask_file": "masks/fixture.tif"}


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(row)+"\n" for row in records))


class QueueTests(unittest.TestCase):
    def build(self, events, records, summary=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation_path, quality_path = root/"annotations.jsonl", root/"quality.jsonl"
            write_jsonl(annotation_path, events); write_jsonl(quality_path, records)
            output = root/"output"; output.mkdir()
            metadata = queue.build_review_queue(annotation_path, quality_path, output, summary)
            rows = [json.loads(line) for line in (output/"review_queue.jsonl").read_text().splitlines()]
            index = [json.loads(line) for line in (output/"quality_result_index.jsonl").read_text().splitlines()]
            return metadata, rows, index

    def test_ranking_preserves_all_candidates_and_uses_quality_then_dates(self):
        records = [result("excluded", EXCLUDED), result("label", "needs_label_review"),
                   result("review", REVIEW), result("late", post="2018-08-10"),
                   result("early-far", pre="2018-07-26"), result("early-close-a"),
                   result("early-close-z"), result("error", "processing_error")]
        metadata, rows, index = self.build([event(candidates=[r["pair_id"] for r in records])], records)
        row = rows[0]
        self.assertEqual(row["candidate_pair_ids"], ["early-close-a", "early-close-z", "early-far", "late",
                                                     "review", "label", "excluded", "error"])
        self.assertEqual(row["proposed_pair_ids"], ["early-close-a", "early-close-z", "early-far"])
        self.assertEqual(row["screened_candidate_count"], 8)
        self.assertEqual(len(row["candidate_quality"]), 8)
        self.assertEqual(len(index), 8)
        self.assertEqual(metadata["input_quality_pair_count"], 8)
        self.assertEqual(row["candidate_quality"][-1]["quality_reasons"], ["reason:error"])
        self.assertIs(row["training_approved"], False)

    def test_order_and_forest_performance_fields_do_not_change_queue(self):
        records = [result("first"), result("later", post="2018-08-10")]
        events = [event(candidates=["first", "later"]), event("outside")]
        _, expected, _ = self.build(events, records)
        changed = copy.deepcopy(records)
        changed[0].update(dice_score=0, forest_model_score=0)
        changed[1].update(dice_score=1, forest_model_score=1)
        metadata, actual, _ = self.build(list(reversed(events)), list(reversed(changed)))
        self.assertEqual(actual, expected)
        self.assertEqual(metadata["catalogue_events_outside_queue"], 1)

    def test_distinct_events_in_same_group_are_not_collapsed(self):
        records = [result("shared", pre="2018-07-31", post="2018-08-10", anchors=["event-1", "event-2"])]
        events = [event(candidates=["shared"]), event("event-2", ["shared"], "2018-08-10")]
        metadata, rows, index = self.build(events, records)
        self.assertEqual([r["event_id"] for r in rows], ["event-1", "event-2"])
        self.assertEqual(len(index), 1)
        self.assertEqual(metadata["event_pair_link_count"], 2)
        self.assertTrue(all(r["group_id"] == "group-1" for r in rows))

    def test_all_excluded_candidates_remain_visible_without_eligible_proposals(self):
        records = [result("bad-a", EXCLUDED), result("bad-b", EXCLUDED)]
        _, rows, index = self.build([event(candidates=["bad-a", "bad-b"])], records)
        self.assertEqual(rows[0]["queue_status"], "requires_alternate_imagery_or_quality_review")
        self.assertEqual(rows[0]["proposed_pair_ids"], [])
        self.assertEqual(rows[0]["eligible_review_candidate_count"], 0)
        self.assertEqual(len(rows[0]["top_candidates"]), 2)
        self.assertTrue(all(not r["eligible_for_review_proposal"] for r in rows[0]["top_candidates"]))
        self.assertEqual({r["quality_reasons"][0] for r in index}, {"reason:bad-a", "reason:bad-b"})

    def test_label_review_and_processing_recovery_are_not_quality_approval(self):
        for status, expected in (("needs_label_review", "requires_label_review"),
                                 ("processing_error", "requires_processing_recovery")):
            with self.subTest(status=status):
                _, rows, _ = self.build([event(candidates=["only"])], [result("only", status)])
                self.assertEqual(rows[0]["queue_status"], expected)
                self.assertEqual(rows[0]["proposed_pair_ids"], [])
                self.assertIs(rows[0]["training_approved"], False)

    def test_group_split_tile_or_date_join_mismatch_is_rejected(self):
        for field, value in (("group_id", "other"), ("split", "test"), ("tile", "36UYA"),
                             ("post_date", "2018-08-04")):
            with self.subTest(field=field):
                record = result("pair"); record[field] = value
                with self.assertRaises(ValueError):
                    self.build([event(candidates=["pair"])], [record])

    def test_pair_anchor_join_is_bidirectional_and_exact(self):
        events = [event(candidates=["shared"]), event("event-2", ["shared"])]
        with self.assertRaisesRegex(ValueError, "anchors disagree"):
            self.build(events, [result("shared", anchors=["event-1"])])

    def test_duplicate_unknown_or_approved_quality_result_is_rejected(self):
        for change in ("duplicate", "unknown", "approval"):
            with self.subTest(change=change):
                records = [result("pair")]
                if change == "duplicate": records.append(result("pair"))
                elif change == "unknown": records[0]["pair_id"] = "missing-pair"
                else: records[0]["training_approved"] = True
                with self.assertRaises(ValueError):
                    self.build([event(candidates=["pair"])], records)

    def test_mixed_policies_and_unknown_quality_status_are_rejected(self):
        records = [result("a"), result("b")]
        records[1]["config_id"] = "different-policy"
        with self.assertRaisesRegex(ValueError, "different frozen policies"):
            self.build([event(candidates=["a", "b"])], records)
        with self.assertRaisesRegex(ValueError, "Unknown quality status"):
            self.build([event(candidates=["a"])], [result("a", "automatically-approved")])

    def test_completed_summary_checks_and_counts_are_enforced(self):
        summary = {"pair_count": 1, "config_id": "frozen-policy", "training_approved": False,
                   "checks": {key: True for key in ("exact_candidate_coverage", "no_duplicate_pairs",
                                                     "single_frozen_quality_policy", "splits_preserved")}}
        metadata, _, _ = self.build([event(candidates=["a"])], [result("a")], summary)
        self.assertEqual(metadata["screened_anchor_event_count"], 1)
        for changed in ({**summary, "pair_count": 2}, {**summary, "checks": {}}):
            with self.assertRaisesRegex(ValueError, "completed summary|publication checks"):
                self.build([event(candidates=["a"])], [result("a")], changed)


if __name__ == "__main__":
    unittest.main()
