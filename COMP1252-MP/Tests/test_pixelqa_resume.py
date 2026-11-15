"""Metadata-only resume/publication regressions; no models or raster imports.

Every file below is a tiny temporary fixture. Source-stat checks use an ordinary
text file, and worker startup is mocked so no child processing can begin.
"""

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Solutions"))
import assess_pixel_quality as quality


IDENTITY = {
    "s2cloudless_version": "fixture", "lightgbm_version": "fixture",
    "model_sha256": "model-v1", "model_path": "/never-open/model.txt",
    "runtime_versions": {"rasterio": "fixture"},
    "algorithm_sources_sha256": {"quality.py": "code-v1"},
}


def pair(number):
    return {"pair_id": f"pair-{number}", "group_id": f"group-{number}",
            "split": "train" if number == 0 else "validation", "tile": "36UXA",
            "pre_date": "2018-07-31", "post_date": "2018-08-05",
            "pre_scene_id": f"pre-{number}", "post_scene_id": f"post-{number}",
            "anchor_event_ids": [f"event-{number}"],
            "grid": {"bounds": [number, 0, number+1, 1]},
            "status": "candidate_for_pixel_review"}


def write_json(path, value):
    path.write_text(json.dumps(value)+"\n")


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value)+"\n" for value in values))


@contextmanager
def completed_fixture():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audit = root/"audit"; audit.mkdir()
        output_base = root/"quality"; batch = output_base/"batches/00000"
        batch.mkdir(parents=True)
        records = [pair(0), pair(1)]
        write_json(audit/"metadata.json", {"frozen_audit": "v1"})
        write_json(audit/"annotations_clean.geojson", {"type": "FeatureCollection", "features": []})
        write_jsonl(audit/"candidate_pairs.jsonl", records)
        source = root/"source-stat-fixture.txt"; source.write_text("small metadata fixture")
        stat = source.stat()
        policy = quality.policy_id(IDENTITY)
        metadata = {
            "mode": "batch", "batch_index": 0, "batch_size": 2, "seed": 42,
            "pair_count": 2, "config_id": policy,
            "audit_metadata_sha256": quality.digest(audit/"metadata.json"),
            "candidate_manifest_sha256": quality.digest(audit/"candidate_pairs.jsonl"),
            "annotation_catalogue_sha256": quality.digest(audit/"annotations_clean.geojson"),
            "selection_hash": quality.selection_hash(records),
            "source_file_stats": {str(source): [stat.st_size, stat.st_mtime_ns]},
            "resources": {"worker_high_water_rss_mib": 1, "elapsed_seconds": .01},
        }
        rows = [{**{name: record[name] for name in ("pair_id", "group_id", "split", "tile",
                                                  "pre_date", "post_date", "anchor_event_ids")},
                 "training_approved": False, "config_id": policy,
                 "quality_status": "passed_automatic_screen_pending_visual_review",
                 "quality_reasons": []} for record in records]
        write_json(batch/"metadata.json", metadata)
        write_jsonl(batch/"results.jsonl", rows)
        args = SimpleNamespace(audit=audit, output_base=output_base,
                               cache_root=root/"cache", batch_size=2, batch_index=0,
                               seed=42, max_seconds=300, max_rss_mib=512,
                               min_available_gib=1.5, mode="batch")
        yield args, batch, metadata, rows, records, source


class ResumeMetadataTests(unittest.TestCase):
    def run_existing(self, args):
        # Importing validate_sample only loads its standard-library definitions.
        with mock.patch("validate_sample.run_guarded") as guard:
            quality.run_one(args, "batch", 0)
        guard.assert_not_called()

    def test_matching_completed_batch_resumes_without_starting_a_worker(self):
        with completed_fixture() as (args, *_), \
                mock.patch.object(quality, "model_identity", return_value=copy.deepcopy(IDENTITY)):
            self.run_existing(args)

    def test_changed_policy_threshold_refuses_stale_batch(self):
        with completed_fixture() as (args, *_), \
                mock.patch.object(quality, "model_identity", return_value=copy.deepcopy(IDENTITY)), \
                mock.patch.dict(quality.CONFIG, {"cloud_probability_threshold": .9}):
            with self.assertRaisesRegex(ValueError, "different audit, manifest or policy"):
                self.run_existing(args)

    def test_model_runtime_and_implementation_changes_refuse_stale_batch(self):
        changes = [
            {"model_sha256": "model-v2"},
            {"runtime_versions": {"rasterio": "different-version"}},
            {"algorithm_sources_sha256": {"quality.py": "fixed-code"}},
        ]
        for change in changes:
            with self.subTest(change=change), completed_fixture() as (args, *_), \
                    mock.patch.object(quality, "model_identity", return_value={**IDENTITY, **change}):
                with self.assertRaisesRegex(ValueError, "different audit, manifest or policy"):
                    self.run_existing(args)

    def test_changed_audit_catalogue_or_candidate_file_refuses_stale_batch(self):
        for name in ("metadata.json", "annotations_clean.geojson", "candidate_pairs.jsonl"):
            with self.subTest(file=name), completed_fixture() as (args, *_), \
                    mock.patch.object(quality, "model_identity", return_value=copy.deepcopy(IDENTITY)):
                path = args.audit/name
                path.write_text(path.read_text()+" ")
                with self.assertRaisesRegex(ValueError, "different audit, manifest or policy"):
                    self.run_existing(args)

    def test_changed_seed_or_batch_size_refuses_stale_selection(self):
        for name, value in (("seed", 43), ("batch_size", 1)):
            with self.subTest(argument=name), completed_fixture() as (args, *_), \
                    mock.patch.object(quality, "model_identity", return_value=copy.deepcopy(IDENTITY)):
                setattr(args, name, value)
                with self.assertRaisesRegex(ValueError, "different frozen selection"):
                    self.run_existing(args)

    def test_stale_selection_hash_refuses_resume(self):
        with completed_fixture() as (args, batch, metadata, *_), \
                mock.patch.object(quality, "model_identity", return_value=copy.deepcopy(IDENTITY)):
            metadata["selection_hash"] = "wrong-selection"
            write_json(batch/"metadata.json", metadata)
            with self.assertRaisesRegex(ValueError, "different frozen selection"):
                self.run_existing(args)

    def test_changed_source_size_or_timestamp_refuses_resume(self):
        for change in ("size", "mtime"):
            with self.subTest(change=change), completed_fixture() as (args, _, _, _, _, source), \
                    mock.patch.object(quality, "model_identity", return_value=copy.deepcopy(IDENTITY)):
                if change == "size":
                    source.write_text("a different fixture size")
                else:
                    stat = source.stat()
                    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns+10_000_000))
                with self.assertRaisesRegex(ValueError, "source file changed"):
                    self.run_existing(args)


class SummaryIdentityTests(unittest.TestCase):
    def summarize(self, args, directory):
        output = Path(directory)/"summary-fixture"; output.mkdir()
        quality.summarize(args, output)
        return json.loads((output/"metadata.json").read_text())

    def test_matching_summary_preserves_unapproved_status_and_exact_coverage(self):
        with completed_fixture() as (args, *_), tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(quality, "model_identity", return_value=copy.deepcopy(IDENTITY)):
            result = self.summarize(args, directory)
        self.assertEqual(result["pair_count"], 2)
        self.assertTrue(result["checks"]["exact_candidate_coverage"])
        self.assertTrue(result["checks"]["splits_preserved"])
        self.assertIs(result["training_approved"], False)

    def test_summary_rejects_frozen_identity_changes(self):
        altered = {"split": "test", "group_id": "different-group", "tile": "36UYA",
                   "pre_date": "2018-07-30", "post_date": "2018-08-06",
                   "anchor_event_ids": ["different-event"]}
        for field, value in altered.items():
            with self.subTest(field=field), completed_fixture() as (args, batch, _, rows, *_), \
                    tempfile.TemporaryDirectory() as directory, \
                    mock.patch.object(quality, "model_identity", return_value=copy.deepcopy(IDENTITY)):
                rows[0][field] = value
                write_jsonl(batch/"results.jsonl", rows)
                with self.assertRaisesRegex(ValueError, "identities disagree"):
                    self.summarize(args, directory)

    def test_summary_rejects_unexpected_training_approval(self):
        with completed_fixture() as (args, batch, _, rows, *_), \
                tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(quality, "model_identity", return_value=copy.deepcopy(IDENTITY)):
            rows[0]["training_approved"] = True
            write_jsonl(batch/"results.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "unexpected approval or policy"):
                self.summarize(args, directory)

    def test_summary_rejects_duplicate_pairs(self):
        with completed_fixture() as (args, batch, _, rows, *_), \
                tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(quality, "model_identity", return_value=copy.deepcopy(IDENTITY)):
            write_jsonl(batch/"results.jsonl", rows+[rows[0]])
            with self.assertRaisesRegex(ValueError, "Duplicate pair"):
                self.summarize(args, directory)

    def test_summary_rejects_incomplete_candidate_coverage(self):
        with completed_fixture() as (args, batch, _, rows, *_), \
                tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(quality, "model_identity", return_value=copy.deepcopy(IDENTITY)):
            write_jsonl(batch/"results.jsonl", rows[:1])
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                self.summarize(args, directory)


if __name__ == "__main__":
    unittest.main()
