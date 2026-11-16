"""Regressions for preserving reviewed partial labels at the model boundary."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "GDAL_NUM_THREADS", "OPJ_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import rasterio
from rasterio.transform import Affine
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Solutions"))
from pilot_dataset import ARTIFACTS, CHANNELS, COUNT_FIELDS, REQUIRED_CHECKS, ReviewedPilotDataset


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PilotFixture:
    """A tiny published export; no source satellite scenes are involved."""

    def __init__(self, root):
        self.root = Path(root)
        self.event = self.root/"examples/event-test"
        self.event.mkdir(parents=True)
        self.grid = {"crs": "EPSG:32636", "width": 64, "height": 64,
                     "transform": [10, 0, 1000, 0, -10, 1640], "bounds": [1000, 1000, 1640, 1640]}
        before = np.full((9, 64, 64), .8, dtype="float32")
        after = np.full((9, 64, 64), .2, dtype="float32")
        before[7], after[7] = -.7, .9
        before[8], after[8] = .4, -.6
        self.features = np.concatenate((before, after, after-before))
        self.labels = np.full((64, 64), 255, dtype="uint8")
        self.labels[5:7, 5:7] = 1
        self.labels[30:38, 30:38] = 0
        self.valid = (self.labels != 255).astype("uint8")
        counts = {"supervised_positive_pixels": 4, "supervised_stable_pixels": 64, "ignored_pixels": 4028}
        policy = "a"*64
        self.row = {"event_id": "event-test", "pair_id": "pair-test", "group_id": "group-test",
                    "split": "train", "tile": "36UXA", "pre_date": "2018-07-31", "post_date": "2018-08-05",
                    "grid": self.grid, "artifact_paths": {key: f"examples/event-test/{name}" for key, name in ARTIFACTS.items()},
                    **counts, "qa_policy_id": policy, "training_approved": True,
                    "visibility_evidence": {"decision_status": "accepted", "training_approved": True}}
        self.example = {key: self.row[key] for key in ("pair_id", "group_id", "split", "tile", "pre_date", "post_date", "grid", "qa_policy_id", *COUNT_FIELDS)}
        self.example.update(anchor_event_ids=["event-test"], channels=list(CHANNELS), training_approved=True,
                            difference_direction="post minus pre, without clipping",
                            decision={"visual_review": "accepted", "temporal_label_review": "accepted", "qa_policy_id": policy})
        self.meta = {"accepted_example_count": 1, "accepted_event_count": 1, "accepted_geographic_group_count": 1,
                     "ready_for_partial_pilot_training": True, "qa_policy_id": policy,
                     "checks": {key: True for key in REQUIRED_CHECKS}, "supervision_pixel_counts": counts}
        self.write_tiff("input_features", self.features)
        self.write_tiff("supervision_mask", self.labels)
        self.write_tiff("valid_mask", self.valid)
        self.publish()

    def write_tiff(self, role, values, transform=None, channels=None):
        data = values if values.ndim == 3 else values[None]
        nodata = float("nan") if role == "input_features" else 255 if role == "supervision_mask" else None
        with rasterio.open(self.event/ARTIFACTS[role], "w", driver="GTiff", crs=self.grid["crs"],
                           transform=transform or Affine(*self.grid["transform"]), width=64, height=64,
                           count=data.shape[0], dtype=data.dtype, nodata=nodata) as output:
            output.write(data)
            output.descriptions = tuple(channels or CHANNELS) if role == "input_features" else (role,)

    def publish(self, rows=None):
        self.example["output_sha256"] = {ARTIFACTS[role]: checksum(self.event/ARTIFACTS[role])
                                         for role in ("input_features", "supervision_mask", "valid_mask")}
        (self.event/"metadata.json").write_text(json.dumps(self.example)+"\n")
        rows = [self.row] if rows is None else rows
        (self.root/"training_manifest.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        names = ["training_manifest.jsonl", *self.row["artifact_paths"].values()]
        self.meta["output_sha256_excluding_this_metadata"] = {name: checksum(self.root/name) for name in names}
        (self.root/"metadata.json").write_text(json.dumps(self.meta)+"\n")


class ReviewedPilotDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = PilotFixture(self.temporary.name)

    def dataset(self):
        return ReviewedPilotDataset(self.fixture.root)

    def test_signed_float_features_and_255_labels_are_preserved_exactly(self):
        sample = self.dataset()[0]
        self.assertEqual(sample["features"].shape, (27, 64, 64))
        self.assertEqual(sample["features"].dtype, torch.float32)
        self.assertEqual(sample["labels"].dtype, torch.uint8)
        self.assertEqual(sample["valid"].dtype, torch.bool)
        np.testing.assert_array_equal(sample["features"].numpy(), self.fixture.features)
        np.testing.assert_array_equal(sample["labels"].numpy(), self.fixture.labels)
        self.assertEqual(int(sample["labels"][0, 0]), 255)
        self.assertFalse(bool(sample["valid"][0, 0]))
        self.assertAlmostEqual(float(sample["features"][25, 5, 5]), 1.6, places=6)
        self.assertAlmostEqual(float(sample["features"][26, 5, 5]), -1., places=6)
        self.assertAlmostEqual(float(sample["features"][18, 5, 5]), -.6, places=6)
        self.assertEqual((sample["event_id"], sample["pair_id"], sample["group_id"]), ("event-test", "pair-test", "group-test"))

    def test_audit_matches_real_pixels_and_does_not_modify_the_artifacts(self):
        paths = list(self.fixture.event.glob("*"))
        before = {p.name: checksum(p) for p in paths}
        data = self.dataset()
        result = data.audit()
        self.assertEqual(result["example_count"], 1)
        self.assertEqual({key: result[key] for key in COUNT_FIELDS}, self.fixture.meta["supervision_pixel_counts"])
        self.assertEqual(result["nonfinite_feature_values_replaced"], 0)
        self.assertEqual(result["dataset_metadata_sha256"], data.metadata_sha256)
        self.assertEqual(result["manifest_sha256"], checksum(self.fixture.root/"training_manifest.jsonl"))
        self.assertEqual({p.name: checksum(p) for p in paths}, before)

    def test_changed_manifest_and_changed_tiff_cannot_reuse_published_hashes(self):
        manifest = self.fixture.root/"training_manifest.jsonl"
        manifest.write_text(manifest.read_text()+" ")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.dataset()
        self.fixture.publish()
        with (self.fixture.event/"input_features.tif").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.dataset()

    def test_changed_artifact_after_construction_is_detected_on_access(self):
        data = self.dataset()
        with (self.fixture.event/"valid_mask.tif").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            data[0]

    def test_nonfinite_ignored_inputs_are_zeroed_only_in_memory_and_reported(self):
        self.fixture.features[0, 0, 0] = np.nan
        self.fixture.features[18, 0, 0] = np.nan
        self.fixture.write_tiff("input_features", self.fixture.features)
        self.fixture.publish()
        data = self.dataset()
        sample = data[0]
        self.assertEqual(sample["nonfinite_replacements"], 2)
        self.assertTrue(torch.isfinite(sample["features"]).all())
        self.assertEqual(float(sample["features"][0, 0, 0]), 0.)
        self.assertEqual(float(sample["features"][18, 0, 0]), 0.)
        self.assertEqual(float(sample["features"][7, 0, 0]), float(self.fixture.features[7, 0, 0]))
        self.assertEqual(int(sample["labels"][0, 0]), 255)
        self.assertEqual(data.audit()["nonfinite_feature_values_replaced"], 2)
        with rasterio.open(self.fixture.event/"input_features.tif") as source:
            self.assertTrue(np.isnan(source.read(1)[0, 0]))

    def test_nonfinite_supervised_inputs_are_rejected_instead_of_zeroed(self):
        self.fixture.features[0, 5, 5] = np.inf
        self.fixture.features[18, 5, 5] = -np.inf
        self.fixture.write_tiff("input_features", self.fixture.features)
        self.fixture.publish()
        with self.assertRaisesRegex(ValueError, "Supervised pixels contain non-finite"):
            self.dataset()[0]

    def test_wrong_valid_mask_and_unknown_label_values_are_rejected(self):
        self.fixture.valid[0, 0] = 1
        self.fixture.write_tiff("valid_mask", self.fixture.valid)
        self.fixture.publish()
        with self.assertRaisesRegex(ValueError, "Valid mask must equal"):
            self.dataset()[0]
        self.fixture.valid[0, 0] = 0
        self.fixture.write_tiff("valid_mask", self.fixture.valid)
        self.fixture.labels[0, 0] = 2
        self.fixture.write_tiff("supervision_mask", self.fixture.labels)
        self.fixture.publish()
        with self.assertRaisesRegex(ValueError, "labels must remain"):
            self.dataset()[0]

    def test_transposed_channels_and_changed_grid_are_rejected_even_with_new_hashes(self):
        channels = list(CHANNELS)
        channels[0], channels[1] = channels[1], channels[0]
        self.fixture.write_tiff("input_features", self.fixture.features, channels=channels)
        self.fixture.publish()
        with self.assertRaisesRegex(ValueError, "channel order changed"):
            self.dataset()[0]
        self.fixture.write_tiff("input_features", self.fixture.features, transform=Affine(10, 0, 1010, 0, -10, 1640))
        self.fixture.publish()
        with self.assertRaisesRegex(ValueError, "geographic grid"):
            self.dataset()[0]

    def test_finite_difference_corruption_is_rejected_without_recomputing_inputs(self):
        self.fixture.features[18, 0, 0] = .5
        self.fixture.write_tiff("input_features", self.fixture.features)
        self.fixture.publish()
        with self.assertRaisesRegex(ValueError, "not post minus pre"):
            self.dataset()[0]

    def test_path_traversal_and_symlinks_outside_the_dataset_are_rejected(self):
        self.fixture.row["artifact_paths"]["input_features"] = "../outside.tif"
        # The unsafe path need not exist: the exact event-directory contract
        # rejects this signed manifest before any array can be read.
        (self.fixture.root/"training_manifest.jsonl").write_text(json.dumps(self.fixture.row)+"\n")
        self.fixture.meta["output_sha256_excluding_this_metadata"]["training_manifest.jsonl"] = checksum(self.fixture.root/"training_manifest.jsonl")
        (self.fixture.root/"metadata.json").write_text(json.dumps(self.fixture.meta)+"\n")
        with self.assertRaisesRegex(ValueError, "exact event export"):
            self.dataset()
        self.fixture.row["artifact_paths"]["input_features"] = "examples/event-test/input_features.tif"
        self.fixture.publish()
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside)/"input_features.tif"
            target.write_bytes((self.fixture.event/"input_features.tif").read_bytes())
            (self.fixture.event/"input_features.tif").unlink()
            (self.fixture.event/"input_features.tif").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "escapes the dataset"):
                self.dataset()

    def test_held_out_unapproved_or_policy_mismatched_records_are_rejected(self):
        for field, value, message in (("split", "validation", "approved training"),
                                      ("training_approved", False, "approved training"),
                                      ("qa_policy_id", "b"*64, "different QA policies")):
            original = self.fixture.row[field]
            self.fixture.row[field] = value
            self.fixture.publish()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.dataset()
            self.fixture.row[field] = original

    def test_duplicate_geographic_groups_and_missing_accepted_evidence_are_rejected(self):
        second = {**self.fixture.row, "event_id": "event-second", "pair_id": "pair-second"}
        self.fixture.meta.update(accepted_example_count=2, accepted_event_count=2, accepted_geographic_group_count=2)
        self.fixture.publish(rows=[self.fixture.row, second])
        with self.assertRaisesRegex(ValueError, "Repeated group_id"):
            self.dataset()
        self.fixture.meta.update(accepted_example_count=1, accepted_event_count=1, accepted_geographic_group_count=1)
        self.fixture.row.pop("visibility_evidence")
        self.fixture.publish()
        with self.assertRaisesRegex(ValueError, "accepted visual/temporal evidence"):
            self.dataset()


if __name__ == "__main__":
    unittest.main()
