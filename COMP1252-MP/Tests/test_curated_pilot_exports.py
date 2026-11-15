"""Small regressions for reviewed partial supervision and signed pair exports."""

from contextlib import ExitStack, contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import warnings


for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "GDAL_NUM_THREADS", "OPJ_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import rasterio


SOLUTIONS = Path(__file__).resolve().parents[1]/"Solutions"
sys.path.insert(0, str(SOLUTIONS))
spec = importlib.util.spec_from_file_location("curated_pilot_exports", SOLUTIONS/"curated_pilot_exports.py")
exports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exports)


class SignedPairTests(unittest.TestCase):
    def test_channel_order_and_signed_differences_are_preserved_without_clipping(self):
        pre = np.full((9, 2, 3), .8, dtype="float64")
        post = np.full((9, 2, 3), .2, dtype="float32")
        pre[7], post[7] = -.9, .9
        pre[8], post[8] = .7, -.8
        actual = exports.stack_pair(pre, post)
        self.assertEqual(actual.dtype, np.dtype("float32"))
        self.assertEqual(actual.shape, (27, 2, 3))
        np.testing.assert_array_equal(actual[:9], pre.astype("float32"))
        np.testing.assert_array_equal(actual[9:18], post)
        np.testing.assert_allclose(actual[18:], post-pre.astype("float32"), atol=0)
        self.assertAlmostEqual(float(actual[25, 0, 0]), 1.8, places=6)
        self.assertAlmostEqual(float(actual[26, 0, 0]), -1.5, places=6)
        self.assertAlmostEqual(float(actual[18, 0, 0]), -.6, places=6)

    def test_shape_raw_integer_infinity_and_float32_overflow_are_rejected(self):
        valid = np.zeros((9, 1, 1), dtype="float32")
        for wrong in (np.zeros((8, 1, 1), dtype="float32"),
                      np.zeros((9, 2, 1), dtype="float32"),
                      valid.astype("uint16"), valid+np.inf):
            with self.subTest(shape=wrong.shape, dtype=wrong.dtype), self.assertRaises(ValueError):
                exports.stack_pair(valid, wrong)
        huge = np.full((9, 1, 1), np.finfo("float32").max, dtype="float32")
        with self.assertRaisesRegex(ValueError, "differences exceed"):
            exports.stack_pair(-huge, huge)

    def test_missing_features_remain_missing_instead_of_becoming_zero(self):
        pre, post = np.ones((9, 2, 2), dtype="float32"), np.ones((9, 2, 2), dtype="float32")
        pre[:, 0, 1] = np.nan
        actual = exports.stack_pair(pre, post)
        self.assertTrue(np.isnan(actual[:9, 0, 1]).all())
        self.assertTrue(np.isnan(actual[18:, 0, 1]).all())


class PartialSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.positive = np.zeros((4, 4), dtype=bool)
        self.stable = np.zeros((4, 4), dtype=bool)
        self.qa = np.zeros((4, 4), dtype=bool)
        self.positive[0, 0] = True
        self.stable[3, 3] = True

    def test_only_reviewed_disjoint_regions_receive_zero_or_one(self):
        self.qa[1, 1] = True
        labels = exports.build_supervision(self.positive, self.stable, self.qa)
        self.assertEqual(labels.dtype, np.dtype("uint8"))
        self.assertEqual(labels[0, 0], 1)
        self.assertEqual(labels[3, 3], 0)
        self.assertEqual(int((labels == 255).sum()), 14)
        self.assertEqual(labels[1, 1], 255)

    def test_overlapping_or_qa_obscured_review_rois_are_rejected(self):
        self.stable[0, 0] = True
        with self.assertRaisesRegex(ValueError, "overlap"):
            exports.build_supervision(self.positive, self.stable, self.qa)
        self.stable[0, 0] = False
        self.qa[0, 0] = True
        with self.assertRaisesRegex(ValueError, "unusable QA"):
            exports.build_supervision(self.positive, self.stable, self.qa)

    def test_integer_masks_and_shape_mismatch_cannot_be_silently_broadcast(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            exports.build_supervision(self.positive.astype("uint8"), self.stable, self.qa)
        with self.assertRaisesRegex(ValueError, "same shape"):
            exports.build_supervision(self.positive, self.stable[:1], self.qa)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.pair = {
            "pair_id": "pair-test", "group_id": "group-train", "split": "train", "tile": "36UXA",
            "anchor_event_ids": ["event-test"], "pre_date": "2018-07-31", "post_date": "2018-08-05",
            "pre_path": "pre.SAFE", "post_path": "post.SAFE", "pre_scene_id": "pre", "post_scene_id": "post",
            "grid": {"crs": "EPSG:32636", "transform": [10, 0, 0, 0, -10, 640],
                     "width": 64, "height": 64, "bounds": [0, 0, 640, 640]},
        }
        self.positive, self.stable, self.qa = [np.zeros((64, 64), dtype=bool) for _ in range(3)]
        self.positive[5, 5] = True
        self.stable[30:34, 30:34] = True
        self.pre = np.full((9, 64, 64), .8, dtype="float32")
        self.post = np.full((9, 64, 64), .2, dtype="float32")
        self.pre[7], self.post[7] = .9, -.5
        self.native_valid = np.ones((64, 64), dtype=bool)
        self.decision = {"visual_review": "accepted", "temporal_label_review": "accepted", "qa_policy_id": "policy-test"}

    @contextmanager
    def environment(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            summary = root/"Datasets/PixelQA/v1/summary/metadata.json"
            summary.parent.mkdir(parents=True)
            summary.write_text('{"frozen":true}\n')
            stack.enter_context(mock.patch.object(exports, "ROOT", root))
            stack.enter_context(mock.patch.object(exports, "_frozen_summary", return_value={"config_id": "policy-test"}))
            stack.enter_context(mock.patch.object(exports, "_pair_source_statistics", return_value={"source": [1, 2]}))
            stack.enter_context(mock.patch.object(exports, "_published_qa", return_value=(self.qa.copy(), self.positive.copy())))
            stack.enter_context(mock.patch.object(exports, "_catalogue_loss_mask", return_value=self.positive.copy()))
            stack.enter_context(mock.patch.object(exports, "read_pair_features", return_value=(self.pre, self.post, self.native_valid)))
            yield root/"example"

    def export(self, output):
        return exports.export_example(output, self.pair, self.positive, self.stable, self.qa, self.decision)

    def test_round_trip_signed_27_channels_partial_labels_and_checksums(self):
        with self.environment() as output:
            metadata = self.export(output)
            with rasterio.open(output/"input_features.tif") as source:
                self.assertEqual(source.count, 27)
                self.assertEqual(source.dtypes, ("float32",)*27)
                self.assertEqual(source.descriptions, exports.CHANNELS)
                self.assertEqual(source.res, (10, 10))
                self.assertEqual(source.bounds, (0, 0, 640, 640))
                self.assertAlmostEqual(float(source.read(26)[5, 5]), -1.4, places=6)
            with rasterio.open(output/"supervision_mask.tif") as source:
                labels = source.read(1)
                self.assertEqual(source.nodata, 255)
                self.assertEqual(int((labels == 1).sum()), 1)
                self.assertEqual(int((labels == 0).sum()), 16)
                self.assertEqual(int((labels == 255).sum()), 4079)
            with rasterio.open(output/"valid_mask.tif") as source:
                np.testing.assert_array_equal(source.read(1), labels != 255)
            for name, checksum in metadata["output_sha256"].items():
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(), checksum)
            self.assertTrue(metadata["training_approved"])
            self.assertTrue(all(metadata["checks"].values()))
            self.assertEqual(json.loads((output/"metadata.json").read_text())["decision"], self.decision)

    def test_held_out_split_and_unaccepted_review_cannot_export(self):
        with self.environment() as output:
            self.pair["split"] = "test"
            with self.assertRaisesRegex(ValueError, "training-split"):
                self.export(output)
            self.pair["split"] = "train"
            self.decision["visual_review"] = "pending"
            with self.assertRaisesRegex(ValueError, "Explicit accepted"):
                self.export(output)
            self.assertFalse(output.exists())

    def test_missing_stable_roi_and_inconsistent_grid_are_rejected(self):
        with self.environment() as output:
            self.stable[:] = False
            with self.assertRaisesRegex(ValueError, "at least one positive and stable"):
                self.export(output)
            self.stable[30, 30] = True
            self.pair["grid"]["bounds"][0] = 10
            with self.assertRaisesRegex(ValueError, "bounds and transform"):
                self.export(output)

    def test_qa_override_and_positive_outside_reference_are_rejected(self):
        with self.environment() as output:
            self.qa[10, 10] = True
            with self.assertRaisesRegex(ValueError, "Caller QA differs"):
                self.export(output)
            self.qa[10, 10] = False
            self.positive[6, 6] = True
            with self.assertRaisesRegex(ValueError, "beyond the frozen temporal"):
                self.export(output)

    def test_native_invalid_supervised_pixels_are_rejected(self):
        with self.environment() as output:
            self.native_valid[5, 5] = False
            with self.assertRaisesRegex(ValueError, "invalid source features"):
                self.export(output)
            self.assertFalse(output.exists())

    def test_known_earlier_or_later_loss_cannot_become_stable_supervision(self):
        with self.environment() as output, mock.patch.object(exports, "_catalogue_loss_mask", return_value=self.stable.copy()):
            with self.assertRaisesRegex(ValueError, "known loss"):
                self.export(output)

    def test_existing_output_is_not_overwritten(self):
        with self.environment() as output:
            output.mkdir()
            (output/"keep.txt").write_text("preserve")
            with self.assertRaises(FileExistsError):
                self.export(output)
            self.assertEqual((output/"keep.txt").read_text(), "preserve")

    def test_relevant_projected_invalid_geometry_is_repaired_for_control_exclusion(self):
        from shapely import make_valid
        from shapely.geometry import Polygon

        nearby = Polygon([(300, 300), (340, 340), (300, 340), (340, 300), (300, 300)])
        remote = Polygon([(10000, 10000), (10040, 10040), (10000, 10040), (10040, 10000), (10000, 10000)])
        self.assertFalse(nearby.is_valid)
        with mock.patch.object(exports, "_digest", return_value="catalogue-hash"), \
                mock.patch.object(exports, "_frozen_summary", return_value={"annotation_catalogue_sha256": "catalogue-hash"}), \
                mock.patch.object(exports, "_projected_catalogue", return_value=(nearby, remote)), \
                mock.patch("shapely.make_valid", wraps=make_valid) as repair, \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            excluded = exports._catalogue_loss_mask(self.pair)
        self.assertTrue(excluded[30:34, 30:34].any())
        self.assertFalse(excluded[:10, :10].any())
        self.assertEqual(repair.call_count, 1)
        self.assertFalse(any(issubclass(item.category, RuntimeWarning) for item in caught))


class BoundedFeatureReadTests(unittest.TestCase):
    def metadata(self):
        return [(Path("unused"), {"feature_band_paths": {name: name for name in exports.BANDS},
                                  "quantification_value": 10000}) for _ in range(2)]

    def fused(self, path, grid, include_feature):
        values = {"B08": 2000, "B04": 5000, "B8A": 1000, "B11": 3000}
        data = np.full((64, 64), values.get(path.name, 2000), dtype="float32")
        valid = np.ones((64, 64), dtype=bool)
        if path.name == "B12":
            valid[0, 0] = False
        return None, None, data, valid

    def test_reads_only_seven_bounded_bands_per_date_and_preserves_invalidity(self):
        with mock.patch.object(exports, "_pair_metadata", return_value=self.metadata()), \
                mock.patch.object(exports, "_pair_source_statistics", return_value={"source": [1, 2]}), \
                mock.patch("assess_pixel_quality.read_fused_band", side_effect=self.fused) as reader:
            pre, post, valid = exports.read_pair_features({"grid": {"frozen": True}})
        self.assertEqual(reader.call_count, 14)
        self.assertTrue(all(call.args[2] is True for call in reader.call_args_list))
        self.assertEqual(pre.dtype, np.dtype("float32"))
        self.assertEqual(pre.shape, (9, 64, 64))
        self.assertFalse(valid[0, 0])
        self.assertTrue(np.isnan(pre[:, 0, 0]).all())
        self.assertTrue(np.isnan(post[:, 0, 0]).all())
        self.assertAlmostEqual(float(pre[7, 1, 1]), -3/7, places=6)
        self.assertAlmostEqual(float(pre[8, 1, 1]), -.5, places=6)
        self.assertAlmostEqual(float(pre[3, 1, 1]), .2, places=6)

    def test_source_change_during_bounded_reads_is_rejected(self):
        with mock.patch.object(exports, "_pair_metadata", return_value=self.metadata()), \
                mock.patch.object(exports, "_pair_source_statistics", side_effect=[{"source": [1, 2]}, {"source": [1, 3]}]), \
                mock.patch("assess_pixel_quality.read_fused_band", side_effect=self.fused):
            with self.assertRaisesRegex(ValueError, "changed during feature"):
                exports.read_pair_features({"grid": {"frozen": True}})


if __name__ == "__main__":
    unittest.main()
