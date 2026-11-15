"""Small pure-helper regressions for frozen pixel-quality screening.

These fixtures never open satellite imagery, instantiate a classifier, or run
model predictions. Native libraries are restricted to one thread.
"""

from collections import Counter
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "GDAL_NUM_THREADS",
    "OPJ_NUM_THREADS",
):
    os.environ[name] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Solutions"))
import numpy as np
import rasterio
from affine import Affine
from rasterio.transform import from_origin
from shapely.geometry import box

import assess_pixel_quality as quality


def zero_observation():
    record = {name: np.zeros((64, 64), dtype=bool) for name in quality.MASK_NAMES}
    record.update(
        ndvi=np.full((64, 64), .5, dtype=np.float32),
        product_cloud=np.zeros((64, 64), dtype=bool),
        metadata={"context_incomplete": False},
    )
    return record


def core():
    return {"crs": "EPSG:32636", "width": 64, "height": 64,
            "transform": [10., 0., 1000., 0., -10., 2640.],
            "bounds": [1000., 2000., 1640., 2640.]}


class PilotSelectionTests(unittest.TestCase):
    def records(self):
        result = []
        for split in ("train", "validation", "test"):
            for tile in ("36UXA", "36UYA"):
                for group in range(8):
                    for interval in (5, 20):
                        result.append({
                            "pair_id": f"{split}-{tile}-{group}-{interval}",
                            "group_id": f"{split}-{tile}-group-{group}",
                            "split": split, "tile": tile, "interval_days": interval,
                            "pre_scene_id": f"pre-{interval}", "post_scene_id": f"post-{interval}",
                            "temporal_source_ids": ["36UXA_time-dependent" if tile == "36UXA" else
                                                    "36UYA_Spring_time-dependent" if group % 2 == 0 else
                                                    "36UYA_Summer_time-dependent"],
                            "grid": {"bounds": [group, 0, group+1, 1]},
                        })
        return result

    def test_pilot_is_deterministic_balanced_and_uses_distinct_groups(self):
        records = self.records()
        first = quality.choose_pilot(records, seed=42)
        second = quality.choose_pilot(list(reversed(records)), seed=42)
        self.assertEqual([p["pair_id"] for p in first], [p["pair_id"] for p in second])
        self.assertEqual(len(first), 24)
        self.assertEqual(len({p["group_id"] for p in first}), 24)
        self.assertEqual(Counter((p["split"], p["tile"]) for p in first),
                         Counter({(s, t): 4 for s in ("train", "validation", "test")
                                  for t in ("36UXA", "36UYA")}))
        self.assertEqual(Counter(p["interval_days"] <= 10 for p in first), {True: 12, False: 12})

    def test_quality_outcomes_do_not_change_selection(self):
        records = self.records()
        expected = [p["pair_id"] for p in quality.choose_pilot(records)]
        altered = copy.deepcopy(records)
        for index, record in enumerate(altered):
            record["cloud_fraction"] = index % 2
            record["quality_status"] = "clear" if index % 2 else "cloudy"
            record["forest_model_score"] = 1-index/len(records)
        self.assertEqual(expected, [p["pair_id"] for p in quality.choose_pilot(altered)])

    def test_uya_pilot_covers_both_temporal_seasons_in_each_split(self):
        selected = quality.choose_pilot(self.records(), seed=42)
        for split in ("train", "validation", "test"):
            counts = Counter(source for p in selected if p["tile"] == "36UYA" and p["split"] == split
                             for source in p["temporal_source_ids"])
            self.assertEqual(counts, {"36UYA_Spring_time-dependent": 2,
                                      "36UYA_Summer_time-dependent": 2})

    def test_insufficient_geographic_groups_are_explicit(self):
        records = [p for p in self.records() if p["group_id"].endswith("group-0")]
        with self.assertRaisesRegex(ValueError, "Insufficient distinct geographic groups"):
            quality.choose_pilot(records)


class ContextAlignmentTests(unittest.TestCase):
    def test_padded_20m_context_maps_exactly_to_64_pixel_10m_core(self):
        raw_grid = core()
        context = quality.context_grid(raw_grid)
        self.assertEqual((context["height"], context["width"]), (152, 152))
        self.assertEqual(context["transform"] * (60, 60),
                         Affine(*raw_grid["transform"]) * (0, 0))
        self.assertEqual(context["transform"] * (92, 92),
                         Affine(*raw_grid["transform"]) * (64, 64))
        values = np.arange(152*152).reshape(152, 152)
        expanded = quality.crop_context(values)
        self.assertEqual(expanded.shape, (64, 64))
        self.assertTrue((expanded[:2, :2] == values[60, 60]).all())
        self.assertTrue((expanded[-2:, -2:] == values[91, 91]).all())
        np.testing.assert_array_equal(expanded[::2, ::2], values[60:92, 60:92])


class FusedRasterReadTests(unittest.TestCase):
    def grid(self):
        return {"crs": "EPSG:32636", "width": 64, "height": 64,
                "transform": [10., 0., 1680., 0., -10., 2320.],
                "bounds": [1680., 1680., 2320., 2320.]}

    def write_raster(self, path, data, resolution):
        with rasterio.open(path, "w", driver="GTiff", crs="EPSG:32636",
                           transform=from_origin(0, 4000, resolution, resolution),
                           width=data.shape[1], height=data.shape[0], count=1,
                           dtype=data.dtype, nodata=0) as destination:
            destination.write(data, 1)

    def test_fused_10m_and_20m_core_matches_existing_reader_on_physical_plane(self):
        from validate_sample import read_band_on_grid

        fine_x, fine_y = np.meshgrid(np.arange(5, 4000, 10), np.arange(3995, 0, -10))
        coarse_x, coarse_y = np.meshgrid(np.arange(10, 4000, 20), np.arange(3990, 0, -20))
        fine = (1000+fine_x+2*fine_y).astype(np.uint16)
        coarse = (1000+coarse_x+2*coarse_y).astype(np.uint16)
        grid = self.grid()
        with tempfile.TemporaryDirectory() as directory, \
                rasterio.Env(GDAL_CACHEMAX=8*2**20, GDAL_NUM_THREADS="1"):
            fine_path, coarse_path = Path(directory)/"fine.tif", Path(directory)/"coarse.tif"
            self.write_raster(fine_path, fine, 10)
            self.write_raster(coarse_path, coarse, 20)
            readings = []
            for path in (fine_path, coarse_path):
                coarse_values, coarse_valid, core_values, core_valid = quality.read_fused_band(path, grid, True)
                existing_values, existing_valid = read_band_on_grid(path, quality.core_grid(grid))
                self.assertEqual(coarse_values.shape, (152, 152))
                self.assertTrue(coarse_valid.all())
                np.testing.assert_allclose(core_values, existing_values, rtol=0, atol=1e-4)
                np.testing.assert_array_equal(core_valid, existing_valid)
                readings.append((coarse_values, core_values))
        np.testing.assert_allclose(readings[0][0], readings[1][0], rtol=0, atol=1e-4)
        np.testing.assert_allclose(readings[0][1], fine[168:232, 168:232], rtol=0, atol=1e-4)

    def test_partial_invalid_10m_block_keeps_20m_cloud_input_unknown(self):
        values = np.full((400, 400), 1000, dtype=np.uint16)
        values[168, 168], values[168, 169] = 0, 65535
        with tempfile.TemporaryDirectory() as directory, \
                rasterio.Env(GDAL_CACHEMAX=8*2**20, GDAL_NUM_THREADS="1"):
            path = Path(directory)/"fine-invalid.tif"
            self.write_raster(path, values, 10)
            _, coarse_valid, _, core_valid = quality.read_fused_band(path, self.grid(), True)
        self.assertFalse(coarse_valid[60, 60])
        self.assertTrue(coarse_valid[60, 61])
        self.assertFalse(core_valid[0, 0])
        self.assertFalse(core_valid[0, 1])
        self.assertTrue(core_valid[0, 2])

    def test_native_20m_saturation_is_not_hidden_by_bilinear_upsampling(self):
        values = np.full((200, 200), 1000, dtype=np.uint16)
        values[100, 100] = 65535
        with tempfile.TemporaryDirectory() as directory, \
                rasterio.Env(GDAL_CACHEMAX=8*2**20, GDAL_NUM_THREADS="1"):
            path = Path(directory)/"coarse-saturated.tif"
            self.write_raster(path, values, 20)
            for include_feature in (False, True):
                with self.subTest(include_feature=include_feature):
                    _, coarse_valid, _, _ = quality.read_fused_band(path, self.grid(), include_feature)
                    self.assertFalse(coarse_valid[76, 76])

    def test_native_20m_nodata_block_remains_unknown(self):
        values = np.full((200, 200), 1000, dtype=np.uint16)
        values[100, 100] = 0
        with tempfile.TemporaryDirectory() as directory, \
                rasterio.Env(GDAL_CACHEMAX=8*2**20, GDAL_NUM_THREADS="1"):
            path = Path(directory)/"coarse-nodata.tif"
            self.write_raster(path, values, 20)
            _, coarse_valid, _, _ = quality.read_fused_band(path, self.grid(), False)
        self.assertFalse(coarse_valid[76, 76])


class QualityClassificationTests(unittest.TestCase):
    def test_target_and_crop_fractions_use_different_explicit_denominators(self):
        pre, post = zero_observation(), zero_observation()
        target = np.zeros((64, 64), dtype=np.uint8)
        target[10, 10:12] = 1
        pre["cloud"][10, 10] = True
        pre["cloud_buffer"][10, 10] = True
        status, reasons, _, full, region = quality.classify_quality(pre, post, target, False)
        self.assertEqual(status, "excluded_by_automatic_quality_screen")
        self.assertIn("cloud_or_buffer_on_target", reasons)
        self.assertEqual(full["denominator_pixels"], 4096)
        self.assertEqual(region["denominator_pixels"], 2)
        self.assertEqual(full["cloud_fraction"], 1/4096)
        self.assertEqual(region["cloud_fraction"], .5)

    def test_two_date_union_does_not_double_count_the_same_cloud_pixel(self):
        pre, post = zero_observation(), zero_observation()
        target = np.zeros((64, 64), dtype=np.uint8); target[10, 10] = 1
        for record in (pre, post):
            record["cloud"][10, 10] = record["cloud_buffer"][10, 10] = True
        _, _, _, full, region = quality.classify_quality(pre, post, target, False)
        self.assertEqual(full["cloud_pixels"], 1)
        self.assertEqual(region["cloud_fraction"], 1)

    def test_empty_target_remains_a_label_review_case(self):
        target = np.zeros((64, 64), dtype=np.uint8)
        status, reasons, _, _, region = quality.classify_quality(
            zero_observation(), zero_observation(), target, False)
        self.assertEqual(status, "needs_label_review")
        self.assertEqual(reasons, ["empty_rasterized_new_loss_target"])
        self.assertEqual(region["region_pixels"], 0)
        self.assertIsNone(region["usable_fraction"])

    def test_invalid_target_is_excluded_while_shadow_and_haze_remain_review_flags(self):
        target = np.zeros((64, 64), dtype=np.uint8); target[20, 20] = 1
        for name, expected_status, expected_reason in (
            ("invalid", "excluded_by_automatic_quality_screen", "invalid_target_pixels"),
            ("shadow_risk", "needs_visual_review", "suspected_shadow_on_target"),
            ("haze_risk", "needs_visual_review", "suspected_haze_on_target"),
        ):
            with self.subTest(mask=name):
                pre, post = zero_observation(), zero_observation()
                pre[name][20, 20] = True
                status, reasons, _, _, _ = quality.classify_quality(pre, post, target, False)
                self.assertEqual(status, expected_status)
                self.assertIn(expected_reason, reasons)

    def test_incomplete_context_requires_review_even_for_unflagged_core(self):
        target = np.zeros((64, 64), dtype=np.uint8); target[20, 20] = 1
        status, reasons, _, _, _ = quality.classify_quality(
            zero_observation(), zero_observation(), target, True)
        self.assertEqual(status, "needs_visual_review")
        self.assertIn("incomplete_off_crop_context", reasons)

    def test_clean_screen_is_explicitly_pending_visual_review(self):
        target = np.zeros((64, 64), dtype=np.uint8); target[20, 20] = 1
        status, reasons, _, _, _ = quality.classify_quality(
            zero_observation(), zero_observation(), target, False)
        self.assertEqual(status, "passed_automatic_screen_pending_visual_review")
        self.assertEqual(reasons, [])

    def test_pair_assessment_never_automatically_approves_training(self):
        target = np.zeros((64, 64), dtype=np.uint8); target[20, 20] = 1
        reader = SimpleNamespace(config_id="frozen", observation=mock.Mock(side_effect=[zero_observation(), zero_observation()]))
        pair = {"pair_id": "pair-fixture", "group_id": "group-fixture", "split": "train",
                "tile": "36UXA", "pre_date": "2018-07-31", "post_date": "2018-08-05",
                "anchor_event_ids": ["event-fixture"], "pre_path": "never-open-before",
                "post_path": "never-open-after", "grid": core()}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(quality, "label_target", return_value=target), \
                mock.patch("validate_sample.write_tif"):
            result = quality.assess_pair(pair, reader, {}, Path(directory))
        self.assertEqual(result["quality_status"], "passed_automatic_screen_pending_visual_review")
        self.assertIs(result["training_approved"], False)
        self.assertEqual(result["visual_review"], "unreviewed")
        self.assertEqual((result["pair_id"], result["group_id"], result["split"]),
                         (pair["pair_id"], pair["group_id"], pair["split"]))


class LabelTargetTests(unittest.TestCase):
    def fixture(self):
        grid = {"crs": "EPSG:32636", "width": 64, "height": 64,
                "transform": [10., 0., 0., 0., -10., 640.], "bounds": [0, 0, 640, 640]}
        events = {
            "old": {"properties": {"temporal_source_ids": ["temporal"], "observation_date": "2018-07-31"},
                    "geometry": box(0, 0, 320, 640)},
            "new": {"properties": {"temporal_source_ids": ["temporal"], "observation_date": "2018-08-05"},
                    "geometry": box(300, 0, 340, 640)},
            "baseline": {"properties": {"temporal_source_ids": [], "observation_date": "2018-08-05"},
                         "geometry": box(0, 0, 640, 640)},
        }
        pair = {"pre_date": "2018-07-31", "post_date": "2018-08-05", "grid": grid,
                "label_event_ids_by_post_date": ["old", "new"], "context_event_ids": ["old", "new", "baseline"]}
        return pair, events

    def test_only_manifest_temporal_labels_supply_the_new_loss_target(self):
        pair, events = self.fixture()
        target = quality.label_target(pair, events)
        expected = np.zeros((64, 64), dtype=np.uint8); expected[:, 32:34] = 1
        np.testing.assert_array_equal(target, expected)

    def test_manifest_cannot_silently_reintroduce_baseline_only_labels(self):
        pair, events = self.fixture()
        pair["label_event_ids_by_post_date"].append("baseline")
        with self.assertRaisesRegex(ValueError, "baseline-only event"):
            quality.label_target(pair, events)


if __name__ == "__main__":
    unittest.main()
