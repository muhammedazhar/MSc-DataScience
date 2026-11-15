"""Tiny array-only cloud/shadow screening regressions; no model or imagery."""

import importlib.util
import os
from pathlib import Path
import unittest

for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "GDAL_NUM_THREADS",
    "OPJ_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "Solutions/pixelqa_masks.py"
SPEC = importlib.util.spec_from_file_location("pixelqa_masks", MODULE_PATH)
qa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa)


class MaskTests(unittest.TestCase):
    def inputs(self, shape=(9, 9)):
        return dict(probability=np.zeros(shape), product_cloud=np.zeros(shape, bool),
                    invalid=np.zeros(shape, bool), nir=np.full(shape, .1),
                    blue=np.full(shape, .1), red=np.full(shape, .1),
                    solar_azimuth_deg=180, resolution_m=20, cloud_buffer_m=0,
                    shadow_distance_m=60)

    def test_southern_sun_projects_shadows_north(self):
        inputs = self.inputs()
        inputs["probability"][4, 4] = .8
        masks = qa.assess_masks(**inputs)
        expected = np.zeros((9, 9), bool)
        expected[1:4, 4] = True
        np.testing.assert_array_equal(masks["shadow_risk"], expected)

    def test_eastern_sun_projects_shadows_west(self):
        inputs = self.inputs()
        inputs["solar_azimuth_deg"] = 90
        inputs["probability"][4, 4] = .8
        masks = qa.assess_masks(**inputs)
        expected = np.zeros((9, 9), bool)
        expected[4, 1:4] = True
        np.testing.assert_array_equal(masks["shadow_risk"], expected)

    def test_projection_does_not_wrap_at_edges(self):
        inputs = self.inputs()
        inputs["solar_azimuth_deg"] = 90
        inputs["probability"][4, 0] = .8
        inputs["shadow_distance_m"] = 10000
        masks = qa.assess_masks(**inputs)
        self.assertFalse(masks["shadow_risk"].any())
        self.assertFalse(masks["shadow_risk"][:, -1].any())

    def test_dark_forest_without_cloud_is_not_shadow(self):
        inputs = self.inputs()
        inputs["nir"][:] = .01
        masks = qa.assess_masks(**inputs)
        self.assertFalse(masks["shadow_risk"].any())

    def test_cloud_union_threshold_and_physical_buffer(self):
        inputs = self.inputs()
        inputs["probability"][4, 4] = .4
        inputs["product_cloud"][0, 0] = True
        inputs["cloud_buffer_m"] = 20
        masks = qa.assess_masks(**inputs)
        self.assertTrue(masks["cloud"][4, 4])
        self.assertTrue(masks["cloud"][0, 0])
        self.assertTrue(masks["cloud_buffer"][4, 3])
        self.assertTrue(masks["cloud_buffer"][3, 4])
        self.assertFalse(masks["cloud_buffer"][3, 3])
        self.assertFalse(masks["shadow_risk"][3, 4])

    def test_bright_nir_projection_is_not_shadow(self):
        inputs = self.inputs()
        inputs["probability"][4, 4] = .9
        inputs["nir"][:] = .3
        self.assertFalse(qa.assess_masks(**inputs)["shadow_risk"].any())

    def test_bad_values_remain_invalid_instead_of_clear(self):
        inputs = self.inputs()
        inputs["probability"][0, 0] = np.nan
        inputs["probability"][0, 1] = 1.5
        inputs["nir"][0, 2] = np.inf
        inputs["blue"][0, 3] = np.nan
        inputs["product_cloud"][0, :4] = True
        masks = qa.assess_masks(**inputs)
        self.assertTrue(masks["invalid"][0, :4].all())
        self.assertFalse(masks["cloud"][0, :4].any())
        self.assertFalse(masks["shadow_risk"][0, :4].any())
        self.assertFalse(masks["haze_risk"][0, :4].any())

    def test_haze_is_a_separate_blue_excess_screen(self):
        inputs = self.inputs()
        inputs["blue"][1, 1] = .20
        inputs["red"][1, 1] = .10
        inputs["blue"][1, 2] = .20
        inputs["red"][1, 2] = .19
        masks = qa.assess_masks(**inputs)
        self.assertTrue(masks["haze_risk"][1, 1])
        self.assertFalse(masks["haze_risk"][1, 2])
        self.assertFalse(masks["cloud"].any())

    def test_custom_haze_thresholds_are_respected(self):
        inputs = self.inputs()
        inputs["blue"][1, 1] = .20
        inputs["red"][1, 1] = .10
        self.assertTrue(qa.assess_masks(**inputs)["haze_risk"][1, 1])
        raised_blue = qa.assess_masks(**inputs, haze_blue_threshold=.25)
        raised_difference = qa.assess_masks(**inputs, haze_blue_minus_red_threshold=.15)
        self.assertFalse(raised_blue["haze_risk"].any())
        self.assertFalse(raised_difference["haze_risk"].any())
        inputs["blue"][2, 2] = .12
        inputs["red"][2, 2] = .10
        lowered = qa.assess_masks(**inputs, haze_blue_threshold=.1,
                                  haze_blue_minus_red_threshold=.01)
        self.assertTrue(lowered["haze_risk"][2, 2])

    def test_invalid_solar_metadata_and_shapes_are_rejected(self):
        for azimuth in (np.nan, np.inf, -1, 361):
            with self.subTest(azimuth=azimuth), self.assertRaises(ValueError):
                qa.assess_masks(**{**self.inputs(), "solar_azimuth_deg": azimuth})
        inputs = self.inputs()
        inputs["nir"] = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            qa.assess_masks(**inputs)


class RegionMetricTests(unittest.TestCase):
    def masks(self, shape=(3, 3)):
        return {name: np.zeros(shape, bool) for name in qa.MASK_NAMES}

    def test_empty_region_fractions_are_unknown(self):
        metrics = qa.region_metrics(self.masks(), np.zeros((3, 3), bool))
        self.assertEqual(metrics["denominator_pixels"], 0)
        self.assertEqual(metrics["usable_pixels"], 0)
        self.assertTrue(all(value is None for key, value in metrics.items() if key.endswith("_fraction")))

    def test_union_does_not_double_count_and_region_limits_counts(self):
        masks = self.masks()
        masks["cloud"][0, 0] = True
        masks["cloud_buffer"][0, :2] = True
        masks["shadow_risk"][1, 0] = True
        masks["haze_risk"][1, 0] = True
        masks["invalid"][1, 1] = True
        masks["invalid"][2, 2] = True
        region = np.zeros((3, 3), bool)
        region[:2, :2] = True
        metrics = qa.region_metrics(masks, region)
        self.assertEqual(metrics["region_pixels"], 4)
        self.assertEqual(metrics["valid_pixels"], 3)
        self.assertEqual(metrics["unusable_pixels"], 4)
        self.assertEqual(metrics["usable_pixels"], 0)
        self.assertEqual(metrics["cloud_fraction"], .25)
        self.assertEqual(metrics["cloud_buffer_fraction"], .5)
        self.assertEqual(metrics["shadow_risk_fraction"], .25)
        self.assertEqual(metrics["haze_risk_fraction"], .25)
        self.assertEqual(metrics["invalid_fraction"], .25)
        self.assertEqual(metrics["unusable_fraction"], 1)

    def test_missing_or_mismatched_region_masks_are_rejected(self):
        with self.assertRaises(ValueError):
            qa.region_metrics({}, np.zeros((3, 3), bool))
        with self.assertRaises(ValueError):
            qa.region_metrics(self.masks(), np.zeros((2, 2), bool))


if __name__ == "__main__":
    unittest.main()
