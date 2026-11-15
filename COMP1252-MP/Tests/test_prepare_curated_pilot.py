"""Small proposal/window fixtures; no imagery, QA classifier or JP2 reads."""

import os
from pathlib import Path
import sys
import unittest

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Solutions"))

import numpy as np

from prepare_curated_pilot import propose_control, select_timeline


def observation():
    result = {name: np.zeros((64, 64), dtype=bool) for name in
              ("cloud", "cloud_buffer", "shadow_risk", "haze_risk", "invalid")}
    result["ndvi"] = np.full((64, 64), .75, dtype="float32")
    return result


class TimelineTests(unittest.TestCase):
    def test_complete_window_is_sorted_and_retains_unusable_metadata(self):
        scenes = [{"tile": "36UXA", "acquisition_date": day, "scene_id": day, "metadata_usable": day != "2018-08-05"}
                  for day in ("2018-09-20", "2018-07-01", "2018-08-05", "2018-09-19", "2018-06-20")]
        selected, bounds = select_timeline(scenes, "36UXA", "2018-08-05")
        self.assertEqual([row["acquisition_date"] for row in selected],
                         ["2018-07-01", "2018-08-05", "2018-09-19"])
        self.assertFalse(selected[1]["metadata_usable"])
        self.assertEqual(bounds["start"], "2018-06-21")
        self.assertEqual(bounds["end"], "2018-09-19")

    def test_cross_year_and_other_tile_are_not_silently_claimed_as_reviewed(self):
        scenes = [{"tile": tile, "acquisition_date": day, "scene_id": tile+day}
                  for tile, day in (("36UXA", "2017-12-20"), ("36UXA", "2018-01-15"), ("36UYA", "2018-01-15"))]
        selected, bounds = select_timeline(scenes, "36UXA", "2018-01-15")
        self.assertEqual(len(selected), 1)
        self.assertEqual(bounds["same_year_only"], 2018)


class ControlProposalTests(unittest.TestCase):
    def test_tied_clear_proposal_is_deterministic_and_unapproved(self):
        pre, post = observation(), observation()
        mask, metadata = propose_control(pre, post, np.zeros((64, 64), dtype=bool))
        self.assertEqual((metadata["row"], metadata["col"], metadata["height"]), (0, 0, 16))
        self.assertEqual(int(mask.sum()), 256)
        self.assertIs(metadata["training_approved"], False)
        self.assertEqual(metadata["control_label_review"], "unreviewed")

    def test_excluded_geometry_quality_and_ndvi_changes_cannot_supply_control_pixels(self):
        pre, post = observation(), observation()
        excluded = np.ones((64, 64), dtype=bool)
        excluded[30:46, 25:41] = False
        post["haze_risk"][30, 25] = True
        post["ndvi"][31, 26] = .50
        mask, metadata = propose_control(pre, post, excluded)
        self.assertTrue(metadata["proposal_found"])
        self.assertEqual(metadata["height"], 12)
        self.assertFalse(mask[30, 25])
        self.assertFalse(mask[31, 26])
        self.assertFalse((mask & excluded).any())

    def test_absence_of_adequate_roi_is_explicit_without_empty_negative_label(self):
        pre, post = observation(), observation()
        mask, metadata = propose_control(pre, post, np.ones((64, 64), dtype=bool))
        self.assertFalse(metadata["proposal_found"])
        self.assertFalse(mask.any())
        self.assertIs(metadata["training_approved"], False)

    def test_eight_pixel_fallback_preserves_explicit_size_and_unknown_review(self):
        pre, post = observation(), observation()
        excluded = np.ones((64, 64), dtype=bool)
        excluded[22:30, 40:48] = False
        mask, metadata = propose_control(pre, post, excluded)
        self.assertEqual((metadata["row"], metadata["col"], metadata["height"], metadata["width"]),
                         (22, 40, 8, 8))
        self.assertEqual(int(mask.sum()), 64)
        self.assertEqual(metadata["control_label_review"], "unreviewed")

    def test_spectral_stability_alone_does_not_override_cloud_or_invalid(self):
        pre, post = observation(), observation()
        pre["invalid"][:] = True
        mask, metadata = propose_control(pre, post, np.zeros((64, 64), dtype=bool))
        self.assertFalse(mask.any())
        self.assertFalse(metadata["proposal_found"])


if __name__ == "__main__":
    unittest.main()
