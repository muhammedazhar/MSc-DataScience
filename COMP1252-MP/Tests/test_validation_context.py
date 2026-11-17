"""Context QA must protect the full input region used by each reviewed pixel."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from audit_validation_context import context_mask, propose_context_control, qa_region_counts, validate_protocol


QA_NAMES = ('cloud', 'cloud_buffer', 'shadow_risk', 'haze_risk', 'invalid')


def observation(ndvi=.8):
    return {**{name: np.zeros((64, 64), dtype=bool) for name in QA_NAMES},
            'rgb': np.full((3, 64, 64), .15, dtype='float32'),
            'ndvi': np.full((64, 64), ndvi, dtype='float32'),
            'feature_valid': np.ones((64, 64), dtype=bool)}


def unique_eight_pixel_control():
    """Only ROI[2:10,2:10] has its complete halo inside feature-valid context."""
    pre, post = observation(), observation()
    for record in (pre, post):
        record['feature_valid'][:] = False
        record['feature_valid'][:12, :12] = True
    return pre, post, np.zeros((64, 64), dtype=bool)


class ValidationContextTests(unittest.TestCase):
    def test_fixed_protocol_rejects_context_support_architecture_and_capacity_changes(self):
        protocol = json.loads((Path(__file__).resolve().parents[1]/'Docs/Reviews/ValidationContextV3/protocol.json').read_text())
        validate_protocol(protocol)
        for section, key, invalid in (('rule', 'context_radius_pixels', 1),
                ('rule', 'minimum_positive_interior_pixels', 1),
                ('rule', 'require_full_target_and_interior_context', False),
                ('architecture', 'global_context_operations', True),
                ('resources', 'workers', 2), ('selection', 'maximum_cases', 2)):
            changed = deepcopy(protocol); changed[section][key] = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol(changed)

    def test_two_pixel_context_is_exact_square_union_and_preserves_input(self):
        roi = np.zeros((64, 64), dtype=bool)
        roi[32, 32] = True
        expected = np.zeros_like(roi); expected[30:35, 30:35] = True
        original = roi.copy()
        np.testing.assert_array_equal(context_mask(roi), expected)
        self.assertEqual(int(context_mask(roi).sum()), 25)
        # Square corners are part of two successive 3x3 convolutions; a
        # Euclidean-radius-two disk would incorrectly omit these inputs.
        self.assertTrue(context_mask(roi)[30, 30])
        np.testing.assert_array_equal(roi, original)
        roi[32, 34] = True
        self.assertEqual(int(context_mask(roi).sum()), 35)

    def test_empty_wrong_type_shape_and_incomplete_edge_context_are_rejected(self):
        roi = np.zeros((64, 64), dtype=bool); roi[32, 32] = True
        invalid = (np.zeros_like(roi), roi.astype('uint8'),
                   np.ones((63, 64), dtype=bool), np.ones((64, 64, 1), dtype=bool))
        for candidate in invalid:
            with self.subTest(shape=candidate.shape, dtype=candidate.dtype), self.assertRaises(ValueError):
                context_mask(candidate)
        for row, col in ((1, 20), (20, 1), (62, 20), (20, 62)):
            edge = np.zeros_like(roi); edge[row, col] = True
            with self.subTest(row=row, col=col), self.assertRaises(ValueError):
                context_mask(edge)
        boundary = np.zeros_like(roi); boundary[2, 2] = True
        expected = np.zeros_like(roi); expected[:5, :5] = True
        np.testing.assert_array_equal(context_mask(boundary), expected)

    def test_quality_union_counts_overlap_once_and_keeps_both_dates_separate(self):
        roi = np.zeros((64, 64), dtype=bool); roi[16, 16] = True
        region = context_mask(roi); pre, post = observation(), observation()
        pre['cloud'][15, 15] = pre['cloud_buffer'][15, 15] = True
        pre['invalid'][15, 16] = True
        post['shadow_risk'][17, 17] = post['haze_risk'][17, 17] = True
        original = deepcopy((region, pre, post))
        counts = qa_region_counts(region, pre, post)
        self.assertEqual(counts['pixels'], 25)
        self.assertEqual(counts['pre_unusable_pixels'], 2)
        self.assertEqual(counts['post_unusable_pixels'], 1)
        self.assertEqual(counts['pre_components'], {'cloud': 1, 'cloud_buffer': 1,
            'shadow_risk': 0, 'haze_risk': 0, 'invalid': 1})
        self.assertEqual(counts['post_components'], {'cloud': 0, 'cloud_buffer': 0,
            'shadow_risk': 1, 'haze_risk': 1, 'invalid': 0})
        np.testing.assert_array_equal(region, original[0])
        for actual, before in zip((pre, post), original[1:]):
            for key in actual:
                np.testing.assert_array_equal(actual[key], before[key])

    def test_flag_outside_roi_but_inside_halo_rejects_control_and_distant_flag_does_not(self):
        pre, post, excluded = unique_eight_pixel_control()
        wanted = np.zeros((64, 64), dtype=bool); wanted[2:10, 2:10] = True
        mask, metadata = propose_context_control(pre, post, excluded, sizes=(8,))
        np.testing.assert_array_equal(mask, wanted)
        self.assertIs(metadata['proposal_found'], True)
        post['cloud'][0, 0] = True  # Beyond ROI, but within its required input context.
        mask, metadata = propose_context_control(pre, post, excluded, sizes=(8,))
        self.assertFalse(mask.any())
        self.assertIs(metadata['proposal_found'], False)
        post['cloud'][0, 0] = False; post['cloud'][50, 50] = True
        mask, metadata = propose_context_control(pre, post, excluded, sizes=(8,))
        np.testing.assert_array_equal(mask, wanted)
        self.assertIs(metadata['proposal_found'], True)

    def test_full_halo_catalogue_exclusion_is_required_without_shrinking_roi(self):
        pre, post, excluded = unique_eight_pixel_control()
        excluded[0, 0] = True
        mask, metadata = propose_context_control(pre, post, excluded, sizes=(8,))
        self.assertFalse(mask.any())
        self.assertIs(metadata['proposal_found'], False)
        # No partial control or smaller implicit fallback can hide the excluded
        # context; only the explicitly passed search sizes are allowed.
        self.assertEqual(int(mask.sum()), 0)

    def test_control_search_prefers_largest_size_before_first_position_not_best_ndvi(self):
        pre, post = observation(.7), observation(.7)
        excluded = np.zeros((64, 64), dtype=bool)
        for record in (pre, post):
            record['feature_valid'][:] = False
            record['feature_valid'][:12, :12] = True       # First possible 8x8.
            record['feature_valid'][18:38, 18:38] = True  # Later 16x16.
        mask, metadata = propose_context_control(pre, post, excluded)
        expected = np.zeros_like(excluded); expected[20:36, 20:36] = True
        np.testing.assert_array_equal(mask, expected)
        self.assertIs(metadata['proposal_found'], True)
        # With equal valid sizes, deterministic row/column order wins even
        # when a later patch has a numerically greener proposal score.
        pre, post = observation(.7), observation(.7)
        for record in (pre, post):
            record['ndvi'][30:50, 30:50] = .95
        mask, _ = propose_context_control(pre, post, excluded)
        expected[:] = False; expected[2:18, 2:18] = True
        np.testing.assert_array_equal(mask, expected)

    def test_feature_invalid_or_nonfinite_halo_blocks_otherwise_green_control(self):
        for corruption in ('feature_valid', 'rgb', 'ndvi'):
            pre, post, excluded = unique_eight_pixel_control()
            if corruption == 'feature_valid':
                pre['feature_valid'][0, 0] = False
            elif corruption == 'rgb':
                post['rgb'][1, 0, 0] = float('nan')
            else:
                pre['ndvi'][0, 0] = float('inf')
            with self.subTest(corruption=corruption):
                mask, metadata = propose_context_control(pre, post, excluded, sizes=(8,))
                self.assertFalse(mask.any())
                self.assertIs(metadata['proposal_found'], False)

    def test_green_stable_threshold_is_on_roi_while_halo_still_requires_finite_inputs(self):
        pre, post, excluded = unique_eight_pixel_control()
        pre['ndvi'][0, 0] = post['ndvi'][0, 0] = .2  # Halo can contain clear nonforest.
        wanted = np.zeros((64, 64), dtype=bool); wanted[2:10, 2:10] = True
        mask, _ = propose_context_control(pre, post, excluded, sizes=(8,))
        np.testing.assert_array_equal(mask, wanted)
        for corruption in ('low_pre_ndvi', 'endpoint_difference'):
            before, after, forbidden = unique_eight_pixel_control()
            if corruption == 'low_pre_ndvi':
                before['ndvi'][2, 2] = .5
            else:
                after['ndvi'][2, 2] = .68
            with self.subTest(corruption=corruption):
                mask, metadata = propose_context_control(before, after, forbidden, sizes=(8,))
                self.assertFalse(mask.any())
                self.assertIs(metadata['proposal_found'], False)


if __name__ == '__main__':
    unittest.main()
