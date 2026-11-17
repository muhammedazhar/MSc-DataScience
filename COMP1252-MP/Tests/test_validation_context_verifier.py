"""Independent local-context checks must retain halo, scope and model binding."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Solutions'))
from verify_validation_context import (architecture_contract, independent_control,
                                       control_gate_counts, inspect_context, square_halo)


class ContextVerifierTests(unittest.TestCase):
    def arrays(self):
        return (np.zeros((5, 64, 64), dtype=bool), np.zeros((5, 64, 64), dtype=bool),
                np.zeros((27, 64, 64), dtype='float32'), np.ones((64, 64), dtype=bool))

    def test_square_halo_includes_diagonal_dependencies_and_exposes_clipped_context(self):
        roi = np.zeros((64, 64), dtype=bool); roi[2, 2] = True
        halo, inside = square_halo(roi)
        self.assertTrue(inside)
        self.assertEqual(int(halo.sum()), 25)
        self.assertTrue(halo[0, 0])
        roi[:] = False; roi[0, 0] = True
        halo, inside = square_halo(roi)
        self.assertFalse(inside)
        self.assertEqual(int(halo.sum()), 9)
        pre, post, features, valid = self.arrays()
        self.assertFalse(inspect_context(roi, pre, post, features, valid)['context_clear'])

    def test_full_target_halo_failure_cannot_be_hidden_by_a_smaller_clean_interior(self):
        pre, post, features, valid = self.arrays()
        target = np.zeros((64, 64), dtype=bool); target[30:36, 30:36] = True
        interior = np.zeros((64, 64), dtype=bool); interior[32:34, 32:34] = True
        pre[0, 28, 28] = True
        self.assertFalse(inspect_context(target, pre, post, features, valid)['context_clear'])
        self.assertTrue(inspect_context(interior, pre, post, features, valid)['context_clear'])
        pre[:] = False; features[26, 28, 28] = np.nan
        self.assertFalse(inspect_context(target, pre, post, features, valid)['context_clear'])
        self.assertTrue(inspect_context(interior, pre, post, features, valid)['context_clear'])

    def test_control_search_rechecks_full_context_and_keeps_first_fixed_geometry(self):
        pre, post, features, valid = self.arrays()
        excluded = np.zeros((64, 64), dtype=bool)
        target_context = excluded.copy(); target_context[:20, :20] = True
        ndvi = np.full((64, 64), .8, dtype='float32')
        control, proposal = independent_control(target_context, excluded, pre, post,
                                                ndvi, ndvi, features, valid)
        self.assertEqual((proposal['row'], proposal['col'], proposal['height']), (2, 22, 16))
        self.assertEqual(int(control.sum()), 256)
        # The flagged pixel is outside that control ROI, but inside its halo.
        pre[0, 0, 20] = True
        control, changed = independent_control(target_context, excluded, pre, post,
                                              ndvi, ndvi, features, valid)
        self.assertEqual((changed['row'], changed['col'], changed['height']), (2, 23, 16))
        context, inside = square_halo(control)
        self.assertTrue(inside)
        self.assertFalse((context & pre[0]).any())
        self.assertFalse(changed['training_approved'])
        self.assertFalse(changed['evaluation_approved'])
        ndvi[:] = 0.; ndvi[30:38, 30:38] = .8
        _, smallest = independent_control(target_context, excluded, pre, post,
                                         ndvi, ndvi, features, valid)
        self.assertEqual((smallest['row'], smallest['col'], smallest['height']), (30, 30, 8))

    def test_actual_factory_proves_radius_two_and_rejects_global_or_changed_context(self):
        source = ROOT/'Solutions/train_pilot_baseline.py'
        contract = architecture_contract(source)
        self.assertEqual((contract['context_radius_pixels'], contract['spatial_receptive_field_pixels'],
                          contract['parameter_count']), (2, 5, 2545))
        content = source.read_text()
        changed = content.replace('nn.Conv2d(8, 1, 1)',
                                  'nn.AdaptiveAvgPool2d(1), nn.Conv2d(8, 1, 1)', 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.py'; path.write_text(changed)
            with self.assertRaises(ValueError):
                architecture_contract(path)

    def test_qa_safe_contexts_do_not_override_failed_ndvi_control_proposals(self):
        pre, post, features, valid = self.arrays()
        excluded = np.zeros((64, 64), dtype=bool)
        ndvi = np.full((64, 64), .5, dtype='float32')
        arrays = {'pre_qa': pre, 'post_qa': post, 'features': features,
                  'feature_valid': valid, 'control_exclusion': excluded,
                  'pre_ndvi': ndvi, 'post_ndvi': ndvi,
                  'pre_rgb': np.zeros((3, 64, 64), dtype='float32'),
                  'post_rgb': np.zeros((3, 64, 64), dtype='float32')}
        control, proposal = independent_control(excluded, excluded, pre, post, ndvi, ndvi, features, valid)
        self.assertFalse(proposal['proposal_found'])
        self.assertFalse(control.any())
        result = control_gate_counts(arrays)
        self.assertTrue(result['diagnostic_only'])
        self.assertEqual(result['new_regions_selected'], 0)
        for row in result['per_size']:
            self.assertGreater(row['inbounds_squares'], 0)
            self.assertEqual(row['qa_finite_clear_context_squares'], row['inbounds_squares'])
            self.assertEqual(row['qa_finite_exclusion_clear_context_squares'], row['inbounds_squares'])
            self.assertEqual(row['absolute_ndvi_change_below_point1_roi_squares'], row['inbounds_squares'])
            self.assertEqual(row['both_ndvi_above_point6_roi_squares'], 0)
            self.assertEqual(row['all_gates_pass_squares'], 0)


if __name__ == '__main__':
    unittest.main()
