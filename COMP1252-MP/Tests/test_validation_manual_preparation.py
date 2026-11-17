"""Synthetic manual-review roster and context QA regressions; run under guard."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from prepare_validation_manual import control_roster, observation_metrics, validate_protocol


def arrays():
    import numpy as np
    result = {'features': np.ones((27, 64, 64), dtype='float32'),
              'feature_valid': np.ones((64, 64), dtype=bool),
              'control_exclusion': np.zeros((64, 64), dtype=bool)}
    for side in ('pre', 'post'):
        result[side+'_qa'] = np.zeros((5, 64, 64), dtype=bool)
        result[side+'_rgb'] = np.full((3, 64, 64), .15, dtype='float32')
        result[side+'_ndvi'] = np.full((64, 64), -.4 if side == 'pre' else .9, dtype='float32')
    return result


def observation():
    import numpy as np
    return {**{name: np.zeros((64, 64), dtype=bool) for name in
               ('cloud', 'cloud_buffer', 'shadow_risk', 'haze_risk', 'invalid')},
            'rgb': np.full((3, 64, 64), .15, dtype='float32'),
            'ndvi': np.full((64, 64), -.5, dtype='float32')}


class ManualPreparationTests(unittest.TestCase):
    def test_roster_uses_geometry_order_and_disjoint_full_halos_without_ndvi_class_gate(self):
        import numpy as np
        source = arrays()
        roster, controls, halos = control_roster(source)
        self.assertEqual([(r['height'], r['row'], r['col']) for r in roster],
                         [(16, 2, 2), (16, 2, 22), (16, 2, 42)])
        self.assertEqual(controls.shape, (3, 64, 64))
        self.assertTrue(all(r['pixels'] == 256 and r['context_pixels'] == 400 for r in roster))
        self.assertEqual(int(halos.sum()), 1200)
        self.assertFalse(np.any(halos.sum(axis=0) > 1))
        self.assertTrue(all(r['visual_review'] == 'unreviewed' and r['evaluation_approved'] is False for r in roster))
        source['pre_ndvi'][:] = .99; source['post_ndvi'][:] = -.99
        for col in range(64):
            source['post_rgb'][:, :, col] = col/100
        changed, changed_controls, changed_halos = control_roster(source)
        self.assertEqual(changed, roster)
        self.assertTrue(np.array_equal(changed_controls, controls))
        self.assertTrue(np.array_equal(changed_halos, halos))

    def test_flag_exclusion_or_nonfinite_value_only_in_halo_rejects_that_entire_square(self):
        for kind in ('qa', 'exclusion', 'feature_valid', 'feature_nan', 'rgb_nan', 'ndvi_nan'):
            import numpy as np
            with self.subTest(kind=kind):
                source = arrays()
                if kind == 'qa': source['post_qa'][2, 0, 0] = True
                elif kind == 'exclusion': source['control_exclusion'][0, 0] = True
                elif kind == 'feature_valid': source['feature_valid'][0, 0] = False
                elif kind == 'feature_nan': source['features'][26, 0, 0] = np.nan
                elif kind == 'rgb_nan': source['pre_rgb'][0, 0, 0] = np.nan
                else: source['post_ndvi'][0, 0] = np.nan
                roster, _, halos = control_roster(source, maximum=1)
                self.assertEqual((roster[0]['row'], roster[0]['col']), (2, 3))
                self.assertFalse(halos[0, 0, 0])

    def test_largest_fitting_size_first_then_exhaustion_keeps_short_roster(self):
        source = arrays(); source['control_exclusion'][:] = True
        source['control_exclusion'][10:26, 10:26] = False
        roster, controls, halos = control_roster(source)
        self.assertEqual([(r['height'], r['row'], r['col']) for r in roster], [(12, 12, 12)])
        self.assertEqual(int(controls.sum()), 144)
        self.assertEqual(int(halos.sum()), 256)

    def test_no_safe_square_is_explicit_empty_roster_not_clipped_control(self):
        source = arrays(); source['control_exclusion'][:] = True
        source['control_exclusion'][:11, :11] = False
        roster, controls, halos = control_roster(source)
        self.assertEqual(roster, [])
        self.assertEqual(controls.shape, (0, 64, 64))
        self.assertEqual(halos.shape, (0, 64, 64))
        with self.assertRaises(ValueError): control_roster(source, maximum=4)

    def test_visual_metrics_distinguish_roi_from_full_context_and_union_overlap(self):
        import numpy as np
        from audit_validation_context import context_mask
        roi = np.zeros((64, 64), dtype=bool); roi[20:24, 20:24] = True
        halo = context_mask(roi)
        record = observation(); record['cloud'][18, 18] = record['shadow_risk'][18, 18] = True
        record['haze_risk'][0, 0] = True
        result = observation_metrics(record, roi, halo)
        self.assertEqual(result['roi_pixels'], 16)
        self.assertEqual(result['context_pixels'], 64)
        self.assertEqual(result['roi']['unusable_pixels'], 0)
        self.assertEqual(result['context']['unusable_pixels'], 1)
        self.assertEqual(result['context']['cloud_pixels'], 1)
        self.assertEqual(result['context']['shadow_risk_pixels'], 1)
        self.assertEqual(result['context']['haze_risk_pixels'], 0)
        self.assertFalse(result['context_clear'])

    def test_visual_context_requires_finite_rgb_ndvi_without_greenness_claim(self):
        import numpy as np
        from audit_validation_context import context_mask
        roi = np.zeros((64, 64), dtype=bool); roi[20:24, 20:24] = True
        halo = context_mask(roi); record = observation()
        self.assertTrue(observation_metrics(record, roi, halo)['context_clear'])
        record['rgb'][1, 18, 18] = np.nan; record['ndvi'][25, 25] = np.nan
        result = observation_metrics(record, roi, halo)
        self.assertEqual(result['roi_nonfinite_rgb_ndvi_pixels'], 0)
        self.assertEqual(result['context_nonfinite_rgb_ndvi_pixels'], 2)
        self.assertFalse(result['context_clear'])

    def test_clipped_or_redrawn_context_cannot_be_used_for_visual_quality_claim(self):
        import numpy as np
        from audit_validation_context import context_mask
        roi = np.zeros((64, 64), dtype=bool); roi[20:24, 20:24] = True
        with self.assertRaises(ValueError): observation_metrics(observation(), roi, roi.copy())
        roi[:] = False; roi[:4, :4] = True
        with self.assertRaises(ValueError): observation_metrics(observation(), roi, np.ones((64, 64), dtype=bool))
        source = arrays(); source['control_exclusion'] = source['control_exclusion'].astype('uint8')
        with self.assertRaises(ValueError): control_roster(source)

    def test_protocol_requires_distinct_manual_scope_and_frozen_capacity_rules(self):
        from copy import deepcopy
        import json
        protocol = json.loads((Path(__file__).resolve().parents[1]/'Docs/Reviews/ValidationManualV4/protocol.json').read_text())
        validate_protocol(protocol)
        mutations = [('schema_version', 3), ('rule.maximum_control_candidates', 4),
                     ('rule.no_ndvi_threshold_or_ranking', False),
                     ('rule.control_contexts_mutually_disjoint', False),
                     ('resources.worker_rss_mib', 1024), ('resources.available_memory_reserve_gib', .5)]
        for name, value in mutations:
            changed = deepcopy(protocol); group, _, field = name.partition('.')
            if field: changed[group][field] = value
            else: changed[group] = value
            with self.subTest(name=name), self.assertRaises(ValueError): validate_protocol(changed)


if __name__ == '__main__':
    unittest.main()
