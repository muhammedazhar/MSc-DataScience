"""Unscreened geometry must preserve blockers, complete anchors and unknown QA."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Solutions'))
from audit_validation_unscreened import complement, erode_numpy, reasons, support


class UnscreenedInventoryTests(unittest.TestCase):
    def pair(self):
        return {'pair_id': 'p', 'group_id': 'g', 'split': 'validation',
            'status': 'blocked_for_temporal_training', 'blocking_reasons': ['baseline_only_provenance'],
            'anchor_event_ids': ['old', 'new'], 'label_event_ids_by_post_date': ['old', 'new'],
            'pre_date': '2019-01-01', 'post_date': '2019-01-10',
            'grid': {'width': 64, 'height': 64, 'crs': 'EPSG:32636',
                'transform': [10., 0., 0., 0., -10., 640.], 'bounds': [0., 0., 640., 640.]}}

    def test_blocked_multi_anchor_complement_is_preserved_without_status_filter(self):
        p = self.pair()
        result = complement([p], [], [p])
        self.assertEqual(result, [p])
        self.assertEqual(result[0]['anchor_event_ids'], ['old', 'new'])
        self.assertEqual(result[0]['blocking_reasons'], ['baseline_only_provenance'])
        q = dict(p, pair_id='q')
        self.assertEqual(complement([q, p], [q], [p]), [p])

    def test_complement_rejects_silent_omission_changed_blocker_or_duplicate(self):
        p = self.pair()
        for retained in ([], [dict(p, blocking_reasons=[])], [dict(p, anchor_event_ids=['old'])]):
            with self.subTest(retained=retained), self.assertRaises(ValueError):
                complement([p], [], retained)
        with self.assertRaises(ValueError): complement([p, p], [], [p])
        with self.assertRaises(ValueError): complement([dict(p, anchor_event_ids=[])], [], [p])

    def test_temporal_reference_excludes_overlap_with_cumulative_prior_loss(self):
        p = self.pair()
        def event(day, geom):
            return {'properties': {'observation_date': day, 'temporal_source_ids': ['s']}, 'geometry': geom}
        events = {'old': event('2019-01-01', box(200, 200, 260, 260)),
                  'new': event('2019-01-05', box(220, 220, 280, 280))}
        producer = support(p, 'new', events)
        independent = support(p, 'new', events, independent=True)
        self.assertEqual(producer, independent)
        self.assertEqual(producer['anchor_pixels'], 36)
        self.assertEqual(producer['temporal_clipped_pixels'], 16)
        self.assertEqual(producer['target_pixels'], 20)
        self.assertLess(producer['positive_interior_pixels'], producer['anchor_interior_pixels'])

    def test_numpy_erosion_retains_real_interior_and_no_clipped_edge_fallback(self):
        mask = np.zeros((64, 64), dtype=bool); mask[20:24, 20:24] = True
        self.assertEqual(int(erode_numpy(mask).sum()), 4)
        mask[:] = False; mask[:2, :8] = True
        self.assertEqual(int(erode_numpy(mask).sum()), 0)

    def test_geometry_never_overrides_provenance_status_or_coverage_blocks(self):
        p = self.pair(); p['blocking_reasons'].append('post_scene_does_not_cover_crop')
        result = reasons(p, {'positive_interior_pixels': 100}, True, ['later'], False, True)
        self.assertEqual(result, ['original:baseline_only_provenance',
            'original:post_scene_does_not_cover_crop', 'original_pair_status_blocked'])

    def test_all_metadata_failures_and_exposed_group_are_retained(self):
        p = self.pair()
        result = reasons(p, {'positive_interior_pixels': 0}, False, [], True, False)
        self.assertTrue({'insufficient_positive_interior', 'train_test_footprint_overlap',
            'no_later_metadata_usable_observation', 'previously_inspected_group',
            'endpoint_metadata_unusable'} <= set(result))


if __name__ == '__main__':
    unittest.main()
