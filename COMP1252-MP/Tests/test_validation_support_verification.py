"""Independent support verification must preserve clipping and blind selection."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from verify_validation_support_pool import (independent_gate_flags,
                                            independent_shortlist,
                                            reconstruct_support)


def candidate(event='event-a', group='group-a', pair='pair-a'):
    return {'event_id': event, 'group_id': group, 'pair_id': pair,
            'stratum': '36UXA_time-dependent', 'qa_status': 'passed_automatic_screen_pending_visual_review',
            'annotation_date': '2019-06-11', 'post_date': '2019-06-11', 'interval_days': 5,
            'positive_interior_pixels': 4, 'endpoint_crop_unusable_pixels': [0, 0],
            'endpoint_target_unusable_pixels': [0, 0], 'interior_unusable_pixels': 0,
            'footprint_isolated': True, 'later_usable_scene_ids': ['scene-later']}


class IndependentSupportVerificationTests(unittest.TestCase):
    def test_published_temporal_clipping_changes_support_for_the_same_anchor(self):
        from shapely.geometry import box
        grid = {'crs': 'EPSG:32636', 'width': 64, 'height': 64,
                'transform': [10., 0., 500000., 0., -10., 5600000.]}
        anchor = box(500100., 5599800., 500140., 5599840.)
        full = np.zeros((6, 64, 64), dtype='uint8'); full[5] = 1
        broad = reconstruct_support(anchor, grid, full)
        self.assertEqual((broad['anchor_pixels'], broad['target_pixels'], broad['positive_interior_pixels']), (16, 16, 4))
        clipped = full.copy(); clipped[5, :, 10] = 0
        narrow = reconstruct_support(anchor, grid, clipped)
        self.assertEqual((narrow['anchor_pixels'], narrow['target_pixels'], narrow['positive_interior_pixels']), (16, 12, 2))
        contaminated = full.copy(); contaminated[0] = 1
        self.assertEqual(reconstruct_support(anchor, grid, contaminated)['interior_unusable_pixels'], 4)

    def test_prior_group_and_claimed_eligibility_do_not_override_evidence(self):
        row = candidate(); row['eligible'] = True
        self.assertFalse(all(independent_gate_flags(row, set(), {'group-a'}).values()))
        self.assertFalse(all(independent_gate_flags(row, {'event-a'}, set()).values()))
        row['positive_interior_pixels'] = 3
        self.assertFalse(all(independent_gate_flags(row, set(), set()).values()))

    def test_shortlist_rebuild_is_order_invariant_and_retains_missing_slots(self):
        first = candidate(); second = candidate('event-b', 'group-b', 'pair-b')
        rows = [first, second]; original = deepcopy(rows)
        selected, missing = independent_shortlist(rows, set(), set())
        self.assertEqual(selected, independent_shortlist(list(reversed(rows)), set(), set())[0])
        self.assertEqual([row['role'] for row in selected], ['primary', 'alternate'])
        self.assertEqual(len(missing), 4)
        self.assertEqual(rows, original)
        self.assertEqual(independent_shortlist(rows, set(), {'group-a', 'group-b'})[0], [])


if __name__ == '__main__':
    unittest.main()
