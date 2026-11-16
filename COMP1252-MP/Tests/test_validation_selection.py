"""Metadata selection must preserve split isolation and predeclared ranking."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from select_validation_cases import POLICY, STRATA, choose_candidates


def grid(west):
    return {'crs': 'EPSG:32636', 'width': 64, 'height': 64,
            'bounds': [west, 0., west+640, 640.],
            'transform': [10., 0., west, 0., -10., 640.]}


class ValidationSelectionTests(unittest.TestCase):
    def fixture(self):
        events, pairs, quality, properties = [], {}, {}, {}
        for source_index, source in enumerate(STRATA):
            for option in range(2):
                number = source_index*2+option
                eid, pid, group = f'event-{number}', f'pair-{number}', f'group-{number}'
                tile = '36UXA' if source_index == 0 else '36UYA'
                g = grid(number*1000.)
                properties[eid] = {'event_id': eid, 'split': 'validation', 'group_id': group,
                    'temporal_source_ids': [source], 'area_m2': 2000., 'width_m': 50., 'height_m': 50.,
                    'group_footprint_bounds': g['bounds']}
                candidate = {'pair_id': pid, 'quality_status': 'passed_automatic_screen_pending_visual_review',
                             'pre_date': '2019-06-01', 'post_date': '2019-06-06'}
                events.append({'event_id': eid, 'split': 'validation', 'group_id': group,
                               'tile': tile, 'observation_date': '2019-06-06', 'top_candidates': [candidate]})
                pairs[pid] = {'pair_id': pid, 'split': 'validation', 'group_id': group,
                              'tile': tile, 'anchor_event_ids': [eid], 'grid': g,
                              'pre_date': '2019-06-01', 'post_date': '2019-06-06'}
                quality[pid] = {**deepcopy(pairs[pid]), 'quality_status': candidate['quality_status'],
                               'config_id': POLICY, 'training_approved': False,
                               'target_metrics': {'region_pixels': 16},
                               'per_date': {d: {'crop': {'unusable_pixels': 0},
                                               'target': {'unusable_pixels': 0},
                                               'target_median_ndvi': .5,
                                               'observation': {'cache_path': f'cache/{pid}-{d}.npz'}}
                                            for d in ('pre', 'post')}}
        scenes = [{'tile': tile, 'acquisition_date': '2019-06-11'} for tile in ('36UXA', '36UYA')]
        training = [{'split': 'train', 'grid': grid(100000.+i*1000)} for i in range(13)]
        return [events, pairs, quality, properties, scenes, training, 'EPSG:32636']

    def test_primary_then_alternate_ranking_is_deterministic_without_spectral_or_model_scores(self):
        args = self.fixture(); expected = choose_candidates(*args)
        self.assertEqual([c['event_id'] for c in expected['candidates']],
                         ['event-0', 'event-2', 'event-4', 'event-1', 'event-3', 'event-5'])
        args[0].reverse()
        for qa in args[2].values():
            qa['forest_model_dice'] = 1. if qa['pair_id'].endswith('1') else 0.
            qa['per_date']['pre']['target_median_ndvi'] = -100.
            qa['per_date']['post']['target_median_ndvi'] = 100.
        self.assertEqual(expected, choose_candidates(*args))

    def test_alternates_never_reuse_primary_or_other_selected_groups(self):
        args = self.fixture()
        for key in ('event-3',):
            args[3][key]['group_id'] = args[3]['event-2']['group_id']
        args[0][3]['group_id'] = args[0][2]['group_id']
        args[1]['pair-3']['group_id'] = args[1]['pair-2']['group_id']
        args[2]['pair-3']['group_id'] = args[2]['pair-2']['group_id']
        result = choose_candidates(*args)
        self.assertEqual(len(result['candidates']), 5)
        self.assertEqual(len({c['group_id'] for c in result['candidates']}), 5)
        self.assertEqual(result['missing_slots'], [{'role': 'alternate', 'stratum': STRATA[1],
                           'reason': 'no_eligible_unused_geographic_group'}])

    def test_full_grid_touching_train_test_barrier_or_actual_training_grid_is_rejected(self):
        for kind in ('conservative', 'actual'):
            args = self.fixture()
            if kind == 'conservative':
                args[3]['barrier'] = {'event_id': 'barrier', 'split': 'test',
                                      'group_footprint_bounds': [640., 0., 700., 640.]}
            else:
                args[5][0]['grid'] = grid(640.)
            result = choose_candidates(*args)
            self.assertNotIn('event-0', [c['event_id'] for c in result['candidates']])
            row = next(r for r in result['dispositions'] if r['event_id'] == 'event-0')
            reason = ('train_or_test_conservative_footprint_overlap' if kind == 'conservative'
                      else 'actual_training_feature_footprint_overlap')
            self.assertIn(reason, row['considered_pairs'][0]['reasons'])

    def test_quality_unknown_crs_and_missing_followup_remain_explicit_dispositions(self):
        args = self.fixture()
        args[2]['pair-0']['per_date']['pre']['crop']['unusable_pixels'] = 1
        args[1]['pair-2']['grid']['crs'] = 'EPSG:4326'
        args[4] = [s for s in args[4] if s['tile'] == '36UXA']
        result = choose_candidates(*args)
        reasons = {r['event_id']:r['considered_pairs'][0]['reasons'] for r in result['dispositions']}
        self.assertIn('endpoint_core_or_target_quality_flags', reasons['event-0'])
        self.assertIn('unknown_crs_or_changed_grid', reasons['event-2'])
        self.assertIn('no_later_local_scene_for_persistence_review', reasons['event-4'])
        self.assertEqual(len(result['dispositions']), 6)

    def test_cross_split_or_approved_quality_cannot_be_repurposed_as_validation(self):
        for field, invalid in (('split', 'train'), ('training_approved', True), ('config_id', 'different')):
            args = self.fixture(); args[2]['pair-0'][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                choose_candidates(*args)

    def test_metadata_selection_never_invents_interior_support_or_label_approval(self):
        result = choose_candidates(*self.fixture())
        self.assertTrue(result['rule']['geometry_does_not_guarantee_eroded_interior'])
        for candidate in result['candidates']:
            self.assertIsNone(candidate['search_aid_metrics']['eroded_anchor_interior_pixels'])
            self.assertFalse(candidate['validation_approved'])
            self.assertFalse(candidate['training_approved'])
        self.assertTrue(all(r['disposition'] == 'selected_for_review_not_approved'
                            for r in result['dispositions']))


if __name__ == '__main__':
    unittest.main()
