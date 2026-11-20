"""Remaining-date lineage and cumulative references without satellite inputs."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Solutions'))
from prepare_validation_review import validate_grid
from screen_earlier_validation import GROUP, POST_DATE, object_hash
from screen_remaining_validation import PARENTS, PRE_DATES, SCENES, context_ids, derive_pair, temporal_summary


class RemainingPairQATests(unittest.TestCase):
    def setUp(self):
        self.paths = {day: 'fixture/' + scene + '.SAFE' for day, scene in SCENES.items()}
        self.frozen_ids = set(PARENTS.values())

    def original(self, event_id):
        grid = {'crs': 'EPSG:32636', 'transform': [10, 0, 663640, 0, -10, 5605640],
                'width': 64, 'height': 64, 'bounds': [663640, 5605000, 664280, 5605640]}
        return {'pair_id': PARENTS[event_id], 'group_id': GROUP, 'split': 'validation',
                'tile': '36UXA', 'pre_date': '2018-07-26', 'post_date': POST_DATE,
                'pre_scene_id': 'original-pre', 'post_scene_id': 'original-post',
                'pre_path': 'original/pre', 'post_path': 'original/post',
                'interval_days': 5, 'grid': grid, 'anchor_event_ids': [event_id],
                'context_event_ids': [event_id], 'label_event_ids_by_pre_date': [],
                'label_event_ids_by_post_date': [event_id], 'conflicting_label_event_ids': [],
                'temporal_source_ids': ['36UXA_time-dependent'],
                'status': 'blocked_for_temporal_training',
                'blocking_reasons': ['crop_crosses_tile_edge', 'post_scene_does_not_cover_crop']}

    def derive(self, parent, day, ids=None):
        event_id = parent['anchor_event_ids'][0]
        retained = {key: deepcopy(parent[key]) for key in
                    ('pair_id', 'group_id', 'anchor_event_ids', 'status', 'blocking_reasons')}
        event = {'event_id': event_id, 'annotation_image_date': POST_DATE,
                 'crop_grid': deepcopy(parent['grid'])}
        if ids is None:
            ids = ([event_id], [], [event_id])
        return derive_pair(parent, retained, event, day, self.paths, self.frozen_ids, ids)

    def test_six_unique_date_event_pairs_preserve_frozen_parents(self):
        intervals = {'2018-07-11': 20, '2018-07-06': 25, '2018-07-01': 30}
        self.assertEqual(PRE_DATES, tuple(intervals))
        pair_ids = set()
        for day, interval in intervals.items():
            for event_id in PARENTS:
                with self.subTest(day=day, event=event_id):
                    parent = self.original(event_id)
                    original = deepcopy(parent)
                    pair = self.derive(parent, day)
                    self.assertEqual(pair, self.derive(parent, day))
                    self.assertEqual((pair['pre_date'], pair['post_date'], pair['interval_days']),
                                     (day, POST_DATE, interval))
                    self.assertEqual(pair['parent_pair_id'], original['pair_id'])
                    self.assertEqual(pair['parent_pair_sha256'], object_hash(original))
                    self.assertEqual((pair['original_tile'], pair['tile']), ('36UXA', '36UXB'))
                    for key in ('grid', 'group_id', 'split', 'anchor_event_ids', 'temporal_source_ids'):
                        self.assertEqual(pair[key], original[key])
                    self.assertEqual(pair['source_replacement'], {
                        side: {'scene_id': SCENES[date], 'path': self.paths[date]}
                        for side, date in (('pre', day), ('post', POST_DATE))})
                    self.assertFalse(pair['training_approved'] or pair['evaluation_approved'])
                    pair_ids.add(pair['pair_id'])
                    pair['grid']['bounds'][0] = 0
                    self.assertEqual(parent, original)
        self.assertEqual(len(pair_ids), 6)
        self.assertFalse(pair_ids & self.frozen_ids)

    def test_thirty_day_boundary_is_inclusive_and_roster_is_fixed(self):
        parent = self.original(next(iter(PARENTS)))
        pair = self.derive(parent, '2018-07-01')
        self.assertEqual(pair['interval_days'], 30)
        validate_grid(pair)
        with self.assertRaises(ValueError):
            validate_grid({**pair, 'pre_date': '2018-06-30', 'interval_days': 31})
        for day in ('2018-06-30', '2018-07-02', '2018-07-16', '2018-07-26', POST_DATE):
            with self.subTest(day=day), self.assertRaises(ValueError):
                self.derive(parent, day)

    def test_july_six_events_enter_earlier_change_mask_with_cumulative_subtraction(self):
        from shapely.geometry import box

        event_id = next(iter(PARENTS))
        parent = self.original(event_id)
        events = {}
        for key, day, left, right in (
                ('june-loss', '2018-06-21', 663740, 663840),
                ('july-six-loss', '2018-07-06', 663790, 663890),
                (event_id, POST_DATE, 663830, 663930)):
            events[key] = {'geometry': box(left, 5605100, right, 5605200), 'properties': {
                'observation_date': day, 'temporal_source_ids': ['36UXA_time-dependent'],
                'group_id': GROUP, 'split': 'validation', 'flags': []}}
        properties = {key: record['properties'] for key, record in events.items()}
        parent.update(context_event_ids=sorted(events), label_event_ids_by_post_date=sorted(events),
                      label_event_ids_by_pre_date=['july-six-loss', 'june-loss'])

        summaries = {}
        for day in PRE_DATES:
            ids = context_ids(parent, events, properties, day)
            summaries[day] = temporal_summary(self.derive(parent, day, ids), parent, events)
        july11, july6, july1 = (summaries[day] for day in PRE_DATES)
        self.assertEqual(july6['pre_event_ids'], ['july-six-loss', 'june-loss'])
        self.assertEqual(july1['pre_event_ids'], ['june-loss'])
        self.assertEqual(july11['mask_sha256'], july6['mask_sha256'])
        self.assertEqual(july6['pixels']['cumulative_pre'], 150)
        self.assertEqual(july1['pixels']['cumulative_pre'], 100)
        self.assertEqual(july1['pixels']['cumulative_post'], 190)
        self.assertEqual(july6['pixels']['reference_new_loss'], 40)
        self.assertEqual(july1['pixels']['reference_new_loss'], 90)
        self.assertEqual(july1['interval_only_excess_pixels'], 50)
        self.assertEqual(july6['changed_reference_pixels_from_parent'], 0)
        self.assertEqual(july1['changed_reference_pixels_from_parent'], 50)


if __name__ == '__main__':
    unittest.main()
