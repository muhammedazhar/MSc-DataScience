"""Earlier-date lineage, cumulative targets and bounded cache checks; no satellite I/O."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from screen_earlier_validation import (GROUP, PARENTS, POLICY, POST_DATE, PRE_DATES,
    SCENES, context_ids, derive_pair, digest, object_hash, prevalidate_cache,
    require_lineage, temporal_summary, collect_roster)


class EarlierPairQATests(unittest.TestCase):
    def setUp(self):
        from shapely.geometry import box
        self.event_id = next(iter(PARENTS))
        self.grid = {'crs': 'EPSG:32636', 'transform': [10., 0., 663640., 0., -10., 5605640.],
                     'width': 64, 'height': 64, 'bounds': [663640., 5605000., 664280., 5605640.]}
        self.parent = {'pair_id': PARENTS[self.event_id], 'group_id': GROUP, 'split': 'validation',
                       'tile': '36UXA', 'pre_date': '2018-07-26', 'post_date': POST_DATE,
                       'interval_days': 5, 'pre_scene_id': 'original-pre', 'post_scene_id': 'original-post',
                       'pre_path': 'original/pre', 'post_path': 'original/post', 'grid': deepcopy(self.grid),
                       'anchor_event_ids': [self.event_id], 'context_event_ids': [self.event_id],
                       'label_event_ids_by_post_date': [self.event_id], 'conflicting_label_event_ids': [],
                       'temporal_source_ids': ['36UXA_time-dependent'],
                       'status': 'blocked_for_temporal_training',
                       'blocking_reasons': ['crop_crosses_tile_edge', 'post_scene_does_not_cover_crop']}
        self.retained = {key: deepcopy(self.parent[key]) for key in
                         ('pair_id', 'group_id', 'anchor_event_ids', 'status', 'blocking_reasons')}
        self.event = {'event_id': self.event_id, 'annotation_image_date': POST_DATE,
                      'crop_grid': deepcopy(self.grid)}
        self.paths = {day: 'fake/'+scene+'.SAFE' for day, scene in SCENES.items()}
        self.frozen_ids = set(PARENTS.values())
        self.events = {}
        for key, day, bounds, temporal in (
                ('old', '2018-07-11', (663740, 5605100, 663840, 5605200), True),
                ('middle', '2018-07-20', (663790, 5605100, 663890, 5605200), True),
                (self.event_id, POST_DATE, (663790, 5605100, 663940, 5605200), True),
                ('baseline', '2018-07-01', (664000, 5605300, 664010, 5605310), False),
                ('future', '2018-08-05', (664050, 5605350, 664060, 5605360), True),
                ('outside', '2018-07-20', (665000, 5606000, 665010, 5606010), True)):
            self.events[key] = {'geometry': box(*bounds), 'properties': {
                'observation_date': day, 'temporal_source_ids': ['36UXA_time-dependent'] if temporal else [],
                'group_id': GROUP, 'split': 'validation', 'flags': []}}

    def derive(self, day=PRE_DATES[0], parent=None, retained=None, event=None, frozen_ids=None, ids=None):
        parent = self.parent if parent is None else parent
        return derive_pair(parent, self.retained if retained is None else retained,
                           self.event if event is None else event, day, self.paths,
                           self.frozen_ids if frozen_ids is None else frozen_ids,
                           ([self.event_id], [], [self.event_id]) if ids is None else ids)

    def test_processing_error_retains_all_four_fixed_outcomes_without_success_claim(self):
        pairs = [{**self.derive(), 'pair_id': str(index)} for index in range(4)]
        seen = []
        def operation(pair):
            seen.append(pair['pair_id'])
            if pair['pair_id'] == '0':
                return ({'pair_id': '0', 'quality_status': 'processing_error',
                         'quality_reasons': ['explicit read failure'], 'qa_mask_file': None}, None, [None, None])
            if pair['pair_id'] == '2':
                raise OSError('another fixed source read failed')
            return ({'pair_id': pair['pair_id'], 'quality_status': 'needs_visual_review',
                     'qa_mask_file': 'qa.tif', 'per_date': {}}, {'target': 'retained'},
                    [{'cache_key': 'pre'}, {'cache_key': 'post'}])
        rows, contexts, receipts, errors = collect_roster(pairs, operation)
        self.assertEqual(seen, ['0', '1', '2', '3'])
        self.assertEqual(len(rows), 4); self.assertEqual(len(receipts), 8)
        self.assertEqual([r['pair_id'] for r in rows], seen)
        self.assertEqual([r['pair_id'] for r in errors], ['0', '2'])
        self.assertEqual(rows[1]['quality_status'], 'needs_visual_review')
        self.assertEqual(rows[3]['quality_status'], 'needs_visual_review')
        self.assertIsNone(receipts[0]); self.assertIsNone(receipts[4])
        self.assertEqual(contexts[0]['status'], 'unavailable_due_to_processing_error')

    def catalogue_parent(self):
        parent = deepcopy(self.parent)
        parent['context_event_ids'] = sorted(set(self.events)-{'outside'})
        parent['label_event_ids_by_post_date'] = sorted([self.event_id, 'old', 'middle'])
        return parent

    def test_four_date_event_identities_preserve_original_annotation_lineage(self):
        pairs = []
        for day in PRE_DATES:
            for event_id, parent_id in PARENTS.items():
                parent = deepcopy(self.parent); parent.update(pair_id=parent_id, anchor_event_ids=[event_id],
                    context_event_ids=[event_id], label_event_ids_by_post_date=[event_id])
                retained = {key: deepcopy(parent[key]) for key in self.retained}
                event = {**self.event, 'event_id': event_id}
                original = deepcopy(parent)
                pair = self.derive(day, parent, retained, event, ids=([event_id], [], [event_id]))
                self.assertEqual(pair, self.derive(day, parent, retained, event, ids=([event_id], [], [event_id])))
                self.assertEqual((pair['pre_date'], pair['post_date'], pair['interval_days']),
                                 (day, POST_DATE, 10 if day == '2018-07-21' else 15))
                self.assertEqual(pair['parent_pair_id'], parent_id)
                self.assertEqual(pair['parent_pair_sha256'], object_hash(original))
                self.assertEqual(pair['parent_pre_date'], '2018-07-26')
                self.assertEqual((pair['original_tile'], pair['tile']), ('36UXA', '36UXB'))
                for key in ('grid', 'group_id', 'split', 'anchor_event_ids', 'context_event_ids',
                            'label_event_ids_by_post_date', 'temporal_source_ids'):
                    self.assertEqual(pair[key], original[key])
                self.assertEqual(pair['source_replacement']['pre'],
                                 {'scene_id': SCENES[day], 'path': self.paths[day]})
                self.assertFalse(pair['training_approved'] or pair['evaluation_approved'])
                pair['grid']['bounds'][0] = 0
                self.assertEqual(parent, original)
                pairs.append(pair)
        self.assertEqual(len({pair['pair_id'] for pair in pairs}), 4)
        self.assertFalse({pair['pair_id'] for pair in pairs} & self.frozen_ids)

    def test_fixed_dates_parent_identity_and_only_coverage_blockers_are_required(self):
        for day in ('2018-07-26', '2018-07-11', POST_DATE):
            with self.subTest(day=day), self.assertRaisesRegex(ValueError, 'Exact fixed'):
                self.derive(day)
        for key, value in (('pre_date', '2018-07-21'), ('post_date', '2018-08-05'),
                           ('interval_days', 10), ('split', 'train'), ('group_id', 'other'),
                           ('tile', '36UXB'), ('blocking_reasons', [*self.parent['blocking_reasons'],
                                                                    'baseline_only_provenance'])):
            with self.subTest(key=key):
                parent = deepcopy(self.parent); parent[key] = value
                retained = {field: deepcopy(parent[field]) for field in self.retained}
                with self.assertRaises(ValueError):
                    self.derive(parent=parent, retained=retained)

    def test_retained_join_original_grid_and_derived_id_collision_are_rejected(self):
        retained = deepcopy(self.retained); retained['status'] = 'candidate_for_pixel_review'
        with self.assertRaisesRegex(ValueError, 'V5 retained'):
            self.derive(retained=retained)
        parent = deepcopy(self.parent); parent['grid']['bounds'][0] += 10
        with self.assertRaisesRegex(ValueError, 'grid/anchor/date'):
            self.derive(parent=parent)
        event = deepcopy(self.event); event['annotation_image_date'] = '2018-07-26'
        with self.assertRaisesRegex(ValueError, 'grid/anchor/date'):
            self.derive(event=event)
        pair = self.derive()
        with self.assertRaisesRegex(ValueError, 'impersonate'):
            self.derive(frozen_ids=self.frozen_ids | {pair['pair_id']})

    def test_context_recomputes_dates_and_entire_original_crop(self):
        parent = self.catalogue_parent()
        properties = {key: record['properties'] for key, record in self.events.items()}
        properties['outside']['group_id'] = 'outside-group'
        context21, before21, after21 = context_ids(parent, self.events, properties, '2018-07-21')
        context16, before16, after16 = context_ids(parent, self.events, properties, '2018-07-16')
        self.assertEqual(context21, parent['context_event_ids'])
        self.assertEqual(context16, context21)
        self.assertEqual(before21, ['middle', 'old'])
        self.assertEqual(before16, ['old'])
        self.assertEqual(after21, parent['label_event_ids_by_post_date'])
        self.assertEqual(after16, after21)
        self.assertIn('future', context21)
        self.assertIn('baseline', context21)
        self.assertNotIn('future', after21)
        self.assertNotIn('baseline', after21)

    def test_expanded_interval_baseline_and_context_provenance_changes_are_rejected(self):
        parent = self.catalogue_parent()
        properties = {key: deepcopy(record['properties']) for key, record in self.events.items()}
        properties['baseline']['observation_date'] = '2018-07-18'
        context_ids(parent, self.events, properties, '2018-07-21')
        with self.assertRaisesRegex(ValueError, 'baseline-only'):
            context_ids(parent, self.events, properties, '2018-07-16')
        properties['baseline']['observation_date'] = '2018-07-01'
        for field, value in (('group_id', 'other-group'), ('split', 'train'),
                             ('flags', ['conflicting_observation_dates'])):
            with self.subTest(field=field):
                bad = deepcopy(properties); bad['baseline'][field] = value
                with self.assertRaisesRegex(ValueError, 'Conflicting context'):
                    context_ids(parent, self.events, bad, '2018-07-16')
        for field in ('context_event_ids', 'label_event_ids_by_post_date'):
            with self.subTest(field=field):
                bad = deepcopy(parent); bad[field] = bad[field][1:]
                with self.assertRaises(ValueError):
                    context_ids(bad, self.events, properties, '2018-07-16')

    def test_cumulative_overlap_differs_from_interval_union_and_between_pre_dates(self):
        parent = self.catalogue_parent()
        properties = {key: record['properties'] for key, record in self.events.items()}
        summaries = []
        for day in PRE_DATES:
            ids = context_ids(parent, self.events, properties, day)
            pair = self.derive(day, parent=parent, ids=ids)
            summaries.append(temporal_summary(pair, parent, self.events))
        july21, july16 = summaries
        self.assertTrue(all(result['independent_boolean_difference_checked'] for result in summaries))
        self.assertEqual(july21['pixels']['cumulative_pre'], 150)
        self.assertEqual(july16['pixels']['cumulative_pre'], 100)
        self.assertEqual(july21['pixels']['cumulative_post'], 200)
        self.assertEqual(july21['pixels']['reference_new_loss'], 50)
        self.assertEqual(july16['pixels']['reference_new_loss'], 100)
        self.assertEqual(july21['interval_only_excess_pixels'], 100)
        self.assertEqual(july16['interval_only_excess_pixels'], 50)
        self.assertNotEqual(july21['mask_sha256']['reference_new_loss'],
                            july16['mask_sha256']['reference_new_loss'])
        self.assertEqual(july21['changed_reference_pixels_from_parent'], 0)
        self.assertEqual(july16['changed_reference_pixels_from_parent'], 50)

    def test_cache_archive_and_metadata_size_are_bounded_before_numpy_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cache.npz'
            with path.open('wb') as stream:
                stream.truncate(2*2**20+1)
            with patch('numpy.load') as load, patch('verify_validation_dataset.read_cache') as read_cache:
                with self.assertRaisesRegex(ValueError, 'Oversized QA cache'):
                    prevalidate_cache(path, 'source', self.grid)
                load.assert_not_called(); read_cache.assert_not_called()
            with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('metadata.npy', b'x'*(2**20+1))
            with patch('numpy.load') as load, patch('verify_validation_dataset.read_cache') as read_cache:
                with self.assertRaisesRegex(ValueError, 'Oversized cache metadata'):
                    prevalidate_cache(path, 'source', self.grid)
                load.assert_not_called(); read_cache.assert_not_called()

    def test_unsafe_scalar_dtype_shape_and_payload_headers_precede_allocation(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cache.npz'
            for shape, dtype, fortran in (((), 'O', False), ((1,), '<U2', False),
                                           ((), '<U300000', False), ((), '<f8', False),
                                           ((), '<U2', True), ((), '<U2', False)):
                with self.subTest(shape=shape, dtype=dtype, fortran=fortran):
                    stream = io.BytesIO()
                    np.lib.format.write_array_header_1_0(stream, {'descr': np.dtype(dtype).str,
                        'fortran_order': fortran, 'shape': shape})
                    with zipfile.ZipFile(path, 'w') as archive:
                        archive.writestr('metadata.npy', stream.getvalue())
                    with patch('numpy.load') as load, patch('verify_validation_dataset.read_cache') as read_cache:
                        with self.assertRaisesRegex(ValueError, 'Unsafe cache scalar allocation'):
                            prevalidate_cache(path, 'source', self.grid)
                        load.assert_not_called(); read_cache.assert_not_called()

    def test_cache_scalar_is_bound_to_source_grid_path_before_complete_verifier(self):
        import numpy as np
        expected = {'source_id': 'source', 'grid': self.grid, 'cache_path': 'cache.npz'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cache.npz'
            for key, value in ((None, None), ('source_id', 'other'), ('grid', {}), ('cache_path', 'other.npz')):
                with self.subTest(key=key):
                    metadata = deepcopy(expected)
                    if key is not None:
                        metadata[key] = value
                    stream = io.BytesIO(); np.save(stream, np.array(json.dumps(metadata)), allow_pickle=False)
                    with zipfile.ZipFile(path, 'w') as archive:
                        archive.writestr('metadata.npy', stream.getvalue())
                    with patch('screen_earlier_validation.relative', return_value='cache.npz'), \
                         patch('verify_validation_dataset.read_cache') as read_cache:
                        if key is None:
                            prevalidate_cache(path, 'source', self.grid)
                            read_cache.assert_called_once_with(expected, POLICY)
                        else:
                            with self.assertRaisesRegex(ValueError, 'source/grid binding'):
                                prevalidate_cache(path, 'source', self.grid)
                            read_cache.assert_not_called()

    def test_completed_lineage_rejects_changed_geometry_targets_and_artifacts(self):
        inputs = {'pairs': [self.derive()], 'geometry': [{'pixels': {'reference_new_loss': 50}}],
                  'assets': {'fake/source.jp2': {'sha256': 'frozen'}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run = root/'lineage'; run.mkdir()
            protocol = root/'protocol.json'; protocol.write_text('{}')
            result = {'protocol_sha256': digest(protocol), 'derived_pairs': inputs['pairs'],
                      'temporal_geometry': inputs['geometry'], 'source_asset_receipts': inputs['assets']}
            (run/'lineage.json').write_text(json.dumps(result))
            (run/'report.md').write_text('verified geometry')
            meta = {'execution_status': 'completed', 'mode': 'prepare', 'artifact_sha256': {
                name: digest(run/name) for name in ('lineage.json', 'report.md')}}
            (run/'metadata.json').write_text(json.dumps(meta))
            args = SimpleNamespace(lineage=run, protocol=protocol)
            with patch('screen_earlier_validation.ROOT', root):
                require_lineage(args, inputs)
                changed = deepcopy(inputs); changed['geometry'][0]['pixels']['reference_new_loss'] += 1
                with self.assertRaisesRegex(ValueError, 'exact sources/targets'):
                    require_lineage(args, changed)
                changed = deepcopy(inputs); changed['pairs'][0]['label_event_ids_by_pre_date'] = [self.event_id]
                with self.assertRaisesRegex(ValueError, 'exact sources/targets'):
                    require_lineage(args, changed)
                (run/'report.md').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    require_lineage(args, inputs)


if __name__ == '__main__':
    unittest.main()
