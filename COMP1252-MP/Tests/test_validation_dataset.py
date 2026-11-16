"""Meaningful loader invariants without a mirrored full-publication fixture."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from validation_review import select_stable_pair
from verify_validation_dataset import (canonical, expected_derived_pair, validate_arrays,
                                       validate_cache_binding, validate_record_role)


def record(kind='loss'):
    grid = {'crs': 'EPSG:32636', 'width': 64, 'height': 64,
            'transform': [10, 0, 500000, 0, -10, 5600000],
            'bounds': [500000, 5599360, 500640, 5600000]}
    pair = {'pair_id': 'pair-parent', 'group_id': 'group-validation', 'split': 'validation',
            'grid': grid, 'pre_date': '2019-06-01', 'post_date': '2019-06-06'}
    if kind == 'stable_roi':
        pair.update(pair_id='validation-stable-example', kind='derived_pre_event_stable_roi',
                    frozen_parent_pair_id='pair-parent', not_an_original_frozen_candidate=True)
    positives = 4 if kind == 'loss' else 0
    return {'sample_id': ('loss-' if kind == 'loss' else 'stable-')+'event-example',
            'kind': kind, 'event_id': 'event-example', 'group_id': 'group-validation',
            'split': 'validation', 'training_approved': False, 'evaluation_approved': True,
            'frozen_parent_pair_id': 'pair-parent', 'pair': pair, 'grid': grid,
            'pre_date': pair['pre_date'], 'post_date': pair['post_date'],
            'supervised_positive_pixels': positives, 'supervised_stable_pixels': 64,
            'ignored_pixels': 4096-64-positives}


def arrays():
    def frame(values):
        a = np.asarray(values, dtype='float32')
        a = np.concatenate((a, [(a[3]-a[2])/(a[3]+a[2]), (a[4]-a[5])/(a[4]+a[5])])).astype('float32')
        return np.broadcast_to(a[:, None, None], (9, 64, 64)).copy()
    pre = frame([.1, .12, .2, .6, .5, .3, .25])
    post = frame([.1, .1, .3, .4, .4, .35, .2])
    features = np.concatenate((pre, post, post-pre))
    labels = np.full((64, 64), 255, dtype='uint8')
    labels[8:16, 8:16] = 0; labels[30:32, 30:32] = 1
    features[:, 0, 0] = np.nan
    return features, labels, labels != 255, np.isfinite(features).all(axis=0)


class ValidationDatasetTests(unittest.TestCase):
    def test_explicit_loss_and_derived_stable_roi_roles_have_distinct_class_contracts(self):
        validate_record_role(record('loss'))
        validate_record_role(record('stable_roi'))

    def test_unsafe_roles_zero_positive_loss_and_missing_derived_lineage_fail(self):
        for change in ('train', 'test', 'training_approved', 'evaluation_approved', 'empty_loss',
                       'unknown_kind', 'missing_lineage', 'count_type', 'count_total'):
            row = record('stable_roi' if change == 'missing_lineage' else 'loss')
            if change in ('train', 'test'):
                row['split'] = change
            elif change == 'training_approved':
                row['training_approved'] = True
            elif change == 'evaluation_approved':
                row['evaluation_approved'] = False
            elif change == 'empty_loss':
                row['supervised_positive_pixels'] = 0; row['ignored_pixels'] += 4
            elif change == 'unknown_kind':
                row['kind'] = 'unreviewed_background'
            elif change == 'missing_lineage':
                row['pair']['not_an_original_frozen_candidate'] = False
            elif change == 'count_type':
                row['supervised_positive_pixels'] = True
            else:
                row['ignored_pixels'] -= 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_record_role(row)

    def test_sanitization_changes_only_returned_nonfinite_unknown_context(self):
        features, labels, valid, feature_valid = arrays()
        original = features.copy()
        result, replacements = validate_arrays(features, labels, valid, feature_valid)
        self.assertEqual(replacements, 27)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.array_equal(result[:, valid], original[:, valid]))
        self.assertTrue(np.array_equal(features, original, equal_nan=True))
        self.assertTrue(np.array_equal(result[:, 0, 0], np.zeros(27, dtype='float32')))
        self.assertFalse(np.shares_memory(result, features))
        self.assertTrue((result[18:, valid] < 0).any())

    def test_unknown_labels_validity_and_nonfinite_reviewed_features_cannot_be_hidden(self):
        for change in ('label_value', 'validity', 'nonfinite_reviewed', 'feature_validity', 'infinite_context'):
            features, labels, valid, feature_valid = arrays()
            if change == 'label_value':
                labels[0, 1] = 2
            elif change == 'validity':
                valid[0, 1] = True
            elif change == 'nonfinite_reviewed':
                features[:, 8, 8] = np.nan; feature_valid[8, 8] = False
            elif change == 'feature_validity':
                feature_valid[0, 0] = True
            else:
                features[0, 0, 1] = np.inf
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_arrays(features, labels, valid, feature_valid)

    def test_signed_differences_and_saved_index_scales_are_checked_independently(self):
        for change in ('difference', 'index'):
            features, labels, valid, feature_valid = arrays()
            if change == 'difference':
                features[18, 8, 8] += .1
            else:
                features[7, 8, 8] = .99
                features[25, 8, 8] = features[16, 8, 8]-features[7, 8, 8]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_arrays(features, labels, valid, feature_valid)

    def test_cache_binding_requires_actual_source_grid_policy_and_self_identity(self):
        metadata = {'scene_id': 'scene-example', 'source_id': 'source-example',
                    'grid': record()['grid'], 'config_id': 'policy-example'}
        metadata['cache_key'] = canonical({'source': metadata['source_id'], 'grid': metadata['grid'],
                                           'config_id': metadata['config_id']})
        metadata['cache_path'] = 'Datasets/PixelQA/cache/'+metadata['cache_key']+'.npz'
        validate_cache_binding(metadata, deepcopy(metadata), 'policy-example')
        for change in ('actual_source', 'policy', 'forged_key', 'forged_path'):
            actual, expected = deepcopy(metadata), deepcopy(metadata)
            if change == 'actual_source':
                actual['source_id'] = 'changed-source'
            elif change == 'policy':
                actual['config_id'] = expected['config_id'] = 'other-policy'
            elif change == 'forged_key':
                actual['cache_key'] = expected['cache_key'] = 'a'*64
            else:
                actual['cache_path'] = expected['cache_path'] = 'Datasets/PixelQA/cache/other.npz'
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_cache_binding(actual, expected, 'policy-example')

    def test_derived_lineage_rebuilds_earliest_pair_and_clears_original_loss_claims(self):
        parent = record()['pair']
        parent.update(pre_date='2019-06-11', post_date='2019-06-16',
                      label_event_ids_by_post_date=['event-example'], status='candidate_for_pixel_review')
        observations = [{'scene_id': f'scene-{i}', 'date': day,
                         'target_state': 'forest', 'control_state': 'stable_forest'}
                        for i, day in enumerate(('2019-06-01', '2019-06-06', '2019-06-11'))]
        case = {'event_id': 'event-example', 'pair': parent,
                'timeline': [{**row, 'scene_path': 'Datasets/Sentinel-2/'+row['scene_id']+'.SAFE'} for row in observations]}
        decision = {'observations': observations,
                    'stable_pair': select_stable_pair({r['scene_id']: r for r in observations}, parent)}
        original = deepcopy(parent)
        pair = expected_derived_pair(case, decision)
        self.assertEqual((pair['pre_scene_id'], pair['post_scene_id']), ('scene-0', 'scene-1'))
        self.assertEqual(pair['group_id'], parent['group_id'])
        self.assertEqual(pair['grid'], parent['grid'])
        self.assertEqual(pair['label_event_ids_by_post_date'], [])
        self.assertIs(pair['not_an_original_frozen_candidate'], True)
        self.assertIs(pair['training_approved'], False)
        self.assertEqual(pair['frozen_parent_pair_id'], parent['pair_id'])
        self.assertEqual(parent, original)
        decision['stable_pair']['post_scene_id'] = 'scene-2'
        with self.assertRaisesRegex(ValueError, 'earliest eligible'):
            expected_derived_pair(case, decision)


if __name__ == '__main__':
    unittest.main()
