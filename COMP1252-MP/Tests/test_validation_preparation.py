"""Metadata-only isolation checks; no satellite arrays or model execution."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from prepare_validation_review import validate_grid, validate_validation_pair, validate_quality_summary


def grid(west=500000., north=5600000., crs='EPSG:32636'):
    return {'crs': crs, 'width': 64, 'height': 64,
            'transform': [10., 0., west, 0., -10., north],
            'bounds': [west, north-640, west+640, north]}


def fixture():
    pair = {'pair_id': 'pair-validation', 'group_id': 'group-validation',
            'split': 'validation', 'pre_scene_id': 'scene-before',
            'post_scene_id': 'scene-after', 'pre_date': '2019-06-11',
            'post_date': '2019-06-26', 'anchor_event_ids': ['event-validation'],
            'status': 'candidate_for_pixel_review', 'blocking_reasons': [],
            'grid': grid()}
    catalogue = {'features': [
        {'id': 'event-validation', 'properties': {'split': 'validation',
          'group_id': 'group-validation', 'group_footprint_bounds': [499000, 5598000, 501000, 5601000]}},
        {'id': 'event-train', 'properties': {'split': 'train',
          'group_id': 'group-other-train', 'group_footprint_bounds': [600000, 5598000, 601000, 5601000]}},
        {'id': 'event-test', 'properties': {'split': 'test',
          'group_id': 'group-other-test', 'group_footprint_bounds': [610000, 5598000, 611000, 5601000]}}
    ]}
    splits = {'crs': 'EPSG:32636', 'groups': [
        {'group_id': 'group-validation', 'split': 'validation', 'event_ids': ['event-validation']},
        {'group_id': 'group-other-train', 'split': 'train', 'event_ids': ['event-train']},
        {'group_id': 'group-other-test', 'split': 'test', 'event_ids': ['event-test']}
    ]}
    training = [{'split': 'train', 'group_id': f'group-training-{index}',
                 'grid': grid(west=700000.+index*1000)} for index in range(13)]
    return pair, catalogue, splits, training


class ValidationPreparationTests(unittest.TestCase):
    def test_frozen_quality_error_list_contract_is_preserved(self):
        summary = {'checks': {'complete': True}, 'processing_errors': []}
        validate_quality_summary(summary)
        for errors in (0, None, {}, ['failed-pair']):
            with self.subTest(errors=errors), self.assertRaises(ValueError):
                validate_quality_summary({**summary, 'processing_errors': errors})
        with self.assertRaises(ValueError):
            validate_quality_summary({**summary, 'checks': {'complete': False}})

    def test_separate_validation_core_and_exact_thirteen_training_crops_pass_unchanged(self):
        pair, catalogue, splits, training = fixture()
        originals = deepcopy((pair, catalogue, splits, training))
        validated = validate_grid(pair)
        self.assertEqual(tuple(validated['transform'])[:6], tuple(pair['grid']['transform']))
        self.assertIsNone(validate_validation_pair(pair, 'event-validation', catalogue, splits, training))
        self.assertEqual((pair, catalogue, splits, training), originals)

    def test_grid_rejects_role_shape_resolution_rotation_bounds_and_nonfinite_origin(self):
        for change in ('train', 'test', 'shape', 'resolution', 'rotation', 'bounds', 'nonfinite', 'scene'):
            pair, _, _, _ = fixture()
            if change in ('train', 'test'):
                pair['split'] = change
            elif change == 'shape':
                pair['grid']['height'] = 63
            elif change == 'resolution':
                pair['grid']['transform'][0] = 20.
            elif change == 'rotation':
                pair['grid']['transform'][1] = 1.
            elif change == 'bounds':
                pair['grid']['bounds'][2] += 10.
            elif change == 'nonfinite':
                pair['grid']['transform'][2] = float('nan')
            else:
                pair['pre_scene_id'] = ''
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_grid(pair)

    def test_forward_five_to_thirty_day_intervals_are_required(self):
        for after in ('2019-06-10', '2019-06-11', '2019-06-15', '2019-07-12'):
            pair, _, _, _ = fixture()
            pair['post_date'] = after
            with self.subTest(after=after), self.assertRaises(ValueError):
                validate_grid(pair)
        for after in ('2019-06-16', '2019-07-11'):
            pair, _, _, _ = fixture()
            pair['post_date'] = after
            validate_grid(pair)

    def test_anchor_must_be_unblocked_and_bound_to_exact_validation_group_and_split(self):
        for change in ('blocked', 'status', 'anchor', 'missing_catalogue',
                       'catalogue_split', 'catalogue_group', 'split_plan_role', 'split_plan_membership'):
            pair, catalogue, splits, training = fixture()
            if change == 'blocked':
                pair['blocking_reasons'] = ['crop_crosses_tile_edge']
            elif change == 'status':
                pair['status'] = 'blocked_for_temporal_training'
            elif change == 'anchor':
                pair['anchor_event_ids'] = ['other-event']
            elif change == 'missing_catalogue':
                catalogue['features'].pop(0)
            elif change == 'catalogue_split':
                catalogue['features'][0]['properties']['split'] = 'train'
            elif change == 'catalogue_group':
                catalogue['features'][0]['properties']['group_id'] = 'other-group'
            elif change == 'split_plan_role':
                splits['groups'][0]['split'] = 'test'
            else:
                splits['groups'][0]['event_ids'] = ['other-event']
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_validation_pair(pair, 'event-validation', catalogue, splits, training)

    def test_full_core_corner_overlap_with_any_train_or_test_conservative_footprint_fails(self):
        for role in ('train', 'test'):
            pair, catalogue, splits, training = fixture()
            row = next(row for row in catalogue['features'] if row['properties']['split'] == role)
            # Only a small corner overlaps: checking centroids or annotation
            # centres would miss this conservative full-feature-footprint leak.
            row['properties']['group_footprint_bounds'] = [500620, 5599320, 500900, 5599380]
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'conservative feature footprint'):
                validate_validation_pair(pair, 'event-validation', catalogue, splits, training)

    def test_actual_training_crop_overlap_and_reused_group_fail_even_when_catalogue_bounds_do_not(self):
        for change in ('overlap', 'group'):
            pair, catalogue, splits, training = fixture()
            if change == 'overlap':
                training[5]['grid'] = grid(west=500630.)
            else:
                training[5]['group_id'] = pair['group_id']
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'actual reviewed training crop/group'):
                validate_validation_pair(pair, 'event-validation', catalogue, splits, training)

    def test_isolation_requires_all_thirteen_training_rows_with_train_role(self):
        for change in ('missing', 'extra', 'wrong_split'):
            pair, catalogue, splits, training = fixture()
            if change == 'missing':
                training.pop()
            elif change == 'extra':
                training.append(deepcopy(training[-1]))
            else:
                training[-1]['split'] = 'validation'
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'thirteen training crops'):
                validate_validation_pair(pair, 'event-validation', catalogue, splits, training)

    def test_actual_training_footprint_is_reprojected_before_overlap_comparison(self):
        # Imports execute only when the parent runs this suite inside its guard.
        # No imagery or raster pixels are read by these geometry operations.
        from pyproj import Transformer

        pair, catalogue, splits, training = fixture()
        east, north = Transformer.from_crs('EPSG:32636', 'EPSG:32637', always_xy=True).transform(
            500320., 5599680.)
        training[0]['grid'] = grid(west=east-320, north=north+320, crs='EPSG:32637')
        with self.assertRaisesRegex(ValueError, 'actual reviewed training crop/group'):
            validate_validation_pair(pair, 'event-validation', catalogue, splits, training)


if __name__ == '__main__':
    unittest.main()
