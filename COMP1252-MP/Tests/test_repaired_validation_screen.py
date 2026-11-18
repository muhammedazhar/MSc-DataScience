"""Derived source lineage and failure publication checks; no satellite I/O."""
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from screen_repaired_validation import (DATES, GROUP, PARENTS, SCENES,
    count_band_requests, derive_pair, digest, load_safe_cache, require_processed,
    validate_lineage, validate_scene)


class RepairedValidationScreenTests(unittest.TestCase):
    def setUp(self):
        self.event_id = next(iter(PARENTS))
        self.grid = {'crs': 'EPSG:32636', 'transform': [10., 0., 663640., 0., -10., 5605640.],
                     'width': 64, 'height': 64, 'bounds': [663640., 5605000., 664280., 5605640.]}
        self.parent = {'pair_id': PARENTS[self.event_id], 'group_id': GROUP, 'split': 'validation',
                       'tile': '36UXA', 'pre_date': DATES[0], 'post_date': DATES[1],
                       'pre_scene_id': 'original-pre', 'post_scene_id': 'original-post',
                       'pre_path': 'original/pre', 'post_path': 'original/post',
                       'grid': deepcopy(self.grid), 'anchor_event_ids': [self.event_id],
                       'label_event_ids_by_post_date': [self.event_id, 'other-temporal-event'],
                       'context_event_ids': [self.event_id], 'conflicting_label_event_ids': [],
                       'status': 'blocked_for_temporal_training',
                       'blocking_reasons': ['crop_crosses_tile_edge', 'post_scene_does_not_cover_crop']}
        self.retained = {k: deepcopy(self.parent[k]) for k in
                         ('pair_id', 'group_id', 'anchor_event_ids', 'status', 'blocking_reasons')}
        self.event = {'event_id': self.event_id, 'annotation_image_date': DATES[1],
                      'crop_grid': deepcopy(self.grid),
                      'required_qa_context_bounds': [662440., 5603800., 665480., 5606840.]}
        self.paths = {day: 'Datasets/Sentinel-2/'+identifier+'/'+identifier+'.SAFE'
                      for day, identifier in SCENES.items()}
        self.frozen_ids = set(PARENTS.values())

    def derive(self):
        return derive_pair(self.parent, self.retained, self.event, self.paths, self.frozen_ids)

    def test_source_pair_has_distinct_deterministic_identity_and_preserves_parent(self):
        original = deepcopy(self.parent)
        pair = self.derive()
        self.assertEqual(pair, self.derive())
        self.assertNotIn(pair['pair_id'], self.frozen_ids)
        self.assertEqual(pair['parent_pair_id'], original['pair_id'])
        self.assertEqual(pair['pair_kind'], 'derived_validation_source_pair')
        self.assertEqual(pair['tile'], '36UXB')
        for key in ('grid', 'group_id', 'split', 'anchor_event_ids', 'label_event_ids_by_post_date', 'context_event_ids'):
            self.assertEqual(pair[key], original[key])
        self.assertEqual(self.parent, original)
        pair['grid']['bounds'][0] = 0
        self.assertEqual(self.parent, original)

    def test_parent_with_noncoverage_block_is_rejected(self):
        for reason in ('baseline_only_provenance', 'unverified_temporal_label_in_crop', 'missing_cloud_qa'):
            with self.subTest(reason=reason):
                parent = deepcopy(self.parent); retained = deepcopy(self.retained)
                parent['blocking_reasons'].append(reason); retained['blocking_reasons'].append(reason)
                with self.assertRaisesRegex(ValueError, 'source-coverage'):
                    derive_pair(parent, retained, self.event, self.paths, self.frozen_ids)

    def test_v5_frozen_join_and_original_grid_are_required(self):
        self.retained['status'] = 'candidate_for_pixel_review'
        with self.assertRaisesRegex(ValueError, 'V5 retained'):
            self.derive()
        self.retained['status'] = self.parent['status']
        self.parent['grid']['bounds'][0] += 10
        with self.assertRaisesRegex(ValueError, 'grid'):
            self.derive()

    def test_parent_split_anchor_group_and_dates_cannot_be_reassigned(self):
        mutations = {'split': 'train', 'anchor_event_ids': ['different-event'],
                     'group_id': 'different-group', 'post_date': '2018-08-05', 'tile': '36UYA'}
        for key, value in mutations.items():
            with self.subTest(key=key):
                parent = deepcopy(self.parent); retained = deepcopy(self.retained)
                parent[key] = value
                if key in retained:
                    retained[key] = value
                with self.assertRaises(ValueError):
                    derive_pair(parent, retained, self.event, self.paths, self.frozen_ids)

    def test_changed_derived_grid_labels_split_source_or_identity_is_rejected(self):
        valid = self.derive()
        for key, value in [('pair_id', self.parent['pair_id']), ('split', 'train'),
                           ('label_event_ids_by_post_date', [self.event_id]),
                           ('anchor_event_ids', ['other']), ('pre_scene_id', 'unverified-scene'),
                           ('parent_pair_id', 'other-parent'), ('blocking_reasons', ['other'])]:
            with self.subTest(key=key):
                pair = deepcopy(valid); pair[key] = value
                with self.assertRaisesRegex(ValueError, 'lineage'):
                    validate_lineage(pair, self.parent, self.retained, self.event, self.paths, self.frozen_ids)
        pair = deepcopy(valid); pair['grid']['transform'][2] += 10
        with self.assertRaisesRegex(ValueError, 'lineage'):
            validate_lineage(pair, self.parent, self.retained, self.event, self.paths, self.frozen_ids)

    def test_processing_errors_are_fatal_but_adverse_qa_is_retained(self):
        with self.assertRaisesRegex(ValueError, 'QA processing failed'):
            require_processed({'quality_status': 'processing_error', 'quality_reasons': ['ValueError: missing B01']})
        require_processed({'quality_status': 'excluded_by_automatic_quality_screen',
                           'qa_mask_file': 'masks/derived.tif', 'per_date': {'pre': {}, 'post': {}}})

    def test_full_actual_qa_prerequisites_and_both_contexts_are_required(self):
        meta = {'metadata_usable': True, 'cloud_model_usable': True, 'shadow_geometry_usable': True,
                'issues': [], 'scene_id': SCENES[DATES[0]], 'acquisition_date': DATES[0],
                'tile': '36UXB', 'processing_level': 'L1C', 'processing_baseline': '02.06',
                'nonzero_offset': False, 'quantification_value': 10000., 'crs': 'EPSG:32636',
                'grid_10m': {'bounds': [600000., 5590200., 709800., 5700000.]}}
        validate_scene(meta, DATES[0], [self.event])
        for key, value in [('metadata_usable', False), ('cloud_model_usable', False),
                           ('processing_level', 'L2A'), ('nonzero_offset', True),
                           ('quantification_value', 1000.), ('scene_id', SCENES[DATES[1]])]:
            with self.subTest(key=key):
                bad = deepcopy(meta); bad[key] = value
                with self.assertRaises(ValueError):
                    validate_scene(bad, DATES[0], [self.event])
        bad = deepcopy(meta); bad['grid_10m']['bounds'] = list(self.grid['bounds'])
        with self.assertRaisesRegex(ValueError, 'full QA context'):
            validate_scene(bad, DATES[0], [self.event])

    def test_window_counter_restores_frozen_reader_even_on_error(self):
        module = Mock(); original = Mock(return_value='decoded'); module.read_fused_band = original
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with count_band_requests(module) as count:
                self.assertEqual(module.read_fused_band('a.jp2'), 'decoded')
                self.assertEqual(count['requests'], 1)
                raise RuntimeError('interrupted')
        self.assertIs(module.read_fused_band, original)

    def test_cache_allocation_headers_are_checked_before_any_numpy_load(self):
        import numpy as np
        shapes = {name: (64, 64) for name in
                  ('cloud', 'cloud_buffer', 'shadow_risk', 'haze_risk', 'invalid',
                   'probability', 'product_cloud', 'ndvi')}
        shapes.update(rgb=(3, 64, 64), metadata=())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cache.npz'
            with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for name, shape in shapes.items():
                    header = io.BytesIO()
                    dtype = '<U2' if name == 'metadata' else ('bool' if name in
                        ('cloud', 'cloud_buffer', 'shadow_risk', 'haze_risk', 'invalid', 'product_cloud') else 'float32')
                    np.lib.format.write_array_header_1_0(header, {'descr': np.dtype(dtype).str,
                        'fortran_order': False, 'shape': (999999999,) if name == 'cloud' else shape})
                    archive.writestr(name+'.npy', header.getvalue())
            receipt = {'path': 'cache.npz', 'bytes': path.stat().st_size,
                       'sha256': digest(path), 'cache_key': 'irrelevant'}
            with patch('screen_repaired_validation.ROOT', Path(directory)), \
                 patch('verify_validation_dataset.ROOT', Path(directory)), \
                 patch('numpy.load', side_effect=AssertionError('Must reject before decompression')):
                with self.assertRaisesRegex(ValueError, 'shape/type changed'):
                    load_safe_cache(receipt, {'cache_path': 'cache.npz'})


if __name__ == '__main__':
    unittest.main()
