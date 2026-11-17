"""Independent manual review must retain scope, full context and every date."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Solutions'))
from verify_validation_manual import (bound_review, digest, expected_render_views,
    independent_roster, observation_metrics, review_decision, square_context, verify_render_views)


class ManualVerificationTests(unittest.TestCase):
    def arrays(self):
        return {'pre_qa': np.zeros((5, 64, 64), dtype=bool),
            'post_qa': np.zeros((5, 64, 64), dtype=bool),
            'pre_rgb': np.zeros((3, 64, 64), dtype='float32'),
            'post_rgb': np.zeros((3, 64, 64), dtype='float32'),
            'pre_ndvi': np.full((64, 64), .1, dtype='float32'),
            'post_ndvi': np.full((64, 64), .9, dtype='float32'),
            'features': np.zeros((27, 64, 64), dtype='float32'),
            'feature_valid': np.ones((64, 64), dtype=bool),
            'control_exclusion': np.zeros((64, 64), dtype=bool)}

    def review_fixture(self):
        dates = ['2019-04-02', '2019-04-27', '2019-05-17', '2019-06-01']
        controls = [{'control_id': 'control-'+str(index+1)} for index in range(3)]
        packet = {'timeline': [{'scene_id': 'scene-'+str(i), 'date': day,
            'observation': {'present': True}, 'target_metrics': {'context_clear': True},
            'control_metrics': [{'context_clear': True} for _ in controls]} for i, day in enumerate(dates)],
            'controls': controls, 'pair': {'pre_scene_id': 'scene-0', 'post_scene_id': 'scene-1',
                'pre_date': dates[0], 'post_date': dates[1]},
            'positive_interior_pixels': 4, 'artifact_sha256': {'observations.npz': 'a'*64}}
        statements = [{'scene_id': row['scene_id'], 'date': row['date'],
            'target_state': 'forest' if index == 0 else 'clearing_visible',
            'controls': {control['control_id']: 'stable_forest' for control in controls},
            'notes': 'Synthetic explicit reviewed date'} for index, row in enumerate(packet['timeline'])]
        review = {'schema_version': 4, 'reviewer': 'Synthetic unit fixture', 'notes': 'Explicit manual evidence',
            'used_model_predictions': False, 'labels_approved': 0, 'training_approved': False,
            'evaluation_approved': False, 'observations': statements}
        return packet, review

    def test_context_exposes_boundary_loss_and_diagonal_risk(self):
        mask = np.zeros((64, 64), dtype=bool); mask[2, 2] = True
        halo, inside = square_context(mask)
        self.assertTrue(inside); self.assertEqual(int(halo.sum()), 25); self.assertTrue(halo[0, 0])
        mask[:] = False; mask[0, 0] = True
        halo, inside = square_context(mask)
        self.assertFalse(inside); self.assertEqual(int(halo.sum()), 9)

    def test_roster_keeps_largest_first_order_and_reserves_full_disjoint_halos(self):
        arrays = self.arrays()
        rois, halos, records = independent_roster(arrays)
        self.assertEqual([(row['row'], row['col'], row['height']) for row in records],
                         [(2, 2, 16), (2, 22, 16), (2, 42, 16)])
        self.assertEqual(list(rois.sum(axis=(1, 2))), [256]*3)
        self.assertTrue((halos.sum(axis=0) <= 1).all())
        self.assertFalse(records[0]['evaluation_approved'])
        # NDVI values are deliberately below/unstable by V3's proxy. They never
        # affect the new pre-image geometry roster, apart from finiteness.
        arrays['post_ndvi'][:] = -.5
        self.assertTrue(np.array_equal(rois, independent_roster(arrays)[0]))

    def test_roster_rejects_qa_and_nonfinite_values_outside_roi_inside_halo(self):
        arrays = self.arrays(); arrays['pre_qa'][0, 0, 0] = True
        _, halos, records = independent_roster(arrays)
        self.assertEqual((records[0]['row'], records[0]['col']), (2, 3))
        self.assertFalse((halos & arrays['pre_qa'][0]).any())
        arrays = self.arrays(); arrays['features'][26, 0, 0] = np.nan
        self.assertEqual(independent_roster(arrays)[2][0]['col'], 3)

    def test_metrics_count_qa_union_once_and_include_finite_display_context(self):
        qa = np.zeros((5, 64, 64), dtype=bool); qa[:2, 18, 18] = True
        rgb = np.zeros((3, 64, 64), dtype='float32'); ndvi = np.zeros((64, 64), dtype='float32')
        roi = np.zeros((64, 64), dtype=bool); roi[20:28, 20:28] = True
        result = observation_metrics(qa, rgb, ndvi, roi)
        self.assertEqual(result['roi']['unusable_pixels'], 0)
        self.assertEqual(result['context']['unusable_pixels'], 1)
        self.assertEqual(result['context']['cloud_pixels'], 1)
        self.assertEqual(result['context']['cloud_buffer_pixels'], 1)
        self.assertFalse(result['context_clear'])
        qa[:] = False; rgb[2, 18, 18] = np.nan
        result = observation_metrics(qa, rgb, ndvi, roi)
        self.assertEqual(result['context_nonfinite_rgb_ndvi_pixels'], 1)
        self.assertFalse(result['context_clear'])

    def test_first_qualifying_locked_control_follows_all_date_dispositions(self):
        packet, review = self.review_fixture()
        review['observations'][2]['controls']['control-1'] = 'nonforest_or_changed'
        result = review_decision(packet, review)
        self.assertEqual(result['decision_status'], 'manual_evidence_supported')
        self.assertEqual(result['selected_control_id'], 'control-2')
        self.assertEqual(result['candidate_controls'][0]['decision_status'], 'rejected')
        self.assertEqual(result['observed_change_interval']['lower_bound_exclusive'], '2019-04-02')
        self.assertEqual(result['observed_change_interval']['upper_bound_inclusive'], '2019-04-27')
        self.assertEqual(result['labels_approved'], 0); self.assertFalse(result['evaluation_approved'])

    def test_uncertain_target_cannot_supply_negative_only_labels(self):
        packet, review = self.review_fixture()
        review['observations'][0]['target_state'] = 'uncertain'
        result = review_decision(packet, review)
        self.assertEqual(result['decision_status'], 'rejected')
        self.assertIsNone(result['selected_control_id']); self.assertIsNone(result['observed_change_interval'])
        self.assertEqual(result['labels_approved'], 0)

    def test_later_persistence_required_and_later_forest_contradiction_retained(self):
        packet, review = self.review_fixture()
        review['observations'][2]['target_state'] = review['observations'][3]['target_state'] = 'uncertain'
        self.assertEqual(review_decision(packet, review)['decision_status'], 'rejected')
        packet, review = self.review_fixture(); review['observations'][3]['target_state'] = 'forest'
        self.assertEqual(review_decision(packet, review)['decision_status'], 'rejected')
        packet, review = self.review_fixture()
        for row in review['observations'][2:]:
            row['controls'] = {key: 'uncertain' for key in row['controls']}
        self.assertEqual(review_decision(packet, review)['decision_status'], 'rejected')

    def test_definite_claims_cannot_bypass_full_context_qa_or_hide_extant_date(self):
        packet, review = self.review_fixture(); packet['timeline'][0]['target_metrics']['context_clear'] = False
        with self.assertRaises(ValueError): review_decision(packet, review)
        packet, review = self.review_fixture(); packet['timeline'][1]['control_metrics'][0]['context_clear'] = False
        with self.assertRaises(ValueError): review_decision(packet, review)
        packet, review = self.review_fixture(); review['observations'][2]['target_state'] = 'unavailable'
        with self.assertRaises(ValueError): review_decision(packet, review)

    def test_every_control_and_timeline_row_requires_exact_coverage(self):
        packet, review = self.review_fixture(); review['observations'].pop()
        with self.assertRaises(ValueError): review_decision(packet, review)
        packet, review = self.review_fixture(); del review['observations'][2]['controls']['control-3']
        with self.assertRaises(ValueError): review_decision(packet, review)
        packet, review = self.review_fixture(); review['observations'].reverse()
        with self.assertRaises(ValueError): review_decision(packet, review)

    def test_bound_review_rejects_changed_artifacts_decision_or_temporal_claim(self):
        packet, review = self.review_fixture()
        review.update(protocol_sha256='b'*64, artifact_sha256=packet['artifact_sha256'],
            decision_status='manual_evidence_supported', selected_control_id='control-1',
            candidate_controls=[{'control_id': row['control_id'], 'decision_status': 'supported',
                'notes': 'Three clear dates including both endpoints and later.'} for row in packet['controls']])
        with tempfile.TemporaryDirectory() as temporary:
            metadata = Path(temporary)/'metadata.json'; metadata.write_text('{"fixed": true}\n')
            review['prepared_metadata_sha256'] = digest(metadata)
            self.assertEqual(bound_review(packet, review, metadata, 'b'*64)['selected_control_id'], 'control-1')
            for changed in [dict(review, artifact_sha256={}), dict(review, selected_control_id='control-2'),
                            dict(review, observed_change_interval={'exact_felling_date': '2019-04-22'})]:
                with self.assertRaises(ValueError): bound_review(packet, changed, metadata, 'b'*64)
            metadata.write_text('{"fixed": false}\n')
            with self.assertRaises(ValueError): bound_review(packet, review, metadata, 'b'*64)

    def render_fixture(self):
        timeline = [{'scene_id': 'scene-'+str(index)} for index in range(12)]
        target = np.zeros((64, 64), dtype=bool); target[30:36, 30:36] = True
        context, _ = square_context(target)
        arrays = {'target_context': context, 'rgb': np.zeros((12, 3, 64, 64), dtype='float32'),
                  'qa': np.zeros((12, 5, 64, 64), dtype=bool)}
        packet = {'timeline': timeline, 'pair': {'pre_scene_id': 'scene-2', 'post_scene_id': 'scene-6'}}
        packet['render_views'] = expected_render_views(packet, arrays)
        packet['artifact_sha256'] = {'observations.npz': 'f'*64,
            **{row['file']: 'e'*64 for row in packet['render_views']}}
        return packet, arrays

    def test_missing_timeline_page_or_wrong_scene_coverage_fails_before_image_load(self):
        packet, arrays = self.render_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            del packet['artifact_sha256']['timeline-02.png']
            with self.assertRaises(ValueError): verify_render_views(temporary, packet, arrays)
            packet, arrays = self.render_fixture()
            packet['render_views'][1]['scene_ids'][0] = 'scene-0'
            with self.assertRaises(ValueError): verify_render_views(temporary, packet, arrays)

    def test_shown_plain_pixels_must_match_actual_date_rgb(self):
        from PIL import Image
        packet, arrays = self.render_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            for view in packet['render_views']:
                Image.new('RGB', (view['width'], view['height']), 'black').save(Path(temporary)/view['file'])
            result = verify_render_views(temporary, packet, arrays)
            self.assertEqual(result['plain_rgb_panels_checked'], 16)
            self.assertEqual(result['qa_panels_checked'], 14)
            path = Path(temporary)/'timeline-02.png'
            with Image.open(path) as image:
                image.putpixel((14, 116), (255, 255, 255)); image.save(path)
            with self.assertRaisesRegex(ValueError, 'plain RGB'):
                verify_render_views(temporary, packet, arrays)

    def test_actual_renderer_keeps_fixed_plain_and_qa_pixels_separate_from_annotations(self):
        from prepare_validation_manual import render_packet
        packet, arrays = self.render_fixture()
        target = np.zeros((64, 64), dtype=bool); target[30:36, 30:36] = True
        controls, contexts, roster = independent_roster(self.arrays())
        arrays.update(target=target, controls=controls, control_contexts=contexts)
        # Spatial colour variation catches swapped dates, crops or resampling.
        for index in range(12):
            arrays['rgb'][index, 0] = np.arange(64, dtype='float32')[None, :]/400+index/1000
            arrays['rgb'][index, 1] = np.arange(64, dtype='float32')[:, None]/400
            arrays['qa'][index, index % 5, 2:4, 3:5] = True
        observations = [{'rgb': arrays['rgb'][index],
            **{name: arrays['qa'][index, plane] for plane, name in enumerate(
                ('cloud', 'cloud_buffer', 'shadow_risk', 'haze_risk', 'invalid'))}} for index in range(12)]
        for index, row in enumerate(packet['timeline']):
            row.update(date=f'2019-04-{index+1:02d}', target_metrics={'context': {'unusable_pixels': 0}})
        render_arrays = {**arrays, 'pair_pre': packet['pair']['pre_scene_id'], 'pair_post': packet['pair']['post_scene_id']}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            packet['render_views'] = render_packet(directory, observations, packet['timeline'], render_arrays, roster)
            result = verify_render_views(directory, packet, arrays)
            self.assertEqual(result['plain_rgb_panels_checked'], 16)
            self.assertEqual(result['qa_panels_checked'], 14)


if __name__ == '__main__':
    unittest.main()
