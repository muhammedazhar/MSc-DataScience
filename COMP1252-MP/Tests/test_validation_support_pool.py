"""Broad support screening must preserve coverage and a prediction-blind roster."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from audit_validation_support_pool import (choose_shortlist, eligibility_reasons,
                                          enumerate_validation_anchors,
                                          followup_scene_ids, validate_protocol)

POLICY = 'fe2472ec030f466bebbb9dc2cb033ed6175de489bcf022f650081dd8e9347560'
STRATA = ('36UXA_time-dependent', '36UYA_Summer_time-dependent',
          '36UYA_Spring_time-dependent')
PASSED = 'passed_automatic_screen_pending_visual_review'


def grid(west=500000.):
    return {'crs': 'EPSG:32636', 'width': 64, 'height': 64,
            'bounds': [west, 5599360., west+640, 5600000.],
            'transform': [10., 0., west, 0., -10., 5600000.]}


def row(event='event-0', group='group-0', pair='pair-0', stratum=STRATA[0]):
    return {'event_id': event, 'group_id': group, 'pair_id': pair,
            'stratum': stratum, 'split': 'validation',
            'annotation_date': '2019-06-11', 'pre_date': '2019-06-06',
            'post_date': '2019-06-11', 'interval_days': 5, 'grid': grid(),
            'qa_status': PASSED, 'positive_interior_pixels': 4,
            'endpoint_crop_unusable_pixels': [0, 0],
            'endpoint_target_unusable_pixels': [0, 0],
            'interior_unusable_pixels': 0, 'footprint_isolated': True,
            'later_usable_scene_ids': ['scene-later'],
            'training_approved': False, 'evaluation_approved': False}


def pair(event='event-0', identifier='pair-0', split='validation', anchors=None):
    return {'pair_id': identifier, 'group_id': 'group-'+identifier,
            'split': split, 'tile': '36UXA',
            'pre_date': '2019-06-06', 'post_date': '2019-06-11',
            'interval_days': 5, 'pre_scene_id': 'scene-before',
            'post_scene_id': 'scene-after', 'pre_path': 'scenes/before.SAFE',
            'post_path': 'scenes/after.SAFE', 'grid': grid(),
            'anchor_event_ids': [event] if anchors is None else list(anchors),
            'status': 'candidate_for_pixel_review', 'blocking_reasons': [],
            'training_approved': False, 'pixel_quality': 'unreviewed'}


def qa_and_index(frozen, status=PASSED):
    identity = {key: deepcopy(frozen[key]) for key in
                ('pair_id', 'group_id', 'split', 'tile', 'pre_date', 'post_date',
                 'anchor_event_ids')}
    qa = {**identity, 'config_id': POLICY, 'quality_status': status,
          'training_approved': False, 'visual_review': 'unreviewed',
          'quality_reasons': [] if status == PASSED else ['recorded_quality_exclusion'],
          'target_metrics': {'region_pixels': 20, 'unusable_pixels': 0},
          'per_date': {side: {'crop': {'region_pixels': 4096, 'unusable_pixels': 0},
                              'target': {'region_pixels': 20, 'unusable_pixels': 0},
                              'observation': {'scene_id': frozen[side+'_scene_id'],
                                  'config_id': POLICY, 'grid': deepcopy(frozen['grid']),
                                  'cache_path': 'cache/'+frozen['pair_id']+'-'+side+'.npz'}}
                       for side in ('pre', 'post')}}
    index = {**identity, 'config_id': POLICY, 'quality_status': status,
             'training_approved': False, 'visual_review': 'unreviewed',
             'batch_path': 'qa/batch', 'qa_mask_file': frozen['pair_id']+'.tif'}
    return qa, index


class BroadValidationSupportTests(unittest.TestCase):
    def test_predeclared_protocol_cannot_silently_change_support_history_or_capacity(self):
        protocol = json.loads((Path(__file__).resolve().parents[1]/'Docs/Reviews/ValidationSupportV2/protocol.json').read_text())
        validate_protocol(protocol)
        for section, key, value in (('rule', 'minimum_interior_pixels', 1),
                ('rule', 'exclude_prior_groups', False), ('rule', 'maximum_shortlist', 12),
                ('rule', 'ranking_uses_ndvi_or_model_scores', True),
                ('resources', 'workers', 2), ('resources', 'worker_rss_mib', 1024)):
            changed = deepcopy(protocol); changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol(changed)

    def roster_rows(self):
        return [row(f'event-{source}-{option}', f'group-{source}-{option}',
                    f'pair-{source}-{option}', stratum)
                for source, stratum in enumerate(STRATA) for option in range(2)]

    def test_every_qa_anchor_is_audited_without_old_top_three_area_or_quality_filter(self):
        first = pair(); first.update(area_m2=1., width_m=1., top_candidates=[])
        shared = pair(identifier='pair-shared', anchors=['event-thin', 'event-wide'])
        training = pair(event='event-train', identifier='pair-train', split='train')
        frozen = [first, shared, training]
        receipts = [qa_and_index(first),
                    qa_and_index(shared, 'excluded_by_automatic_quality_screen'),
                    qa_and_index(training)]
        quality = [q for q, _ in receipts]; indices = [i for _, i in receipts]
        originals = deepcopy((frozen, quality, indices))
        result = enumerate_validation_anchors(frozen, quality, indices, POLICY)
        self.assertEqual({(r['event_id'], r['pair']['pair_id']) for r in result},
                         {('event-0', 'pair-0'), ('event-thin', 'pair-shared'),
                          ('event-wide', 'pair-shared')})
        self.assertEqual(len(result), 3)
        self.assertEqual((frozen, quality, indices), originals)
        # Neither an old search-aid size nor a caller's stale eligibility flag
        # can replace the newly measured anchor-specific support criterion.
        candidate = row(); candidate.update(area_m2=1., width_m=1.,
                                             top_candidates=[], eligible=False)
        self.assertEqual(eligibility_reasons(candidate, set()), [])
        self.assertEqual(choose_shortlist([candidate], set())['candidates'][0]['event_id'], 'event-0')

    def test_frozen_qa_index_identity_mismatches_fail_instead_of_losing_coverage(self):
        frozen = pair(); original_qa, original_index = qa_and_index(frozen)
        for source, field, value in (
                ('qa', 'group_id', 'different-group'), ('qa', 'split', 'train'),
                ('qa', 'anchor_event_ids', ['different-event']),
                ('qa', 'config_id', 'different-policy'),
                ('index', 'group_id', 'different-group'), ('index', 'split', 'test'),
                ('index', 'anchor_event_ids', ['different-event']),
                ('index', 'config_id', 'different-policy')):
            qa, index = deepcopy(original_qa), deepcopy(original_index)
            (qa if source == 'qa' else index)[field] = value
            with self.subTest(source=source, field=field), self.assertRaises(ValueError):
                enumerate_validation_anchors([frozen], [qa], [index], POLICY)

    def test_actual_support_and_all_qa_isolation_followup_gates_are_required(self):
        valid = row()
        self.assertEqual(eligibility_reasons(valid, set()), [])
        for field, invalid in (
                ('positive_interior_pixels', 3),
                ('qa_status', 'needs_visual_review'),
                ('endpoint_crop_unusable_pixels', [0, 1]),
                ('endpoint_target_unusable_pixels', [1, 0]),
                ('interior_unusable_pixels', 1), ('footprint_isolated', False),
                ('later_usable_scene_ids', []), ('stratum', 'unknown-source')):
            candidate = {**deepcopy(valid), field: invalid, 'eligible': True}
            with self.subTest(field=field):
                self.assertTrue(eligibility_reasons(candidate, set()))
                self.assertFalse(choose_shortlist([candidate], set())['candidates'])

    def test_prior_event_rejection_applies_to_every_other_temporal_pair(self):
        old = row(); alternative = {**deepcopy(old), 'pair_id': 'pair-new'}
        for candidate in (old, alternative):
            self.assertEqual(eligibility_reasons(candidate, set()), [])
            self.assertTrue(eligibility_reasons(candidate, {'event-0'}))
        result = choose_shortlist([old, alternative], {'event-0'})
        self.assertFalse(result['candidates'])

    def test_prior_group_excludes_an_unreviewed_event_without_discarding_its_audit_record(self):
        fresh_event = row(event='event-unreviewed', group='previously-reviewed-group')
        self.assertEqual(eligibility_reasons(fresh_event, {'different-event'}), [])
        reasons = eligibility_reasons(fresh_event, {'different-event'}, {'previously-reviewed-group'})
        self.assertTrue(reasons)
        self.assertFalse(choose_shortlist([fresh_event], {'different-event'},
                                         {'previously-reviewed-group'})['candidates'])
        # Eligibility helpers return decisions; they must not mutate the audit
        # record or invent a preparation/label approval while rejecting it.
        self.assertEqual(fresh_event['event_id'], 'event-unreviewed')
        self.assertIs(fresh_event['evaluation_approved'], False)

    def test_input_order_and_spectral_or_model_scores_do_not_change_fixed_shortlist(self):
        rows = self.roster_rows(); expected = choose_shortlist(rows, set())
        mutated = deepcopy(rows)
        for index, candidate in enumerate(mutated):
            candidate.update(forest_model_dice=index/10, model_score=-index,
                             target_median_ndvi_drop=100-index, positive_probability=index/7)
        mutated.reverse()
        self.assertEqual(choose_shortlist(mutated, set()), expected)

    def test_exact_annotation_post_then_shorter_interval_then_pair_id_rank(self):
        exact_long = row(pair='pair-exact-long')
        exact_long.update(pre_date='2019-06-01', interval_days=10)
        later_short = row(pair='pair-later-short')
        later_short.update(pre_date='2019-06-08', post_date='2019-06-13', interval_days=5)
        exact_short_z = row(pair='pair-z')
        exact_short_a = row(pair='pair-a')
        result = choose_shortlist([later_short, exact_long, exact_short_z, exact_short_a], set())
        self.assertEqual(result['candidates'][0]['pair_id'], 'pair-a')
        self.assertEqual(len(result['candidates']), 1)

    def test_all_primaries_precede_alternates_and_groups_never_repeat(self):
        rows = self.roster_rows()
        result = choose_shortlist(rows, set())
        self.assertEqual([r['role'] for r in result['candidates']],
                         ['primary']*3+['alternate']*3)
        self.assertEqual([r['stratum'] for r in result['candidates'][:3]], list(STRATA))
        shared = deepcopy(rows)
        for candidate in shared:
            if candidate['stratum'] == STRATA[1]:
                candidate['group_id'] = rows[0]['group_id']
        reduced = choose_shortlist(shared, set())
        self.assertEqual(len({r['group_id'] for r in reduced['candidates']}), len(reduced['candidates']))
        missing = [(slot['stratum'], slot['role']) for slot in reduced['missing_slots']]
        self.assertIn((STRATA[1], 'primary'), missing)
        self.assertIn((STRATA[1], 'alternate'), missing)
        self.assertLessEqual(len(reduced['candidates']), 6)

    def test_followup_requires_later_usable_same_year_scene_within_declared_window(self):
        frozen = pair()
        scenes = [
            {'scene_id': 'endpoint', 'tile': '36UXA', 'acquisition_date': '2019-06-11', 'metadata_usable': True},
            {'scene_id': 'metadata-unusable', 'tile': '36UXA', 'acquisition_date': '2019-06-16', 'metadata_usable': False},
            {'scene_id': 'wrong-tile', 'tile': '36UYA', 'acquisition_date': '2019-06-16', 'metadata_usable': True},
            {'scene_id': 'usable', 'tile': '36UXA', 'acquisition_date': '2019-06-21', 'metadata_usable': True},
            {'scene_id': 'outside-window', 'tile': '36UXA', 'acquisition_date': '2019-07-27', 'metadata_usable': True},
            {'scene_id': 'wrong-year', 'tile': '36UXA', 'acquisition_date': '2020-06-16', 'metadata_usable': True}]
        self.assertEqual(followup_scene_ids(frozen, '2019-06-11', scenes), ['usable'])
        scenes[3]['metadata_usable'] = False
        self.assertEqual(followup_scene_ids(frozen, '2019-06-11', scenes), [])


if __name__ == '__main__':
    unittest.main()
