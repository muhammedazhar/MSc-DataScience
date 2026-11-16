"""Metadata-only protections for prediction-blind validation review evidence."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from curated_pilot_review import _hash
from validation_review import select_stable_pair, validate_validation_review


def bind(case, review):
    """Bind synthetic reviewed evidence after intentional fixture alterations."""
    review['prepared_case_sha256'] = _hash(case)
    review['artifact_sha256'] = deepcopy(case['artifact_sha256'])


def fixture():
    days = ('2019-06-01', '2019-06-06', '2019-06-11',
            '2019-06-16', '2019-06-21', '2019-06-26')
    states = ('forest', 'forest', 'forest', 'clearing_visible',
              'uncertain', 'clearing_visible')
    timeline, inspected = [], []
    for index, (day, state) in enumerate(zip(days, states)):
        event = {'region_pixels': 40, 'unusable_pixels': 0, 'invalid_pixels': 0,
                 'valid_pixels': 40, 'usable_pixels': 40, 'denominator_pixels': 40}
        control = {**event, 'region_pixels': 64, 'valid_pixels': 64,
                   'usable_pixels': 64, 'denominator_pixels': 64}
        timeline.append({'scene_id': f'scene-{index}', 'date': day,
                         'metadata_usable': True, 'event_metrics': event,
                         'control_metrics': control,
                         'observation': {'config_id': 'frozen-policy'}})
        inspected.append({'scene_id': f'scene-{index}', 'date': day,
                          'target_state': state, 'control_state': 'stable_forest',
                          'notes': f'Explicit visual inspection of {day}.'})
    case = {'event_id': 'event-validation', 'pair_id': 'pair-validation',
            'group_id': 'group-validation', 'split': 'validation',
            'training_approved': False, 'evaluation_approved': False,
            'config_id': 'frozen-policy', 'target_pixels': 40,
            'positive_interior_pixels': 16,
            'control_proposal': {'proposal_found': True, 'pixels': 64},
            'artifact_sha256': {'pair.png': 'a'*64, 'timeline.png': 'b'*64,
                                'positive_interior.tif': 'c'*64,
                                'proposed_control.tif': 'd'*64},
            'pair': {'pair_id': 'pair-validation', 'group_id': 'group-validation',
                     'split': 'validation', 'anchor_event_ids': ['event-validation'],
                     'pre_scene_id': 'scene-2', 'post_scene_id': 'scene-3',
                     'pre_date': days[2], 'post_date': days[3]},
            'timeline': timeline}
    review = {'event_id': 'event-validation', 'pair_id': 'pair-validation',
              'visual_review': 'accepted', 'temporal_label_review': 'accepted',
              'reviewer': 'assistant visual inspection', 'used_model_predictions': False,
              'qa_policy_id': 'frozen-policy',
              'notes': 'All prepared observations inspected without model predictions.',
              'observations': inspected}
    bind(case, review)
    return case, review


class ValidationReviewTests(unittest.TestCase):
    def test_accepted_review_approves_evaluation_only_and_derives_observed_bounds(self):
        case, review = fixture()
        originals = deepcopy((case, review))
        result = validate_validation_review(case, review)
        self.assertEqual(result['decision_status'], 'accepted')
        self.assertEqual(result['split'], 'validation')
        self.assertIs(result['training_approved'], False)
        self.assertIs(result['evaluation_approved'], True)
        self.assertEqual(result['observed_change_interval']['lower_bound_exclusive'], '2019-06-11')
        self.assertEqual(result['observed_change_interval']['upper_bound_inclusive'], '2019-06-16')
        self.assertEqual(result['observed_change_interval']['width_days'], 5)
        self.assertEqual(result['observed_change_interval']['persistence_scene_ids'], ['scene-5'])
        self.assertEqual(result['stable_pair']['pre_date'], '2019-06-01')
        self.assertEqual(result['stable_pair']['post_date'], '2019-06-06')
        self.assertEqual((case, review), originals)
        result['observations'][0]['notes'] = 'Changed returned copy'
        self.assertEqual((case, review), originals)

    def test_training_test_and_already_approved_evidence_are_rejected(self):
        for change in ('train', 'test', 'mixed_split', 'training_approved', 'evaluation_approved'):
            case, review = fixture()
            if change in ('train', 'test'):
                case['split'] = case['pair']['split'] = change
            elif change == 'mixed_split':
                case['pair']['split'] = 'train'
            else:
                case[change] = True
            bind(case, review)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'validation-only'):
                validate_validation_review(case, review)

    def test_model_assisted_or_missing_blind_review_declaration_cannot_approve_labels(self):
        for value in (True, None, 0):
            case, review = fixture()
            review['used_model_predictions'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'without predictions'):
                validate_validation_review(case, review)
        case, review = fixture()
        del review['used_model_predictions']
        with self.assertRaisesRegex(ValueError, 'without predictions'):
            validate_validation_review(case, review)

    def test_case_and_artifact_mutations_invalidate_review_bindings(self):
        for change in ('case', 'review_artifact', 'prepared_artifact'):
            case, review = fixture()
            if change == 'case':
                case['target_pixels'] = 41
            elif change == 'review_artifact':
                review['artifact_sha256']['pair.png'] = 'e'*64
            else:
                case['artifact_sha256']['pair.png'] = 'e'*64
                # Bind the changed metadata but deliberately retain old reviewed
                # artifact bytes to exercise the independent artifact binding.
                review['prepared_case_sha256'] = _hash(case)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_validation_review(case, review)

    def test_clear_claim_cannot_override_qa_metadata_validity_or_policy(self):
        for change in ('target_qa', 'control_qa', 'component', 'validity', 'metadata', 'cache_policy'):
            case, review = fixture()
            row = case['timeline'][2]
            if change == 'target_qa':
                row['event_metrics']['unusable_pixels'] = 1
            elif change == 'control_qa':
                row['control_metrics']['unusable_pixels'] = 1
            elif change == 'component':
                row['event_metrics']['shadow_risk_pixels'] = 1
            elif change == 'validity':
                row['event_metrics']['valid_pixels'] = 39
            elif change == 'metadata':
                row['metadata_usable'] = False
            else:
                row['observation']['config_id'] = 'other-policy'
            bind(case, review)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_validation_review(case, review)

    def test_persistence_and_no_later_forest_are_required(self):
        for change in ('missing_persistence', 'later_forest'):
            case, review = fixture()
            if change == 'missing_persistence':
                review['observations'][5]['target_state'] = 'uncertain'
            else:
                review['observations'][4]['target_state'] = 'forest'
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_validation_review(case, review)

    def test_exact_scene_date_coverage_and_order_are_required(self):
        for change in ('missing', 'duplicate', 'unknown', 'misdated', 'review_order', 'prepared_order'):
            case, review = fixture()
            if change == 'missing':
                review['observations'].pop()
            elif change == 'duplicate':
                review['observations'].append(deepcopy(review['observations'][0]))
            elif change == 'unknown':
                review['observations'][0]['scene_id'] = 'not-prepared'
            elif change == 'misdated':
                review['observations'][0]['date'] = '2019-05-31'
            elif change == 'review_order':
                review['observations'].reverse()
            else:
                case['timeline'].reverse()
                bind(case, review)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_validation_review(case, review)

    def test_stable_pair_is_earliest_valid_pre_loss_combination_not_empty_target(self):
        pair = {'pre_date': '2019-06-11'}
        rows = [
            {'scene_id': 'late', 'date': '2019-06-16', 'target_state': 'forest', 'control_state': 'stable_forest'},
            {'scene_id': 'earliest', 'date': '2019-06-01', 'target_state': 'forest', 'control_state': 'stable_forest'},
            {'scene_id': 'too_close', 'date': '2019-06-02', 'target_state': 'forest', 'control_state': 'stable_forest'},
            {'scene_id': 'eligible', 'date': '2019-06-06', 'target_state': 'forest', 'control_state': 'stable_forest'},
            {'scene_id': 'not_forest', 'date': '2019-05-27', 'target_state': 'uncertain', 'control_state': 'stable_forest'},
            {'scene_id': 'not_control', 'date': '2019-05-22', 'target_state': 'forest', 'control_state': 'uncertain'}]
        result = select_stable_pair({row['scene_id']: row for row in rows}, pair)
        self.assertEqual((result['pre_scene_id'], result['post_scene_id']), ('earliest', 'eligible'))
        self.assertEqual(result['kind'], 'derived_pre_event_stable_roi')
        self.assertIn('control ROI only', result['supervision_scope'])

    def test_no_eligible_stable_pair_remains_none_without_imputed_negative(self):
        case, review = fixture()
        # The loss evidence remains valid; only one reviewed forest date remains.
        review['observations'][0]['target_state'] = 'uncertain'
        review['observations'][1]['target_state'] = 'uncertain'
        result = validate_validation_review(case, review)
        self.assertIs(result['evaluation_approved'], True)
        self.assertIsNone(result['stable_pair'])

    def test_rejection_retains_reasons_and_never_approves_or_derives_intervals(self):
        case, review = fixture()
        review.update(visual_review='rejected', temporal_label_review='rejected',
                      rejection_reasons=['uncertain_visible_change'], training_approved=True,
                      evaluation_approved=True, observed_change_interval={'invented': 'date'})
        for row in review['observations']:
            row.update(target_state='uncertain', control_state='uncertain')
        result = validate_validation_review(case, review)
        self.assertEqual(result['decision_status'], 'rejected')
        self.assertEqual(result['rejection_reasons'], ['uncertain_visible_change'])
        self.assertIs(result['training_approved'], False)
        self.assertIs(result['evaluation_approved'], False)
        self.assertIsNone(result['observed_change_interval'])
        self.assertIsNone(result['stable_pair'])


if __name__ == '__main__':
    unittest.main()
