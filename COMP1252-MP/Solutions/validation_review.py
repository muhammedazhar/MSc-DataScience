"""Explicit, prediction-blind review rules for held-out partial ROI validation.

Generic statement/QA checks are reused without changing or bypassing the
training-only review validator. This module approves evaluation only.
"""

from copy import deepcopy
from itertools import combinations

from curated_pilot_review import (_bound_review, _clear_claim, _day, _hash,
                                  _integer, _require, _string)


def prepared_validation(case):
    pair = case.get('pair', {})
    _require(case.get('split') == pair.get('split') == 'validation'
             and case.get('training_approved') is False
             and case.get('evaluation_approved') is False,
             'Prepared evidence must be unapproved validation-only data')
    _require(case.get('event_id') in pair.get('anchor_event_ids', [])
             and case.get('pair_id') == pair.get('pair_id')
             and case.get('group_id') == pair.get('group_id'), 'Exact validation anchor identity required')
    for field in ('pair_id', 'group_id', 'pre_scene_id', 'post_scene_id'):
        _string(pair.get(field), field)
    _require(5 <= (_day(pair['post_date'])-_day(pair['pre_date'])).days <= 30,
             'Loss pair must retain a 5–30 day interval')
    policy = _string(case.get('config_id'), 'frozen QA policy')
    rows = case.get('timeline')
    _require(isinstance(rows, list) and 2 <= len(rows) <= 40, 'Bounded complete timeline required')
    scenes, dates = {}, []
    for row in rows:
        identifier = _string(row.get('scene_id'), 'scene ID'); day = _day(row.get('date'))
        _require(identifier not in scenes and day not in dates, 'Timeline scenes/dates must be unique')
        scenes[identifier] = row; dates.append(day)
    _require(dates == sorted(dates), 'Timeline must be chronological')
    for side in ('pre', 'post'):
        identifier = pair[side+'_scene_id']
        _require(identifier in scenes and scenes[identifier]['date'] == pair[side+'_date'],
                 'Primary scene/date must match prepared timeline')
    return pair, policy, scenes


def select_stable_pair(inspected, pair):
    """Earliest eligible reviewed forest pair; no empty-target negative labels."""
    rows = [row for row in inspected.values()
            if row['target_state'] == 'forest' and row['control_state'] == 'stable_forest'
            and _day(row['date']) <= _day(pair['pre_date'])]
    rows.sort(key=lambda row: (row['date'], row['scene_id']))
    for before, after in combinations(rows, 2):
        if 5 <= (_day(after['date'])-_day(before['date'])).days <= 30:
            return {'pre_scene_id': before['scene_id'], 'post_scene_id': after['scene_id'],
                    'pre_date': before['date'], 'post_date': after['date'],
                    'kind': 'derived_pre_event_stable_roi',
                    'selection_rule': 'Earliest chronological 5–30 day combination with both reviewed forest and stable control, no later than loss pre-date',
                    'supervision_scope': 'Reviewed control ROI only; all other pixels remain 255'}
    return None


def validate_validation_review(case, review):
    pair, policy, prepared = prepared_validation(case)
    _require(review.get('used_model_predictions') is False, 'Validation labels must be reviewed without predictions')
    _require(review.get('prepared_case_sha256') == _hash(case), 'Review must bind the exact prepared case')
    _require(isinstance(case.get('artifact_sha256'), dict) and bool(case['artifact_sha256'])
             and review.get('artifact_sha256') == case['artifact_sha256'], 'Review artifact bindings changed')
    inspected = _bound_review(case, review, pair, policy, prepared)
    result = deepcopy(review)
    result.update({'split': 'validation', 'group_id': pair['group_id'], 'pair_id': pair['pair_id'],
                   'training_approved': False, 'evaluation_approved': False,
                   'observed_change_interval': None, 'stable_pair': None,
                   'approval_scope': 'Assistant-reviewed partial validation ROIs; no expert ground-truth certification'})
    if review['visual_review'] != 'accepted' or review['temporal_label_review'] != 'accepted':
        result['decision_status'] = 'rejected'
        result['rejection_reasons'] = deepcopy(review.get('rejection_reasons', [review['notes']]))
        return result
    target = _integer(case.get('target_pixels'), 'target pixels', 1)
    interior = _integer(case.get('positive_interior_pixels'), 'positive interior', 4)
    proposal = case.get('control_proposal', {})
    _require(interior <= target and proposal.get('proposal_found') is True,
             'Accepted validation needs an interior and a control proposal')
    controls = _integer(proposal.get('pixels'), 'control pixels', 64)
    for identifier, row in inspected.items():
        if row['target_state'] in ('forest', 'clearing_visible'):
            _clear_claim(prepared[identifier], 'event_metrics', target, policy)
        if row['control_state'] == 'stable_forest':
            _clear_claim(prepared[identifier], 'control_metrics', controls, policy)
    pre, post = (inspected[pair[side+'_scene_id']] for side in ('pre', 'post'))
    _require(pre['target_state'] == 'forest' and post['target_state'] == 'clearing_visible',
             'Loss endpoints must explicitly show forest then clearing')
    _require(pre['control_state'] == post['control_state'] == 'stable_forest',
             'Reviewed forest control must be clear at both endpoints')
    rows = list(inspected.values())
    clearings = [r for r in rows if r['target_state'] == 'clearing_visible']
    first = clearings[0]
    _require(len(clearings) >= 2, 'Later clear observation must confirm persistence')
    _require(not any(r['target_state'] == 'forest' and r['date'] >= first['date'] for r in rows),
             'Later forest contradicts persistent loss')
    forests = [r for r in rows if r['target_state'] == 'forest' and r['date'] < first['date']]
    _require(bool(forests), 'Observed clearing requires a prior forest bound')
    last = forests[-1]
    stable = [r for r in rows if r['control_state'] == 'stable_forest']
    _require(len(stable) >= 3, 'Control needs at least three clear reviewed dates')
    result.update({'decision_status': 'accepted', 'evaluation_approved': True,
        'observed_change_interval': {'lower_bound_exclusive': last['date'],
            'upper_bound_inclusive': first['date'],
            'width_days': (_day(first['date'])-_day(last['date'])).days,
            'persistence_scene_ids': [r['scene_id'] for r in clearings[1:]],
            'interpretation': 'Inspected local observation interval, not exact felling time'},
        'controls': {'supervised_pixels': controls, 'clear_observation_count': len(stable),
                     'stable_scene_ids': [r['scene_id'] for r in stable]},
        'stable_pair': select_stable_pair(inspected, pair)})
    return result
