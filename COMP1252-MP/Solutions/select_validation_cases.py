#!/usr/bin/env python3
"""Select a bounded validation review roster from frozen metadata, without pixels.

This is a deliberately curated review pilot, not a population-representative
sample. Geometry thresholds are search aids and do not prove eroded support or
visual labels. No image arrays, model predictions, NDVI rankings or test imagery
are read. All source validation-event dispositions remain in the output.
"""

import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
POLICY = 'fe2472ec030f466bebbb9dc2cb033ed6175de489bcf022f650081dd8e9347560'
STRATA = ('36UXA_time-dependent', '36UYA_Summer_time-dependent',
          '36UYA_Spring_time-dependent')
PASSED = 'passed_automatic_screen_pending_visual_review'
RULE = {
    'selection_version': 1,
    'strata_in_order': list(STRATA),
    'selection_order': 'one primary per stratum, then at most one alternate per stratum',
    'ranking': ['group_id', 'event_id', 'existing_top_candidate_rank', 'pair_id'],
    'all_roster_groups_unique': True,
    'required_split': 'validation',
    'quality_status': PASSED,
    'both_endpoint_core_and_target_unusable_pixels': 0,
    'minimum_union_target_pixels_search_aid': 4,
    'minimum_anchor_area_m2_search_aid': 1600,
    'minimum_anchor_width_height_m_search_aid': 40,
    'require_later_local_scene_in_same_year_45_day_window': True,
    'reject_touching_or_overlapping_train_test_conservative_footprints': True,
    'reject_touching_or_overlapping_actual_training_feature_grids': True,
    'unknown_crs_handling': 'reject; do not transform in metadata-only selection',
    'geometry_does_not_guarantee_eroded_interior': True,
    'minimum_eroded_interior_pixels_deferred_to_guarded_preparation': 4,
    'no_ndvi_or_model_output_ranking': True,
    'maximum_cases': 6,
    'missing_alternate_handling': 'retain missing slot; never relax filters or reuse a selected group',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def json_file(path, maximum=16*2**20):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= maximum, 'Missing or oversized JSON source')
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def records(path, maximum=20000):
    with Path(path).open('rb') as stream:
        for index in range(maximum+1):
            line = stream.readline(512*1024+1)
            if not line:
                return
            require(index < maximum and 0 < len(line) <= 512*1024 and line.strip(), 'JSONL size/coverage bound exceeded')
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
            require(isinstance(row, dict), 'Metadata records must be objects')
            yield row


def bounds(value):
    require(isinstance(value, (list, tuple)) and len(value) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in value)
            and value[0] < value[2] and value[1] < value[3], 'Invalid projected bounds')
    return tuple(value)


def touches(a, b):
    a, b = bounds(a), bounds(b)
    return min(a[2], b[2]) >= max(a[0], b[0]) and min(a[3], b[3]) >= max(a[1], b[1])


def grid_bounds(grid, crs):
    require(isinstance(grid, dict) and grid.get('crs') == crs, 'Unknown or mismatched grid CRS')
    require(grid.get('width') == grid.get('height') == 64, 'Feature grid must remain 64x64')
    b = bounds(grid['bounds']); transform = grid.get('transform')
    require(transform == [10., 0., b[0], 0., -10., b[3]]
            and b[2]-b[0] == b[3]-b[1] == 640, 'Frozen feature-grid geometry changed')
    return b


def choose_candidates(events, pairs, quality, properties, scenes, training, crs):
    """Return deterministic reviewed-case proposals plus all event dispositions.

    Inputs are metadata dictionaries/lists. Every passed candidate is validated
    against exact frozen joins, masks' numeric receipts and complete feature
    footprints. Selection grants no label or validation approval.
    """
    require(crs == 'EPSG:32636', 'This metadata selector accepts only the frozen projected CRS')
    require(len(training) == 13 and all(t.get('split') == 'train' for t in training),
            'Exactly the 13 immutable training examples are required')
    training_bounds = [grid_bounds(t['grid'], crs) for t in training]
    barriers = [(p['event_id'], bounds(p['group_footprint_bounds']))
                for p in properties.values() if p['split'] in ('train', 'test')]
    eligible = []; dispositions = []
    validation_events = sorted((e for e in events if e['split'] == 'validation'), key=lambda e: e['event_id'])
    require(len({e['event_id'] for e in validation_events}) == len(validation_events), 'Duplicate validation queue event')
    for event in validation_events:
        identifier = event['event_id']; prop = properties[identifier]
        require(prop['split'] == event['split'] == 'validation'
                and prop['group_id'] == event['group_id'], 'Frozen annotation/queue identity changed')
        sources = prop['temporal_source_ids']; matches = [s for s in STRATA if s in sources]
        reasons = []; considered = []
        if len(matches) != 1:
            reasons.append('source_stratum_missing_or_ambiguous')
        if not (prop['area_m2'] >= RULE['minimum_anchor_area_m2_search_aid']
                and min(prop['width_m'], prop['height_m']) >= RULE['minimum_anchor_width_height_m_search_aid']):
            reasons.append('anchor_geometry_below_declared_search_aid_threshold')
        if reasons:
            dispositions.append({'event_id': identifier, 'group_id': event['group_id'],
                                 'disposition': 'metadata_ineligible', 'reasons': reasons, 'considered_pairs': []})
            continue
        observed = date.fromisoformat(event['observation_date'])
        for rank, candidate in enumerate(event['top_candidates']):
            pid = candidate['pair_id']; pair = pairs[pid]; qa = quality[pid]
            require(pair['pair_id'] == qa['pair_id'] == pid and identifier in pair['anchor_event_ids']
                    and pair['split'] == qa['split'] == 'validation'
                    and pair['group_id'] == qa['group_id'] == event['group_id']
                    and pair['tile'] == qa['tile'] == event['tile'], 'Frozen pair/quality/event join changed')
            require(qa['config_id'] == POLICY and qa['training_approved'] is False,
                    'Quality policy or unapproved state changed')
            require(pair['pre_date'] == qa['pre_date'] == candidate['pre_date']
                    and pair['post_date'] == qa['post_date'] == candidate['post_date'], 'Frozen observation dates changed')
            before, after = date.fromisoformat(pair['pre_date']), date.fromisoformat(pair['post_date'])
            require(before < observed <= after and 5 <= (after-before).days <= 30,
                    'Candidate does not bracket the annotation observation')
            rejected = []
            if qa['quality_status'] != PASSED or candidate['quality_status'] != PASSED:
                rejected.append('not_passed_automatic_quality_screen')
            if any(qa['per_date'][d][region]['unusable_pixels'] != 0
                   for d in ('pre', 'post') for region in ('crop', 'target')):
                rejected.append('endpoint_core_or_target_quality_flags')
            if qa['target_metrics']['region_pixels'] < RULE['minimum_union_target_pixels_search_aid']:
                rejected.append('union_target_below_search_aid_count')
            later = sorted({s['acquisition_date'] for s in scenes if s['tile'] == pair['tile']
                            and after < date.fromisoformat(s['acquisition_date']) <= observed+timedelta(days=45)
                            and date.fromisoformat(s['acquisition_date']).year == observed.year})
            if not later:
                rejected.append('no_later_local_scene_for_persistence_review')
            try:
                footprint = grid_bounds(pair['grid'], crs)
            except ValueError:
                rejected.append('unknown_crs_or_changed_grid')
                footprint = None
            overlap_ids = []
            if footprint is not None:
                overlap_ids = [eid for eid, barrier in barriers if touches(footprint, barrier)]
                if overlap_ids:
                    rejected.append('train_or_test_conservative_footprint_overlap')
                if any(touches(footprint, b) for b in training_bounds):
                    rejected.append('actual_training_feature_footprint_overlap')
            considered.append({'pair_id': pid, 'existing_top_candidate_rank': rank,
                               'reasons': rejected, 'eligible': not rejected,
                               'overlapping_train_test_event_ids': overlap_ids})
            if rejected:
                continue
            paths = [qa['per_date'][d]['observation']['cache_path'] for d in ('pre', 'post')]
            require(all(isinstance(p, str) and not Path(p).is_absolute() and '..' not in Path(p).parts for p in paths),
                    'Observation cache paths must remain project-relative')
            eligible.append({'event_id': identifier, 'pair_id': pid, 'group_id': event['group_id'],
                             'stratum': matches[0], 'existing_top_candidate_rank': rank,
                             'split': 'validation', 'tile': pair['tile'],
                             'observation_date': event['observation_date'], 'pre_date': pair['pre_date'],
                             'post_date': pair['post_date'], 'grid': pair['grid'],
                             'search_aid_metrics': {'anchor_area_m2': prop['area_m2'],
                                 'anchor_width_m': prop['width_m'], 'anchor_height_m': prop['height_m'],
                                 'union_target_pixels': qa['target_metrics']['region_pixels'],
                                 'eroded_anchor_interior_pixels': None,
                                 'later_local_scene_dates': later},
                             'endpoint_cache_paths': paths, 'qa_policy_id': POLICY,
                             'validation_approved': False, 'training_approved': False})
        dispositions.append({'event_id': identifier, 'group_id': event['group_id'],
                             'disposition': 'metadata_eligible_pending_roster' if any(c['eligible'] for c in considered) else 'metadata_ineligible',
                             'reasons': [] if any(c['eligible'] for c in considered) else ['no_candidate_passed_metadata_rules'],
                             'considered_pairs': considered})
    eligible.sort(key=lambda r: (r['group_id'], r['event_id'], r['existing_top_candidate_rank'], r['pair_id']))
    selected = []; used_groups = set(); missing = []
    for role in ('primary', 'alternate'):
        for stratum in STRATA:
            options = [r for r in eligible if r['stratum'] == stratum and r['group_id'] not in used_groups]
            if options:
                chosen = {**options[0], 'role': role, 'rank': len(selected)+1}
                selected.append(chosen); used_groups.add(chosen['group_id'])
            else:
                missing.append({'role': role, 'stratum': stratum, 'reason': 'no_eligible_unused_geographic_group'})
    selected_events = {r['event_id'] for r in selected}
    for row in dispositions:
        if row['event_id'] in selected_events:
            row['disposition'] = 'selected_for_review_not_approved'
        elif row['disposition'] == 'metadata_eligible_pending_roster':
            row['disposition'] = 'not_selected_by_predeclared_bounded_roster'
    return {'rule': RULE, 'candidates': selected, 'missing_slots': missing,
            'eligible_pair_count': len(eligible), 'eligible_event_count': len({r['event_id'] for r in eligible}),
            'eligible_group_count': len({r['group_id'] for r in eligible}),
            'validation_queue_event_count': len(validation_events), 'dispositions': dispositions,
            'barrier_event_count': len(barriers), 'actual_training_grid_count': len(training)}


def worker(args, output):
    os.nice(10)
    audit, qa, training = args.audit.resolve(), args.qa.resolve(), args.training.resolve()
    sources = {'audit_metadata': audit/'metadata.json', 'splits': audit/'splits.json',
               'catalogue': audit/'annotations_clean.geojson', 'annotations': audit/'annotation_manifest.jsonl',
               'pairs': audit/'candidate_pairs.jsonl', 'scenes': audit/'scene_inventory.jsonl',
               'quality_summary': qa/'summary/metadata.json', 'quality': qa/'summary/quality_manifest.jsonl',
               'review_metadata': qa/'review/metadata.json', 'queue': qa/'review/review_queue.jsonl',
               'training_metadata': training/'metadata.json', 'training_manifest': training/'training_manifest.jsonl'}
    identities = {key: {'path': str(path), 'sha256': digest(path)} for key, path in sources.items()}
    audit_meta, split = json_file(sources['audit_metadata']), json_file(sources['splits'])
    require(all(audit_meta['checks'].get(k) is True for k in ('no_duplicate_event_ids',
            'no_cross_split_label_or_crop_overlap', 'all_candidate_intervals_5_to_30_days')), 'Frozen audit checks incomplete')
    summary, review_meta = json_file(sources['quality_summary']), json_file(sources['review_metadata'])
    require(summary['config_id'] == review_meta['config_id'] == POLICY
            and all(summary['checks'].get(k) is True for k in ('exact_candidate_coverage',
                'no_duplicate_pairs', 'single_frozen_quality_policy', 'splits_preserved')),
            'Frozen QA publication checks incomplete')
    require(review_meta['checks'].get('no_forest_model_or_dice_scores_used') is True
            and review_meta['checks'].get('frozen_splits_preserved') is True, 'Review queue provenance incomplete')
    require(review_meta['quality_manifest_sha256'] == identities['quality']['sha256']
            and review_meta['annotation_manifest_sha256'] == identities['annotations']['sha256'], 'Frozen review source hashes changed')
    require(summary['audit_metadata_sha256'] == identities['audit_metadata']['sha256']
            and summary['candidate_manifest_sha256'] == identities['pairs']['sha256']
            and summary['annotation_catalogue_sha256'] == identities['catalogue']['sha256'],
            'Completed QA no longer binds the frozen audit sources')
    catalogue = json_file(sources['catalogue'])
    props = {f['properties']['event_id']: f['properties'] for f in catalogue['features']}
    require(len(props) == len(catalogue['features']), 'Duplicate catalogue event identity')
    annotations = {r['event_id']: r for r in records(sources['annotations'])}
    require(set(props) == set(annotations), 'Catalogue/annotation coverage changed')
    assignments = {}
    for group in split['groups']:
        require(group['split'] in ('train', 'validation', 'test'), 'Unknown frozen split role')
        for eid in group['event_ids']:
            require(eid not in assignments, 'An event appears in multiple frozen geographic groups')
            assignments[eid] = (group['group_id'], group['split'])
    require(set(assignments) == set(props), 'Frozen split/catalogue event coverage changed')
    for eid, prop in props.items():
        require(all(prop[k] == annotations[eid][k] for k in ('group_id', 'split', 'tile', 'observation_date', 'temporal_source_ids')),
                'Catalogue/frozen annotation identity changed')
        require((prop['group_id'], prop['split']) == assignments[eid], 'Frozen geographic assignment changed')
    pair_counts, quality_counts = Counter(), Counter()
    pair_index = {}; quality_index = {}
    for row in records(sources['pairs']):
        pair_counts[row['split']] += 1
        if row['split'] == 'validation': pair_index[row['pair_id']] = row
    for row in records(sources['quality']):
        quality_counts[row['split']] += 1
        if row['split'] == 'validation': quality_index[row['pair_id']] = row
    require(sum(quality_counts.values()) == summary['pair_count'] == review_meta['input_quality_pair_count'],
            'Complete QA coverage changed')
    queue = list(records(sources['queue'])); scenes = list(records(sources['scenes']))
    result = choose_candidates(queue, pair_index, quality_index, props, scenes,
                               list(records(sources['training_manifest'])), split['crs'])
    require(len(queue) == review_meta['screened_anchor_event_count'], 'Review event coverage changed')
    training_meta = json_file(sources['training_metadata'])
    require(training_meta['qa_policy_id'] == POLICY and training_meta['accepted_example_count'] == 13,
            'Frozen training reference changed')
    for candidate in result['candidates']:
        candidate['endpoint_caches_present'] = all((ROOT/p).is_file() for p in candidate['endpoint_cache_paths'])
        require(candidate['endpoint_caches_present'], 'Selected endpoint cache is missing')
    require(all(digest(path) == identities[key]['sha256'] for key, path in sources.items()),
            'Metadata inputs changed during selection')
    result.update(schema_version=1, frozen_policy_id=POLICY, source_files=identities,
                  counts={'candidate_pairs_by_split': dict(pair_counts), 'quality_pairs_by_split': dict(quality_counts),
                          'review_events_by_split': dict(Counter(r['split'] for r in queue)),
                          'dispositions': dict(Counter(r['disposition'] for r in result['dispositions']))},
                  selection_uses_model_predictions=False, pixel_arrays_read=0,
                  model_executions=0, training_approved=False, validation_approved=False,
                  limitations=['Curated metadata-filtered pilot; not population-representative.',
                               'Geometry thresholds and union target counts are search aids, not reviewed labels or eroded-interior proof.',
                               'A later scene is only an opportunity for persistence review, not a clear-observation claim.',
                               'Stable/no-change validation pairs still require explicit image review and unknown-pixel masking.'])
    (output/'selection.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    import resource
    metadata = {'execution_status': 'completed', 'selection_sha256': digest(output/'selection.json'),
                'selected_case_count': len(result['candidates']), 'source_files': identities,
                'implementation_sha256': digest(Path(__file__)), 'pixel_arrays_read': 0,
                'model_executions': 0, 'worker_high_water_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                     (2**20 if sys.platform == 'darwin' else 1024), 2)}
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=ROOT/'Datasets/AnnotationAudit/v1')
    parser.add_argument('--qa', type=Path, default=ROOT/'Datasets/PixelQA/v1')
    parser.add_argument('--training', type=Path, default=ROOT/'Datasets/CuratedPilot/v1/dataset')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=120)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Use 128–512 MiB RSS, >=1.5 GiB reserve and <=300 seconds')
    for key in THREAD_VARIABLES: os.environ[key] = '1'
    if args.worker_output: worker(args, args.worker_output)
    else: run_guarded(args, script_path=__file__, operation='validation-selection', gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
