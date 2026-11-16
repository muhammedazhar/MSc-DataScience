#!/usr/bin/env python3
"""Independently verify a broad validation support audit and fixed shortlist.

Only small frozen QA masks and vector metadata are read. Satellite bands,
forest-model predictions and new image reviews are outside this check.
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
STRATA = ('36UXA_time-dependent', '36UYA_Summer_time-dependent',
          '36UYA_Spring_time-dependent')
PASSED = 'passed_automatic_screen_pending_visual_review'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 32*2**20, 'Missing or oversized audit evidence')
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite audit evidence')))


def records(path, maximum=20000):
    with Path(path).open('rb') as stream:
        for index in range(maximum+1):
            line = stream.readline(512*1024+1)
            if not line:
                return
            require(index < maximum and 0 < len(line) <= 512*1024 and line.strip(), 'Audit JSONL bounds exceeded')
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite audit evidence')))
            require(isinstance(row, dict), 'Audit JSONL records must be objects')
            yield row


def within(base, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'Relative evidence path required')
    base = Path(base).resolve(); path = (base/relative).resolve()
    require(path.is_relative_to(base) and path.is_file(), 'Audit evidence path is missing or escaped')
    return path


def reconstruct_support(projected_anchor, grid, published):
    """Burn only the anchor, intersect published temporal loss, erode independently.

    The worker computes its temporal mask from catalogue dates. This verifier
    instead reads the frozen published sixth QA plane. Erosion is reconstructed
    with nine NumPy slices, without calling the worker or SciPy implementation.
    """
    import numpy as np
    from affine import Affine
    from rasterio.features import rasterize

    require((grid['width'], grid['height']) == (64, 64), 'Frozen core must remain 64x64')
    values = np.asarray(published)
    require(values.shape == (6, 64, 64) and values.dtype == np.dtype('uint8')
            and np.isin(values, (0, 1)).all(), 'Published QA must retain six binary uint8 planes')
    anchor = rasterize([(projected_anchor, 1)], out_shape=(64, 64),
                       transform=Affine(*grid['transform']), fill=0,
                       dtype='uint8', all_touched=False).astype(bool)
    temporal = values[5].astype(bool)
    target = anchor & temporal
    def eroded(values):
        result = np.zeros((64, 64), dtype=bool)
        result[1:-1, 1:-1] = np.logical_and.reduce([
            values[1+dy:63+dy, 1+dx:63+dx]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        return result
    interior = eroded(target)
    unusable = np.any(values[:5].astype(bool), axis=0)
    return {'anchor_pixels': int(anchor.sum()), 'anchor_interior_pixels': int(eroded(anchor).sum()),
            'temporal_union_pixels': int(temporal.sum()),
            'target_pixels': int(target.sum()), 'positive_interior_pixels': int(interior.sum()),
            'temporal_clipped_pixels': int(anchor.sum()-target.sum()),
            'interior_unusable_pixels': int((interior & unusable).sum()),
            'target_mask_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
            'positive_interior_sha256': hashlib.sha256(interior.tobytes()).hexdigest()}


def independent_gate_flags(row, prior_events, prior_groups):
    """Re-derive eligibility from measured evidence, ignoring claimed decisions."""
    return {
        'automatic_pass': row['qa_status'] == PASSED,
        'endpoint_crop_clear': row['endpoint_crop_unusable_pixels'] == [0, 0],
        'endpoint_target_clear': row['endpoint_target_unusable_pixels'] == [0, 0],
        'minimum_interior': type(row['positive_interior_pixels']) is int and row['positive_interior_pixels'] >= 4,
        'interior_clear': row['interior_unusable_pixels'] == 0,
        'footprint_isolated': row['footprint_isolated'] is True,
        'later_usable_scene': bool(row['later_usable_scene_ids']),
        'source_stratum': row['stratum'] in STRATA,
        'fresh_event': row['event_id'] not in prior_events,
        'fresh_group': row['group_id'] not in prior_groups,
    }


def independent_shortlist(rows, prior_events, prior_groups):
    """Choose all primaries before alternates using only predeclared fields."""
    eligible = [row for row in rows if all(independent_gate_flags(row, prior_events, prior_groups).values())]
    eligible.sort(key=lambda row: (row['group_id'], row['event_id'],
                                  row['post_date'] != row['annotation_date'],
                                  row['interval_days'], row['pair_id']))
    used = set(); selected = []; missing = []
    for role in ('primary', 'alternate'):
        for stratum in STRATA:
            available = [row for row in eligible if row['stratum'] == stratum and row['group_id'] not in used]
            if not available:
                missing.append((stratum, role))
                continue
            chosen = available[0]
            used.add(chosen['group_id'])
            selected.append({'event_id': chosen['event_id'], 'pair_id': chosen['pair_id'],
                             'group_id': chosen['group_id'], 'stratum': stratum,
                             'role': role, 'rank': len(selected)+1})
    return selected, missing


def independent_reasons(row, prior_events, prior_groups):
    flags = independent_gate_flags(row, prior_events, prior_groups)
    mapping = (('automatic_pass', 'not_automatic_quality_pass'),
               ('minimum_interior', 'anchor_interior_below_four_pixels'),
               ('endpoint_crop_clear', 'endpoint_crop_qa_flags'),
               ('endpoint_target_clear', 'endpoint_target_qa_flags'),
               ('interior_clear', 'actual_interior_qa_flags'),
               ('footprint_isolated', 'full_feature_footprint_not_isolated'),
               ('later_usable_scene', 'no_later_metadata_usable_observation'),
               ('source_stratum', 'source_stratum_missing_or_ambiguous'),
               ('fresh_event', 'previously_rejected_event'),
               ('fresh_group', 'previously_reviewed_geographic_group'))
    return [reason for key, reason in mapping if not flags[key]]


def count_coverage(rows):
    supported = [row for row in rows if row['positive_interior_pixels'] >= 4]
    return {'anchor_pair_count': len(rows), 'pair_count': len({row['pair_id'] for row in rows}),
            'event_count': len({row['event_id'] for row in rows}),
            'group_count': len({row['group_id'] for row in rows}),
            'anchor_pairs_with_interior_at_least_four': len(supported),
            'events_with_interior_at_least_four': len({row['event_id'] for row in supported})}


def compare(actual, expected, label):
    """Strict JSON reconstruction: do not equate booleans with integer counts."""
    require(type(actual) is type(expected), 'Evidence type mismatch: '+label)
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Evidence key coverage mismatch: '+label)
        for key, value in expected.items():
            compare(actual[key], value, label+'.'+key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'Evidence list coverage mismatch: '+label)
        for index, value in enumerate(expected):
            compare(actual[index], value, label+f'[{index}]')
    else:
        require(actual == expected, 'Evidence value mismatch: '+label)


def validate_protocol(protocol):
    require(protocol['schema_version'] == 2
            and protocol['status'] == 'predeclared_before_geometry_audit_and_new_imagery',
            'The fixed broader support protocol is required')
    rule = protocol['rule']
    expected = {'erosion_pixels': 1, 'minimum_interior_pixels': 4,
        'require_automatic_pass': True, 'require_zero_endpoint_crop_and_target_qa': True,
        'require_zero_actual_interior_qa': True, 'require_full_footprint_isolation': True,
        'require_later_metadata_usable_scene': True, 'followup_window_days': 45,
        'followup_same_year': True, 'exclude_prior_events': True, 'exclude_prior_groups': True,
        'source_strata': list(STRATA), 'primaries_per_stratum': 1,
        'alternates_per_stratum_at_most': 1, 'maximum_shortlist': 6,
        'ranking': ['group_id', 'event_id', 'post_date_not_annotation_date', 'interval_days', 'pair_id'],
        'ranking_uses_ndvi_or_model_scores': False, 'shortlist_approves_labels': False,
        'new_imagery_reviewed': False, 'no_fallback_when_slots_missing': True}
    for key, value in expected.items():
        compare(rule.get(key), value, 'Protocol '+key)
    limits = protocol['resources']
    require(limits['workers'] == limits['native_threads'] == 1 and limits['gpu_used'] is False
            and limits['nice_increment'] == 10 and limits['worker_rss_mib'] == 512
            and limits['available_memory_reserve_gib'] == 1.5
            and limits['max_seconds_per_worker'] <= 300,
            'Frozen Mac limits changed')


def verify_resources(metadata):
    resource = metadata['resources']
    require(resource['workers'] == resource['native_thread_limit'] == 1
            and resource['gpu_used'] is False and resource['nice_increment'] == 10
            and 128 <= resource['worker_limit_mib'] <= 512
            and resource['minimum_available_gib_limit'] >= 1.5
            and 0 < resource['runtime_limit_seconds'] <= 300, 'Support worker resource guard changed')
    numbers = [resource[key] for key in ('worker_peak_rss_mib_sampled', 'worker_high_water_rss_mib',
                'minimum_available_gib_sampled', 'elapsed_seconds')]
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in numbers),
            'Support worker resource measurements must be finite')
    peak = max(numbers[:2])
    require(peak <= resource['worker_limit_mib']
            and numbers[2] >= resource['minimum_available_gib_limit']
            and numbers[3] <= resource['runtime_limit_seconds'], 'Support worker exceeded resource limits')
    return {'peak_worker_mib': peak, 'minimum_available_gib': numbers[2], 'elapsed_seconds': numbers[3]}


def keyed(rows, name):
    result = {}
    for row in rows:
        require(row[name] not in result, 'Repeated frozen '+name)
        result[row[name]] = row
    return result


def verify(args, output):
    import numpy as np
    import rasterio
    from affine import Affine
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as project_geometry

    require('torch' not in sys.modules, 'Independent support verification must not import Torch')
    protocol_path = args.protocol.resolve(); directory = args.run.resolve()
    protocol, metadata = read_json(protocol_path), read_json(directory/'metadata.json')
    validate_protocol(protocol)
    audit, selection = read_json(directory/'audit.json'), read_json(directory/'selection.json')
    protocol_sha = digest(protocol_path)
    require(metadata['execution_status'] == 'completed' and metadata['mode'] == 'validation_support_pool',
            'A completed support-pool run is required')
    require(metadata['protocol_sha256'] == audit['protocol_sha256'] == selection['protocol_sha256'] == protocol_sha,
            'The predeclared protocol binding changed')
    require(set(metadata['artifact_sha256']) == {'audit.json', 'selection.json', 'report.md'},
            'Support audit artifact coverage changed')
    for name, checksum in metadata['artifact_sha256'].items():
        require(digest(within(directory, name)) == checksum, 'Support audit artifact changed: '+name)
    sources = {}; source_hashes = {str(protocol_path.relative_to(ROOT)): protocol_sha}
    for name, receipt in protocol['source_files'].items():
        path = within(ROOT, receipt['path'])
        require(digest(path) == receipt['sha256'], 'Frozen audit input changed: '+name)
        sources[name] = path; source_hashes[receipt['path']] = receipt['sha256']
    for name, checksum in protocol['immutable_sources_sha256'].items():
        require(digest(within(ROOT/'Solutions', name)) == checksum, 'Archived implementation changed: '+name)
    expected_implementations = ('audit_validation_support_pool.py', 'audit_validation_geometry.py',
        'prepare_validation_review.py', 'select_validation_cases.py', 'assess_pixel_quality.py', 'validate_sample.py')
    implementation = {name: digest(within(ROOT/'Solutions', name)) for name in expected_implementations}
    compare(metadata['implementation_sha256'], implementation, 'Run source implementations')
    compare(audit['implementation_sha256'], implementation, 'Audit source implementations')
    resources = verify_resources(metadata)
    summary = read_json(sources['quality_summary'])
    require(summary['processing_errors'] == [] and summary['checks']
            and all(value is True for value in summary['checks'].values())
            and summary['config_id'] == protocol['qa_policy_id']
            and summary['candidate_manifest_sha256'] == digest(sources['pairs'])
            and summary['annotation_catalogue_sha256'] == digest(sources['catalogue'])
            and summary['audit_metadata_sha256'] == digest(sources['audit_metadata']),
            'The frozen quality publication no longer binds its audit inputs')
    all_pairs = keyed(records(sources['pairs']), 'pair_id')
    candidates = [row for row in all_pairs.values() if row['split'] == 'validation']
    quality = keyed(records(sources['quality']), 'pair_id')
    quality = {pid: row for pid, row in quality.items() if row['split'] == 'validation'}
    indices = keyed(records(sources['quality_index']), 'pair_id')
    indices = {pid: row for pid, row in indices.items() if row['split'] == 'validation'}
    require(set(quality) == set(indices) <= {row['pair_id'] for row in candidates},
            'All screened validation pairs must have exact candidate/index coverage')
    require(0 < len(candidates) <= protocol['resources']['maximum_validation_pairs'], 'Frozen validation pair bound exceeded')
    catalogue, splits = read_json(sources['catalogue']), read_json(sources['splits'])
    features = keyed(catalogue['features'], 'id')
    annotations = keyed(records(sources['annotations']), 'event_id')
    require(set(features) == set(annotations), 'Frozen annotation/catalogue coverage differs')
    assignment = {}
    for group in splits['groups']:
        for event_id in group['event_ids']:
            require(event_id not in assignment, 'An event belongs to multiple frozen groups')
            assignment[event_id] = (group['group_id'], group['split'])
    require(set(assignment) == set(features), 'Frozen group/catalogue coverage differs')
    for event_id, feature in features.items():
        prop = feature['properties']; annotation = annotations[event_id]
        require(assignment[event_id] == (prop['group_id'], prop['split'])
                and all(prop[key] == annotation[key] for key in
                        ('group_id', 'split', 'tile', 'observation_date', 'temporal_source_ids')),
                'Frozen event/group/date/source assignment changed')
    training = list(records(sources['training_manifest'], maximum=13))
    training_plan = read_json(ROOT/'Docs/Reviews/PixelPilotV1/experiment_plan.json')
    require(len(training) == 13 and all(row['split'] == 'train' for row in training)
            and digest(sources['training_manifest']) == training_plan['selection']['manifest_sha256']
            and digest(sources['training_metadata']) == training_plan['selection']['dataset_metadata_sha256'],
            'The exact trained thirteen-grid isolation reference changed')
    prior = read_json(sources['prior_verification']); prior_reviews = read_json(sources['prior_reviews'])
    require(prior['verification_status'] == 'passed' and prior['accepted_loss_case_count'] == 0
            and prior['model_executions'] == 0 and prior['reviews_sha256'] == digest(sources['prior_reviews']),
            'The prior rejected-review evidence changed')
    prior_events = {row['event_id'] for row in prior['review_dispositions']}
    prior_groups = {row['group_id'] for row in prior['review_dispositions']}
    require(prior_events == {row['event_id'] for row in prior_reviews['reviews']}, 'Prior reviewed event coverage differs')
    scenes = list(records(sources['scenes']))
    barriers = [feature['properties']['group_footprint_bounds'] for feature in features.values()
                if feature['properties']['split'] in ('train', 'test')]

    def bounds(grid):
        require(grid['crs'] == splits['crs'] == 'EPSG:32636'
                and (grid['width'], grid['height']) == (64, 64), 'Unsupported frozen validation grid')
        west, south, east, north = grid['bounds']
        require(all(type(value) in (int, float) and math.isfinite(value) for value in grid['bounds'])
                and east-west == north-south == 640
                and grid['transform'] == [10., 0., west, 0., -10., north], 'Frozen grid/transform changed')
        return west, south, east, north

    def overlapping(a, b):
        return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]

    train_bounds = [bounds(row['grid']) for row in training]
    source_crs = (catalogue.get('crs') or {}).get('properties', {}).get('name', 'EPSG:4326')
    transformer = Transformer.from_crs(source_crs, splits['crs'], always_xy=True).transform
    projected = {}
    rebuilt = []; masks_seen = set()
    identity = ('pair_id', 'group_id', 'split', 'tile', 'pre_date', 'post_date', 'anchor_event_ids')
    with rasterio.Env(GDAL_CACHEMAX=16*2**20, GDAL_NUM_THREADS='1'):
        for pair_id in sorted(quality):
            pair, qa, index = all_pairs[pair_id], quality[pair_id], indices[pair_id]
            require(all(pair[key] == qa[key] == index[key] for key in identity)
                    and pair['split'] == 'validation' and pair['status'] == 'candidate_for_pixel_review'
                    and not pair['blocking_reasons'] and qa['config_id'] == index['config_id'] == protocol['qa_policy_id']
                    and qa['quality_status'] == index['quality_status']
                    and qa['training_approved'] is False and index['training_approved'] is False,
                    'Exact frozen validation QA/index/candidate identity changed')
            require(pair['anchor_event_ids'] and len(pair['anchor_event_ids']) == len(set(pair['anchor_event_ids'])),
                    'Frozen validation anchors must remain unique and nonempty')
            core = bounds(pair['grid'])
            isolated = (not any(overlapping(core, other) for other in barriers)
                        and not any(overlapping(core, other) for other in train_bounds)
                        and pair['group_id'] not in {row['group_id'] for row in training})
            mask_name = index['batch_path']+'/'+index['qa_mask_file']; mask_path = within(ROOT, mask_name)
            source_hashes[mask_name] = digest(mask_path); masks_seen.add(mask_name)
            with rasterio.open(mask_path) as source:
                require(source.shape == (64, 64) and source.count == 6 and source.dtypes == ('uint8',)*6
                        and source.transform == Affine(*pair['grid']['transform'])
                        and source.crs == rasterio.crs.CRS.from_user_input(pair['grid']['crs']),
                        'Frozen QA TIFF grid/type changed')
                published = source.read()
            require(np.isin(published, (0, 1)).all(), 'Published QA planes must be binary')
            unusable = np.any(published[:5].astype(bool), axis=0); temporal = published[5].astype(bool)
            require(int(temporal.sum()) == qa['target_metrics']['region_pixels']
                    and int(unusable.sum()) == qa['crop_metrics']['unusable_pixels']
                    and int((unusable & temporal).sum()) == qa['target_metrics']['unusable_pixels'],
                    'Actual published QA counts disagree with frozen metadata')
            for event_id in pair['anchor_event_ids']:
                feature = features[event_id]; prop = feature['properties']; observed = prop['observation_date']
                before, after, observation = map(date.fromisoformat, (pair['pre_date'], pair['post_date'], observed))
                require((prop['group_id'], prop['split'], prop['tile']) == (pair['group_id'], 'validation', pair['tile'])
                        and before < observation <= after and 5 <= (after-before).days <= 30
                        and pair['interval_days'] == (after-before).days,
                        'Frozen pair does not bracket its exact annotation/group')
                if event_id not in projected:
                    projected[event_id] = project_geometry(transformer, shape(feature['geometry']))
                support = reconstruct_support(projected[event_id], pair['grid'], published)
                followups = sorted((scene for scene in scenes
                    if scene['tile'] == pair['tile'] and scene.get('metadata_usable') is True
                    and after < date.fromisoformat(scene['acquisition_date']) <= observation+timedelta(days=45)
                    and date.fromisoformat(scene['acquisition_date']).year == observation.year),
                    key=lambda scene: (scene['acquisition_date'], scene['scene_id']))
                strata = [name for name in STRATA if name in prop['temporal_source_ids']]
                row = {'event_id': event_id, 'pair_id': pair_id, 'group_id': pair['group_id'],
                    'stratum': strata[0] if len(strata) == 1 else None, 'split': 'validation',
                    'annotation_date': observed, 'pre_date': pair['pre_date'], 'post_date': pair['post_date'],
                    'interval_days': pair['interval_days'], 'grid': pair['grid'], 'qa_policy_id': protocol['qa_policy_id'],
                    'qa_status': qa['quality_status'],
                    'endpoint_crop_unusable_pixels': [qa['per_date'][side]['crop']['unusable_pixels'] for side in ('pre', 'post')],
                    'endpoint_target_unusable_pixels': [qa['per_date'][side]['target']['unusable_pixels'] for side in ('pre', 'post')],
                    'footprint_isolated': isolated, 'later_usable_scene_ids': [scene['scene_id'] for scene in followups],
                    'published_qa_file': mask_name, 'published_qa_sha256': source_hashes[mask_name],
                    **support, 'training_approved': False, 'evaluation_approved': False}
                row['eligibility_reasons'] = independent_reasons(row, prior_events, prior_groups)
                row['eligible_for_fresh_review_shortlist'] = not row['eligibility_reasons']
                rebuilt.append(row)
    require(0 < len(rebuilt) <= protocol['resources']['maximum_anchor_records'], 'Frozen anchor record bound exceeded')
    rebuilt.sort(key=lambda row: (row['event_id'], row['pair_id']))
    compare(audit['records'], rebuilt, 'Every independently reconstructed anchor record')
    screened = set(quality)
    unscreened = [{'pair_id': pair['pair_id'], 'group_id': pair['group_id'], 'anchor_event_ids': pair['anchor_event_ids'],
                   'status': pair['status'], 'blocking_reasons': pair['blocking_reasons'],
                   'disposition': 'no_completed_published_qa', 'positive_interior_pixels': None}
                  for pair in candidates if pair['pair_id'] not in screened]
    expected_counts = {'coverage': count_coverage(rebuilt),
        'by_stratum': {name: count_coverage([row for row in rebuilt if row['stratum'] == name]) for name in STRATA},
        'eligible_before_history': count_coverage([row for row in rebuilt if not independent_reasons(row, set(), set())]),
        'eligible_after_history': count_coverage([row for row in rebuilt if row['eligible_for_fresh_review_shortlist']]),
        'quality_status_counts': dict(Counter(row['quality_status'] for row in quality.values())),
        'exclusion_reason_counts': dict(Counter(reason for row in rebuilt for reason in row['eligibility_reasons'])),
        'validation_candidate_pair_count': len(candidates), 'unscreened_candidate_pair_count': len(unscreened),
        'prior_rejected_event_ids': sorted(prior_events), 'prior_reviewed_group_ids': sorted(prior_groups),
        'unscreened_candidates': unscreened, 'source_sha256': source_hashes}
    for key, value in expected_counts.items():
        compare(audit[key], value, 'Audit '+key)
    for key, value in (('source_jp2_pixel_reads', 0), ('model_executions', 0),
                       ('new_imagery_reviewed', 0), ('labels_approved', 0), ('qa_mask_files_read', len(masks_seen))):
        compare(audit[key], value, 'Audit counter '+key)
        if key != 'qa_mask_files_read':
            compare(metadata[key], value, 'Worker counter '+key)
    compare(metadata['coverage'], expected_counts['coverage'], 'Worker coverage')
    compare(metadata['eligible_after_history'], expected_counts['eligible_after_history'], 'Worker fresh eligibility')
    chosen, missing = independent_shortlist(rebuilt, prior_events, prior_groups)
    by_identity = {(row['event_id'], row['pair_id']): row for row in rebuilt}
    fields = ('event_id', 'pair_id', 'group_id', 'stratum', 'split', 'pre_date', 'post_date',
              'interval_days', 'grid', 'annotation_date', 'positive_interior_pixels',
              'anchor_pixels', 'anchor_interior_pixels', 'target_pixels', 'temporal_union_pixels',
              'temporal_clipped_pixels', 'target_mask_sha256', 'positive_interior_sha256', 'qa_policy_id')
    expected_candidates = []
    for item in chosen:
        row = by_identity[(item['event_id'], item['pair_id'])]
        expected_candidates.append({**{key: row[key] for key in fields}, 'role': item['role'], 'rank': item['rank'],
            'observation_date': row['annotation_date'], 'training_approved': False, 'evaluation_approved': False,
            'visual_review': 'unreviewed', 'temporal_label_review': 'unreviewed'})
    eligible = [row for row in rebuilt if row['eligible_for_fresh_review_shortlist']]
    expected_selection = {'schema_version': 2, 'scope': protocol['scope'], 'rule': protocol['rule'],
        'protocol_sha256': protocol_sha, 'source_audit_sha256': digest(directory/'audit.json'),
        'selection_uses_model_predictions': False, 'source_jp2_pixel_reads': 0,
        'published_qa_masks_read': len(masks_seen), 'model_executions': 0,
        'training_approved': False, 'evaluation_approved': False,
        'candidates': expected_candidates,
        'missing_slots': [{'stratum': stratum, 'role': role,
                          'reason': 'no_eligible_unreviewed_unused_geographic_group'} for stratum, role in missing],
        'eligible_anchor_pair_count': len(eligible),
        'eligible_event_count': len({row['event_id'] for row in eligible}),
        'eligible_group_count': len({row['group_id'] for row in eligible})}
    compare(selection, expected_selection, 'Complete fixed shortlist')
    compare(metadata['shortlist_count'], len(chosen), 'Worker shortlist count')
    require(all(digest(within(ROOT, name)) == checksum for name, checksum in source_hashes.items())
            and all(digest(within(ROOT/'Solutions', name)) == checksum for name, checksum in implementation.items()),
            'Verified source evidence changed during independent reconstruction')
    require('torch' not in sys.modules, 'Independent support verifier unexpectedly imported Torch')
    diagnostics = {}
    for gate in ('endpoint_crop_clear', 'fresh_group'):
        passing = []
        for row in rebuilt:
            flags = independent_gate_flags(row, prior_events, prior_groups)
            if all(value for key, value in flags.items() if key != gate):
                passing.append(row)
        diagnostics['if_only_'+gate+'_omitted'] = count_coverage(passing)
    result = {'verification_status': 'passed', 'scope': protocol['scope'],
        'audit_metadata_sha256': digest(directory/'metadata.json'), 'audit_sha256': digest(directory/'audit.json'),
        'selection_sha256': digest(directory/'selection.json'), 'protocol_sha256': protocol_sha,
        **expected_counts, 'shortlist_count': len(chosen), 'missing_slot_count': len(missing),
        'diagnostic_gate_combinations': diagnostics,
        'run_resources': resources, 'source_jp2_pixel_reads': 0, 'qa_mask_files_read': len(masks_seen),
        'model_executions': 0, 'new_imagery_reviewed': 0, 'labels_approved': 0, 'torch_imported': False,
        'verification_implementation_sha256': digest(Path(__file__)),
        'limitations': ['Geometry and mask reconstruction do not approve visual or temporal labels.',
                        'Frozen QA masks and metadata are checked; cloud/QA models and satellite bands are not re-executed.',
                        'The filtered shortlist remains a convenience sample, not representative accuracy evidence.']}
    write_report(output/'report.md', result)
    return result


def write_report(path, result):
    coverage = result['coverage']; fresh = result['eligible_after_history']
    lines = ['# Independently verified broad validation support audit', '',
        f"Verified {coverage['anchor_pair_count']} anchor records from {coverage['pair_count']} QA-screened pairs, "
        f"plus {result['unscreened_candidate_pair_count']} retained unscreened candidates.",
        f"Reconstructed actual support, clipping, QA, complete exclusions and the fixed {result['shortlist_count']}-case shortlist.", '',
        f"Fresh-review eligible: {fresh['anchor_pair_count']} anchor-pairs / {fresh['event_count']} events / {fresh['group_count']} groups.",
        f"Missing shortlist slots: {result['missing_slot_count']}; no fallback or label approval.", '',
        'Anchor rasterization and nine-slice NumPy erosion were reconstructed separately from the worker.',
        'Temporal targets came from the frozen published sixth QA plane, not the worker catalogue-mask routine.',
        'The verifier independently rederived the geographic/history/quality gates and complete deterministic shortlist.',
        'No Torch, satellite JP2 pixels, model execution, new image review or approval.', '',
        *result['limitations']]
    lines += ['', '## Diagnostic gate combinations', '',
              'These counts omit one gate at a time for diagnosis only. The protocol, shortlist and approvals remain unchanged.', '',
              '| Only omitted gate | Anchor-pairs | Events | Groups |', '| --- | ---: | ---: | ---: |']
    for key, counts in result['diagnostic_gate_combinations'].items():
        lines.append(f"| {key} | {counts['anchor_pair_count']} | {counts['event_count']} | {counts['group_count']} |")
    Path(path).write_text('\n'.join(lines).rstrip()+'\n')


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationSupportV2/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Use 128–512 MiB RSS, >=1.5 GiB reserve and <=300 seconds')
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        os.nice(10)
        result = verify(args, args.worker_output)
        import resource
        result['worker_high_water_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                                   (2**20 if sys.platform == 'darwin' else 1024), 2)
        (args.worker_output/'metadata.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    else:
        run_guarded(args, script_path=__file__, operation='validation-support-independent-verification',
                    gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
