#!/usr/bin/env python3
"""Verify and load reviewed validation exports without executing a model.

This audit reads only the small exported TIFFs, prepared review artifacts and
completed observation caches. Source JP2 files are fingerprinted, never read.
The loader may be called by a guarded inference worker that already has Torch;
this module itself never imports Torch or performs model operations.
"""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MASK_NAMES = ('cloud', 'cloud_buffer', 'shadow_risk', 'haze_risk', 'invalid')
CHANNELS = tuple(f'{period}_{band}' for period in ('pre', 'post', 'difference')
                 for band in ('B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12', 'NDVI', 'NDMI'))
COUNTS = ('supervised_positive_pixels', 'supervised_stable_pixels', 'ignored_pixels')
PREPARED_ARTIFACTS = {'pair.png', 'timeline.png', 'target_mask.tif', 'positive_interior.tif',
                      'proposed_control.tif', 'catalogue_exclusion.tif', 'endpoint_unusable.tif',
                      'feature_valid.tif', 'input_features.tif'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 32*2**20, 'Missing or oversized JSON evidence')
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON evidence')))


def records(path, maximum=20000):
    with Path(path).open('rb') as stream:
        for index in range(maximum+1):
            line = stream.readline(512*1024+1)
            if not line:
                return
            require(index < maximum and 0 < len(line) <= 512*1024 and line.strip(), 'JSONL evidence bounds exceeded')
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSONL evidence')))
            require(isinstance(row, dict), 'JSONL evidence must contain objects')
            yield row


def within(base, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'Relative evidence path required')
    base = Path(base).resolve(); path = (base/relative).resolve()
    require(path.is_relative_to(base) and path.is_file(), 'Evidence path is missing or escaped its root')
    return path


def validate_record_role(row):
    """A stable-only ROI is the sole supported zero-positive validation kind."""
    require(row.get('split') == 'validation' and row.get('training_approved') is False
            and row.get('evaluation_approved') is True, 'Only explicitly approved validation data may be loaded')
    require(row.get('kind') in ('loss', 'stable_roi'), 'Unknown validation sample kind')
    for key in ('sample_id', 'event_id', 'group_id', 'frozen_parent_pair_id'):
        require(isinstance(row.get(key), str) and row[key], 'Validation sample identity is missing')
    prefix = 'loss-' if row['kind'] == 'loss' else 'stable-'
    require(row['sample_id'] == prefix+row['event_id'], 'Validation sample ID does not encode its reviewed role/event')
    for key in COUNTS:
        require(type(row.get(key)) is int and row[key] >= 0, 'Validation counts must be nonnegative integers')
    p, n, ignored = (row[key] for key in COUNTS)
    require(p+n+ignored == 4096 and n >= 64
            and ((row['kind'] == 'loss' and p >= 4) or (row['kind'] == 'stable_roi' and p == 0)),
            'Validation role and reviewed class support disagree')
    pair = row.get('pair', {})
    require(pair.get('split') == row['split'] and pair.get('group_id') == row['group_id']
            and pair.get('grid') == row.get('grid')
            and pair.get('pre_date') == row.get('pre_date') and pair.get('post_date') == row.get('post_date'),
            'Validation sample and pair identity disagree')
    if row['kind'] == 'stable_roi':
        require(pair.get('kind') == 'derived_pre_event_stable_roi'
                and pair.get('not_an_original_frozen_candidate') is True
                and pair.get('frozen_parent_pair_id') == row['frozen_parent_pair_id'],
                'Zero-positive samples require explicit derived stable-ROI lineage')
    else:
        require(pair.get('pair_id') == row['frozen_parent_pair_id'], 'Loss sample must retain its frozen parent pair')


def validate_arrays(features, labels, valid, feature_valid):
    """Check signed arrays and return a finite model-input copy, preserving ROIs."""
    import numpy as np

    require(isinstance(features, np.ndarray) and features.shape == (27, 64, 64)
            and features.dtype == np.dtype('float32') and not np.isinf(features).any(),
            'Validation features must retain finite-or-NaN float32 signed channels')
    require(isinstance(labels, np.ndarray) and labels.shape == (64, 64) and labels.dtype == np.dtype('uint8')
            and np.isin(labels, (0, 1, 255)).all(), 'Validation labels must retain uint8 0/1/255')
    for name, mask in (('review validity', valid), ('feature validity', feature_valid)):
        require(isinstance(mask, np.ndarray) and mask.shape == labels.shape
                and mask.dtype in (np.dtype(bool), np.dtype('uint8')) and np.isin(mask, (0, 1)).all(),
                name+' must remain a binary 64x64 mask')
    reviewed = labels != 255
    require(np.array_equal(valid.astype(bool), reviewed), 'Validity must equal labels != 255 exactly')
    finite = np.isfinite(features).all(axis=0)
    require(np.array_equal(feature_valid.astype(bool), finite) and finite[reviewed].all(),
            'Reviewed features must be finite and feature validity must match actual inputs')
    require(np.array_equal(features[18:], features[9:18]-features[:9], equal_nan=True),
            'Signed post-minus-pre differences changed')
    for start in (0, 9):
        for left, right, index in ((3, 2, 7), (4, 5, 8)):
            a, b = features[start+left], features[start+right]
            denominator = a+b
            expected = np.divide(a-b, denominator, out=np.zeros_like(a),
                                 where=np.isfinite(denominator) & (denominator != 0))
            require(np.allclose(features[start+index][finite], expected[finite], rtol=0, atol=2e-6),
                    'Saved vegetation index signs/scales disagree with reflectance channels')
    model_input = features.copy()
    replacements = int((~np.isfinite(model_input)).sum())
    model_input[~np.isfinite(model_input)] = 0.
    require(np.isfinite(model_input).all() and np.array_equal(model_input[:, reviewed], features[:, reviewed]),
            'Input sanitization changed reviewed feature values')
    return model_input, replacements


def validate_cache_binding(actual, expected, policy):
    require(actual == expected and actual.get('config_id') == policy,
            'Observation cache metadata or frozen QA policy changed')
    key = canonical({'source': actual.get('source_id'), 'grid': actual.get('grid'), 'config_id': policy})
    require(actual.get('cache_key') == key and isinstance(actual.get('cache_path'), str)
            and Path(actual['cache_path']).name == key+'.npz', 'Observation cache source/grid identity changed')


def read_cache(expected, policy):
    """Bound compressed cache headers before loading small QA arrays."""
    import numpy as np

    path = within(ROOT, expected['cache_path'])
    require(path.stat().st_size <= 2*2**20, 'Observation cache compressed bound exceeded')
    shapes = {name: (64, 64) for name in (*MASK_NAMES, 'probability', 'product_cloud', 'ndvi')}
    shapes.update(rgb=(3, 64, 64), metadata=())
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(shapes)
                and set(archive.namelist()) == {name+'.npy' for name in shapes}
                and sum(item.file_size for item in archive.infolist()) <= 2*2**20,
                'Observation cache member coverage or allocation bound changed')
        for name, shape in shapes.items():
            with archive.open(name+'.npy') as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'Unsupported cached array header')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                actual_shape, _, dtype = reader(stream)
                require(actual_shape == shape and not dtype.hasobject, 'Observation cache shape/type changed')
                expected_dtype = 'bool' if name in (*MASK_NAMES, 'product_cloud') else 'float32'
                require((name == 'metadata' and dtype.kind == 'U' and dtype.itemsize <= 2**20)
                        or (name != 'metadata' and dtype == np.dtype(expected_dtype)), 'Observation cache dtype changed')
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        validate_cache_binding(metadata, expected, policy)
        masks = {name: archive[name].copy() for name in MASK_NAMES}
    return masks


def read_tif(path, grid, channels=False, nodata=None):
    import rasterio
    from affine import Affine

    with rasterio.open(path) as source:
        require(source.shape == (64, 64) and source.transform == Affine(*grid['transform'])
                and source.crs == rasterio.crs.CRS.from_user_input(grid['crs']), 'Evidence TIFF grid changed')
        if channels:
            require(source.count == 27 and source.dtypes == ('float32',)*27 and source.descriptions == CHANNELS,
                    'Feature dtype/count/channel order changed')
        else:
            require(source.count == 1 and source.dtypes == ('uint8',), 'Mask dtype/count changed')
            if nodata is not None:
                require(source.nodata == nodata, 'Unknown label nodata convention changed')
        return source.read() if channels else source.read(1)


def read_binary(path, grid):
    import numpy as np
    values = read_tif(path, grid)
    require(np.isin(values, (0, 1)).all(), 'Prepared QA/ROI masks must remain binary')
    return values.astype(bool)


def expected_derived_pair(case, decision):
    """Reconstruct the explicit lineage, never treating an empty target as stable."""
    from validation_review import select_stable_pair

    parent = case['pair']; derived = decision.get('stable_pair')
    inspected = {row['scene_id']: row for row in decision['observations']}
    require(derived is not None and derived == select_stable_pair(inspected, parent),
            'Derived stable pair differs from the earliest eligible reviewed pre-event combination')
    timeline = {row['scene_id']: row for row in case['timeline']}
    pair = deepcopy(parent)
    pair.update({key: derived[key] for key in ('pre_scene_id', 'post_scene_id', 'pre_date', 'post_date')})
    pair['pair_id'] = 'validation-stable-'+canonical({'parent_pair_id': parent['pair_id'], 'event_id': case['event_id'],
        'pre_scene_id': derived['pre_scene_id'], 'post_scene_id': derived['post_scene_id'], 'grid': parent['grid']})[:20]
    pair['interval_days'] = (date.fromisoformat(derived['post_date'])-date.fromisoformat(derived['pre_date'])).days
    for side in ('pre', 'post'):
        pair[side+'_path'] = timeline[derived[side+'_scene_id']]['scene_path']
    pair.update({'kind': 'derived_pre_event_stable_roi', 'frozen_parent_pair_id': parent['pair_id'],
                 'not_an_original_frozen_candidate': True,
                 'label_event_ids_by_post_date': [], 'status': 'derived_validation_pair_reviewed_stable_roi',
                 'pixel_quality': 'reviewed_control_roi_only', 'training_approved': False,
                 'label_interpretation': 'Reviewed stable control ROI only; unknown pixels ignored'})
    return pair


def _check_stats(stats):
    for name, expected in stats.items():
        path = within(ROOT, name); value = path.stat()
        require([value.st_size, value.st_mtime_ns] == expected, 'Frozen validation source fingerprint changed: '+name)


def load_and_verify(dataset, protocol):
    """Return original metadata, manifest records and finite NumPy input copies."""
    import numpy as np
    from scipy.ndimage import binary_erosion
    from prepare_validation_review import validate_validation_pair, validate_grid, catalogue_exclusion
    from prepare_curated_pilot import select_timeline
    from validation_review import validate_validation_review

    dataset, protocol_path = Path(dataset).resolve(), Path(protocol).resolve()
    metadata, policy = read_json(dataset/'metadata.json'), read_json(protocol_path)
    require(policy['status'] == 'predeclared_before_validation_predictions', 'Validation protocol is not predeclared')
    evaluation = policy['evaluation']
    require(evaluation['checkpoint_attempt'] == 1664 and evaluation['probability_threshold'] == .5
            and evaluation['retraining'] is False and evaluation['threshold_tuning'] is False
            and evaluation['best_checkpoint_selection'] is False, 'Frozen inference checkpoint/threshold protocol changed')
    for name, checksum in policy['immutable_sources_sha256'].items():
        require(re.fullmatch(r'[A-Za-z0-9_]+\.py', name) and digest(within(ROOT/'Solutions', name)) == checksum,
                'Immutable archived source changed: '+name)
    require(digest(within(ROOT, evaluation['checkpoint'])) == evaluation['weights_sha256']
            and digest(ROOT/'Docs/Reviews/PixelPilotV1/run/metadata.json') == evaluation['run_metadata_sha256'],
            'Frozen final checkpoint/model receipt changed')
    require(metadata.get('ready_for_partial_validation') is True and metadata.get('evaluation_approved') is True
            and metadata.get('training_approved') is False and metadata.get('model_predictions_read') is False
            and metadata.get('forest_model_training_run') is False
            and metadata.get('qa_policy_id') == policy['qa_policy_id'], 'Dataset role/label approval is unsafe')
    require(metadata['protocol_sha256'] == digest(protocol_path)
            and within(ROOT, metadata['protocol_file']) == protocol_path
            and metadata['model_weights_sha256'] == evaluation['weights_sha256'], 'Dataset protocol/model lock changed')
    require(metadata.get('checks') and all(value is True for value in metadata['checks'].values()), 'Dataset publication checks incomplete')
    payload = metadata['output_sha256_excluding_this_metadata']
    require(isinstance(payload, dict) and payload
            and set(payload) == {str(path.relative_to(dataset)) for path in dataset.rglob('*') if path.is_file() and path.name != 'metadata.json'}
                | {str(path.relative_to(dataset)) for path in (dataset/'examples').rglob('metadata.json')},
            'Validation payload coverage changed')
    for name, checksum in payload.items():
        require(digest(within(dataset, name)) == checksum, 'Validation payload changed: '+name)
    for name, checksum in metadata['source_identity_sha256'].items():
        require(digest(within(ROOT, name)) == checksum, 'Frozen validation source/review evidence changed: '+name)
    for name, checksum in metadata['implementation_sha256'].items():
        require(digest(within(ROOT/'Solutions', name)) == checksum, 'Validation publication implementation changed: '+name)
    _check_stats(metadata['source_file_stats'])
    reviews_path, selection_path = within(ROOT, metadata['reviews_file']), within(ROOT, metadata['selection_file'])
    require(digest(reviews_path) == metadata['reviews_sha256'] and digest(selection_path) == metadata['selection_sha256'],
            'Finalized review/selection lock changed')
    review_bundle, selection = read_json(reviews_path), read_json(selection_path)
    require(review_bundle['selection_sha256'] == digest(selection_path)
            and within(ROOT, review_bundle['selection_file']) == selection_path
            and selection['selection_uses_model_predictions'] is False, 'Review selection provenance changed')
    selected = {row['event_id']: row for row in selection['candidates']}
    require(len(selected) == len(selection['candidates']) <= 6, 'Predeclared selection coverage changed')
    dispositions = list(records(dataset/'review_decisions.jsonl', maximum=6))
    require(len(dispositions) == metadata['review_count'] == len(review_bundle['reviews'])
            and {row['event_id'] for row in dispositions} == {row['event_id'] for row in review_bundle['reviews']},
            'Retained review disposition coverage changed')
    review_by_event = {row['event_id']: row for row in review_bundle['reviews']}
    require(len(review_by_event) == len(dispositions), 'Duplicate review events')
    require(isinstance(review_bundle.get('prepared_case_files'), list)
            and set(review_bundle['prepared_case_files']) == {row['case_file'] for row in review_bundle['reviews']},
            'Every prepared case must retain a review disposition')
    audit = ROOT/'Datasets/AnnotationAudit/v1'
    catalogue, split_plan = read_json(audit/'annotations_clean.geojson'), read_json(audit/'splits.json')
    scenes = list(records(audit/'scene_inventory.jsonl'))
    quality_summary = read_json(ROOT/'Datasets/PixelQA/v1/summary/metadata.json')
    require(quality_summary['config_id'] == policy['qa_policy_id']
            and quality_summary['processing_errors'] == 0 and all(quality_summary['checks'].values()),
            'Frozen QA publication is incomplete or changed')
    for name, values in metadata['source_file_stats'].items():
        require(quality_summary['source_file_stats'].get(name) == values, 'Source fingerprints differ from frozen QA')
    training = list(records(ROOT/'Datasets/CuratedPilot/v1/dataset/training_manifest.jsonl', maximum=13))
    train_plan = read_json(ROOT/'Docs/Reviews/PixelPilotV1/experiment_plan.json')
    require(digest(ROOT/'Datasets/CuratedPilot/v1/dataset/training_manifest.jsonl') == train_plan['selection']['manifest_sha256'],
            'Isolation reference differs from completed training')
    wanted_pairs = {row['pair_id'] for row in selection['candidates']}
    frozen = {row['pair_id']: row for row in records(audit/'candidate_pairs.jsonl') if row['pair_id'] in wanted_pairs}
    cases, accepted, groups, strata = {}, {}, set(), set()
    for disposition in dispositions:
        event_id = disposition['event_id']; require(event_id in selected, 'Review event is absent from selection')
        review = review_by_event[event_id]; require(review == disposition['review'], 'Retained review statements changed')
        case_path = within(ROOT, review['case_file']); case = read_json(case_path)
        require(review['case_sha256'] == digest(case_path) and review['prepared_case_sha256'] == canonical(case)
                and set(case['artifact_sha256']) == PREPARED_ARTIFACTS
                and review['artifact_sha256'] == case['artifact_sha256'], 'Prepared evidence review binding changed')
        for name, checksum in case['artifact_sha256'].items():
            require(digest(within(case_path.parent, name)) == checksum, 'Prepared artifact changed after review: '+name)
        item = selected[event_id]; parent = frozen[item['pair_id']]
        require(case['pair'] == parent and case['event_id'] == event_id
                and case['stratum'] == item['stratum']
                and all(disposition[key] == item[key] for key in ('stratum', 'role', 'pair_id', 'group_id')),
                'Frozen parent/selection disposition identity changed')
        validate_validation_pair(parent, event_id, catalogue, split_plan, training)
        expected_scenes, window = select_timeline(scenes, parent['tile'], case['annotation_observation_date'])
        require(case['timeline_window'] == window
                and [(row['scene_id'], row['date'], row['scene_path'], row['metadata_usable']) for row in case['timeline']]
                    == [(row['scene_id'], row['acquisition_date'], row['path'], row['metadata_usable']) for row in expected_scenes],
                'Prepared timeline omitted or changed a local scene/date')
        batch = read_json(within(ROOT, review['prepared_batch_file']))
        require(batch['mode'] == 'prepare_validation_review' and batch['config_id'] == policy['qa_policy_id']
                and batch['implementation_sha256'] == digest(ROOT/'Solutions/prepare_validation_review.py')
                and batch['selection_sha256'] == metadata['selection_sha256']
                and batch['candidate_manifest_sha256'] == digest(audit/'candidate_pairs.jsonl')
                and batch['annotation_catalogue_sha256'] == digest(audit/'annotations_clean.geojson')
                and batch['split_plan_sha256'] == digest(audit/'splits.json')
                and batch['audit_metadata_sha256'] == digest(audit/'metadata.json')
                and batch['quality_summary_sha256'] == digest(ROOT/'Datasets/PixelQA/v1/summary/metadata.json')
                and all(batch['checks'].values())
                and any(row['event_id'] == event_id and row['pair_id'] == parent['pair_id'] for row in batch['cases']),
                'Prepared batch case/selection/source receipt changed')
        result = validate_validation_review(case, review)
        require(disposition['validation'] == result and disposition['decision_status'] == result['decision_status'],
                'Validated review decision changed')
        cases[event_id] = (case, case_path.parent, result)
        if result['evaluation_approved']:
            require(parent['group_id'] not in groups and item['stratum'] not in strata, 'Accepted validation groups/strata repeat')
            groups.add(parent['group_id']); strata.add(item['stratum']); accepted[event_id] = result
    for item in selection['candidates']:
        if item['role'] == 'primary':
            require(item['event_id'] in cases, 'Every predeclared primary needs a retained disposition')
        elif item['event_id'] in accepted:
            primary = [p for p in selection['candidates'] if p['role'] == 'primary' and p['stratum'] == item['stratum']]
            require(len(primary) == 1 and primary[0]['event_id'] in cases
                    and cases[primary[0]['event_id']][2]['decision_status'] == 'rejected',
                    'Accepted alternate requires its primary rejection')
    rows = list(records(dataset/'validation_manifest.jsonl', maximum=6))
    require(1 <= len(accepted) <= 3 and len(rows) == metadata['sample_count']
            and sum(row['kind'] == 'loss' for row in rows) == metadata['loss_case_count'] == len(accepted)
            and sum(row['kind'] == 'stable_roi' for row in rows) == metadata['stable_pair_count'] <= 3
            and set(metadata['group_ids']) == groups and len(metadata['group_ids']) == len(groups)
            and len({row['sample_id'] for row in rows}) == len(rows), 'Accepted sample/group coverage changed')
    samples, totals = [], Counter()
    expected_ids = {'loss-'+event_id for event_id in accepted}
    expected_ids |= {'stable-'+event_id for event_id, decision in accepted.items() if decision['stable_pair'] is not None}
    require({row['sample_id'] for row in rows} == expected_ids, 'Derived stable-pair availability or loss coverage changed')
    prepared_arrays, endpoint_masks = {}, {}
    for event_id, (case, directory, decision) in cases.items():
        if event_id not in accepted:
            continue
        grid = case['pair']['grid']
        target = read_binary(directory/'target_mask.tif', grid)
        interior = read_binary(directory/'positive_interior.tif', grid)
        control = read_binary(directory/'proposed_control.tif', grid)
        excluded = read_binary(directory/'catalogue_exclusion.tif', grid)
        require(np.array_equal(interior, binary_erosion(target, structure=np.ones((3, 3), dtype=bool)))
                and int(target.sum()) == case['target_pixels'] and int(interior.sum()) == case['positive_interior_pixels']
                and int(control.sum()) == case['control_proposal']['pixels']
                and np.array_equal(excluded, catalogue_exclusion(case['pair']))
                and not (control & excluded).any() and not (control & interior).any(), 'Prepared ROI/exclusion reconstruction changed')
        prepared_arrays[event_id] = {'positive': interior, 'stable': control,
            'features': read_tif(directory/'input_features.tif', grid, channels=True),
            'feature_valid': read_binary(directory/'feature_valid.tif', grid),
            'qa': read_binary(directory/'endpoint_unusable.tif', grid)}
        inspected = {row['scene_id']: row for row in decision['observations']}
        endpoints = {}
        for observation in case['timeline']:
            statement = inspected[observation['scene_id']]
            if observation.get('observation') is None:
                require(statement['target_state'] not in ('forest', 'clearing_visible')
                        and statement['control_state'] != 'stable_forest', 'Unavailable cache claimed clear')
                continue
            expected = observation['observation']
            require(expected['scene_id'] == observation['scene_id'] and expected['grid'] == grid,
                    'Observation cache scene/grid differs from prepared timeline')
            masks = read_cache(expected, policy['qa_policy_id'])
            unusable = np.logical_or.reduce([masks[name] for name in MASK_NAMES])
            for region_name, region in (('event_metrics', target), ('control_metrics', control)):
                numeric = observation[region_name]
                require(numeric['region_pixels'] == int(region.sum())
                        and numeric['unusable_pixels'] == int((unusable & region).sum()), 'Cached ROI quality counts changed')
                for name in MASK_NAMES:
                    require(numeric[name+'_pixels'] == int((masks[name] & region).sum()), 'Cached ROI QA component changed')
            if statement['target_state'] in ('forest', 'clearing_visible'):
                require(not (unusable & target).any(), 'Reviewed clear target intersects actual cache QA')
            if statement['control_state'] == 'stable_forest':
                require(not (unusable & control).any(), 'Reviewed stable control intersects actual cache QA')
            endpoints[observation['scene_id']] = unusable
        endpoint_masks[event_id] = endpoints
        primary = case['pair']
        require(np.array_equal(prepared_arrays[event_id]['qa'], endpoints[primary['pre_scene_id']] | endpoints[primary['post_scene_id']]),
                'Prepared loss QA differs from actual endpoint caches')
    for row in rows:
        validate_record_role(row); require(row['event_id'] in accepted, 'Exported sample lacks an accepted review')
        case, _, decision = cases[row['event_id']]; expected_pair = case['pair'] if row['kind'] == 'loss' else expected_derived_pair(case, decision)
        require(row['pair'] == expected_pair and row['qa_policy_id'] == policy['qa_policy_id']
                and row['review_sha256'] == canonical(review_by_event[row['event_id']])
                and row['prepared_case_sha256'] == canonical(case), 'Sample pair/review lineage changed')
        validate_grid(row['pair']); grid = row['grid']; sample_id = row['sample_id']
        paths = {role: f'examples/{sample_id}/{name}' for role, name in
                 (('input_features', 'input_features.tif'), ('supervision_mask', 'supervision_mask.tif'),
                  ('valid_mask', 'valid_mask.tif'), ('metadata', 'metadata.json'))}
        require(row['artifact_paths'] == paths, 'Export paths differ from their exact sample roles')
        sample_meta = read_json(within(dataset, paths['metadata']))
        require(all(sample_meta.get(key) == value for key, value in row.items())
                and sample_meta['channels'] == list(CHANNELS) and sample_meta['decision'] == decision
                and sample_meta['derived_pair'] is (row['kind'] == 'stable_roi')
                and sample_meta['lineage'] == {'frozen_parent_pair': case['pair'], 'reviewed_event_id': case['event_id'],
                    'stable_pair_rule': decision['stable_pair'] if row['kind'] == 'stable_roi' else None},
                'Sample metadata lineage or approved decision changed')
        extras = {role: f'examples/{sample_id}/{name}.tif' for role, name in
                  (('endpoint_unusable', 'endpoint_unusable'), ('feature_valid', 'feature_valid'))}
        require(sample_meta['extra_artifact_paths'] == extras, 'Sample QA/feature-validity paths changed')
        directory = within(dataset, paths['metadata']).parent
        require(set(sample_meta['output_sha256']) == {'input_features.tif', 'supervision_mask.tif', 'valid_mask.tif',
                                                     'endpoint_unusable.tif', 'feature_valid.tif'}, 'Sample payload hash coverage changed')
        for name, checksum in sample_meta['output_sha256'].items():
            require(digest(within(directory, name)) == checksum, 'Sample payload checksum changed')
        _check_stats(sample_meta['feature_source_file_stats'])
        require(all(metadata['source_file_stats'].get(name) == values
                    and quality_summary['source_file_stats'].get(name) == values
                    for name, values in sample_meta['feature_source_file_stats'].items()),
                'Sample feature fingerprints are absent from frozen export/QA sources')
        features = read_tif(within(dataset, paths['input_features']), grid, channels=True)
        labels = read_tif(within(dataset, paths['supervision_mask']), grid, nodata=255)
        valid = read_tif(within(dataset, paths['valid_mask']), grid)
        qa = read_binary(within(dataset, extras['endpoint_unusable']), grid)
        feature_valid = read_binary(within(dataset, extras['feature_valid']), grid)
        original = prepared_arrays[row['event_id']]
        endpoints = endpoint_masks[row['event_id']]
        expected_qa = endpoints[row['pair']['pre_scene_id']] | endpoints[row['pair']['post_scene_id']]
        require(np.array_equal(qa, expected_qa), 'Exported QA differs from exact source observation caches')
        if row['kind'] == 'loss':
            require(np.array_equal(features, original['features'], equal_nan=True)
                    and np.array_equal(feature_valid, original['feature_valid']), 'Primary features differ from prepared reviewed arrays')
        expected_labels = np.full((64, 64), 255, dtype='uint8')
        expected_labels[original['stable']] = 0
        if row['kind'] == 'loss':
            expected_labels[original['positive']] = 1
        require(np.array_equal(labels, expected_labels) and not qa[labels != 255].any(), 'Exported reviewed ROI labels or QA changed')
        model_input, replacements = validate_arrays(features, labels, valid, feature_valid)
        counts = {key: int((labels == value).sum()) for key, value in zip(COUNTS, (1, 0, 255))}
        require(all(row[key] == value for key, value in counts.items()), 'Exported reviewed class counts changed')
        totals.update(counts)
        samples.append({'features': model_input, 'labels': labels.copy(), 'valid': labels != 255,
                        'sample_id': sample_id, 'kind': row['kind'], 'event_id': row['event_id'],
                        'group_id': row['group_id'], 'nonfinite_replacements': replacements})
    require(dict(totals) == metadata['supervision_pixel_counts'], 'Dataset aggregate reviewed support changed')
    _check_stats(metadata['source_file_stats'])
    require(digest(protocol_path) == metadata['protocol_sha256'] and digest(reviews_path) == metadata['reviews_sha256'],
            'Label/protocol lock changed during verification')
    return metadata, rows, samples


def verify_review_dispositions(reviews_path, protocol_path):
    """Independently audit a fully rejected bounded roster, without a dataset.

    Rejected evidence is a completed curation outcome. This function must not
    turn it into an empty evaluation set, synthesize labels or report accuracy.
    """
    import numpy as np
    import rasterio
    from affine import Affine
    from pyproj import Transformer
    from scipy.ndimage import binary_erosion
    from shapely.geometry import shape
    from shapely.ops import transform
    from prepare_curated_pilot import select_timeline
    from prepare_validation_review import validate_validation_pair, validate_grid, catalogue_exclusion
    from validation_review import validate_validation_review
    from validate_sample import burn

    reviews_path, protocol_path = Path(reviews_path).resolve(), Path(protocol_path).resolve()
    bundle, protocol = read_json(reviews_path), read_json(protocol_path)
    require(protocol['status'] == 'predeclared_before_validation_predictions'
            and protocol['review']['used_model_predictions'] is False, 'Prediction-blind validation protocol required')
    for name, checksum in protocol['immutable_sources_sha256'].items():
        require(digest(within(ROOT/'Solutions', name)) == checksum, 'Immutable archived source changed: '+name)
    evaluation = protocol['evaluation']
    require(evaluation['checkpoint_attempt'] == 1664 and evaluation['probability_threshold'] == .5
            and evaluation['retraining'] is False and evaluation['threshold_tuning'] is False
            and evaluation['best_checkpoint_selection'] is False
            and digest(within(ROOT, evaluation['checkpoint'])) == evaluation['weights_sha256']
            and digest(ROOT/'Docs/Reviews/PixelPilotV1/run/metadata.json') == evaluation['run_metadata_sha256'],
            'Frozen model/checkpoint protocol changed during review audit')
    selection_path = within(ROOT, bundle['selection_file']); selection = read_json(selection_path)
    require(digest(selection_path) == bundle['selection_sha256']
            and selection['selection_uses_model_predictions'] is False
            and selection['model_executions'] == selection['pixel_arrays_read'] == 0,
            'Blind metadata-only selection lock changed')
    candidates = selection['candidates']; by_event = {row['event_id']: row for row in candidates}
    reviews = bundle['reviews']; review_map = {row['event_id']: row for row in reviews}
    require(1 <= len(reviews) <= 6 and len(review_map) == len(reviews)
            and len(by_event) == len(candidates) and set(review_map) == set(by_event),
            'Fully rejected review roster must retain every predeclared candidate once')
    require(isinstance(bundle.get('prepared_case_files'), list)
            and len(bundle['prepared_case_files']) == len(reviews)
            and set(bundle['prepared_case_files']) == {row['case_file'] for row in reviews},
            'Prepared-case review coverage changed')
    require(len({row['group_id'] for row in candidates}) == len(candidates), 'Selected geographic groups repeat')
    slots = [(row['stratum'], row['role']) for row in candidates]
    require(len(set(slots)) == len(slots) and all(role in ('primary', 'alternate') for _, role in slots),
            'Predeclared primary/alternate slots changed')
    audit = ROOT/'Datasets/AnnotationAudit/v1'; qa_root = ROOT/'Datasets/PixelQA/v1'
    catalogue = read_json(audit/'annotations_clean.geojson'); split_plan = read_json(audit/'splits.json')
    scenes = list(records(audit/'scene_inventory.jsonl'))
    training_path = ROOT/'Datasets/CuratedPilot/v1/dataset/training_manifest.jsonl'
    training = list(records(training_path, maximum=13))
    training_plan = read_json(ROOT/'Docs/Reviews/PixelPilotV1/experiment_plan.json')
    require(digest(training_path) == training_plan['selection']['manifest_sha256'], 'Frozen training isolation reference changed')
    summary_path = qa_root/'summary/metadata.json'; summary = read_json(summary_path)
    require(summary['config_id'] == protocol['qa_policy_id'] and summary['processing_errors'] == 0
            and all(summary['checks'].values()), 'Completed frozen quality publication changed')
    wanted = {row['pair_id'] for row in candidates}
    frozen_pairs = {row['pair_id']: row for row in records(audit/'candidate_pairs.jsonl') if row['pair_id'] in wanted}
    quality = {row['pair_id']: row for row in records(qa_root/'review/quality_result_index.jsonl') if row['pair_id'] in wanted}
    require(set(frozen_pairs) == set(quality) == wanted, 'Selected frozen pair/QA coverage changed')
    evidence_paths = [reviews_path, protocol_path, selection_path, audit/'metadata.json', audit/'candidate_pairs.jsonl',
                      audit/'annotations_clean.geojson', audit/'splits.json', audit/'scene_inventory.jsonl',
                      training_path, summary_path, qa_root/'review/quality_result_index.jsonl']
    evidence_hashes = {str(path.relative_to(ROOT)): digest(path) for path in evidence_paths}
    properties = {row['id']: row for row in catalogue['features']}
    decisions, cache_hashes, statistics, timeline_count = [], {}, {}, 0
    for review in reviews:
        event_id = review['event_id']; selected = by_event[event_id]; parent = frozen_pairs[selected['pair_id']]
        case_path = within(ROOT, review['case_file']); case = read_json(case_path)
        require(review['case_sha256'] == digest(case_path) and review['prepared_case_sha256'] == canonical(case)
                and case['pair'] == parent and case['event_id'] == event_id and case['stratum'] == selected['stratum']
                and case['group_id'] == selected['group_id'] and case['model_predictions_read'] is False,
                'Exact reviewed case/selection/frozen pair binding changed')
        require(set(case['artifact_sha256']) == PREPARED_ARTIFACTS
                and review['artifact_sha256'] == case['artifact_sha256'], 'Prepared artifact review bindings incomplete')
        evidence_hashes[review['case_file']] = digest(case_path)
        for name, checksum in case['artifact_sha256'].items():
            path = within(case_path.parent, name)
            require(digest(path) == checksum, 'Rejected prepared artifact changed after review: '+name)
            evidence_hashes[str(path.relative_to(ROOT))] = checksum
        validate_validation_pair(parent, event_id, catalogue, split_plan, training)
        batch_path = within(ROOT, review['prepared_batch_file']); batch = read_json(batch_path)
        expected_batch = {'selection_sha256': digest(selection_path), 'audit_metadata_sha256': digest(audit/'metadata.json'),
            'candidate_manifest_sha256': digest(audit/'candidate_pairs.jsonl'),
            'annotation_catalogue_sha256': digest(audit/'annotations_clean.geojson'),
            'split_plan_sha256': digest(audit/'splits.json'), 'quality_summary_sha256': digest(summary_path)}
        require(batch['mode'] == 'prepare_validation_review' and batch['config_id'] == protocol['qa_policy_id']
                and batch['implementation_sha256'] == digest(ROOT/'Solutions/prepare_validation_review.py')
                and all(batch.get(key) == value for key, value in expected_batch.items())
                and all(batch['checks'].values()) and batch['model_predictions_read'] is False
                and batch['training_approved'] is False and batch['evaluation_approved'] is False
                and any(row['event_id'] == event_id and row['pair_id'] == parent['pair_id'] for row in batch['cases']),
                'Rejected prepared-batch receipt changed')
        evidence_hashes[review['prepared_batch_file']] = digest(batch_path)
        for name, checksum in batch['source_identity_sha256'].items():
            require(digest(within(ROOT, name)) == checksum, 'Prepared frozen source changed: '+name)
            evidence_hashes[name] = checksum
        for name, values in {**batch['source_file_stats'], **case['feature_source_file_stats']}.items():
            require(summary['source_file_stats'].get(name) == values
                    and (name not in statistics or statistics[name] == values), 'Prepared source fingerprints contradict frozen QA')
            statistics[name] = values
        decision = validate_validation_review(case, review)
        require(decision['decision_status'] == 'rejected' and decision['training_approved'] is False
                and decision['evaluation_approved'] is False and decision['observed_change_interval'] is None
                and decision['stable_pair'] is None, 'Fully rejected roster must not create labels or evaluation approval')
        grid = parent['grid']; actual_grid = validate_grid(parent)
        target = read_binary(case_path.parent/'target_mask.tif', grid)
        interior = read_binary(case_path.parent/'positive_interior.tif', grid)
        control = read_binary(case_path.parent/'proposed_control.tif', grid)
        excluded = read_binary(case_path.parent/'catalogue_exclusion.tif', grid)
        feature_valid = read_binary(case_path.parent/'feature_valid.tif', grid)
        features = read_tif(case_path.parent/'input_features.tif', grid, channels=True)
        prepared_qa = read_binary(case_path.parent/'endpoint_unusable.tif', grid)
        require(int(target.sum()) == case['target_pixels'] and int(interior.sum()) == case['positive_interior_pixels']
                and int(control.sum()) == case['control_proposal']['pixels']
                and np.array_equal(interior, binary_erosion(target, structure=np.ones((3, 3), dtype=bool)))
                and np.array_equal(excluded, catalogue_exclusion(parent)) and not (control & excluded).any(),
                'Rejected-case actual ROI counts, erosion or exclusion differ from preparation')
        # These are feature-integrity checks, not approval of unknown pixels.
        empty_labels = np.full((64, 64), 255, dtype='uint8')
        validate_arrays(features, empty_labels, np.zeros((64, 64), dtype=bool), feature_valid)
        expected_scenes, window = select_timeline(scenes, parent['tile'], case['annotation_observation_date'])
        require(case['timeline_window'] == window
                and [(row['scene_id'], row['date'], row['scene_path'], row['metadata_usable']) for row in case['timeline']]
                    == [(row['scene_id'], row['acquisition_date'], row['path'], row['metadata_usable']) for row in expected_scenes],
                'Rejected timeline omitted or changed an available observation')
        statements = {row['scene_id']: row for row in review['observations']}; endpoints = {}
        for observation in case['timeline']:
            statement = statements[observation['scene_id']]
            if observation.get('observation') is None:
                require(statement['target_state'] not in ('forest', 'clearing_visible')
                        and statement['control_state'] != 'stable_forest', 'Unavailable rejected observation claimed clear')
                continue
            expected = observation['observation']
            require(expected['scene_id'] == observation['scene_id'] and expected['grid'] == grid,
                    'Rejected cache scene/grid differs from its timeline')
            masks = read_cache(expected, protocol['qa_policy_id'])
            cache_path = within(ROOT, expected['cache_path']); cache_hashes[expected['cache_path']] = digest(cache_path)
            unusable = np.logical_or.reduce([masks[name] for name in MASK_NAMES]); endpoints[observation['scene_id']] = masks
            for key, region in (('event_metrics', target), ('control_metrics', control)):
                numeric = observation[key]
                require(numeric['region_pixels'] == int(region.sum())
                        and numeric['unusable_pixels'] == int((unusable & region).sum())
                        and numeric['valid_pixels'] == int(region.sum())-int((masks['invalid'] & region).sum()),
                        'Rejected-case ROI quality denominator/union changed')
                require(all(numeric[name+'_pixels'] == int((masks[name] & region).sum()) for name in MASK_NAMES),
                        'Rejected-case actual QA components differ from recorded metrics')
            if statement['target_state'] in ('forest', 'clearing_visible'):
                require(not (unusable & target).any(), 'Rejected review still cannot make a clear target claim across QA flags')
            if statement['control_state'] == 'stable_forest':
                require(not (unusable & control).any(), 'Rejected review still cannot make a clear control claim across QA flags')
            timeline_count += 1
        before, after = endpoints[parent['pre_scene_id']], endpoints[parent['post_scene_id']]
        qa_union = np.logical_or.reduce([before[name] | after[name] for name in MASK_NAMES])
        require(np.array_equal(prepared_qa, qa_union), 'Rejected prepared endpoint QA differs from observation caches')
        published = case['published_qa']; published_path = within(ROOT, published['qa_file'])
        require(published['index_row'] == quality[parent['pair_id']]
                and digest(published_path) == published['qa_file_sha256'], 'Frozen published QA receipt changed')
        with rasterio.open(published_path) as source:
            require(source.shape == (64, 64) and source.count == 6 and source.dtypes == ('uint8',)*6
                    and source.transform == Affine(*grid['transform'])
                    and source.crs == rasterio.crs.CRS.from_user_input(grid['crs']), 'Frozen rejected-case QA grid changed')
            published_arrays = source.read()
        require(np.isin(published_arrays, (0, 1)).all()
                and np.array_equal(published_arrays[:5].astype(bool), np.stack([before[name] | after[name] for name in MASK_NAMES])),
                'Frozen QA planes differ from actual endpoint caches')
        anchor = transform(Transformer.from_crs('EPSG:4326', grid['crs'], always_xy=True).transform,
                           shape(properties[event_id]['geometry']))
        expected_target = burn([anchor], actual_grid).astype(bool) & published_arrays[5].astype(bool)
        require(np.array_equal(target, expected_target), 'Rejected anchor target differs from frozen temporal reference')
        evidence_hashes[published['qa_file']] = published['qa_file_sha256']
        decisions.append({'event_id': event_id, 'pair_id': parent['pair_id'], 'group_id': parent['group_id'],
            'stratum': selected['stratum'], 'role': selected['role'], 'decision_status': 'rejected',
            'training_approved': False, 'evaluation_approved': False,
            'target_pixels': int(target.sum()), 'positive_interior_pixels': int(interior.sum()),
            'control_pixels': int(control.sum()), 'rejection_reasons': decision['rejection_reasons']})
    for row in candidates:
        require(row['role'] != 'alternate' or any(p['role'] == 'primary' and p['stratum'] == row['stratum']
                for p in candidates), 'Alternate lacks a predeclared reviewed primary')
    _check_stats(statistics)
    require(all(digest(within(ROOT, name)) == checksum for name, checksum in {**evidence_hashes, **cache_hashes}.items()),
            'Rejected review evidence changed during verification')
    return {'verification_status': 'passed', 'outcome': 'bounded_roster_fully_rejected',
        'review_count': len(decisions), 'rejected_case_count': len(decisions), 'accepted_loss_case_count': 0,
        'published_validation_sample_count': 0, 'training_approved': False, 'evaluation_approved': False,
        'model_evaluation_performed': False, 'accuracy_metrics': None,
        'timeline_observations_checked': timeline_count, 'unique_observation_caches_checked': len(cache_hashes),
        'review_dispositions': decisions, 'reviews_sha256': digest(reviews_path), 'protocol_sha256': digest(protocol_path),
        'selection_sha256': digest(selection_path), 'source_evidence_sha256': evidence_hashes,
        'observation_cache_sha256': cache_hashes, 'verification_implementation_sha256': digest(Path(__file__)),
        'source_jp2_pixel_reads': 0, 'model_executions': 0,
        'limitations': ['Rejected evidence does not supply an evaluation dataset or an accuracy estimate.',
                        'Assistant visual review is not independent expert certification.',
                        'Prepared feature arrays were checked without re-extracting JP2 source pixels.']}


def worker(args, output):
    os.nice(10)
    require('torch' not in sys.modules, 'Standalone dataset audit must not import Torch')
    if args.mode == 'reviews':
        result = verify_review_dispositions(args.reviews, args.protocol)
        require('torch' not in sys.modules, 'Rejected-review verification unexpectedly imported Torch')
        result['torch_imported'] = False
        lines = ['# Rejected validation review verification', '',
                 f"Verified all {result['rejected_case_count']} rejected candidates and every prepared observation.",
                 '**No validation dataset was published and no model evaluation or accuracy estimate was produced.**', '',
                 '| Event | Role | Target pixels | Eroded interior pixels | Rejection reasons |',
                 '| --- | --- | ---: | ---: | --- |']
        for row in result['review_dispositions']:
            lines.append('| '+' | '.join((row['event_id'], row['role'], str(row['target_pixels']),
                str(row['positive_interior_pixels']), '; '.join(row['rejection_reasons'])))+' |')
        lines += ['', 'Checked frozen selection/pair/group joins, complete geographic footprints, review-time artifact hashes,',
                  'actual target erosion, local timeline coverage, cached QA, source fingerprints and retained rejection decisions.',
                  'No Torch, model execution, raw JP2 pixel reads or training.']
        (output/'report.md').write_text('\n'.join(lines).rstrip()+'\n')
        import resource
        result['worker_high_water_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024), 2)
        (output/'metadata.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        return
    metadata, rows, samples = load_and_verify(args.dataset, args.protocol)
    require('torch' not in sys.modules, 'Standalone dataset audit unexpectedly imported Torch')
    result = {'verification_status': 'passed', 'sample_count': len(rows),
        'loss_case_count': metadata['loss_case_count'], 'stable_pair_count': metadata['stable_pair_count'],
        'geographic_group_count': len(metadata['group_ids']), 'retained_review_count': metadata['review_count'],
        'supervision_pixel_counts': metadata['supervision_pixel_counts'],
        'nonfinite_input_values_zeroed_in_returned_copies': sum(row['nonfinite_replacements'] for row in samples),
        'dataset_metadata_sha256': digest(args.dataset/'metadata.json'), 'protocol_sha256': digest(args.protocol),
        'implementation_sha256': digest(Path(__file__)), 'torch_imported': False,
        'source_jp2_pixel_reads': 0, 'model_executions': 0,
        'limitations': ['Assistant reviews are not independent expert ground truth.',
                        'Derived features are checked against signed exported arrays, cache QA and source fingerprints; source JP2 pixels are not re-extracted.',
                        'Partial ROIs do not certify complete masks, clearing boundaries or areas.']}
    (output/'report.md').write_text('# Validation dataset verification\n\n'
        f"Passed {len(rows)} samples from {len(metadata['group_ids'])} geographic validation groups, "
        f"retaining all {metadata['review_count']} review dispositions.\n\n"
        'Checked role/label locks, all payload/evidence hashes, frozen parent and derived-pair lineage, '
        'full feature-footprint isolation, actual observation-cache QA, reviewed ROI reconstruction, '
        'TIFF grids, signed feature scales and source fingerprints. No Torch, model execution or JP2 pixel reads.\n\n'
        'Derived feature values were not independently re-extracted from JP2 source bands.\n')
    import resource
    result['worker_high_water_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024), 2)
    (output/'metadata.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('dataset', 'reviews'), default='dataset')
    parser.add_argument('--reviews', type=Path)
    parser.add_argument('--dataset', type=Path, default=ROOT/'Datasets/ValidationPilot/v1/dataset')
    parser.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationPilotV1/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=180)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == 'reviews' and args.reviews is None:
        parser.error('--reviews is required in reviews mode')
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Use 128–512 MiB RSS, >=1.5 GiB reserve and <=300 seconds')
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='validation-dataset-verification', gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
