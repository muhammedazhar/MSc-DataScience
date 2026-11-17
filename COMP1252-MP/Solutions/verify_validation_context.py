#!/usr/bin/env python3
"""Verify a predeclared local-context QA audit without model execution.

Independent masks and control search use NumPy. Saved input features are
checked against hashes, scale identities and source fingerprints, without
re-extracting source satellite bands.
"""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MASK_NAMES = ('cloud', 'cloud_buffer', 'shadow_risk', 'haze_risk', 'invalid')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 32*2**20, 'Missing or oversized context evidence')
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite context evidence')))


def compare(actual, expected, label):
    require(type(actual) is type(expected), 'Context evidence type mismatch: '+label)
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Context evidence field mismatch: '+label)
        for key, value in expected.items():
            compare(actual[key], value, label+'.'+key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'Context evidence coverage mismatch: '+label)
        for index, value in enumerate(expected):
            compare(actual[index], value, label+f'[{index}]')
    else:
        require(actual == expected, 'Context evidence value mismatch: '+label)


def square_halo(mask, radius=2):
    """Return exact Chebyshev context plus whether no dependency was clipped."""
    import numpy as np

    value = np.asarray(mask)
    require(value.shape == (64, 64) and value.dtype == np.dtype(bool) and value.any(),
            'A nonempty boolean 64x64 ROI is required')
    require(type(radius) is int and radius == 2, 'This architecture requires radius two')
    padded = np.pad(value, radius)
    enlarged = np.zeros_like(padded)
    # Dilation is a union of shifted source pixels; no SciPy morphology call.
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            enlarged[radius+dy:radius+dy+64, radius+dx:radius+dx+64] |= value
    core = enlarged[radius:radius+64, radius:radius+64].copy()
    return core, int(enlarged.sum()) == int(core.sum())


def inspect_context(roi, qa_pre, qa_post, features, feature_valid):
    import numpy as np

    require(roi.shape == (64, 64) and roi.dtype == np.dtype(bool), 'ROI must be a boolean 64x64 mask')
    # A missing control is a retained negative result, never vacuous clearance.
    halo, inside = square_halo(roi) if roi.any() else (np.zeros_like(roi), False)
    require(qa_pre.shape == qa_post.shape == (5, 64, 64)
            and qa_pre.dtype == qa_post.dtype == np.dtype(bool), 'Both dates need five boolean QA masks')
    require(features.shape == (27, 64, 64) and features.dtype == np.dtype('float32')
            and feature_valid.shape == (64, 64) and feature_valid.dtype == np.dtype(bool),
            'Exact float32 model channels and feature validity are required')
    masks = (qa_pre, qa_post)
    counts = {}
    for side, planes in zip(('pre', 'post'), masks):
        counts[side] = {'roi_unusable_pixels': int((np.any(planes, axis=0) & roi).sum()),
                       'context_unusable_pixels': int((np.any(planes, axis=0) & halo).sum()),
                       'context_components': {name: int((plane & halo).sum()) for name, plane in zip(MASK_NAMES, planes)}}
    finite = np.isfinite(features).all(axis=0)
    invalid_context = int((halo & (~finite | ~feature_valid)).sum())
    return {'roi_pixels': int(roi.sum()), 'context_pixels': int(halo.sum()),
            'full_context_inside_grid': inside, 'dates': counts,
            'invalid_or_nonfinite_context_pixels': invalid_context,
            'context_clear': bool(inside and invalid_context == 0
                and all(counts[side]['context_unusable_pixels'] == 0 for side in counts))}


def independent_control(positive_context, exclusion, qa_pre, qa_post, pre_ndvi,
                        post_ndvi, features, feature_valid, sizes=(16, 12, 8)):
    """Fixed largest-size then row/column-first numerical proposal only."""
    import numpy as np

    for value in (positive_context, exclusion, feature_valid):
        require(value.shape == (64, 64) and value.dtype == np.dtype(bool), 'Control context masks must be boolean64x64')
    require(pre_ndvi.shape == post_ndvi.shape == (64, 64), 'Both endpoint NDVI arrays are required')
    require(tuple(sizes) == (16, 12, 8), 'Control sizes must retain their declared order')
    finite = np.isfinite(features).all(axis=0)
    bad_context = (np.any(qa_pre, axis=0) | np.any(qa_post, axis=0)
                   | ~feature_valid | ~finite | exclusion | positive_context)
    green = (np.isfinite(pre_ndvi) & np.isfinite(post_ndvi)
             & (pre_ndvi > .6) & (post_ndvi > .6) & (np.abs(post_ndvi-pre_ndvi) < .10))
    # Prefix sums check whole squares independently of the producer's search.
    def prefix(mask):
        return np.pad(mask.astype('int64').cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    bad, not_green = prefix(bad_context), prefix(~green)
    def count(sums, r, c, size):
        return int(sums[r+size, c+size]-sums[r, c+size]-sums[r+size, c]+sums[r, c])
    for size in sizes:
        for row in range(2, 65-size-2):
            for col in range(2, 65-size-2):
                if count(not_green, row, col, size) or count(bad, row-2, col-2, size+4):
                    continue
                mask = np.zeros((64, 64), dtype=bool); mask[row:row+size, col:col+size] = True
                return mask, {'proposal_found': True, 'row': row, 'col': col,
                              'height': size, 'width': size, 'pixels': size*size,
                              'context_radius_pixels': 2, 'context_pixels': (size+4)**2,
                              'minimum_endpoint_ndvi': float(np.minimum(pre_ndvi[mask], post_ndvi[mask]).min()),
                              'maximum_absolute_ndvi_change': float(np.abs(post_ndvi[mask]-pre_ndvi[mask]).max()),
                              'search_order': 'largest declared size, ascending row, ascending column',
                              'control_label_review': 'unreviewed',
                              'training_approved': False, 'evaluation_approved': False}
    return np.zeros((64, 64), dtype=bool), {'proposal_found': False, 'pixels': 0,
                                         'reason': 'No declared square has green stable endpoint ROI and fully clear, finite, unexcluded input context',
                                         'control_label_review': 'unreviewed',
                                         'training_approved': False, 'evaluation_approved': False}


def control_gate_counts(arrays):
    """Report every fixed control square's gate counts without proposing a ROI.

    Marginals overlap; their counts are not sequential attrition totals.
    Composite names explicitly identify the gates counted together. Context
    gates cover the full square plus radius two, while NDVI gates cover its ROI.
    """
    import numpy as np

    qa_pre, qa_post = arrays['pre_qa'], arrays['post_qa']
    features, valid = arrays['features'], arrays['feature_valid']
    before, after = arrays['pre_ndvi'], arrays['post_ndvi']
    excluded = arrays['control_exclusion']
    require(qa_pre.shape == qa_post.shape == (5, 64, 64)
            and qa_pre.dtype == qa_post.dtype == np.dtype(bool), 'Diagnostics require both frozen endpoint QA masks')
    require(features.shape == (27, 64, 64) and features.dtype == np.dtype('float32')
            and valid.shape == excluded.shape == (64, 64)
            and valid.dtype == excluded.dtype == np.dtype(bool), 'Diagnostics require exact saved features and masks')
    require(before.shape == after.shape == (64, 64)
            and arrays['pre_rgb'].shape == arrays['post_rgb'].shape == (3, 64, 64),
            'Diagnostics require saved endpoint NDVI/RGB')
    finite = (np.isfinite(features).all(axis=0) & valid
              & np.isfinite(arrays['pre_rgb']).all(axis=0)
              & np.isfinite(arrays['post_rgb']).all(axis=0)
              & np.isfinite(before) & np.isfinite(after))
    qa_finite = ~np.any(qa_pre | qa_post, axis=0) & finite
    safe_context = qa_finite & ~excluded
    pre_green = np.isfinite(before) & (before > .6)
    post_green = np.isfinite(after) & (after > .6)
    stable = np.isfinite(before) & np.isfinite(after) & (np.abs(after-before) < .1)

    def prefix(mask):
        return np.pad((~mask).astype('int64').cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    prefixes = {name: prefix(mask) for name, mask in
                (('qa_finite', qa_finite), ('safe_context', safe_context),
                 ('pre_green', pre_green), ('post_green', post_green),
                 ('both_green', pre_green & post_green), ('stable', stable))}
    def passes(name, row, col, size):
        sums = prefixes[name]
        return sums[row+size, col+size]-sums[row, col+size]-sums[row+size, col]+sums[row, col] == 0

    results = []
    for size in (16, 12, 8):
        counts = {name: 0 for name in ('inbounds_squares', 'qa_finite_clear_context_squares',
            'qa_finite_exclusion_clear_context_squares', 'pre_ndvi_above_point6_roi_squares',
            'post_ndvi_above_point6_roi_squares', 'both_ndvi_above_point6_roi_squares',
            'absolute_ndvi_change_below_point1_roi_squares',
            'safe_context_and_both_ndvi_roi_squares', 'safe_context_and_stable_change_roi_squares',
            'all_gates_pass_squares')}
        for row in range(2, 65-size-2):
            for col in range(2, 65-size-2):
                clear = bool(passes('qa_finite', row-2, col-2, size+4))
                safe = bool(passes('safe_context', row-2, col-2, size+4))
                pre = bool(passes('pre_green', row, col, size))
                post = bool(passes('post_green', row, col, size))
                both = bool(passes('both_green', row, col, size))
                unchanged = bool(passes('stable', row, col, size))
                flags = {'inbounds_squares': True, 'qa_finite_clear_context_squares': clear,
                    'qa_finite_exclusion_clear_context_squares': safe,
                    'pre_ndvi_above_point6_roi_squares': pre,
                    'post_ndvi_above_point6_roi_squares': post,
                    'both_ndvi_above_point6_roi_squares': both,
                    'absolute_ndvi_change_below_point1_roi_squares': unchanged,
                    'safe_context_and_both_ndvi_roi_squares': safe and both,
                    'safe_context_and_stable_change_roi_squares': safe and unchanged,
                    'all_gates_pass_squares': safe and both and unchanged}
                for name, value in flags.items():
                    counts[name] += int(value)
        results.append({'size_pixels': size, 'roi_pixels': size*size,
                        'context_radius_pixels': 2, **counts})
    return {'diagnostic_only': True, 'counts_are_overlapping_not_cumulative': True,
            'search_order_and_thresholds_unchanged': True,
            'new_regions_selected': 0, 'per_size': results}


def architecture_contract(factory_path):
    """Parse the actual frozen factory, proving local receptive field sans Torch."""
    tree = ast.parse(Path(factory_path).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'make_model']
    require(len(functions) == 1, 'The frozen model factory must be unique')
    returned = [node for node in functions[0].body if isinstance(node, ast.Return)]
    require(len(returned) == 1, 'The frozen model factory must have one explicit return')
    call = returned[0].value
    require(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'to',
            'The model factory must retain explicit CPU conversion')
    conversion = {item.arg: item.value for item in call.keywords}
    require(ast.literal_eval(conversion['device']) == 'cpu', 'The model must stay on CPU')
    sequential = call.func.value
    require(isinstance(sequential, ast.Call) and isinstance(sequential.func, ast.Attribute)
            and sequential.func.attr == 'Sequential' and not sequential.keywords,
            'Only the frozen sequential architecture is supported')
    expected_names = ('Conv2d', 'ReLU', 'Conv2d', 'ReLU', 'Conv2d')
    require(len(sequential.args) == len(expected_names), 'The model layer count changed')
    convs = []
    for node, name in zip(sequential.args, expected_names):
        require(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == name,
                'The model contains a changed/global-context operation')
        if name == 'ReLU':
            require(not node.args and not node.keywords, 'The frozen activation changed')
            continue
        require(len(node.args) == 3 and all(item.arg in ('padding',) for item in node.keywords),
                'Convolution stride/dilation/group/kernel contract changed')
        channels_in, channels_out, kernel = [ast.literal_eval(value) for value in node.args]
        options = {item.arg: ast.literal_eval(item.value) for item in node.keywords}
        convs.append((channels_in, channels_out, kernel, options.get('padding', 0)))
    require(convs == [(27, 8, 3, 1), (8, 8, 3, 1), (8, 1, 1, 0)], 'The frozen convolution architecture changed')
    parameters = sum(c_in*c_out*kernel*kernel+c_out for c_in, c_out, kernel, _ in convs)
    return {'spatial_receptive_field_pixels': 5, 'context_radius_pixels': 2,
            'parameter_count': parameters, 'global_context_operations': False,
            'factory_sha256': digest(factory_path)}


def within(base, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'Relative context evidence path required')
    base = Path(base).resolve(); path = (base/relative).resolve()
    require(path.is_relative_to(base) and path.is_file(), 'Context evidence escaped its root or is missing')
    return path


def records(path):
    with Path(path).open('rb') as stream:
        for index in range(20001):
            line = stream.readline(512*1024+1)
            if not line:
                return
            require(index < 20000 and 0 < len(line) <= 512*1024 and line.strip(), 'Context JSONL bound exceeded')
            yield json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite context evidence')))


def validate_protocol(protocol):
    require(protocol['schema_version'] == 3
            and protocol['status'] == 'predeclared_context_quality_only_before_new_visual_review',
            'The fixed context-only protocol is required')
    rule = protocol['rule']
    for key, expected in {'context_radius_pixels': 2, 'minimum_positive_interior_pixels': 4,
            'target_erosion_pixels': 1, 'require_full_halo_inside_core': True,
            'require_full_target_and_interior_context': True, 'require_zero_qa_at_both_dates': True,
            'qa_components': list(MASK_NAMES), 'require_finite_all27_features_on_context': True,
            'retain_unchanged_full_crop_flags': True, 'control_sizes': [16, 12, 8],
            'minimum_control_pixels': 64, 'control_minimum_endpoint_ndvi_exclusive': .6,
            'control_maximum_absolute_ndvi_difference_exclusive': .1,
            'control_full_halo_avoids_catalogue_buffer_and_target_context': True,
            'catalogue_exclusion_buffer_metres': 20, 'no_control_is_valid_negative_outcome': True}.items():
        compare(rule[key], expected, 'Protocol '+key)
    compare(protocol['selection']['maximum_cases'], 1, 'Protocol one-case bound')
    compare(protocol['resources'], {'workers': 1, 'native_threads': 1, 'gpu_used': False,
        'nice_increment': 10, 'worker_rss_mib': 512, 'available_memory_reserve_gib': 1.5,
        'max_seconds_per_worker': 300, 'maximum_cases': 1, 'feature_band_window_requests': 14}, 'Protocol resources')


def feature_integrity(features, feature_valid, endpoints):
    import numpy as np

    require(features.shape == (27, 64, 64) and features.dtype == np.dtype('float32')
            and not np.isinf(features).any()
            and feature_valid.shape == (64, 64) and feature_valid.dtype == np.dtype(bool)
            and np.array_equal(feature_valid, np.isfinite(features).all(axis=0)),
            'Saved model channels or conservative feature validity changed')
    require(np.array_equal(features[18:], features[9:18]-features[:9], equal_nan=True),
            'Saved post-minus-pre signs or scales changed')
    for side, start in (('pre', 0), ('post', 9)):
        endpoint = endpoints[side]
        require(np.allclose(endpoint['rgb'][:, feature_valid], features[[start+2, start+1, start]][:, feature_valid], rtol=0, atol=2e-6)
                and np.allclose(endpoint['ndvi'][feature_valid], features[start+7][feature_valid], rtol=0, atol=2e-6),
                'Frozen cached RGB/NDVI disagree with saved source features')
        for left, right, index in ((3, 2, 7), (4, 5, 8)):
            a, b = features[start+left], features[start+right]
            denominator = a+b
            value = np.divide(a-b, denominator, out=np.zeros_like(a),
                              where=np.isfinite(denominator) & (denominator != 0))
            require(np.allclose(features[start+index][feature_valid], value[feature_valid], rtol=0, atol=2e-6),
                    'Vegetation-index scales differ from original reflectance ratios')


def verify(args, output):
    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    from affine import Affine
    from pyproj import Transformer
    from shapely import make_valid
    from shapely.geometry import box, shape
    from shapely.ops import transform
    from summarize_pilot_overfit import checked_npz
    from verify_validation_dataset import read_cache
    from verify_validation_support_pool import verify_resources
    from prepare_validation_review import validate_validation_pair

    require('torch' not in sys.modules, 'Context verifier must not import Torch')
    protocol_path, directory = args.protocol.resolve(), args.run.resolve()
    protocol, metadata = read_json(protocol_path), read_json(directory/'metadata.json')
    validate_protocol(protocol); protocol_sha = digest(protocol_path)
    require(metadata['execution_status'] == 'completed' and metadata['protocol_sha256'] == protocol_sha,
            'Completed context audit/protocol binding changed')
    require(set(metadata['artifact_sha256']) == {'context_arrays.npz', 'report.md'}, 'Context artifact coverage changed')
    for name, checksum in metadata['artifact_sha256'].items():
        require(digest(within(directory, name)) == checksum, 'Context artifact changed: '+name)
    sources, hashes = {}, {str(protocol_path.relative_to(ROOT)): protocol_sha}
    for role, item in protocol['source_files'].items():
        path = within(ROOT, item['path'])
        require(digest(path) == item['sha256'], 'Predeclared context source changed: '+role)
        sources[role] = path; hashes[item['path']] = item['sha256']
    for name, checksum in protocol['immutable_sources_sha256'].items():
        require(digest(within(ROOT/'Solutions', name)) == checksum, 'Immutable archived source changed: '+name)
    source_names = ('validation_context_worker.py', 'audit_validation_context.py',
        'verify_validation_dataset.py', 'prepare_validation_review.py', 'prepare_curated_pilot.py',
        'curated_pilot_exports.py', 'assess_pixel_quality.py', 'pixelqa_masks.py', 'pixelqa_scene_metadata.py',
        'validate_sample.py', 'annotation_scene_inventory.py', 'train_pilot_baseline.py')
    implementation = {name: digest(within(ROOT/'Solutions', name)) for name in source_names}
    compare(metadata['implementation_sha256'], implementation, 'Context worker implementation')
    if 'implementation_sha256' in protocol:
        compare(protocol['implementation_sha256'], implementation, 'Pre-run implementation lock')
    resources = verify_resources(metadata)
    previous, receipt = read_json(sources['v2_audit']), read_json(sources['v2_verification'])
    previous_metadata = read_json(sources['v2_metadata'])
    require(receipt['verification_status'] == 'passed'
            and receipt['audit_sha256'] == digest(sources['v2_audit'])
            and receipt['audit_metadata_sha256'] == digest(sources['v2_metadata'])
            and receipt['selection_sha256'] == digest(sources['v2_selection'])
            and receipt['protocol_sha256'] == digest(sources['v2_protocol'])
            and previous_metadata['execution_status'] == 'completed', 'The broad-audit candidate proof changed')
    unique = [row for row in previous['records'] if row['eligibility_reasons'] == ['endpoint_crop_qa_flags']]
    require(len(unique) == 1 and unique[0] == protocol['selection']['candidate'], 'The one-case diagnostic selection changed')
    candidate = unique[0]; event_id, pair_id = candidate['event_id'], candidate['pair_id']
    require(candidate['group_id'] not in previous['prior_reviewed_group_ids']
            and event_id not in previous['prior_rejected_event_ids'], 'The diagnostic candidate is not a fresh review location')
    matches = [[row for row in records(sources[key]) if row['pair_id'] == pair_id]
               for key in ('pairs', 'quality', 'quality_index')]
    require(all(len(items) == 1 for items in matches), 'Exact context pair/QA/index coverage changed')
    pair, qa, index = [items[0] for items in matches]
    identity = ('pair_id', 'group_id', 'split', 'tile', 'pre_date', 'post_date', 'anchor_event_ids')
    require(all(pair[key] == qa[key] == index[key] for key in identity)
            and qa['config_id'] == index['config_id'] == protocol['qa_policy_id']
            and qa['quality_status'] == index['quality_status'] == candidate['qa_status']
            and pair['split'] == 'validation' and event_id in pair['anchor_event_ids'],
            'Frozen context pair identity or policy changed')
    compare(metadata['pair'], pair, 'Actual frozen pair')
    compare(metadata['candidate'], candidate, 'Actual selected V2 record')
    for field, value in (('event_id', event_id), ('pair_id', pair_id), ('group_id', pair['group_id']), ('split', 'validation')):
        compare(metadata[field], value, 'Context identity '+field)
    catalogue, splits = read_json(sources['catalogue']), read_json(sources['splits'])
    training = list(records(sources['training_manifest']))
    trained_plan = read_json(ROOT/'Docs/Reviews/PixelPilotV1/experiment_plan.json')
    require(digest(sources['training_manifest']) == trained_plan['selection']['manifest_sha256']
            and digest(sources['training_metadata']) == trained_plan['selection']['dataset_metadata_sha256'],
            'The exact trained-grid isolation reference changed')
    validate_validation_pair(pair, event_id, catalogue, splits, training)
    summary = read_json(sources['quality_summary'])
    require(summary['config_id'] == protocol['qa_policy_id'] and summary['processing_errors'] == []
            and summary['checks'] and all(value is True for value in summary['checks'].values()), 'Frozen QA publication changed')
    factory = within(ROOT, protocol['architecture']['factory_file'])
    architecture = architecture_contract(factory)
    for key, value in architecture.items():
        compare(protocol['architecture'][key], value, 'Predeclared architecture '+key)
    compare(metadata['architecture'], architecture, 'Actual parsed architecture')
    hashes[str(factory.relative_to(ROOT))] = digest(factory)
    trained = read_json(sources['trained_metadata'])
    require(trained['parameter_count'] == 2545
            and trained['implementation_sha256']['train_pilot_baseline.py'] == digest(factory), 'Trained architecture receipt changed')

    shapes = {'pre_qa': (5, 64, 64), 'post_qa': (5, 64, 64),
        'pre_rgb': (3, 64, 64), 'post_rgb': (3, 64, 64), 'pre_ndvi': (64, 64), 'post_ndvi': (64, 64),
        'features': (27, 64, 64)}
    for name in ('feature_valid', 'target', 'positive_interior', 'target_context', 'positive_context',
                 'catalogue_exclusion', 'control_exclusion', 'proposed_control', 'control_context'):
        shapes[name] = (64, 64)
    types = {name: ('float32' if name in ('pre_rgb', 'post_rgb', 'pre_ndvi', 'post_ndvi', 'features') else 'bool') for name in shapes}
    arrays = checked_npz(directory/'context_arrays.npz', shapes, types)
    require(set(metadata['array_sha256']) == set(arrays), 'Context array hash coverage changed')
    for name, value in arrays.items():
        require(hashlib.sha256(value.tobytes()).hexdigest() == metadata['array_sha256'][name], 'Context array hash changed: '+name)
    endpoints, expected_caches = {}, {}
    for side in ('pre', 'post'):
        expected = qa['per_date'][side]['observation']
        require(expected['scene_id'] == pair[side+'_scene_id'] and expected['grid'] == pair['grid']
                and expected['config_id'] == protocol['qa_policy_id'], 'Endpoint scene/grid/policy changed')
        cache_path = within(ROOT, expected['cache_path']); sha = digest(cache_path)
        require(sources[side+'_cache'] == cache_path, 'Predeclared endpoint cache differs from the exact pair')
        masks = read_cache(expected, protocol['qa_policy_id'])
        with np.load(cache_path, allow_pickle=False) as cache:
            rgb, ndvi = cache['rgb'].copy(), cache['ndvi'].copy()
        require(digest(cache_path) == sha
                and np.array_equal(arrays[side+'_qa'], np.stack([masks[name] for name in MASK_NAMES]))
                and np.array_equal(arrays[side+'_rgb'], rgb, equal_nan=True)
                and np.array_equal(arrays[side+'_ndvi'], ndvi, equal_nan=True), 'Saved endpoint fields differ from frozen source cache')
        hashes[expected['cache_path']] = sha
        expected_caches[side] = {'path': expected['cache_path'], 'sha256': sha, 'observation': expected}
        endpoints[side] = {**masks, 'rgb': rgb, 'ndvi': ndvi}
    compare(metadata['cache_sources'], expected_caches, 'Actual endpoint cache sources')
    feature_integrity(arrays['features'], arrays['feature_valid'], endpoints)
    source_stats = metadata['feature_source_file_stats']
    require(isinstance(source_stats, dict) and source_stats, 'Read source fingerprints are missing')
    for name, expected in source_stats.items():
        path = within(ROOT, name); stat = path.stat()
        require([stat.st_size, stat.st_mtime_ns] == expected
                and summary['source_file_stats'].get(name) == expected, 'Feature source fingerprints changed')

    qa_path = within(ROOT, index['batch_path']+'/'+index['qa_mask_file'])
    require(sources['published_qa'] == qa_path, 'Predeclared published QA differs from the exact pair')
    hashes[str(qa_path.relative_to(ROOT))] = digest(qa_path)
    with rasterio.open(qa_path) as source:
        require(source.shape == (64, 64) and source.count == 6 and source.dtypes == ('uint8',)*6
                and source.transform == Affine(*pair['grid']['transform'])
                and source.crs == rasterio.crs.CRS.from_user_input(pair['grid']['crs']), 'Published context QA grid changed')
        published = source.read()
    require(np.isin(published, (0, 1)).all()
            and np.array_equal(published[:5].astype(bool), arrays['pre_qa'] | arrays['post_qa']), 'Published endpoint QA differs from actual caches')
    projection = Transformer.from_crs('EPSG:4326', pair['grid']['crs'], always_xy=True).transform
    projected = {feature['id']: transform(projection, shape(feature['geometry'])) for feature in catalogue['features']}
    grid = pair['grid']; affine = Affine(*grid['transform'])
    anchor = rasterize([(projected[event_id], 1)], out_shape=(64, 64), transform=affine,
                                         fill=0, dtype='uint8', all_touched=False).astype(bool)
    target = anchor & published[5].astype(bool)
    positive = np.zeros_like(target)
    positive[1:-1, 1:-1] = np.logical_and.reduce([target[1+dy:63+dy, 1+dx:63+dx]
                                                for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    require(np.array_equal(arrays['target'], target) and np.array_equal(arrays['positive_interior'], positive)
            and int(target.sum()) == candidate['target_pixels'] and int(positive.sum()) == candidate['positive_interior_pixels']
            and int(positive.sum()) >= 4
            and hashlib.sha256(target.tobytes()).hexdigest() == candidate['target_mask_sha256']
            and hashlib.sha256(positive.tobytes()).hexdigest() == candidate['positive_interior_sha256'],
            'Saved target/interior differs from frozen geometry and published temporal reference')
    halo_masks, contexts = {}, {}
    for name, roi, saved in (('target', target, 'target_context'),
                             ('positive_interior', positive, 'positive_context')):
        halo, _ = square_halo(roi)
        require(np.array_equal(arrays[saved], halo), 'Saved spatial dependencies changed: '+name)
        halo_masks[name] = halo
        contexts[name] = inspect_context(roi, arrays['pre_qa'], arrays['post_qa'], arrays['features'], arrays['feature_valid'])
    core = box(*grid['bounds']); geometries = []
    for geometry in projected.values():
        require(len(geometry.bounds) == 4 and all(math.isfinite(v) for v in geometry.bounds), 'Catalogue projected geometry is invalid')
        if box(*geometry.bounds).distance(core) > 20:
            continue
        repaired = make_valid(geometry) if not geometry.is_valid else geometry
        buffered = repaired.buffer(20)
        geometries.append(make_valid(buffered) if not buffered.is_valid else buffered)
    excluded = rasterize([(g, 1) for g in geometries], out_shape=(64, 64),
        transform=affine, fill=0, dtype='uint8', all_touched=True).astype(bool) if geometries else np.zeros((64, 64), dtype=bool)
    require(np.array_equal(arrays['catalogue_exclusion'], excluded)
            and np.array_equal(arrays['control_exclusion'], excluded | halo_masks['target']), 'Catalogue/context control exclusion changed')
    control, proposal = independent_control(halo_masks['target'], excluded, arrays['pre_qa'], arrays['post_qa'],
        arrays['pre_ndvi'], arrays['post_ndvi'], arrays['features'], arrays['feature_valid'])
    require(np.array_equal(arrays['proposed_control'], control), 'Proposed control differs from fixed independent first-match search')
    compare(metadata['proposal_metadata'], proposal, 'Actual numerical control proposal')
    control_halo, _ = square_halo(control) if control.any() else (np.zeros_like(control), False)
    require(np.array_equal(arrays['control_context'], control_halo), 'Saved control input dependencies changed')
    contexts['control'] = inspect_context(control, arrays['pre_qa'], arrays['post_qa'], arrays['features'], arrays['feature_valid'])
    compare(metadata['contexts'], contexts, 'All independently reconstructed contexts')
    eligible = bool(proposal['proposal_found'] and all(value['context_clear'] for value in contexts.values()))
    compare(metadata['qa_eligible_for_visual_review'], eligible, 'Numeric context eligibility')
    compare(metadata['numeric_context_passed'], eligible, 'Numeric context pass alias')
    crop_qa = {side: {name: int(endpoints[side][name].sum()) for name in MASK_NAMES}
              | {'unusable_pixels': int(np.logical_or.reduce([endpoints[side][name] for name in MASK_NAMES]).sum())}
              for side in endpoints}
    compare(metadata['full_crop_qa'], crop_qa, 'Retained original full-core flags')
    for field, value in (('training_approved', False), ('evaluation_approved', False), ('labels_approved', 0),
            ('model_executions', 0), ('validation_predictions_run', False), ('torch_imported', False),
            ('raw_feature_band_window_reads', 14), ('source_jp2_band_window_reads', 14),
            ('feature_band_window_requests', 14),
            ('cloud_model_executions', 0), ('new_imagery_reviewed', 0), ('context_radius_pixels', 2)):
        compare(metadata[field], value, 'Audit receipt '+field)
    compare(metadata['source_input_sha256'], hashes, 'All bound source hashes')
    require(all(digest(within(ROOT, name)) == checksum for name, checksum in hashes.items())
            and all(digest(within(ROOT/'Solutions', name)) == checksum for name, checksum in implementation.items()),
            'Context evidence changed during independent verification')
    for name, expected in source_stats.items():
        stat = within(ROOT, name).stat()
        require([stat.st_size, stat.st_mtime_ns] == expected, 'Input source changed during verification')
    require('torch' not in sys.modules, 'Independent context verification unexpectedly imported Torch')
    diagnostics = control_gate_counts(arrays)
    passing_windows = sum(row['all_gates_pass_squares'] for row in diagnostics['per_size'])
    require((passing_windows > 0) == proposal['proposal_found'],
            'Report-only gate diagnostics disagree with the unchanged control outcome')
    result = {'verification_status': 'passed', 'event_id': event_id, 'pair_id': pair_id,
        'scope': protocol['scope'], 'protocol_sha256': protocol_sha,
        'audit_metadata_sha256': digest(directory/'metadata.json'),
        'audit_artifact_sha256': metadata['artifact_sha256'], 'contexts': contexts,
        'proposal_metadata': proposal, 'qa_eligible_for_visual_review': eligible, 'numeric_context_passed': eligible,
        'training_approved': False, 'evaluation_approved': False, 'labels_approved': 0,
        'architecture': architecture, 'run_resources': resources,
        'control_search_diagnostics': diagnostics,
        'source_jp2_pixel_reads': 0, 'model_executions': 0, 'cloud_model_executions': 0,
        'new_imagery_reviewed': 0, 'torch_imported': False,
        'verification_implementation_sha256': digest(Path(__file__)),
        'limitations': ['Numeric QA and NDVI proposals do not approve forest stability or temporal loss labels.',
                        'Source feature values were not independently re-extracted from satellite bands.',
                        'The two-pixel context contract belongs only to the exact frozen tiny CNN.']}
    lines = ['# Independently verified input-context audit', '',
             f"Verified `{event_id}` / `{pair_id}`. Eligible for visual review: **{eligible}**.",
             '**No labels, training, model prediction or accuracy estimate were produced.**', '',
             '| Region | ROI pixels | Context pixels | Full context inside | Clear/finite context |',
             '| --- | ---: | ---: | --- | --- |']
    for name, context in contexts.items():
        lines.append(f"| {name} | {context['roi_pixels']} | {context['context_pixels']} | {context['full_context_inside_grid']} | {context['context_clear']} |")
    lines += ['', 'Independent NumPy mask dilation and prefix-sum control search reconstruct the producer decisions.',
              'Factory AST, source/caches, all saved feature channels, source fingerprints and original QA flags are bound.',
              'Raw source bands were read by the audit worker; this verification did not re-extract them.', '', *result['limitations'],
              '', '## Report-only control gate diagnostics', '',
              'These counts describe every square already covered by the fixed search. Marginals overlap; they are not cumulative filtering totals.',
              'Only explicitly named composite columns count gates together. No threshold, search order, proposal or approval changed.', '',
              '| ROI width | Inbounds | QA/finite context | QA/finite + exclusions | Pre NDVI >.6 | Post NDVI >.6 | Both >.6 | Delta <.1 | Safe context + both NDVI | Safe context + delta | All gates |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in diagnostics['per_size']:
        keys = ('size_pixels', 'inbounds_squares', 'qa_finite_clear_context_squares',
                'qa_finite_exclusion_clear_context_squares', 'pre_ndvi_above_point6_roi_squares',
                'post_ndvi_above_point6_roi_squares', 'both_ndvi_above_point6_roi_squares',
                'absolute_ndvi_change_below_point1_roi_squares', 'safe_context_and_both_ndvi_roi_squares',
                'safe_context_and_stable_change_roi_squares', 'all_gates_pass_squares')
        lines.append('| '+' | '.join(str(row[key]) for key in keys)+' |')
    (output/'report.md').write_text('\n'.join(lines).rstrip()+'\n')
    return result


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationContextV3/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Use 128–512 MiB RSS, >=1.5GiB reserve and <=300s')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10); metadata = verify(args, args.worker_output)
        import resource
        metadata['worker_high_water_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024), 2)
        (args.worker_output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    else:
        run_guarded(args, script_path=__file__, operation='validation-context-independent-verification', gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
