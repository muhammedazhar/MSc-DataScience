#!/usr/bin/env python3
"""Independently verify a fixed manual-review packet and its observations.

This verifier reads bounded saved arrays/caches and source fingerprints. It does
not extract satellite pixels, execute QA or forest models, or approve labels.
"""

import argparse
from datetime import date, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MASK_NAMES = ('cloud', 'cloud_buffer', 'shadow_risk', 'haze_risk', 'invalid')
TARGET_STATES = {'forest', 'clearing_visible', 'uncertain', 'obscured', 'unavailable'}
CONTROL_STATES = {'stable_forest', 'nonforest_or_changed', 'uncertain', 'obscured', 'unavailable'}


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
    require(path.is_file() and path.stat().st_size <= 32*2**20, 'Missing or oversized manual-review evidence')
    value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON evidence')))
    require(isinstance(value, dict), 'Manual-review evidence must be an object')
    return value


def within(base, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'Relative evidence path required')
    base = Path(base).resolve(); path = (base/relative).resolve()
    require(path.is_relative_to(base) and path.is_file(), 'Evidence path escaped its root or is missing')
    return path


def compare(actual, expected, label):
    require(type(actual) is type(expected), 'Evidence type changed: '+label)
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Evidence fields changed: '+label)
        for key, value in expected.items():
            compare(actual[key], value, label+'.'+key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'Evidence coverage changed: '+label)
        for index, value in enumerate(expected):
            compare(actual[index], value, label+f'[{index}]')
    else:
        require(actual == expected, 'Evidence value changed: '+label)


def square_context(region):
    """Independently union all 25 shifted dependencies, retaining off-grid risk."""
    import numpy as np

    require(region.shape == (64, 64) and region.dtype == np.dtype(bool), 'Boolean64x64 ROI required')
    if not region.any():
        return np.zeros_like(region), False
    expanded = np.zeros((68, 68), dtype=bool)
    for dy in range(5):
        for dx in range(5):
            expanded[dy:dy+64, dx:dx+64] |= region
    context = expanded[2:66, 2:66].copy()
    return context, bool(context.sum() == expanded.sum())


def independent_roster(arrays, maximum=3):
    """Greedy largest/row/column roster; numerical finiteness, never greenness."""
    import numpy as np

    require(type(maximum) is int and maximum == 3, 'Exactly the predeclared three-slot bound is required')
    bad = (np.any(arrays['pre_qa'] | arrays['post_qa'], axis=0)
           | ~arrays['feature_valid'] | ~np.isfinite(arrays['features']).all(axis=0)
           | ~np.isfinite(arrays['pre_rgb']).all(axis=0)
           | ~np.isfinite(arrays['post_rgb']).all(axis=0)
           | ~np.isfinite(arrays['pre_ndvi']) | ~np.isfinite(arrays['post_ndvi'])
           | arrays['control_exclusion'])
    def prefix(mask):
        return np.pad(mask.astype('int64').cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums = prefix(bad)
    rois, contexts, records = [], [], []
    for size in (16, 12, 8):
        for row in range(2, 63-size):
            for col in range(2, 63-size):
                # Prefix sums are independently rebuilt after a committed slot.
                r, c, width = row-2, col-2, size+4
                count = sums[r+width, c+width]-sums[r, c+width]-sums[r+width, c]+sums[r, c]
                if count:
                    continue
                roi = np.zeros((64, 64), dtype=bool); roi[row:row+size, col:col+size] = True
                context, inside = square_context(roi)
                require(inside and not (bad & context).any(), 'Prefix search context mismatch')
                identifier = 'control-'+str(len(rois)+1)
                rois.append(roi); contexts.append(context)
                records.append({'control_id': identifier, 'row': row, 'col': col,
                    'height': size, 'width': size, 'pixels': size*size,
                    'context_radius_pixels': 2, 'context_pixels': (size+4)**2,
                    'selection_order': len(rois), 'visual_review': 'unreviewed',
                    'evaluation_approved': False, 'training_approved': False})
                bad |= context
                sums = prefix(bad)
                if len(rois) == maximum:
                    return np.stack(rois), np.stack(contexts), records
    shape = (0, 64, 64)
    return (np.stack(rois) if rois else np.zeros(shape, dtype=bool),
            np.stack(contexts) if contexts else np.zeros(shape, dtype=bool), records)


def region_counts(planes, region):
    """Count each frozen flag and its union with all ROI pixels as denominator."""
    import numpy as np

    denominator = int(region.sum())
    counts = {name: int(planes[index][region].sum()) for index, name in enumerate(MASK_NAMES)}
    union = int(np.any(planes, axis=0)[region].sum())
    result = {'region_pixels': denominator, 'denominator_pixels': denominator,
              'valid_pixels': denominator-counts['invalid'], 'usable_pixels': denominator-union,
              'unusable_pixels': union}
    result.update({name+'_pixels': value for name, value in counts.items()})
    result.update({name+'_fraction': result[name+'_pixels']/denominator if denominator else None
                   for name in (*MASK_NAMES, 'valid', 'usable', 'unusable')})
    return result


def observation_metrics(planes, rgb, ndvi, roi):
    import numpy as np

    require(planes.shape == (5, 64, 64) and planes.dtype == np.dtype(bool), 'Five boolean QA planes required')
    require(rgb.shape == (3, 64, 64) and ndvi.shape == (64, 64), 'Display arrays changed shape')
    context, inside = square_context(roi)
    nonfinite = ~np.isfinite(rgb).all(axis=0) | ~np.isfinite(ndvi)
    roi_metrics, context_metrics = region_counts(planes, roi), region_counts(planes, context)
    invalid = int(nonfinite[context].sum())
    return {'roi_pixels': int(roi.sum()), 'context_pixels': int(context.sum()),
            'full_context_inside_grid': inside, 'roi': roi_metrics, 'context': context_metrics,
            'roi_nonfinite_rgb_ndvi_pixels': int(nonfinite[roi].sum()),
            'context_nonfinite_rgb_ndvi_pixels': invalid,
            'context_clear': bool(inside and invalid == 0 and context_metrics['unusable_pixels'] == 0)}


def review_decision(packet, review):
    """Validate complete statements and derive support without approving labels."""
    rows, controls, pair = packet['timeline'], packet['controls'], packet['pair']
    require(review.get('used_model_predictions') is False, 'Manual review must be prediction-blind')
    require(type(review.get('labels_approved')) is int and review['labels_approved'] == 0 and review.get('training_approved') is False
            and review.get('evaluation_approved') is False, 'Manual evidence must not approve labels or evaluation')
    for name in ('reviewer', 'notes'):
        require(isinstance(review.get(name), str) and review[name].strip(), 'Explicit review '+name+' required')
    inspected = review.get('observations')
    require(isinstance(inspected, list) and len(inspected) == len(rows), 'Review must cover the complete timeline')
    identifiers = [record['control_id'] for record in controls]
    for source, statement in zip(rows, inspected):
        require(statement.get('scene_id') == source['scene_id'] and statement.get('date') == source['date'],
                'Review date/scene order differs from the prepared complete timeline')
        require(statement.get('target_state') in TARGET_STATES and isinstance(statement.get('notes'), str)
                and statement['notes'].strip(), 'Explicit target state and date notes required')
        states = statement.get('controls')
        require(isinstance(states, dict) and set(states) == set(identifiers)
                and all(state in CONTROL_STATES for state in states.values()), 'Every locked control requires a state on every date')
        available = source.get('observation') is not None
        target_state = statement['target_state']
        if available:
            require(target_state != 'unavailable' and all(state != 'unavailable' for state in states.values()),
                    'Prepared observations cannot be called unavailable')
        else:
            require(target_state == 'unavailable' and all(state == 'unavailable' for state in states.values()),
                    'Missing observations must remain unavailable')
        if target_state in ('forest', 'clearing_visible'):
            require(available and source['target_metrics']['context_clear'] is True,
                    'Clear target claim intersects flagged or nonfinite full context')
        for index, identifier in enumerate(identifiers):
            if states[identifier] in ('stable_forest', 'nonforest_or_changed'):
                require(available and source['control_metrics'][index]['context_clear'] is True,
                        'Definite control claim intersects flagged or nonfinite full context')
    by_scene = {row['scene_id']: row for row in inspected}
    pre, post = by_scene[pair['pre_scene_id']], by_scene[pair['post_scene_id']]
    clearings = [row for row in inspected if row['target_state'] == 'clearing_visible']
    first = clearings[0] if clearings else None
    positive_pixels = packet.get('positive_interior_pixels', packet.get('candidate', {}).get('positive_interior_pixels'))
    require(type(positive_pixels) is int and positive_pixels >= 4, 'Original minimum positive interior is required')
    target_supported = bool(positive_pixels >= 4
        and pre['target_state'] == 'forest' and post['target_state'] == 'clearing_visible'
        and any(row['date'] > pair['post_date'] for row in clearings)
        and first is not None and not any(row['target_state'] == 'forest' and row['date'] >= first['date'] for row in inspected))
    dispositions = []
    for identifier in identifiers:
        stable = [row for row in inspected if row['controls'][identifier] == 'stable_forest']
        supported = bool(pre['controls'][identifier] == post['controls'][identifier] == 'stable_forest'
            and len({row['date'] for row in stable}) >= 3
            and any(row['date'] > pair['post_date'] for row in stable)
            and not any(row['controls'][identifier] == 'nonforest_or_changed' for row in inspected))
        dispositions.append({'control_id': identifier, 'decision_status': 'supported' if supported else 'rejected'})
    selected = next((row['control_id'] for row in dispositions if row['decision_status'] == 'supported'), None)
    supported = bool(target_supported and selected is not None)
    interval = None
    if target_supported:
        forests = [row for row in inspected if row['target_state'] == 'forest' and row['date'] < first['date']]
        require(forests, 'Supported temporal change needs a prior observed forest bound')
        interval = {'lower_bound_exclusive': forests[-1]['date'], 'upper_bound_inclusive': first['date'],
                    'width_days': (date.fromisoformat(first['date'])-date.fromisoformat(forests[-1]['date'])).days,
                    'interpretation': 'Available inspected image interval, not exact felling time or global first visibility'}
    return {'decision_status': 'manual_evidence_supported' if supported else 'rejected',
            'target_evidence_supported': target_supported, 'selected_control_id': selected if supported else None,
            'candidate_controls': dispositions, 'observed_change_interval': interval if supported else None,
            'labels_approved': 0, 'training_approved': False, 'evaluation_approved': False}


def validate_protocol(protocol):
    require(protocol.get('schema_version') == 4
            and protocol.get('status') == 'predeclared_manual_review_before_new_imagery_or_validation_predictions',
            'The separate predeclared manual-review protocol is required')
    expected = {'context_radius_pixels': 2, 'control_sizes': [16, 12, 8],
        'maximum_control_candidates': 3, 'minimum_control_pixels': 64,
        'minimum_positive_interior_pixels': 4, 'require_full_target_and_interior_context': True,
        'require_full_halo_inside_core': True, 'zero_all_five_qa_at_both_endpoints': True,
        'finite_all27_endpoint_context': True, 'finite_rgb_ndvi_each_clear_visual_context': True,
        'catalogue_exclusion_buffer_metres': 20,
        'control_full_context_avoids_target_context_and_catalogue': True,
        'control_contexts_mutually_disjoint': True, 'no_ndvi_threshold_or_ranking': True,
        'roster_committed_before_images': True, 'no_roi_shrink_redraw_or_replacement': True,
        'v3_outcome_unchanged': True}
    for key, value in expected.items():
        compare(protocol['rule'][key], value, 'Protocol '+key)
    require(protocol['selection']['maximum_cases'] == 1 and protocol['selection']['replacement_cases'] == 0,
            'The fixed one-case, no-replacement limit changed')
    compare(protocol['resources'], {'workers': 1, 'native_threads': 1, 'torch_threads': 1,
        'gpu_used': False, 'nice_increment': 10, 'worker_rss_mib': 512,
        'available_memory_reserve_gib': 1.5, 'max_seconds_per_worker': 300,
        'preparation_batch_size': 1}, 'Mac capacity contract')
    require(protocol['review']['used_model_predictions'] is False
            and protocol['review']['label_approval'] is False
            and protocol['review']['training_approved'] is False
            and protocol['review']['evaluation_approved'] is False
            and protocol['review']['dataset_export'] is False
            and protocol['review']['validation_inference'] is False, 'Manual scope cannot approve labels or inference')
    require(protocol['cache_preparation']['existing_caches'] == 8
            and protocol['cache_preparation']['expected_new_observations'] == 4
            and protocol['cache_preparation']['new_endpoint_feature_band_window_reads'] == 0
            and protocol['cache_preparation']['expected_quality_fused_band_window_requests'] == 44
            and protocol['cache_preparation']['expected_quality_include_feature_window_requests'] == 28,
            'The exact bounded cache preparation contract changed')
    plan = protocol['timeline']
    require(len(plan) == 12 and all(row['metadata_usable'] is True for row in plan)
            and sum(not row['cache_exists'] for row in plan) == 4
            and len({row['scene_id'] for row in plan}) == len({row['date'] for row in plan}) == 12,
            'The exact twelve-date cache plan changed')


def source_identity(plan, grid, policy):
    """Reconstruct source/cache keys from XML, file stats and quality-file hashes."""
    from pixelqa_scene_metadata import qa_scene_metadata

    scene = ROOT/plan['scene_path']
    require(scene.is_dir() and scene.resolve().is_relative_to(ROOT.resolve()), 'Scene source escaped project')
    metadata = qa_scene_metadata(scene, ROOT)
    require(metadata['metadata_usable'] is True and metadata['scene_id'] == plan['scene_id']
            and metadata['acquisition_date'] == plan['date'] and metadata['metadata_uid'] == plan['metadata_uid'],
            'Timeline XML metadata differs from its frozen plan')
    raster_paths = sorted(set(metadata['cloud_band_paths'].values()) | set(metadata['feature_band_paths'].values()))
    raster_stats = {}
    for name in raster_paths:
        info = within(ROOT, name).stat(); raster_stats[name] = [info.st_size, info.st_mtime_ns]
    quality_paths = [metadata['quality_paths']['cloud']]
    quality_paths += [name for kind in ('nodata', 'satura', 'defect', 'tecqua') for name in metadata['quality_paths'][kind]]
    source_id = canonical({'metadata_uid': metadata['metadata_uid'], 'raster_stats': raster_stats,
                           'quality_hashes': {name: digest(within(ROOT, name)) for name in quality_paths}})
    key = canonical({'source': source_id, 'grid': grid, 'config_id': policy})
    require(source_id == plan['source_id'] and key == plan['cache_key']
            and Path(plan['cache_path']).name == key+'.npz', 'Independent cache source/grid/policy identity changed')
    metadata_paths = [str(path.relative_to(ROOT)) for path in scene.glob('GRANULE/*/MTD_TL.xml')]
    metadata_paths += [str((scene/'MTD_MSIL1C.xml').relative_to(ROOT))]
    statistics = {}
    for name in (*raster_paths, *quality_paths, *metadata_paths):
        info = within(ROOT, name).stat(); statistics[name] = [info.st_size, info.st_mtime_ns]
    compare(statistics, plan['source_file_stats'], 'All timeline source fingerprints')
    return statistics


def checked_arrays(directory, timeline_count, control_count):
    from summarize_pilot_overfit import checked_npz

    shapes = {name: (64, 64) for name in ('target', 'positive_interior', 'target_context',
        'positive_context', 'catalogue_exclusion', 'control_exclusion', 'feature_valid')}
    shapes.update(qa=(timeline_count, 5, 64, 64), rgb=(timeline_count, 3, 64, 64),
                  ndvi=(timeline_count, 64, 64), features=(27, 64, 64),
                  controls=(control_count, 64, 64), control_contexts=(control_count, 64, 64))
    types = {name: 'float32' if name in ('rgb', 'ndvi', 'features') else 'bool' for name in shapes}
    return checked_npz(directory/'observations.npz', shapes, types)


def display_pixels(rgb, qa=None):
    """Independent numerical rendering contract; no per-image contrast fitting."""
    import numpy as np

    values = np.moveaxis(rgb, 0, -1)
    pixels = (np.clip(np.nan_to_num(values, nan=0, posinf=0, neginf=0)/.30, 0, 1)*255).astype('uint8')
    if qa is not None:
        colors = ((255, 255, 255), (255, 170, 35), (180, 70, 255), (40, 210, 255), (255, 30, 70))
        for index, color in enumerate(colors):
            pixels[qa[index]] = ((pixels[qa[index]].astype('float64')+np.asarray(color))/2).astype('uint8')
    return pixels


def expected_render_views(packet, arrays):
    import numpy as np

    scenes = [row['scene_id'] for row in packet['timeline']]
    require(len(scenes) == 12, 'Exactly three complete four-date view pages are required')
    modes = ['plain', 'overlay', 'qa']
    views = [{'file': f'timeline-{index+1:02d}.png', 'scene_ids': scenes[index*4:index*4+4],
              'kinds': modes, 'width': 1016, 'height': 1541} for index in range(3)]
    endpoints = [packet['pair'][side+'_scene_id'] for side in ('pre', 'post')]
    views.append({'file': 'pair.png', 'scene_ids': endpoints, 'kinds': modes, 'width': 816, 'height': 1363})
    yy, xx = np.nonzero(arrays['target_context'])
    require(yy.size > 0, 'Target detail cannot display an empty context')
    side = min(64, max(int(yy.max()-yy.min()+1), int(xx.max()-xx.min()+1))+4)
    row = max(0, min(int(yy.min())-2, 64-side)); col = max(0, min(int(xx.min())-2, 64-side))
    window = {'row_start': row, 'row_stop_exclusive': row+side, 'col_start': col, 'col_stop_exclusive': col+side}
    views.append({'file': 'target-detail.png', 'scene_ids': endpoints, 'kinds': ['plain', 'overlay'],
                  'width': 816, 'height': 958, 'source_window': window})
    return views


def verify_render_views(directory, packet, arrays):
    """Bind every shown date to actual saved RGB/QA with independent pixel checks."""
    import numpy as np
    from PIL import Image

    views = expected_render_views(packet, arrays)
    compare(packet['render_views'], views, 'All fixed image dates, kinds and dimensions')
    require(set(packet['artifact_sha256']) == {'observations.npz', *(row['file'] for row in views)},
            'Every fixed timeline, endpoint and target-detail image must remain bound')
    scenes = {row['scene_id']: index for index, row in enumerate(packet['timeline'])}
    checked_plain, checked_qa = 0, 0
    for view in views:
        path = within(directory, view['file'])
        require(path.stat().st_size <= 8*2**20, 'Review PNG exceeds its bounded size')
        with Image.open(path) as image:
            require(image.format == 'PNG' and image.mode == 'RGB'
                    and image.size == (view['width'], view['height']), 'Review PNG header/type/dimensions changed')
            for position, identifier in enumerate(view['scene_ids']):
                index = scenes[identifier]
                plain = display_pixels(arrays['rgb'][index])
                if view['file'].startswith('timeline-'):
                    size, x, y = 320, 14, 116+position*358
                    expected = np.repeat(np.repeat(plain, 5, axis=0), 5, axis=1)
                    qa_x, qa_y = 682, y
                elif view['file'] == 'pair.png':
                    size, x, y = 384, 16+position*400, 115
                    expected = np.repeat(np.repeat(plain, 6, axis=0), 6, axis=1)
                    qa_x, qa_y = x, 951
                else:
                    size, x, y = 384, 16+position*400, 96
                    window = view['source_window']
                    crop = plain[window['row_start']:window['row_stop_exclusive'],
                                 window['col_start']:window['col_stop_exclusive']]
                    # Pixel-centre nearest sampling is independent of PIL.resize.
                    indices = np.floor((np.arange(size)+.5)*crop.shape[0]/size).astype('int64')
                    expected = crop[indices[:, None], indices[None, :]]
                    qa_x, qa_y = None, None
                actual = np.asarray(image.crop((x, y, x+size, y+size)))
                if not np.array_equal(actual, expected):
                    row, col = np.argwhere(np.any(actual != expected, axis=2))[0]
                    raise ValueError(f"Shown plain RGB differs: {view['file']} {identifier}, "
                        f"panel row {row} col {col}; canvas row {y+row} col {x+col}")
                checked_plain += 1
                if qa_x is not None:
                    pixels = display_pixels(arrays['rgb'][index], arrays['qa'][index])
                    multiple = size//64
                    expected_qa = np.repeat(np.repeat(pixels, multiple, axis=0), multiple, axis=1)
                    actual_qa = np.asarray(image.crop((qa_x, qa_y, qa_x+size, qa_y+size)))
                    if not np.array_equal(actual_qa, expected_qa):
                        row, col = np.argwhere(np.any(actual_qa != expected_qa, axis=2))[0]
                        raise ValueError(f"Shown QA differs: {view['file']} {identifier}, "
                            f"panel row {row} col {col}; canvas row {qa_y+row} col {qa_x+col}")
                    checked_qa += 1
    return {'timeline_pages_checked': 3, 'views_checked': 5,
            'plain_rgb_panels_checked': checked_plain, 'qa_panels_checked': checked_qa,
            'overlays_scope': 'Image hashes and exact saved target/control/context masks are bound; plain and QA panels independently reconstructed.'}


def bound_review(packet, review, metadata_path, protocol_sha):
    require(review.get('schema_version') == 4 and review.get('protocol_sha256') == protocol_sha
            and review.get('prepared_metadata_sha256') == digest(metadata_path)
            and review.get('artifact_sha256') == packet['artifact_sha256'],
            'Review changed its exact protocol/prepared evidence bindings')
    decision = review_decision(packet, review)
    for key in ('decision_status', 'selected_control_id'):
        compare(review.get(key), decision[key], 'Independent review '+key)
    for key in ('target_evidence_supported', 'observed_change_interval'):
        if key in review:
            compare(review[key], decision[key], 'Independent review '+key)
    controls = review.get('candidate_controls')
    require(isinstance(controls, list) and len(controls) == len(decision['candidate_controls']),
            'Every fixed control requires an explicit disposition')
    for actual, expected in zip(controls, decision['candidate_controls']):
        require(all(actual.get(key) == value for key, value in expected.items())
                and isinstance(actual.get('notes'), str) and actual['notes'].strip(),
                'Control disposition differs from its complete observation evidence')
    if decision['decision_status'] == 'rejected':
        reasons = review.get('rejection_reasons')
        require(isinstance(reasons, list) and reasons and all(isinstance(reason, str) and reason.strip() for reason in reasons),
                'Rejected review must retain explicit reasons')
    return decision


def verify(args, output):
    import numpy as np
    from summarize_pilot_overfit import checked_npz
    from verify_validation_dataset import read_cache, records
    from verify_validation_support_pool import verify_resources
    from prepare_validation_review import validate_validation_pair

    require('torch' not in sys.modules, 'Manual verifier must not import Torch')
    protocol_path, directory = args.protocol.resolve(), args.run.resolve()
    protocol, packet = read_json(protocol_path), read_json(directory/'metadata.json')
    validate_protocol(protocol); protocol_sha = digest(protocol_path)
    require(packet['schema_version'] == 4 and packet['mode'] == 'prepare_validation_manual'
            and packet['execution_status'] == 'completed' and packet['protocol_sha256'] == protocol_sha,
            'Completed packet/protocol binding changed')
    resources = verify_resources(packet)
    source_paths, expected_hashes = {}, {str(protocol_path.relative_to(ROOT)): protocol_sha}
    for role, item in protocol['source_files'].items():
        path = within(ROOT, item['path'])
        require(digest(path) == item['sha256'], 'Predeclared source changed: '+role)
        source_paths[role] = path; expected_hashes[item['path']] = item['sha256']
    compare(packet['source_input_sha256'], expected_hashes, 'Prepared source bindings')
    for name, checksum in protocol['immutable_sources_sha256'].items():
        require(digest(within(ROOT/'Solutions', name)) == checksum, 'Frozen source changed: '+name)
    require(packet['implementation_sha256'] == protocol['implementation_sha256'], 'Predeclared packet implementation changed')
    for name, checksum in packet['implementation_sha256'].items():
        require(digest(within(ROOT/'Solutions', name)) == checksum, 'Packet implementation source changed: '+name)
    v3 = read_json(source_paths['v3_audit_metadata']); v3_verification = read_json(source_paths['v3_verification'])
    require(v3_verification['verification_status'] == 'passed'
            and v3_verification['audit_metadata_sha256'] == digest(source_paths['v3_audit_metadata'])
            and v3['numeric_context_passed'] is False and v3['qa_eligible_for_visual_review'] is False
            and v3_verification['numeric_context_passed'] is False
            and v3['artifact_sha256']['context_arrays.npz'] == digest(source_paths['v3_arrays']),
            'Previously verified negative V3 result changed')
    candidate, pair = protocol['candidate'], protocol['pair']
    compare(v3['candidate'], candidate, 'Original V3 candidate')
    compare(v3['pair'], pair, 'Original V3 pair')
    compare(packet['candidate'], candidate, 'Prepared candidate')
    compare(packet['pair'], pair, 'Prepared pair')
    compare(packet['architecture'], v3['architecture'], 'Frozen local-context architecture')
    for key in ('event_id', 'pair_id', 'group_id', 'split'):
        compare(packet[key], candidate[key], 'Packet identity '+key)
    require(packet['qa_policy_id'] == protocol['qa_policy_id'], 'Frozen manual-review QA policy changed')
    frozen_pairs = [row for row in records(source_paths['candidate_pairs']) if row['pair_id'] == pair['pair_id']]
    require(len(frozen_pairs) == 1 and frozen_pairs[0] == pair, 'Frozen candidate pair join changed')
    catalogue, splits = read_json(source_paths['catalogue']), read_json(source_paths['splits'])
    training = list(records(source_paths['training_manifest']))
    validate_validation_pair(pair, candidate['event_id'], catalogue, splits, training)
    summary = read_json(source_paths['quality_summary'])
    require(summary['config_id'] == protocol['qa_policy_id'], 'Frozen QA summary policy changed')
    for key, expected in {'model_executions': 0, 'forest_model_executions': 0,
            'cloud_model_executions': 4, 'new_observations': 4, 'cache_hits': 8,
            'new_pair_feature_extractions': 0, 'source_jp2_band_window_requests_for_new_qa': 44,
            'source_jp2_include_feature_window_requests_for_new_qa': 28,
            'model_predictions_read': False, 'v3_numeric_context_passed': False,
            'control_search_uses_ndvi_thresholds_or_ranking': False, 'torch_imported': False,
            'new_imagery_reviewed': 0, 'labels_approved': 0,
            'training_approved': False, 'evaluation_approved': False}.items():
        compare(packet[key], expected, 'Preparation receipt '+key)
    controls = packet['controls']
    require(isinstance(controls, list) and len(controls) <= 3, 'Bounded locked control roster required')
    timeline = packet['timeline']
    require(isinstance(timeline, list) and len(timeline) == 12, 'All twelve observations must remain in the packet')
    arrays = checked_arrays(directory, len(timeline), len(controls))
    require(set(packet['array_sha256']) == set(arrays), 'Array hash coverage changed')
    for name, values in arrays.items():
        require(hashlib.sha256(values.tobytes()).hexdigest() == packet['array_sha256'][name], 'Saved array changed: '+name)
    artifacts = packet['artifact_sha256']
    require(isinstance(artifacts, dict) and 'observations.npz' in artifacts and 'pair.png' in artifacts
            and all(name == 'observations.npz' or name.endswith('.png') for name in artifacts), 'Review artifact coverage changed')
    for name, checksum in artifacts.items():
        require(digest(within(directory, name)) == checksum, 'Prepared review artifact changed: '+name)

    shapes = {name: (64, 64) for name in ('feature_valid', 'target', 'positive_interior',
        'target_context', 'positive_context', 'catalogue_exclusion', 'control_exclusion', 'proposed_control', 'control_context')}
    shapes.update(pre_qa=(5, 64, 64), post_qa=(5, 64, 64), pre_rgb=(3, 64, 64),
                  post_rgb=(3, 64, 64), pre_ndvi=(64, 64), post_ndvi=(64, 64), features=(27, 64, 64))
    types = {name: 'float32' if name in ('pre_rgb', 'post_rgb', 'pre_ndvi', 'post_ndvi', 'features') else 'bool' for name in shapes}
    old = checked_npz(source_paths['v3_arrays'], shapes, types)
    for name in ('target', 'positive_interior', 'target_context', 'positive_context',
                 'catalogue_exclusion', 'control_exclusion', 'feature_valid', 'features'):
        require(np.array_equal(arrays[name], old[name], equal_nan=True), 'Previously verified V3 evidence changed: '+name)
    for name, saved in (('target', 'target_context'), ('positive_interior', 'positive_context')):
        context, inside = square_context(arrays[name])
        require(inside and np.array_equal(context, arrays[saved])
                and arrays['feature_valid'][context].all() and np.isfinite(arrays['features'][:, context]).all(),
                'Full target context changed or lost finite original model inputs')
    target, interior = arrays['target'], arrays['positive_interior']
    eroded = np.zeros_like(target)
    eroded[1:-1, 1:-1] = np.logical_and.reduce([target[1+y:63+y, 1+x:63+x] for y in (-1, 0, 1) for x in (-1, 0, 1)])
    require(np.array_equal(interior, eroded) and int(interior.sum()) == candidate['positive_interior_pixels'] >= 4
            and int(target.sum()) == candidate['target_pixels'], 'Original target/interior support changed')
    rois, contexts, roster = independent_roster(old)
    require(np.array_equal(arrays['controls'], rois) and np.array_equal(arrays['control_contexts'], contexts),
            'Locked control geometry differs from independent prefix-sum selection')
    compare(controls, roster, 'Exact predeclared no-NDVI control roster')
    require(not (arrays['control_exclusion'] != (arrays['catalogue_exclusion'] | arrays['target_context'])).any(),
            'Control exclusions differ from catalogue plus full target context')

    when = date.fromisoformat(candidate['annotation_date'])
    start, end = when-timedelta(days=45), when+timedelta(days=45)
    scenes = [row for row in records(source_paths['scene_inventory']) if row['tile'] == pair['tile']
        and start <= date.fromisoformat(row['acquisition_date']) <= end
        and date.fromisoformat(row['acquisition_date']).year == when.year]
    scenes.sort(key=lambda row: (row['acquisition_date'], row['scene_id']))
    require(len(scenes) == 12, 'Complete frozen local observation universe changed')
    compare(packet['timeline_window'], protocol['timeline_window'], 'Timeline window')
    require(protocol['timeline_window']['start'] == start.isoformat()
            and protocol['timeline_window']['end'] == end.isoformat(), 'Declared timeline bounds changed')
    all_statistics, cache_hashes = {}, {}
    for index, (scene, plan, row) in enumerate(zip(scenes, protocol['timeline'], timeline)):
        require((scene['scene_id'], scene['acquisition_date'], scene['path'], scene['metadata_usable'])
                == (plan['scene_id'], plan['date'], plan['scene_path'], plan['metadata_usable']), 'Timeline plan omitted or changed a scene')
        for key, value in plan.items():
            if key != 'cache_sha256':
                compare(row[key], value, 'Timeline identity '+key)
        source_stats = source_identity(plan, pair['grid'], protocol['qa_policy_id'])
        all_statistics.update(source_stats)
        expected = row['observation']
        require(isinstance(expected, dict) and expected['scene_id'] == plan['scene_id']
                and expected['grid'] == pair['grid'] and expected['metadata_uid'] == plan['metadata_uid']
                and expected['source_id'] == plan['source_id'] and expected['cache_key'] == plan['cache_key']
                and expected['cache_path'] == plan['cache_path'], 'Actual cache metadata differs from independent source binding')
        path = within(ROOT, plan['cache_path']); cache_sha = digest(path)
        require(row['cache_sha256'] == cache_sha and row['cache_created_this_run'] is (not plan['cache_exists']),
                'Cache completion identity differs from the frozen eight-existing/four-missing plan')
        if plan['cache_exists']:
            require(cache_sha == plan['cache_sha256'], 'A previously completed cache changed during preparation')
        else:
            require(plan['cache_sha256'] is None, 'A missing cache had a fabricated original hash')
        masks = read_cache(expected, protocol['qa_policy_id'])
        with np.load(path, allow_pickle=False) as cache:
            rgb, ndvi = cache['rgb'].copy(), cache['ndvi'].copy()
        planes = np.stack([masks[name] for name in MASK_NAMES])
        require(digest(path) == cache_sha and np.array_equal(arrays['qa'][index], planes)
                and np.array_equal(arrays['rgb'][index], rgb, equal_nan=True)
                and np.array_equal(arrays['ndvi'][index], ndvi, equal_nan=True), 'Packet observations differ from bound actual cache fields')
        compare(row['target_metrics'], observation_metrics(planes, rgb, ndvi, target), 'Actual target metrics')
        compare(row['interior_metrics'], observation_metrics(planes, rgb, ndvi, interior), 'Actual interior metrics')
        compare(row['control_metrics'], [observation_metrics(planes, rgb, ndvi, roi) for roi in rois], 'Actual control metrics')
        cache_hashes[plan['cache_path']] = cache_sha
        for side in ('pre', 'post'):
            if scene['scene_id'] == pair[side+'_scene_id']:
                require(np.array_equal(planes, old[side+'_qa']) and np.array_equal(rgb, old[side+'_rgb'], equal_nan=True)
                        and np.array_equal(ndvi, old[side+'_ndvi'], equal_nan=True), 'Endpoint fields differ from verified V3 input evidence')
    compare(packet['observation_cache_sha256'], cache_hashes, 'Complete actual cache hash coverage')
    compare(packet['source_file_stats'], all_statistics, 'Complete timeline source-stat coverage')
    compare(packet['feature_source_file_stats'], v3['feature_source_file_stats'], 'Retained V3 feature-source proof')
    for name, expected in {**all_statistics, **packet['feature_source_file_stats']}.items():
        stat = within(ROOT, name).stat()
        require([stat.st_size, stat.st_mtime_ns] == expected, 'Source fingerprint changed: '+name)
        if name in summary['source_file_stats']:
            require(summary['source_file_stats'][name] == expected, 'Source fingerprint contradicts archived QA')

    render_check = verify_render_views(directory, packet, arrays)
    decision, review_hash, review_canonical = None, None, None
    if args.reviews:
        review = read_json(args.reviews)
        reviewed_packet = {**packet, 'positive_interior_pixels': int(interior.sum())}
        decision = bound_review(reviewed_packet, review, directory/'metadata.json', protocol_sha)
        review_hash, review_canonical = digest(args.reviews), canonical(review)
    require(all(digest(within(ROOT, name)) == checksum for name, checksum in expected_hashes.items())
            and all(digest(within(ROOT, name)) == checksum for name, checksum in cache_hashes.items())
            and 'torch' not in sys.modules, 'Evidence changed during verification or Torch was imported')
    result = {'verification_status': 'passed', 'mode': 'manual_reviews' if args.reviews else 'prepared_packet',
        'event_id': candidate['event_id'], 'pair_id': pair['pair_id'], 'protocol_sha256': protocol_sha,
        'prepared_metadata_sha256': digest(directory/'metadata.json'), 'prepared_metadata_canonical_sha256': canonical(packet),
        'artifact_sha256': artifacts, 'controls': roster, 'timeline_observations_checked': 12,
        'render_verification': render_check,
        'cache_sha256': cache_hashes, 'source_file_stats_checked': len(all_statistics),
        'review_sha256': review_hash, 'review_canonical_sha256': review_canonical, 'decision': decision,
        'source_jp2_pixel_reads': 0, 'model_executions': 0, 'cloud_model_executions': 0,
        'labels_approved': 0, 'training_approved': False, 'evaluation_approved': False,
        'new_imagery_reviewed': 0, 'torch_imported': False, 'preparation_resources': resources,
        'verification_implementation_sha256': digest(Path(__file__)),
        'limitations': ['This verifies evidence consistency and review rules, not independent visual truth.',
                       'Later RGB/NDVI values and frozen QA were not independently re-extracted from satellite bands.',
                       'V3 automatic control eligibility remains false; manual evidence never approves labels here.']}
    lines = ['# Independent manual-review evidence verification', '',
             f"Verified `{candidate['event_id']}` / `{pair['pair_id']}`: all 12 dates and {len(roster)} fixed controls.",
             'No labels, model execution or accuracy estimate were produced.', '',
             'Separate NumPy shifted halos and prefix sums reconstruct the control roster and every full-context quality count.',
             'All actual cache fields, source/cache identities, source fingerprints and original V3 evidence are bound.']
    if decision:
        lines += ['', 'Manual evidence decision: **'+decision['decision_status']+'**.',
                  'Selected control: '+str(decision['selected_control_id'])+'.']
    lines += ['', *result['limitations']]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationManualV4/protocol.json')
    parser.add_argument('--reviews', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Use 128–512MiB RSS, >=1.5GiB reserve and <=300s')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10); metadata = verify(args, args.worker_output)
        import resource
        metadata['worker_high_water_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024), 2)
        (args.worker_output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    else:
        run_guarded(args, script_path=__file__, operation='validation-manual-independent-verification', gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
