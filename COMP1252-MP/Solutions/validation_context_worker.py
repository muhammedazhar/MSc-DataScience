"""One guarded worker for the predeclared local-context validation QA check.

Only this new worker reads the fourteen feature-band windows. Frozen QA caches
are reused without cloud/forest model execution. No label is approved here.
"""

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MASK_NAMES = ('cloud', 'cloud_buffer', 'shadow_risk', 'haze_risk', 'invalid')
SOURCE_NAMES = ('validation_context_worker.py', 'audit_validation_context.py',
                'verify_validation_dataset.py', 'prepare_validation_review.py',
                'prepare_curated_pilot.py', 'curated_pilot_exports.py',
                'assess_pixel_quality.py', 'pixelqa_masks.py', 'pixelqa_scene_metadata.py',
                'validate_sample.py', 'annotation_scene_inventory.py', 'train_pilot_baseline.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 32*2**20, 'Missing or oversized context input')
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite context input')))


def rooted(relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            'Context inputs must be project-relative')
    path = (ROOT/relative).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), 'Context input is missing or escaped the project')
    return path


def _context_result(roi, pre, post, features, feature_valid):
    """Keep incomplete/absent contexts as explicit negative audit results."""
    import numpy as np
    from scipy.ndimage import binary_dilation
    from audit_validation_context import context_mask, qa_region_counts

    require(roi.shape == (64, 64) and roi.dtype == np.dtype(bool), 'A context ROI must retain its boolean grid')
    empty_counts = {side: {'roi_unusable_pixels': 0, 'context_unusable_pixels': 0,
                          'context_components': {name: 0 for name in MASK_NAMES}}
                    for side in ('pre', 'post')}
    if not roi.any():
        return np.zeros_like(roi), {'roi_pixels': 0, 'context_pixels': 0,
            'full_context_inside_grid': False, 'dates': empty_counts,
            'invalid_or_nonfinite_context_pixels': 0, 'context_clear': False}
    try:
        context = context_mask(roi, radius=2)
        inside = True
    except ValueError:
        # A clipped array can describe the audit evidence, but cannot be used
        # to claim that a missing dependency is clear or approve the case.
        context = binary_dilation(roi, structure=np.ones((5, 5), dtype=bool))
        inside = False
    roi_counts = qa_region_counts(roi, pre, post)
    context_counts = qa_region_counts(context, pre, post)
    finite = np.isfinite(features).all(axis=0)
    invalid = int((context & (~finite | ~feature_valid)).sum())
    dates = {side: {'roi_unusable_pixels': roi_counts[side+'_unusable_pixels'],
                   'context_unusable_pixels': context_counts[side+'_unusable_pixels'],
                   'context_components': context_counts[side+'_components']}
             for side in ('pre', 'post')}
    clear = bool(inside and invalid == 0 and all(dates[side]['context_unusable_pixels'] == 0 for side in dates))
    return context, {'roi_pixels': int(roi.sum()), 'context_pixels': int(context.sum()),
        'full_context_inside_grid': inside, 'dates': dates,
        'invalid_or_nonfinite_context_pixels': invalid, 'context_clear': clear}


def run(args, output):
    """Return source-bound context evidence; the root CLI adds report/RSS."""
    require('torch' not in sys.modules, 'Context QA must not import Torch or load a model')
    from audit_validation_context import (architecture_contract, propose_context_control,
                                         validate_protocol)
    from prepare_validation_review import (catalogue_exclusion, read_validation_features,
        records, validate_grid, validate_quality_summary, validate_validation_pair)
    from verify_validation_dataset import read_cache
    from assess_pixel_quality import core_grid, label_target, load_events, validate_source_stats
    from validate_sample import burn
    import numpy as np
    import rasterio
    from affine import Affine
    from scipy.ndimage import binary_erosion

    output, protocol_path = Path(output), Path(args.protocol).resolve()
    protocol = read_json(protocol_path); validate_protocol(protocol)
    protocol_sha = digest(protocol_path)
    sources = {}
    source_hashes = {str(protocol_path.relative_to(ROOT)): protocol_sha}
    for role, item in protocol['source_files'].items():
        path = rooted(item['path'])
        require(digest(path) == item['sha256'], 'A predeclared context input changed: '+role)
        sources[role] = path; source_hashes[item['path']] = item['sha256']
    for name, checksum in protocol['immutable_sources_sha256'].items():
        require(digest(ROOT/'Solutions'/name) == checksum, 'An immutable archived source changed: '+name)
    implementation = {name: digest(ROOT/'Solutions'/name) for name in SOURCE_NAMES}
    if 'implementation_sha256' in protocol:
        require(protocol['implementation_sha256'] == implementation, 'Context implementation differs from the frozen protocol')

    previous = read_json(sources['v2_audit'])
    previous_meta = read_json(sources['v2_metadata'])
    previous_verified = read_json(sources['v2_verification'])
    previous_protocol = read_json(sources['v2_protocol'])
    require(previous_meta['execution_status'] == 'completed'
            and previous_verified['verification_status'] == 'passed'
            and previous_verified['audit_sha256'] == digest(sources['v2_audit'])
            and previous_verified['audit_metadata_sha256'] == digest(sources['v2_metadata'])
            and previous_verified['selection_sha256'] == digest(sources['v2_selection'])
            and previous_verified['protocol_sha256'] == digest(sources['v2_protocol'])
            and previous['protocol_sha256'] == digest(sources['v2_protocol']),
            'The unique-candidate source audit lacks its completed independent proof')
    unique = [row for row in previous['records']
              if row['eligibility_reasons'] == ['endpoint_crop_qa_flags']]
    require(len(unique) == 1 and unique[0] == protocol['selection']['candidate'],
            'Exactly the predeclared unique full-crop-only exclusion must be retained')
    candidate = unique[0]; event_id, pair_id = candidate['event_id'], candidate['pair_id']

    # Split/catalogue/training references are inherited by content identity,
    # including when an explicit alias was omitted from the new protocol.
    for role in ('splits', 'catalogue', 'pairs', 'scenes', 'training_manifest', 'training_metadata'):
        if role not in sources:
            receipt = previous_protocol['source_files'][role]
            path = rooted(receipt['path'])
            require(digest(path) == receipt['sha256'], 'A frozen inherited context reference changed: '+role)
            sources[role] = path; source_hashes[receipt['path']] = receipt['sha256']
    matched = [row for row in records(sources['pairs']) if row['pair_id'] == pair_id]
    quality = [row for row in records(sources['quality']) if row['pair_id'] == pair_id]
    indexed = [row for row in records(sources['quality_index']) if row['pair_id'] == pair_id]
    require(len(matched) == len(quality) == len(indexed) == 1, 'Exact frozen pair/QA/index coverage is required')
    pair, qa, index = matched[0], quality[0], indexed[0]
    identity = ('pair_id', 'group_id', 'split', 'tile', 'pre_date', 'post_date', 'anchor_event_ids')
    policy = protocol['qa_policy_id']
    require(all(pair[key] == qa[key] == index[key] for key in identity)
            and qa['config_id'] == index['config_id'] == policy
            and qa['quality_status'] == index['quality_status'] == candidate['qa_status']
            and qa['training_approved'] is False and index['training_approved'] is False
            and event_id in pair['anchor_event_ids'] and pair['split'] == 'validation'
            and all(pair[key] == candidate[key] for key in ('pair_id', 'group_id', 'split', 'pre_date', 'post_date', 'grid')),
            'Frozen candidate identity, geographic grid or policy changed')
    catalogue, splits = read_json(sources['catalogue']), read_json(sources['splits'])
    selected_features = [feature for feature in catalogue['features'] if feature['id'] == event_id]
    require(len(selected_features) == 1
            and selected_features[0]['properties']['observation_date'] == candidate['annotation_date'],
            'The candidate annotation observation date changed')
    training = list(records(sources['training_manifest']))
    trained_plan = read_json(ROOT/'Docs/Reviews/PixelPilotV1/experiment_plan.json')
    require(digest(sources['training_manifest']) == trained_plan['selection']['manifest_sha256']
            and digest(sources['training_metadata']) == trained_plan['selection']['dataset_metadata_sha256'],
            'The exact thirteen trained grids changed')
    validate_validation_pair(pair, event_id, catalogue, splits, training)
    actual_grid = validate_grid(pair)
    summary = read_json(sources['quality_summary']); validate_quality_summary(summary)
    require(summary['config_id'] == policy, 'The frozen QA policy changed')
    factory = ROOT/'Solutions/train_pilot_baseline.py'
    architecture = architecture_contract(factory)
    require(architecture['spatial_receptive_field_pixels'] == 5
            and architecture['context_radius_pixels'] == 2
            and architecture['parameter_count'] == 2545
            and architecture['global_context_operations'] is False,
            'Local context is supported only for the exact frozen tiny CNN')
    source_hashes[str(factory.relative_to(ROOT))] = digest(factory)
    if 'trained_metadata' in sources:
        trained_metadata = read_json(sources['trained_metadata'])
        require(trained_metadata['parameter_count'] == architecture['parameter_count']
                and trained_metadata['implementation_sha256']['train_pilot_baseline.py'] == digest(factory),
                'The trained architecture receipt differs from the parsed factory')

    endpoints, cache_sources = {}, {}
    for side in ('pre', 'post'):
        expected = qa['per_date'][side]['observation']
        require(expected['config_id'] == policy and expected['scene_id'] == pair[side+'_scene_id']
                and expected['grid'] == pair['grid'], 'Endpoint QA cache scene/grid/policy differs from the frozen pair')
        cache_path = rooted(expected['cache_path']); before = digest(cache_path)
        require(sources.get(side+'_cache', cache_path) == cache_path,
                'The endpoint cache differs from its explicitly bound protocol source')
        masks = read_cache(expected, policy)
        # read_cache bounds every compressed member, dtype and allocation
        # before these two small, exact float32 fields are loaded.
        with np.load(cache_path, allow_pickle=False) as archive:
            rgb, ndvi = archive['rgb'].copy(), archive['ndvi'].copy()
        require(rgb.shape == (3, 64, 64) and rgb.dtype == np.dtype('float32')
                and ndvi.shape == (64, 64) and ndvi.dtype == np.dtype('float32')
                and digest(cache_path) == before, 'Cached endpoint arrays changed during reading')
        source_hashes[expected['cache_path']] = before
        cache_sources[side] = {'path': expected['cache_path'], 'sha256': before, 'observation': expected}
        endpoints[side] = {**masks, 'rgb': rgb, 'ndvi': ndvi}
    published_relative = index['batch_path']+'/'+index['qa_mask_file']
    published_path = rooted(published_relative); published_sha = digest(published_path)
    require(published_relative == candidate['published_qa_file']
            and published_sha == candidate['published_qa_sha256'],
            'The published QA artifact differs from the fixed V2 candidate')
    with rasterio.open(published_path) as source:
        require(source.shape == (64, 64) and source.count == 6 and source.dtypes == ('uint8',)*6
                and source.transform == Affine(*pair['grid']['transform'])
                and source.crs == rasterio.crs.CRS.from_user_input(pair['grid']['crs']),
                'Published endpoint QA grid or array type changed')
        published = source.read()
    expected_union = np.stack([endpoints['pre'][name] | endpoints['post'][name] for name in MASK_NAMES])
    require(np.isin(published, (0, 1)).all() and np.array_equal(published[:5].astype(bool), expected_union),
            'Published QA masks differ from the actual frozen endpoint caches')
    source_hashes[published_relative] = published_sha

    features, feature_valid, feature_stats = read_validation_features(pair, summary)
    require(features.shape == (27, 64, 64) and features.dtype == np.dtype('float32')
            and feature_valid.shape == (64, 64) and feature_valid.dtype == np.dtype(bool)
            and np.array_equal(feature_valid, np.isfinite(features).all(axis=0)),
            'All twenty-seven model-input channels require exact conservative finite validity')
    for side, start in (('pre', 0), ('post', 9)):
        observation = endpoints[side]
        require(np.allclose(observation['rgb'][:, feature_valid], features[[start+2, start+1, start]][:, feature_valid], rtol=0, atol=2e-6)
                and np.allclose(observation['ndvi'][feature_valid], features[start+7][feature_valid], rtol=0, atol=2e-6),
                'Cached RGB/NDVI do not match the newly checked original feature channels')
        observation['feature_valid'] = feature_valid.copy()

    events = load_events(sources['catalogue'].parent)
    require(pair['grid']['crs'] == 'EPSG:32636', 'Context audit preserves the frozen projected CRS')
    temporal = label_target(pair, events).astype(bool)
    anchor = burn([events[event_id]['geometry']], core_grid(pair['grid'])).astype(bool)
    target = anchor & temporal
    positive = binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
    require(np.array_equal(temporal, published[5].astype(bool))
            and int(target.sum()) == candidate['target_pixels']
            and int(positive.sum()) == candidate['positive_interior_pixels']
            and digest(published_path) == published_sha
            and hashlib.sha256(target.tobytes()).hexdigest() == candidate['target_mask_sha256']
            and hashlib.sha256(positive.tobytes()).hexdigest() == candidate['positive_interior_sha256'],
            'Context target/interior differs from the exact independently verified V2 candidate')
    pre, post = endpoints['pre'], endpoints['post']
    target_context, target_result = _context_result(target, pre, post, features, feature_valid)
    positive_context, positive_result = _context_result(positive, pre, post, features, feature_valid)
    excluded = catalogue_exclusion(pair)
    control_exclusion = excluded | target_context
    control, proposal = propose_context_control(pre, post, control_exclusion,
        radius=2, sizes=(16, 12, 8))
    require(control.shape == (64, 64) and control.dtype == np.dtype(bool), 'Proposed control must retain its boolean core grid')
    control_context, control_result = _context_result(control, pre, post, features, feature_valid)
    contexts = {'target': target_result, 'positive_interior': positive_result, 'control': control_result}
    eligible = bool(proposal['proposal_found'] and all(value['context_clear'] for value in contexts.values()))
    full_crop = {side: {name: int(record[name].sum()) for name in MASK_NAMES}
                 | {'unusable_pixels': int(np.logical_or.reduce([record[name] for name in MASK_NAMES]).sum())}
                 for side, record in (('pre', pre), ('post', post))}
    require([full_crop[side]['unusable_pixels'] for side in ('pre', 'post')]
            == candidate['endpoint_crop_unusable_pixels'], 'Original full-crop QA flags changed')
    arrays = {'pre_qa': np.stack([pre[name] for name in MASK_NAMES]),
        'post_qa': np.stack([post[name] for name in MASK_NAMES]),
        'pre_rgb': pre['rgb'], 'post_rgb': post['rgb'], 'pre_ndvi': pre['ndvi'], 'post_ndvi': post['ndvi'],
        'features': features, 'feature_valid': feature_valid, 'target': target,
        'positive_interior': positive, 'target_context': target_context, 'positive_context': positive_context,
        'catalogue_exclusion': excluded, 'control_exclusion': control_exclusion,
        'proposed_control': control, 'control_context': control_context}
    archive_path = output/'context_arrays.npz'
    np.savez_compressed(archive_path, **arrays)
    require(archive_path.stat().st_size <= 2*2**20, 'Saved context arrays exceed their bounded size')
    with np.load(archive_path, allow_pickle=False) as archive:
        require(set(archive.files) == set(arrays)
                and all(np.array_equal(archive[name], values, equal_nan=True) for name, values in arrays.items()),
                'Safe saved context arrays failed their exact round-trip')
    validate_source_stats({'source_file_stats': feature_stats})
    require(all(digest(rooted(name)) == checksum for name, checksum in source_hashes.items())
            and all(digest(ROOT/'Solutions'/name) == checksum for name, checksum in implementation.items()),
            'Context inputs or implementation changed during the single audit')
    for name, checksum in protocol['immutable_sources_sha256'].items():
        require(digest(ROOT/'Solutions'/name) == checksum, 'An immutable source changed during context auditing')
    require('torch' not in sys.modules, 'Context QA unexpectedly imported Torch')
    return {'execution_status': 'completed', 'scope': protocol['scope'],
        'event_id': event_id, 'pair_id': pair_id, 'group_id': pair['group_id'], 'split': 'validation',
        'pair': pair, 'candidate': candidate, 'protocol_sha256': protocol_sha,
        'architecture': architecture, 'context_radius_pixels': 2, 'contexts': contexts,
        'proposal_metadata': proposal, 'qa_eligible_for_visual_review': eligible,
        'numeric_context_passed': eligible,
        'training_approved': False, 'evaluation_approved': False, 'labels_approved': 0,
        'full_crop_qa': full_crop,
        'source_input_sha256': source_hashes, 'cache_sources': cache_sources,
        'feature_source_file_stats': feature_stats, 'implementation_sha256': implementation,
        'array_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()},
        'artifact_sha256': {'context_arrays.npz': digest(archive_path)},
        'model_executions': 0, 'validation_predictions_run': False, 'torch_imported': False,
        'feature_band_window_requests': 14,
        'raw_feature_band_window_reads': 14, 'source_jp2_band_window_reads': 14,
        'cloud_model_executions': 0, 'new_imagery_reviewed': 0,
        'limitations': ['Numeric context/control eligibility does not approve forest/clearing or temporal labels.',
            'The two-pixel context contract applies only to the exact parsed tiny CNN architecture.',
            'Full feature windows were read, without executing cloud or deforestation models.',
            'All review, model selection and validation inference remain later tasks.']}
