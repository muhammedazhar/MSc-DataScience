#!/usr/bin/env python3
"""Screen two explicitly derived source pairs; never approve validation labels.

Only the screen worker reads JP2 windows or runs the frozen cloud classifier.
Verification replays saved QA/cache arrays and geometry, without source pixels.
Original blocked pairs and every frozen implementation remain unchanged.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
GROUP = 'group-96c27871459d03c08bcc'
PARENTS = {
    'event-71d8da17cbbfd9babdee': 'pair-4644e1fb9fe295425239',
    'event-ce1c1df8d02be33bb3c8': 'pair-7250e1f130f05c216746',
}
DATES = ('2018-07-26', '2018-07-31')
SCENES = {
    DATES[0]: 'S2B_MSIL1C_20180726T084009_N0206_R064_T36UXB_20180726T104227',
    DATES[1]: 'S2A_MSIL1C_20180731T083601_N0206_R064_T36UXB_20180731T110233',
}
POLICY = 'fe2472ec030f466bebbb9dc2cb033ed6175de489bcf022f650081dd8e9347560'
COVERAGE_BLOCKS = {'crop_crosses_tile_edge', 'post_scene_does_not_cover_crop'}
RESOURCES = {'workers': 1, 'native_threads': 1, 'nice_increment': 10,
             'gpu_used': False, 'worker_rss_mib': 512,
             'available_memory_reserve_gib': 1.5, 'max_seconds_per_worker': 300}
RULE = {'events': list(PARENTS), 'group_id': GROUP, 'pre_date': DATES[0],
        'post_date': DATES[1], 'source_tile': '36UXB', 'source_crs': 'EPSG:32636',
        'qa_policy_id': POLICY, 'context_padding_m': 1200,
        'context_radius_pixels': 2, 'phase': 'qa_only_no_27_channel_eligibility'}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 32*2**20, 'Missing or oversized JSON source')
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def rooted(name):
    require(isinstance(name, str) and name and not Path(name).is_absolute(), 'Project-relative path required')
    path = (ROOT/name).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.exists(), 'Source escapes the project or is missing')
    return path


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def unique(rows, key):
    result = {}
    for row in rows:
        require(row[key] not in result, 'Duplicate '+key)
        result[row[key]] = row
    return result


def derive_pair(parent, retained, event, scene_paths, frozen_ids):
    """Validate the frozen join before creating a distinct, explicit source pair."""
    event_id = event['event_id']
    require(event_id in PARENTS and parent.get('pair_id') == PARENTS[event_id]
            and parent['pair_id'] in frozen_ids, 'Exact original frozen parent required')
    require(all(retained.get(k) == parent.get(k) for k in
                ('pair_id', 'group_id', 'anchor_event_ids', 'status', 'blocking_reasons')),
            'V5 retained parent differs from the original frozen pair')
    require(parent.get('group_id') == GROUP and parent.get('split') == 'validation'
            and parent.get('anchor_event_ids') == [event_id]
            and parent.get('tile') == '36UXA'
            and (parent.get('pre_date'), parent.get('post_date')) == DATES,
            'Parent group, split, anchor, dates or tile changed')
    require(parent.get('status') == 'blocked_for_temporal_training'
            and set(parent.get('blocking_reasons', [])) == COVERAGE_BLOCKS
            and len(parent['blocking_reasons']) == len(COVERAGE_BLOCKS),
            'Only the exact original source-coverage blocks may be repaired')
    require(parent.get('grid') == event.get('crop_grid')
            and parent['grid']['crs'] == 'EPSG:32636'
            and event.get('annotation_image_date') == DATES[1]
            and parent.get('label_event_ids_by_post_date')
            and not parent.get('conflicting_label_event_ids'),
            'Frozen grid or temporal-label provenance changed')
    source = {side: {'scene_id': SCENES[day], 'path': scene_paths[day]}
              for side, day in zip(('pre', 'post'), DATES)}
    identifier = 'derived-source-'+object_hash({'parent_pair_id': parent['pair_id'], 'sources': source})[:20]
    require(identifier not in frozen_ids, 'Derived pair must never impersonate a frozen pair')
    pair = deepcopy(parent)
    pair.update(pair_id=identifier, pair_kind='derived_validation_source_pair',
                parent_pair_id=parent['pair_id'], parent_pair_sha256=object_hash(parent),
                parent_status=parent['status'], parent_blocking_reasons=list(parent['blocking_reasons']),
                source_replacement=source, original_tile=parent['tile'], tile='36UXB',
                status='candidate_for_pixel_review', blocking_reasons=[],
                pixel_quality='unreviewed', visual_review='unreviewed', training_approved=False,
                evaluation_approved=False)
    for side in ('pre', 'post'):
        pair[side+'_scene_id'] = source[side]['scene_id']
        pair[side+'_path'] = source[side]['path']
    return pair


def validate_lineage(pair, parent, retained, event, scene_paths, frozen_ids):
    require(pair == derive_pair(parent, retained, event, scene_paths, frozen_ids),
            'Derived source pair differs from its exact frozen lineage')


def require_processed(row):
    """The frozen assessor returns exceptions as rows; never publish them as QA."""
    require(row.get('quality_status') != 'processing_error'
            and row.get('qa_mask_file') and isinstance(row.get('per_date'), dict),
            'QA processing failed: '+str(row.get('quality_reasons')))


def contains(outer, inner):
    return (len(outer) == len(inner) == 4 and all(math.isfinite(v) for v in (*outer, *inner))
            and outer[0] <= inner[0] < inner[2] <= outer[2]
            and outer[1] <= inner[1] < inner[3] <= outer[3])


def validate_scene(meta, day, events):
    require(meta.get('metadata_usable') is True and meta.get('cloud_model_usable') is True
            and meta.get('shadow_geometry_usable') is True and not meta.get('issues')
            and meta.get('scene_id') == SCENES[day] and meta.get('acquisition_date') == day
            and meta.get('tile') == '36UXB' and meta.get('processing_level') == 'L1C'
            and meta.get('processing_baseline') == '02.06' and meta.get('nonzero_offset') is False
            and meta.get('quantification_value') == 10000.0 and meta.get('crs') == 'EPSG:32636',
            'Actual repaired source scene fails frozen QA prerequisites or declared identity')
    for event in events:
        require(contains(meta['grid_10m']['bounds'], event['crop_grid']['bounds'])
                and contains(meta['grid_10m']['bounds'], event['required_qa_context_bounds']),
                'Actual repaired source does not contain the original core and full QA context')


def check_asset(asset):
    name = asset['path']
    path = Path(name)
    if path.is_absolute():
        require(path.resolve().is_relative_to(ROOT.resolve()), 'Asset path escapes the project')
        name = relative(path)
    path = rooted(name)
    size = asset.get('bytes', asset.get('size_bytes'))
    require(type(size) is int and size > 0 and path.stat().st_size == size
            and digest(path) == asset['sha256'], 'Source asset differs from the completed receipt: '+name)
    stat = path.stat()
    return name, {'bytes': size, 'sha256': asset['sha256'], 'stat': [size, stat.st_mtime_ns]}


def validate_assets(ingestion, completion):
    """Verify actual original and added files, not just the receipt's own hash."""
    require(completion.get('new_assets') == 96 and completion.get('original_assets_preserved') == 20
            and completion.get('satellite_pixel_reads') == completion.get('model_executions')
            == completion.get('labels_approved') == 0, 'Unexpected V10 completion scope')
    assets = [a for scene in ingestion['ingestion_results'] for a in scene['assets']]
    originals = unique(completion['original_checks'], 'path')
    require(len(assets) == len(originals) == 20, 'Exact original V9 receipts required')
    for asset in assets:
        name = relative(asset['path']) if Path(asset['path']).is_absolute() else asset['path']
        require(name in originals and all(asset[k] == originals[name][k] for k in ('bytes', 'sha256')),
                'V10 original preservation receipt differs from V9')
    added = completion['source_checks']
    require(len(added) == 96, 'Exactly the 96 repaired QA assets are required')
    for asset in added:
        require(asset['scene_id'] in SCENES.values() and asset.get('if_match_verified') is True
                and asset.get('content_length_verified') is True, 'Unverified V10 remote source receipt')
        name = Path('Datasets/Sentinel-2')/asset['scene_id']/asset['target_relpath']
        assets.append({**asset, 'path': str(name)})
    checked = {}
    for asset in assets:
        name, receipt = check_asset(asset)
        require(name not in checked, 'Duplicate receipt for a source asset')
        checked[name] = receipt
    require(len(checked) == 116, 'Both scenes require exactly 58 full QA source files each')
    return checked


def build_inputs(args):
    from select_validation_cases import records
    from pixelqa_scene_metadata import qa_scene_metadata
    from prepare_validation_review import validate_validation_pair

    protocol = read(args.protocol)
    require(protocol.get('schema_version') == 11 and protocol.get('rule') == RULE
            and protocol.get('resources') == RESOURCES, 'Frozen V11 scope or resource contract changed')
    sources = {}
    for role, receipt in protocol['source_files'].items():
        path = rooted(receipt['path'])
        require(digest(path) == receipt['sha256'], 'Frozen source changed: '+role)
        sources[role] = path
    for name, checksum in {**protocol['immutable_sources_sha256'], **protocol['implementation_sha256']}.items():
        require(digest(ROOT/'Solutions'/name) == checksum, 'Frozen implementation changed: '+name)
    inventory = read(sources['inventory']); reqs = read(sources['source_requirements'])
    require(GROUP not in inventory['exposed_group_ids'] and reqs['group_id'] == GROUP
            and reqs['crs'] == 'EPSG:32636' and reqs['context_padding_metres'] == 1200,
            'Prior exposure or frozen source requirements changed')
    events = unique(reqs['events'], 'event_id')
    require(set(events) == set(PARENTS), 'Exactly the two predeclared events are required')
    parents, frozen_ids = {}, set()
    for parent in records(sources['pairs']):
        require(parent['pair_id'] not in frozen_ids, 'Duplicate original frozen pair')
        frozen_ids.add(parent['pair_id'])
        if parent['pair_id'] in PARENTS.values():
            parents[parent['pair_id']] = parent
    retained = unique(inventory['pairs'], 'pair_id')
    ingestion = read(sources['ingestion']); completion = read(sources['completion'])
    completion_metadata = read(sources['completion_metadata'])
    require(completion_metadata.get('execution_status') == 'completed'
            and completion_metadata['artifact_sha256']['completion.json'] == digest(sources['completion']),
            'V10 completion was not published by the guarded worker')
    scene_paths = {}
    for scene in ingestion['ingestion_results']:
        day = next((d for d in DATES if SCENES[d] == scene['scene_id']), None)
        require(day is not None and day not in scene_paths, 'Unexpected/duplicate ingested source scene')
        scene_paths[day] = relative(scene['safe_path'])
    require(set(scene_paths) == set(DATES), 'Exact two ingested scene identities required')
    assets = validate_assets(ingestion, completion)
    scene_metadata = {}
    for day, scene_path in scene_paths.items():
        meta = qa_scene_metadata(rooted(scene_path), ROOT)
        validate_scene(meta, day, events.values())
        scene_metadata[day] = meta
    final_scenes = unique(completion['final_scenes'], 'scene_id')
    require(set(final_scenes) == set(SCENES.values())
            and all(final_scenes[meta['scene_id']] == meta for meta in scene_metadata.values()),
            'Actual source metadata differs from completed V10 final scenes')
    catalogue, splits = read(sources['catalogue']), read(sources['splits'])
    training = list(records(sources['training_manifest']))
    # training_metadata is independently frozen even though footprint checks use the manifest.
    read(sources['training_metadata'])
    derived = []
    for event_id, parent_id in PARENTS.items():
        require(parent_id in parents and parent_id in retained, 'Frozen/V5 original parent is missing')
        pair = derive_pair(parents[parent_id], retained[parent_id], events[event_id], scene_paths, frozen_ids)
        validate_validation_pair(pair, event_id, catalogue, splits, training)
        derived.append(pair)
    require(len({p['pair_id'] for p in derived}) == 2, 'Derived pair IDs must be unique')
    return protocol, sources, assets, scene_metadata, derived, inventory['exposed_group_ids']


def unchanged_assets(assets):
    for name, receipt in assets.items():
        stat = rooted(name).stat()
        require([stat.st_size, stat.st_mtime_ns] == receipt['stat'], 'Source changed during screening: '+name)


def region_result(roi, pre, post):
    """Retain empty/boundary support and adverse QA as numeric outcomes."""
    import numpy as np
    from audit_validation_context import context_mask, qa_region_counts

    result = {'pixels': int(roi.sum()), 'full_context_available': False,
              'context_radius_pixels': 2, 'all_27_channels_checked': False}
    if not roi.any():
        result['reason'] = 'empty_reference_geometry'
        return result
    try:
        context = context_mask(roi, radius=2)
    except ValueError as error:
        result['reason'] = str(error)
        return result
    result.update(full_context_available=True, context_pixels=int(context.sum()),
                  qa=qa_region_counts(context, pre, post),
                  roi_qa=qa_region_counts(roi, pre, post),
                  context_mask_sha256=hashlib.sha256(context.tobytes()).hexdigest())
    result['finite_cached_inputs'] = {}
    for side, observation in (('pre', pre), ('post', post)):
        finite = np.isfinite(observation['rgb']).all(axis=0) & np.isfinite(observation['ndvi'])
        result['finite_cached_inputs'][side] = {
            'rgb_and_ndvi_nonfinite_context_pixels': int((~finite & context).sum()),
            'scope': 'Cached RGB and NDVI only; no complete 27-channel eligibility proof'}
    return result


def anchor_contexts(pair, event_id, events, pre, post):
    import numpy as np
    from scipy.ndimage import binary_erosion
    from assess_pixel_quality import core_grid, label_target
    from validate_sample import burn

    anchor = burn([events[event_id]['geometry']], core_grid(pair['grid'])).astype(bool)
    temporal = label_target(pair, events).astype(bool)
    target = anchor & temporal
    interior = binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
    return {'event_id': event_id, 'target': region_result(target, pre, post),
            'positive_interior': region_result(interior, pre, post),
            'anchor_pixels': int(anchor.sum()), 'temporal_union_pixels': int(temporal.sum()),
            'temporal_clipped_pixels': int((anchor & ~temporal).sum()),
            'target_mask_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
            'interior_mask_sha256': hashlib.sha256(interior.tobytes()).hexdigest(),
            'geometry_scope': 'Anchor intersected with unchanged cumulative-post minus cumulative-pre reference',
            'interior_rule': 'One-pixel 3x3 erosion of anchor-specific temporal target',
            'class_labels_approved': 0, 'all_27_channels_checked': False}


@contextmanager
def count_band_requests(module):
    original = module.read_fused_band
    counter = {'requests': 0}
    def counted(*args, **kwargs):
        counter['requests'] += 1
        return original(*args, **kwargs)
    module.read_fused_band = counted
    try:
        yield counter
    finally:
        module.read_fused_band = original


def save_result(output, result, metadata):
    write(output/'screen.json', result)
    (output/'derived_pairs.jsonl').write_text(''.join(json.dumps(p, sort_keys=True, allow_nan=False)+'\n'
                                                     for p in result['derived_pairs']))
    (output/'report.md').write_text(
        '# Repaired-source validation quality screen\n\n'
        'Two derived pairs retain the original grids, dates and temporal provenance in one validation group. '
        'Original blocked records are unchanged. Numeric QA outcomes and anchor-specific target/interior '
        'contexts are retained even when adverse. No imagery was displayed or manually reviewed.\n\n'
        'This phase checks frozen QA and cached RGB/NDVI finiteness only. Complete 27-channel eligibility, '
        'forest/clearing state, labels, dates of clearing and validation accuracy remain unestablished.\n')
    metadata['artifact_sha256'] = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output/'metadata.json', metadata)


def screen(args, output, inputs):
    import numpy as np
    import rasterio
    import assess_pixel_quality as quality
    from assess_pixel_quality import validate_source_stats

    protocol, sources, assets, scene_metadata, pairs, exposed = inputs
    identity = quality.model_identity()
    require(quality.policy_id(identity) == POLICY, 'Frozen cloud QA policy/runtime identity changed')
    events = quality.load_events(sources['catalogue'].parent)

    class CapturedReader(quality.ObservationReader):
        def __init__(self):
            super().__init__(args.cache_root, identity)
            self.observations = []
        def observation(self, scene_path, raw_grid):
            record = super().observation(scene_path, raw_grid)
            self.observations.append(record)
            return record

    reader = CapturedReader(); rows, contexts, caches = [], [], []
    output.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(GDAL_CACHEMAX=32*2**20, GDAL_NUM_THREADS='1'), count_band_requests(quality) as counts:
        for pair in pairs:
            start = len(reader.observations)
            row = quality.assess_pair(pair, reader, events, output, preview=False)
            require_processed(row)
            require(len(reader.observations) == start+2, 'Each pair requires exactly two observations')
            pre, post = reader.observations[start:]
            rows.append(row)
            contexts.append(anchor_contexts(pair, pair['anchor_event_ids'][0], events, pre, post))
            for record in (pre, post):
                path = rooted(record['metadata']['cache_path'])
                require(path.stat().st_size <= 2*2**20, 'QA cache exceeds the frozen size bound')
                caches.append({'path': relative(path), 'bytes': path.stat().st_size, 'sha256': digest(path),
                               'cache_key': record['metadata']['cache_key']})
    require(len(reader.observations) == 4 and reader.hits+reader.misses == 4
            and counts['requests'] == reader.misses*11, 'Bounded four-observation/11-band request contract changed')
    source_stats = {}
    for meta in reader.scenes.values():
        validate_source_stats(meta)
        source_stats.update(meta['source_file_stats'])
    require(set(source_stats) == set(assets), 'QA discovered files differ from complete original/added receipts')
    unchanged_assets(assets)
    result = {'schema_version': 11, 'protocol_sha256': digest(args.protocol), 'qa_policy_id': POLICY,
              'phase': RULE['phase'], 'derived_pairs': pairs, 'results': rows,
              'anchor_contexts': contexts, 'cache_receipts': caches,
              'source_asset_receipts': assets, 'source_file_stats': source_stats,
              'scene_metadata': scene_metadata, 'prior_exposed_group_ids': exposed,
              'groups_observed_by_automatic_qa': [GROUP],
              'future_freshness_note': 'This source/group has now undergone numeric QA; never erase this exposure history',
              'processing_errors': [], 'observation_count': 4, 'new_cache_hits': reader.hits,
              'new_cache_misses': reader.misses, 'cloud_model_executions': reader.misses,
              'fused_band_window_requests': counts['requests'], 'maximum_fused_band_window_requests': 44,
              'all_27_channels_checked': False, 'validation_eligible': False,
              'labels_approved': 0, 'new_imagery_reviewed': 0, 'forest_model_executions': 0,
              'training_approved': False, 'evaluation_approved': False, 'model_identity': identity}
    metadata = base_metadata(args, result)
    metadata.update(observation_count=4, new_cache_hits=reader.hits, new_cache_misses=reader.misses,
                    cloud_model_executions=reader.misses, fused_band_window_requests=counts['requests'],
                    checks={'exact_derived_lineage': True, 'full_train_test_footprint_isolation': True,
                            'frozen_qa_prerequisites': True, 'source_file_stats_unchanged': True,
                            'zero_processing_errors': True})
    save_result(output, result, metadata)


def base_metadata(args, result):
    import resource
    return {'execution_status': 'completed', 'schema_version': 11, 'mode': args.mode,
            'protocol_sha256': digest(args.protocol), 'implementation_sha256': digest(__file__),
            'phase': RULE['phase'], 'processing_errors': [], 'pair_count': 2, 'group_count': 1,
            'labels_approved': 0, 'new_imagery_reviewed': 0, 'forest_model_executions': 0,
            'validation_predictions_run': False, 'all_27_channels_checked': False,
            'validation_eligible': False, 'training_approved': False, 'evaluation_approved': False,
            'torch_imported': 'torch' in sys.modules,
            'worker_high_water_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                             (2**20 if sys.platform == 'darwin' else 1024), 2)}


def load_safe_cache(receipt, expected):
    """Validate every allocation header with the frozen verifier before loading."""
    import numpy as np
    from verify_validation_dataset import read_cache

    require(expected.get('cache_path') == receipt['path'], 'Expected cache path differs from its receipt')
    path = rooted(receipt['path'])
    require(path.stat().st_size == receipt['bytes'] <= 2*2**20 and digest(path) == receipt['sha256'],
            'Completed safe cache changed')
    read_cache(expected, POLICY)
    with np.load(path, allow_pickle=False) as data:
        observation = {name: data[name] for name in data.files if name != 'metadata'}
        observation['metadata'] = json.loads(str(data['metadata']))
    require(observation['metadata']['cache_key'] == receipt['cache_key'], 'Safe cache identity mismatch')
    return observation


def verify(args, output, inputs):
    import numpy as np
    import rasterio
    from assess_pixel_quality import (MASK_NAMES, classify_quality, core_grid, label_target,
                                      load_events, model_identity, policy_id)

    _, sources, assets, _, pairs, _ = inputs
    run = args.run.resolve()
    require(run.is_dir() and run.is_relative_to(ROOT.resolve()), 'Completed screen must be inside the project')
    meta = read(run/'metadata.json'); result = read(run/'screen.json')
    require(meta['execution_status'] == 'completed' and meta['mode'] == 'screen'
            and result['protocol_sha256'] == digest(args.protocol) and result['derived_pairs'] == pairs
            and result['processing_errors'] == [] and result['source_asset_receipts'] == assets,
            'Completed screen does not match frozen lineage and current source receipts')
    for name, checksum in meta['artifact_sha256'].items():
        path = (run/name).resolve()
        require(path.is_relative_to(run) and path.is_file() and digest(path) == checksum, 'Screen artifact changed')
    require(policy_id(model_identity()) == POLICY and result['qa_policy_id'] == POLICY,
            'Cloud QA policy/runtime differs from the completed screen')
    require(len(result['cache_receipts']) == 4 and len(result['results']) == len(result['anchor_contexts']) == 2,
            'Expected four caches and two pair results')
    cached = []
    for index, receipt in enumerate(result['cache_receipts']):
        side = 'pre' if index % 2 == 0 else 'post'
        expected = result['results'][index//2]['per_date'][side]['observation']
        cached.append(load_safe_cache(receipt, expected))
    events = load_events(sources['catalogue'].parent)
    for index, pair in enumerate(pairs):
        pre, post = cached[2*index:2*index+2]
        row = result['results'][index]; require_processed(row)
        for side, observation in (('pre', pre), ('post', post)):
            require(observation['metadata']['scene_id'] == pair[side+'_scene_id']
                    and observation['metadata']['grid'] == pair['grid']
                    and row['per_date'][side]['observation'] == observation['metadata'],
                    'Cache observation/grid differs from its derived pair')
        target = label_target(pair, events)
        status, reasons, masks, crop, target_metrics = classify_quality(
            pre, post, target, pre['metadata']['context_incomplete'] or post['metadata']['context_incomplete'])
        require((row['quality_status'], row['quality_reasons'], row['crop_metrics'], row['target_metrics'])
                == (status, reasons, crop, target_metrics), 'Saved numeric quality outcome differs from safe caches')
        path = (run/row['qa_mask_file']).resolve()
        require(path.is_relative_to(run), 'QA mask path escapes completed screen')
        with rasterio.open(path) as source:
            expected = np.stack([*(masks[n].astype('uint8') for n in MASK_NAMES), target])
            require(source.count == 6 and source.shape == (64, 64)
                    and source.crs.to_string() == pair['grid']['crs']
                    and source.transform == core_grid(pair['grid'])['transform']
                    and source.descriptions == (*MASK_NAMES, 'reference_new_loss')
                    and np.array_equal(source.read(), expected), 'Saved six-plane QA mask differs from safe caches/reference')
        require(anchor_contexts(pair, pair['anchor_event_ids'][0], events, pre, post)
                == result['anchor_contexts'][index], 'Anchor-specific context replay differs')
    unchanged_assets(assets)
    metadata = base_metadata(args, result)
    require(metadata['torch_imported'] is False, 'Torch/model inference is forbidden')
    metadata.update(source_run=relative(run), source_metadata_sha256=digest(run/'metadata.json'),
                    source_jp2_window_reads=0, cloud_model_executions=0, safe_caches_checked=4,
                    qa_tifs_checked=2, verification_scope='Shared lineage/QA helpers and cached-array replay; no independent cloud prediction or visual class truth',
                    checks={'source_asset_hashes': True, 'derived_lineage': True, 'safe_cache_hashes': True,
                            'six_plane_qa_reconstruction': True, 'anchor_context_reconstruction': True})
    output.mkdir(parents=True, exist_ok=True)
    write(output/'metadata.json', metadata)
    (output/'report.md').write_text('# Repaired-source QA verification\n\n'
        'Replayed four safe caches, two six-plane QA files and anchor-specific contexts. '
        'Original/added source hashes and frozen lineage match. Shared helpers are reused; '
        'cloud predictions and visual classes were not independently verified. '
        'No source JP2 windows, model runs, labels or full 27-channel eligibility checks.\n')


def worker(args, output):
    require('torch' not in sys.modules, 'Torch/model inference is forbidden')
    inputs = build_inputs(args)
    if args.mode == 'verify':
        verify(args, output, inputs)
    else:
        screen(args, output, inputs)
    require('torch' not in sys.modules, 'Torch/model inference is forbidden')


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationSourceScreenV11/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=ROOT/'Datasets/PixelQA/cache')
    parser.add_argument('--mode', choices=('screen', 'verify'), default='screen')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Retain 512 MiB/1.5 GiB/300 s bounds')
    if (args.mode == 'verify') != (args.run is not None):
        parser.error('--run is required only in verify mode')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='validation-source-qa', gdal_cache_mib=32, warp_memory_mib=16)


if __name__ == '__main__':
    main()
