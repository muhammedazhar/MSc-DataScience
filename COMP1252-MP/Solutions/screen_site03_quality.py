"""Screen the frozen site-03 crop and replay its bounded QA archives."""

from pathlib import Path
import json
import shutil
import tempfile

from prepare_validation_manual import _npz_headers, observation_metrics
from screen_earlier_validation import prevalidate_cache
from screen_fresh_validation_sources import review_candidates
from screen_repaired_validation import (
    POLICY, ROOT, anchor_contexts, count_band_requests, digest, read, relative,
    require, unchanged_assets, write,
)
from verify_validation_dataset import validate_cache_binding


DATES = ('2018-05-10', '2018-05-30', '2018-06-04')
ROLES = ('earlier_control_observation', 'before_observation', 'after_observation')
CACHE_ROOT = ROOT/'Datasets/ValidationPilot/v30/cache'
MAX_ARCHIVE_BYTES = 2*2**20
BANDS_PER_OBSERVATION = 11


def bound_inputs(inputs):
    import assess_pixel_quality as quality

    pair, site, control = (inputs[key] for key in ('pair', 'site', 'control'))
    scenes, assets = inputs['scenes'], inputs['source_asset_receipts']
    require(set(scenes) == set(DATES) and site['site_id'] == 'site-03'
            and (pair['pre_date'], pair['post_date']) == DATES[1:]
            and pair['grid'] == site['grid'] and pair['group_id'] == site['group_id']
            and pair['anchor_event_ids'] == [site['event_id']],
            'Quality phase must use the exact frozen site-03 crop and dates')
    require(all(scenes[pair[side+'_date']][key] == pair[side+'_'+key]
                for side in ('pre', 'post') for key in ('path', 'scene_id')),
            'Quality endpoints must match the selected scene paths and IDs')
    require(all(control[key] == value for key, value in {
        'row': 2, 'col': 2, 'width': 16, 'height': 16, 'pixels': 256,
        'context_radius_pixels': 2, 'context_pixels': 400,
    }.items()), 'Quality phase must use the exact frozen control')

    bindings = {}
    for day, role in zip(DATES, ROLES):
        scene = scenes[day]
        rasters = set(scene['cloud_band_paths'].values()) | set(scene['feature_band_paths'].values())
        qa_paths = [scene['quality_paths']['cloud']] + [
            path for kind in ('nodata', 'satura', 'defect', 'tecqua')
            for path in scene['quality_paths'][kind]
        ]
        require(len(rasters) == BANDS_PER_OBSERVATION and scene['acquisition_date'] == day,
                'Unexpected QA band inventory or acquisition date: '+day)
        source_id = quality.object_hash({
            'metadata_uid': scene['metadata_uid'],
            'raster_stats': {path: assets[path]['stat'] for path in rasters},
            'quality_hashes': {path: assets[path]['sha256'] for path in qa_paths},
        })
        key = quality.object_hash({'source': source_id, 'grid': pair['grid'], 'config_id': POLICY})
        bindings[day] = {
            'scene_path': scene['path'], 'scene_id': scene['scene_id'], 'role': role,
            'metadata_uid': scene['metadata_uid'], 'source_id': source_id,
            'cache_key': key, 'cache_path': relative(CACHE_ROOT/POLICY/(key+'.npz')),
            'grid': pair['grid'],
        }
    require(len({binding['cache_key'] for binding in bindings.values()}) == len(DATES),
            'Each observation date must bind to its own cache')
    common = {
        'schema_version': 30, 'protocol_sha256': inputs['protocol_sha256'], 'qa_policy_id': POLICY,
        'pair': pair, 'site': site, 'temporal_geometry': inputs['geometry'],
        'control_opportunity': control, 'source_asset_receipts': assets,
        'labels_approved': 0, 'new_imagery_reviewed': 0, 'forest_model_executions': 0,
        'all_27_channels_checked': False, 'validation_eligible': False,
    }
    return common, bindings


def archived_observation(run, receipt, expected, binding):
    """Load a self-contained QA archive without following its original cache path."""
    import numpy as np
    from assess_pixel_quality import MASK_NAMES

    key = binding['cache_key']
    require(all(expected[field] == binding[field] for field in (
        'scene_id', 'metadata_uid', 'source_id', 'cache_key', 'cache_path', 'grid')),
        'Archived QA observation does not bind to its source and grid')
    require(receipt['path'] == binding['cache_path'] and receipt['cache_key'] == key
            and receipt['snapshot'] == 'cache_evidence/'+key+'.npz',
            'Unexpected QA archive path or cache key')
    run = run.resolve()
    path = (run/receipt['snapshot']).resolve()
    require(path.is_relative_to(run) and path.stat().st_size == receipt['bytes'] <= MAX_ARCHIVE_BYTES
            and digest(path) == receipt['sha256'], 'Archived QA cache receipt changed')
    shapes = {name: (64, 64) for name in (*MASK_NAMES, 'probability', 'product_cloud', 'ndvi')}
    shapes.update(rgb=(3, 64, 64), metadata=())
    _npz_headers(path, shapes, ('probability', 'rgb', 'ndvi'), metadata=True)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        validate_cache_binding(metadata, expected, POLICY)
        observation = {name: data[name] for name in shapes if name != 'metadata'}
    require(digest(path) == receipt['sha256'], 'Archived QA cache changed while reading')
    return {**observation, 'metadata': metadata}


def evaluate(output, inputs, observations, processing_errors):
    import assess_pixel_quality as quality
    from plan_site03_qa import site03_masks

    pair, events, control = (inputs[key] for key in ('pair', 'events', 'control'))
    by_path = {inputs['scenes'][day]['path']: observations[day] for day in observations}

    class CachedReader:
        config_id = POLICY

        def observation(self, path, grid):
            require(grid == pair['grid'] and path in by_path, 'QA endpoint observation unavailable')
            return by_path[path]

    errors = list(processing_errors)
    result = quality.assess_pair(pair, CachedReader(), events, output, preview=False)
    contexts = None
    if result['quality_status'] == 'processing_error':
        errors.append({'scope': 'pair_assessment', 'reasons': result['quality_reasons']})
    else:
        try:
            contexts = anchor_contexts(pair, pair['anchor_event_ids'][0], events,
                                       observations[DATES[1]], observations[DATES[2]])
        except Exception as error:
            errors.append({'scope': 'anchor_contexts', 'reason': f'{type(error).__name__}: {error}'})

    masks = site03_masks(pair, events, control)
    roi, context = masks['control'], masks['control_context']
    require(int(roi.sum()) == control['pixels'] and int(context.sum()) == control['context_pixels'],
            'Control or full context size differs from the frozen geometry')
    control_qa = {}
    for day in DATES:
        if day not in observations:
            control_qa[day] = {'qa_status': 'observation_unavailable'}
            continue
        try:
            observation = observations[day]
            metrics = observation_metrics(observation, roi, context)
            incomplete = observation['metadata']['context_incomplete']
            control_qa[day] = {
                **metrics, 'off_crop_context_incomplete': incomplete,
                'qa_status': 'clear' if metrics['context_clear'] and not incomplete else 'flagged',
            }
        except Exception as error:
            errors.append({'scope': 'control_context', 'date': day, 'reason': f'{type(error).__name__}: {error}'})
            control_qa[day] = {'qa_status': 'processing_error'}

    eligible = review_candidates([pair], [result], [contexts], errors)
    review_status = ('blocked_by_processing_errors' if errors else
                     'candidates_ready_for_separate_review' if eligible else 'no_candidates_passed_qa')
    clear_controls = sum(row['qa_status'] == 'clear' for row in control_qa.values())
    control_status = ('blocked_by_processing_errors' if errors else
                      'clear_on_all_three_dates' if clear_controls == len(DATES) else 'qa_flags_present')
    return {
        'results': [result], 'anchor_contexts': [contexts], 'processing_errors': errors,
        'review_candidate_status': review_status, 'review_candidate_pair_ids': eligible,
        'control_qa_by_date': control_qa, 'control_qa_status': control_status,
        'control_clear_observation_count': clear_controls, 'control_class_status': 'unknown',
    }


def screen(output, inputs):
    import assess_pixel_quality as quality
    import rasterio

    common, bindings = bound_inputs(inputs)
    identity = quality.model_identity()
    require(quality.policy_id(identity) == POLICY, 'Frozen pixel QA policy or runtime changed')
    unchanged_assets(inputs['source_asset_receipts'])
    reader = quality.ObservationReader(CACHE_ROOT, identity)
    cloud_model = reader.detector.get_cloud_probability_maps
    cloud_calls = 0

    def counted_cloud_model(*args, **kwargs):
        nonlocal cloud_calls
        require(cloud_calls < len(DATES), 'V30 cloud execution budget exceeded')
        cloud_calls += 1
        return cloud_model(*args, **kwargs)

    reader.detector.get_cloud_probability_maps = counted_cloud_model
    (output/'cache_evidence').mkdir()
    observations, records, errors = {}, {}, []
    with rasterio.Env(GDAL_CACHEMAX=32*2**20, GDAL_NUM_THREADS='1'), count_band_requests(quality) as counts:
        for day, binding in bindings.items():
            record = {key: binding[key] for key in ('role', 'scene_id', 'scene_path')}
            records[day] = record
            try:
                scene_path, grid = binding['scene_path'], binding['grid']
                scene = reader.metadata(scene_path)
                require(scene['cache_source_id'] == binding['source_id']
                        and scene['scene_id'] == binding['scene_id']
                        and all(inputs['source_asset_receipts'][path]['stat'] == stat
                                for path, stat in scene['source_file_stats'].items()),
                        'Live QA source differs from verified publication')
                cache_path = ROOT/binding['cache_path']
                require(cache_path == reader.cache_root/(binding['cache_key']+'.npz'),
                        'Unexpected QA cache allocation path')
                if cache_path.exists():
                    prevalidate_cache(cache_path, binding['source_id'], grid)
                value = reader.observation(scene_path, grid)
                key = binding['cache_key']
                require(cache_path.stat().st_size <= MAX_ARCHIVE_BYTES, 'QA cache exceeds archive byte limit')
                snapshot = output/'cache_evidence'/(key+'.npz')
                receipt = {
                    'path': binding['cache_path'], 'bytes': cache_path.stat().st_size,
                    'sha256': digest(cache_path), 'cache_key': key,
                    'snapshot': snapshot.relative_to(output).as_posix(),
                }
                shutil.copyfile(cache_path, snapshot)
                observations[day] = archived_observation(output, receipt, value['metadata'], binding)
                record.update(status='processed', observation=value['metadata'], cache_receipt=receipt)
            except Exception as error:
                reason = f'{type(error).__name__}: {error}'
                errors.append({'scope': 'observation', 'date': day, 'reason': reason})
                record.update(status='processing_error', reason=reason)
        outcomes = evaluate(output, inputs, observations, errors)
    unchanged_assets(inputs['source_asset_receipts'])
    require(reader.hits+reader.misses <= len(DATES) and counts['requests'] <= len(DATES)*BANDS_PER_OBSERVATION,
            'V30 source QA resource budget exceeded')
    if not outcomes['processing_errors']:
        require(len(observations) == reader.hits+reader.misses == len(DATES)
                and reader.misses == cloud_calls and counts['requests'] == cloud_calls*BANDS_PER_OBSERVATION,
                'Completed QA observations differ from the planned source-read budget')
    summary = {
        'execution_status': 'incomplete_processing_errors' if outcomes['processing_errors'] else 'completed',
        'source_jp2_window_reads': counts['requests'], 'cloud_model_executions': cloud_calls,
        'review_candidate_status': outcomes['review_candidate_status'],
        'review_candidate_pair_ids': outcomes['review_candidate_pair_ids'],
        'control_qa_status': outcomes['control_qa_status'],
        'control_clear_observation_count': outcomes['control_clear_observation_count'],
        'status_counts': {outcomes['results'][0]['quality_status']: 1},
    }
    write(output/'screen.json', {
        **common, **outcomes, **summary, 'observations': records, 'distinct_observations': len(observations),
        'cache_hits': reader.hits, 'fused_band_window_requests': counts['requests'],
    })
    return summary


def verify(output, source, inputs):
    import assess_pixel_quality as quality
    import numpy as np
    import rasterio

    common, bindings = bound_inputs(inputs)
    require(quality.policy_id(quality.model_identity()) == POLICY, 'Frozen pixel QA policy or runtime changed')
    result = read(source/'screen.json')
    require(result['execution_status'] == 'completed' and not result['processing_errors']
            and all(result[key] == value for key, value in common.items())
            and set(result['observations']) == set(DATES) and result['distinct_observations'] == len(DATES),
            'Completed QA archive differs from the frozen phase or observation dates')
    calls, hits = result['cloud_model_executions'], result['cache_hits']
    require(0 <= calls <= len(DATES) and calls+hits == len(DATES)
            and result['source_jp2_window_reads'] == result['fused_band_window_requests'] == calls*BANDS_PER_OBSERVATION,
            'Archived source and cloud call accounting differs')

    observations = {}
    for day, binding in bindings.items():
        record = result['observations'][day]
        require(record['status'] == 'processed' and all(record[key] == binding[key]
                for key in ('role', 'scene_id', 'scene_path')), 'Archived QA observation roster changed: '+day)
        observations[day] = archived_observation(source, record['cache_receipt'], record['observation'], binding)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        replay = Path(temporary)
        outcomes = evaluate(replay, inputs, observations, [])
        require(not outcomes['processing_errors'] and all(result[key] == value for key, value in outcomes.items()),
                'Replayed positive or control numeric QA outcomes differ')
        mask = (source/outcomes['results'][0]['qa_mask_file']).resolve()
        require(mask.is_relative_to(source.resolve()) and mask.stat().st_size <= MAX_ARCHIVE_BYTES,
                'Archived QA mask path or size differs')
        with rasterio.open(mask) as saved, rasterio.open(replay/outcomes['results'][0]['qa_mask_file']) as expected:
            require(saved.profile == expected.profile and saved.descriptions == expected.descriptions
                    and np.array_equal(saved.read(), expected.read()), 'QA mask differs from archived inputs')
    proof = {
        'execution_status': 'completed', 'protocol_sha256': inputs['protocol_sha256'],
        'checks': {
            'exact_lineage': True, 'cumulative_targets': True, 'safe_archived_cache_bindings': True,
            'numeric_qa_outcomes': True, 'qa_masks': True, 'anchor_contexts': True,
            'three_date_control_context': True, 'candidate_gate': True,
        },
        'pairs_replayed': 1, 'cache_bindings_replayed': len(observations),
        'verified_screen_metadata_sha256': digest(source/'metadata.json'),
        'review_candidate_pair_ids': outcomes['review_candidate_pair_ids'],
        'control_qa_status': outcomes['control_qa_status'],
        'cloud_model_executions': 0, 'source_jp2_window_reads': 0,
    }
    write(output/'verification.json', proof)
    return proof
