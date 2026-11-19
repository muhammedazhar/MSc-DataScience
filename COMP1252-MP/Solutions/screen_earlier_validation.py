#!/usr/bin/env python3
"""Prepare four explicit earlier-date pairs, screen all, then replay saved QA."""
import argparse
from copy import deepcopy
from datetime import date
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

from screen_repaired_validation import (ROOT, GROUP, PARENTS, POLICY, RESOURCES, COVERAGE_BLOCKS,
    require, digest, object_hash, read, write, rooted, relative, unique, contains, check_asset,
    unchanged_assets, require_processed, anchor_contexts, count_band_requests, load_safe_cache)

PRE_DATES = ('2018-07-21', '2018-07-16')
POST_DATE = '2018-07-31'
SCENES = {
    PRE_DATES[0]: 'S2A_MSIL1C_20180721T083601_N0206_R064_T36UXB_20180721T105541',
    PRE_DATES[1]: 'S2B_MSIL1C_20180716T084009_N0206_R064_T36UXB_20180716T122805',
    POST_DATE: 'S2A_MSIL1C_20180731T083601_N0206_R064_T36UXB_20180731T110233',
}


def source_paths():
    return {day: 'Datasets/Sentinel-2/'+scene+'/'+scene+'.SAFE' for day, scene in SCENES.items()}


def context_ids(parent, events, properties, pre_date):
    from shapely.geometry import box
    footprint = box(*parent['grid']['bounds'])
    ids = sorted(key for key, value in events.items() if value['geometry'].intersects(footprint))
    require(ids == parent['context_event_ids'], 'Recomputed context differs from frozen crop provenance')
    for key in ids:
        prop = properties[key]
        require(prop['group_id'] == GROUP and prop['split'] == 'validation'
                and 'conflicting_observation_dates' not in prop['flags'], 'Conflicting context group/split/date provenance')
        require(prop['temporal_source_ids'] or not pre_date < prop['observation_date'] <= POST_DATE,
                'Earlier interval introduces an unverified baseline-only temporal label')
    after = sorted(key for key in ids if properties[key]['temporal_source_ids']
                   and properties[key]['observation_date'] <= POST_DATE)
    before = sorted(key for key in after if properties[key]['observation_date'] <= pre_date)
    require(after == parent['label_event_ids_by_post_date'], 'Recomputed post-date catalogue set changed')
    return ids, before, after


def derive_pair(parent, retained, event, pre_date, paths, frozen_ids, ids):
    event_id = event['event_id']
    require(pre_date in PRE_DATES and event_id in PARENTS and parent['pair_id'] == PARENTS[event_id]
            and parent['pair_id'] in frozen_ids, 'Exact fixed earlier date/event/frozen parent required')
    require(all(retained.get(k) == parent.get(k) for k in
                ('pair_id', 'group_id', 'anchor_event_ids', 'status', 'blocking_reasons')),
            'V5 retained parent differs from frozen original')
    require(parent['group_id'] == GROUP and parent['split'] == 'validation' and parent['tile'] == '36UXA'
            and parent['anchor_event_ids'] == [event_id]
            and (parent['pre_date'], parent['post_date'], parent['interval_days']) == ('2018-07-26', POST_DATE, 5)
            and parent['status'] == 'blocked_for_temporal_training'
            and set(parent['blocking_reasons']) == COVERAGE_BLOCKS and len(parent['blocking_reasons']) == 2,
            'Frozen parent identity or noncoverage provenance blockers changed')
    require(parent['grid'] == event['crop_grid'] and parent['grid']['crs'] == 'EPSG:32636'
            and event['annotation_image_date'] == POST_DATE and not parent['conflicting_label_event_ids'],
            'Frozen grid/anchor/date provenance changed')
    context, before, after = ids
    require(context == parent['context_event_ids'] and after == parent['label_event_ids_by_post_date']
            and set(before) <= set(after), 'Temporal catalogue lineage changed')
    sources = {side: {'scene_id': SCENES[day], 'path': paths[day]}
               for side, day in (('pre', pre_date), ('post', POST_DATE))}
    lineage = {'parent_pair_id': parent['pair_id'], 'parent_pair_sha256': object_hash(parent),
               'pre_date': pre_date, 'post_date': POST_DATE, 'sources': sources,
               'grid': parent['grid'], 'post_label_ids': after, 'pre_label_ids': before}
    identifier = 'derived-earlier-'+object_hash(lineage)[:20]
    require(identifier not in frozen_ids, 'Derived pair must not impersonate frozen/V11 identity')
    pair = deepcopy(parent)
    pair.update(pair_id=identifier, pair_kind='derived_earlier_validation_source_pair',
                parent_pair_id=parent['pair_id'], parent_pair_sha256=object_hash(parent),
                parent_status=parent['status'], parent_blocking_reasons=list(parent['blocking_reasons']),
                parent_pre_date=parent['pre_date'], parent_post_date=parent['post_date'],
                source_replacement=sources, original_tile=parent['tile'], tile='36UXB',
                pre_date=pre_date, post_date=POST_DATE,
                interval_days=(date.fromisoformat(POST_DATE)-date.fromisoformat(pre_date)).days,
                label_event_ids_by_pre_date=before, label_event_ids_by_post_date=after,
                temporal_reference_rule='cumulative_post_AND_NOT_cumulative_pre_recomputed',
                status='candidate_for_pixel_review', blocking_reasons=[], pixel_quality='unreviewed',
                visual_review='unreviewed', training_approved=False, evaluation_approved=False)
    for side in ('pre', 'post'):
        pair[side+'_scene_id'] = sources[side]['scene_id']; pair[side+'_path'] = sources[side]['path']
    return pair


def temporal_summary(pair, parent, events):
    import numpy as np
    from scipy.ndimage import binary_erosion
    from assess_pixel_quality import core_grid, label_target
    from validate_sample import burn, temporal_masks
    grid = core_grid(pair['grid']); before_day, after_day = map(date.fromisoformat, (pair['pre_date'], pair['post_date']))
    records = [(date.fromisoformat(events[key]['properties']['observation_date']), events[key]['geometry'])
               for key in pair['label_event_ids_by_post_date']]
    before, after, change = temporal_masks(records, before_day, after_day, grid)
    independent_before = burn([g for d, g in records if d <= before_day], grid)
    independent_after = burn([g for d, g in records if d <= after_day], grid)
    expected = ((independent_after > 0) & (independent_before == 0)).astype('uint8')
    require(np.array_equal(before, independent_before) and np.array_equal(after, independent_after)
            and np.array_equal(change, expected) and np.array_equal(change, label_target(pair, events)),
            'Cumulative temporal reference differs from recomputed boolean difference')
    anchor = burn([events[pair['anchor_event_ids'][0]]['geometry']], grid).astype(bool)
    target = anchor & change.astype(bool); interior = binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
    interval_only = burn([g for d, g in records if before_day < d <= after_day], grid)
    old = label_target(parent, events)
    masks = {'cumulative_pre': before, 'cumulative_post': after, 'reference_new_loss': change,
             'anchor_target': target, 'anchor_interior': interior}
    return {'pair_id': pair['pair_id'], 'pre_date': pair['pre_date'], 'post_date': pair['post_date'],
            'interval_days': pair['interval_days'], 'pre_event_ids': pair['label_event_ids_by_pre_date'],
            'post_event_ids': pair['label_event_ids_by_post_date'],
            'pixels': {name: int(value.sum()) for name, value in masks.items()},
            'mask_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in masks.items()},
            'interval_only_excess_pixels': int(((interval_only > 0) & (change == 0)).sum()),
            'changed_reference_pixels_from_parent': int((old != change).sum()),
            'parent_reference_sha256': hashlib.sha256(old.tobytes()).hexdigest(),
            'independent_boolean_difference_checked': True,
            'verification_scope': 'Date-aware cumulative burns and boolean difference; shared rasterizer, no independent annotation truth'}


def build_inputs(args):
    from select_validation_cases import records
    from prepare_validation_review import validate_validation_pair
    from pixelqa_scene_metadata import qa_scene_metadata
    from assess_pixel_quality import load_events
    protocol = read(args.protocol); rule = protocol['rule']
    expected = {'group_id': GROUP, 'events': PARENTS, 'pre_dates': list(PRE_DATES), 'post_date': POST_DATE,
                'source_scenes': SCENES, 'source_tile': '36UXB', 'source_crs': 'EPSG:32636',
                'qa_policy_id': POLICY, 'context_padding_m': 1200, 'context_radius_pixels': 2,
                'expected_pairs': 4, 'observation_requests': 8, 'maximum_distinct_observations': 6,
                'maximum_cloud_model_calls': 6, 'maximum_fused_band_requests': 66,
                'phase': 'qa_only_no_27_channel_eligibility'}
    require(protocol['schema_version'] == 15 and protocol['resources'] == RESOURCES
            and all(rule.get(k) == v for k, v in expected.items()), 'V15 fixed scope/resource contract changed')
    sources = {role: rooted(receipt['path']) for role, receipt in protocol['source_files'].items()}
    for role, path in sources.items():
        require(digest(path) == protocol['source_files'][role]['sha256'], 'Frozen evidence changed: '+role)
    require('Solutions/screen_earlier_validation.py' in protocol.get('implementation_sha256', {}), 'Freeze tested implementation first')
    for name, checksum in {**protocol['frozen_implementation_sha256'], **protocol['implementation_sha256']}.items():
        require(digest(rooted(name)) == checksum, 'Frozen implementation changed: '+name)
    prior = read(sources['v11_screen']); old_meta = read(sources['v11_metadata'])
    require(old_meta['execution_status'] == 'completed' and old_meta['artifact_sha256']['screen.json'] == digest(sources['v11_screen'])
            and len(prior['results']) == 2 and all(r['quality_status'] == 'excluded_by_automatic_quality_screen' for r in prior['results'])
            and prior['groups_observed_by_automatic_qa'] == [GROUP], 'Prior V11 exclusions/exposure changed')
    unchanged_assets(prior['source_asset_receipts'])
    acquired = read(sources['acquisition_result']); acquired_meta = read(sources['acquisition_metadata'])
    require(acquired['transaction_state'] == 'complete' and acquired['new_assets'] == 116
            and acquired_meta['execution_status'] == 'completed' and acquired_meta['mode'] == 'publish'
            and acquired_meta['artifact_sha256']['result.json'] == digest(sources['acquisition_result']), 'V14 publication changed/incomplete')
    assets = {}
    for scene, receipts in acquired['source_receipts'].items():
        require(scene in (SCENES[d] for d in PRE_DATES), 'Unexpected earlier acquisition source')
        for target, receipt in receipts.items():
            name, checked = check_asset({**receipt, 'path': str(Path('Datasets/Sentinel-2')/scene/target)})
            require(name not in assets, 'Duplicate source receipt'); assets[name] = checked
    for name, receipt in prior['source_asset_receipts'].items():
        if '/'+SCENES[POST_DATE]+'/' in name:
            _, checked = check_asset({**receipt, 'path': name}); require(name not in assets, 'Duplicate post receipt'); assets[name] = checked
    require(len(assets) == 174, 'Exactly three complete 58-file source scenes required')
    requirements = read(sources['source_requirements']); req_events = unique(requirements['events'], 'event_id')
    require(requirements['group_id'] == GROUP and set(req_events) == set(PARENTS)
            and requirements['context_padding_metres'] == 1200, 'Original event context changed')
    paths = source_paths(); scene_metadata = {}; expected_meta = unique(acquired['final_scenes'], 'scene_id')
    expected_meta[SCENES[POST_DATE]] = prior['scene_metadata'][POST_DATE]
    for day, path in paths.items():
        meta = qa_scene_metadata(rooted(path), ROOT)
        require(path == expected_meta[SCENES[day]]['path'] and meta == expected_meta[SCENES[day]]
                and meta['metadata_usable'] and meta['cloud_model_usable']
                and meta['shadow_geometry_usable'] and not meta['issues'] and meta['acquisition_date'] == day
                and meta['crs'] == 'EPSG:32636' and meta['processing_baseline'] == '02.06'
                and meta['nonzero_offset'] is False and meta['quantification_value'] == 10000, 'Actual source prerequisites changed')
        for event in req_events.values():
            require(contains(meta['grid_10m']['bounds'], event['crop_grid']['bounds'])
                    and contains(meta['grid_10m']['bounds'], event['required_qa_context_bounds']), 'Full original source context is missing')
        scene_metadata[day] = meta
    catalogue, splits, inventory = (read(sources[k]) for k in ('catalogue', 'splits', 'inventory'))
    properties = {f['id']: f['properties'] for f in catalogue['features']}; events = load_events(sources['catalogue'].parent)
    parents, frozen_ids = {}, set()
    for parent in records(sources['pairs']):
        require(parent['pair_id'] not in frozen_ids, 'Duplicate frozen pair ID'); frozen_ids.add(parent['pair_id'])
        if parent['pair_id'] in PARENTS.values(): parents[parent['pair_id']] = parent
    frozen_ids.update(p['pair_id'] for p in prior['derived_pairs'])
    retained = unique(inventory['pairs'], 'pair_id'); training = list(records(sources['training_manifest']))
    read(sources['training_metadata'])
    pairs, geometry = [], []
    for day in PRE_DATES:
        for event_id, parent_id in PARENTS.items():
            parent = parents[parent_id]; ids = context_ids(parent, events, properties, day)
            pair = derive_pair(parent, retained[parent_id], req_events[event_id], day, paths, frozen_ids, ids)
            validate_validation_pair(pair, event_id, catalogue, splits, training)
            pairs.append(pair); geometry.append(temporal_summary(pair, parent, events))
    require(len(pairs) == len({p['pair_id'] for p in pairs}) == 4, 'Four distinct derived pairs required')
    unchanged_assets(assets)
    return {'protocol': protocol, 'sources': sources, 'assets': assets, 'scene_metadata': scene_metadata,
            'pairs': pairs, 'geometry': geometry, 'events': events,
            'prior_cache_receipts': {r['cache_key']: r for r in prior['cache_receipts']},
            'prior_manual_exposed_group_ids': inventory['exposed_group_ids'],
            'prior_automatic_exposure': {'screen_sha256': digest(sources['v11_screen']), 'group_ids': [GROUP],
                                         'excluded_pair_ids': [r['pair_id'] for r in prior['results']]}}


def metadata(args):
    import resource
    return {'execution_status': 'completed', 'schema_version': 15, 'mode': args.mode,
            'protocol_sha256': digest(args.protocol), 'implementation_sha256': digest(__file__),
            'pair_count': 4, 'group_count': 1, 'labels_approved': 0, 'new_imagery_reviewed': 0,
            'forest_model_executions': 0, 'validation_predictions_run': False,
            'all_27_channels_checked': False, 'validation_eligible': False,
            'training_approved': False, 'evaluation_approved': False,
            'worker_high_water_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                             (2**20 if sys.platform == 'darwin' else 1024), 2)}


def save_artifacts(output, result, meta, name):
    write(output/(name+'.json'), result)
    (output/'derived_pairs.jsonl').write_text(''.join(json.dumps(p, sort_keys=True, allow_nan=False)+'\n' for p in result['derived_pairs']))
    (output/'report.md').write_text('# Earlier-date validation '+name+'\n\nFour fixed pairs, one geographic group. '
        'No class labels, forest predictions or full 27-channel eligibility. See JSON for every outcome and reference hash.\n\n'
        'Execution: '+meta['execution_status']+'; processing errors: '+str(len(result.get('processing_errors', [])))+'.\n')
    meta['artifact_sha256'] = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output/'metadata.json', meta)


def prepare(args, output, inputs):
    result = {'schema_version': 15, 'protocol_sha256': digest(args.protocol), 'derived_pairs': inputs['pairs'],
              'temporal_geometry': inputs['geometry'], 'source_asset_receipts': inputs['assets'],
              'source_scene_metadata': inputs['scene_metadata'],
              'prior_manual_exposed_group_ids': inputs['prior_manual_exposed_group_ids'],
              'prior_automatic_exposure': inputs['prior_automatic_exposure'],
              'satellite_pixel_decodes': 0, 'cloud_model_executions': 0, 'labels_approved': 0}
    meta = metadata(args); meta.update(checks={'exact_date_aware_lineage': True, 'full_train_test_isolation': True,
        'cumulative_reference_recomputed': True, 'source_hashes_and_prerequisites': True}, satellite_pixel_decodes=0, cloud_model_executions=0)
    save_artifacts(output, result, meta, 'lineage')


def require_lineage(args, inputs):
    require(args.lineage is not None, 'Completed lineage preparation required')
    run = args.lineage.resolve(); require(run.is_relative_to(ROOT.resolve()), 'Lineage path outside project')
    meta = read(run/'metadata.json'); result = read(run/'lineage.json')
    require(meta['execution_status'] == 'completed' and meta['mode'] == 'prepare'
            and result['protocol_sha256'] == digest(args.protocol) and result['derived_pairs'] == inputs['pairs']
            and result['temporal_geometry'] == inputs['geometry'] and result['source_asset_receipts'] == inputs['assets'],
            'Completed lineage differs from current exact sources/targets')
    for name, checksum in meta['artifact_sha256'].items():
        path = (run/name).resolve(); require(path.is_relative_to(run) and digest(path) == checksum, 'Lineage artifact changed')


def prevalidate_cache(path, source_id, grid):
    """Read only bounded scalar metadata, then use the frozen complete cache verifier."""
    import numpy as np
    from verify_validation_dataset import read_cache
    require(path.stat().st_size <= 2*2**20, 'Oversized QA cache')
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo('metadata.npy'); require(info.file_size <= 2**20, 'Oversized cache metadata')
        raw = archive.read(info)
    stream = io.BytesIO(raw); version = np.lib.format.read_magic(stream)
    require(version in ((1, 0), (2, 0)), 'Unsupported cache metadata format')
    header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = header(stream)
    require(shape == () and not fortran and dtype.kind == 'U' and dtype.itemsize <= 2**20
            and len(raw) == stream.tell()+dtype.itemsize, 'Unsafe cache scalar allocation')
    expected = json.loads(str(np.load(io.BytesIO(raw), allow_pickle=False)))
    require(expected['source_id'] == source_id and expected['grid'] == grid
            and expected['cache_path'] == relative(path), 'Cache source/grid binding changed')
    read_cache(expected, POLICY)


def collect_roster(pairs, operation):
    """Keep all fixed cases on per-pair errors, without a completed-QA claim."""
    rows, contexts, caches, errors = [], [], [], []
    for pair in pairs:
        row, context, receipts = None, None, [None, None]
        try:
            row, context, receipts = operation(pair)
            require_processed(row)
            require(len(receipts) == 2 and all(receipts), 'Two completed observation receipts required')
        except Exception as error:
            if row is None or row.get('quality_status') != 'processing_error':
                row = {'pair_id': pair['pair_id'], 'group_id': pair['group_id'],
                       'pre_date': pair['pre_date'], 'post_date': pair['post_date'],
                       'quality_status': 'processing_error', 'quality_reasons': [type(error).__name__+': '+str(error)],
                       'intermediate_quality_outcome': row, 'qa_mask_file': None, 'preview_file': None}
            errors.append({'pair_id': pair['pair_id'], 'reasons': row['quality_reasons']})
            context = {'event_id': pair['anchor_event_ids'][0], 'status': 'unavailable_due_to_processing_error'}
            receipts = [None, None]
        rows.append(row); contexts.append(context); caches.extend(receipts)
    return rows, contexts, caches, errors


def screen(args, output, inputs):
    import rasterio
    import assess_pixel_quality as quality
    require_lineage(args, inputs)
    model = quality.model_identity(); require(quality.policy_id(model) == POLICY, 'Frozen cloud QA policy/runtime changed')
    class CapturedReader(quality.ObservationReader):
        def __init__(self):
            super().__init__(args.cache_root, model); self.observations = []; self.failed_keys = {}
        def observation(self, scene_path, grid):
            meta = self.metadata(scene_path)
            key = quality.object_hash({'source': meta['cache_source_id'], 'grid': grid, 'config_id': self.config_id})
            path = self.cache_root/(key+'.npz')
            require(key not in self.failed_keys, 'Earlier failed observation retained; no automatic retry')
            try:
                if key not in self.memory and path.exists():
                    known = inputs['prior_cache_receipts'].get(key)
                    if known:
                        require(path.stat().st_size == known['bytes'] and digest(path) == known['sha256'], 'Prior V11 cache changed')
                    prevalidate_cache(path, meta['cache_source_id'], grid)
                observation = super().observation(scene_path, grid); self.observations.append(observation)
                return observation
            except Exception as error:
                self.failed_keys[key] = type(error).__name__+': '+str(error)
                raise
    reader = CapturedReader()
    def one_pair(pair):
        start = len(reader.observations)
        row = quality.assess_pair(pair, reader, inputs['events'], output, preview=False)
        if row.get('quality_status') == 'processing_error':
            return row, None, [None, None]
        require(len(reader.observations) == start+2, 'Each completed pair requires two observations')
        pre, post = reader.observations[start:]; receipts = []
        context = anchor_contexts(pair, pair['anchor_event_ids'][0], inputs['events'], pre, post)
        for record in (pre, post):
            path = rooted(record['metadata']['cache_path'])
            require(path.stat().st_size <= 2*2**20, 'Oversized completed cache')
            receipt = {'path': relative(path), 'bytes': path.stat().st_size, 'sha256': digest(path),
                       'cache_key': record['metadata']['cache_key']}
            known = inputs['prior_cache_receipts'].get(receipt['cache_key'])
            require(known is None or all(receipt[k] == known[k] for k in ('path', 'bytes', 'sha256', 'cache_key')),
                    'Prior V11 cached output changed during screen')
            receipts.append(receipt)
        return row, context, receipts
    with rasterio.Env(GDAL_CACHEMAX=32*2**20, GDAL_NUM_THREADS='1'), count_band_requests(quality) as counts:
        rows, contexts, caches, errors = collect_roster(inputs['pairs'], one_pair)
    require(len(reader.observations) <= reader.hits+reader.misses <= 8 and reader.misses <= 6
            and counts['requests'] <= 66, 'Observation/window bound exceeded')
    if not errors:
        require(len(reader.observations) == reader.hits+reader.misses == 8
                and len({c['cache_key'] for c in caches}) == 6 and counts['requests'] == reader.misses*11,
                'Completed observation/request contract changed')
    source_stats = {}
    try:
        for scene in reader.scenes.values():
            quality.validate_source_stats(scene); source_stats.update(scene['source_file_stats'])
        require(set(source_stats) <= set(inputs['assets']), 'QA discovered an unexpected source asset set')
        unchanged_assets(inputs['assets'])
    except Exception as error:
        errors.append({'scope': 'source_integrity', 'reasons': [type(error).__name__+': '+str(error)]})
    if not errors: require(set(source_stats) == set(inputs['assets']), 'Completed source set differs')
    result = {'schema_version': 15, 'protocol_sha256': digest(args.protocol), 'qa_policy_id': POLICY,
        'derived_pairs': inputs['pairs'], 'temporal_geometry': inputs['geometry'], 'results': rows,
        'anchor_contexts': contexts, 'cache_receipts': caches, 'source_asset_receipts': inputs['assets'],
        'source_file_stats': source_stats, 'prior_manual_exposed_group_ids': inputs['prior_manual_exposed_group_ids'],
        'prior_automatic_exposure': inputs['prior_automatic_exposure'], 'groups_observed_by_automatic_qa': [GROUP],
        'processing_errors': errors, 'observation_requests': len(reader.observations),
        'distinct_observations': len({c['cache_key'] for c in caches if c}),
        'new_cache_hits': reader.hits, 'new_cache_misses': reader.misses, 'cloud_model_executions': reader.misses,
        'fused_band_window_requests': counts['requests'], 'labels_approved': 0, 'new_imagery_reviewed': 0,
        'all_27_channels_checked': False, 'validation_eligible': False, 'forest_model_executions': 0}
    meta = metadata(args); meta.update(cloud_model_executions=reader.misses, fused_band_window_requests=counts['requests'],
        checks={'exact_lineage': True, 'cumulative_targets': True, 'full_train_test_isolation': True,
                'source_hashes_and_stats': not any(e.get('scope') == 'source_integrity' for e in errors),
                'zero_processing_errors': not errors})
    if errors: meta['execution_status'] = 'incomplete_processing_errors'
    save_artifacts(output, result, meta, 'screen')


def verify(args, output, inputs):
    import numpy as np
    import rasterio
    from assess_pixel_quality import MASK_NAMES, classify_quality, core_grid, label_target, model_identity, policy_id
    require_lineage(args, inputs)
    run = args.run.resolve(); require(run.is_relative_to(ROOT.resolve()), 'Screen path outside project')
    meta = read(run/'metadata.json'); result = read(run/'screen.json')
    require(meta['execution_status'] == 'completed' and meta['mode'] == 'screen'
            and result['protocol_sha256'] == digest(args.protocol) and result['derived_pairs'] == inputs['pairs']
            and result['temporal_geometry'] == inputs['geometry'] and result['source_asset_receipts'] == inputs['assets']
            and result['processing_errors'] == [] and result['qa_policy_id'] == policy_id(model_identity()) == POLICY,
            'Completed screen differs from lineage/sources/frozen QA')
    for name, checksum in meta['artifact_sha256'].items():
        path = (run/name).resolve(); require(path.is_relative_to(run) and digest(path) == checksum, 'Screen artifact changed')
    require(len(result['results']) == len(result['anchor_contexts']) == 4 and len(result['cache_receipts']) == 8,
            'Expected four outcomes and eight bound cache requests')
    for index, pair in enumerate(inputs['pairs']):
        row = result['results'][index]; require_processed(row); observations = []
        for side, receipt in zip(('pre', 'post'), result['cache_receipts'][2*index:2*index+2]):
            observation = load_safe_cache(receipt, row['per_date'][side]['observation'])
            require(observation['metadata']['scene_id'] == pair[side+'_scene_id']
                    and observation['metadata']['grid'] == pair['grid'], 'Replay source/grid differs')
            observations.append(observation)
        pre, post = observations; target = label_target(pair, inputs['events'])
        status, reasons, masks, crop, target_metrics = classify_quality(pre, post, target,
            pre['metadata']['context_incomplete'] or post['metadata']['context_incomplete'])
        require((row['quality_status'], row['quality_reasons'], row['crop_metrics'], row['target_metrics'])
                == (status, reasons, crop, target_metrics), 'Cached numeric QA outcome differs')
        path = (run/row['qa_mask_file']).resolve(); require(path.is_relative_to(run), 'QA TIFF outside completed screen')
        with rasterio.open(path) as source:
            expected = np.stack([*(masks[name].astype('uint8') for name in MASK_NAMES), target])
            require(source.count == 6 and source.shape == (64, 64) and source.crs.to_string() == pair['grid']['crs']
                    and source.transform == core_grid(pair['grid'])['transform']
                    and source.descriptions == (*MASK_NAMES, 'reference_new_loss') and np.array_equal(source.read(), expected),
                    'Six-plane QA/reference TIFF reconstruction differs')
        require(anchor_contexts(pair, pair['anchor_event_ids'][0], inputs['events'], pre, post)
                == result['anchor_contexts'][index], 'Anchor context replay differs')
    unchanged_assets(inputs['assets'])
    final = metadata(args); final.update(cloud_model_executions=0, source_jp2_window_reads=0,
        checks={'four_lineages_and_targets': True, 'eight_safe_cache_bindings': True, 'four_qa_tiffs': True,
                'numeric_quality_outcomes': True, 'anchor_contexts': True, 'source_hashes_and_stats': True},
        verification_scope='Shared geometry/QA helpers and saved-array replay; no independent cloud/class truth')
    write(output/'metadata.json', final)
    (output/'report.md').write_text('# Earlier-date QA replay\n\nFour pair outcomes, eight cache bindings and four six-plane QA files replayed. '
                                  'No source pixels decoded, cloud/forest models run or class labels approved.\n')


def worker(args, output):
    require('torch' not in sys.modules, 'Forest model/Torch forbidden')
    inputs = build_inputs(args)
    {'prepare': prepare, 'screen': screen, 'verify': verify}[args.mode](args, output, inputs)
    if args.mode != 'screen': unchanged_assets(inputs['assets'])
    require('torch' not in sys.modules, 'Forest model/Torch forbidden')


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationEarlierScreenV15/protocol.json')
    parser.add_argument('--mode', choices=('prepare', 'screen', 'verify'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lineage', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--cache-root', type=Path, default=ROOT/'Datasets/PixelQA/cache')
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.mode != 'prepare') != (args.lineage is not None) or (args.mode == 'verify') != (args.run is not None):
        parser.error('screen/verify require --lineage; only verify requires --run')
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Retain 512 MiB/1.5 GiB/300 s bounds')
    for name in THREAD_VARIABLES: os.environ[name] = '1'
    if args.worker_output:
        os.nice(10); worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='earlier-date-qa', gdal_cache_mib=32, warp_memory_mib=16)


if __name__ == '__main__': main()
