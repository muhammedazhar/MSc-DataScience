#!/usr/bin/env python3
"""Prepare, screen, and replay the six remaining July validation pairs."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import date
import json
import os
from pathlib import Path
import resource
import sys
import tempfile

from screen_earlier_validation import (
    POST_DATE, SCENES as PRIOR_SCENES, context_ids, temporal_summary,
    require_lineage, prevalidate_cache, collect_roster,
)
from screen_repaired_validation import (
    ROOT, GROUP, PARENTS, POLICY, RESOURCES, COVERAGE_BLOCKS, require, digest,
    object_hash, read, write, rooted, relative, unique, contains, check_asset,
    unchanged_assets, require_processed, anchor_contexts, count_band_requests,
    load_safe_cache,
)

PRE_DATES = ('2018-07-11', '2018-07-06', '2018-07-01')
SCENES = {
    '2018-07-11': 'S2A_MSIL1C_20180711T083601_N0206_R064_T36UXB_20180711T122613',
    '2018-07-06': 'S2B_MSIL1C_20180706T083559_N0206_R064_T36UXB_20180706T123246',
    '2018-07-01': 'S2A_MSIL1C_20180701T083601_N0206_R064_T36UXB_20180701T104347',
    POST_DATE: PRIOR_SCENES[POST_DATE],
}
PROTOCOL = ROOT/'Docs/Reviews/ValidationRemainingScreenV18/protocol.json'
PAIR_COUNT = len(PRE_DATES)*len(PARENTS)
OBSERVATIONS = 2*PAIR_COUNT
DISTINCT = len(SCENES)*len(PARENTS)
WINDOWS = 11*DISTINCT


def derive_pair(parent, retained, event, pre_date, paths, frozen_ids, ids):
    event_id = event['event_id']
    require(pre_date in PRE_DATES and event_id in PARENTS
            and parent['pair_id'] == PARENTS[event_id] and parent['pair_id'] in frozen_ids,
            'Exact fixed date, event and frozen parent required')
    require(all(retained.get(key) == parent.get(key) for key in
                ('pair_id', 'group_id', 'anchor_event_ids', 'status', 'blocking_reasons')),
            'V5 retained parent differs from frozen original')
    require(parent['group_id'] == GROUP and parent['split'] == 'validation'
            and parent['tile'] == '36UXA' and parent['anchor_event_ids'] == [event_id]
            and (parent['pre_date'], parent['post_date'], parent['interval_days']) ==
            ('2018-07-26', POST_DATE, 5) and parent['status'] == 'blocked_for_temporal_training'
            and set(parent['blocking_reasons']) == COVERAGE_BLOCKS and len(parent['blocking_reasons']) == 2,
            'Frozen parent identity or noncoverage provenance blockers changed')
    require(parent['grid'] == event['crop_grid'] and parent['grid']['crs'] == 'EPSG:32636'
            and event['annotation_image_date'] == POST_DATE and not parent['conflicting_label_event_ids'],
            'Frozen grid, anchor or date provenance changed')
    context, before, after = ids
    require(context == parent['context_event_ids'] and after == parent['label_event_ids_by_post_date']
            and set(before) <= set(after), 'Temporal catalogue lineage changed')
    sources = {side: {'scene_id': SCENES[day], 'path': paths[day]}
               for side, day in (('pre', pre_date), ('post', POST_DATE))}
    parent_hash = object_hash(parent)
    lineage = {'parent_pair_id': parent['pair_id'], 'parent_pair_sha256': parent_hash,
               'pre_date': pre_date, 'post_date': POST_DATE, 'sources': sources, 'grid': parent['grid'],
               'post_label_ids': after, 'pre_label_ids': before}
    identifier = 'derived-remaining-'+object_hash(lineage)[:20]
    interval = (date.fromisoformat(POST_DATE)-date.fromisoformat(pre_date)).days
    require(identifier not in frozen_ids and 5 <= interval <= 30, 'Pair identity or date interval differs')
    pair = deepcopy(parent)
    pair.update(pair_id=identifier, pair_kind='derived_earlier_validation_source_pair',
                parent_pair_id=parent['pair_id'], parent_pair_sha256=parent_hash,
                parent_status=parent['status'], parent_blocking_reasons=list(parent['blocking_reasons']),
                parent_pre_date=parent['pre_date'], parent_post_date=parent['post_date'],
                source_replacement=sources, original_tile=parent['tile'], tile='36UXB',
                pre_date=pre_date, post_date=POST_DATE, interval_days=interval,
                label_event_ids_by_pre_date=list(before), label_event_ids_by_post_date=list(after),
                temporal_reference_rule='cumulative_post_AND_NOT_cumulative_pre_recomputed',
                status='candidate_for_pixel_review', blocking_reasons=[], pixel_quality='unreviewed',
                visual_review='unreviewed', training_approved=False, evaluation_approved=False)
    for side, source in sources.items():
        pair[side+'_scene_id'] = source['scene_id']
        pair[side+'_path'] = source['path']
    return pair


def checked_inputs():
    protocol = read(PROTOCOL)
    expected = {'group_id': GROUP, 'events': PARENTS, 'pre_dates': list(PRE_DATES),
                'post_date': POST_DATE, 'source_scenes': SCENES, 'source_tile': '36UXB',
                'source_crs': 'EPSG:32636', 'qa_policy_id': POLICY, 'context_padding_m': 1200,
                'context_radius_pixels': 2, 'expected_pairs': PAIR_COUNT,
                'observation_requests': OBSERVATIONS, 'maximum_distinct_observations': DISTINCT,
                'maximum_cloud_model_calls': DISTINCT, 'maximum_fused_band_requests': WINDOWS,
                'phase': 'qa_only_no_27_channel_eligibility'}
    require(protocol['schema_version'] == 18 and protocol['resources'] == RESOURCES
            and all(protocol['rule'].get(k) == v for k, v in expected.items()), 'V18 protocol differs')
    sources = {}
    for role, receipt in protocol['source_files'].items():
        path = rooted(receipt['path'])
        require(digest(path) == receipt['sha256'], 'Pinned evidence changed: '+role)
        sources[role] = path
    implementation = protocol['implementation_sha256']
    require(str(Path(__file__).relative_to(ROOT)) in implementation, 'Pin the tested implementation first')
    for name, checksum in {**protocol['frozen_implementation_sha256'], **implementation}.items():
        require(digest(rooted(name)) == checksum, 'Pinned implementation changed: '+name)
    return sources


def checked_scenes(sources, prior, requirements):
    from pixelqa_scene_metadata import qa_scene_metadata

    acquired = read(sources['acquisition_result'])
    meta = read(sources['acquisition_metadata'])
    acquired_assets = acquired['source_receipts']
    require(acquired['transaction_state'] == 'complete'
            and set(acquired_assets) == {SCENES[day] for day in PRE_DATES}
            and acquired['new_assets'] == 174 and meta['execution_status'] == 'completed'
            and meta['mode'] == 'publish'
            and meta['artifact_sha256']['result.json'] == digest(sources['acquisition_result']),
            'V17 publication is incomplete or differs')
    pinned = unique(acquired['final_scenes'], 'scene_id')
    pinned[SCENES[POST_DATE]] = prior['scene_metadata'][POST_DATE]
    assets, paths, scene_metadata = {}, {}, {}
    for day, scene in SCENES.items():
        folder = Path('Datasets/Sentinel-2')/scene
        path = str(folder/(scene+'.SAFE'))
        receipts = ({str(folder/target): record for target, record in acquired_assets[scene].items()}
                    if day in PRE_DATES else
                    {name: record for name, record in prior['source_asset_receipts'].items()
                     if Path(name).is_relative_to(folder)})
        require(len(receipts) == 58, 'Each scene requires 58 published assets')
        for name, record in receipts.items():
            _, checked = check_asset({**record, 'path': name})
            require(name not in assets and checked['stat'] == record['stat'], 'Source identity or stats changed')
            assets[name] = checked
        actual = qa_scene_metadata(rooted(path), ROOT)
        require(path == actual['path'] and actual == pinned[scene] and actual['acquisition_date'] == day
                and actual['metadata_usable'] and actual['cloud_model_usable']
                and actual['shadow_geometry_usable'] and not actual['issues'], 'Scene prerequisites changed')
        for event in requirements.values():
            bounds = actual['grid_10m']['bounds']
            require(contains(bounds, event['crop_grid']['bounds'])
                    and contains(bounds, event['required_qa_context_bounds']), 'Fixed QA context lacks coverage')
        paths[day], scene_metadata[day] = path, actual
    return assets, paths, scene_metadata


def build_inputs():
    from assess_pixel_quality import load_events
    from prepare_validation_review import validate_validation_pair
    from select_validation_cases import records

    sources = checked_inputs()
    prior_screens, cache_receipts, prior_assets = [], {}, {}
    for version, count in ((11, 2), (15, 4)):
        screen_path = sources[f'v{version}_screen']
        screen, meta = read(screen_path), read(sources[f'v{version}_metadata'])
        require(meta['execution_status'] == 'completed'
                and meta['artifact_sha256']['screen.json'] == digest(screen_path)
                and screen['groups_observed_by_automatic_qa'] == [GROUP]
                and len(screen['results']) == count
                and all(r['quality_status'] == 'excluded_by_automatic_quality_screen' for r in screen['results']),
                'Earlier exclusions or automatic exposure differ')
        for receipt in screen['cache_receipts']:
            key = receipt['cache_key']
            require(cache_receipts.setdefault(key, receipt) == receipt, 'Prior cache receipts conflict')
        prior_assets.update(screen['source_asset_receipts'])
        prior_screens.append(screen)
    unchanged_assets(prior_assets)
    requirements = read(sources['source_requirements'])
    anchors = unique(requirements['events'], 'event_id')
    require(requirements['group_id'] == GROUP and set(anchors) == set(PARENTS)
            and requirements['context_padding_metres'] == 1200, 'Fixed event context changed')
    assets, paths, scene_metadata = checked_scenes(sources, prior_screens[0], anchors)
    catalogue, splits, inventory = (read(sources[key]) for key in ('catalogue', 'splits', 'inventory'))
    properties = {item['id']: item['properties'] for item in catalogue['features']}
    events = load_events(sources['catalogue'].parent)
    original = unique(records(sources['pairs']), 'pair_id')
    frozen_ids = set(original) | {p['pair_id'] for screen in prior_screens for p in screen['derived_pairs']}
    retained = unique(inventory['pairs'], 'pair_id')
    training = list(records(sources['training_manifest']))
    pairs, geometry = [], []
    for day in PRE_DATES:
        for event_id, parent_id in PARENTS.items():
            parent = original[parent_id]
            ids = context_ids(parent, events, properties, day)
            pair = derive_pair(parent, retained[parent_id], anchors[event_id], day, paths, frozen_ids, ids)
            validate_validation_pair(pair, event_id, catalogue, splits, training)
            frozen_ids.add(pair['pair_id'])
            pairs.append(pair)
            geometry.append(temporal_summary(pair, parent, events))
    unchanged_assets(assets)
    exposure = {
        'screen_sha256': {f'v{version}': digest(sources[f'v{version}_screen']) for version in (11, 15)},
        'group_ids': [GROUP],
        'excluded_pair_ids': [r['pair_id'] for screen in prior_screens for r in screen['results']],
    }
    return {'pairs': pairs, 'geometry': geometry, 'events': events, 'assets': assets,
            'scene_metadata': scene_metadata, 'prior_assets': prior_assets, 'prior_cache_receipts': cache_receipts,
            'prior_manual_exposed_group_ids': inventory['exposed_group_ids'], 'prior_automatic_exposure': exposure}


def common_result(inputs):
    return {'schema_version': 18, 'protocol_sha256': digest(PROTOCOL), 'derived_pairs': inputs['pairs'],
            'temporal_geometry': inputs['geometry'], 'source_asset_receipts': inputs['assets'],
            'source_scene_metadata': inputs['scene_metadata'],
            'prior_manual_exposed_group_ids': inputs['prior_manual_exposed_group_ids'],
            'prior_automatic_exposure': inputs['prior_automatic_exposure'], 'labels_approved': 0,
            'new_imagery_reviewed': 0, 'forest_model_executions': 0, 'all_27_channels_checked': False,
            'validation_eligible': False}


def prepare(args, output, inputs):
    write(output/'lineage.json', {**common_result(inputs), 'satellite_pixel_decodes': 0,
                                 'cloud_model_executions': 0})
    (output/'derived_pairs.jsonl').write_text(''.join(json.dumps(pair, sort_keys=True)+'\n'
                                                    for pair in inputs['pairs']))
    return {'checks': {'exact_lineage': True, 'cumulative_targets': True,
                       'full_train_test_isolation': True, 'source_hashes_and_prerequisites': True}}


def screen(args, output, inputs):
    import rasterio
    import assess_pixel_quality as quality

    require_lineage(args, inputs)
    model = quality.model_identity()
    require(quality.policy_id(model) == POLICY, 'Frozen QA policy or model identity differs')
    known_caches = inputs['prior_cache_receipts']

    class Reader(quality.ObservationReader):
        def __init__(self):
            super().__init__(ROOT/'Datasets/PixelQA/cache', model)
            self.observations = []
            self.failed = set()

        def observation(self, scene_path, grid):
            source_id = self.metadata(scene_path)['cache_source_id']
            key = quality.object_hash({'source': source_id, 'grid': grid, 'config_id': self.config_id})
            path = self.cache_root/(key+'.npz')
            require(key not in self.failed, 'Previously failed observation is retained')
            try:
                if key not in self.memory and path.exists():
                    known = known_caches.get(key)
                    if known:
                        require(path.stat().st_size == known['bytes'] and digest(path) == known['sha256'],
                                'Prior cache differs')
                    prevalidate_cache(path, source_id, grid)
                record = super().observation(scene_path, grid)
                self.observations.append(record)
                return record
            except Exception:
                self.failed.add(key)
                raise

    reader = Reader()

    def one_pair(pair):
        start = len(reader.observations)
        row = quality.assess_pair(pair, reader, inputs['events'], output, preview=False)
        if row['quality_status'] == 'processing_error':
            return row, None, [None, None]
        pre, post = reader.observations[start:]
        context = anchor_contexts(pair, pair['anchor_event_ids'][0], inputs['events'], pre, post)
        receipts = []
        for record in (pre, post):
            path = rooted(record['metadata']['cache_path'])
            require(path.stat().st_size <= 2*2**20, 'Cache exceeds bound')
            receipt = {'path': relative(path), 'bytes': path.stat().st_size, 'sha256': digest(path),
                       'cache_key': record['metadata']['cache_key']}
            require(known_caches.get(receipt['cache_key'], receipt) == receipt, 'Prior cache changed')
            receipts.append(receipt)
        return row, context, receipts

    with rasterio.Env(GDAL_CACHEMAX=32*2**20, GDAL_NUM_THREADS='1'), count_band_requests(quality) as counts:
        rows, contexts, caches, errors = collect_roster(inputs['pairs'], one_pair)
    distinct = len({receipt['cache_key'] for receipt in caches if receipt})
    stats = {path: stat for record in reader.scenes.values()
             for path, stat in record['source_file_stats'].items()}
    require(reader.hits+reader.misses <= OBSERVATIONS and reader.misses <= DISTINCT
            and counts['requests'] <= WINDOWS, 'Observation or window bound exceeded')
    if not errors:
        require(len(reader.observations) == reader.hits+reader.misses == OBSERVATIONS
                and distinct == DISTINCT and counts['requests'] == reader.misses*11,
                'Completed observation contract differs')
        require(stats == {name: receipt['stat'] for name, receipt in inputs['assets'].items()},
                'Observed source file stats differ from verified assets')
    result = {**common_result(inputs), 'qa_policy_id': POLICY, 'results': rows,
              'anchor_contexts': contexts, 'cache_receipts': caches, 'source_file_stats': stats,
              'groups_observed_by_automatic_qa': [GROUP], 'processing_errors': errors,
              'observation_requests': len(reader.observations), 'distinct_observations': distinct,
              'new_cache_hits': reader.hits, 'new_cache_misses': reader.misses,
              'cloud_model_executions': reader.misses, 'fused_band_window_requests': counts['requests']}
    write(output/'screen.json', result)
    return {'execution_status': 'incomplete_processing_errors' if errors else 'completed',
            'checks': {'exact_lineage': True, 'zero_processing_errors': not errors},
            'status_counts': dict(Counter(row['quality_status'] for row in rows))}


def verify(args, output, inputs):
    import numpy as np
    import rasterio
    import assess_pixel_quality as quality

    require_lineage(args, inputs)
    run = args.run.resolve()
    require(run.is_relative_to(ROOT), 'Screen path outside project')
    meta, result = read(run/'metadata.json'), read(run/'screen.json')
    require(meta['execution_status'] == 'completed' and meta['mode'] == 'screen'
            and result['qa_policy_id'] == quality.policy_id(quality.model_identity()) == POLICY
            and all(result[key] == value for key, value in common_result(inputs).items())
            and not result['processing_errors'], 'Screen differs from the verified inputs or is incomplete')
    for name, checksum in meta['artifact_sha256'].items():
        path = (run/name).resolve()
        require(path.is_relative_to(run) and digest(path) == checksum, 'Screen artifact changed')
    require(len(result['results']) == len(result['anchor_contexts']) == PAIR_COUNT
            and len(result['cache_receipts']) == OBSERVATIONS, 'Incorrect replay roster size')

    class ReplayReader:
        config_id = POLICY

        def __init__(self, bindings):
            self.bindings = iter(bindings)

        def observation(self, path, grid):
            expected_path, expected_grid, record = next(self.bindings)
            require(path == expected_path and grid == expected_grid, 'Replay source or grid differs')
            return record

    with tempfile.TemporaryDirectory(dir=output) as directory:
        replay = Path(directory)
        for index, pair in enumerate(inputs['pairs']):
            row = result['results'][index]
            records = []
            bindings = []
            for side, receipt in zip(('pre', 'post'), result['cache_receipts'][2*index:2*index+2]):
                record = load_safe_cache(receipt, row['per_date'][side]['observation'])
                require(record['metadata']['scene_id'] == pair[side+'_scene_id']
                        and record['metadata']['grid'] == pair['grid'], 'Replay cache identity differs')
                records.append(record)
                bindings.append((pair[side+'_path'], pair['grid'], record))
            checked = quality.assess_pair(pair, ReplayReader(bindings), inputs['events'], replay, preview=False)
            require_processed(checked)
            require(checked == row, 'Replayed numeric QA outcome differs')
            path = (run/row['qa_mask_file']).resolve()
            require(path.is_relative_to(run) and path.stat().st_size <= 2*2**20, 'QA mask path or size differs')
            with rasterio.open(path) as saved, rasterio.open(replay/row['qa_mask_file']) as expected:
                require(saved.profile == expected.profile and saved.descriptions == expected.descriptions
                        and np.array_equal(saved.read(), expected.read()), 'QA mask differs from cache replay')
            checked_context = anchor_contexts(pair, pair['anchor_event_ids'][0], inputs['events'], *records)
            require(checked_context == result['anchor_contexts'][index], 'Replayed anchor context differs')
    return {'checks': {'numeric_qa_outcomes': True, 'safe_cache_bindings': True,
                       'qa_masks': True, 'anchor_contexts': True},
            'pairs_replayed': PAIR_COUNT, 'cache_bindings_replayed': OBSERVATIONS,
            'cloud_model_executions': 0, 'source_jp2_window_reads': 0}


def worker(args, output):
    inputs = build_inputs()
    details = {'prepare': prepare, 'screen': screen, 'verify': verify}[args.mode](args, output, inputs)
    unchanged_assets(inputs['assets'])
    unchanged_assets(inputs['prior_assets'])
    require('torch' not in sys.modules, 'Unexpected Torch import during automatic QA')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'execution_status': 'completed', 'schema_version': 18, 'mode': args.mode,
        'protocol_sha256': digest(PROTOCOL), 'pair_count': PAIR_COUNT, 'group_count': 1,
        'labels_approved': 0, 'validation_eligible': False, 'all_27_channels_checked': False,
        'forest_model_executions': 0, 'new_imagery_reviewed': 0,
        'source_hashes_checked': len(inputs['assets']),
        'artifact_sha256': {str(path.relative_to(output)): digest(path)
                            for path in sorted(output.rglob('*')) if path.is_file()},
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2),
        **details,
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'screen', 'verify'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lineage', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(protocol=PROTOCOL, max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    if (args.mode != 'prepare') != (args.lineage is not None) or (args.mode == 'verify') != (args.run is not None):
        parser.error('screen and verify require --lineage. Only verify requires --run.')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='remaining-pair-qa',
                    gdal_cache_mib=32, warp_memory_mib=16)


if __name__ == '__main__':
    main()
