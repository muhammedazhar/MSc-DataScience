#!/usr/bin/env python3
"""Prepare four fresh pairs, screen their inputs, and replay saved QA exactly."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile

from preflight_fresh_validation_sources import PROTOCOL, checked_protocol
from screen_earlier_validation import collect_roster, prevalidate_cache, require_lineage, temporal_summary
from screen_repaired_validation import (
    POLICY, ROOT, anchor_contexts, contains, count_band_requests, digest, load_safe_cache,
    object_hash, read, require, require_processed, rooted, unchanged_assets, unique, write,
)


PRE_DATES = ('2018-07-21', '2018-07-16')
PARENTS = ('pair-4da0c1fed3e02525a046', 'pair-21bb6f74457c8d1631a5')
QA_PASSED = 'passed_automatic_screen_pending_visual_review'
QA_EXCLUDED = 'excluded_by_automatic_quality_screen'
PAIR_COUNT = 4
OBSERVATION_COUNT = 8
NEW_CACHE = ROOT/'Datasets/ValidationPilot/v24/cache'


def checked_artifacts(directory, mode):
    run = directory.resolve()
    require(run.is_relative_to(ROOT.resolve()), 'Artifact directory outside the project')
    meta = read(run/'metadata.json')
    require(meta['execution_status'] == 'completed' and meta['mode'] == mode
            and meta['protocol_sha256'] == digest(PROTOCOL), 'Phase is incomplete or bound to another protocol')
    for name, checksum in meta['artifact_sha256'].items():
        path = (run/name).resolve()
        require(path.is_relative_to(run) and digest(path) == checksum, 'Phase artifact changed: '+name)
    return run, meta


def checked_publication(directory, protocol):
    from pixelqa_scene_metadata import qa_scene_metadata

    run, meta = checked_artifacts(directory, 'verify')
    require({'result.json', 'journal_snapshot.json'} <= meta['artifact_sha256'].keys(),
            'Expected an independently verified acquisition publication')
    result = read(run/'result.json')
    selected = {row['product_uri'].removesuffix('.SAFE'): row['acquisition_date'] for row in protocol['roster']}
    require(tuple(row['acquisition_date'] for row in protocol['roster']) == PRE_DATES
            and len(selected) == result['new_scenes'] == 2 and result['new_assets'] == 116
            and result['transaction_state'] == 'complete' and result['protocol_sha256'] == digest(PROTOCOL)
            and result['satellite_pixel_reads'] == result['model_executions'] == result['labels_approved'] == 0
            and set(result['source_receipts']) == selected.keys(), 'Published acquisition scope differs')
    scenes = unique(result['final_scenes'], 'scene_id')
    require(scenes.keys() == selected.keys(), 'Published scene roster differs')
    assets, by_date = {}, {}
    for scene, day in selected.items():
        product = Path('Datasets/Sentinel-2')/scene
        path = str(product/(scene+'.SAFE'))
        post = qa_scene_metadata(rooted(path), ROOT)
        require(post == scenes[scene] and post['path'] == path and post['acquisition_date'] == day
                and post['metadata_usable'] and post['cloud_model_usable'] and post['shadow_geometry_usable']
                and post['tile'] == '36UXA' and post['baseline_code'] == 'N0206'
                and '_R064_T36UXA_' in scene, 'Published source metadata differs: '+scene)
        by_date[day] = post
        receipts = result['source_receipts'][scene]
        require(len(receipts) == 58, 'Published source asset coverage differs: '+scene)
        for target, receipt in receipts.items():
            name = str(product/target)
            resolved = rooted(name)
            require(resolved.is_relative_to(rooted(path)) and name not in assets
                    and receipt['bytes'] == receipt['stat'][0] and receipt['if_match_verified']
                    and receipt['content_length_verified'], 'Invalid published asset receipt')
            assets[name] = receipt
    unchanged_assets(assets)
    return by_date, assets, digest(run/'metadata.json')


def frozen_posts(plan, old_quality, old_summary):
    from pixelqa_scene_metadata import qa_scene_metadata

    require(old_summary['config_id'] == POLICY and not old_summary['processing_errors']
            and all(old_summary['checks'].values()), 'Original QA publication is incomplete')
    posts, assets, originals = {}, {}, []
    for row in plan['roster']:
        evidence = row['evidence']
        parent = old_quality[row['pair_id']]
        before, after = (parent['per_date'][side] for side in ('pre', 'post'))
        require(parent['quality_status'] == evidence['qa_status'] == QA_EXCLUDED
                and parent['config_id'] == POLICY and not parent['training_approved']
                and all(parent[key] == row[key] for key in ('group_id', 'pre_date', 'post_date'))
                and before['observation']['scene_id'] == row['pre_scene_id']
                and after['observation']['scene_id'] == row['post_scene_id']
                and before['observation']['grid'] == after['observation']['grid'] == row['grid']
                and [image['crop']['unusable_pixels'] for image in (before, after)] == evidence['endpoint_crop_unusable_pixels']
                and [image['target']['unusable_pixels'] for image in (before, after)] == evidence['endpoint_target_unusable_pixels']
                and after['crop']['unusable_pixels'] == after['target']['unusable_pixels'] == 0,
                'Original QA disposition or endpoint lineage changed: '+row['pair_id'])
        post = qa_scene_metadata(rooted(row['post_path']), ROOT)
        require(all(post[key] == value for key, value in row['verified_post_metadata'].items())
                and post['metadata_usable'] and post['cloud_model_usable'] and post['shadow_geometry_usable']
                and contains(post['grid_10m']['bounds'], row['qa_context_bounds']), 'Original post source metadata changed')
        stats = {name: {'stat': stat} for name, stat in old_summary['source_file_stats'].items()
                 if Path(name).is_relative_to(row['post_path'])}
        require(len(stats) == 58 and not assets.keys() & stats.keys(), 'Original post prerequisite set changed')
        assets.update(stats)
        expected = after['observation']
        path = rooted(expected['cache_path'])
        require(path.stat().st_size <= 2*2**20, 'Original post cache exceeds its limit')
        receipt = {'path': expected['cache_path'], 'bytes': path.stat().st_size,
                   'sha256': digest(path), 'cache_key': expected['cache_key']}
        posts[row['pair_id']] = (post, expected, receipt)
        originals.append({'pair_id': row['pair_id'], 'quality_status': parent['quality_status'],
                          'quality_reasons': parent['quality_reasons'], 'historical_evidence': evidence})
    unchanged_assets(assets)
    return posts, assets, originals


def temporal_event_sets(parent, events, properties, pre_date):
    from shapely.geometry import box

    area = box(*parent['grid']['bounds'])
    context = sorted(key for key, event in events.items() if event['geometry'].intersects(area))
    require(context == parent['context_event_ids'], 'Recomputed temporal context differs from the original pair')
    pre, post = [], []
    for key in context:
        prop = properties[key]
        observed = prop['observation_date']
        require(prop['group_id'] == parent['group_id'] and prop['split'] == 'validation'
                and 'conflicting_observation_dates' not in prop['flags']
                and (prop['temporal_source_ids'] or not pre_date < observed <= parent['post_date']),
                'Conflicting, cross-split, or baseline-only source would enter the derived reference')
        if prop['temporal_source_ids'] and observed <= parent['post_date']:
            post.append(key)
            if observed <= pre_date:
                pre.append(key)
    require(post == parent['label_event_ids_by_post_date'], 'Original post-date catalogue set changed')
    return pre, post


def derive_pair(parent, row, scene, events, properties, frozen_ids):
    before, after = scene['acquisition_date'], parent['post_date']
    interval = (date.fromisoformat(after)-date.fromisoformat(before)).days
    require(parent['pair_id'] == row['pair_id'] and object_hash(parent) == row['parent_pair_sha256']
            and parent['grid'] == row['grid'] and parent['tile'] == '36UXA'
            and parent['grid']['crs'] == scene['crs'] == 'EPSG:32636'
            and parent['anchor_event_ids'] == [row['event_id']] and not parent['blocking_reasons']
            and before in PRE_DATES and 5 <= interval <= 30
            and before < parent['pre_date'] and before < row['annotation_date'] <= after
            and contains(scene['grid_10m']['bounds'], row['qa_context_bounds']),
            'Derived pair changes frozen dates, support, geometry, or source coverage')
    pre_events, post_events = temporal_event_sets(parent, events, properties, before)
    pre = {'scene_id': scene['scene_id'], 'path': scene['path']}
    post = {'scene_id': parent['post_scene_id'], 'path': parent['post_path']}
    sources = {'pre': pre, 'post': post}
    lineage = {'parent_pair_id': parent['pair_id'], 'parent_pair_sha256': row['parent_pair_sha256'],
               'pre_date': before, 'post_date': after, 'sources': sources, 'grid': parent['grid'],
               'post_label_ids': post_events, 'pre_label_ids': pre_events}
    identifier = 'derived-fresh-'+object_hash(lineage)[:20]
    require(identifier not in frozen_ids, 'Derived identity would impersonate a frozen pair')
    pair = deepcopy(parent)
    pair.update(pair_id=identifier, pair_kind='derived_earlier_fresh_validation_source_pair',
                parent_pair_id=parent['pair_id'], parent_pair_sha256=row['parent_pair_sha256'],
                parent_pre_date=parent['pre_date'], parent_post_date=after,
                parent_quality_status=row['evidence']['qa_status'], parent_status=parent['status'],
                parent_blocking_reasons=parent['blocking_reasons'], source_replacement=sources,
                pre_date=before, interval_days=interval, pre_scene_id=pre['scene_id'], pre_path=pre['path'],
                label_event_ids_by_pre_date=pre_events, label_event_ids_by_post_date=post_events,
                temporal_reference_rule='cumulative_post_AND_NOT_cumulative_pre_recomputed',
                training_approved=False, evaluation_approved=False)
    return pair


def build_inputs(args):
    from assess_pixel_quality import load_events
    from prepare_validation_review import validate_validation_pair
    from select_validation_cases import records

    protocol = checked_protocol()
    require(protocol['schema_version'] == 24 and tuple(protocol['candidate_dates']) == PRE_DATES
            and str(Path(__file__).resolve().relative_to(ROOT)) in protocol['implementation_sha256'],
            'Fixed fresh QA scope or worker pin differs')
    sources = {key: rooted(receipt['path']) for key, receipt in protocol['source_files'].items()}
    plan, inventory = (read(sources[key]) for key in ('v23_plan', 'v23_inventory'))
    require(tuple(row['pair_id'] for row in plan['roster']) == PARENTS and not plan['validation_eligible']
            and not plan['labels_approved'] and len(plan['excluded_previous_review_groups']) == 7
            and all(row['group_id'] not in plan['excluded_previous_review_groups'] for row in plan['roster'])
            and inventory['nominal_roster'] == protocol['roster'], 'Frozen V23 selection or prior exposure differs')
    before, assets, publication_sha = checked_publication(args.publication, protocol)
    old_quality = unique((row for row in records(sources['quality']) if row['pair_id'] in PARENTS), 'pair_id')
    after, original_assets, originals = frozen_posts(plan, old_quality, read(sources['old_quality_summary']))
    require(not assets.keys() & original_assets.keys(), 'Acquisition overlaps a frozen original post asset')
    fresh_stats = {name: receipt['stat'] for name, receipt in assets.items()}
    assets.update(original_assets)
    original = unique(records(sources['pairs']), 'pair_id')
    events = load_events(sources['catalogue'].parent)
    catalogue, splits = (read(sources[key]) for key in ('catalogue', 'splits'))
    properties = {feature['id']: feature['properties'] for feature in catalogue['features']}
    training = list(records(sources['training_manifest']))
    pairs, geometry, ids = [], [], set(original)
    for day in PRE_DATES:
        for row in plan['roster']:
            parent = original[row['pair_id']]
            pair = derive_pair(parent, row, before[day], events, properties, ids)
            validate_validation_pair(pair, row['event_id'], catalogue, splits, training)
            pairs.append(pair)
            geometry.append(temporal_summary(pair, parent, events))
            ids.add(pair['pair_id'])
    groups = sorted({pair['group_id'] for pair in pairs})
    require(len(pairs) == PAIR_COUNT and len(groups) == 2, 'Expected four comparisons in two distinct groups')
    result = {'schema_version': 24, 'protocol_sha256': digest(PROTOCOL), 'qa_policy_id': POLICY,
              'derived_pairs': pairs, 'temporal_geometry': geometry, 'source_asset_receipts': assets,
              'publication_verification_metadata_sha256': publication_sha,
              'prior_manual_exposed_group_ids': plan['excluded_previous_review_groups'],
              'original_parent_qa_dispositions': originals, 'planned_geographic_groups': groups,
              'labels_approved': 0, 'validation_eligible': False, 'all_27_channels_checked': False,
              'new_imagery_reviewed': 0, 'forest_model_executions': 0}
    return {'pairs': pairs, 'geometry': geometry, 'assets': assets, 'events': events, 'posts': after,
            'fresh_stats': fresh_stats, 'common': result}


def prepare(args, output, inputs):
    write(output/'lineage.json', inputs['common'])
    (output/'derived_pairs.jsonl').write_text(''.join(json.dumps(pair, sort_keys=True)+'\n' for pair in inputs['pairs']))
    return {'cloud_model_executions': 0, 'source_jp2_window_reads': 0,
            'checks': {'exact_lineage': True, 'cumulative_targets': True, 'split_isolation': True}}


def review_candidates(pairs, rows, contexts, errors):
    if errors:
        return []
    eligible = []
    for pair, row, context in zip(pairs, rows, contexts):
        if (row['quality_status'] == QA_PASSED and row['crop_metrics']['unusable_pixels'] == 0
                and row['target_metrics']['unusable_pixels'] == 0
                and context['positive_interior']['pixels'] >= 4
                and all(context[key]['full_context_available']
                        and all(context[key]['qa'][side+'_unusable_pixels'] == 0
                                and context[key]['finite_cached_inputs'][side]['rgb_and_ndvi_nonfinite_context_pixels'] == 0
                                for side in ('pre', 'post')) for key in ('target', 'positive_interior'))):
            eligible.append(pair['pair_id'])
    return eligible


def screen(args, output, inputs):
    import assess_pixel_quality as quality
    import rasterio

    require_lineage(args, inputs)
    identity = quality.model_identity()
    require(quality.policy_id(identity) == POLICY, 'Frozen pixel QA policy or runtime changed')
    reader = quality.ObservationReader(NEW_CACHE, identity)
    post_bindings = {(record['path'], object_hash(expected['grid'])): (expected, receipt)
                     for record, expected, receipt in inputs['posts'].values()}
    old_cache = {}
    failed_keys = set()
    snapshots = output/'cache_evidence'
    snapshots.mkdir()

    def one_pair(pair):
        observations, receipts = [], []

        class BoundReader:
            config_id = POLICY

            def observation(self, path, grid):
                if path == pair['post_path']:
                    key = (path, object_hash(grid))
                    expected, receipt = post_bindings[key]
                    if key not in old_cache:
                        old_cache[key] = load_safe_cache(receipt, expected)
                    value = old_cache[key]
                else:
                    require(path == pair['pre_path'] and grid == pair['grid'], 'Unexpected fresh QA source or grid')
                    source_id = reader.metadata(path)['cache_source_id']
                    cache_key = quality.object_hash({'source': source_id, 'grid': grid, 'config_id': POLICY})
                    cache_path = reader.cache_root/(cache_key+'.npz')
                    require(cache_key not in failed_keys, 'Prior attempt at this observation failed')
                    try:
                        if cache_path.exists() and cache_key not in reader.memory:
                            prevalidate_cache(cache_path, source_id, grid)
                        value = reader.observation(path, grid)
                    except Exception:
                        failed_keys.add(cache_key)
                        raise
                    require(rooted(value['metadata']['cache_path']) == cache_path.resolve(), 'Unexpected fresh cache path')
                    receipt = {'path': value['metadata']['cache_path'], 'bytes': cache_path.stat().st_size,
                               'sha256': digest(cache_path), 'cache_key': cache_key}
                side = 'pre' if path == pair['pre_path'] else 'post'
                require(value['metadata']['scene_id'] == pair[side+'_scene_id'] and value['metadata']['grid'] == grid
                        and receipt['bytes'] <= 2*2**20, 'QA cache source/grid identity differs')
                path = snapshots/(receipt['cache_key']+'.npz')
                if not path.exists():
                    shutil.copyfile(rooted(receipt['path']), path)
                require(digest(path) == receipt['sha256'], 'Archived QA observation differs from its cache')
                observations.append(value)
                receipts.append({**receipt, 'snapshot': str(path.relative_to(output))})
                return value

        row = quality.assess_pair(pair, BoundReader(), inputs['events'], output, preview=False)
        if row['quality_status'] == 'processing_error':
            return row, None, [None, None]
        require(len(observations) == 2, 'Each pair must be bound to two endpoint observations')
        return row, anchor_contexts(pair, pair['anchor_event_ids'][0], inputs['events'], *observations), receipts

    with rasterio.Env(GDAL_CACHEMAX=32*2**20, GDAL_NUM_THREADS='1'), count_band_requests(quality) as counts:
        rows, contexts, caches, errors = collect_roster(inputs['pairs'], one_pair)
    distinct = len({receipt['cache_key'] for receipt in caches if receipt})
    require(reader.hits+reader.misses <= 4 and counts['requests'] <= 44, 'Fresh source QA resource bound exceeded')
    if not errors:
        stats = {name: stat for scene in reader.scenes.values() for name, stat in scene['source_file_stats'].items()}
        require(reader.hits+reader.misses == 4 and len(old_cache) == 2 and distinct == 6
                and counts['requests'] == reader.misses*11 and stats == inputs['fresh_stats'],
                'Fresh and frozen endpoint bindings differ')
    eligible = review_candidates(inputs['pairs'], rows, contexts, errors)
    status = ('blocked_by_processing_errors' if errors else
              'candidates_ready_for_separate_review' if eligible else 'no_candidates_passed_qa')
    result = {**inputs['common'], 'results': rows, 'anchor_contexts': contexts,
              'cache_receipts': caches, 'processing_errors': errors, 'distinct_observations': distinct,
              'source_file_stats': {name: receipt['stat'] for name, receipt in inputs['assets'].items()},
              'groups_observed_by_automatic_qa': inputs['common']['planned_geographic_groups'],
              'original_post_cache_reuses': 2*len(old_cache) if not errors else None,
              'cloud_model_executions': reader.misses, 'fused_band_window_requests': counts['requests'],
              'review_candidate_status': status, 'review_candidate_pair_ids': eligible}
    write(output/'screen.json', result)
    return {'execution_status': 'incomplete_processing_errors' if errors else 'completed',
            'status_counts': dict(Counter(row['quality_status'] for row in rows)),
            'review_candidate_status': status, 'review_candidate_pair_ids': eligible,
            'cloud_model_executions': reader.misses, 'source_jp2_window_reads': counts['requests']}


def verify(args, output, inputs):
    import assess_pixel_quality as quality
    import numpy as np
    import rasterio

    require_lineage(args, inputs)
    run, meta = checked_artifacts(args.screen, 'screen')
    require('screen.json' in meta['artifact_sha256'], 'Completed screen receipt is missing')
    result = read(run/'screen.json')
    require(not result['processing_errors'] and result['qa_policy_id'] == POLICY == quality.policy_id(quality.model_identity())
            and all(result[key] == value for key, value in inputs['common'].items())
            and len(result['results']) == len(result['anchor_contexts']) == PAIR_COUNT
            and len(result['cache_receipts']) == OBSERVATION_COUNT, 'Screen differs from completed bound inputs')

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
            observations, bindings = [], []
            for side, receipt in zip(('pre', 'post'), result['cache_receipts'][2*index:2*index+2]):
                snapshot = (run/receipt['snapshot']).resolve()
                record = load_safe_cache(receipt, row['per_date'][side]['observation'])
                require(snapshot.is_relative_to(run) and digest(snapshot) == receipt['sha256']
                        and record['metadata']['scene_id'] == pair[side+'_scene_id']
                        and record['metadata']['grid'] == pair['grid'], 'Replay cache binding or snapshot changed')
                observations.append(record)
                bindings.append((pair[side+'_path'], pair['grid'], record))
            checked = quality.assess_pair(pair, ReplayReader(bindings), inputs['events'], replay, preview=False)
            require_processed(checked)
            require(checked == row and anchor_contexts(pair, pair['anchor_event_ids'][0], inputs['events'], *observations)
                    == result['anchor_contexts'][index], 'Replayed numeric QA or anchor context differs')
            path = (run/row['qa_mask_file']).resolve()
            require(path.is_relative_to(run) and path.stat().st_size <= 2*2**20, 'QA mask path or size differs')
            with rasterio.open(path) as saved, rasterio.open(replay/row['qa_mask_file']) as expected:
                require(saved.profile == expected.profile and saved.descriptions == expected.descriptions
                        and np.array_equal(saved.read(), expected.read()), 'QA mask differs from observation replay')
    eligible = review_candidates(inputs['pairs'], result['results'], result['anchor_contexts'], [])
    require(eligible == result['review_candidate_pair_ids'] == meta['review_candidate_pair_ids'], 'Replayed candidate gate differs')
    proof = {'checks': {'exact_lineage': True, 'cumulative_targets': True, 'safe_cache_bindings': True,
              'numeric_qa_outcomes': True, 'qa_masks': True, 'anchor_contexts': True, 'candidate_gate': True},
              'pairs_replayed': PAIR_COUNT, 'cache_bindings_replayed': OBSERVATION_COUNT,
              'verified_screen_metadata_sha256': digest(run/'metadata.json'),
              'review_candidate_pair_ids': eligible, 'cloud_model_executions': 0, 'source_jp2_window_reads': 0}
    write(output/'verification.json', proof)
    return proof


def worker(args, output):
    inputs = build_inputs(args)
    details = {'prepare': prepare, 'screen': screen, 'verify': verify}[args.mode](args, output, inputs)
    unchanged_assets(inputs['assets'])
    require('torch' not in sys.modules, 'Forest model runtime imported during quality checks')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'execution_status': 'completed', 'schema_version': 24, 'mode': args.mode,
        'protocol_sha256': digest(PROTOCOL), 'pair_count': PAIR_COUNT, 'group_count': 2,
        'labels_approved': 0, 'all_27_channels_checked': False, 'validation_eligible': False,
        'forest_model_executions': 0, 'new_imagery_reviewed': 0,
        'artifact_sha256': {str(path.relative_to(output)): digest(path)
                            for path in sorted(output.rglob('*')) if path.is_file()},
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2),
        **details,
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'screen', 'verify'), required=True)
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lineage', type=Path)
    parser.add_argument('--screen', type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(protocol=PROTOCOL, max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    if (args.mode != 'prepare') != (args.lineage is not None) or (args.mode == 'verify') != (args.screen is not None):
        parser.error('screen and verify require --lineage. Only verify requires --screen.')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='fresh-pair-qa', gdal_cache_mib=32, warp_memory_mib=16)


if __name__ == '__main__':
    main()
