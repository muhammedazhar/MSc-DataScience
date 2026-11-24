#!/usr/bin/env python3
"""Bind the four pinned 2019 comparisons, run frozen QA and replay its evidence."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import tempfile

from complete_validation_qa_inputs import expected_md5
from inventory_earlier_validation_sources import RESOURCES, canonical, sha
from preflight_remaining_validation_metadata import READY
from screen_earlier_validation import collect_roster, prevalidate_cache, require_lineage, temporal_summary
from screen_fresh_validation_sources import review_candidates, temporal_event_sets
from screen_repaired_validation import (
    POLICY, ROOT, anchor_contexts, contains, count_band_requests, digest, load_safe_cache,
    object_hash, read, require, require_processed, rooted, unchanged_assets, unique, write,
)


PROTOCOL = ROOT/'Docs/Reviews/Validation2019ScreenV27/protocol.json'
NEW_CACHE = ROOT/'Datasets/ValidationPilot/v27/cache'
SITE_DATES = {'march': ('2019-03-03', '2019-03-08'), 'may': ('2019-05-07', '2019-05-12')}
QA_EXCLUDED = 'excluded_by_automatic_quality_screen'
PAIR_COUNT = 4
OBSERVATION_COUNT = 8
SOURCE_ROLES = {'v25_protocol', 'v26_preflight', 'pairs', 'catalogue', 'splits', 'training_manifest',
                'quality', 'old_quality_summary', 'v24_screen'}


def protocol_sources(path):
    protocol = read(path)
    require(protocol['schema_version'] == 27 and protocol['resources'] == RESOURCES
            and SOURCE_ROLES <= protocol['source_files'].keys(), 'V27 scope or resource contract differs')
    source_files = {key: rooted(receipt['path']) for key, receipt in protocol['source_files'].items()}
    for key, source in source_files.items():
        require(digest(source) == protocol['source_files'][key]['sha256'], 'Frozen source changed: '+key)
    implementations = {**protocol['frozen_implementation_sha256'], **protocol['implementation_sha256']}
    require(str(Path(__file__).resolve().relative_to(ROOT)) in protocol['implementation_sha256'],
            'V27 screening implementation must be pinned before running')
    for name, checksum in implementations.items():
        require(digest(rooted(name)) == checksum, 'Frozen implementation changed: '+name)
    return source_files


def checked_artifacts(directory, filename):
    run = directory.resolve()
    require(run.is_relative_to(ROOT), 'Artifact directory outside the project')
    meta = read(run/'metadata.json')
    hashes = meta['artifact_sha256']
    require(filename in hashes, 'Missing bounded artifact: '+filename)
    for name, checksum in hashes.items():
        path = (run/name).resolve()
        require(path.is_relative_to(run) and digest(path) == checksum, 'Phase artifact changed: '+name)
    return run, meta


def validate_scene(meta, scene_id, day, site):
    require(meta['scene_id'] == scene_id and meta['acquisition_date'] == day
            and meta['metadata_usable'] and meta['cloud_model_usable'] and meta['shadow_geometry_usable']
            and not meta['issues'] and meta['processing_level'] == 'L1C' and meta['tile'] == '36UYA'
            and meta['baseline_code'] == 'N0207' and '_R064_T36UYA_' in scene_id
            and meta['crs'] == site['grid']['crs'] == 'EPSG:32636'
            and contains(meta['grid_10m']['bounds'], site['qa_context_bounds']),
            'Frozen SAFE prerequisites or full QA footprint changed: '+scene_id)


def checked_publication(directory, preflight, sites):
    from pixelqa_scene_metadata import qa_scene_metadata

    require(preflight['schema_version'] == 26 and preflight['candidate_count'] == preflight['ready_count'] == PAIR_COUNT
            and preflight['image_assets_downloaded'] == preflight['labels_approved'] == 0,
            'Expected the complete V26 native SAFE preflight')
    candidates = preflight['results']
    require([(row['site_id'], row['acquisition_date']) for row in candidates]
            == [(site, day) for site, dates in SITE_DATES.items() for day in dates]
            and len({row['scene_id'] for row in candidates}) == PAIR_COUNT,
            'V26 candidate identities or date order changed')
    records, assets, proofs = {}, {}, {}
    for row in candidates:
        day, scene_id = (row[key] for key in ('acquisition_date', 'scene_id'))
        site = sites[row['site_id']]
        require(all(row[key] == site[key] for key in ('event_id', 'group_id', 'pair_id', 'post_date', 'post_scene_id'))
                and row['status'] == READY and not row['issues']
                and row['metadata']['fixed_event_coverage'][site['event_id']] == {'core': True, 'qa_context': True},
                'Scene preflight does not cover its exact site: '+scene_id)
        run, meta = checked_artifacts(directory/day, 'download.json')
        result = read(run/'download.json')
        scene = Path('Datasets/Sentinel-2')/scene_id
        safe = scene/(scene_id+'.SAFE')
        require(meta['schema_version'] == 1 and meta['command'] == 'download' and meta['satellite_pixel_reads'] == 0
                and result['transaction_state'] in ('complete', 'existing_verified')
                and result['scene_id'] == scene_id and result['acquisition_date'] == day
                and Path(result['product_path']).resolve() == rooted(str(scene))
                and result['asset_count'] == len(row['assets']) == 58
                and result['payload_bytes'] == row['measured_payload_bytes'] == sum(a['size'] for a in row['assets'])
                and result['manifest_sha256'] == sha(canonical({'scene_id': scene_id, 'assets': row['assets']})),
                'Completed download differs from preflight: '+scene_id)
        record = qa_scene_metadata(rooted(str(safe)), ROOT)
        require(record['path'] == str(safe) and record == result['final_metadata'],
                'Published SAFE metadata differs: '+scene_id)
        validate_scene(record, scene_id, day, site)
        receipts = result['source_receipts']
        require(set(receipts) == {asset['target_relpath'] for asset in row['assets']}, 'Exact 58 asset receipts required')
        for asset in row['assets']:
            target = asset['target_relpath']
            receipt = receipts[target]
            name = str(scene/target)
            path = rooted(name)
            require(path.is_relative_to(rooted(str(safe))) and name not in assets
                    and receipt['bytes'] == asset['size'] and receipt['etag'] == asset['etag']
                    and receipt['source_url'] == asset['url'] and receipt['md5'] == expected_md5(asset)
                    and re.fullmatch(r'[0-9a-f]{64}', receipt['sha256']),
                    'Invalid provider-bound asset receipt: '+name)
            stat = path.stat()
            require(stat.st_size == asset['size'], 'Published asset size changed: '+name)
            assets[name] = {**receipt, 'stat': [stat.st_size, stat.st_mtime_ns]}
        records[day] = record
        proofs[day] = digest(run/'metadata.json')
    require(len(assets) == 232, 'Exactly 232 new SAFE asset receipts required')
    return candidates, records, assets, proofs


def frozen_posts(sites, original, old_quality, summary):
    from pixelqa_scene_metadata import qa_scene_metadata

    require(summary['config_id'] == POLICY and not summary['processing_errors'] and all(summary['checks'].values()),
            'Original QA publication is incomplete')
    posts, assets, history = {}, {}, []
    for site in sites.values():
        parent = original[site['pair_id']]
        require(all(parent[key] == site[key] for key in ('group_id', 'grid', 'pre_date', 'post_date',
                                                       'pre_scene_id', 'post_scene_id'))
                and parent['anchor_event_ids'] == [site['event_id']], 'Original pair geometry or identity differs')
        for day in site['prior_rejected_pre_dates']:
            matching = [row for row in original.values() if row['group_id'] == site['group_id']
                        and row['grid'] == site['grid'] and row['anchor_event_ids'] == [site['event_id']]
                        and row['pre_date'] == day and row['post_date'] == site['post_date']]
            require(len(matching) == 1, 'Known cloudy prior comparison is missing: '+day)
            prior = old_quality[matching[0]['pair_id']]
            require(prior['quality_status'] == QA_EXCLUDED and prior['config_id'] == POLICY
                    and not prior['training_approved'], 'Known cloudy automatic QA disposition differs')
            history.append({key: prior[key] for key in ('pair_id', 'group_id', 'pre_date', 'post_date',
                                                       'quality_status', 'quality_reasons')})
        after = old_quality[parent['pair_id']]['per_date']['post']
        observed = after['observation']
        post = qa_scene_metadata(rooted(parent['post_path']), ROOT)
        validate_scene(post, parent['post_scene_id'], parent['post_date'], site)
        require(after['crop']['unusable_pixels'] == after['target']['unusable_pixels'] == 0
                and observed['metadata_uid'] == post['metadata_uid'] and observed['scene_id'] == post['scene_id']
                and observed['grid'] == parent['grid'], 'Fixed clear post observation binding differs')
        stats = {name: {'stat': stat} for name, stat in summary['source_file_stats'].items()
                 if Path(name).is_relative_to(parent['post_path'])}
        require(len(stats) == 58 and not assets.keys() & stats.keys(), 'Fixed post prerequisite set changed')
        assets.update(stats)
        path = rooted(observed['cache_path'])
        require(path.stat().st_size <= 2*2**20, 'Fixed original post cache exceeds its bound')
        receipt = {'path': observed['cache_path'], 'bytes': path.stat().st_size,
                   'sha256': digest(path), 'cache_key': observed['cache_key']}
        posts[parent['pair_id']] = (post, observed, receipt)
    require(len(history) == 3, 'Expected all three prior cloudy 2019 comparison dispositions')
    unchanged_assets(assets)
    return posts, assets, history


def derive_pair(parent, site, scene, events, properties, known_ids):
    before, after = scene['acquisition_date'], parent['post_date']
    interval = (date.fromisoformat(after)-date.fromisoformat(before)).days
    require(parent['tile'] == '36UYA' and parent['split'] == 'validation'
            and not parent['blocking_reasons'] and not parent['conflicting_label_event_ids']
            and before in SITE_DATES[site['site_id']] and 5 <= interval <= 30
            and before < parent['pre_date'] and before < site['annotation_date'] <= after
            and contains(scene['grid_10m']['bounds'], site['qa_context_bounds']),
            'Derived pair would change frozen coverage or temporal provenance')
    pre_events, post_events = temporal_event_sets(parent, events, properties, before)
    sources = {'pre': {'scene_id': scene['scene_id'], 'path': scene['path']},
               'post': {'scene_id': parent['post_scene_id'], 'path': parent['post_path']}}
    lineage = {'parent_pair_id': parent['pair_id'], 'parent_pair_sha256': object_hash(parent),
               'pre_date': before, 'post_date': after, 'sources': sources, 'grid': parent['grid'],
               'post_label_ids': post_events, 'pre_label_ids': pre_events}
    identifier = 'derived-2019-'+object_hash(lineage)[:20]
    require(identifier not in known_ids, 'Derived identity would impersonate a frozen pair')
    pair = deepcopy(parent)
    pair.update(pair_id=identifier, pair_kind='derived_2019_validation_source_pair',
                parent_pair_id=parent['pair_id'], parent_pair_sha256=lineage['parent_pair_sha256'],
                parent_pre_date=parent['pre_date'], parent_post_date=after,
                parent_quality_status=QA_EXCLUDED, parent_status=parent['status'],
                parent_blocking_reasons=parent['blocking_reasons'], source_replacement=sources,
                pre_date=before, interval_days=interval, pre_scene_id=scene['scene_id'], pre_path=scene['path'],
                label_event_ids_by_pre_date=pre_events, label_event_ids_by_post_date=post_events,
                temporal_reference_rule='cumulative_post_AND_NOT_cumulative_pre_recomputed',
                training_approved=False, evaluation_approved=False)
    return pair


def build_inputs(args):
    from assess_pixel_quality import load_events
    from prepare_validation_review import validate_validation_pair
    from select_validation_cases import records

    sources = protocol_sources(args.protocol)
    reference = read(sources['v25_protocol'])
    sites = unique(reference['sites'], 'site_id')
    exposed = reference['prior_manual_exposed_group_ids']
    groups = sorted(site['group_id'] for site in sites.values())
    require(list(sites) == list(SITE_DATES) and len(set(exposed)) == 7
            and len(set(groups)) == 2 and not set(groups) & set(exposed), 'V25 site scope or prior exposure differs')
    v24 = read(sources['v24_screen'])
    require(not v24['processing_errors'] and not v24['review_candidate_pair_ids']
            and len(v24['results']) == 4 and all(row['quality_status'] == QA_EXCLUDED for row in v24['results'])
            and set(v24['prior_manual_exposed_group_ids']) == set(exposed), 'V24 automatic exclusions differ')
    candidates, fresh, assets, publications = checked_publication(args.publication, read(sources['v26_preflight']), sites)
    fresh_stats = {name: receipt['stat'] for name, receipt in assets.items()}
    original, ids = {}, set()
    fixed_posts = {site['group_id']: site['post_date'] for site in sites.values()}
    for pair in records(sources['pairs']):
        identifier = pair['pair_id']
        require(identifier not in ids, 'Duplicate frozen comparison identity: '+identifier)
        ids.add(identifier)
        if fixed_posts.get(pair['group_id']) == pair['post_date']:
            original[identifier] = pair
    old_quality = unique((row for row in records(sources['quality']) if row['pair_id'] in original), 'pair_id')
    summary = read(sources['old_quality_summary'])
    posts, original_assets, history = frozen_posts(sites, original, old_quality, summary)
    summary_stats, later_stats = summary['source_file_stats'], v24['source_file_stats']
    require(all(summary_stats[name] == later_stats[name] for name in summary_stats.keys() & later_stats.keys()),
            'Original and V24 recorded source stats disagree')
    original_stats = {**summary_stats, **later_stats}
    prior_assets = {name: {'stat': stat} for name, stat in original_stats.items()}
    require(not assets.keys() & prior_assets.keys(), 'New scenes must be distinct from all prior QA source files')
    unchanged_assets(prior_assets)
    assets.update(original_assets)
    events = load_events(sources['catalogue'].parent)
    catalogue, splits = (read(sources[key]) for key in ('catalogue', 'splits'))
    properties = {feature['id']: feature['properties'] for feature in catalogue['features']}
    training = list(records(sources['training_manifest']))
    pairs, geometry = [], []
    ids.update(row['pair_id'] for row in v24['derived_pairs'])
    for row in candidates:
        site = sites[row['site_id']]
        parent = original[site['pair_id']]
        pair = derive_pair(parent, site, fresh[row['acquisition_date']], events, properties, ids)
        validate_validation_pair(pair, site['event_id'], catalogue, splits, training)
        measured = temporal_summary(pair, parent, events)
        require(measured['pixels']['anchor_interior'] == site['eroded_support_pixels'],
                'Recomputed interior support differs from the pinned V25 crop')
        pairs.append(pair)
        geometry.append(measured)
        ids.add(pair['pair_id'])
    common = {'schema_version': 27, 'protocol_sha256': digest(args.protocol), 'qa_policy_id': POLICY,
              'derived_pairs': pairs, 'temporal_geometry': geometry, 'source_asset_receipts': assets,
              'publication_metadata_sha256_by_date': publications, 'prior_manual_exposed_group_ids': exposed,
              'historical_2019_exclusions': history, 'prior_v24_excluded_pair_ids': [row['pair_id'] for row in v24['results']],
              'planned_geographic_groups': groups, 'labels_approved': 0, 'new_imagery_reviewed': 0,
              'all_27_channels_checked': False, 'validation_eligible': False, 'forest_model_executions': 0}
    return {'pairs': pairs, 'geometry': geometry, 'assets': assets, 'prior_assets': prior_assets,
            'events': events, 'posts': posts, 'fresh_stats': fresh_stats, 'common': common}


def prepare(args, output, inputs):
    write(output/'lineage.json', inputs['common'])
    (output/'derived_pairs.jsonl').write_text(''.join(json.dumps(pair, sort_keys=True)+'\n' for pair in inputs['pairs']))
    return {'cloud_model_executions': 0, 'source_jp2_window_reads': 0,
            'checks': {'exact_lineage': True, 'cumulative_targets': True, 'split_isolation': True}}

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
    require(reader.hits+reader.misses <= 4 and counts['requests'] <= 44, 'V27 source QA resource bound exceeded')
    if not errors:
        stats = {name: stat for scene in reader.scenes.values() for name, stat in scene['source_file_stats'].items()}
        require(reader.hits+reader.misses == 4 and len(old_cache) == 2 and distinct == 6
                and counts['requests'] == reader.misses*11 and stats == inputs['fresh_stats'],
                'V27 new and frozen endpoint bindings differ')
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
    run, meta = checked_artifacts(args.screen, 'screen.json')
    require(meta['execution_status'] == 'completed' and meta['mode'] == 'screen'
            and meta['protocol_sha256'] == digest(args.protocol), 'Completed screen phase differs')
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
    proof = {'protocol_sha256': digest(args.protocol), 'checks': {'exact_lineage': True, 'cumulative_targets': True, 'safe_cache_bindings': True,
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
    unchanged_assets(inputs['prior_assets'])
    require('torch' not in sys.modules, 'Forest model runtime imported during quality checks')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'execution_status': 'completed', 'schema_version': 27, 'mode': args.mode,
        'protocol_sha256': digest(args.protocol), 'pair_count': PAIR_COUNT, 'group_count': 2,
        'labels_approved': 0, 'all_27_channels_checked': False, 'validation_eligible': False,
        'forest_model_executions': 0, 'new_imagery_reviewed': 0,
        'prior_source_file_stat_count': len(inputs['prior_assets']),
        'artifact_sha256': {str(path.relative_to(output)): digest(path)
                            for path in sorted(output.rglob('*')) if path.is_file()},
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2),
        **details,
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'screen', 'verify'), required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lineage', type=Path)
    parser.add_argument('--screen', type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=RESOURCES['worker_rss_mib'],
                        min_available_gib=RESOURCES['available_memory_reserve_gib'],
                        max_seconds=RESOURCES['max_seconds_per_worker'])
    args = parser.parse_args()
    if args.mode == 'prepare' and (args.lineage or args.screen):
        parser.error('prepare takes neither --lineage nor --screen')
    if args.mode != 'prepare' and (args.lineage is None or (args.mode == 'verify') != bool(args.screen)):
        parser.error('screen and verify require --lineage. Only verify requires --screen.')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(RESOURCES['nice_increment'])
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='2019-pair-qa', gdal_cache_mib=32, warp_memory_mib=16)


if __name__ == '__main__':
    main()
