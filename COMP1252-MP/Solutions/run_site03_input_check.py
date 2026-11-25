#!/usr/bin/env python3
"""Audit the 27 frozen input channels for site-03 before any class review."""

import argparse
import os
from pathlib import Path
import resource
import sys

from inventory_earlier_validation_sources import RESOURCES, ROOT, digest, require, write
from preflight_fresh_validation_sources import checked_artifacts, evidence_hashes, load
from run_site03_quality import DATES, PROTOCOL, inputs, prior_assets, read_publications, require_phase


V30 = PROTOCOL.parent
V30_INTEGRITY_SHA256 = 'df49d8214a05601fcc74582595e7135e0a18ad70db07a4c8225c227d25325d2e'
PAIR_ID = 'derived-site03-891e388f6c6aaee1a9ea'
SCHEMA_VERSION = 31
TENSORS = {
    'control_baseline': 'control-baseline-features.npz',
    'target_endpoints': 'target-endpoint-features.npz',
}
BAND_WINDOWS_PER_PAIR = 14


def checked_inputs():
    """Reconcile the V30 receipts and exact geometry before extracting any bands."""
    from audit_validation_context import architecture_contract
    from plan_site03_qa import prepare_geometry, site03_masks
    from screen_site03_quality import archived_observation, bound_inputs

    final_path = V30/'final_integrity.json'
    require(digest(final_path) == V30_INTEGRITY_SHA256, 'Frozen V30 integrity record changed')
    final = load(final_path)
    require(final['execution_status'] == 'completed' and final['guard_limits_verified'] is True,
            'Completed V30 final integrity required')
    checked_artifacts(V30, final['git_index_sha256'])
    protocol, paths = inputs(PROTOCOL)
    checksum = digest(PROTOCOL)
    require(checksum == final['protocol_sha256'], 'V30 protocol binding differs')
    lineage = require_phase(V30/'lineage', 'lineage.json', 'prepare', checksum)
    screen = require_phase(V30/'screen', 'screen.json', 'screen', checksum)
    replay = require_phase(V30/'verification', 'verification.json', 'verify', checksum)
    require(replay['verified_screen_metadata_sha256'] == digest(V30/'screen/metadata.json')
            and len(replay['checks']) == 8 and all(value is True for value in replay['checks'].values())
            and replay['review_candidate_pair_ids'] == screen['review_candidate_pair_ids'] == [PAIR_ID]
            and screen['processing_errors'] == []
            and replay['control_qa_status'] == screen['control_qa_status'] == 'clear_on_all_three_dates'
            and all(screen['control_qa_by_date'][day]['qa_status'] == 'clear' for day in DATES),
            'V30 replay, review shortlist or three-date control QA differs')

    site = load(paths['v29_protocol'])['site']
    scenes, receipts, publication_sha = read_publications(V30/'publication', load(paths['acquisition_plan']), site)
    geometry = prepare_geometry(protocol, paths, scenes)
    bound = {**geometry, 'site': site, 'scenes': scenes, 'source_asset_receipts': receipts,
             'protocol_sha256': checksum}
    require(all(bound[key] == value for key, value in lineage.items()
                if key != 'publication_metadata_sha256_by_date')
            and publication_sha == lineage['publication_metadata_sha256_by_date'],
            'Rebuilt site-03 geometry or published receipts differ from V30')
    common, bindings = bound_inputs(bound)
    require(all(screen[key] == value for key, value in common.items())
            and set(screen['observations']) == set(DATES), 'V30 screen source or geometry binding differs')
    observations = {}
    for day in DATES:
        record = screen['observations'][day]
        binding = bindings[day]
        require(record['status'] == 'processed' and record['scene_path'] == binding['scene_path']
                and record['scene_id'] == binding['scene_id'], 'V30 observation roster changed: '+day)
        observations[day] = archived_observation(V30/'screen', record['cache_receipt'],
                                                 record['observation'], binding)
    older = prior_assets(paths)
    require(len(receipts) == 174 and len(older) == 2494 and not receipts.keys() & older.keys(),
            'Site-03 and prior source inventory differ')
    pair = geometry['pair']
    control_pair = {
        'pair_id': 'site-03-control-baseline-2018-05-10-2018-05-30',
        **{key: pair[key] for key in ('group_id', 'grid', 'split')},
    }
    for key in ('date', 'path', 'scene_id'):
        control_pair['pre_'+key] = pair['control_'+key]
        control_pair['post_'+key] = pair['pre_'+key]
    architecture = architecture_contract(ROOT/'Solutions/train_pilot_baseline.py')
    require(architecture['context_radius_pixels'] == lineage['control']['context_radius_pixels'] == 2,
            'Frozen model receptive field differs from the V30 geometry')
    return {
        'pairs': {'control_baseline': control_pair, 'target_endpoints': pair},
        'masks': site03_masks(pair, geometry['events'], geometry['control']),
        'lineage': lineage, 'architecture': architecture, 'screen': screen, 'replay': replay,
        'observations': observations, 'receipts': receipts, 'prior_assets': older,
    }


def read_tensor(path):
    import numpy as np
    from prepare_validation_manual import _npz_headers

    _npz_headers(path, {'features': (27, 64, 64), 'feature_valid': (64, 64)}, ('features',))
    with np.load(path, allow_pickle=False) as values:
        return values['features'], values['feature_valid']


def audit_result(bound, tensors):
    import numpy as np
    from curated_pilot_exports import CHANNELS
    from prepare_july_visual_review import array_hash
    from verify_validation_context import feature_integrity
    from validation_context_worker import _context_result

    lineage, masks, observations = (bound[key] for key in ('lineage', 'masks', 'observations'))
    scope = {
        'control_baseline': {'control': ('control', 'control_context')},
        'target_endpoints': {'target': ('anchor_target', 'anchor_context'),
                             'eroded_interior': ('anchor_interior', 'interior_context'),
                             'control': ('control', 'control_context')},
    }
    outcomes = {}
    for name, (features, valid) in tensors.items():
        pair = bound['pairs'][name]
        pre_date, post_date = pair['pre_date'], pair['post_date']
        pre, post = observations[pre_date], observations[post_date]
        feature_integrity(features, valid, {'pre': pre, 'post': post})
        contexts = {}
        for key, (roi_key, context_key) in scope[name].items():
            context, metrics = _context_result(masks[roi_key], pre, post, features, valid)
            require(np.array_equal(context, masks[context_key])
                    and array_hash(context) == lineage['geometry']['mask_sha256'][context_key]
                    and metrics['context_pixels'] == lineage['geometry']['pixels'][context_key],
                    'Feature audit context differs from frozen V30 geometry: '+key)
            metrics['nonfinite_by_channel'] = dict(zip(CHANNELS, (~np.isfinite(features[:, context])).sum(axis=1).tolist()))
            contexts[key] = metrics
        outcomes[name] = {
            'pre_date': pre_date, 'post_date': post_date,
            'pre_scene_id': pair['pre_scene_id'], 'post_scene_id': pair['post_scene_id'],
            'feature_values_sha256': array_hash(features), 'feature_valid_sha256': array_hash(valid),
            'finite_crop_pixels': int(valid.sum()), 'contexts': contexts,
        }
    baseline, baseline_valid = tensors['control_baseline']
    endpoints, endpoints_valid = tensors['target_endpoints']
    common = baseline_valid & endpoints_valid
    require(np.array_equal(baseline[9:18, common], endpoints[:9, common]),
            'Shared 30 May observation differs between the two feature tensors')
    positive = outcomes['target_endpoints']['contexts']
    return {
        'schema_version': SCHEMA_VERSION, 'pair_id': PAIR_ID, 'grid': lineage['pair']['grid'],
        'control': lineage['control'], 'architecture': bound['architecture'], 'channels': list(CHANNELS),
        'pairs': outcomes, 'shared_observation_valid_pixels_compared': int(common.sum()),
        'all_27_channels_checked': True,
        'target_inputs_complete': positive['target']['context_clear'] and positive['eroded_interior']['context_clear'],
        'control_inputs_complete': all(row['contexts']['control']['context_clear'] for row in outcomes.values()),
        'control_class_status': 'unknown', 'labels_approved': 0, 'validation_eligible': False,
        'v30_final_integrity_sha256': V30_INTEGRITY_SHA256,
    }


def worker(args, output):
    import numpy as np
    import assess_pixel_quality as quality
    from prepare_validation_review import read_validation_features
    from screen_repaired_validation import count_band_requests, unchanged_assets

    bound = checked_inputs()
    checksum = digest(__file__)
    receipts = bound['receipts']
    stats = {name: row['stat'] for name, row in receipts.items()}
    tensors = {}
    if args.mode == 'prepare':
        summary = {'checks': bound['replay']['checks'], 'processing_errors': bound['screen']['processing_errors'],
                   'source_file_stats': stats}
        with count_band_requests(quality) as count:
            for role, pair in bound['pairs'].items():
                previous = count['requests']
                features, valid, measured = read_validation_features(pair, summary)
                expected = {name: stat for name, stat in stats.items()
                            if name.startswith((pair['pre_path']+'/', pair['post_path']+'/'))}
                require(count['requests']-previous == BAND_WINDOWS_PER_PAIR
                        and measured == expected and len(expected) == 116,
                        'Feature source inventory or bounded window count differs')
                target = output/TENSORS[role]
                np.savez_compressed(target, features=features, feature_valid=valid)
                saved, saved_valid = read_tensor(target)
                require(np.array_equal(features, saved, equal_nan=True) and np.array_equal(valid, saved_valid),
                        'Feature tensor archive round-trip differs')
                tensors[role] = features, valid
        requests = count['requests']
        write(output/'audit.json', audit_result(bound, tensors))
    else:
        packet = args.packet.resolve()
        require(packet.is_relative_to(ROOT) and packet.is_dir(), 'Feature packet must be inside the project')
        metadata_path = packet/'metadata.json'
        metadata = load(metadata_path)
        artifacts = metadata['artifact_sha256']
        require(metadata['mode'] == 'prepare' and metadata['schema_version'] == SCHEMA_VERSION
                and metadata['execution_status'] == 'completed' and metadata['implementation_sha256'] == checksum
                and metadata['v30_final_integrity_sha256'] == V30_INTEGRITY_SHA256
                and artifacts.keys() == {*TENSORS.values(), 'audit.json'}
                and metadata['feature_band_window_requests'] == len(TENSORS)*BAND_WINDOWS_PER_PAIR,
                'Completed site-03 feature audit with the same worker is required')
        checked_artifacts(packet, artifacts)
        for role, filename in TENSORS.items():
            tensors[role] = read_tensor(packet/filename)
        result = audit_result(bound, tensors)
        require(result == load(packet/'audit.json'), 'Recomputed site-03 input completeness differs')
        checked_artifacts(packet, artifacts)
        requests = 0
        write(output/'verification.json', {
            'execution_status': 'completed', 'audit_metadata_sha256': digest(metadata_path),
            'verified_artifact_sha256': artifacts, 'tensors_replayed': len(tensors),
            'feature_channels_and_values': 'passed', 'frozen_roi_and_context_completeness': 'passed',
            'target_inputs_complete': result['target_inputs_complete'],
            'control_inputs_complete': result['control_inputs_complete'],
            'feature_band_window_requests': requests,
        })
    unchanged_assets(receipts)
    unchanged_assets(bound['prior_assets'])
    require(digest(__file__) == checksum and not {'torch', 's2cloudless'} & sys.modules.keys(),
            'Feature worker changed or a model runtime was imported')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'schema_version': SCHEMA_VERSION, 'execution_status': 'completed', 'mode': args.mode,
        'implementation_sha256': checksum, 'v30_final_integrity_sha256': V30_INTEGRITY_SHA256,
        'artifact_sha256': evidence_hashes(output), 'source_file_stat_count': len(receipts),
        'prior_source_file_stat_count': len(bound['prior_assets']),
        'feature_band_window_requests': requests, 'cloud_model_executions': 0, 'forest_model_executions': 0,
        'labels_approved': 0, 'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2),
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'verify'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=RESOURCES['worker_rss_mib'],
                        min_available_gib=RESOURCES['available_memory_reserve_gib'],
                        max_seconds=RESOURCES['max_seconds_per_worker'])
    args = parser.parse_args()
    if (args.mode == 'verify') != (args.packet is not None):
        parser.error('Only verify mode uses --packet')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output is not None:
        os.nice(RESOURCES['nice_increment'])
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='site03-input-check', gdal_cache_mib=32, warp_memory_mib=16)


if __name__ == '__main__':
    main()
