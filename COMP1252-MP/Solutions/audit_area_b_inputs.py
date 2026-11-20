#!/usr/bin/env python3
"""Audit the frozen Area B feature tensor and prepare its control review."""

import argparse
import os
from pathlib import Path
import resource
import sys
import tempfile

from screen_repaired_validation import (
    GROUP, POLICY, RESOURCES, ROOT, count_band_requests, digest, load_safe_cache,
    object_hash, read, require, rooted, write,
)

REVIEW = ROOT/'Docs/Reviews/ValidationAreaBInputsV21'
PROTOCOL = REVIEW/'protocol.json'
PAIR = 'derived-remaining-e61a8a8e623222b69efb'
DATES = ('2018-07-06', '2018-07-31', '2018-08-10')
FEATURES = 'endpoint-features.npz'


def checked_inputs():
    from audit_validation_context import architecture_contract
    from check_persistence_observations import CLEAR, checked_publication, frozen_area
    from preflight_persistence_source import checked_protocol as checked_v20_protocol
    from prepare_validation_review import records, validate_validation_pair
    from validation_context_worker import read_json

    protocol = read(PROTOCOL)
    require(protocol['schema_version'] == 21 and protocol['resources'] == RESOURCES
            and protocol['pair_id'] == PAIR and protocol['dates'] == list(DATES),
            'The fixed Area B scope or resource limits differ')
    paths = {}
    for role, receipt in protocol['source_files'].items():
        path = rooted(receipt['path'])
        require(digest(path) == receipt['sha256'], 'Pinned Area B source changed: '+role)
        paths[role] = path
    for name, checksum in {**protocol['frozen_implementation_sha256'], **protocol['implementation_sha256']}.items():
        require(digest(rooted(name)) == checksum, 'Pinned Area B code changed: '+name)
    architecture = architecture_contract(ROOT/'Solutions/train_pilot_baseline.py')
    require(architecture == protocol['architecture'], 'Original 5 x 5 context contract differs')

    info, masks, july = frozen_area()
    screen, replay = (read(paths[key]) for key in ('v18_screen', 'v18_replay'))
    pair = next(pair for pair in screen['derived_pairs'] if pair['pair_id'] == PAIR)
    require(pair['group_id'] == info['group_id'] == GROUP and pair['grid'] == info['grid']
            and (pair['pre_date'], pair['post_date']) == DATES[:2]
            and info['pair_id'] == PAIR and screen['qa_policy_id'] == POLICY
            and replay['execution_status'] == 'completed' and all(replay['checks'].values())
            and not screen['processing_errors'], 'Derived July pair or QA binding differs')
    parent = next(row for row in records(paths['pairs']) if row['pair_id'] == pair['parent_pair_id'])
    require(object_hash(parent) == pair['parent_pair_sha256'] and parent['grid'] == pair['grid'],
            'Original validation parent or grid changed')
    plan = read(paths['training_plan'])
    require(digest(paths['training_manifest']) == plan['selection']['manifest_sha256']
            and digest(paths['training_metadata']) == plan['selection']['dataset_metadata_sha256'],
            'The 13 actual training crops differ from the frozen model plan')
    event = info['event_id']
    validate_validation_pair(pair, event, read_json(paths['catalogue']), read_json(paths['splits']),
                             list(records(paths['training_manifest'])))
    summary = {'checks': replay['checks'], 'processing_errors': screen['processing_errors'],
               'source_file_stats': screen['source_file_stats']}

    persisted, followup_qa = (read(paths[key]) for key in ('v20_reviews', 'v20_screen'))
    followup_replay = read(paths['v20_replay'])
    require(checked_v20_protocol()['group_id'] == GROUP, 'Persistence protocol group differs')
    require(persisted['status'] == 'visual_persistence_review_completed' and persisted['pair_id'] == PAIR
            and persisted['screen_metadata_sha256'] == digest(paths['v20_screen_meta'])
            and followup_replay['screen_metadata_sha256'] == digest(paths['v20_screen_meta'])
            and followup_replay['screen_artifact_sha256']['result.json'] == digest(paths['v20_screen'])
            and all(followup_replay['checks'].values()) and followup_qa['area'] == info,
            'Area B persistence or replay evidence differs')
    by_date, hashes = checked_publication(paths['v20_publication'].parent, pair['grid'], ['2018-08-05', DATES[2]])
    require(hashes == followup_qa['publication_sha256'], 'Follow-up source verification differs')
    clear = next(row for row in followup_qa['observations'] if row['date'] == DATES[2])
    receipt = clear['cache_receipt']
    snapshot = paths['v20_screen'].parent/clear['snapshot']
    require(clear['status'] == CLEAR and clear['target']['context_clear']
            and clear['scene_id'] == by_date[DATES[2]]['scene_id'] and digest(snapshot) == receipt['sha256'],
            'Verified Area B follow-up cache differs')
    august = load_safe_cache(receipt, clear['observation'])
    require(august['metadata']['grid'] == pair['grid'] and august['metadata']['scene_id'] == clear['scene_id'],
            'Follow-up cache scene or grid differs')
    return pair, summary, info, masks, [*july, august], receipt, paths


def source_stats(pair, summary, paths):
    combined = {name: values for name, values in summary['source_file_stats'].items()
                if name.startswith((pair['pre_path']+'/', pair['post_path']+'/'))}
    require(len(combined) == 116, 'Expected only the two verified 58-file July sources')
    asset_receipts = read(paths['v18_screen'])['source_asset_receipts']
    for name, values in combined.items():
        stat = rooted(name).stat()
        require([stat.st_size, stat.st_mtime_ns] == values == asset_receipts[name]['stat'],
                'The frozen July source fingerprint changed: '+name)
    return combined


def verify_features(path):
    import numpy as np
    from prepare_validation_manual import _npz_headers

    _npz_headers(path, {'features': (27, 64, 64), 'feature_valid': (64, 64)}, {'features'})
    with np.load(path, allow_pickle=False) as archive:
        return archive['features'], archive['feature_valid']


def build_result(output, inputs, features, validity, stats, *, verify=False):
    import numpy as np
    from curated_pilot_exports import CHANNELS
    from prepare_area_b_controls import prepare_controls
    from prepare_july_visual_review import array_hash
    from validation_context_worker import _context_result
    from verify_validation_context import feature_integrity

    pair, _, info, masks, observations, receipt, _ = inputs
    before, after = observations[:2]
    feature_integrity(features, validity, {'pre': before, 'post': after})
    contexts = {}
    for key, context_key in (('target', 'context'), ('interior', 'interior_context')):
        context, metrics = _context_result(masks[key], before, after, features, validity)
        require(np.array_equal(context, masks[context_key]), 'Area B feature context differs')
        metrics['nonfinite_by_channel'] = dict(zip(CHANNELS, (~np.isfinite(features[:, context])).sum(axis=1).tolist()))
        contexts[key] = metrics
    controls = prepare_controls(output, pair, masks['target'], masks['context'], features, validity,
                                observations, verify=verify)
    return {
        'schema_version': 21, 'protocol_sha256': digest(PROTOCOL), 'qa_policy_id': POLICY,
        'area': info, 'source_stats': stats, 'dates': list(DATES), 'august_cache_receipt': receipt,
        'endpoint_channels': list(CHANNELS), 'feature_values_sha256': array_hash(features),
        'feature_valid_sha256': array_hash(validity), 'finite_crop_pixels': int(validity.sum()),
        'contexts': contexts, 'target_endpoint_inputs_eligible': all(row['context_clear'] for row in contexts.values()),
        'control_review_packet': controls, 'labels_approved': 0, 'validation_eligible': False,
    }


def artifact_hashes(path):
    return {file.name: digest(file) for file in sorted(path.iterdir()) if file.is_file()}


def worker(args, output):
    import numpy as np
    import assess_pixel_quality as quality
    from prepare_validation_review import read_validation_features

    inputs = checked_inputs()
    pair, summary, _, _, _, _, paths = inputs
    stats = source_stats(pair, summary, paths)
    if args.mode == 'prepare':
        with count_band_requests(quality) as calls:
            features, validity, measured = read_validation_features(pair, summary)
        require(calls['requests'] == 14 and measured == stats, 'Frozen feature read scope differs')
        np.savez_compressed(output/FEATURES, features=features, feature_valid=validity)
        check, check_valid = verify_features(output/FEATURES)
        require(np.array_equal(features, check, equal_nan=True) and np.array_equal(validity, check_valid),
                'Feature archive round-trip differs')
        write(output/'audit.json', build_result(output, inputs, features, validity, stats))
        requests = calls['requests']
    else:
        packet = args.packet.resolve()
        require(packet.is_relative_to(ROOT), 'Review packet must be inside the project')
        metadata = read(packet/'metadata.json')
        require(metadata['execution_status'] == 'completed' and metadata['mode'] == 'prepare'
                and metadata['protocol_sha256'] == digest(PROTOCOL), 'Completed Area B audit required')
        artifacts = metadata['artifact_sha256']
        require(artifact_hashes(packet) == {**artifacts, 'metadata.json': digest(packet/'metadata.json')},
                'Area B packet archive changed')
        features, validity = verify_features(packet/FEATURES)
        with tempfile.TemporaryDirectory(dir=output) as directory:
            stage = Path(directory)
            actual = build_result(stage, inputs, features, validity, stats, verify=True)
            require(actual == read(packet/'audit.json'), 'Area B finite-context or control replay differs')
            require(artifact_hashes(stage) == {name: sha for name, sha in artifacts.items()
                    if name not in (FEATURES, 'audit.json')}, 'Area B comparison panels differ')
        write(output/'verification.json', {'audit_metadata_sha256': digest(packet/'metadata.json'),
              'audit_artifact_sha256': artifacts, 'feature_and_context_replay': 'passed',
              'independent_control_roster_and_view_replay': 'passed'})
        requests = 0
    require(source_stats(pair, summary, paths) == stats, 'Feature sources changed')
    require(not {'torch', 's2cloudless'} & sys.modules.keys(), 'Unexpected model import')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {'execution_status': 'completed', 'mode': args.mode,
          'protocol_sha256': digest(PROTOCOL), 'artifact_sha256': artifact_hashes(output),
          'feature_band_window_requests': requests, 'cloud_model_executions': 0, 'forest_model_executions': 0,
          'labels_approved': 0, 'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2)})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'verify'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    if (args.mode == 'verify') != (args.packet is not None):
        parser.error('Only verify mode uses --packet')
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='area-b-input-audit',
                    gdal_cache_mib=32, warp_memory_mib=0)


if __name__ == '__main__':
    main()
