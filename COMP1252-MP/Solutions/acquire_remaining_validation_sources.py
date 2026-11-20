#!/usr/bin/env python3
"""Acquire the three pinned V16 scenes using the V14 transaction engine."""
import argparse
import fcntl
import os
from pathlib import Path
import resource
import shutil
import sys

from acquire_earlier_validation_sources import (
    BASE, RESERVE, SOURCE_REL, acquire_scene, completion_result, event, open_transaction,
    publish, save,
)
from inventory_earlier_validation_sources import (
    ROOT, RESOURCES, digest, read_json, require, rooted, write,
)
from preflight_earlier_validation_metadata import asset_manifest

DATES = ('2018-07-11', '2018-07-06', '2018-07-01')
TX_REL = 'Datasets/ValidationPilot/v17/source_transaction'
PROTOCOL = ROOT/'Docs/Reviews/ValidationRemainingAcquisitionV17/protocol.json'
PASS = 'remote_metadata_and_asset_listing_passed'


def validate_roster(protocol, preflight):
    scenes = preflight['results']
    require(preflight['status'] == 'metadata_preflight_passed_no_acquisition'
            and tuple(scene['acquisition_date'] for scene in scenes) == DATES,
            'V16 preflight or fixed dates differ')
    roster = []
    for scene in scenes:
        scene_id = scene['scene_id']
        day = int(scene['acquisition_date'][-2:])
        prefix = f'tiles/36/U/XB/2018/7/{day}/0/'
        product_key = f'products/2018/7/{day}/{scene_id}/metadata.xml'
        assets = scene['assets']
        require(scene['status'] == PASS and not scene['issues'] and len(assets) == 58,
                'Incomplete V16 manifest')
        pins = {asset['key']: {key: asset[key] for key in ('key', 'url', 'size', 'etag')}
                for asset in assets}
        require(len(pins) == 58 and product_key in pins
                and all(pin['url'] == BASE+key for key, pin in pins.items()),
                'Duplicate, missing or foreign asset mapping')
        product_pin = {key: pins[product_key][key] for key in ('size', 'etag')}
        expected, issues = asset_manifest({'scene_id': scene_id, 'tile_prefix': prefix,
                                           'product_key': product_key}, scene['metadata'], pins, product_pin)
        require(not issues and assets == expected, 'Source-to-SAFE mapping differs from V16')
        payload = sum(asset['size'] for asset in assets)
        require(payload == scene['measured_payload_bytes'], 'Manifest payload differs')
        roster.append({'scene_id': scene_id, 'acquisition_date': scene['acquisition_date'],
                       'asset_count': len(assets), 'payload_bytes': payload})
    require(protocol['roster'] == roster and len({s['scene_id'] for s in scenes}) == 3
            and sum(row['payload_bytes'] for row in roster) == 1_664_246_549
            and protocol['stage_allowance_bytes'] ==
            preflight['disk_budget']['future_two_copy_staging_plus_temporary_bytes'],
            'Fixed roster, byte total or staging allowance differs')
    return scenes


def load_inputs():
    protocol = read_json(PROTOCOL.read_bytes())
    require(protocol['schema_version'] == 17 and protocol['resources'] == RESOURCES
            and protocol['transaction_root'] == TX_REL and protocol['sentinel_root'] == SOURCE_REL
            and protocol['disk_reserve_bytes'] == RESERVE, 'Acquisition protocol differs')
    receipts = dict(protocol['source_files'])
    implementation = protocol['implementation_sha256']
    require(str(Path(__file__).relative_to(ROOT)) in implementation, 'Pin the tested runner first')
    for name, checksum in {**protocol['frozen_implementation_sha256'], **implementation}.items():
        receipts['code/'+name] = {'path': name, 'sha256': checksum}
    for receipt in receipts.values():
        require(digest(rooted(receipt['path'])) == receipt['sha256'], 'Pinned input changed: '+receipt['path'])
    preflight = read_json(rooted(receipts['preflight']['path']).read_bytes())
    return protocol, validate_roster(protocol, preflight), receipts


def check_prior_sources(protocol):
    paths = set()
    for role in ('v11_screen', 'v15_screen'):
        screen = read_json(rooted(protocol['source_files'][role]['path']).read_bytes())
        for name, receipt in screen['source_asset_receipts'].items():
            stat = rooted(name).stat()
            require([stat.st_size, stat.st_mtime_ns] == receipt['stat'], 'Prior source changed: '+name)
            paths.add(name)
    return len(paths)


def run_transaction(args, transaction, scenes):
    if args.mode == 'verify':
        require(transaction['state'] == 'complete', 'Publication must be complete before verification')
        result = completion_result(transaction, scenes)
        require(result == read_json((Path(transaction['root'])/'completion.json').read_bytes()),
                'Persistent completion receipt differs from the published sources')
        return result

    require(transaction['state'] != 'complete', 'Source publication is already complete')
    try:
        if args.mode == 'download':
            return acquire_scene(transaction, scenes, args.scene_index)
        return publish(transaction, scenes)
    except Exception as error:
        event(transaction, 'worker_failed', mode=args.mode, scene_index=args.scene_index,
              error=f'{type(error).__name__}: {error}')
        save(transaction)
        raise


def worker(args, output):
    protocol, scenes, receipts = load_inputs()
    prior_count = check_prior_sources(protocol)
    root, sentinel = ROOT/TX_REL, ROOT/SOURCE_REL
    sha = digest(PROTOCOL)
    free_before = shutil.disk_usage(sentinel).free
    allowance = protocol['stage_allowance_bytes'] if args.mode == 'download' else 0
    require(free_before >= RESERVE+allowance, 'Insufficient space for staging and disk reserve')
    if args.mode != 'download':
        require((root/'journal.json').is_file(), 'Acquisition journal is missing')
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.with_name(root.name+'.lock')
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transaction = open_transaction(root, sentinel, sha, scenes)
        result = run_transaction(args, transaction, scenes)
        free_after = shutil.disk_usage(sentinel).free
        require(free_after >= RESERVE, 'Disk reserve breached')
        require(check_prior_sources(protocol) == prior_count, 'Prior source inventory changed')
        require(digest(PROTOCOL) == sha, 'Protocol changed during acquisition')
        journal = root/'journal.json'
        shutil.copyfile(journal, output/'journal_snapshot.json')
        result.update(disk_free_bytes_before=free_before, disk_free_bytes_after=free_after,
                      prior_source_stats_unchanged=prior_count)
        write(output/'result.json', result)
    require(not {'torch', 'numpy', 'rasterio', 's2cloudless'} & sys.modules.keys(),
            'Unexpected pixel or model import during acquisition')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'execution_status': 'completed', 'mode': args.mode, 'scene_index': args.scene_index,
        'protocol_sha256': sha, 'input_receipts': receipts,
        'transaction_journal': TX_REL+'/journal.json',
        'artifact_sha256': {name: digest(output/name) for name in ('result.json', 'journal_snapshot.json')},
        'satellite_pixel_reads': 0, 'model_executions': 0, 'labels_approved': 0,
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2),
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('download', 'publish', 'verify'), required=True)
    parser.add_argument('--scene-index', type=int, choices=range(len(DATES)))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    if (args.mode == 'download') != (args.scene_index is not None):
        parser.error('--scene-index is required only for download')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='remaining-source-acquisition',
                    gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
