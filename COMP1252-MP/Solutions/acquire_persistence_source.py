#!/usr/bin/env python3
"""Acquire only the compatible V20 persistence sources using the V14 engine."""
import argparse
import fcntl
import os
from pathlib import Path
import re
import resource
import shutil
import sys

from acquire_earlier_validation_sources import (
    BASE, RESERVE, SOURCE_REL, open_transaction, safe_path, save,
)
from acquire_remaining_validation_sources import run_transaction
from inventory_earlier_validation_sources import ROOT, digest, read_json, require, rooted, write
from preflight_earlier_validation_metadata import asset_manifest, disk_budget
from preflight_persistence_source import PROTOCOL, checked_protocol

TX_REL = 'Datasets/ValidationPilot/v20/source_transaction'
PASS = 'remote_metadata_and_asset_listing_passed'


def verify_manifest(scene):
    year, month, day = map(int, scene['acquisition_date'].split('-'))
    stamp = f'{year}/{month}/{day}'
    compact_date = f'{year:04}{month:02}{day:02}'
    scene_id = scene['scene_id']
    prefix = scene['tile_prefix']
    product_key = scene['product_key']
    require(re.fullmatch(f'S2[AB]_MSIL1C_{compact_date}T[0-9]{{6}}_N0206_R064_T36UXB_'
                         r'[0-9]{8}T[0-9]{6}', scene_id)
            and re.fullmatch(f'tiles/36/U/XB/{stamp}/[0-9]+/', prefix)
            and product_key == f'products/{stamp}/{scene_id}/metadata.xml',
            'Invalid pinned persistence source identity')
    assets = scene['assets']
    require(not scene['issues'] and len(assets) == 58, 'Incomplete persistence manifest')
    pins = {asset['key']: {key: asset[key] for key in ('key', 'url', 'size', 'etag')}
            for asset in assets}
    require(len(pins) == 58 and product_key in pins
            and all(pin['url'] == BASE + key for key, pin in pins.items()),
            'Duplicate, missing or foreign asset mapping')
    product_pin = {key: pins[product_key][key] for key in ('size', 'etag')}
    expected, issues = asset_manifest(scene, scene['metadata'], pins, product_pin)
    require(not issues and assets == expected, 'Source-to-SAFE mapping differs from the frozen manifest')
    require(sum(asset['size'] for asset in assets) == scene['measured_payload_bytes'],
            'Manifest payload differs')


def preflight_inputs(protocol, protocol_sha, directory):
    directory = directory.resolve()
    require(directory.is_relative_to(ROOT) and directory.is_dir(), 'Invalid preflight directory')
    path = safe_path(directory, 'metadata.json')
    metadata = read_json(path.read_bytes())
    require(metadata['execution_status'] == 'completed' and metadata['protocol_sha256'] == protocol_sha,
            'Preflight did not finish under the frozen persistence protocol')
    hashes = metadata['artifact_sha256']
    require('preflight.json' in hashes and all(digest(safe_path(directory, name)) == checksum
                                               for name, checksum in hashes.items()),
            'Preflight artifact set missing or modified')
    preflight = read_json(safe_path(directory, 'preflight.json').read_bytes())
    require(preflight['status'] == 'metadata_preflight_passed_no_acquisition'
            and preflight['protocol_sha256'] == protocol_sha
            and [row['acquisition_date'] for row in preflight['results']] == protocol['candidate_dates']
            and all(row['status'] in (PASS, 'blocked_metadata_preflight') for row in preflight['results']),
            'Preflight results differ from the fixed roster')
    require(all(all(row[key] == value for key, value in expected.items())
                for row, expected in zip(preflight['results'], protocol['catalogue_roster'])),
            'Preflight scene identities differ from the fixed catalogue roster')
    scenes = [scene for scene in preflight['results'] if scene['status'] == PASS]
    require(scenes and len({scene['scene_id'] for scene in scenes}) == len(scenes),
            'No distinct compatible persistence sources')
    for scene in scenes:
        verify_manifest(scene)
    budget = preflight['disk_budget']
    measured = disk_budget(scenes, budget['measured_free_bytes'], budget['current_metadata_artifact_bytes'])
    require(all(budget.get(key) == value for key, value in measured.items())
            and budget['sufficient_for_conservative_future_acquisition'],
            'Preflight staging allowance or measured disk reserve differs')
    return scenes, budget['future_two_copy_staging_plus_temporary_bytes'], digest(path)


def check_prior_sources(protocol):
    """Preserve every earlier used source, including pre-V18 exclusions."""
    sources = protocol['source_files']
    history = read_json(rooted(sources['v18_protocol']['path']).read_bytes())['source_files']
    paths = set()
    for source in (history['v11_screen'], history['v15_screen'], sources['v18_screen']):
        path = rooted(source['path'])
        require(digest(path) == source['sha256'], 'Prior screen changed: ' + source['path'])
        screen = read_json(path.read_bytes())
        for name, receipt in screen['source_asset_receipts'].items():
            stat = rooted(name).stat()
            require([stat.st_size, stat.st_mtime_ns] == receipt['stat'], 'Prior source changed: ' + name)
            paths.add(name)
    return len(paths)


def worker(args, output):
    protocol = checked_protocol()
    protocol_sha = digest(PROTOCOL)
    scenes, allowance, preflight_sha = preflight_inputs(protocol, protocol_sha, args.preflight)
    if args.mode == 'download':
        require(0 <= args.scene_index < len(scenes), 'Scene index outside the compatible roster')
    prior_count = check_prior_sources(protocol)
    root, sentinel = ROOT / TX_REL, ROOT / SOURCE_REL
    free_before = shutil.disk_usage(sentinel).free
    require(free_before >= RESERVE + (allowance if args.mode == 'download' else 0),
            'Insufficient space for staging and disk reserve')
    if args.mode != 'download':
        require((root / 'journal.json').is_file(), 'Acquisition journal is missing')
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.with_name(root.name + '.lock')
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transaction = open_transaction(root, sentinel, protocol_sha, scenes)
        if 'preflight_metadata_sha256' not in transaction:
            require(transaction['state'] == 'acquiring' and not transaction['events'],
                    'Acquisition journal is missing its preflight pin')
            transaction['preflight_metadata_sha256'] = preflight_sha
            save(transaction)
        require(transaction['preflight_metadata_sha256'] == preflight_sha,
                'Preflight changed between acquisition steps')
        result = run_transaction(args, transaction, scenes)
        free_after = shutil.disk_usage(sentinel).free
        require(free_after >= RESERVE, 'Disk reserve breached')
        require(check_prior_sources(protocol) == prior_count and digest(PROTOCOL) == protocol_sha,
                'Pinned evidence changed during acquisition')
        shutil.copyfile(root / 'journal.json', output / 'journal_snapshot.json')
        result.update(disk_free_bytes_before=free_before, disk_free_bytes_after=free_after,
                      prior_source_stats_unchanged=prior_count)
        write(output / 'result.json', result)
    require(not {'torch', 'numpy', 'rasterio', 's2cloudless'} & sys.modules.keys(),
            'Unexpected pixel or model import during acquisition')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output / 'metadata.json', {
        'execution_status': 'completed', 'mode': args.mode, 'scene_index': args.scene_index,
        'protocol_sha256': protocol_sha, 'preflight_metadata_sha256': preflight_sha,
        'transaction_journal': TX_REL + '/journal.json',
        'artifact_sha256': {name: digest(output / name) for name in ('result.json', 'journal_snapshot.json')},
        'satellite_pixel_reads': 0, 'model_executions': 0, 'labels_approved': 0,
        'worker_high_water_rss_mib': round(peak / (2**20 if sys.platform == 'darwin' else 1024), 2),
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('download', 'publish', 'verify'), required=True)
    parser.add_argument('--scene-index', type=int, help='Zero-based index among passing metadata rows')
    parser.add_argument('--preflight', type=Path, default=PROTOCOL.parent / 'preflight')
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
        run_guarded(args, script_path=__file__, operation='persistence-source-acquisition',
                    gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
