#!/usr/bin/env python3
"""Publish the fixed 36UXA sources using the existing verified transaction engine."""

import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import resource
import shutil
import sys

import acquire_earlier_validation_sources as engine
from acquire_remaining_validation_sources import run_transaction
from inventory_earlier_validation_sources import ROOT, digest, read_json, require, rooted, write
from preflight_earlier_validation_metadata import BASE, asset_manifest, disk_budget
from preflight_fresh_validation_sources import PROTOCOL, checked_protocol

TX = ROOT/'Datasets/ValidationPilot/v24/source_transaction'
READY = 'remote_metadata_and_asset_listing_passed'


def load_preflight(directory, protocol, protocol_sha):
    directory = directory.resolve()
    require(directory.is_relative_to(ROOT), 'Preflight directory must stay inside the project')
    metadata_path = engine.safe_path(directory, 'metadata.json')
    metadata = read_json(metadata_path.read_bytes())
    hashes = metadata['artifact_sha256']
    require(metadata['execution_status'] == 'completed' and metadata['protocol_sha256'] == protocol_sha
            and 'preflight.json' in hashes and all(digest(engine.safe_path(directory, name)) == checksum
                                                   for name, checksum in hashes.items()), 'Preflight artifacts differ')
    result = read_json((directory/'preflight.json').read_bytes())
    require(result['protocol_sha256'] == protocol_sha
            and result['status'] == 'metadata_preflight_passed_no_acquisition', 'Compatible preflight required')
    scenes = result['results']
    require(len(scenes) == len(protocol['catalogue_roster']) == 2, 'Fixed two-scene roster differs')
    for pinned, scene in zip(protocol['catalogue_roster'], scenes):
        day = int(pinned['acquisition_date'][-2:])
        scene_id = pinned['scene_id']
        prefix = f'tiles/36/U/XA/2018/7/{day}/0/'
        product_key = f'products/2018/7/{day}/{scene_id}/metadata.xml'
        require(all(scene[key] == value for key, value in pinned.items()) and scene['status'] == READY
                and not scene['issues'] and scene['tile_prefix'] == prefix and scene['product_key'] == product_key,
                'Pinned 36UXA identity or manifest differs')
        assets = scene['assets']
        require(len(assets) == 58 and len({asset['key'] for asset in assets}) == 58,
                'Expected 58 unique assets before acquisition')
        listed = {asset['key']: {key: asset[key] for key in ('key', 'url', 'size', 'etag')} for asset in assets}
        require(all(item['url'] == BASE+key for key, item in listed.items()), 'Asset outside the pinned L1C bucket')
        product_pin = {key: listed[product_key][key] for key in ('size', 'etag')}
        expected, issues = asset_manifest(scene, scene['metadata'], listed, product_pin)
        require(not issues and expected == assets and sum(a['size'] for a in assets) == scene['measured_payload_bytes'],
                'Asset roles, destinations or payload differ')
    budget = result['disk_budget']
    require(disk_budget(scenes, budget['measured_free_bytes'], budget['current_metadata_artifact_bytes']) == budget
            and budget['sufficient_for_conservative_future_acquisition'], 'Preflight disk allowance differs')
    return scenes, budget['future_two_copy_staging_plus_temporary_bytes'], digest(metadata_path)


def validate_uxa(product, scene, receipts, qa_reader=None):
    from pixelqa_scene_metadata import qa_scene_metadata

    engine.check_tree(product, scene)
    require(set(receipts) == {asset['target_relpath'] for asset in scene['assets']}, 'A pinned asset is missing')
    for asset in scene['assets']:
        name = asset['target_relpath']
        path = engine.safe_path(product, name)
        engine.verify_asset(path, asset, receipts[name]['sha256'])
        stat = path.stat()
        receipts[name]['stat'] = [stat.st_size, stat.st_mtime_ns]
    record = (qa_reader or (lambda path: qa_scene_metadata(path, ROOT)))(product)
    metadata = scene['metadata']
    require(record['scene_id'] == scene['scene_id'] and record['acquisition_date'] == scene['acquisition_date']
            and record['tile'] == '36UXA' and record['processing_level'] == 'L1C'
            and record['processing_baseline'] == '02.06' and record['quantification_value'] == 10000
            and not record['nonzero_offset'] and record['radiometric_offsets'] == metadata['radio_add_offsets']
            and all(record[key] == metadata[key] for key in ('crs', 'grid_10m', 'solar_zenith_deg', 'solar_azimuth_deg'))
            and all(record[key] for key in ('metadata_usable', 'cloud_model_usable', 'shadow_geometry_usable'))
            and not record['issues'], 'Published 36UXA source or QA metadata differs from preflight')
    return record


@contextmanager
def use_uxa_validator():
    """Replace only the UXB-specific validator inside this serial worker."""
    original = engine.validate_local
    engine.validate_local = validate_uxa
    try:
        yield
    finally:
        engine.validate_local = original


def preserve_source_stats(protocol):
    summary = read_json(rooted(protocol['source_files']['old_quality_summary']['path']).read_bytes())
    stats = summary['source_file_stats']
    for name, expected in stats.items():
        stat = rooted(name).stat()
        require([stat.st_size, stat.st_mtime_ns] == expected, 'Original source changed: '+name)
    return len(stats)


def worker(args, output):
    protocol = checked_protocol()
    protocol_sha = digest(PROTOCOL)
    scenes, allowance, preflight_sha = load_preflight(args.preflight, protocol, protocol_sha)
    preserved = preserve_source_stats(protocol)
    sentinel = ROOT/engine.SOURCE_REL
    free_before = shutil.disk_usage(sentinel).free
    require(free_before >= engine.RESERVE+(allowance if args.mode == 'download' else 0),
            'Insufficient disk for staging and reserve')
    TX.parent.mkdir(parents=True, exist_ok=True)
    require(args.mode == 'download' or (TX/'journal.json').is_file(), 'Acquisition journal is missing')
    lock = os.open(TX.with_name(TX.name+'.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock, 'a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transaction = engine.open_transaction(TX, sentinel, protocol_sha, scenes)
        if 'preflight_metadata_sha256' not in transaction:
            require(transaction['state'] == 'acquiring' and not transaction['events'], 'Missing preflight journal pin')
            transaction['preflight_metadata_sha256'] = preflight_sha
            engine.save(transaction)
        require(transaction['preflight_metadata_sha256'] == preflight_sha, 'Preflight changed while staging')
        with use_uxa_validator():
            result = run_transaction(args, transaction, scenes)
        free_after = shutil.disk_usage(sentinel).free
        require(free_after >= engine.RESERVE and preserve_source_stats(protocol) == preserved
                and digest(PROTOCOL) == protocol_sha, 'Disk reserve or original inputs changed')
        shutil.copyfile(TX/'journal.json', output/'journal_snapshot.json')
        result.update(disk_free_bytes_before=free_before, disk_free_bytes_after=free_after,
                      prior_source_stats_unchanged=preserved)
        write(output/'result.json', result)
    require(not {'torch', 'numpy', 'rasterio', 's2cloudless'} & sys.modules.keys(),
            'Source verification imported a pixel or model library')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'execution_status': 'completed', 'mode': args.mode, 'scene_index': args.scene_index,
        'protocol_sha256': protocol_sha, 'preflight_metadata_sha256': preflight_sha,
        'transaction_journal': str((TX/'journal.json').relative_to(ROOT)),
        'artifact_sha256': {name: digest(output/name) for name in ('result.json', 'journal_snapshot.json')},
        'satellite_pixel_reads': 0, 'model_executions': 0, 'labels_approved': 0,
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2),
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('download', 'publish', 'verify'), required=True)
    parser.add_argument('--scene-index', type=int, choices=(0, 1))
    parser.add_argument('--preflight', type=Path, default=PROTOCOL.parent/'preflight')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    if (args.mode == 'download') != (args.scene_index is not None):
        parser.error('Scene index is required only for download')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='fresh-source-acquisition',
                    gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
