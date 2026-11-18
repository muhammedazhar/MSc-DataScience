#!/usr/bin/env python3
"""Stream two fixed sources into durable staging; publish new directories only."""
import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import Request, build_opener

from complete_validation_qa_inputs import (identity, matches_identity,
    sync_directory, verify_asset, expected_md5)
from inventory_earlier_validation_sources import (ROOT, RESOURCES, NoRedirect, digest,
                                                read_json, require, rooted, write)

TX_REL = 'Datasets/ValidationPilot/v14/source_transaction'
SOURCE_REL = 'Datasets/Sentinel-2'
BASE = 'https://sentinel-s2-l1c.s3.amazonaws.com/'
DATES = ('2018-07-21', '2018-07-16')
RESERVE = 2*2**30


def exclusive_move(source, destination):
    """Native no-replace rename, including an existing empty destination directory."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin':
        function = libc.renamex_np
        function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        args = (os.fsencode(source), os.fsencode(destination), 4)  # macOS SDK RENAME_EXCL
    elif sys.platform == 'linux' and hasattr(libc, 'renameat2'):
        function = libc.renameat2
        function.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        args = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    else:
        raise RuntimeError('Native exclusive rename is unavailable; refusing publication')
    function.restype = ctypes.c_int
    if function(*args) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))
    sync_directory(Path(source).parent); sync_directory(Path(destination).parent)


def safe_path(parent, name):
    parent = Path(parent)
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts and relative.parts,
            'Unsafe transaction-relative path')
    current = parent
    require(not current.is_symlink(), 'Symlink transaction root')
    for part in relative.parts:
        current = current/part
        require(not current.is_symlink(), 'Symlink in transaction path')
    require(current.resolve().is_relative_to(parent.resolve()), 'Transaction path escapes owner')
    return current


def inputs(protocol):
    require(protocol['schema_version'] == 14 and protocol['resources'] == RESOURCES, 'V14/resource scope changed')
    rules = protocol['rules']
    fixed = {'total_assets': 116, 'total_payload_bytes': 1050527541, 'transaction_root': TX_REL,
             'sentinel_root': SOURCE_REL, 'max_download_attempts_per_asset_per_worker': 1,
             'request_timeout_seconds': 30, 'stream_chunk_bytes': 1048576,
             'minimum_disk_reserve_bytes': RESERVE, 'one_guarded_worker_per_scene': True,
             'no_paid_access_or_redirects': True}
    require(all(rules.get(k) == v for k, v in fixed.items()), 'Fixed acquisition rule changed')
    receipts = dict(protocol['source_files'])
    for name, checksum in {**protocol['frozen_implementation_sha256'], **protocol.get('implementation_sha256', {})}.items():
        receipts['code/'+name] = {'path': name, 'sha256': checksum}
    require('Solutions/acquire_earlier_validation_sources.py' in protocol.get('implementation_sha256', {}),
            'Freeze tested acquisition implementation before work')
    for receipt in receipts.values():
        require(digest(rooted(receipt['path'])) == receipt['sha256'], 'Frozen input changed: '+receipt['path'])
    preflight = read_json(rooted(receipts['preflight']['path']).read_bytes())
    require(preflight['status'] == 'metadata_preflight_passed_no_acquisition'
            and tuple(r['acquisition_date'] for r in preflight['results']) == DATES
            and rules['initial_conservative_disk_allowance_bytes'] == preflight['disk_budget']['future_two_copy_staging_plus_temporary_bytes'],
            'V13 preflight/roster changed')
    scenes = preflight['results']
    expected = [{'scene_id': s['scene_id'], 'acquisition_date': s['acquisition_date'],
                 'asset_count': len(s['assets']), 'payload_bytes': s['measured_payload_bytes']} for s in scenes]
    require(protocol['roster'] == expected and sum(s['payload_bytes'] for s in expected) == rules['total_payload_bytes'],
            'Acquisition roster differs from V13 pins')
    for scene in scenes:
        require(not scene['issues'] and len(scene['assets']) == 58, 'Incomplete V13 source manifest')
        targets = []
        for asset in scene['assets']:
            expected_md5(asset)
            require(type(asset['size']) is int and asset['size'] > 0
                    and asset['url'] == BASE+asset['key'], 'Unsupported source pin/URL')
            path = Path(asset['target_relpath'])
            require(not path.is_absolute() and '..' not in path.parts
                    and path.parts[0] == scene['scene_id']+'.SAFE', 'Invalid source target')
            targets.append(str(path))
        require(len(set(targets)) == 58, 'Duplicate source target')
    return scenes, receipts


def preserved_sources(protocol):
    screen = read_json(rooted(protocol['source_files']['v11_screen']['path']).read_bytes())
    for name, receipt in screen['source_asset_receipts'].items():
        path = rooted(name); stat = path.stat()
        require([stat.st_size, stat.st_mtime_ns] == receipt['stat'], 'Prior source changed: '+name)
    return len(screen['source_asset_receipts'])


def atomic_json(path, value):
    """Never open a predictable existing temporary file or follow its symlink."""
    path = Path(path)
    require(not path.is_symlink(), 'Symlink journal/completion destination')
    descriptor, name = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
    temporary = Path(name); stat = os.fstat(descriptor); owned = [stat.st_dev, stat.st_ino]
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        require(not temporary.is_symlink() and identity(temporary) == owned and not path.is_symlink(),
                'Atomic journal file ownership changed')
        os.replace(temporary, path); sync_directory(path.parent)
    finally:
        if temporary.exists():
            require(not temporary.is_symlink() and identity(temporary) == owned, 'Unowned journal temporary retained')
            temporary.unlink()


def save(transaction):
    atomic_json(Path(transaction['root'])/'journal.json', transaction)


def event(transaction, kind, **details):
    transaction.setdefault('events', []).append({'kind': kind, 'utc': datetime.now(timezone.utc).isoformat(), **details})


def open_transaction(root, sentinel, protocol_sha256, scenes):
    root, sentinel = Path(root), Path(sentinel)
    require(not root.is_symlink() and not sentinel.is_symlink(), 'Symlink transaction/source root')
    root, sentinel = root.resolve(), sentinel.resolve()
    require(not root.is_relative_to(sentinel) and sentinel.is_dir(), 'Staging must be outside source discovery')
    path = root/'journal.json'
    require(not path.is_symlink(), 'Symlink journal is not supported')
    if path.exists():
        transaction = read_json(path.read_bytes())
        require(transaction['schema_version'] == 14 and transaction['root'] == str(root)
                and transaction['sentinel'] == str(sentinel) and transaction['protocol_sha256'] == protocol_sha256
                and matches_identity(root, transaction['root_identity'])
                and matches_identity(sentinel, transaction['sentinel_identity'])
                and [r['scene_id'] for r in transaction['scenes']] == [s['scene_id'] for s in scenes]
                and transaction['state'] in ('acquiring', 'ready', 'publishing', 'complete'), 'Transaction binding/ownership changed')
        require(all(item['targets'] == [a['target_relpath'] for a in scene['assets']]
                    for item, scene in zip(transaction['scenes'], scenes)), 'Transaction target allowlist changed')
        return transaction
    require(not root.exists() or not any(root.iterdir()), 'Unrecognized transaction without journal; inspect before recovery')
    root.mkdir(parents=True, exist_ok=True)
    require(root.stat().st_dev == sentinel.stat().st_dev, 'Staging/publication must share a filesystem')
    (root/'staged').mkdir()
    transaction = {'schema_version': 14, 'root': str(root), 'root_identity': identity(root),
                   'sentinel': str(sentinel), 'sentinel_identity': identity(sentinel),
                   'protocol_sha256': protocol_sha256, 'state': 'acquiring', 'events': [],
                   'scenes': [{'scene_id': s['scene_id'], 'state': 'new', 'identity': None,
                               'targets': [a['target_relpath'] for a in s['assets']],
                               'partial': None, 'receipts': {}, 'last_error': None} for s in scenes]}
    save(transaction)
    return transaction


def locations(transaction, index):
    item = transaction['scenes'][index]
    return (safe_path(Path(transaction['root'])/'staged', item['scene_id']),
            safe_path(Path(transaction['sentinel']), item['scene_id']))


def recover_partial(transaction, index):
    item = transaction['scenes'][index]; partial = item.get('partial')
    if not partial:
        return
    staged, _ = locations(transaction, index)
    path = safe_path(staged, partial['path'])
    require(partial['path'] in {name+'.download' for name in item['targets']}, 'Unexpected partial filename')
    if path.exists():
        require(partial.get('identity') is not None and path.is_file()
                and identity(path) == partial['identity'], 'Ambiguous partial ownership; inspect without deletion')
        event(transaction, 'owned_partial_removed', scene_id=item['scene_id'], path=partial['path'],
              discarded_bytes=path.stat().st_size)
        path.unlink(); sync_directory(path.parent)
    item['partial'] = None; save(transaction)


def check_tree(product, scene, partial=None):
    expected = {asset['target_relpath'] for asset in scene['assets']}
    if partial:
        expected.add(partial['path'])
    directories = {str(parent) for name in expected for parent in Path(name).parents if str(parent) != '.'}
    for path in product.rglob('*'):
        require(not path.is_symlink(), 'Symlink in staged source')
        relative = str(path.relative_to(product))
        require(relative in (directories if path.is_dir() else expected), 'Unexpected staged file/directory: '+relative)


def stream_asset(asset, target, transaction, index, opener=None):
    """Durable partial ownership precedes payload bytes; one request, no retries."""
    expected_md5(asset); target = Path(target)
    require(not target.exists(), 'Refusing to overwrite downloaded asset')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.download')
    require(not temporary.exists() and not temporary.is_symlink(), 'Unexpected partial file')
    item = transaction['scenes'][index]; staged, _ = locations(transaction, index)
    item['partial'] = {'path': str(temporary.relative_to(staged)), 'identity': None}
    event(transaction, 'transfer_started', scene_id=item['scene_id'], target=asset['target_relpath'])
    save(transaction)
    opener = opener or build_opener(NoRedirect()).open
    checksum, md5, count = hashlib.sha256(), hashlib.md5(), 0
    try:
        with temporary.open('xb') as stream:
            item['partial']['identity'] = identity(temporary); sync_directory(temporary.parent); save(transaction)
            request = Request(asset['url'], headers={'If-Match': asset['etag'], 'Accept-Encoding': 'identity',
                                                     'User-Agent': 'COMP1252-earlier-source-acquisition/14'})
            with opener(request, timeout=30) as response:
                require(response.status == 200 and response.geturl() == asset['url'], 'Full pinned HTTP 200 source required')
                require(response.headers.get('ETag') == asset['etag']
                        and int(response.headers.get('Content-Length', '-1')) == asset['size'], 'Source ETag/length changed')
                while True:
                    chunk = response.read(min(2**20, asset['size']-count+1))
                    if not chunk:
                        break
                    count += len(chunk); require(count <= asset['size'], 'Source exceeded declared size')
                    require(shutil.disk_usage(staged).free >= RESERVE+len(chunk), 'Disk reserve breached during transfer')
                    stream.write(chunk); checksum.update(chunk); md5.update(chunk)
            stream.flush(); os.fsync(stream.fileno())
        require(count == asset['size'] and md5.hexdigest() == expected_md5(asset), 'Source truncated or MD5 mismatch')
        exclusive_move(temporary, target)
        receipt = {'bytes': count, 'sha256': checksum.hexdigest(), 'md5': md5.hexdigest(),
                   'etag': asset['etag'], 'source_url': asset['url'], 'if_match_verified': True,
                   'content_length_verified': True}
        item['receipts'][asset['target_relpath']] = receipt
        event(transaction, 'transfer_verified', scene_id=item['scene_id'], target=asset['target_relpath'], bytes=count)
        item['partial'] = None; save(transaction)
        return receipt
    except BaseException as error:
        item['last_error'] = type(error).__name__+': '+str(error); save(transaction)
        event(transaction, 'transfer_failed', scene_id=item['scene_id'], target=asset['target_relpath'],
              observed_response_bytes=count, error=item['last_error'])
        save(transaction)
        recover_partial(transaction, index)
        raise


def validate_local(product, scene, receipts, qa_reader=None):
    from pixelqa_scene_metadata import qa_scene_metadata
    check_tree(product, scene)
    require(set(receipts) == {a['target_relpath'] for a in scene['assets']}, 'Complete source receipts required')
    for asset in scene['assets']:
        path = safe_path(product, asset['target_relpath'])
        verify_asset(path, asset, receipts[asset['target_relpath']]['sha256'])
        stat = path.stat(); receipts[asset['target_relpath']]['stat'] = [stat.st_size, stat.st_mtime_ns]
    reader = qa_reader or (lambda path: qa_scene_metadata(path, ROOT))
    record = reader(product)
    require(record['metadata_usable'] is True and record['cloud_model_usable'] is True
            and record['shadow_geometry_usable'] is True and not record['issues']
            and record['scene_id'] == scene['scene_id'] and record['acquisition_date'] == scene['acquisition_date']
            and record['processing_level'] == 'L1C' and record['tile'] == '36UXB'
            and record['processing_baseline'] == '02.06' and record['quantification_value'] == 10000
            and record['nonzero_offset'] is False and record['radiometric_offsets'] == scene['metadata']['radio_add_offsets']
            and record['crs'] == scene['metadata']['crs'] and record['grid_10m'] == scene['metadata']['grid_10m']
            and record['solar_zenith_deg'] == scene['metadata']['solar_zenith_deg']
            and record['solar_azimuth_deg'] == scene['metadata']['solar_azimuth_deg'],
            'Actual local frozen QA metadata prerequisites/identity changed')
    return record


def acquire_scene(transaction, scenes, index, qa_reader=None, opener=None):
    require(transaction['state'] in ('acquiring', 'ready'), 'Cannot download into publishing/completed transaction')
    scene = scenes[index]; item = transaction['scenes'][index]; staged, final = locations(transaction, index)
    require(not final.exists(), 'Source destination already exists; preserve user tree')
    if item['identity'] is None:
        require(not staged.exists(), 'Unowned staged tree; manual inspection required')
        staged.mkdir(); sync_directory(staged.parent); item['identity'] = identity(staged); save(transaction)
    require(matches_identity(staged, item['identity']), 'Staged tree ownership changed')
    recover_partial(transaction, index); check_tree(staged, scene)
    downloaded, reused = 0, 0
    for asset in scene['assets']:
        target = safe_path(staged, asset['target_relpath']); known = item['receipts'].get(asset['target_relpath'])
        if target.exists():
            checked = verify_asset(target, asset, known['sha256'] if known else None)
            # A completed rename may precede its journal update; size+source MD5 allow recovery.
            if not known:
                item['receipts'][asset['target_relpath']] = {**checked, 'etag': asset['etag'],
                    'source_url': asset['url'], 'recovered_verified_complete_file': True}
                save(transaction)
            reused += 1
        else:
            require(known is None, 'Receipt exists for missing source file')
            require(shutil.disk_usage(staged).free >= RESERVE+asset['size'], 'Insufficient disk for next pinned asset')
            stream_asset(asset, target, transaction, index, opener); downloaded += 1
        if (downloaded+reused) % 10 == 0:
            print({'scene': scene['acquisition_date'], 'verified_assets': downloaded+reused}, flush=True)
    record = validate_local(staged, scene, item['receipts'], qa_reader)
    item.update(state='prepared', staged_metadata=record, last_error=None); save(transaction)
    event(transaction, 'scene_prepared', scene_id=item['scene_id'], assets=len(item['receipts'])); save(transaction)
    if all(r['state'] == 'prepared' for r in transaction['scenes']):
        transaction['state'] = 'ready'; save(transaction)
    return {'scene_id': scene['scene_id'], 'assets': len(item['receipts']), 'new_downloads': downloaded,
            'verified_reused_assets': reused, 'payload_bytes': sum(a['size'] for a in scene['assets']),
            'source_receipts': item['receipts'], 'staged_metadata': record, 'transaction_state': transaction['state']}


def rollback_publication(transaction):
    """Move only owned published trees back; never delete/replace a user tree."""
    for index in reversed(range(len(transaction['scenes']))):
        item = transaction['scenes'][index]; staged, final = locations(transaction, index)
        if matches_identity(final, item['identity']):
            require(not staged.exists(), 'Rollback staging occupied; manual inspection required')
            exclusive_move(final, staged)
        require(matches_identity(staged, item['identity']), 'Prepared tree unavailable; manual recovery required')
        item['state'] = 'prepared'
    transaction['state'] = 'ready'; event(transaction, 'publication_rolled_back'); save(transaction)


def publish(transaction, scenes, qa_reader=None, after_move=None):
    require(transaction['state'] in ('ready', 'publishing'), 'Both prepared scenes required before publication')
    # On restart, a rename may have happened before its journal update; inode ownership is authoritative.
    try:
        for index, scene in enumerate(scenes):
            item = transaction['scenes'][index]; staged, final = locations(transaction, index)
            product = final if matches_identity(final, item['identity']) else staged
            require(matches_identity(product, item['identity']), 'Prepared source ownership changed')
            require(product == final or not final.exists(), 'Refusing existing unowned final tree')
            validate_local(product, scene, item['receipts'], qa_reader)
        transaction['state'] = 'publishing'; save(transaction)
        for index in range(len(scenes)):
            item = transaction['scenes'][index]; staged, final = locations(transaction, index)
            if not matches_identity(final, item['identity']):
                require(matches_identity(staged, item['identity']), 'Prepared inode changed')
                exclusive_move(staged, final)
                if after_move:
                    after_move(index)
            item['state'] = 'published'; save(transaction)
        result = completion_result(transaction, scenes, qa_reader)
        for item, record in zip(transaction['scenes'], result['final_scenes']):
            item['final_metadata'] = record
        atomic_json(Path(transaction['root'])/'completion.json', result)
        transaction['state'] = 'complete'; event(transaction, 'publication_complete'); save(transaction)
        return result
    except BaseException:
        rollback_publication(transaction)
        raise


def completion_result(transaction, scenes, qa_reader=None):
    records = []
    for index, scene in enumerate(scenes):
        item = transaction['scenes'][index]; _, final = locations(transaction, index)
        require(matches_identity(final, item['identity']), 'Published tree ownership changed')
        records.append(validate_local(final, scene, item['receipts'], qa_reader))
    return {'protocol_sha256': transaction['protocol_sha256'], 'transaction_state': 'complete',
            'new_scenes': len(scenes), 'new_assets': sum(len(i['receipts']) for i in transaction['scenes']),
            'payload_bytes': sum(a['size'] for s in scenes for a in s['assets']),
            'source_receipts': {i['scene_id']: i['receipts'] for i in transaction['scenes']},
            'final_scenes': records, 'satellite_pixel_reads': 0, 'model_executions': 0, 'labels_approved': 0}


def worker(args, output):
    import resource
    protocol = read_json(args.protocol.read_bytes()); scenes, receipts = inputs(protocol)
    prior_count = preserved_sources(protocol)
    root = ROOT/TX_REL; sentinel = ROOT/SOURCE_REL
    require(args.transaction_root.resolve() == root.resolve(), 'Fixed transaction root required')
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.with_name(root.name+'.lock')
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transaction = open_transaction(root, sentinel, digest(args.protocol), scenes)
        require((transaction['state'] == 'complete') == (args.mode == 'verify'),
                'Completed transaction requires read-only verify; download/publication cannot repeat')
        free_before = shutil.disk_usage(sentinel).free
        allowance = protocol['rules']['initial_conservative_disk_allowance_bytes'] if args.mode == 'download' else 0
        require(free_before >= RESERVE+allowance,
                'Insufficient live disk for conservative staging allowance and reserve')
        try:
            if args.mode == 'download':
                result = acquire_scene(transaction, scenes, args.scene_index)
            elif args.mode == 'publish':
                result = publish(transaction, scenes)
            else:
                result = completion_result(transaction, scenes)
                require(result == read_json((root/'completion.json').read_bytes()), 'Persistent completion receipt changed')
        except BaseException as error:
            transaction['last_error'] = {'mode': args.mode, 'scene_index': args.scene_index,
                                       'error': type(error).__name__+': '+str(error)}
            event(transaction, 'worker_failed', **transaction['last_error'])
            save(transaction)
            raise
        require(shutil.disk_usage(sentinel).free >= RESERVE, 'Disk reserve breached')
        preserved_sources(protocol); inputs(protocol)
        result.update(disk_free_bytes_before=free_before, disk_free_bytes_after=shutil.disk_usage(sentinel).free,
                      prior_source_stats_unchanged=prior_count)
    write(output/'result.json', result)
    (output/'report.md').write_text('# Earlier source acquisition\n\nMode: '+args.mode+'; transaction: '+
        transaction['state']+'. No source pixels decoded, cloud/forest models run or labels approved.\n')
    require(not any(name in sys.modules for name in ('torch', 'numpy', 'rasterio', 's2cloudless')), 'Pixel/model library imported')
    write(output/'metadata.json', {'execution_status': 'completed', 'mode': args.mode, 'scene_index': args.scene_index,
        'protocol_sha256': digest(args.protocol), 'input_receipts': receipts,
        'transaction_journal': str(root/'journal.json'), 'transaction_journal_sha256': digest(root/'journal.json'),
        'artifact_sha256': {name: digest(output/name) for name in ('result.json', 'report.md')},
        'satellite_pixel_reads': 0, 'model_executions': 0, 'labels_approved': 0,
        'worker_high_water_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                         (2**20 if sys.platform == 'darwin' else 1024), 2)})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationEarlierAcquisitionV14/protocol.json')
    parser.add_argument('--transaction-root', type=Path, default=ROOT/TX_REL)
    parser.add_argument('--mode', choices=('download', 'publish', 'verify'), required=True)
    parser.add_argument('--scene-index', type=int, choices=(0, 1))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.mode == 'download') != (args.scene_index is not None):
        parser.error('--scene-index is required only for download')
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Retain 512 MiB/1.5 GiB/300 s bounds')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10); worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='earlier-source-acquisition', gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
