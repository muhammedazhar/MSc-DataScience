"""Stream one preflighted Sentinel-2 scene into the project's SAFE layout."""

import fcntl
import os
from pathlib import Path
import re
import shutil
import stat

import acquire_earlier_validation_sources as engine
from inventory_earlier_validation_sources import PRODUCT, ROOT, canonical, read_json, require, sha
from preflight_earlier_validation_metadata import BANDS, BASE, asset_manifest
from preflight_remaining_validation_metadata import READY


def checked_scene(scene):
    scene_id = scene['scene_id']
    product = PRODUCT.fullmatch(scene_id)
    require(product is not None and not scene_id.endswith('.SAFE')
            and scene.get('status') == READY and scene.get('issues') == [],
            'Choose a scene with a complete native L1C preflight')
    day = scene['acquisition_date']
    require(product['acquisition'][:8] == day.replace('-', ''), 'Scene acquisition date differs')
    year, month, day_of_month = map(int, day.split('-'))
    tile = product['tile']
    sequence = scene['sequence']
    require(isinstance(sequence, str) and re.fullmatch(r'0|[1-9][0-9]*', sequence),
            'Invalid native catalogue sequence')
    prefix = f'tiles/{int(tile[:2])}/{tile[2]}/{tile[3:]}/{year}/{month}/{day_of_month}/{sequence}/'
    product_key = f'products/{year}/{month}/{day_of_month}/{scene_id}/metadata.xml'
    require(scene['tile_prefix'] == prefix and scene['product_key'] == product_key,
            'Native L1C provider keys differ')
    metadata = scene['metadata']
    folder = metadata['granule_folder']
    require(re.fullmatch(r'L1C_T'+tile+r'_A\d{6}_\d{8}T\d{6}', folder) is not None,
            'Native SAFE granule folder differs')
    require(all(metadata['image_paths'].get(band) ==
                f'GRANULE/{folder}/IMG_DATA/T{tile}_{product["acquisition"]}_{band}.jp2' for band in BANDS),
            'Native SAFE band paths differ')
    assets = scene['assets']
    require(isinstance(assets, list) and len(assets) == 58, 'Expected 58 preflight assets')
    listed = {asset['key']: {key: asset[key] for key in ('key', 'url', 'size', 'etag')} for asset in assets}
    require(len(assets) == len(listed) == 58 and product_key in listed
            and all(asset['url'] == BASE+key for key, asset in listed.items()),
            'Expected 58 exact native L1C provider pins')
    product_pin = {key: listed[product_key][key] for key in ('size', 'etag')}
    expected, issues = asset_manifest(scene, metadata, listed, product_pin)
    require(not issues and expected == assets and sum(asset['size'] for asset in assets) == scene['measured_payload_bytes'],
            'Preflight paths, roles or byte budget differ')
    return product


def validate_metadata(path, scene, product, qa_reader):
    from pixelqa_scene_metadata import qa_scene_metadata

    record = qa_reader(path) if qa_reader is not None else qa_scene_metadata(path, ROOT)
    expected = scene['metadata']
    baseline = product['baseline'][1:3]+'.'+product['baseline'][3:]
    require(record['scene_id'] == scene['scene_id'] and record['acquisition_date'] == scene['acquisition_date']
            and record['processing_level'] == 'L1C' and record['tile'] == product['tile']
            and record['baseline_code'] == product['baseline']
            and record['processing_baseline'] == expected['identity']['PROCESSING_BASELINE'] == baseline
            and record['quantification_value'] == expected['quantification_value']
            and record['radiometric_offsets'] == expected['radio_add_offsets'] and not record['nonzero_offset']
            and all(record[key] == expected[key] for key in ('crs', 'grid_10m', 'solar_zenith_deg', 'solar_azimuth_deg'))
            and all(record[key] for key in ('metadata_usable', 'cloud_model_usable', 'shadow_geometry_usable'))
            and not record['issues'], 'Local SAFE metadata or QA prerequisites differ from preflight')
    return record


def scene_receipts(path, scene, recorded):
    receipts = {}
    for asset in scene['assets']:
        name = asset['target_relpath']
        target = engine.safe_path(path, name)
        previous = recorded.get(name, {})
        checked = engine.verify_asset(target, asset, previous.get('sha256'))
        receipts[name] = {**previous, **checked, 'etag': asset['etag'], 'source_url': asset['url']}
    return receipts


def transfer_scene(scene, destination, transaction_root, *, opener=None, qa_reader=None):
    """Stage and publish a single scene, or verify a matching existing scene."""
    product = checked_scene(scene)
    scene_id = scene['scene_id']
    destination, transaction_root = Path(destination), Path(transaction_root)
    require(not destination.is_symlink() and not transaction_root.is_symlink(), 'Use real destination and staging directories')
    destination = destination.resolve()
    root = engine.safe_path(transaction_root.resolve(), scene_id)
    require(not root.is_relative_to(destination) and not destination.is_relative_to(root),
            'Keep staging outside Sentinel-2 scene discovery')
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent/'.sentinel-2-acquisition.lock'
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(lock_path, flags, 0o600), 'rb') as handle:
        require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode), 'Acquisition lock must be a regular file')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination.mkdir(exist_ok=True)
        free_before = shutil.disk_usage(destination).free
        require(free_before >= engine.RESERVE, 'Insufficient free disk for the Sentinel-2 reserve')
        final = engine.safe_path(destination, scene_id)
        manifest = sha(canonical({'scene_id': scene_id, 'assets': scene['assets']}))
        result = {'scene_id': scene_id, 'acquisition_date': scene['acquisition_date'], 'product_path': str(final),
                  'payload_bytes': scene['measured_payload_bytes'], 'asset_count': len(scene['assets']),
                  'manifest_sha256': manifest, 'disk_free_bytes_before': free_before, 'new_downloads': 0}

        journal = engine.safe_path(root, 'journal.json')
        if final.exists() and not journal.exists():
            require(final.is_dir() and not root.exists(), 'Existing scene or staging ownership is ambiguous')
            result.update(transaction_state='existing_verified', source_receipts=scene_receipts(final, scene, {}),
                          final_metadata=validate_metadata(final, scene, product, qa_reader))
            return result

        if not root.exists():
            require(free_before >= engine.RESERVE+scene['measured_payload_bytes'], 'Insufficient disk for the full scene and reserve')
        transaction = engine.open_transaction(root, destination, manifest, [scene])
        staged, final = engine.locations(transaction, 0)
        item = transaction['scenes'][0]
        require(item['receipts'].keys() <= set(item['targets']), 'Journal contains an unexpected asset receipt')
        if final.exists():
            require(transaction['state'] in ('publishing', 'complete') and not staged.exists()
                    and engine.matches_identity(final, item['identity']), 'Existing scene belongs to another acquisition')
        else:
            require(transaction['state'] != 'complete', 'A completed local scene is missing')
            if item['identity'] is None:
                require(not staged.exists(), 'Existing staged scene has no ownership record')
                staged.mkdir()
                engine.sync_directory(staged.parent)
                item['identity'] = engine.identity(staged)
                engine.save(transaction)
            require(engine.matches_identity(staged, item['identity']), 'Staged scene ownership differs')
            engine.recover_partial(transaction, 0)
            engine.check_tree(staged, scene)
            remaining_bytes = sum(asset['size'] for asset in scene['assets']
                                  if not engine.safe_path(staged, asset['target_relpath']).exists())
            require(shutil.disk_usage(staged).free >= engine.RESERVE+remaining_bytes,
                    'Insufficient disk for the remaining assets and reserve')
            for asset in scene['assets']:
                name = asset['target_relpath']
                target = engine.safe_path(staged, name)
                known = item['receipts'].get(name)
                if target.exists():
                    verified = engine.verify_asset(target, asset, known['sha256'] if known else None)
                    if known is None:
                        item['receipts'][name] = {**verified, 'etag': asset['etag'], 'source_url': asset['url'],
                                                  'recovered_verified_complete_file': True}
                        engine.save(transaction)
                else:
                    require(known is None, 'An asset with a saved receipt is missing')
                    require(shutil.disk_usage(staged).free >= engine.RESERVE+asset['size'], 'Disk reserve for next asset is insufficient')
                    engine.stream_asset(asset, target, transaction, 0, opener)
                    result['new_downloads'] += 1
            engine.check_tree(staged, scene)
            validate_metadata(staged, scene, product, qa_reader)
            require(shutil.disk_usage(staged).free >= engine.RESERVE, 'Disk reserve breached before publication')
            transaction['state'] = 'publishing'
            engine.save(transaction)
            engine.exclusive_move(staged, final)

        require(engine.matches_identity(final, item['identity']), 'Published scene ownership differs')
        engine.check_tree(final, scene)
        require(item['receipts'].keys() == set(item['targets']), 'Published scene receipts are incomplete')
        completion = {'manifest_sha256': manifest, 'product_path': str(final),
                      'source_receipts': scene_receipts(final, scene, item['receipts']),
                      'final_metadata': validate_metadata(final, scene, product, qa_reader)}
        completion_path = engine.safe_path(root, 'completion.json')
        if completion_path.exists():
            require(read_json(completion_path.read_bytes()) == completion, 'Persistent completion receipt differs')
        else:
            require(transaction['state'] == 'publishing', 'Missing completion receipt for published scene')
            engine.atomic_json(completion_path, completion)
        if transaction['state'] == 'publishing':
            item.update(state='published', last_error=None)
            transaction['state'] = 'complete'
            engine.event(transaction, 'publication_complete', scene_id=scene_id)
            engine.save(transaction)
        result.update(completion, transaction_state='complete', transaction_journal=str(journal),
                      disk_free_bytes_after=shutil.disk_usage(destination).free)
        return result
