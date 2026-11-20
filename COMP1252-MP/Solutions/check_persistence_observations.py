#!/usr/bin/env python3
"""Screen and replay the fixed Area B follow-ups on its original crop grid."""

import argparse
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile

from prepare_july_visual_review import (
    array_hash, board, colour_map, detail_box, panel, rgb, verified_inputs,
)
from preflight_persistence_source import PROTOCOL, checked_protocol
from screen_repaired_validation import (
    GROUP, POLICY, ROOT, contains, count_band_requests, digest, load_safe_cache,
    read, relative, require, rooted, unique, write,
)

DATES = ('2018-08-05', '2018-08-10')
JULY_DATES = ('2018-07-06', '2018-07-31')
CACHE = ROOT/'Datasets/ValidationPilot/v20/observation_cache'
CLEAR = 'context_clear_for_visual_review'


def frozen_area():
    """Bind the 92-pixel original anchor target and its two prior images."""
    import numpy as np
    import rasterio
    from scipy.ndimage import binary_erosion

    from assess_pixel_quality import MASK_NAMES, core_grid, label_target, load_events
    from audit_validation_context import context_mask
    from validate_sample import burn

    paths, screen, metadata, picks = verified_inputs()
    entry, (index, pair, row) = next(pick for pick in picks if pick[0]['area'] == 'B')
    require(pair['group_id'] == GROUP and pair['split'] == 'validation'
            and tuple(pair[key] for key in ('pre_date', 'post_date')) == JULY_DATES,
            'Area B frozen split, group or dates differ')
    events = load_events(paths['catalogue'].parent)
    grid = core_grid(pair['grid'])
    temporal = label_target(pair, events).astype(bool)
    anchor = burn([events[entry['event_id']]['geometry']], grid).astype(bool)
    target = anchor & temporal
    interior = binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
    context, interior_context = context_mask(target, 2), context_mask(interior, 2)
    checked = screen['anchor_contexts'][index]
    require(array_hash(target) == checked['target_mask_sha256']
            and array_hash(interior) == checked['interior_mask_sha256']
            and array_hash(context) == checked['target']['context_mask_sha256']
            and array_hash(interior_context) == checked['positive_interior']['context_mask_sha256']
            and [roi.sum() for roi in (target, interior, context, interior_context)] == [92, 25, 274, 104],
            'Area B support or full QA margins differ')
    mask_name = row['qa_mask_file']
    mask_path = paths['screen'].parent/mask_name
    require(digest(mask_path) == metadata['artifact_sha256'][mask_name], 'Frozen reference mask changed')
    with rasterio.open(mask_path) as tif:
        require(tif.shape == (64, 64) and tif.count == 6 and tif.transform == grid['transform']
                and tif.crs.to_string() == pair['grid']['crs'] and tif.descriptions == (*MASK_NAMES, 'reference_new_loss')
                and np.array_equal(tif.read(6), temporal), 'Frozen reference plane or grid differs')

    archive_root = paths['cache_index'].parent.parent
    archived = unique(read(paths['cache_index']), 'cache_key')
    endpoints, caches = [], []
    for side, receipt in zip(('pre', 'post'), screen['cache_receipts'][index*2:index*2+2]):
        cached = archived[receipt['cache_key']]
        require(all(cached[key] == value for key, value in receipt.items())
                and digest(archive_root/cached['snapshot']) == receipt['sha256']
                and {'pair_id': pair['pair_id'], 'side': side} in cached['bindings'],
                'Earlier observation snapshot differs')
        record = load_safe_cache(receipt, row['per_date'][side]['observation'])
        require(record['metadata']['scene_id'] == pair[side+'_scene_id']
                and record['metadata']['grid'] == pair['grid'], 'Earlier source or crop differs')
        endpoints.append(record)
        caches.append(receipt)
    geometry = {'target': target, 'interior': interior, 'context': context, 'interior_context': interior_context}
    info = {**entry, 'group_id': GROUP, 'grid': pair['grid'], 'qa_reference_sha256': digest(mask_path),
            'mask_sha256': {key: array_hash(mask) for key, mask in geometry.items()},
            'july_cache_receipts': caches}
    return info, geometry, endpoints


def checked_publication(directory, grid, candidates):
    """Use only dates published and checked by the bounded V20 transaction."""
    from pixelqa_scene_metadata import qa_scene_metadata

    directory = rooted(relative(directory))
    files = {name: digest(directory/name) for name in ('result.json', 'journal_snapshot.json')}
    meta, result = (read(directory/name) for name in ('metadata.json', 'result.json'))
    require(meta['mode'] == 'verify' and meta['execution_status'] == 'completed'
            and meta['protocol_sha256'] == result['protocol_sha256'] == digest(PROTOCOL)
            and meta['artifact_sha256'] == files and result['transaction_state'] == 'complete',
            'Require the verified V20 publication receipt')
    by_date = unique(result['final_scenes'], 'acquisition_date')
    receipts = result['source_receipts']
    require(set(by_date) <= set(candidates) and len(by_date) == result['new_scenes']
            and {record['scene_id'] for record in by_date.values()} == set(receipts)
            and result['new_assets'] == 58*len(by_date), 'Publication contains unexpected dates or assets')
    west, south, east, north = grid['bounds']
    padded = [west-1200, south-1200, east+1200, north+1200]
    for day, record in by_date.items():
        scene = record['scene_id']
        product = Path('Datasets/Sentinel-2')/scene
        require('_N0206_R064_T36UXB_' in scene and record['path'] == str(product/(scene+'.SAFE'))
                and record['metadata_usable'] and record['cloud_model_usable'] and record['shadow_geometry_usable']
                and record['crs'] == grid['crs'] and contains(record['grid_10m']['bounds'], padded)
                and qa_scene_metadata(rooted(record['path']), ROOT) == record, 'Follow-up QA prerequisites differ')
        require(len(receipts[scene]) == 58, 'Missing follow-up asset receipt')
        for name, receipt in receipts[scene].items():
            path = rooted(str(product/name))
            require(path.is_relative_to(ROOT/product), 'Follow-up asset escapes verified source')
            stat = path.stat()
            require([stat.st_size, stat.st_mtime_ns] == receipt['stat'], 'Follow-up asset fingerprint changed')
    return by_date, {**files, 'metadata.json': digest(directory/'metadata.json')}


def review_views(output, day, records, masks):
    box = detail_box(masks['target'])
    dates = (*JULY_DATES, day)
    rgb_values = [rgb(record['rgb']) for record in records]
    titles = [('plain', 'Area B | Full crop'), ('detail', 'Area B | Fixed target detail'),
              ('ndvi', 'Area B | Vegetation index')]
    names = [day+'-'+name+'.png' for name, _ in titles]
    board(output/names[0], titles[0][1], 'Matched reflectance 0 to 0.30',
          [(date, panel(values, (0, 0, 64, 64), 7)) for date, values in zip(dates, rgb_values)],
          'North up | 64 x 64 source pixels | 10 m per pixel')
    overlays = [(masks['context'], '#70eff3'), (masks['target'], '#ffdb63'), (masks['interior'], '#ffffff')]
    board(output/names[1], titles[1][1], 'Matched reflectance 0 to 0.30 | native pixels enlarged',
          [(date, panel(values, box, 10, overlays)) for date, values in zip(dates, rgb_values)],
          'Yellow: target | White: eroded interior | Cyan: target context')
    greens = [(151, 59, 38), (211, 169, 110), (234, 231, 202), (131, 167, 122), (35, 81, 63)]
    ndvi = [colour_map(record['ndvi'], (-.1, .9), greens) for record in records]
    board(output/names[2], titles[2][1], 'Matched NDVI -0.1 to 0.9 | rust to green',
          [(date, panel(values, box, 10, [(masks['target'], '#ffdb63')])) for date, values in zip(dates, ndvi)],
          'Yellow: target | NDVI is a descriptive check on the RGB interpretation')
    return names


def describe_ndvi(records, masks):
    import numpy as np

    results = {}
    for key in ('target', 'interior'):
        early, july, later = (record['ndvi'][masks[key]] for record in records)
        results[key] = {'pixels': int(early.size), 'median_6_july': float(np.median(early)),
                        'median_31_july': float(np.median(july)), 'median_followup': float(np.median(later)),
                        'median_followup_minus_31_july': float(np.median(later-july)),
                        'followup_below_6_july_pixels': int((later < early).sum())}
    return results


def screen_record(output, date, source, record, receipt, endpoints, masks):
    import numpy as np

    from pixelqa_masks import region_metrics
    from prepare_validation_manual import observation_metrics

    target = observation_metrics(record, masks['target'], masks['context'])
    interior = observation_metrics(record, masks['interior'], masks['interior_context'])
    complete = not record['metadata']['context_incomplete']
    clear = complete and target['context_clear'] and interior['context_clear']
    reasons = []
    if not complete:
        reasons.append('incomplete_off_crop_context')
    if target['context']['unusable_pixels']:
        reasons.append('flags_in_full_target_context')
    if target['context_nonfinite_rgb_ndvi_pixels']:
        reasons.append('nonfinite_rgb_ndvi_in_full_target_context')
    snapshot = date+'-observation.npz'
    shutil.copyfile(rooted(receipt['path']), output/snapshot)
    require(digest(output/snapshot) == receipt['sha256'], 'Follow-up snapshot changed while copying')
    return {'date': date, 'scene_id': source['scene_id'], 'scene_path': source['path'],
            'status': CLEAR if clear else 'qa_obscured_or_invalid', 'reasons': reasons,
            'target': target, 'interior': interior, 'crop_quality': region_metrics(record, np.ones((64, 64), bool)),
            'off_crop_context_complete': complete, 'observation': record['metadata'],
            'cache_receipt': receipt, 'snapshot': snapshot,
            'ndvi_descriptive_only': describe_ndvi([*endpoints, record], masks) if clear else None,
            'views': review_views(output, date, [*endpoints, record], masks) if clear else []}


def screen_later(output, grid, by_date, endpoints, masks):
    import rasterio
    import assess_pixel_quality as quality
    from screen_earlier_validation import prevalidate_cache

    identity = quality.model_identity()
    require(quality.policy_id(identity) == POLICY, 'Frozen cloud QA policy differs')
    reader = quality.ObservationReader(CACHE, identity) if by_date else None
    rows = []
    errors = []
    with rasterio.Env(GDAL_CACHEMAX=32*2**20, GDAL_NUM_THREADS='1'), count_band_requests(quality) as reads:
        for date in DATES:
            source = by_date.get(date)
            if source is None:
                rows.append({'date': date, 'status': 'source_unavailable_in_verified_publication', 'views': []})
                continue
            try:
                scene_meta = reader.metadata(source['path'])
                source_id = scene_meta['cache_source_id']
                key = quality.object_hash({'source': source_id, 'grid': grid, 'config_id': POLICY})
                path = reader.cache_root/(key+'.npz')
                if path.exists():
                    prevalidate_cache(path, source_id, grid)
                record = reader.observation(source['path'], grid)
                require(record['metadata']['scene_id'] == source['scene_id']
                        and record['metadata']['grid'] == grid and record['metadata']['cache_key'] == key,
                        'Follow-up observation source or grid differs')
                receipt = {'path': relative(path), 'bytes': path.stat().st_size,
                           'sha256': digest(path), 'cache_key': key}
                safe_record = load_safe_cache(receipt, record['metadata'])
                rows.append(screen_record(output, date, source, safe_record, receipt, endpoints, masks))
            except Exception as error:
                reason = type(error).__name__+': '+str(error)
                errors.append({'date': date, 'scene_id': source['scene_id'], 'error': reason})
                rows.append({'date': date, 'scene_id': source['scene_id'], 'status': 'processing_error',
                             'error': reason, 'views': []})

    hits, misses = (reader.hits, reader.misses) if reader else (0, 0)
    require(hits+misses <= len(by_date) and reads['requests'] <= 11*misses, 'Fixed QA cost bound exceeded')
    if not errors:
        require(hits+misses == len(by_date) and reads['requests'] == 11*misses, 'QA read count differs')
    return rows, errors, hits, misses, reads['requests']


def artifact_hashes(output):
    return {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}


def replay_screen(output, screen_dir, common, by_date, endpoints, masks):
    screen_dir = rooted(relative(screen_dir))
    metadata = read(screen_dir/'metadata.json')
    prior = read(screen_dir/'result.json')
    require(metadata['mode'] == 'screen' and metadata['execution_status'] == 'completed'
            and metadata['protocol_sha256'] == digest(PROTOCOL) and not prior['processing_errors']
            and all(prior[key] == value for key, value in common.items()), 'Require a complete matching screen')
    require([row['date'] for row in prior['observations']] == list(DATES), 'Follow-up date roster differs')
    rows = []
    with tempfile.TemporaryDirectory(dir=output) as directory:
        stage = Path(directory)
        for row in prior['observations']:
            day = row['date']
            source = by_date.get(day)
            if source is None:
                rows.append({'date': day, 'status': 'source_unavailable_in_verified_publication', 'views': []})
                continue
            record = load_safe_cache(row['cache_receipt'], row['observation'])
            require(record['metadata']['scene_id'] == source['scene_id']
                    and record['metadata']['grid'] == common['area']['grid'], 'Follow-up cache identity differs')
            rows.append(screen_record(stage, day, source, record, row['cache_receipt'], endpoints, masks))
        require(rows == prior['observations'], 'Follow-up QA rows or visual evidence differ on replay')
        write(stage/'result.json', prior)
        hashes = artifact_hashes(stage)
    require(hashes == metadata['artifact_sha256']
            and all(digest(screen_dir/name) == checksum for name, checksum in hashes.items()),
            'Follow-up screen evidence differs on replay')
    return {'screen_metadata_sha256': digest(screen_dir/'metadata.json'), 'screen_artifact_sha256': hashes,
            'replayed_observations': len(by_date), 'replayed_views': sum(len(row['views']) for row in rows),
            'checks': {'fixed_grid_and_geometry': True, 'exact_qa_rows_and_views': True,
                       'original_and_later_cache_bindings': True, 'frozen_published_source_fingerprints': True}}


def worker(args, output):
    protocol = checked_protocol()
    require(protocol['candidate_dates'] == list(DATES), 'Fixed V20 candidate dates differ')
    info, masks, endpoints = frozen_area()
    by_date, publication_hashes = checked_publication(args.publication, info['grid'], DATES)
    common = {'schema_version': 20, 'protocol_sha256': digest(PROTOCOL), 'qa_policy_id': POLICY,
              'area': info, 'publication_sha256': publication_hashes, 'all_27_channels_checked': False,
              'forest_model_executions': 0, 'labels_approved': 0, 'validation_eligible': False}
    if args.mode == 'screen':
        rows, errors, hits, misses, requests = screen_later(output, info['grid'], by_date, endpoints, masks)
        result = {**common, 'observations': rows, 'processing_errors': errors, 'new_cache_hits': hits,
                  'new_cache_misses': misses, 'cloud_model_executions': misses,
                  'fused_band_window_requests': requests}
        status = 'incomplete_processing_errors' if errors else 'completed'
    else:
        result = {**common, **replay_screen(output, args.screen, common, by_date, endpoints, masks),
                  'cloud_model_executions': 0, 'fused_band_window_requests': 0}
        status = 'completed'
    write(output/'result.json', result)
    require('torch' not in sys.modules, 'Unexpected forest-model dependency in persistence worker')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {'execution_status': status, 'mode': args.mode, 'protocol_sha256': digest(PROTOCOL),
                                  'artifact_sha256': artifact_hashes(output), 'labels_approved': 0,
                                  'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2)})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('screen', 'verify'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--screen', type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    if (args.mode == 'verify') != (args.screen is not None):
        parser.error('Only verify mode uses --screen')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='persistence-observations',
                    gdal_cache_mib=32, warp_memory_mib=0)


if __name__ == '__main__':
    main()
