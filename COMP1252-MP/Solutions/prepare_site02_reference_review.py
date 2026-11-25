#!/usr/bin/env python3
"""Prepare two scene-locked site-02 Landsat crops and replay their review boards."""

import argparse
import os
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory

from inventory_unseen_validation_sites import RESOURCES, ROOT, digest, load, require, rooted, write
from preflight_fresh_validation_sources import checked_artifacts, evidence_hashes
from prepare_july_visual_review import array_hash, board, outline
from prepare_site01_reference_review import MAX_BYTES, PREVIEW_SIZE, SCALE, fetch_crop, read_crop, reference_url


REPORT = ROOT/'Docs/Reviews/ValidationSite02ReferenceV34'
LOCAL = ROOT/'Datasets/ValidationPilot/v34'
PROTOCOL = REPORT/'protocol.json'
SITE = 'site-02'
VIEWS = ('landsat-plain.png', 'landsat-outlined.png')
SCENES = ('LC08_L1TP_178025_20171116_20200902_02_T1', 'LC08_L1GT_178025_20180103_20200902_02_T2')


def verified_inputs():
    protocol = load(PROTOCOL)
    require(protocol['schema_version'] == 34 and protocol['site_id'] == SITE and protocol['resources'] == RESOURCES
            and protocol['max_bytes_per_image'] == MAX_BYTES and protocol['preview_size_pixels'] == [PREVIEW_SIZE]*2
            and protocol['view_order'] == list(VIEWS), 'V34 protocol differs')
    paths = {}
    for key, receipt in protocol['input_sha256'].items():
        path = rooted(receipt['path'])
        require(digest(path) == receipt['sha256'], 'V34 pinned input changed: '+key)
        paths[key] = path
    comparison = load(paths['v32_comparison'])
    site = comparison['sites'][SITE]
    v28_site, = [row for row in load(paths['v28_plan'])['sites'] if row['site_id'] == SITE]
    require(comparison['site_02_reserve']['best_nominal_pair_retained'] == site['options'][0] == protocol['sentinel_pair']
            and site['grid'] == protocol['grid'] == v28_site['grid']
            and site['alternative_control'] == protocol['control']
            and all(site[key] == v28_site[key] for key in ('event_id', 'group_id', 'qa_context_bounds')),
            'V28 and V32 site, pair or exact control differs')
    access = {row['scene_id']: row for row in load(paths['v32_access'])['landsat_8']['records']}
    require(tuple(source['scene_id'] for source in protocol['sources']) == SCENES, 'V34 source roster differs')
    for source in protocol['sources']:
        record = access[source['scene_id']]
        require(source['esri_raster_id'] == record['archive_object_id']
                and source['capture_time_utc'] == record['capture_timestamp_utc']
                and source['usgs_scene_cloud_percent'] == record['usgs_scene_cloud_percent']
                and record['esri_catalogue_within_covers_crop'] and record['usgs_stac_footprint_covers_crop'],
                'V34 source differs from the verified V32 record')
    return protocol, load(paths['catalogue']), v28_site


def masks(protocol, catalogue, site):
    import numpy as np
    from pyproj import Transformer
    from scipy.ndimage import binary_erosion
    from shapely.geometry import shape
    from shapely.ops import transform
    from assess_pixel_quality import core_grid
    from validate_sample import burn

    require(site['context_event_ids'] == [site['event_id']] and not site['prior_temporal_context_event_ids'],
            'V28 site-02 target should have no earlier overlapping annotations')
    anchor, = [feature for feature in catalogue['features'] if feature['id'] == site['event_id']]
    project = Transformer.from_crs('EPSG:4326', site['grid']['crs'], always_xy=True).transform
    target = burn([transform(project, shape(anchor['geometry']))], core_grid(site['grid'])).astype(bool)
    interior = binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
    require(int(target.sum()) == site['positive_target_pixels'] == 194
            and int(interior.sum()) == site['positive_interior_pixels'] == 131
            and array_hash(interior) == site['positive_interior_sha256'], 'V28 site-02 mask hashes differ')
    control = protocol['control']
    row, col, width, radius = (control[key] for key in ('row', 'col', 'width', 'context_radius_pixels'))
    require((row, col, width, radius) == (5, 45, 16, 2), 'V32 alternate control geometry differs')
    roi, context = np.zeros_like(target), np.zeros_like(target)
    roi[row:row+width, col:col+width] = True
    context[row-radius:row+width+radius, col-radius:col+width+radius] = True
    require(int(roi.sum()) == control['pixels'] == 256 and int(context.sum()) == control['context_pixels'] == 400
            and not (target & context).any(), 'Fixed target and control context differ')
    return {'target': target, 'interior': interior, 'control': roi, 'control_context': context}


def build_packet(output, original=None):
    from PIL import Image, ImageDraw

    protocol, catalogue, site = verified_inputs()
    areas = masks(protocol, catalogue, site)
    receipts, plain, outlined = [], [], []
    for source in protocol['sources']:
        url = reference_url(source, protocol['grid'])
        day = source['capture_time_utc'].split('T')[0]
        name = f'landsat-{day}.png'
        path = (original or output)/name
        if original is None:
            fetch_crop(url, path)
        crop = read_crop(path).resize((64*SCALE, 64*SCALE), Image.Resampling.NEAREST)
        marked = crop.copy()
        draw = ImageDraw.Draw(marked)
        for key, colour in (('target', '#ffdb63'), ('control_context', '#70eff3'), ('control', '#ffffff')):
            outline(draw, areas[key], SCALE, colour)
        plain.append((source['caption'], crop))
        outlined.append((source['caption'], marked))
        receipts.append({
            'scene_id': source['scene_id'], 'capture_time_utc': source['capture_time_utc'],
            'source_url': url, 'usgs_item_url': source['usgs_item_url'], 'tier': source['tier'],
            'scene_cloud_percent': source['usgs_scene_cloud_percent'], 'filename': name,
            'bytes': path.stat().st_size, 'sha256': digest(path),
        })
    subtitle = 'Landsat 8 pan-sharpened RGB | 43 x 43 source render | North up'
    board(output/VIEWS[0], 'Site-02 | Dated reference crops', subtitle, plain,
          'USGS Landsat 8 via Esri. Native pan sampling 15 m, multispectral 30 m.')
    board(output/VIEWS[1], 'Site-02 | Fixed target and control', subtitle, outlined,
          'Yellow target, white control, cyan control context.')
    write(output/'packet.json', {
        'schema_version': 34, 'site_id': SITE, 'group_id': site['group_id'], 'event_id': site['event_id'],
        'grid': protocol['grid'], 'control': protocol['control'], 'sentinel_pair': protocol['sentinel_pair'],
        'source_preview_size_pixels': [PREVIEW_SIZE]*2, 'display_resampling': 'nearest',
        'mask_sha256': {key: array_hash(mask) for key, mask in areas.items()},
        'receipts': receipts, 'view_order': list(VIEWS), 'boards_sha256': {name: digest(output/name) for name in VIEWS},
    })


def worker(args, output):
    checksum = digest(PROTOCOL)
    if args.mode == 'prepare':
        build_packet(output)
        result = {'image_requests': len(SCENES), 'artifact_sha256': evidence_hashes(output)}
    else:
        original = args.packet.resolve()
        require(original.is_relative_to(LOCAL), 'Review packet must be in the local V34 output area')
        metadata = load(original/'metadata.json')
        require(metadata['execution_status'] == 'completed' and metadata['mode'] == 'prepare'
                and metadata['protocol_sha256'] == checksum and metadata['artifact_sha256'] == evidence_hashes(original),
                'V34 prepared packet binding differs')
        checked_artifacts(original, metadata['artifact_sha256'])
        with TemporaryDirectory(dir=output) as temporary:
            replay = Path(temporary)
            build_packet(replay, original)
            require(all(digest(path) == metadata['artifact_sha256'][path.name] for path in replay.iterdir()),
                    'V34 board or packet replay differs')
        result = {'image_requests': 0, 'verified_packet_metadata_sha256': digest(original/'metadata.json'),
                  'verified_artifact_sha256': metadata['artifact_sha256'], 'byte_identical_replay': True}
    require(digest(PROTOCOL) == checksum and 'torch' not in sys.modules, 'Protocol changed or forest runtime imported')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'schema_version': 34, 'execution_status': 'completed', 'mode': args.mode, 'protocol_sha256': checksum,
        'sentinel_jp2_reads': 0, 'model_executions': 0, 'labels_approved': 0,
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2), **result,
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'verify'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=120)
    args = parser.parse_args()
    if (args.mode == 'verify') != (args.packet is not None):
        parser.error('Only verify mode uses --packet')
    if not args.output.resolve().is_relative_to(LOCAL):
        parser.error('Output must be in the local V34 output area')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='site02-landsat-reference',
                    gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
