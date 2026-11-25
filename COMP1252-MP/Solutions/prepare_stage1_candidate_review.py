#!/usr/bin/env python3
"""Prepare and replay two bounded Landsat previews for a fixed stage 1 site."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory

from inventory_unseen_validation_sites import RESOURCES, ROOT, digest, load, require, rooted, write
from preflight_fresh_validation_sources import checked_artifacts, evidence_hashes
from prepare_july_visual_review import array_hash, board, outline
from prepare_site01_reference_review import MAX_BYTES, PREVIEW_SIZE, SCALE, fetch_crop, read_crop, reference_url


REPORT = ROOT/'Docs/Reviews/ValidationStage1SearchV35'
LOCAL = ROOT/'Datasets/ValidationPilot/v35/candidates'
VIEWS = ('landsat-plain.png', 'landsat-outlined.png')


def pinned_inputs(protocol_path):
    protocol = load(protocol_path)
    require(protocol['schema_version'] == 35 and protocol['resources'] == RESOURCES
            and protocol['max_bytes_per_image'] == MAX_BYTES and protocol['view_order'] == list(VIEWS)
            and protocol['preview_size_pixels'] == [PREVIEW_SIZE]*2, 'Stage 1 preview protocol differs')
    paths = {}
    for role, receipt in protocol['input_sha256'].items():
        path = rooted(receipt['path'])
        require(digest(path) == receipt['sha256'], 'Stage 1 preview pinned input changed: '+role)
        paths[role] = path
    require(Path(__file__).resolve() in paths.values(), 'Stage 1 preview renderer must be pinned')
    site = protocol['site']
    roster = load(paths['roster'])
    require(roster['schema_version'] == 35 and site in roster['sites'] and site['split'] == 'validation'
            and protocol_path.stem == site['site_id'], 'Stage 1 preview site differs from the fixed roster')
    sources = protocol['sources']
    require(len(sources) == 2 and len({source['scene_id'] for source in sources}) == 2
            and len({source['esri_raster_id'] for source in sources}) == 2, 'Two distinct locked scenes are required')
    dates = []
    for source in sources:
        instant = datetime.fromisoformat(source['capture_time_utc'].replace('Z', '+00:00'))
        require(instant.utcoffset() is not None and source['tier'] in ('T1', 'T2'),
                'Capture time and Landsat tier are required')
        dates.append(instant.date())
    require(dates[0] < dates[1], 'Stage 1 preview dates must be distinct and ordered')
    requests = [(source, day.isoformat(), reference_url(source, site['grid']))
                for source, day in zip(sources, dates)]
    return protocol, load(paths['catalogue']), requests


def masks(site, control, catalogue):
    import numpy as np
    from pyproj import Transformer
    from scipy.ndimage import binary_dilation, binary_erosion
    from shapely.geometry import shape
    from shapely.ops import transform
    from assess_pixel_quality import core_grid
    from validate_sample import burn

    raw_grid = site['grid']
    west, south, east, north = raw_grid['bounds']
    require(raw_grid['crs'] == control['crs'] == 'EPSG:32636'
            and raw_grid['width'] == raw_grid['height'] == 64
            and (east-west, north-south) == (640, 640)
            and raw_grid['transform'] == [10, 0, west, 0, -10, north], 'Stage 1 exact core grid is required')
    ids = [site['event_id'], *site['prior_temporal_context_event_ids']]
    features = {feature['id']: feature for feature in catalogue['features'] if feature['id'] in ids}
    require(len(features) == len(ids), 'Stage 1 anchor and all earlier annotations are required')
    project = Transformer.from_crs('EPSG:4326', raw_grid['crs'], always_xy=True).transform
    anchor_geometry, *earlier = [transform(project, shape(features[key]['geometry'])) for key in ids]
    grid = core_grid(raw_grid)
    anchor = burn([anchor_geometry], grid).astype(bool)
    prior = burn(earlier, grid).astype(bool)
    target = anchor & ~prior
    interior = binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
    radius = 2
    context = binary_dilation(anchor, structure=np.ones((5, 5), dtype=bool))
    require(int(anchor.sum()) == site['anchor_pixels'] == int(anchor[2:-2, 2:-2].sum())
            and int(target.sum()) == site['positive_target_pixels']
            and int(interior.sum()) == site['positive_interior_pixels']
            and array_hash(interior) == site['positive_interior_sha256'], 'Stage 1 target support differs from V28')

    row, col, width = (control[key] for key in ('row', 'col', 'width'))
    require(all(type(item) is int for item in (row, col, width))
            and width == control['height'] and width in (8, 12, 16)
            and control['context_radius_pixels'] == radius and min(row, col) >= radius
            and max(row, col) + width + radius <= 64, 'Stage 1 full-context control must fit inside the core')
    roi = np.zeros_like(anchor)
    roi[row:row+width, col:col+width] = True
    halo = binary_dilation(roi, structure=np.ones((5, 5), dtype=bool))
    roi_bounds = [west + 10*col, north - 10*(row+width), west + 10*(col+width), north - 10*row]
    halo_bounds = [value + offset for value, offset in zip(roi_bounds, (-20, -20, 20, 20))]
    require(roi_bounds == control['bounds'] and halo_bounds == control['context_bounds']
            and int(roi.sum()) == control['pixels'] == width**2
            and int(halo.sum()) == control['context_pixels'] == (width+2*radius)**2
            and not (halo & (context | prior)).any(), 'Stage 1 control or full-context geometry differs')
    return {'anchor': anchor, 'target': target, 'interior': interior, 'earlier_annotation': prior,
            'target_context': context, 'control': roi, 'control_context': halo}


def build_packet(output, protocol_path, original=None):
    from PIL import Image, ImageDraw

    protocol, catalogue, requests = pinned_inputs(protocol_path)
    site = protocol['site']
    areas = masks(site, protocol['control'], catalogue)
    receipts, plain, outlined = [], [], []
    for source, day, url in requests:
        name = f'landsat-{day}.png'
        path = (original or output)/name
        if original is None:
            fetch_crop(url, path)
        crop = read_crop(path).resize((64*SCALE, 64*SCALE), Image.Resampling.NEAREST)
        marked = crop.copy()
        drawing = ImageDraw.Draw(marked)
        for key, colour in (('earlier_annotation', '#ff7ac8'), ('target', '#ffdb63'),
                            ('control_context', '#70eff3'), ('control', '#ffffff')):
            outline(drawing, areas[key], SCALE, colour)
        plain.append((source['caption'], crop))
        outlined.append((source['caption'], marked))
        receipts.append({
            'scene_id': source['scene_id'], 'capture_time_utc': source['capture_time_utc'],
            'source_url': url, 'usgs_item_url': source['usgs_item_url'], 'tier': source['tier'],
            'scene_cloud_percent': source['scene_cloud_percent'], 'filename': name,
            'bytes': path.stat().st_size, 'sha256': digest(path),
        })
    title = site['site_id'].capitalize()
    subtitle = 'Landsat 8 pan-sharpened RGB | 43 x 43 source render | North up'
    board(output/VIEWS[0], title+' | Dated reference crops', subtitle, plain,
          'USGS Landsat 8 via Esri. Native pan sampling 15 m, multispectral 30 m.')
    board(output/VIEWS[1], title+' | Fixed target and control', subtitle, outlined,
          'Yellow target, pink earlier annotation, white control, cyan control context.')
    write(output/'packet.json', {
        'schema_version': 35, 'site_id': site['site_id'], 'group_id': site['group_id'], 'event_id': site['event_id'],
        'grid': site['grid'], 'control': protocol['control'], 'sentinel_pair': protocol['sentinel_pair'],
        'source_preview_size_pixels': [PREVIEW_SIZE]*2, 'display_resampling': 'nearest',
        'mask_sha256': {key: array_hash(mask) for key, mask in areas.items()},
        'receipts': receipts, 'view_order': list(VIEWS), 'boards_sha256': {name: digest(output/name) for name in VIEWS},
    })


def worker(args, output):
    checksum = digest(args.protocol)
    if args.mode == 'prepare':
        build_packet(output, args.protocol)
        result = {'image_requests': 2, 'artifact_sha256': evidence_hashes(output)}
    else:
        original = args.packet.resolve()
        require(original.is_relative_to(LOCAL), 'Review packet must be in the local V35 candidate output area')
        metadata = load(original/'metadata.json')
        require(metadata['execution_status'] == 'completed' and metadata['mode'] == 'prepare'
                and metadata['protocol_sha256'] == checksum and metadata['artifact_sha256'] == evidence_hashes(original),
                'Stage 1 prepared packet binding differs')
        checked_artifacts(original, metadata['artifact_sha256'])
        with TemporaryDirectory(dir=output) as temporary:
            replay = Path(temporary)
            build_packet(replay, args.protocol, original)
            require(all(digest(path) == metadata['artifact_sha256'][path.name] for path in replay.iterdir()),
                    'Stage 1 review board or packet replay differs')
        result = {'image_requests': 0, 'verified_packet_metadata_sha256': digest(original/'metadata.json'),
                  'verified_artifact_sha256': metadata['artifact_sha256'], 'byte_identical_replay': True}
    require(digest(args.protocol) == checksum and 'torch' not in sys.modules,
            'Protocol changed or forest runtime imported')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'schema_version': 35, 'execution_status': 'completed', 'mode': args.mode, 'protocol_sha256': checksum,
        'sentinel_jp2_reads': 0, 'model_executions': 0, 'labels_approved': 0,
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2), **result,
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'verify'), required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=120)
    args = parser.parse_args()
    if (args.mode == 'verify') != (args.packet is not None):
        parser.error('Only verify mode uses --packet')
    args.protocol = args.protocol.resolve()
    if not args.protocol.is_relative_to(REPORT) or not args.output.resolve().is_relative_to(LOCAL):
        parser.error('Use the V35 report protocol and local candidate output area')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='stage1-landsat-candidate-review',
                    gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
