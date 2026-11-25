#!/usr/bin/env python3
"""Prepare and replay two bounded Landsat reference views for site-01."""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from inventory_unseen_validation_sites import RESOURCES, ROOT, digest, load, require, rooted, write
from preflight_fresh_validation_sources import checked_artifacts, evidence_hashes
from prepare_july_visual_review import array_hash, board, outline


PROTOCOL = ROOT/'Docs/Reviews/ValidationSite01ReferenceV33/protocol.json'
LOCAL = ROOT/'Datasets/ValidationPilot/v33'
SITE = 'site-01'
MAX_BYTES = 65536
PREVIEW_SIZE = 43
SCALE = 8
DEADLINE_SECONDS = 20
SERVICE = 'https://landsat2.arcgis.com/arcgis/rest/services/Landsat8_PanSharpened/ImageServer/exportImage'
VIEW_NAMES = ('landsat-plain.png', 'landsat-outlined.png')


def pinned_inputs():
    protocol = load(PROTOCOL)
    require(protocol['schema_version'] == 33 and protocol['site_id'] == SITE
            and protocol['resources'] == RESOURCES and protocol['max_bytes_per_image'] == MAX_BYTES
            and protocol['preview_size_pixels'] == [PREVIEW_SIZE]*2 and protocol['view_order'] == list(VIEW_NAMES),
            'V33 reference protocol differs')
    paths = {}
    for key, receipt in protocol['input_sha256'].items():
        path = rooted(receipt['path'])
        require(digest(path) == receipt['sha256'], 'V33 pinned input changed: '+key)
        paths[key] = path
    selection = load(paths['selection'])
    site = selection['sites'][SITE]
    nomination = selection['nomination']
    require(nomination['site_id'] == SITE and nomination['fixed_pair'] == protocol['sentinel_pair']
            and nomination['fixed_alternative_control'] == protocol['control']
            and site['grid'] == protocol['grid'], 'V32 nomination differs')
    plan_site, = [row for row in load(paths['v28_plan'])['sites'] if row['site_id'] == SITE]
    require(all(site[key] == plan_site[key] for key in ('event_id', 'group_id', 'grid', 'qa_context_bounds'))
            and plan_site['positive_interior_sha256'] == site['eroded_target_sha256'],
            'V28 target and V32 nomination differ')
    source_rows = {row['scene_id']: row for row in load(paths['access'])['landsat_8']['records']}
    sources = [source_rows[key] for key in nomination['reference_scene_ids']]
    require([row['scene_id'] for row in sources] == protocol['reference_scene_ids']
            and [row['esri_raster_id'] for row in sources] == protocol['esri_raster_ids'], 'Reference roster differs')
    return protocol, paths, plan_site, sources


def reference_url(source, grid):
    metadata = source['locked_preview_metadata']
    url = urlsplit(metadata['url'])
    query = parse_qs(url.query)
    require(urlunsplit((url.scheme, url.netloc, url.path, '', '')) == SERVICE
            and not url.fragment and query['bboxSR'] == query['imageSR'] == ['32636']
            and query['size'] == ['43,43'] and query['f'] == ['json']
            and query['format'] == ['png24'] and query['interpolation'] == ['RSP_NearestNeighbor']
            and [float(value) for value in query['bbox'][0].split(',')] == metadata['bounds_epsg32636'] == grid['bounds']
            and json.loads(query['mosaicRule'][0]) == {
                'mosaicMethod': 'esriMosaicLockRaster', 'lockRasterIds': [source['esri_raster_id']],
                'mosaicOperation': 'MT_FIRST',
            }, 'Reference request is outside the fixed scene and crop')
    require(url.query.endswith('&f=json'), 'Unexpected ArcGIS request parameters')
    return urlunsplit(url._replace(query=url.query.removesuffix('&f=json')+'&f=image'))


def fetch_crop(url, path):
    request = Request(url, headers={'Accept': 'image/png', 'User-Agent': 'COMP1252-MP bounded reference review'})
    with urlopen(request, timeout=DEADLINE_SECONDS) as response:
        content_type = response.headers.get_content_type()
        require(response.status == 200 and response.geturl() == url and content_type == 'image/png',
                'Scene-locked export did not return the requested PNG')
        data = response.read(MAX_BYTES+1)
    require(0 < len(data) <= MAX_BYTES, 'Reference preview exceeds the fixed byte limit')
    path.write_bytes(data)


def read_crop(path):
    from PIL import Image

    require(0 < path.stat().st_size <= MAX_BYTES, 'Invalid reference preview size')
    with Image.open(BytesIO(path.read_bytes()), formats=['PNG']) as image:
        require(image.size == (PREVIEW_SIZE, PREVIEW_SIZE) and image.mode == 'RGB' and not image.is_animated,
                'Unexpected reference preview format')
        return image.copy()


def crop_masks(paths, site, control):
    import numpy as np
    from pyproj import Transformer
    from scipy.ndimage import binary_erosion
    from shapely.geometry import shape
    from shapely.ops import transform
    from assess_pixel_quality import core_grid
    from validate_sample import burn

    geodata = load(paths['catalogue'])
    event_ids = [site['event_id'], *site['prior_temporal_context_event_ids']]
    matches = {f['id']: f for f in geodata['features'] if f['id'] in event_ids}
    require(len(matches) == len(event_ids), 'Pinned anchor and earlier annotations are required')
    project = Transformer.from_crs('EPSG:4326', site['grid']['crs'], always_xy=True).transform
    anchor, *earlier = [transform(project, shape(matches[key]['geometry'])) for key in event_ids]
    grid = core_grid(site['grid'])
    prior = burn(earlier, grid).astype(bool)
    target = burn([anchor], grid).astype(bool) & ~prior
    interior = binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
    require(int(target.sum()) == site['positive_target_pixels'] == 237
            and int(interior.sum()) == site['positive_interior_pixels'] == 151
            and array_hash(interior) == site['positive_interior_sha256'], 'V28 support masks differ')
    row, col, width, radius = (control[key] for key in ('row', 'col', 'width', 'context_radius_pixels'))
    require((row, col, width, radius) == (44, 41, 16, 2), 'V32 fixed control differs')
    roi, context = np.zeros_like(target), np.zeros_like(target)
    roi[row:row+width, col:col+width] = True
    context[row-radius:row+width+radius, col-radius:col+width+radius] = True
    require(int(roi.sum()) == control['pixels'] == 256 and int(context.sum()) == control['context_pixels'] == 400
            and not (context & (target | prior)).any(), 'V32 fixed control support differs')
    return {'earlier_annotation': prior, 'target': target, 'control_context': context, 'control': roi}


def build_packet(output, original=None):
    from PIL import Image, ImageDraw

    protocol, paths, site, sources = pinned_inputs()
    masks = crop_masks(paths, site, protocol['control'])
    plain, outlined, receipts = [], [], []
    for index, source in enumerate(sources):
        url = reference_url(source, protocol['grid'])
        date = source['capture_time_utc'].split('T')[0]
        name = f'landsat-{date}.png'
        path = (original or output)/name
        if original is None:
            fetch_crop(url, path)
        preview = read_crop(path)
        view = preview.resize((64*SCALE, 64*SCALE), Image.Resampling.NEAREST)
        marked = view.copy()
        drawing = ImageDraw.Draw(marked)
        for key, colour in (('earlier_annotation', '#ff7ac8'), ('target', '#ffdb63'),
                            ('control_context', '#70eff3'), ('control', '#ffffff')):
            outline(drawing, masks[key], SCALE, colour)
        label = ('12 January 2018', '20 February 2018')[index]
        plain.append((label, view))
        outlined.append((label, marked))
        receipts.append({
            'scene_id': source['scene_id'], 'capture_time_utc': source['capture_time_utc'],
            'source_url': url, 'usgs_item_url': source['usgs_stac_receipt']['url'],
            'scene_cloud_percent': source['usgs_scene_cloud_percent'], 'filename': name,
            'bytes': path.stat().st_size, 'sha256': digest(path),
        })
    subtitle = 'Landsat 8 pan-sharpened RGB | 43 x 43 source render | North up'
    board(output/VIEW_NAMES[0], 'Site-01 | Dated reference crops', subtitle, plain,
          'USGS Landsat 8 via Esri. Native pan sampling 15 m, multispectral 30 m.')
    board(output/VIEW_NAMES[1], 'Site-01 | Fixed target and control', subtitle, outlined,
          'Yellow target, pink prior annotations, white control, cyan control context.')
    write(output/'packet.json', {
        'schema_version': 33, 'site_id': SITE, 'group_id': site['group_id'], 'event_id': site['event_id'],
        'grid': protocol['grid'], 'control': protocol['control'], 'sentinel_pair': protocol['sentinel_pair'],
        'source_preview_size_pixels': [43, 43], 'display_resampling': 'nearest',
        'mask_sha256': {key: array_hash(mask) for key, mask in masks.items()},
        'receipts': receipts, 'view_order': list(VIEW_NAMES),
        'boards_sha256': {name: digest(output/name) for name in VIEW_NAMES},
    })


def worker(args, output):
    checksum = digest(PROTOCOL)
    if args.mode == 'prepare':
        build_packet(output)
        details = {'image_requests': 2, 'artifact_sha256': evidence_hashes(output)}
    else:
        original = args.packet.resolve()
        require(original.is_relative_to(LOCAL), 'Reference packet must be in the local V33 output area')
        metadata = load(original/'metadata.json')
        require(metadata['execution_status'] == 'completed' and metadata['mode'] == 'prepare'
                and metadata['protocol_sha256'] == checksum and metadata['artifact_sha256'] == evidence_hashes(original),
                'Prepared reference packet binding differs')
        checked_artifacts(original, metadata['artifact_sha256'])
        with TemporaryDirectory(dir=output) as temporary:
            replay = Path(temporary)
            build_packet(replay, original)
            require(all(digest(path) == metadata['artifact_sha256'][path.name] for path in replay.iterdir()),
                    'Reference board or receipt replay differs')
        details = {'image_requests': 0, 'verified_packet_metadata_sha256': digest(original/'metadata.json'),
                   'verified_artifact_sha256': metadata['artifact_sha256'], 'byte_identical_replay': True}
    require(checksum == digest(PROTOCOL) and 'torch' not in sys.modules, 'Protocol changed or model runtime imported')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'schema_version': 33, 'mode': args.mode, 'execution_status': 'completed', 'protocol_sha256': checksum,
        'sentinel_jp2_reads': 0, 'model_executions': 0, 'labels_approved': 0,
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2), **details,
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
        parser.error('Output must be in the local V33 output area')
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='site01-landsat-reference',
                    gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
