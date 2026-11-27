#!/usr/bin/env python3
"""Search the frozen site-09 crop for a separate radar-supported control."""

import argparse
import os
from pathlib import Path
import resource
import sys
from urllib.parse import urlencode

from check_site09_radar_control import DATA_API, ITEMS, MAX_BYTES, fetch, load_window
from inventory_unseen_validation_sites import (
    PROTOCOL as V28_PROTOCOL, RESOURCES, ROOT, checked_protocol, digest, load, require, rooted, write,
)
from prepare_july_visual_review import array_hash


REPORT = ROOT/'Docs/Reviews/ValidationStage1SearchV35/candidates/site-09'
PROTOCOL = REPORT/'radar-alternate-protocol-v36.json'
LOCAL = ROOT/'Datasets/ValidationPilot/v36/site09-radar-alternate'


def sources(protocol):
    require(protocol['schema_version'] == 36 and protocol['resources'] == RESOURCES
            and protocol['max_bytes_per_crop'] == MAX_BYTES, 'Unexpected alternate control scope')
    paths = {}
    for role, pin in protocol['input_sha256'].items():
        path = rooted(pin['path'])
        require(digest(path) == pin['sha256'], 'Pinned alternate input differs: '+role)
        paths[role] = path
    radar = load(paths['radar_metadata'])
    site_protocol = load(paths['site_protocol'])
    require(site_protocol['site']['site_id'] == 'site-09' and [row['item_id'] for row in radar['items']] == list(ITEMS),
            'Alternate control must use the same site and annual maps')
    col, row, width, height = protocol['crop_api']['native_window_col_start_row_start_width_height']
    require(0 <= col < col+width <= 4500 and 0 <= row < row+height <= 4500 and width*height <= 1600,
            'Alternate control radar crop is larger than the bounded exact site')
    return radar['items'], site_protocol, load(paths['map_snapshot'])


def crop_url(window, item):
    _, _, width, height = window['native_window_col_start_row_start_width_height']
    bounds = ','.join(str(value) for value in window['native_context_cover_wgs84'])
    query = urlencode({'collection': 'alos-fnf-mosaic', 'item': item['item_id'], 'assets': 'C',
                       'coord_crs': 'EPSG:4326', 'dst_crs': 'EPSG:4326', 'resampling': 'nearest',
                       'reproject': 'nearest', 'return_mask': 'false'})
    return f'{DATA_API}/{bounds}/{width}x{height}.tif?{query}'


def search(output, protocol, original):
    import numpy as np
    from rasterio.features import shapes
    from pyproj import Transformer
    from shapely import union_all
    from shapely.geometry import box, shape
    from shapely.ops import transform as project_geometry
    from propose_site09_radar_alternate import propose_alternate

    items, site_protocol, snapshot = sources(protocol)
    window = protocol['crop_api']
    records, annual_classes = [], []
    for item in items:
        url = crop_url(window, item)
        filename = item['item_id']+'.tif'
        path = (original or output)/filename
        if original is None:
            fetch(url, path)
        values, transform = load_window(path, protocol, item)
        annual_classes.append(values)
        records.append({'item_id': item['item_id'], 'url': url, 'filename': filename,
                        'bytes': path.stat().st_size, 'sha256': digest(path),
                        'native_classes_sha256': array_hash(values)})
    stable = np.all(np.isin(annual_classes, (1, 2)), axis=0)
    cells = [shape(geometry) for geometry, value in shapes(stable.astype('uint8'), transform=transform)
             if value == 1]
    site = site_protocol['site']
    project = Transformer.from_crs('EPSG:4326', site['grid']['crs'], always_xy=True).transform
    stable_woodland = project_geometry(project, union_all(cells)).intersection(box(*site['grid']['bounds']))
    require(stable_woodland.is_valid, 'Invalid radar geometry in the bounded crop')
    _, v28_paths, _ = checked_protocol(V28_PROTOCOL)
    older = [site['control_opportunity'], site_protocol['control']]
    result = None if stable_woodland.is_empty else propose_alternate(site, v28_paths, snapshot, stable_woodland, older)
    return {'schema_version': 36, 'site_id': site['site_id'], 'group_id': site['group_id'],
            'grid': site['grid'], 'previous_control_contexts': [control['context_bounds'] for control in older],
            'forest_on_all_three_maps_area_m2': round(stable_woodland.area, 4),
            'forest_on_all_three_maps_mask_sha256': array_hash(stable),
            'selected_control': result, 'recorded_assets': records, 'labels_approved': 0}


def worker(args, output):
    checksum = digest(PROTOCOL)
    original = args.packet.resolve() if args.packet else None
    if original:
        require(original.is_relative_to(LOCAL), 'Alternate control replay must use the local V36 packet')
    result = search(output, load(PROTOCOL), original)
    if original:
        prior = load(original/'metadata.json')
        require(prior['execution_status'] == 'completed' and prior['protocol_sha256'] == checksum
                and result == load(original/'alternate.json'), 'Alternate control replay differs')
    write(output/'alternate.json', result)
    require(digest(PROTOCOL) == checksum and 'torch' not in sys.modules, 'Protocol changed or model runtime imported')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {'schema_version': 36, 'execution_status': 'completed',
          'protocol_sha256': checksum, 'report_sha256': digest(output/'alternate.json'),
          'crop_requests': 0 if original else len(ITEMS), 'replay_verified': bool(original),
          'labels_approved': 0, 'model_executions': 0,
          'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2)})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=120)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(LOCAL):
        parser.error('Use the local V36 alternate control output area')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='site09-radar-alternate',
                    gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
