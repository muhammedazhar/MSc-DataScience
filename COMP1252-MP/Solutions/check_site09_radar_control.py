#!/usr/bin/env python3
"""Audit an exact site-09 control with bounded, native-grid annual SAR crops."""

import argparse
import os
from pathlib import Path
import resource
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from inventory_unseen_validation_sites import RESOURCES, ROOT, digest, load, require, rooted, write


REPORT = ROOT/'Docs/Reviews/ValidationStage1SearchV35/candidates/site-09'
PROTOCOL = REPORT/'radar-protocol-v36.json'
LOCAL = ROOT/'Datasets/ValidationPilot/v36/site09-radar'
DATA_API = 'https://planetarycomputer.microsoft.com/api/data/v1/item/bbox'
ITEMS = ('N51E034_17_FNF', 'N51E034_18_FNF', 'N51E034_19_FNF')
CLASSES = ('no_data', 'dense_forest', 'forest', 'non_forest', 'water')
MAX_BYTES = 65536


def inputs():
    protocol = load(PROTOCOL)
    require(protocol['schema_version'] == 36 and protocol['resources'] == RESOURCES
            and protocol['max_bytes_per_crop'] == MAX_BYTES, 'Radar control protocol has changed')
    paths = {}
    for role, pin in protocol['input_sha256'].items():
        path = rooted(pin['path'])
        require(digest(path) == pin['sha256'], 'Radar control input differs: '+role)
        paths[role] = path
    metadata = load(paths['radar_metadata'])
    control = load(paths['control_protocol'])['control']
    require(metadata['site_id'] == 'site-09' and metadata['control_crs'] == control['crs'] == 'EPSG:32636'
            and metadata['control_bounds'] == control['bounds']
            and metadata['control_context_bounds'] == control['context_bounds']
            and [item['item_id'] for item in metadata['items']] == list(ITEMS)
            and metadata['collection']['id'] == 'alos-fnf-mosaic', 'Unexpected site, geometry or annual radar roster')
    return metadata, control


def crop_request(metadata, item):
    crop = metadata['crop_api']
    width, height = crop['native_context_cover_size']
    require((width, height) == (14, 9) and item['item_id'] in ITEMS, 'Unexpected native window size or source')
    bounds = ','.join(str(value) for value in crop['native_context_cover_wgs84'])
    query = urlencode({'collection': 'alos-fnf-mosaic', 'item': item['item_id'], 'assets': 'C',
                       'coord_crs': 'EPSG:4326', 'dst_crs': 'EPSG:4326', 'resampling': 'nearest',
                       'reproject': 'nearest', 'return_mask': 'false'})
    return f'{DATA_API}/{bounds}/{width}x{height}.tif?{query}'


def fetch(url, output):
    request = Request(url, headers={'Accept': 'image/tiff', 'User-Agent': 'COMP1252-MP bounded control audit'})
    with urlopen(request, timeout=20) as response:
        require(response.status == 200 and response.geturl() == url
                and response.headers.get_content_type() in ('image/tiff', 'application/geotiff'),
                'Bounded Data API request did not return the exact GeoTIFF')
        data = response.read(MAX_BYTES+1)
    require(0 < len(data) <= MAX_BYTES, 'Radar crop exceeds the bounded byte limit')
    output.write_bytes(data)


def load_window(path, metadata, item):
    import numpy as np
    import rasterio
    from affine import Affine

    require(0 < path.stat().st_size <= MAX_BYTES, 'Invalid radar crop byte count')
    col, row, width, height = metadata['crop_api']['native_window_col_start_row_start_width_height']
    source_transform = Affine(*item['native_transform'][:6])
    expected = source_transform * Affine.translation(col, row)
    with rasterio.open(path) as data:
        require(data.driver == 'GTiff' and data.crs.to_epsg() == 4326
                and (data.width, data.height, data.count) == (width, height, 1)
                and data.dtypes == ('uint8',)
                and np.allclose(data.transform[:6], expected[:6], rtol=0, atol=1e-10),
                'Radar response does not match the exact native support')
        values = data.read(1)
    require(set(np.unique(values)) <= set(range(len(CLASSES))), 'Unexpected radar class')
    return values, expected


def measure(values, transform, control):
    import numpy as np
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform as project_geometry

    project = Transformer.from_crs('EPSG:4326', control['crs'], always_xy=True).transform
    scopes = {key: box(*control[bounds]) for key, bounds in (
        ('control', 'bounds'), ('full_context', 'context_bounds'),
    )}
    summaries = {key: {name: {'cells': 0, 'area_m2': 0.0} for name in CLASSES} for key in scopes}
    for (row, col), value in np.ndenumerate(values):
        x0, y0 = transform * (col, row)
        x1, y1 = transform * (col+1, row+1)
        polygon = project_geometry(project, box(x0, y1, x1, y0))
        for scope, area in scopes.items():
            overlap = polygon.intersection(area).area
            if overlap > 1e-7:
                record = summaries[scope][CLASSES[int(value)]]
                record['cells'] += 1
                record['area_m2'] += overlap
    for scope, classes in summaries.items():
        area = scopes[scope].area
        require(abs(sum(record['area_m2'] for record in classes.values()) - area) < 0.1,
                'Raster window does not cover the entire exact control geometry')
        for record in classes.values():
            record['area_percent'] = round(100*record['area_m2']/area, 4)
            record['area_m2'] = round(record['area_m2'], 4)
    return summaries


def audit(output, original=None):
    metadata, control = inputs()
    records = []
    for item in metadata['items']:
        url = crop_request(metadata, item)
        filename = item['item_id']+'.tif'
        path = (original or output)/filename
        if original is None:
            fetch(url, path)
        values, transform = load_window(path, metadata, item)
        records.append({'item_id': item['item_id'], 'item_metadata_url': item['stac_item_url'],
                        'capture_year': item['time_start_utc'][:4], 'request_url': url,
                        'file': filename, 'bytes': path.stat().st_size, 'sha256': digest(path),
                        'classes_native_grid': values.tolist(), 'areas': measure(values, transform, control)})
    return {'schema_version': 36, 'site_id': 'site-09', 'control': control,
            'source_crs': 'EPSG:4326', 'source_window': metadata['crop_api']['native_context_cover_size'],
            'scope': 'Area-weighted intersections of the annual radar classes with the exact frozen UTM areas.',
            'records': records, 'labels_approved': 0}


def worker(args, output):
    checksum = digest(PROTOCOL)
    original = args.packet.resolve() if args.packet else None
    if original:
        require(original.is_relative_to(LOCAL), 'Radar replay requires the local V36 packet')
    result = audit(output, original)
    if original:
        prior = load(original/'metadata.json')
        require(prior['execution_status'] == 'completed' and prior['protocol_sha256'] == checksum
                and result == load(original/'radar-control.json'),
                'Bounded radar audit differs on offline replay')
    write(output/'radar-control.json', result)
    require(digest(PROTOCOL) == checksum and 'torch' not in sys.modules, 'Protocol changed or model runtime imported')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {'schema_version': 36, 'execution_status': 'completed',
          'protocol_sha256': checksum, 'records_sha256': digest(output/'radar-control.json'),
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
        parser.error('Use the local V36 radar output area')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='site09-radar-control',
                    gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
