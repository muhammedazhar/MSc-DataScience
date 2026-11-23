#!/usr/bin/env python3
"""Prepare the fixed Area B historical-reference display without fetching imagery."""

import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys
from urllib.parse import parse_qs, urlsplit

from screen_repaired_validation import RESOURCES, ROOT, digest, read, require, rooted, write

PROTOCOL = ROOT/'Docs/Reviews/ValidationAreaBReferenceV22/protocol.json'
TILE_URL = ('https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/'
            'WMTS/1.0.0/default028mm/MapServer/tile/{release}/17/{row}/{col}')
TILE_SIZE = 256
WORLD_EDGE = math.pi*6378137
METRES_PER_PIXEL = 2*WORLD_EDGE/(2**17*TILE_SIZE)
WAYBACK = {11334: '2010-06-14', 44710: '2020-08-31'}


def checked_inputs():
    protocol = read(PROTOCOL)
    require(protocol['schema_version'] == 22 and protocol['resources'] == RESOURCES,
            'Area B reference protocol or resources differ')
    sources = {}
    for role, receipt in protocol['source_files'].items():
        path = rooted(receipt['path'])
        require(digest(path) == receipt['sha256'], 'Pinned reference source differs: '+role)
        sources[role] = path
    implementation = protocol['implementation_sha256']
    require(str(Path(__file__).resolve().relative_to(ROOT)) in implementation,
            'Pin the reference page runner before use')
    for name, checksum in {**protocol.get('frozen_implementation_sha256', {}), **implementation}.items():
        require(digest(rooted(name)) == checksum, 'Pinned reference code differs: '+name)
    roster = read(sources['reference_roster'])
    audit = read(sources['area_b_audit'])
    reviews = read(sources['area_b_reviews'])
    integrity = read(sources['area_b_integrity'])
    require(integrity['status'] == 'passed'
            and integrity['review_sha256'] == digest(sources['area_b_reviews'])
            and reviews['pair_id'] == audit['area']['pair_id'] == protocol['pair_id']
            and reviews['group_id'] == audit['area']['group_id'] == protocol['group_id']
            and audit['area']['grid'] == protocol['grid'], 'Area B audit or review binding differs')
    return protocol, roster, audit


def world_pixel(x, y):
    return (x+WORLD_EDGE)/METRES_PER_PIXEL, (WORLD_EDGE-y)/METRES_PER_PIXEL


def canvas_geometry(grid):
    from pyproj import Transformer

    transform = Transformer.from_crs(grid['crs'], 'EPSG:3857', always_xy=True)
    west, south, east, north = transform.transform_bounds(*grid['bounds'], densify_pts=21)
    left, top = map(math.floor, world_pixel(west, north))
    right, bottom = map(math.ceil, world_pixel(east, south))
    size = max(right-left, bottom-top)
    require(800 <= size <= 900, 'Area B map size differs from the bounded z17 footprint')
    rows = range(top//TILE_SIZE, (top+size-1)//TILE_SIZE+1)
    cols = range(left//TILE_SIZE, (left+size-1)//TILE_SIZE+1)
    require(len(rows)*len(cols) <= 25, 'Reference layer tile bound exceeded')
    return transform, left, top, size, rows, cols


def overlays(protocol, audit, transform, left, top):
    import numpy as np
    from affine import Affine
    from rasterio.features import shapes
    from check_persistence_observations import frozen_area
    from prepare_july_visual_review import array_hash

    info, masks, _ = frozen_area()
    grid = protocol['grid']
    require(info['grid'] == grid and info['pair_id'] == protocol['pair_id']
            and info['group_id'] == protocol['group_id']
            and info['mask_sha256'] == protocol['area_mask_sha256'], 'Frozen target geometry differs')

    def point(x, y):
        xx, yy = world_pixel(*transform.transform(x, y))
        return f'{xx-left:.3f},{yy-top:.3f}'

    fragments = []
    for name, colour in (('target', '#ffde60'), ('interior', '#ff965e')):
        mask = masks[name]
        for geometry, _ in shapes(mask.astype('uint8'), mask=mask, transform=Affine(*grid['transform'])):
            path = ' '.join('M'+' L'.join(point(x, y) for x, y in ring)+' Z' for ring in geometry['coordinates'])
            fragments.append(f'<path d="{path}" stroke="{colour}"/>')

    keys = ('control_id', 'row', 'col', 'width', 'roi_bounds', 'roi_sha256')
    controls = [{key: row[key] for key in keys} for row in audit['control_review_packet']['controls']]
    require(controls == protocol['control_roster'] and len(controls) == 3, 'Frozen control roster differs')
    for index, row in enumerate(controls, 1):
        start, col, width = (row[key] for key in ('row', 'col', 'width'))
        mask = np.zeros((grid['height'], grid['width']), dtype=bool)
        mask[start:start+width, col:col+width] = True
        x0, y0, x1, y1 = row['roi_bounds']
        affine = Affine(*grid['transform'])
        require(affine*(col, start+width) == (x0, y0) and affine*(col+width, start) == (x1, y1)
                and array_hash(mask) == row['roi_sha256'], 'Frozen control footprint differs')
        corners = ' '.join(point(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
        x, y = map(float, point(x0, y1).split(','))
        fragments.extend([f'<polygon points="{corners}" stroke="#61dcff"/>',
                          f'<text x="{x+6:.3f}" y="{y+19:.3f}">C{index}</text>'])
    return '\n'.join(fragments)


def landsat_image(layer, left, top):
    url = urlsplit(layer['image_url'])
    require(url.scheme == 'https' and url.hostname and url.hostname.endswith('.arcgis.com')
            and url.port is None and not url.username and not url.password and not url.fragment
            and url.path.rstrip('/').endswith('/exportImage'), 'Unexpected reference export URL')
    params = parse_qs(url.query)
    image_sr = params.get('imageSR', [])
    require(image_sr == ['3857'] or image_sr == ['102100'], 'Export image must use Web Mercator')
    if 'bounds_3857' in layer:
        west, south, east, north = layer['bounds_3857']
    else:
        require(params.get('bboxSR') in (['3857'], ['102100']) and len(params.get('bbox', [])) == 1,
                'Reference export requires explicit Web Mercator bounds')
        west, south, east, north = map(float, params['bbox'][0].split(','))
    require(all(math.isfinite(v) for v in (west, south, east, north))
            and west < east and south < north, 'Reference export bounds differ')
    x0, y0 = world_pixel(west, north)
    x1, y1 = world_pixel(east, south)
    return [{'url': layer['image_url'], 'x': x0-left, 'y': y0-top, 'width': x1-x0, 'height': y1-y0}]


def display_layers(roster, left, top, rows, cols):
    layers = roster['layers']
    require(2 <= len(layers) <= 6 and len({row['id'] for row in layers}) == len(layers),
            'Unexpected Area B reference layer roster')
    rendered = []
    releases = set()
    for layer in layers:
        date, label, kind = (layer[key] for key in ('date', 'label', 'kind'))
        if kind == 'wayback':
            release = layer['release']
            require(type(release) is int and WAYBACK.get(release) == date and release not in releases,
                    'Wayback capture or release differs from the fixed historical references')
            releases.add(release)
            imagery = [{'url': TILE_URL.format(release=release, row=row, col=col),
                        'x': col*TILE_SIZE-left, 'y': row*TILE_SIZE-top,
                        'width': TILE_SIZE, 'height': TILE_SIZE} for row in rows for col in cols]
            attribution = 'Imagery © Esri, Vantor, Earthstar Geographics, and the GIS User Community.'
            if release == 11334:
                attribution += ' Additional credits: CNES/Airbus DS, USDA FSA, USGS, Aerogrid, IGN, and IGP.'
            item_id = {11334: '2ae3908a772546828a1436025c4b14a2', 44710: '2cb162f16ead4c4e97033ced48e99635'}[release]
            source = 'https://www.arcgis.com/home/item.html?id='+item_id
        else:
            require(kind == 'landsat', 'Unexpected reference source type')
            imagery = landsat_image(layer, left, top)
            attribution = 'USGS Landsat 8 imagery via Esri. 15 m panchromatic detail and 30 m color.'
            source = layer['source']
        rendered.append({'id': layer['id'], 'label': label, 'date': date, 'kind': kind,
                         'images': imagery, 'attribution': attribution, 'source': source})
    require(releases == set(WAYBACK), 'Missing one of the two historical Wayback views')
    return rendered


def page(size, outlines, layers):
    data = json.dumps(layers, ensure_ascii=True, separators=(',', ':')).replace('<', '\\u003c')
    return ('''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Area B reference imagery</title>
<style>
body { background: #152321; color: #f1f5ef; font: 15px system-ui, sans-serif; margin: 16px; }
h1 { font-size: 21px; margin: 0 0 10px; } p { margin: 8px 0; }
select, input { font: inherit; } label { margin-right: 18px; }
a { color: #a8ecff; } .scroll { overflow: auto; }
#map { position: relative; overflow: hidden; background: #3a4744; }
#imagery, #outlines { position: absolute; inset: 0; }
#imagery img { position: absolute; margin: 0; display: block; }
#outlines { width: 100%; height: 100%; fill: none; stroke-width: 2; pointer-events: none; }
#outlines text { fill: #e8faff; paint-order: stroke; stroke: #152321; font: bold 14px system-ui; }
#outlines[hidden] { display: none; }
</style></head><body>
<h1>Area B reference imagery</h1>
<p><label>Observation <select id="observation"></select></label>
<label><input type="checkbox" id="geometry"> Show target and controls</label></p>
<p id="state" aria-live="polite"></p>
<div class="scroll">'''+f'''<div id="map" data-size="{size}" style="width:min(100%,{size}px);aspect-ratio:1" aria-label="Fixed Area B map">
<div id="imagery"></div><svg id="outlines" width="{size}" height="{size}" viewBox="0 0 {size} {size}" hidden>
{outlines}</svg></div></div>
'''+'''<p>Yellow target · Orange eroded interior · Blue fixed 16 × 16 control squares</p>
<p><span id="attribution"></span> <a id="source" target="_blank" rel="noopener noreferrer">Source</a></p>
<script>
const layers = '''+data+''';
const select = document.getElementById('observation');
const state = document.getElementById('state');
const imagery = document.getElementById('imagery');
const attribution = document.getElementById('attribution');
const source = document.getElementById('source');
const geometry = document.getElementById('geometry');
const outlines = document.getElementById('outlines');
const mapSize = Number(document.getElementById('map').dataset.size);
for (const layer of layers) select.add(new Option(layer.label, layer.id));
let epoch = 0;
function show() {
  const layer = layers.find(item => item.id === select.value);
  const shown = ++epoch;
  imagery.replaceChildren();
  attribution.textContent = layer.attribution;
  source.href = layer.source;
  let loaded = 0;
  let failed = 0;
  function update() {
    if (shown !== epoch) return;
    const total = layer.images.length;
    const pending = total - loaded - failed;
    state.textContent = layer.date + ' capture. ' + (pending ? 'Loading ' + pending + ' of ' + total + ' map images.' : 'Loaded ' + loaded + ' of ' + total + ' map images.') + (failed ? ' ' + failed + ' failed. Imagery is incomplete.' : '');
  }
  update();
  for (const entry of layer.images) {
    const img = document.createElement('img');
    img.alt = '';
    img.style.left = 100 * entry.x / mapSize + '%';
    img.style.top = 100 * entry.y / mapSize + '%';
    img.style.width = 100 * entry.width / mapSize + '%';
    img.style.height = 100 * entry.height / mapSize + '%';
    if (layer.kind === 'landsat') img.style.imageRendering = 'pixelated';
    img.onload = () => { loaded++; update(); };
    img.onerror = () => { failed++; update(); };
    img.src = entry.url;
    imagery.append(img);
  }
}
select.addEventListener('change', show);
geometry.addEventListener('change', () => outlines.toggleAttribute('hidden', !geometry.checked));
show();
</script></body></html>
''')


def worker(output):
    protocol, roster, audit = checked_inputs()
    transform, left, top, size, rows, cols = canvas_geometry(protocol['grid'])
    outlines = overlays(protocol, audit, transform, left, top)
    layers = display_layers(roster, left, top, rows, cols)
    path = output/'reference_map.html'
    path.write_text(page(size, outlines, layers), encoding='utf-8')
    require(not {'torch', 's2cloudless'} & sys.modules.keys(), 'Reference page imported a learned model')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {'execution_status': 'completed', 'protocol_sha256': digest(PROTOCOL),
          'artifact_sha256': {path.name: digest(path)}, 'pair_id': protocol['pair_id'],
          'review_layer_ids': [layer['id'] for layer in layers], 'remote_images': [len(layer['images']) for layer in layers],
          'display_zoom': 17, 'display_size': size, 'display_top_left_world_pixel': [left, top],
          'image_pixel_reads': 0, 'network_requests': 0, 'model_executions': 0, 'labels_approved': 0,
          'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2)})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='area-b-reference-page',
                    gdal_cache_mib=32, warp_memory_mib=0)


if __name__ == '__main__':
    main()
