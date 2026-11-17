#!/usr/bin/env python3
"""Check fixed crop coverage using local directory/XML metadata only."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    require(Path(path).stat().st_size <= 32*2**20, 'Metadata size bound exceeded')
    return json.loads(Path(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def parse_tile(path):
    require(Path(path).stat().st_size <= 5*2**20, 'Tile XML exceeds5MiB')
    root = ET.parse(path).getroot()
    def local(element):
        return element.tag.rsplit('}', 1)[-1]
    def unique(elements, name):
        items = [e for e in elements if local(e) == name]
        require(len(items) == 1, 'Missing/ambiguous tile field: '+name)
        return items[0]
    crs = unique(root.iter(), 'HORIZONTAL_CS_CODE').text
    sizes = [e for e in root.iter() if local(e) == 'Size' and e.get('resolution') == '10']
    positions = [e for e in root.iter() if local(e) == 'Geoposition' and e.get('resolution') == '10']
    require(len(sizes) == len(positions) == 1, 'Unique10m grid required')
    def number(elements, name):
        value = float(unique(elements, name).text)
        require(math.isfinite(value), 'Nonfinite tile grid')
        return value
    width, height = (number(sizes[0], n) for n in ('NCOLS', 'NROWS'))
    west, north, dx, dy = (number(positions[0], n) for n in ('ULX', 'ULY', 'XDIM', 'YDIM'))
    require(width.is_integer() and height.is_integer() and width > 0 and height > 0
            and (dx, dy) == (10, -10), 'Unsupported tile grid')
    return {'crs': crs, 'width': int(width), 'height': int(height), 'ulx': west, 'uly': north,
            'bounds': [west, north-height*10, west+width*10, north]}


def relation(crop, tile):
    require(len(crop) == len(tile) == 4 and all(math.isfinite(v) for v in (*crop, *tile))
            and crop[0] < crop[2] and crop[1] < crop[3]
            and tile[0] < tile[2] and tile[1] < tile[3], 'Valid ordered bounds required')
    overlap = [max(0., min(crop[i+2], tile[i+2])-max(crop[i], tile[i])) for i in (0, 1)]
    return {'fully_covered': bool(tile[0] <= crop[0] and tile[1] <= crop[1]
            and tile[2] >= crop[2] and tile[3] >= crop[3]),
        'positive_area_overlap_m2': overlap[0]*overlap[1],
        'north_gap_m': max(0., crop[1]-tile[3]),
        'east_west_gap_m': max(0., crop[0]-tile[2], tile[0]-crop[2])}


def discover(directory):
    output = []
    for product in sorted(Path(directory).iterdir()):
        if not product.is_dir():
            continue
        if product.name.endswith('.SAFE'):
            output.append(product.resolve())
        else:
            safes = sorted(p.resolve() for p in product.glob('*.SAFE') if p.is_dir())
            require(safes or not product.name.startswith('S2'), 'Product has no SAFE metadata: '+product.name)
            output.extend(safes)
    require(len(output) == len(set(output)), 'Duplicate SAFE paths')
    return sorted(output)


def worker(args, output):
    protocol = read(args.protocol); rule = protocol['rule']
    require(protocol['schema_version'] == 6 and rule['pairs'] == 12 and rule['expected_scenes'] == 50
            and rule['qa_context_padding_m'] == 1200 and rule['fixed_crs'] == 'EPSG:32636', 'Fixed scope changed')
    require(protocol['resources'] == {'workers': 1, 'native_threads': 1, 'nice_increment': 10,
        'gpu_used': False, 'worker_rss_mib': 512, 'available_memory_reserve_gib': 1.5,
        'max_seconds_per_worker': 300}, 'Mac capacity contract changed')
    sources = {}
    for role, item in protocol['source_files'].items():
        path = (ROOT/item['path']).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == item['sha256'], 'Source changed: '+role)
        sources[role] = path
    for name, checksum in {**protocol['immutable_sources_sha256'], **protocol['implementation_sha256']}.items():
        require(sha(ROOT/'Solutions'/name) == checksum, 'Frozen implementation changed: '+name)
    inventory, prior_meta, proof = (read(sources[k]) for k in ('prior_inventory', 'prior_metadata', 'prior_verification'))
    require(proof['verification_status'] == 'passed' and proof['inventory_sha256'] == sha(sources['prior_inventory'])
            and proof['audit_metadata_sha256'] == sha(sources['prior_metadata'])
            and prior_meta['protocol_sha256'] == inventory['protocol_sha256'] == sha(sources['prior_protocol']), 'V5 proof chain changed')
    prior_protocol = read(sources['prior_protocol'])
    for item in prior_protocol['source_files'].values():
        require(sha(ROOT/item['path']) == item['sha256'], 'Inherited frozen input changed')
    plan = read(sources['planning'])
    wanted = [r for r in inventory['records'] if r['positive_interior_pixels'] >= 4
              and r['temporal_source_ids'] and not r['previously_inspected_group']]
    require(len(wanted) == 12 and len({r['pair_id'] for r in wanted}) == 12
            and sorted({r['event_id'] for r in wanted}) == sorted(rule['events']) == plan['events'], 'Fixed V5 planning set changed')
    scenes = [json.loads(line) for line in sources['scenes'].read_text().splitlines()]
    require(len(scenes) == len({s['scene_id'] for s in scenes}) == 50, 'Complete50-scene inventory required')
    paths = discover(ROOT/rule['local_sentinel_root'])
    require(paths == sorted((ROOT/s['path']).resolve() for s in scenes), 'Local SAFE inventory changed; re-inventory before coverage claims')
    xml_hashes, actual = {}, []
    for scene in scenes:
        safe = ROOT/scene['path']; tiles = list(safe.glob('GRANULE/*/MTD_TL.xml'))
        require(len(tiles) == 1, 'Unique tile XML required')
        for path, field in ((tiles[0], 'tile_metadata'), (safe/'MTD_MSIL1C.xml', 'product_metadata')):
            require(sha(path) == scene['metadata_sha256'][field], 'Local scene XML differs from frozen inventory')
            xml_hashes[str(path.relative_to(ROOT))] = sha(path)
        grid = parse_tile(tiles[0]); frozen = scene['grid_10m']
        require(grid['crs'] == scene['crs'] == 'EPSG:32636' and all(grid[k] == frozen[k]
                for k in ('width', 'height', 'ulx', 'uly', 'bounds')), 'Fresh XML grid differs from frozen inventory')
        actual.append({'scene_id': scene['scene_id'], 'date': scene['acquisition_date'], 'tile': scene['tile'],
            'scene_path': scene['path'], 'recorded_metadata_usable': scene['metadata_usable'], **grid})
    from assess_pixel_quality import CONFIG
    require(CONFIG['context_padding_m'] == 1200, 'Frozen QA halo differs')
    events, comparisons = [], []
    for event in rule['events']:
        rows = [r for r in wanted if r['event_id'] == event]
        crop = rows[0]['grid']['bounds']
        require(all(r['grid'] == rows[0]['grid'] for r in rows), 'Fixed crop differs between dates')
        context = [crop[0]-1200, crop[1]-1200, crop[2]+1200, crop[3]+1200]
        for scene in actual:
            check = relation(crop, scene['bounds']); qa = relation(context, scene['bounds'])
            col, row = (crop[0]-scene['ulx'])/10, (scene['uly']-crop[3])/10
            window_covers = 0 <= col and 0 <= row and col+64 <= scene['width'] and row+64 <= scene['height']
            require(check['fully_covered'] == window_covers, 'Bounds and independent native-window containment disagree')
            comparisons.append({'event_id': event, 'scene_id': scene['scene_id'], 'date': scene['date'],
                'tile': scene['tile'], 'crop': check, 'qa_context': qa,
                'native_window': {'column_start': col, 'row_start': row, 'width': 64, 'height': 64}})
        relevant = [c for c in comparisons if c['event_id'] == event]
        events.append({'event_id': event, 'group_id': rows[0]['group_id'], 'annotation_date': rows[0]['annotation_date'],
            'crop_grid': rows[0]['grid'], 'qa_context_bounds': context, 'original_pairs': [
                {k: r[k] for k in ('pair_id', 'pre_date', 'post_date', 'original_status', 'original_blocking_reasons')} for r in rows],
            'covering_local_scene_ids': [c['scene_id'] for c in relevant if c['crop']['fully_covered']],
            'overlapping_local_scene_ids': [c['scene_id'] for c in relevant if c['crop']['positive_area_overlap_m2'] > 0],
            'minimum_north_gap_m': min(c['crop']['north_gap_m'] for c in relevant)})
    dates = sorted({r[side+'_date'] for r in wanted for side in ('pre', 'post')})
    summary = {'fixed_events': 2, 'fixed_original_pairs': 12, 'local_scenes': 50,
        'tile_scene_counts': dict(Counter(s['tile'] for s in actual)), 'event_scene_comparisons': len(comparisons),
        'covering_comparisons': sum(c['crop']['fully_covered'] for c in comparisons),
        'positive_overlap_comparisons': sum(c['crop']['positive_area_overlap_m2'] > 0 for c in comparisons),
        'qa_context_covering_comparisons': sum(c['qa_context']['fully_covered'] for c in comparisons),
        'original_pair_dates': dates, 'scope': 'Rectangular XML coverage only; no valid-pixel, cloud or class approval.'}
    result = {'protocol_sha256': sha(args.protocol), 'summary': summary, 'events': events,
        'scenes': actual, 'comparisons': comparisons, 'xml_sha256': xml_hashes,
        'source_input_sha256': {item['path']: item['sha256'] for item in protocol['source_files'].values()},
        'conclusion': 'No local scene rectangle intersects either fixed crop. Additional correctly covering imagery or separately verified source/georeferencing correction is required; existing blocks and grids stay unchanged.' if summary['positive_overlap_comparisons'] == 0 else 'See complete coverage matrix; no automatic QA or label approval.'}
    for name, checksum in xml_hashes.items():
        require(sha(ROOT/name) == checksum, 'XML changed during check')
    require(paths == discover(ROOT/rule['local_sentinel_root']), 'Local inventory changed during check')
    write_path = output/'coverage.json';write_path.write_text(json.dumps(result, indent=2)+'\n')
    (output/'report.md').write_text('# Metadata coverage feasibility\n\n'+result['conclusion']+'\n\n'+json.dumps(summary, indent=2)+'\n')
    import resource
    metadata = {'execution_status': 'completed', 'protocol_sha256': result['protocol_sha256'], 'summary': summary,
        'implementation_sha256': sha(__file__), 'artifact_sha256': {n: sha(output/n) for n in ('coverage.json', 'report.md')},
        'satellite_pixel_reads': 0, 'qa_or_cache_reads': 0, 'model_executions': 0, 'new_imagery_reviewed': 0,
        'downloads': 0, 'labels_approved': 0, 'xml_files_checked': len(xml_hashes),
        'verification_scope': 'Fresh XML versus frozen inventory; rectangle containment independently agrees with native-pixel window bounds. Directory inventory covers the configured Sentinel-2 root only.',
        'worker_high_water_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024), 2)}
    require(not any(n in sys.modules for n in ('torch', 'rasterio', 'numpy')), 'Coverage must remain metadata-only')
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationCoverageV6/protocol.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-rss-mib', type=int, default=512)
    p.add_argument('--min-available-gib', type=float, default=1.5)
    p.add_argument('--max-seconds', type=int, default=300)
    p.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        p.error('Retain512MiB/1.5GiB/300s bounds')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10);worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='metadata-coverage', gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
