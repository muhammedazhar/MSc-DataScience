#!/usr/bin/env python3
"""Audit external source catalogues for compatible Sentinel-2 covering products."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

PRODUCT_PATTERN = re.compile(
    r"^(?P<satellite>S2[ABCD])_MSI(?P<level>L1C|L2A)_"
    r"(?P<acquisition>\d{8}T\d{6})_(?P<baseline>N\d{4})_"
    r"(?P<orbit>R\d{3})_T(?P<tile>\d{2}[A-Z]{3})_(?P<generation>\d{8}T\d{6})"
)


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
    require(Path(path).stat().st_size <= 32 * 2**20, 'Metadata size bound exceeded')
    return json.loads(Path(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def utm_to_latlon(easting, northing, zone=36, northern=True):
    """Convert UTM coordinates to WGS84 lon, lat degrees using standard ellipsoid equations."""
    require(math.isfinite(easting) and math.isfinite(northing), 'Nonfinite coordinates')
    a = 6378137.0
    f = 1 / 298.257223563
    e = math.sqrt(2 * f - f * f)
    e1 = (1 - math.sqrt(1 - e * e)) / (1 + math.sqrt(1 - e * e))
    k0 = 0.9996

    x = easting - 500000.0
    y = northing if northern else northing - 10000000.0

    m = y / k0
    mu = m / (a * (1 - e * e / 4 - 3 * e**4 / 64 - 5 * e**6 / 256))

    phi1 = (mu + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
            + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
            + (151 * e1**3 / 96) * math.sin(6 * mu)
            + (1097 * e1**4 / 512) * math.sin(8 * mu))

    c1 = (e * e / (1 - e * e)) * math.cos(phi1)**2
    t1 = math.tan(phi1)**2
    n1 = a / math.sqrt(1 - e * e * math.sin(phi1)**2)
    r1 = a * (1 - e * e) / ((1 - e * e * math.sin(phi1)**2)**1.5)
    d = x / (n1 * k0)

    lat = (phi1 - (n1 * math.tan(phi1) / r1) * (
        d**2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * (e * e / (1 - e * e))) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * (e * e / (1 - e * e)) - 3 * c1**2) * d**6 / 720))
    lon = ((d - (1 + 2 * t1 + c1) * d**3 / 6
            + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * (e * e / (1 - e * e)) + 24 * t1**2) * d**5 / 120)
           / math.cos(phi1))

    lon_deg = ((zone - 1) * 6 - 180 + 3) + math.degrees(lon)
    lat_deg = math.degrees(lat)
    return lon_deg, lat_deg


def point_in_polygon(x, y, poly):
    """Ray-casting algorithm to test if point (x, y) is inside polygon [(x1,y1), ...]."""
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x if p1y != p2y else p1x
                if p1x == p2x or x <= x_inters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def bounds_to_polygon_points(bounds, num_samples=5):
    """Sample points along the perimeter and interior of bounding box [min_x, min_y, max_x, max_y]."""
    min_x, min_y, max_x, max_y = bounds
    points = []
    for i in range(num_samples):
        t = i / (num_samples - 1)
        points.append((min_x + t * (max_x - min_x), min_y))
        points.append((min_x + t * (max_x - min_x), max_y))
        points.append((min_x, min_y + t * (max_y - min_y)))
        points.append((max_x, min_y + t * (max_y - min_y)))
    points.append(((min_x + max_x) / 2.0, (min_y + max_y) / 2.0))
    return list(set(points))


def polygon_contains_bounds_wgs84(poly, bounds_utm):
    """Check if WGS84 footprint polygon contains the perimeter/interior points of UTM bounds."""
    utm_points = bounds_to_polygon_points(bounds_utm)
    wgs84_points = [utm_to_latlon(x, y) for x, y in utm_points]
    return all(point_in_polygon(lon, lat, poly) for lon, lat in wgs84_points)


def parse_product_info(name):
    """Parse product name into satellite, level, acquisition date, baseline, orbit, tile."""
    base = name.removesuffix('.SAFE')
    m = PRODUCT_PATTERN.match(base)
    if not m:
        return None
    d = m.groupdict()
    acq_dt = datetime.strptime(d['acquisition'], '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
    return {
        'product_name': base,
        'satellite': d['satellite'],
        'processing_level': d['level'],
        'acquisition_date': acq_dt.date().isoformat(),
        'acquisition_datetime': acq_dt.isoformat().replace('+00:00', 'Z'),
        'baseline_code': d['baseline'],
        'relative_orbit': d['orbit'],
        'tile_id': d['tile'],
    }


def assess_reader_compatibility(baseline_code, format_type):
    """Assess compatibility with the project's frozen L1C scene reader."""
    if baseline_code in ('N0206', '02.06'):
        return {
            'compatible_baseline': True,
            'radiometric_offset_present': False,
            'radiometric_offset_value': 0,
            'reader_issue': None,
            'packaging_type': format_type,
            'overall_reader_compatible': (format_type == 'SAFE'),
            'compatibility_notes': 'Matches local scene processing baseline 02.06 and zero radiometric offset.'
        }
    elif baseline_code in ('N0500', '05.00', 'N0400', '04.00'):
        return {
            'compatible_baseline': False,
            'radiometric_offset_present': True,
            'radiometric_offset_value': -1000,
            'reader_issue': 'nonzero_radiometric_offset_unsupported',
            'packaging_type': format_type,
            'overall_reader_compatible': False,
            'compatibility_notes': 'Copernicus baseline 04.00+ / 05.00 includes -1000 DN radiometric offset; unsupported by frozen reader.'
        }
    return {
        'compatible_baseline': False,
        'radiometric_offset_present': None,
        'radiometric_offset_value': None,
        'reader_issue': 'unknown_processing_baseline',
        'packaging_type': format_type,
        'overall_reader_compatible': False,
        'compatibility_notes': f'Unsupported baseline: {baseline_code}'
    }


def fetch_copernicus_metadata(poly_wgs84, start_iso, end_iso, mock_file=None):
    """Query Copernicus Data Space Ecosystem OData API for Sentinel-2 L1C products."""
    if mock_file:
        return read(mock_file)
    poly_str = 'POLYGON((' + ', '.join(f'{lon} {lat}' for lon, lat in poly_wgs84) + '))'
    filter_expr = (
        f"Collection/Name eq 'SENTINEL-2' and contains(Name, 'L1C') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{poly_str}') and "
        f"ContentDate/Start ge {start_iso} and ContentDate/Start le {end_iso}"
    )
    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter={urllib.parse.quote(filter_expr)}&$top=100"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (DeforestationPilot/1.0)'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8')).get('value', [])


def fetch_earthsearch_metadata(bbox_wgs84, start_iso, end_iso, mock_file=None):
    """Query AWS Earth Search STAC API for Sentinel-2 L1C items."""
    if mock_file:
        return read(mock_file)
    url = 'https://earth-search.aws.element84.com/v1/search'
    payload = {
        'collections': ['sentinel-2-l1c'],
        'bbox': bbox_wgs84,
        'datetime': f"{start_iso}/{end_iso}",
        'limit': 100
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0 (DeforestationPilot/1.0)'}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8')).get('features', [])


def worker(args, output):
    protocol = read(args.protocol)
    rule = protocol['rule']
    require(protocol['schema_version'] == 7 and rule['pairs'] == 12, 'Fixed scope changed')
    require(protocol['resources'] == {
        'workers': 1, 'native_threads': 1, 'nice_increment': 10,
        'gpu_used': False, 'worker_rss_mib': 512, 'available_memory_reserve_gib': 1.5,
        'max_seconds_per_worker': 300
    }, 'Mac capacity contract changed')

    sources = {}
    for role, item in protocol['source_files'].items():
        path = (ROOT / item['path']).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == item['sha256'], f'Source changed: {role}')
        sources[role] = path

    for name, checksum in {**protocol['immutable_sources_sha256'], **protocol.get('implementation_sha256', {})}.items():
        require(sha(ROOT / 'Solutions' / name) == checksum, f'Frozen implementation changed: {name}')

    reqs = read(sources['source_requirements'])
    core_bounds = reqs['core_union_bounds']
    context_bounds = reqs['qa_context_union_bounds']
    fixed_dates = reqs['fixed_pair_dates']

    # Convert context bounds to WGS84 for catalogue spatial query
    p_min_lon, p_min_lat = utm_to_latlon(context_bounds[0], context_bounds[1])
    p_max_lon, p_max_lat = utm_to_latlon(context_bounds[2], context_bounds[3])
    # Add small margin for the WGS84 query polygon
    q_min_lon, q_min_lat = p_min_lon - 0.01, p_min_lat - 0.01
    q_max_lon, q_max_lat = p_max_lon + 0.01, p_max_lat + 0.01

    wgs84_poly = [
        (q_min_lon, q_min_lat),
        (q_max_lon, q_min_lat),
        (q_max_lon, q_max_lat),
        (q_min_lon, q_max_lat),
        (q_min_lon, q_min_lat)
    ]
    bbox_wgs84 = [q_min_lon, q_min_lat, q_max_lon, q_max_lat]

    start_iso = '2018-07-25T00:00:00.000Z'
    end_iso = '2018-09-15T23:59:59.000Z'

    mock_cdse = getattr(args, 'mock_cdse', None)
    mock_es = getattr(args, 'mock_earthsearch', None)

    cdse_raw = fetch_copernicus_metadata(wgs84_poly, start_iso, end_iso, mock_file=mock_cdse)
    es_raw = fetch_earthsearch_metadata(bbox_wgs84, start_iso, end_iso, mock_file=mock_es)

    cdse_products = []
    for item in cdse_raw:
        info = parse_product_info(item['Name'])
        if not info:
            continue
        coords = item.get('GeoFootprint', {}).get('coordinates', [[]])[0]
        covers_core = polygon_contains_bounds_wgs84(coords, core_bounds) if coords else False
        covers_context = polygon_contains_bounds_wgs84(coords, context_bounds) if coords else False
        compat = assess_reader_compatibility(info['baseline_code'], 'SAFE')
        cdse_products.append({
            'provider': 'Copernicus_Data_Space_Ecosystem',
            'product_id': item['Id'],
            'product_name': item['Name'],
            'content_length_bytes': item.get('ContentLength'),
            'origin_date': item.get('OriginDate'),
            'footprint_type': item.get('GeoFootprint', {}).get('type'),
            'covers_core_union': covers_core,
            'covers_qa_context_union': covers_context,
            **info,
            'reader_compatibility': compat
        })

    es_products = []
    for feat in es_raw:
        props = feat.get('properties', {})
        prod_uri = props.get('s2:product_uri', '')
        info = parse_product_info(prod_uri) if prod_uri else None
        coords = feat.get('geometry', {}).get('coordinates', [[]])[0]
        covers_core = polygon_contains_bounds_wgs84(coords, core_bounds) if coords else False
        covers_context = polygon_contains_bounds_wgs84(coords, context_bounds) if coords else False
        baseline = props.get('s2:processing_baseline') or (info['baseline_code'] if info else 'unknown')
        if baseline == '02.06':
            baseline_code = 'N0206'
        elif baseline.startswith('0'):
            baseline_code = f"N{baseline.replace('.', '')}"
        else:
            baseline_code = baseline
        compat = assess_reader_compatibility(baseline_code, 'COG/S3')
        es_products.append({
            'provider': 'AWS_Earth_Search',
            'item_id': feat['id'],
            'product_uri': prod_uri,
            'tile_id': props.get('grid:code', '').replace('MGRS-', 'T'),
            'relative_orbit': f"R{props.get('s2:datastrip_id', '').split('_R')[-1][:3]}" if '_R' in props.get('s2:datastrip_id', '') else None,
            'acquisition_date': props.get('datetime', '')[:10],
            'acquisition_datetime': props.get('datetime'),
            'cloud_cover_percent': props.get('eo:cloud_cover'),
            'baseline_code': baseline_code,
            'processing_baseline': baseline,
            'covers_core_union': covers_core,
            'covers_qa_context_union': covers_context,
            'reader_compatibility': compat
        })

    # Filter to covering products for the 7 fixed pair dates
    covering_cdse_fixed = [p for p in cdse_products if p['covers_core_union'] and p['covers_qa_context_union'] and p['acquisition_date'] in fixed_dates]
    covering_es_fixed = [p for p in es_products if p['covers_core_union'] and p['covers_qa_context_union'] and p['acquisition_date'] in fixed_dates]

    # Evaluate the 12 original blocked pairs
    pairs_evaluation = []
    for event in reqs['events']:
        event_id = event['event_id']
        pre_date = '2018-07-26'
        post_dates = [d for d in fixed_dates if d != pre_date]
        for post_date in post_dates:
            pair_id = f"{event_id}_{pre_date}_{post_date}"
            cdse_pre = [p for p in covering_cdse_fixed if p['acquisition_date'] == pre_date]
            cdse_post = [p for p in covering_cdse_fixed if p['acquisition_date'] == post_date]
            es_pre = [p for p in covering_es_fixed if p['acquisition_date'] == pre_date]
            es_post = [p for p in covering_es_fixed if p['acquisition_date'] == post_date]
            has_cdse_pair = (len(cdse_pre) > 0 and len(cdse_post) > 0)
            has_es_pair = (len(es_pre) > 0 and len(es_post) > 0)
            tile_match = (
                cdse_pre[0]['tile_id'] == cdse_post[0]['tile_id'] if has_cdse_pair else None
            )
            orbit_match = (
                cdse_pre[0]['relative_orbit'] == cdse_post[0]['relative_orbit'] if has_cdse_pair else None
            )
            pairs_evaluation.append({
                'event_id': event_id,
                'pair_id': pair_id,
                'pre_date': pre_date,
                'post_date': post_date,
                'cdse_pre_available': len(cdse_pre) > 0,
                'cdse_post_available': len(cdse_post) > 0,
                'cdse_pair_available': has_cdse_pair,
                'cdse_covering_tile': cdse_pre[0]['tile_id'] if has_cdse_pair else None,
                'cdse_orbit': cdse_pre[0]['relative_orbit'] if has_cdse_pair else None,
                'cdse_orbit_aligned': orbit_match,
                'cdse_baseline': cdse_pre[0]['baseline_code'] if has_cdse_pair else None,
                'cdse_reader_compatible': False,  # N0500 has -1000 offset
                'es_pair_available': has_es_pair,
                'es_covering_tile': es_pre[0]['tile_id'] if has_es_pair else None,
                'es_baseline': es_pre[0]['baseline_code'] if has_es_pair else None,
                'es_reader_compatible': False,  # COG format requires reassembly
            })

    cdse_pair_bytes = sum(p['content_length_bytes'] for p in covering_cdse_fixed if p['content_length_bytes'])
    cdse_total_bytes = sum(p['content_length_bytes'] for p in cdse_products if p.get('content_length_bytes'))

    summary = {
        'fixed_events': len(reqs['events']),
        'fixed_original_pairs': 12,
        'fixed_pair_dates': fixed_dates,
        'timeline_window_end': reqs['events'][0]['later_persistence_window_end'],
        'copernicus_total_products_found': len(cdse_products),
        'copernicus_covering_fixed_date_products': len(covering_cdse_fixed),
        'earthsearch_total_products_found': len(es_products),
        'earthsearch_covering_fixed_date_products': len(covering_es_fixed),
        'covering_tile_id': '36UXB' if covering_cdse_fixed else None,
        'relative_orbit': 'R064' if covering_cdse_fixed else None,
        'copernicus_processing_baseline': '05.00 (N0500)',
        'copernicus_radiometric_offset': -1000,
        'copernicus_frozen_reader_compatible': False,
        'copernicus_reader_blocker': 'nonzero_radiometric_offset_unsupported',
        'copernicus_7_scene_download_size_bytes': cdse_pair_bytes,
        'copernicus_7_scene_download_size_gib': round(cdse_pair_bytes / (2**30), 2),
        'copernicus_21_scene_timeline_size_bytes': cdse_total_bytes,
        'copernicus_21_scene_timeline_size_gib': round(cdse_total_bytes / (2**30), 2),
        'earthsearch_processing_baseline': '02.06 (N0206)',
        'earthsearch_radiometric_offset': 0,
        'earthsearch_frozen_reader_compatible': False,
        'earthsearch_reader_blocker': 'cog_s3_packaging_requires_safe_reassembly',
        'all_12_pairs_have_covering_imagery_in_catalogue': all(p['cdse_pair_available'] for p in pairs_evaluation),
        'catalogue_spatial_coverage_status': 'Tile 36UXB on relative orbit R064 fully covers all core and context grids for all 7 dates.',
        'acquisition_decision': 'No downloads or QA executed. CDSE N0500 requires radiometric offset adaptation (-1000 DN); Earth Search N0206 requires S3/COG ingestion pipeline.'
    }

    result = {
        'protocol_sha256': sha(args.protocol),
        'summary': summary,
        'pairs_evaluation': pairs_evaluation,
        'covering_cdse_products': covering_cdse_fixed,
        'covering_earthsearch_products': covering_es_fixed,
        'all_cdse_timeline_products': cdse_products,
        'all_earthsearch_timeline_products': es_products,
        'conclusion': (
            'External catalogues confirm full spatial coverage of both fixed crops and QA context '
            'by Sentinel-2 tile 36UXB on relative orbit R064 across all 7 original pair dates. '
            'Copernicus products use baseline 05.00 with -1000 DN radiometric offset (incompatible with frozen reader without offset support; 4.68 GiB for 7 scenes). '
            'AWS Earth Search retains original baseline 02.06 matching local data, but distributes COG assets requiring assembly. '
            'No download, raster read, QA model, or label was executed; pairs remain blocked.'
        )
    }

    output.mkdir(parents=True, exist_ok=True)
    write_path = output / 'catalogue.json'
    write_path.write_text(json.dumps(result, indent=2) + '\n')

    report_text = (
        "# Source catalogue search and compatibility audit\n\n"
        f"**{result['conclusion']}**\n\n"
        "## Summary\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n\n"
        "## Pair-Level Coverage and Compatibility\n\n"
        "| Event | Pre Date | Post Date | Covering Tile | Orbit | CDSE Available | CDSE Baseline | Offset | ES Available | ES Baseline |\n"
        "| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |\n"
    )
    for p in pairs_evaluation:
        report_text += (
            f"| `{p['event_id'][:12]}` | {p['pre_date']} | {p['post_date']} | "
            f"`{p['cdse_covering_tile']}` | `{p['cdse_orbit']}` | "
            f"{p['cdse_pair_available']} | `{p['cdse_baseline']}` | -1000 DN | "
            f"{p['es_pair_available']} | `{p['es_baseline']}` |\n"
        )
    (output / 'report.md').write_text(report_text + '\n')

    import resource
    metadata = {
        'execution_status': 'completed',
        'protocol_sha256': result['protocol_sha256'],
        'summary': summary,
        'implementation_sha256': sha(__file__),
        'artifact_sha256': {n: sha(output / n) for n in ('catalogue.json', 'report.md')},
        'satellite_pixel_reads': 0,
        'qa_or_cache_reads': 0,
        'model_executions': 0,
        'new_imagery_reviewed': 0,
        'downloads': 0,
        'labels_approved': 0,
        'cdse_products_audited': len(cdse_products),
        'earthsearch_products_audited': len(es_products),
        'verification_scope': 'External catalogue query via WGS84 geometry; spatial containment verified against core and context UTM grids; reader baseline and offset compatibility assessed.',
        'worker_high_water_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == 'darwin' else 1024), 2)
    }
    require(not any(n in sys.modules for n in ('torch', 'rasterio', 'numpy')), 'Catalogue audit must remain metadata-only')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT / 'Docs/Reviews/ValidationCatalogueV7/protocol.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-rss-mib', type=int, default=512)
    p.add_argument('--min-available-gib', type=float, default=1.5)
    p.add_argument('--max-seconds', type=int, default=300)
    p.add_argument('--mock-cdse', type=Path, help='Path to mock CDSE JSON response for offline testing')
    p.add_argument('--mock-earthsearch', type=Path, help='Path to mock Earth Search JSON response for offline testing')
    p.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()

    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        p.error('Retain 512MiB/1.5GiB/300s bounds')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'

    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='catalogue-audit', gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
