#!/usr/bin/env python3
"""Bounded metadata inventory only; never acquire imagery or approve sources."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = 'https://earth-search.aws.element84.com/v1/search'
KNOWN_CLOUDY = 'S2B_MSIL1C_20180726T084009_N0206_R064_T36UXB_20180726T104227'
RESOURCES = {'workers': 1, 'native_threads': 1, 'nice_increment': 10,
             'gpu_used': False, 'worker_rss_mib': 512,
             'available_memory_reserve_gib': 1.5, 'max_seconds_per_worker': 300}
PRODUCT = re.compile(r'(?P<satellite>S2[AB])_MSIL1C_(?P<acquisition>\d{8}T\d{6})_'
                     r'(?P<baseline>N\d{4})_(?P<orbit>R\d{3})_T(?P<tile>\d{2}[A-Z]{3})_'
                     r'(?P<generation>\d{8}T\d{6})(?:\.SAFE)?')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return sha(Path(path).read_bytes())


def read_json(data):
    return json.loads(data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def rooted(name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'Relative source path required')
    path = (ROOT/name).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.is_file(), 'Missing or escaping source')
    return path


def check_inputs(protocol):
    require(protocol['schema_version'] == 12 and protocol['resources'] == RESOURCES,
            'V12 scope/resource declaration differs')
    q = protocol['query']
    expected = {'url': ENDPOINT, 'collection': 'sentinel-2-l1c',
                'start_inclusive': '2018-07-01T00:00:00Z', 'end_exclusive': '2018-07-31T00:00:00Z',
                'stac_datetime': '2018-07-01T00:00:00Z/2018-07-30T23:59:59.999999Z',
                'page_limit': 100, 'maximum_pages': 10, 'maximum_response_bytes': 2097152,
                'timeout_seconds': 20, 'target_tile': '36UXB', 'relative_orbit': 'R064',
                'post_date': '2018-07-31', 'known_cloudy_date': '2018-07-26',
                'minimum_interval_days': 5, 'maximum_interval_days': 30,
                'nominal_followup_baseline': 'N0206', 'source_crs': 'EPSG:32636',
                'catalogue_cloud_percentage_used_for_selection': False, 'select_roster': False}
    require(all(q.get(k) == v for k, v in expected.items()), 'Declared query policy differs')
    receipts = dict(protocol['source_files'])
    for name, checksum in protocol['immutable_sources_sha256'].items():
        receipts['frozen/'+name] = {'path': 'Solutions/'+name, 'sha256': checksum}
    for name, checksum in protocol.get('implementation_sha256', {}).items():
        receipts['implementation/'+name] = {'path': name, 'sha256': checksum}
    require('Solutions/inventory_earlier_validation_sources.py' in protocol.get('implementation_sha256', {}),
            'Freeze inventory implementation before operational query')
    for receipt in receipts.values():
        require(digest(rooted(receipt['path'])) == receipt['sha256'], 'Frozen input changed: '+receipt['path'])
    requirements = read_json(rooted(receipts['requirements']['path']).read_bytes())
    require(requirements['group_id'] == 'group-96c27871459d03c08bcc'
            and requirements['crs'] == q['source_crs']
            and all(requirements[k] == q[k] for k in ('core_union_bounds', 'qa_context_union_bounds')),
            'Fixed group/context changed')
    return requirements, receipts


def initial_request(requirements, query):
    from pyproj import Transformer

    transform = Transformer.from_crs(requirements['crs'], 'EPSG:4326', always_xy=True)
    bbox = list(transform.transform_bounds(*requirements['qa_context_union_bounds'], densify_pts=21))
    require(all(math.isfinite(v) for v in bbox), 'Nonfinite request bounds')
    return {'url': query['url'], 'method': 'POST', 'body': {
        'collections': [query['collection']], 'datetime': query['stac_datetime'],
        'bbox': bbox, 'limit': query['page_limit']}}


def trusted(url):
    parsed = urlsplit(url)
    require(parsed.scheme == 'https' and parsed.hostname == 'earth-search.aws.element84.com'
            and parsed.port in (None, 443) and parsed.path == '/v1/search'
            and parsed.username is None and parsed.password is None and not parsed.fragment,
            'Untrusted catalogue pagination URL')


def next_request(page, current):
    links = page.get('links')
    require(isinstance(links, list) and all(isinstance(v, dict) for v in links), 'Malformed STAC links')
    following = [link for link in links if link.get('rel') == 'next']
    require(len(following) <= 1, 'Ambiguous next page')
    if not following:
        return None
    link = following[0]
    require(isinstance(link.get('href'), str) and link['href'], 'Missing next URL')
    url = urljoin(current['url'], link['href']); trusted(url)
    method = link.get('method', 'GET').upper()
    require(method in ('GET', 'POST'), 'Unsupported pagination method')
    merge = link.get('merge', False)
    require(type(merge) is bool, 'Malformed pagination merge flag')
    body = link.get('body')
    if method == 'POST':
        require(isinstance(body, dict), 'POST pagination body required')
        if merge:
            require(isinstance(current['body'], dict), 'Cannot merge a missing request body')
            body = {**current['body'], **body}
    else:
        require(body is None and not merge, 'Unexpected GET pagination body')
    return {'url': url, 'method': method, 'body': body}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Catalogue redirect refused; retain trusted endpoint')


def fetch(request, query):
    trusted(request['url'])
    data = canonical(request['body']) if request['body'] is not None else None
    req = Request(request['url'], data=data, method=request['method'],
                  headers={'Content-Type': 'application/json', 'Accept': 'application/geo+json, application/json',
                           'User-Agent': 'COMP1252-bounded-metadata-inventory/12'})
    with build_opener(NoRedirect()).open(req, timeout=query['timeout_seconds']) as response:
        trusted(response.geturl())
        raw = response.read(query['maximum_response_bytes']+1)
        require(response.status == 200 and len(raw) <= query['maximum_response_bytes'],
                'Catalogue response failed or exceeded byte cap')
        return raw


def collect(request, query, output, fetcher=fetch):
    """Keep raw evidence, every distinct item, and exact duplicate occurrences."""
    raw_dir = output/'raw'; raw_dir.mkdir()
    seen_requests = set(); items = {}; receipts = []; totals = set()
    while request is not None:
        require(len(receipts) < query['maximum_pages'], 'Pagination incomplete at page cap')
        fingerprint = sha(canonical(request))
        require(fingerprint not in seen_requests, 'Pagination loop')
        seen_requests.add(fingerprint)
        raw = fetcher(request, query)
        require(isinstance(raw, bytes) and len(raw) <= query['maximum_response_bytes'], 'Response exceeds cap')
        index = len(receipts)+1
        name = f'raw/page-{index:02d}.json'; (output/name).write_bytes(raw)
        write(output/f'raw/request-{index:02d}.json', request)
        page = read_json(raw)
        features = page.get('features')
        require(page.get('type') == 'FeatureCollection' and isinstance(features, list)
                and len(features) <= query['page_limit'], 'Malformed/oversized STAC page')
        context = page.get('context', {})
        require(isinstance(context, dict), 'Malformed result context')
        for returned in (page.get('numberReturned'), context.get('returned')):
            require(returned is None or type(returned) is int and returned == len(features), 'Returned count mismatch')
        for total in (page.get('numberMatched'), context.get('matched')):
            if total is not None:
                require(type(total) is int and total >= 0, 'Malformed matched count')
                totals.add(total)
        for feature in features:
            require(isinstance(feature, dict) and feature.get('type') == 'Feature'
                    and isinstance(feature.get('id'), str) and feature['id']
                    and isinstance(feature.get('collection'), str), 'Missing item identity')
            key = (feature['collection'], feature['id'])
            checksum = sha(canonical(feature))
            if key in items:
                require(items[key]['sha256'] == checksum, 'Conflicting duplicate item identity')
                items[key]['pages'].append(index)
            else:
                items[key] = {'feature': feature, 'sha256': checksum, 'pages': [index]}
        receipts.append({'response': name, 'sha256': sha(raw), 'bytes': len(raw),
                         'request': request, 'returned_items': len(features)})
        request = next_request(page, request)
    require(len(totals) <= 1 and (not totals or len(items) == next(iter(totals))),
            'Incomplete or inconsistent catalogue matched count')
    return [items[key] for key in sorted(items)], receipts, next(iter(totals), None)


def coverage(geometry, requirements):
    """Test full projected geometry, including holes; never repair invalid input."""
    from pyproj import Transformer
    from pyproj.exceptions import ProjError
    from shapely.errors import ShapelyError
    from shapely.geometry import box, shape
    from shapely.ops import transform

    unknown = {'core': None, 'qa_context': None}
    if not geometry:
        return unknown, 'missing_footprint'
    try:
        polygon = shape(geometry)
        require(polygon.geom_type in ('Polygon', 'MultiPolygon') and not polygon.is_empty
                and polygon.is_valid and all(math.isfinite(v) for v in polygon.bounds), 'Invalid geographic footprint')
        require(-180 <= polygon.bounds[0] <= polygon.bounds[2] <= 180
                and -90 <= polygon.bounds[1] <= polygon.bounds[3] <= 90, 'Invalid geographic coordinates')
        project = Transformer.from_crs('EPSG:4326', requirements['crs'], always_xy=True)
        projected = transform(project.transform, polygon)
        require(projected.is_valid and not projected.is_empty
                and all(math.isfinite(v) for v in projected.bounds), 'Invalid projected footprint')
        return {'core': bool(projected.covers(box(*requirements['core_union_bounds']))),
                'qa_context': bool(projected.covers(box(*requirements['qa_context_union_bounds'])))}, None
    except (ValueError, TypeError, KeyError, IndexError, ProjError, ShapelyError):
        return unknown, 'invalid_footprint'


def timestamp(value):
    require(isinstance(value, str), 'Missing datetime')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'Datetime needs timezone')
    return result.astimezone(timezone.utc)


def normalize(item, query, requirements):
    feature = item['feature']; props = feature.get('properties', {})
    require(isinstance(props, dict), 'Malformed item properties')
    issues = []; uri = props.get('s2:product_uri'); parsed = None; day = None
    match = PRODUCT.fullmatch(uri) if isinstance(uri, str) else None
    if match:
        parsed = match.groupdict()
        try:
            day = datetime.strptime(parsed['acquisition'], '%Y%m%dT%H%M%S').date()
            datetime.strptime(parsed['generation'], '%Y%m%dT%H%M%S')
        except ValueError:
            issues.append('invalid_product_timestamp'); day = None
    else:
        issues.append('missing_or_invalid_product_uri')
    try:
        observed = timestamp(props.get('datetime'))
        if day and observed.date() != day:
            issues.append('product_and_item_date_conflict')
        if not timestamp(query['start_inclusive']) <= observed < timestamp(query['end_exclusive']):
            issues.append('item_datetime_outside_query')
    except (ValueError, TypeError):
        issues.append('missing_or_invalid_item_datetime')
    if feature['collection'] != query['collection']:
        issues.append('collection_conflict')
    tiles = [v for v in (props.get('grid:code'), props.get('s2:mgrs_tile')) if v is not None]
    components = [props.get(k) for k in ('mgrs:utm_zone', 'mgrs:latitude_band', 'mgrs:grid_square')]
    if any(v is not None for v in components):
        if (type(components[0]) is int and 1 <= components[0] <= 60
                and isinstance(components[1], str) and re.fullmatch('[A-Z]', components[1])
                and isinstance(components[2], str) and re.fullmatch('[A-Z]{2}', components[2])):
            tiles.append(f'{components[0]:02d}'+components[1]+components[2])
        else:
            issues.append('incomplete_or_invalid_mgrs_components')
    if not tiles:
        issues.append('missing_provider_tile')
    for value in tiles:
        tile = value.removeprefix('MGRS-') if isinstance(value, str) else None
        if parsed and tile != parsed['tile']:
            issues.append('provider_tile_conflict')
    baseline = props.get('s2:processing_baseline')
    normalized_baseline = 'N'+baseline.replace('.', '') if isinstance(baseline, str) and re.fullmatch(r'\d{2}\.\d{2}', baseline) else baseline
    if baseline is None:
        issues.append('missing_provider_baseline')
    elif parsed and normalized_baseline != parsed['baseline']:
        issues.append('provider_baseline_conflict')
    orbit = props.get('sat:relative_orbit')
    if orbit is not None and parsed:
        expected_orbit = int(parsed['orbit'][1:])
        if type(orbit) is not int or orbit != expected_orbit:
            issues.append('provider_orbit_conflict')
    platform = props.get('platform')
    if platform is not None and parsed and platform != 'sentinel-2'+parsed['satellite'][-1].lower():
        issues.append('provider_platform_conflict')
    covered, geometry_issue = coverage(feature.get('geometry'), requirements)
    if geometry_issue:
        issues.append(geometry_issue)
    interval = (datetime.fromisoformat(query['post_date']).date()-day).days if day else None
    blockers = list(issues)
    if parsed:
        for key, wanted in (('tile', query['target_tile']), ('orbit', query['relative_orbit']),
                            ('baseline', query['nominal_followup_baseline'])):
            if parsed[key] != wanted:
                blockers.append('different_'+key)
    if interval is None or not query['minimum_interval_days'] <= interval <= query['maximum_interval_days']:
        blockers.append('outside_forward_interval')
    known_day = day is not None and day.isoformat() == query['known_cloudy_date']
    if day is None or not query['start_inclusive'][:10] <= day.isoformat() < query['known_cloudy_date']:
        blockers.append('not_a_fresh_earlier_date')
    if covered['core'] is not True or covered['qa_context'] is not True:
        blockers.append('full_catalogue_footprint_coverage_not_established')
    exact_cloudy = isinstance(uri, str) and uri.removesuffix('.SAFE') == KNOWN_CLOUDY
    cloud = props.get('eo:cloud_cover')
    # Keep supplied offsets verbatim as evidence, without interpreting missing values as zero.
    offsets = {k: v for k, v in props.items() if 'offset' in k.lower()}
    asset_offsets = {}
    assets = feature.get('assets', {})
    if not isinstance(assets, dict):
        issues.append('malformed_asset_metadata'); blockers.append('malformed_asset_metadata'); assets = {}
    for name, asset in assets.items():
        if isinstance(asset, dict):
            bands = asset.get('raster:bands', [])
            if not isinstance(bands, list):
                issues.append('malformed_asset_band_metadata'); blockers.append('malformed_asset_band_metadata'); continue
            values = [{'band_index': i, 'offset': band['offset']} for i, band in enumerate(bands)
                      if isinstance(band, dict) and 'offset' in band]
            if values:
                asset_offsets[name] = values
    return {'collection': feature['collection'], 'item_id': feature['id'], 'product_uri': uri,
            'product_identity': parsed, 'acquisition_date': day.isoformat() if day else None,
            'provider_datetime': props.get('datetime'), 'provider_tile_fields': tiles,
            'provider_baseline': baseline, 'provider_relative_orbit': orbit,
            'provider_platform': platform, 'catalogue_cloud_percentage': cloud,
            'catalogue_cloud_percentage_used_for_selection': False,
            'supplied_offset_properties': offsets or None, 'supplied_asset_band_offsets': asset_offsets or None,
            'reader_compatibility': 'unverified',
            'source_approved': False, 'roster_selected': False, 'interval_to_post_days': interval,
            'known_cloudy_acquisition_date': known_day,
            'own_qa_status': 'excluded_by_V11_automatic_quality_screen' if exact_cloudy else 'unexamined',
            'catalogue_footprint_covers': covered, 'issues': sorted(set(issues)),
            'nominal_metadata_followup_eligible': not blockers, 'followup_blockers': sorted(set(blockers)),
            'raw_item_sha256': item['sha256'], 'occurrence_pages': item['pages']}


def summarize(rows):
    eligible = [row for row in rows if row['nominal_metadata_followup_eligible']]
    return {'distinct_items': len(rows), 'distinct_product_uris': len({r['product_uri'] for r in rows
            if isinstance(r['product_uri'], str) and r['product_uri']}),
            'acquisition_dates': sorted({r['acquisition_date'] for r in rows if r['acquisition_date']}),
            'baselines': sorted({r['product_identity']['baseline'] for r in rows if r['product_identity']}),
            'nominal_metadata_followup_items': len(eligible),
            'nominal_metadata_followup_dates': sorted({r['acquisition_date'] for r in eligible}),
            'known_cloudy_date_items_retained': sum(r['known_cloudy_acquisition_date'] for r in rows),
            'items_with_issues': sum(bool(r['issues']) for r in rows),
            'source_approvals': 0, 'selected_roster_items': 0, 'geographic_groups': 1}


def worker(args, output):
    import resource

    protocol = read_json(args.protocol.read_bytes())
    requirements, receipts = check_inputs(protocol)
    request = initial_request(requirements, protocol['query'])
    items, pages, total = collect(request, protocol['query'], output)
    rows = [normalize(item, protocol['query'], requirements) for item in items]
    summary = summarize(rows)
    result = {'schema_version': 12, 'status': 'complete_metadata_inventory_no_roster',
              'protocol_sha256': digest(args.protocol), 'group_id': requirements['group_id'],
              'pagination_complete': True, 'provider_number_matched': total, 'pages': pages,
              'summary': summary, 'items': rows,
              'limitations': ['Catalogue footprint is not valid-pixel coverage or cloud QA.',
                             'Nominal N0206 follow-up is not offset/reader/source approval.',
                             'Known-cloudy 26 July and all product variants remain retained.',
                             'Earlier dates require new lineage and cumulative target calculation.',
                             'Two events are one geographic location; no labels or unseen accuracy.']}
    write(output/'inventory.json', result)
    check_inputs(protocol)
    report = '# Earlier pre-image catalogue inventory\n\n'+json.dumps(summary, indent=2)+'\n\n'
    report += '\n'.join('- '+text for text in result['limitations'])+'\n'
    (output/'report.md').write_text(report)
    require(not any(name in sys.modules for name in ('torch', 'rasterio', 's2cloudless')),
            'Metadata worker imported a pixel/model library')
    artifacts = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output/'metadata.json', {'execution_status': 'completed', 'mode': 'metadata_inventory',
          'protocol_sha256': digest(args.protocol), 'input_receipts': receipts, 'artifact_sha256': artifacts,
          'satellite_pixel_reads': 0, 'image_asset_downloads': 0, 'model_executions': 0,
          'labels_approved': 0, 'selected_roster_items': 0, 'native_threads': 1, 'gpu_used': False,
          'worker_high_water_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                           (2**20 if sys.platform == 'darwin' else 1024), 2)})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationEarlierCatalogueV12/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Retain 512 MiB/1.5 GiB/300 s resource bounds')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='earlier-source-metadata', gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
