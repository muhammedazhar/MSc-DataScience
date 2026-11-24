"""Find recent native Sentinel-2 L1C candidates for a finite set of plots."""

from collections import Counter
from datetime import date
from http.client import HTTPException
import math
from pathlib import Path
from urllib.request import Request, build_opener

from bbox_sentinel_preflight import catalogue_source
from inventory_earlier_validation_sources import (
    ENDPOINT, NoRedirect, canonical, coverage, next_request, read_json, require, sha,
    trusted, write,
)
from preflight_2019_validation_sources import native_timestamp


COLLECTION = 'sentinel-2-l1c'
PAGE_LIMIT = 20
MAX_PAGES_PER_PLOT = 5
MAX_PAGES_PER_RUN = 20
MAX_RESPONSE_BYTES = 2 * 2**20
REQUEST_TIMEOUT = 12


def search_request(bounds, start, end, cloud_cover):
    filters = {'s2:processing_baseline': {'lt': '04.00'}}
    if cloud_cover is not None:
        filters['eo:cloud_cover'] = {'lte': cloud_cover}
    return {'url': ENDPOINT, 'method': 'POST', 'body': {
        'collections': [COLLECTION], 'bbox': bounds,
        'datetime': f'{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59.999999Z',
        'limit': PAGE_LIMIT, 'query': filters,
        'sortby': [{'field': 'properties.datetime', 'direction': 'desc'},
                   {'field': 'id', 'direction': 'asc'}],
    }}


def fetch_page(request, opener):
    trusted(request['url'])
    body = None if request['body'] is None else canonical(request['body'])
    http_request = Request(request['url'], data=body, method=request['method'], headers={
        'Content-Type': 'application/json', 'Accept': 'application/geo+json, application/json',
        'User-Agent': 'COMP1252-bounding-box-catalogue',
    })
    with opener(http_request, timeout=REQUEST_TIMEOUT) as response:
        require(response.status == 200 and response.geturl() == request['url'],
                'Expected a successful response from the exact catalogue URL')
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        require(len(raw) <= MAX_RESPONSE_BYTES, 'Catalogue response exceeded the byte limit')
    return raw


def candidate(feature, bounds, start, end, cloud_cover):
    try:
        source = catalogue_source(feature)
        observed = native_timestamp(feature['properties']['datetime'])
    except (ValueError, KeyError, TypeError):
        return None, 'unsupported_catalogue_identity_or_datetime'
    day = date.fromisoformat(source['acquisition_date'])
    if not start <= day <= end:
        return None, 'outside_inclusive_date_range'
    cloud = feature['properties'].get('eo:cloud_cover')
    if cloud_cover is not None and (type(cloud) not in (int, float) or not 0 <= cloud <= cloud_cover):
        return None, 'outside_catalogue_cloud_limit'
    covering, issue = coverage(feature.get('geometry'), {'crs': 'EPSG:4326',
                               'core_union_bounds': bounds, 'qa_context_union_bounds': bounds})
    if not covering['core']:
        return None, issue or 'single_scene_does_not_cover_bbox'
    satellite, _level, datatake, _suffix = source['scene_id'].split('_', 3)
    return (observed, (satellite, datatake, source['relative_orbit'])), None


def search_plot(plot, start, end, count, cloud_cover, directory, remaining_pages, opener):
    bounds = list(plot['bbox'])
    require(len(bounds) == 4 and all(type(x) in (int, float) and math.isfinite(x) for x in bounds)
            and -180 <= bounds[0] < bounds[2] <= 180 and -90 <= bounds[1] < bounds[3] <= 90,
            'A nonempty WGS84 bounding box is required')
    directory.mkdir()
    request = search_request(bounds, start, end, cloud_cover)
    result = {**plot, 'bbox': bounds, 'candidates': [], 'pages': [], 'requests_attempted': 0,
              'discovery_status': 'partial', 'stop_reason': 'page_budget_reached'}
    rejects = Counter()
    seen_items = {}
    seen_requests = set()
    acquisitions = set()
    last_observed = None
    try:
        for index in range(1, min(remaining_pages, MAX_PAGES_PER_PLOT) + 1):
            request_hash = sha(canonical(request))
            require(request_hash not in seen_requests, 'Catalogue repeated a pagination request')
            seen_requests.add(request_hash)
            request_path = directory / f'request-{index:02d}.json'
            response_path = directory / f'page-{index:02d}.json'
            write(request_path, request)
            result['requests_attempted'] += 1
            raw = fetch_page(request, opener)
            response_path.write_bytes(raw)
            result['pages'].append({'request': request_path.name, 'response': response_path.name,
                                    'bytes': len(raw), 'sha256': sha(raw)})
            page = read_json(raw)
            require(isinstance(page, dict) and page.get('type') == 'FeatureCollection'
                    and isinstance(page.get('features'), list) and len(page['features']) <= PAGE_LIMIT,
                    'Invalid or oversized catalogue page')
            for feature in page['features']:
                require(isinstance(feature, dict) and feature.get('type') == 'Feature'
                        and feature.get('collection') == COLLECTION
                        and isinstance(feature.get('id'), str) and feature['id'], 'Invalid native catalogue item')
                item_id = feature['id']
                fingerprint = sha(canonical(feature))
                if item_id in seen_items:
                    require(fingerprint == seen_items[item_id], 'Catalogue returned conflicting copies of an item')
                    rejects['duplicate_catalogue_item'] += 1
                    continue
                seen_items[item_id] = fingerprint
                match, issue = candidate(feature, bounds, start, end, cloud_cover)
                if issue is not None:
                    rejects[issue] += 1
                    continue
                observed, acquisition = match
                require(last_observed is None or observed <= last_observed,
                        'Catalogue did not respect newest-first sorting')
                last_observed = observed
                if acquisition in acquisitions:
                    rejects['duplicate_acquisition'] += 1
                    continue
                if len(result['candidates']) < count:
                    acquisitions.add(acquisition)
                    result['candidates'].append(feature)
            if len(result['candidates']) == count:
                result.update(discovery_status='complete', stop_reason='requested_count_reached')
                break
            request = next_request(page, request)
            if request is None:
                result.update(discovery_status='complete' if result['candidates'] else 'no_match',
                              stop_reason='catalogue_exhausted')
                break
    except (HTTPException, OSError, ValueError, TypeError, KeyError) as error:
        result.update(discovery_status='failed', stop_reason='catalogue_request_or_response_error',
                      error=f'{type(error).__name__}: {error}', candidates=[])
    result['rejected_items'] = dict(sorted(rejects.items()))
    return result


def discover(plots, start, end, count, cloud_cover, output, *, opener=None):
    """Save raw metadata and return unique matching acquisitions for each plot."""
    require(type(start) is date and type(end) is date and start <= end, 'Invalid inclusive date range')
    require(type(count) is int and 1 <= count <= PAGE_LIMIT * MAX_PAGES_PER_PLOT,
            'Count must fit the per-plot catalogue page budget')
    require(cloud_cover is None or type(cloud_cover) in (int, float) and 0 <= cloud_cover <= 100,
            'Cloud cover must be a percentage from 0 to 100')
    output = Path(output)
    archive = output / 'catalogue'
    archive.mkdir()
    results = []
    requests = 0
    opener = opener or build_opener(NoRedirect()).open
    for index, plot in enumerate(plots, 1):
        directory = archive / f'plot-{index:03d}'
        result = search_plot(plot, start, end, count, cloud_cover, directory, MAX_PAGES_PER_RUN - requests, opener)
        result['catalogue_directory'] = directory.relative_to(output).as_posix()
        results.append(result)
        requests += result['requests_attempted']
    return {'plots': results, 'catalogue_requests': requests, 'catalogue_collection': COLLECTION,
            'catalogue_cloud_threshold': cloud_cover, 'count_per_plot': count,
            'start_inclusive': start.isoformat(), 'end_inclusive': end.isoformat()}
