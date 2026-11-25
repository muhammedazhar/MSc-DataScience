"""Bounded independent-reference metadata for the V28 site feasibility check."""

from datetime import date, timedelta
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.parse import urlencode

from inventory_2019_reference_sources import (
    BYTE_LIMIT, WKID, archive, artifact_hashes, build, field_date_in_window,
    fetch, interpret, metadata_date,
)
from inventory_earlier_validation_sources import digest, read_json, require, write


LANDSAT = 'https://landsat2.arcgis.com/arcgis/rest/services/Landsat8_PanSharpened/ImageServer/query'
LANDSAT_LIMIT = 24


def _check_sites(sites):
    ids = [site['site_id'] for site in sites]
    require(ids and len(ids) == len(set(ids)) and all(re.fullmatch(r'[A-Za-z0-9_-]+', key) for key in ids),
            'Missing, repeated or unsafe reference site ID')
    for site in sites:
        grid = site['grid']
        require(grid['crs'] == f'EPSG:{WKID}', 'Reference API crop CRS differs')
        xmin, ymin, xmax, ymax = grid['bounds']
        require(xmax - xmin == ymax - ymin == 640, 'Reference query needs the exact 640 m crop')
        start, anchor, end = (date.fromisoformat(site[key]) for key in
                              ('reference_start_inclusive', 'post_date', 'reference_end_inclusive'))
        require(start <= anchor <= end, 'Reference dates are out of order')


def landsat_request(site):
    start = date.fromisoformat(site['reference_start_inclusive'])
    end_exclusive = date.fromisoformat(site['reference_end_inclusive']) + timedelta(days=1)
    params = {
        'where': f"Category = 1 AND AcquisitionDate >= DATE '{start}' AND AcquisitionDate < DATE '{end_exclusive}'",
        'geometry': ','.join(map(str, site['grid']['bounds'])), 'geometryType': 'esriGeometryEnvelope',
        'inSR': WKID, 'spatialRel': 'esriSpatialRelWithin', 'orderByFields': 'AcquisitionDate ASC,OBJECTID ASC',
        'outFields': 'OBJECTID,LANDSAT_PRODUCT_ID,Name,AcquisitionDate,CloudCover,SensorName,Category',
        'returnGeometry': 'false', 'resultRecordCount': LANDSAT_LIMIT, 'f': 'json',
    }
    return LANDSAT + '?' + urlencode(params)


def landsat_records(payload, site):
    features = payload['features']
    require(isinstance(features, list) and len(features) <= LANDSAT_LIMIT, 'Invalid or oversized Landsat response')
    records = []
    for index, feature in enumerate(features):
        fields = feature['attributes']
        capture = metadata_date(fields.get('AcquisitionDate'))
        scene_id = fields.get('LANDSAT_PRODUCT_ID')
        if not scene_id or fields.get('SensorName') != 'Landsat 8' or fields.get('Category') != 1:
            records.append({'feature_index': index, 'metadata_issue': 'unconfirmed_landsat_8_scene'})
            continue
        records.append({
            'feature_index': index, 'object_id': fields.get('OBJECTID'), 'scene_id': scene_id,
            'capture_date_utc': capture, 'capture_in_reference_window': field_date_in_window(capture, site),
            'full_crop_coverage': True, 'coverage_basis': 'provider Within query on projected crop',
            'catalogue_cloud_pct': fields.get('CloudCover'), 'panchromatic_sampling_m': 15,
            'multispectral_sampling_m': 30, 'reference_role': 'coarse_context_only',
            'access': 'public catalogue metadata; pixels unexamined',
        })
    return records, payload.get('exceededTransferLimit') is True or len(features) == LANDSAT_LIMIT


def _inventory(output, sites, protocol_sha, queried_at, fetcher, origins, prior_sha):
    result = build(output, sites, protocol_sha, queried_at, fetcher)
    for site in sites:
        site_id = site['site_id']
        receipt, payload = archive(output, f'{site_id}-landsat-8', landsat_request(site), fetcher)
        interpreted = interpret(receipt, payload, lambda page: landsat_records(page, site))
        result['queries'].append({**interpreted, 'site_id': site_id, 'catalogue': 'Esri Landsat 8',
                                  'reference_role': 'coarse_context_only'})
    result.update(schema_version=28, request_origins=origins,
                  network_requests=sum(origin == 'network' for origin in origins.values()),
                  retry_from_sha256=prior_sha)
    result['limitations'].append('Landsat 8 provides coarse context at 15 m panchromatic and 30 m multispectral sampling.')
    write(output/'references.json', result)
    return result


def _bound_archive(directory, sites, protocol_sha):
    metadata = read_json((directory/'metadata.json').read_bytes())
    require(metadata['protocol_sha256'] == protocol_sha
            and metadata['artifact_sha256'] == artifact_hashes(directory), 'Reference archive binding differs')
    original = read_json((directory/'references.json').read_bytes())
    require(original['schema_version'] == 28 and original['protocol_sha256'] == protocol_sha
            and original['sites'] == sites, 'Archived reference protocol or sites differ')
    receipts = {record['query_id']: record for record in original['queries']}
    require(len(receipts) == len(original['queries']) and receipts.keys() == original['request_origins'].keys(),
            'Repeated or unbound archived reference query')
    return original, receipts, metadata


def _response(directory, receipt):
    body = None
    if 'response_file' in receipt:
        raw = directory/'raw'/f"{receipt['query_id']}.response"
        require(raw.relative_to(directory).as_posix() == receipt['response_file']
                and raw.resolve().is_relative_to((directory/'raw').resolve())
                and raw.stat().st_size == receipt['response_bytes'] <= BYTE_LIMIT
                and digest(raw) == receipt['response_sha256'], 'Archived reference response differs')
        body = raw.read_bytes()
    return receipt['transport_status'], receipt['http_status'], receipt['transport_error'], body


def _retryable(receipt):
    code = receipt.get('http_status')
    if receipt['status'] == 'provider_error' and isinstance(receipt.get('error'), dict):
        code = receipt['error'].get('code', code)
    return (receipt['transport_status'] == 'request_failed' or code in (408, 429)
            or type(code) is int and 500 <= code < 600)


def build_references(output, sites, protocol_sha, queried_at, *, fetcher=None, retry_from=None):
    """Archive bounded metadata, optionally retrying transient failures from one prior run."""
    _check_sites(sites)
    prior_sha, prior = None, {}
    if retry_from is not None:
        retry_from = Path(retry_from).resolve()
        original, prior, _ = _bound_archive(retry_from, sites, protocol_sha)
        require(original['retry_from_sha256'] is None, 'Reference requests may be retried once')
        prior_sha = digest(retry_from/'references.json')
    if fetcher is None:
        def fetcher(_, url):
            return fetch(url)
    origins = {}

    def collect(query_id, url):
        record = prior.get(query_id)
        if record is not None:
            require(record['url'] == url, 'Retry reference request differs: '+query_id)
            if not _retryable(record):
                origins[query_id] = 'archive'
                return _response(retry_from, record)
        origins[query_id] = 'network'
        return fetcher(query_id, url)

    return _inventory(Path(output), sites, protocol_sha, queried_at, collect, origins, prior_sha)


def verify_references(directory, output, sites, protocol_sha):
    """Rebuild a metadata archive byte for byte with zero live requests."""
    _check_sites(sites)
    directory, output = Path(directory).resolve(), Path(output)
    original, receipts, metadata = _bound_archive(directory, sites, protocol_sha)
    origins = {}

    def collect(query_id, url):
        record = receipts[query_id]
        require(record['url'] == url, 'Replay reference request differs: '+query_id)
        origin = original['request_origins'][query_id]
        require(origin in ('archive', 'network'), 'Unknown reference request origin')
        origins[query_id] = origin
        return _response(directory, record)

    with TemporaryDirectory(prefix='reference-replay-', dir=output) as path:
        rebuilt = Path(path)
        result = _inventory(rebuilt, sites, protocol_sha, original['queried_at_utc'], collect, origins,
                            original['retry_from_sha256'])
        require(artifact_hashes(rebuilt) == metadata['artifact_sha256'], 'Reference archive replay differs')
    write(output/'replay.json', {'schema_version': 28, 'result': 'byte_identical',
                                 'protocol_sha256': protocol_sha, 'network_requests': 0,
                                 'inventory_artifact_sha256': metadata['artifact_sha256'],
                                 'queries_replayed': len(result['queries'])})
    return result
