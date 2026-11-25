"""Archive and replay Landsat catalogue leads for the stage-1 site roster."""

from datetime import date, datetime, timezone
from http.client import HTTPException
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, build_opener

from inventory_2019_reference_sources import archive, artifact_hashes, interpret
from inventory_earlier_validation_sources import NoRedirect, digest, read_json, require, write
from inventory_unseen_validation_references import LANDSAT, _check_sites, _retryable, landsat_records, landsat_request


MAX_SITES = 29
MAX_RESPONSE_BYTES = 128 * 1024
REQUEST_TIMEOUT_SECONDS = 7
PREVIEW_SIZE = 43
EXPORT = LANDSAT.removesuffix('/query') + '/exportImage'
USGS_ITEMS = 'https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l1/items/'
SCENE_ID = re.compile(r'LC08_(L1TP|L1GT|L1GS)_\d{6}_(\d{8})_\d{8}_02_(T1|T2)')


def locked_preview_metadata_url(site, object_id):
    """Build a preview metadata request. The catalogue scan never sends it."""
    require(type(object_id) is int and object_id > 0, 'Invalid Landsat archive object ID')
    rule = {'mosaicMethod': 'esriMosaicLockRaster', 'lockRasterIds': [object_id], 'mosaicOperation': 'MT_FIRST'}
    return EXPORT+'?'+urlencode({
        'bbox': ','.join(map(str, site['grid']['bounds'])), 'bboxSR': 32636, 'imageSR': 32636,
        'size': f'{PREVIEW_SIZE},{PREVIEW_SIZE}', 'format': 'png24',
        'interpolation': 'RSP_NearestNeighbor', 'mosaicRule': json.dumps(rule, separators=(',', ':')),
        'f': 'json',
    })


def _capture_utc(value):
    try:
        if type(value) in (int, float):
            instant = datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        elif isinstance(value, str):
            instant = datetime.fromisoformat(value.replace('Z', '+00:00'))
        else:
            return None
        return instant.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z') if instant.tzinfo else None
    except (OverflowError, OSError, ValueError):
        return None


def _records(payload, site):
    records, limited = landsat_records(payload, site)
    seen = set()
    for row in records:
        if 'metadata_issue' in row:
            continue
        cloud = row.pop('catalogue_cloud_pct')
        row['esri_cloud_fraction'] = cloud
        row['scene_cloud_percent'] = round(cloud * 100, 6) if type(cloud) in (int, float) and 0 <= cloud <= 1 else None
        product = SCENE_ID.fullmatch(row['scene_id'])
        object_id = row['object_id']
        attributes = payload['features'][row['feature_index']]['attributes']
        row['capture_timestamp_utc'] = _capture_utc(attributes.get('AcquisitionDate'))
        if (not product or not row['capture_in_reference_window']
                or product.group(2) != row['capture_date_utc'].replace('-', '')
                or type(object_id) is not int or object_id <= 0):
            row['metadata_issue'] = 'unconfirmed_scene_identity_date_or_raster_id'
            continue
        require(object_id not in seen, 'Repeated Landsat archive object ID')
        seen.add(object_id)
        row.update(
            correction=product.group(1), collection_tier=product.group(3),
            usgs_item_url=USGS_ITEMS+row['scene_id'],
            locked_preview_metadata_url=locked_preview_metadata_url(site, object_id),
            role_relative_to_annotation=('before_annotation' if row['capture_date_utc'] < site['post_date']
                                         else 'on_or_after_annotation'),
        )
    return records, limited


def _rank(scene, anchor):
    cloud = scene['scene_cloud_percent']
    days = abs((date.fromisoformat(scene['capture_date_utc']) - anchor).days)
    return (cloud if cloud is not None else 101, scene['collection_tier'] != 'T1', days,
            scene['capture_date_utc'], scene['scene_id'], scene['object_id'])


def _fetch(request_url):
    request = Request(request_url, headers={'Accept': 'application/json', 'Accept-Encoding': 'identity',
                                           'User-Agent': 'COMP1252-bounded-landsat-catalogue/35'})
    try:
        try:
            response = build_opener(NoRedirect()).open(request, timeout=REQUEST_TIMEOUT_SECONDS)
        except HTTPError as error:
            response = error
        with response:
            require(response.geturl() == request_url, 'Redirected Landsat catalogue response')
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                return 'response_byte_limit', response.status, None, body[:MAX_RESPONSE_BYTES]
            status = 'received' if response.status == 200 else 'http_error'
            return status, response.status, None, body
    except (OSError, HTTPException, ValueError) as error:
        return 'request_failed', None, str(error), None


def _run(output, sites, protocol_sha, queried_at, fetcher, origins, prior_sha):
    (output/'raw').mkdir()
    queries = []
    for site in sites:
        query_id = f"{site['site_id']}-landsat-8"
        receipt, payload = archive(output, query_id, landsat_request(site), fetcher)
        result = interpret(receipt, payload, lambda page: _records(page, site))
        eligible = [row for row in result['records'] if 'metadata_issue' not in row] if result['status'] == 'complete' else []
        anchor = date.fromisoformat(site['post_date'])
        ranked = sorted(eligible, key=lambda scene: _rank(scene, anchor))
        queries.append({
            **result, 'site_id': site['site_id'], 'catalogue': 'Esri Landsat 8',
            'before_options': [row for row in ranked if row['role_relative_to_annotation'] == 'before_annotation'],
            'after_options': [row for row in ranked if row['role_relative_to_annotation'] == 'on_or_after_annotation'],
        })
    result = {
        'schema_version': 35, 'protocol_sha256': protocol_sha, 'queried_at_utc': queried_at,
        'sites': sites, 'queries': queries, 'request_origins': origins,
        'network_requests': sum(origin == 'network' for origin in origins.values()), 'retry_from_sha256': prior_sha,
        'limits': {'maximum_sites': MAX_SITES, 'maximum_response_bytes': MAX_RESPONSE_BYTES,
                   'request_timeout_seconds': REQUEST_TIMEOUT_SECONDS},
        'limitations': [
            'Within is caller-requested provider footprint coverage for the complete projected crop.',
            'Esri CloudCover is a scene fraction. Converted scene percentages cannot establish crop clarity.',
            'Tier 2 geometric uncertainty limits precise interpretation of target and control boundaries.',
            'The 15 m pan and 30 m multispectral sampling constrain class judgments. The archive contains metadata only.',
            'Annotation dates are image observations. Temporal roles need verification against the chosen Sentinel-2 dates.',
        ],
        'reference_imagery_acquired': False, 'labels_approved': 0,
    }
    write(output/'landsat.json', result)
    return result


def _bound_archive(directory, sites, protocol_sha):
    metadata = read_json((directory/'metadata.json').read_bytes())
    require(metadata['artifact_sha256'] == artifact_hashes(directory)
            and metadata['protocol_sha256'] == protocol_sha, 'Landsat archive metadata binding differs')
    original = read_json((directory/'landsat.json').read_bytes())
    require(original['schema_version'] == 35 and original['sites'] == sites
            and original['protocol_sha256'] == protocol_sha, 'Landsat archive scope differs')
    receipts = {row['query_id']: row for row in original['queries']}
    require(len(receipts) == len(sites) and receipts.keys() == original['request_origins'].keys(),
            'Repeated or unbound Landsat archive query')
    return original, receipts, metadata


def _archived_response(directory, receipt):
    body = None
    if 'response_file' in receipt:
        raw = directory/'raw'/f"{receipt['query_id']}.response"
        require(raw.relative_to(directory).as_posix() == receipt['response_file']
                and raw.resolve().is_relative_to((directory/'raw').resolve())
                and raw.stat().st_size == receipt['response_bytes'] <= MAX_RESPONSE_BYTES
                and digest(raw) == receipt['response_sha256'], 'Landsat archived response differs')
        body = raw.read_bytes()
    return receipt['transport_status'], receipt['http_status'], receipt['transport_error'], body


def build_search(output, sites, protocol_sha, queried_at, *, fetcher=None, retry_from=None):
    """Query exact V28 crops once, with one optional transient retry pass."""
    _check_sites(sites)
    require(len(sites) <= MAX_SITES, 'Landsat query roster exceeds the stage-1 bound')
    prior, prior_sha = {}, None
    if retry_from is not None:
        retry_from = Path(retry_from).resolve()
        original, prior, _ = _bound_archive(retry_from, sites, protocol_sha)
        require(original['retry_from_sha256'] is None, 'Landsat catalogue queries may be retried once')
        prior_sha = digest(retry_from/'landsat.json')
    origins = {}

    def collect(query_id, url):
        previous = prior.get(query_id)
        if previous is not None:
            require(previous['url'] == url, 'Landsat retry query differs: '+query_id)
            if not _retryable(previous):
                origins[query_id] = 'archive'
                return _archived_response(retry_from, previous)
        origins[query_id] = 'network'
        response = _fetch(url) if fetcher is None else fetcher(query_id, url)
        require(response[3] is None or len(response[3]) <= MAX_RESPONSE_BYTES, 'Landsat response exceeds byte cap')
        return response

    return _run(Path(output), sites, protocol_sha, queried_at, collect, origins, prior_sha)


def verify_search(directory, output, sites, protocol_sha):
    """Replay a completed Landsat archive without live requests."""
    _check_sites(sites)
    require(len(sites) <= MAX_SITES, 'Landsat replay roster exceeds the stage-1 bound')
    directory, output = Path(directory).resolve(), Path(output)
    original, receipts, metadata = _bound_archive(directory, sites, protocol_sha)
    origins = {}

    def collect(query_id, url):
        receipt = receipts[query_id]
        require(receipt['url'] == url, 'Landsat replay query differs: '+query_id)
        origin = original['request_origins'][query_id]
        require(origin in ('archive', 'network'), 'Unknown Landsat request origin')
        origins[query_id] = origin
        return _archived_response(directory, receipt)

    with TemporaryDirectory(prefix='landsat-replay-', dir=output) as path:
        rebuilt = Path(path)
        result = _run(rebuilt, sites, protocol_sha, original['queried_at_utc'], collect, origins,
                      original['retry_from_sha256'])
        require(artifact_hashes(rebuilt) == metadata['artifact_sha256'], 'Landsat archive replay differs')
    write(output/'replay.json', {'schema_version': 35, 'result': 'byte_identical',
                                 'protocol_sha256': protocol_sha, 'network_requests': 0,
                                 'inventory_artifact_sha256': metadata['artifact_sha256'],
                                 'queries_replayed': len(result['queries'])})
    return result
