#!/usr/bin/env python3
"""Archive independent reference metadata for the two frozen 2019 validation crops."""

import argparse
from datetime import date, datetime, timedelta, timezone
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import resource
import sys
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, build_opener

from inventory_earlier_validation_sources import (
    ROOT, NoRedirect, RESOURCES, digest, read_json, require, rooted, sha, write,
)


IMPLEMENTATION = 'Solutions/inventory_2019_reference_sources.py'
PROTOCOL = ROOT/'Docs/Reviews/Validation2019SourcesV25/protocol.json'
AIRBUS = ('https://portal.intelligence-airbusds.com/arcgis/rest/services/'
          'OneAtlas/OADConstellationArchive/MapServer')
AIRBUS_LAYERS = {9: 'Pléiades', 14: 'SPOT'}
WAYBACK_CONFIG = 'https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json'
WAYBACK = 'https://metadata.maptiles.arcgis.com/arcgis/rest/services/World_Imagery_Metadata_'
WAYBACK_SERVICE = re.compile(r'^'+re.escape(WAYBACK)+r'(\d{4}_r\d{2})/MapServer/?$')
WAYBACK_TITLE = re.compile(r'World Imagery \(Wayback (\d{4}-\d{2}-\d{2})\)')
WAYBACK_LAYER = 6
WKID = 32636
PAGE_LIMIT = 100
BYTE_LIMIT = 2*2**20
TIMEOUT = 15


def read_inputs(protocol_path):
    raw = protocol_path.read_bytes()
    protocol = read_json(raw)
    require(protocol['schema_version'] == 25 and protocol['resources'] == RESOURCES,
            'V25 reference scope or resource declaration differs')
    rules = protocol['reference_rules']
    require(rules['airbus_layers'] == list(AIRBUS_LAYERS) and rules['wayback_layer'] == WAYBACK_LAYER
            and rules['airbus_result_limit_per_layer'] == PAGE_LIMIT
            and rules['airbus_max_response_bytes'] == rules['maximum_response_bytes'] == BYTE_LIMIT
            and rules['request_timeout_seconds'] == TIMEOUT, 'V25 reference query limits differ')
    implementations = protocol['implementation_sha256']
    require(IMPLEMENTATION in implementations, 'Pin the reference implementation before querying')
    for receipt in protocol['source_files'].values():
        require(digest(rooted(receipt['path'])) == receipt['sha256'],
                'Pinned source differs: '+receipt['path'])
    for name, expected in {**protocol['frozen_implementation_sha256'], **implementations}.items():
        require(digest(rooted(name)) == expected, 'Pinned implementation differs: '+name)

    expected = [
        ('march', 'pair-f113871fa642d1296fbc', [725820, 5510500, 726460, 5511140],
         '2019-03-03', '2019-04-02', '2019-05-12'),
        ('may', 'pair-73bad06c9e0d3fe6b7fc', [769800, 5574380, 770440, 5575020],
         '2019-05-07', '2019-06-06', '2019-07-11'),
    ]
    sites = protocol['sites']
    require(all(site['grid']['crs'] == f'EPSG:{WKID}' for site in sites)
            and [(site['site_id'], site['pair_id'], site['grid']['bounds'],
                  site['reference_start_inclusive'], site['post_date'],
                  site['reference_end_inclusive']) for site in sites] == expected,
            'Frozen 2019 reference crops or dates differ')
    return sites, sha(raw)


def airbus_request(site, layer):
    end_exclusive = date.fromisoformat(site['reference_end_inclusive']) + timedelta(days=1)
    start = site['reference_start_inclusive']
    params = {'where': f"acquisitiondate >= DATE '{start}' AND acquisitiondate < DATE '{end_exclusive}'",
              'geometry': ','.join(map(str, site['grid']['bounds'])),
              'geometryType': 'esriGeometryEnvelope', 'inSR': WKID, 'outSR': WKID,
              'spatialRel': 'esriSpatialRelIntersects', 'outFields': '*', 'returnGeometry': 'true',
              'f': 'geojson', 'resultRecordCount': PAGE_LIMIT, 'orderByFields': 'acquisitiondate ASC'}
    return f'{AIRBUS}/{layer}/query?'+urlencode(params)


def wayback_request(site, release):
    envelope = dict(zip(('xmin', 'ymin', 'xmax', 'ymax'), site['grid']['bounds']))
    envelope['spatialReference'] = {'wkid': WKID}
    params = {'where': '1=1', 'geometry': json.dumps(envelope, separators=(',', ':')),
              'geometryType': 'esriGeometryEnvelope', 'inSR': WKID,
              'spatialRel': 'esriSpatialRelWithin', 'outFields': 'OBJECTID,SRC_DATE2,NICE_DESC,SRC_DESC,SAMP_RES,SRC_ACC',
              'returnGeometry': 'false', 'f': 'json'}
    return f'{WAYBACK}{release}/MapServer/{WAYBACK_LAYER}/query?'+urlencode(params)


def fetch(url):
    request = Request(url, headers={'Accept': 'application/json, application/geo+json',
                                   'User-Agent': 'COMP1252-bounded-reference-metadata/25'})
    try:
        try:
            response = build_opener(NoRedirect()).open(request, timeout=TIMEOUT)
        except HTTPError as error:
            response = error
        with response:
            body = response.read(BYTE_LIMIT+1)
            status = 'received' if response.status == 200 else 'http_error'
            if len(body) > BYTE_LIMIT:
                return 'response_byte_limit', response.status, None, body[:BYTE_LIMIT]
            return status, response.status, None, body
    except (OSError, HTTPException, ValueError) as error:
        return 'request_failed', None, str(error), None


def archive(output, query_id, url, fetcher):
    status, http_status, error, body = fetcher(query_id, url)
    receipt = {'query_id': query_id, 'url': url, 'transport_status': status,
               'http_status': http_status, 'transport_error': error}
    if body is not None:
        name = f'raw/{query_id}.response'
        (output/name).write_bytes(body)
        receipt.update(response_file=name, response_sha256=sha(body), response_bytes=len(body))
    if status != 'received':
        return {**receipt, 'status': status}, None
    try:
        payload = read_json(body)
        require(isinstance(payload, dict), 'Expected a JSON object')
    except ValueError as exc:
        return {**receipt, 'status': 'invalid_response', 'error': str(exc)}, None
    if 'error' in payload:
        return {**receipt, 'status': 'provider_error', 'error': payload['error']}, None
    return {**receipt, 'status': 'received'}, payload


def metadata_date(value):
    """Normalize Esri's Unix milliseconds or a provider ISO date or timestamp."""
    try:
        if type(value) in (int, float):
            return datetime.fromtimestamp(value/1000, tz=timezone.utc).date().isoformat()
        if isinstance(value, str):
            instant = datetime.fromisoformat(value)
            if instant.tzinfo:
                instant = instant.astimezone(timezone.utc)
            return instant.date().isoformat()
    except (OverflowError, OSError, ValueError):
        pass
    return None


def field_date_in_window(capture, site):
    if capture is None:
        return None
    return site['reference_start_inclusive'] <= capture <= site['reference_end_inclusive']


def footprint_coverage(geometry, crs_name, bounds):
    from shapely.errors import ShapelyError
    from shapely.geometry import box, shape

    if crs_name != f'EPSG:{WKID}' or not isinstance(geometry, dict):
        return None
    try:
        footprint = shape(geometry)
        if footprint.geom_type in ('Polygon', 'MultiPolygon') and not footprint.is_empty and footprint.is_valid:
            return bool(footprint.covers(box(*bounds)))
    except (ShapelyError, KeyError, TypeError, ValueError):
        pass
    return None


def airbus_records(payload, site):
    require(payload['type'] == 'FeatureCollection', 'Expected Airbus GeoJSON FeatureCollection')
    features = payload['features']
    require(isinstance(features, list), 'Expected Airbus features list')
    crs = payload.get('crs') or {}
    crs_name = crs.get('properties', {}).get('name')
    records = []
    for index, feature in enumerate(features):
        if not isinstance(feature, dict) or not isinstance(feature.get('properties'), dict):
            records.append({'feature_index': index, 'metadata_issue': 'missing_feature_properties'})
            continue
        fields = feature['properties']
        capture = metadata_date(fields.get('acquisitiondate'))
        records.append({'feature_index': index, 'object_id': fields.get('objectid', feature.get('id')),
                        'scene_id': fields.get('parentidentifier'), 'capture_date_utc': capture,
                        'publication_date_utc': metadata_date(fields.get('publicationdate')),
                        'capture_in_reference_window': field_date_in_window(capture, site),
                        'full_crop_coverage': footprint_coverage(feature.get('geometry'), crs_name,
                                                                site['grid']['bounds']),
                        'sampling_m': None, 'sampling_basis': 'scene sampling unverified in archive metadata',
                        'catalogue_cloud_pct': fields.get('cloudcover'),
                        'processing_level': fields.get('processinglevel'),
                        'access': 'public metadata; scene access and terms unresolved'})
    limited = payload.get('exceededTransferLimit') is True or len(features) >= PAGE_LIMIT
    return records, limited


def wayback_releases(config):
    releases = []
    for tile_release, entry in config.items():
        require(isinstance(entry, dict) and tile_release.isascii() and tile_release.isdecimal(),
                'Unexpected Wayback release configuration entry')
        title = entry.get('itemTitle', '')
        title_match = WAYBACK_TITLE.fullmatch(title)
        service = WAYBACK_SERVICE.fullmatch(entry.get('metadataLayerUrl', ''))
        require(title_match and service,
                'Wayback title or metadata service differs from the configured release')
        releases.append({'metadata_release': service[1],
                         'release_date': date.fromisoformat(title_match[1]).isoformat(),
                         'release_title': title, 'release_num': int(tile_release)})
    require(releases, 'No trusted Wayback release entries')
    require(len({release['metadata_release'] for release in releases}) == len(releases),
            'Duplicate Wayback metadata release')
    return sorted(releases, key=lambda item: (item['release_date'], item['metadata_release']))


def selected_releases(releases, site):
    selected = {}
    for anchor in ('reference_start_inclusive', 'post_date', 'reference_end_inclusive'):
        eligible = next((item for item in releases if item['release_date'] >= site[anchor]), None)
        key = eligible['metadata_release'] if eligible else 'unavailable'
        result = selected.setdefault(key, {**(eligible or {}), 'anchors': []})
        result['anchors'].append({'field': anchor, 'date': site[anchor]})
    return list(selected.values())


def wayback_records(payload, site, release):
    features = payload['features']
    require(isinstance(features, list), 'Expected Wayback metadata features list')
    records = []
    for index, feature in enumerate(features):
        if not isinstance(feature, dict) or not isinstance(feature.get('attributes'), dict):
            records.append({'feature_index': index, 'metadata_issue': 'missing_feature_attributes'})
            continue
        fields = feature['attributes']
        capture = metadata_date(fields.get('SRC_DATE2'))
        records.append({'feature_index': index, 'object_id': fields.get('OBJECTID'),
                        'capture_date_utc': capture, 'release_date': release['release_date'],
                        'capture_in_reference_window': field_date_in_window(capture, site),
                        'capture_on_or_before_release': capture <= release['release_date'] if capture else None,
                        'source': fields.get('NICE_DESC'), 'sensor': fields.get('SRC_DESC'),
                        'sampling_m': fields.get('SAMP_RES'), 'accuracy_m': fields.get('SRC_ACC'),
                        'full_crop_coverage': True, 'coverage_basis': 'provider Within query on projected crop',
                        'access': 'public Wayback display metadata; image use terms and pixels unexamined'})
    return records, payload.get('exceededTransferLimit') is True


def interpret(receipt, payload, reader):
    if payload is None:
        return {**receipt, 'records': []}
    try:
        records, limited = reader(payload)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        return {**receipt, 'status': 'invalid_response', 'error': str(error), 'records': []}
    return {**receipt, 'status': 'transfer_limit' if limited else 'complete',
            'records': records}


def build(output, sites, protocol_sha, queried_at, fetcher):
    (output/'raw').mkdir()
    config, payload = archive(output, 'wayback-config', WAYBACK_CONFIG, fetcher)
    releases = []
    if payload is not None:
        try:
            releases = wayback_releases(payload)
        except (TypeError, ValueError) as error:
            config.update(status='invalid_response', error=str(error))
        else:
            config.update(status='complete', parsed_releases=len(releases))
    queries = [config]
    selections = {}
    for site in sites:
        site_id = site['site_id']
        for layer, platform in AIRBUS_LAYERS.items():
            receipt, payload = archive(output, f'{site_id}-airbus-{layer}', airbus_request(site, layer), fetcher)
            result = interpret(receipt, payload, lambda page: airbus_records(page, site))
            queries.append({**result, 'site_id': site_id, 'catalogue': 'Airbus', 'layer': layer,
                            'platform': platform})
        selections[site_id] = selected_releases(releases, site)
        for release in selections[site_id]:
            release_id = release.get('metadata_release')
            if not release_id:
                release['status'] = 'release_unavailable' if releases else 'configuration_unavailable'
                continue
            receipt, payload = archive(output, f'{site_id}-wayback-{release_id}',
                                       wayback_request(site, release_id), fetcher)
            result = interpret(receipt, payload, lambda page: wayback_records(page, site, release))
            queries.append({**result, 'site_id': site_id, 'catalogue': 'Esri Wayback',
                            'layer': WAYBACK_LAYER, 'release': release})
    result = {'schema_version': 25, 'protocol_sha256': protocol_sha, 'queried_at_utc': queried_at,
              'reference_scope': 'public independent reference metadata for the exact frozen crop',
              'sites': sites, 'wayback_release_selections': selections, 'queries': queries,
              'limitations': [
                  'Archive visibility and public access vary. Empty responses cover the queried public layers and dates.',
                  'Airbus acquisition and publication dates are distinct. Catalogue cloud is scene-level metadata.',
                  'Airbus scene sample spacing and scene imagery access remain unverified.',
                  'Wayback release dates date the mosaic version. SRC_DATE2 dates the source listed in that version.',
                  'A full footprint is spatial metadata. Crop clarity, land cover and event timing require pixel evidence.',
              ],
              'reference_imagery_acquired': False, 'reference_roster_selected': False,
              'labels_approved': 0}
    write(output/'references.json', result)
    return result


def artifact_hashes(directory):
    return {str(path.relative_to(directory)): digest(path) for path in sorted(directory.rglob('*'))
            if path.is_file() and path != directory/'metadata.json'}


def replay(output, inventory, sites, protocol_sha):
    metadata = read_json((inventory/'metadata.json').read_bytes())
    require(metadata['execution_status'] == 'completed' and metadata['mode'] == '2019_reference_inventory'
            and metadata['protocol_sha256'] == protocol_sha
            and artifact_hashes(inventory) == metadata['artifact_sha256'], 'Archive binding differs')
    original = read_json((inventory/'references.json').read_bytes())
    receipts = {record['query_id']: record for record in original['queries']}
    require(len(receipts) == len(original['queries']), 'Repeated query ID in archive')

    def read_archive(query_id, url):
        record = receipts[query_id]
        require(record['url'] == url, 'Archived request differs: '+query_id)
        body = (inventory/f'raw/{query_id}.response').read_bytes() if 'response_file' in record else None
        return record['transport_status'], record['http_status'], record['transport_error'], body

    with TemporaryDirectory(prefix='reference-replay-', dir=output) as path:
        rebuilt = Path(path)
        result = build(rebuilt, sites, protocol_sha, original['queried_at_utc'], read_archive)
        require(artifact_hashes(rebuilt) == metadata['artifact_sha256'], 'Reference archive replay differs')
    write(output/'replay.json', {'schema_version': 25, 'result': 'byte_identical',
                                 'protocol_sha256': protocol_sha, 'network_requests': 0,
                                 'inventory_artifact_sha256': metadata['artifact_sha256'],
                                 'queries_replayed': len(result['queries'])})
    return result


def worker(args):
    output = args.worker_output
    sites, protocol_sha = read_inputs(args.protocol)
    if args.mode == 'inventory':
        now = datetime.now(timezone.utc).isoformat(timespec='seconds')
        result = build(output, sites, protocol_sha, now, lambda _, url: fetch(url))
    else:
        result = replay(output, args.inventory, sites, protocol_sha)
    require(not {'torch', 'rasterio', 's2cloudless'} & sys.modules.keys(), 'Unexpected pixel or model import')
    require(digest(args.protocol) == protocol_sha, 'Protocol changed during the reference inventory')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'execution_status': 'completed',
        'mode': '2019_reference_inventory' if args.mode == 'inventory' else '2019_reference_replay',
        'protocol_sha256': protocol_sha, 'artifact_sha256': artifact_hashes(output),
        'network_requests': len(result['queries']) if args.mode == 'inventory' else 0,
        'satellite_pixel_reads': 0, 'image_asset_downloads': 0, 'model_executions': 0,
        'labels_approved': 0, 'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2),
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--mode', choices=('inventory', 'verify'), default='inventory')
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=RESOURCES['worker_rss_mib'],
                        min_available_gib=RESOURCES['available_memory_reserve_gib'],
                        max_seconds=RESOURCES['max_seconds_per_worker'])
    args = parser.parse_args()
    if (args.mode == 'verify') != (args.inventory is not None):
        parser.error('--inventory is required only for verify mode')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(RESOURCES['nice_increment'])
        worker(args)
    else:
        run_guarded(args, script_path=__file__, operation='2019-reference-inventory',
                    gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
