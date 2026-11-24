#!/usr/bin/env python3
"""Check the two V23 36UXA nominations against actual provider XML and listings."""

import argparse
from email.message import Message
from http.client import HTTPException
from io import BytesIO
import math
import os
from pathlib import Path
import re
import resource
import shutil
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

from inventory_earlier_validation_sources import (
    RESOURCES, ROOT, canonical, digest, read_json, require, rooted, sha, timestamp, write,
)
from preflight_earlier_validation_metadata import (
    BANDS, BASE, MetadataNetwork, asset_manifest, disk_budget, listing_url, parse_listing, single, texts,
)
from preflight_remaining_validation_metadata import BLOCKED, READY


PROTOCOL = ROOT/'Docs/Reviews/ValidationFreshSourcesV24/protocol.json'
DATES = ('2018-07-21', '2018-07-16')
TILE = '36UXA'
PROVIDER_RULES = {
    'base_url': BASE, 'allow_automatic_redirects': False, 'listing_max_keys': 300,
    'maximum_listing_bytes': 2*2**20, 'require_listing_not_truncated': True,
    'maximum_xml_bytes': 5*2**20, 'maximum_total_response_bytes': 24*2**20,
    'maximum_requests': 8, 'timeout_seconds': 20,
    'require_xml_size_etag_content_length_md5_if_match': True, 'anonymous_only_no_paid_access': True,
}


def load(path):
    return read_json(path.read_bytes())


def checked_protocol():
    protocol = load(PROTOCOL)
    require(protocol['schema_version'] == 24 and protocol['resources'] == RESOURCES
            and tuple(protocol['candidate_dates']) == DATES
            and all(protocol['provider'].get(key) == value for key, value in PROVIDER_RULES.items()),
            'V24 metadata scope, provider or resource limits differ')
    receipts = {record['path']: record['sha256'] for record in protocol['source_files'].values()}
    receipts.update(protocol['frozen_implementation_sha256'])
    receipts.update(protocol['implementation_sha256'])
    require(str(Path(__file__).resolve().relative_to(ROOT)) in protocol['implementation_sha256'],
            'Pin the V24 preflight code before use')
    for path, checksum in receipts.items():
        require(digest(rooted(path)) == checksum, 'Pinned V24 input changed: '+path)
    return protocol


def checked_artifacts(directory, artifacts):
    for name, checksum in artifacts.items():
        path = (directory/name).resolve()
        require(path.is_relative_to(directory) and path.is_file() and digest(path) == checksum,
                'Missing, escaping or modified evidence artifact: '+name)


def raw_items(inventory_path, inventory):
    results = {}
    root = inventory_path.parent
    for receipt in inventory['pages']:
        path = (root/receipt['response']).resolve()
        require(path.is_relative_to(root/'raw') and path.is_file()
                and path.stat().st_size == receipt['bytes'] <= 2*2**20
                and digest(path) == receipt['sha256'], 'Saved V23 catalogue page differs')
        for feature in load(path)['features']:
            require(feature['collection'] == 'sentinel-2-l1c', 'V23 raw item collection differs')
            identifier = feature['id']
            require(identifier not in results or results[identifier] == feature,
                    'V23 item ID refers to multiple conflicting raw features')
            results[identifier] = feature
    return results


def inputs(protocol):
    files = {role: rooted(receipt['path']) for role, receipt in protocol['source_files'].items()}
    plan, inventory, metadata, replay = (load(files[key]) for key in (
        'v23_plan', 'v23_inventory', 'v23_inventory_metadata', 'v23_inventory_replay'))
    plan_hash = digest(files['v23_plan'])
    require(plan['schema_version'] == inventory['schema_version'] == 23
            and plan['status'] == 'fresh_catalogue_plan_ready'
            and inventory['status'] == 'complete_nominal_catalogue_inventory' and inventory['pagination_complete']
            and metadata['mode'] == 'fresh_source_inventory' and metadata['execution_status'] == 'completed'
            and replay['status'] == 'passed' and replay['network_requests'] == 0
            and plan['protocol_sha256'] == inventory['protocol_sha256'] == metadata['protocol_sha256'] == replay['protocol_sha256']
            and inventory['plan_sha256'] == metadata['plan_sha256'] == replay['plan_sha256'] == plan_hash
            and replay['inventory_metadata_sha256'] == digest(files['v23_inventory_metadata'])
            and replay['inventory_artifact_sha256'] == metadata['artifact_sha256']
            and metadata['artifact_sha256']['inventory.json'] == digest(files['v23_inventory']),
            'V23 plan and catalogue replay provenance differ')
    checked_artifacts(files['v23_inventory'].parent, metadata['artifact_sha256'])
    nominated = inventory['nominal_roster']
    require(nominated == protocol['roster'] and tuple(row['acquisition_date'] for row in nominated) == DATES,
            'Preflight roster must match the exact two V23 nominees')
    planned = plan['roster']
    require(len(planned) == len({row['group_id'] for row in planned}) == 2
            and [row['group_id'] for row in planned] == inventory['group_ids']
            and plan['catalogue_requirements'] == inventory['catalogue_requirements']
            and plan['catalogue_requirements']['target_tile'] == TILE
            and plan['catalogue_requirements']['crs'] == 'EPSG:32636'
            and all(row['grid']['crs'] == 'EPSG:32636' and row['grid']['width'] == row['grid']['height'] == 64
                    and row['qa_context_bounds'] == [coord+delta for coord, delta in
                        zip(row['grid']['bounds'], (-1200, -1200, 1200, 1200))] for row in planned),
            'Fresh group identities or original QA envelope differ')
    features = raw_items(files['v23_inventory'], inventory)
    selections = []
    for row in nominated:
        feature = features[row['item_id']]
        props = feature['properties']
        sequence = props['s2:sequence']
        uri = row['product_uri']
        scene = uri.removesuffix('.SAFE')
        day = row['acquisition_date']
        require(sha(canonical(feature)) == row['raw_item_sha256'] and props['s2:product_uri'] == uri
                and uri.endswith('.SAFE') and re.fullmatch(
                    r'S2[AB]_MSIL1C_\d{8}T\d{6}_N0206_R064_T36UXA_\d{8}T\d{6}', scene)
                and scene.split('_')[2][:8] == day.replace('-', '')
                and isinstance(sequence, str) and re.fullmatch(r'[0-9]{1,3}', sequence)
                and row['item_id'] == f'{scene[:3]}_{TILE}_{day.replace("-", "")}_{sequence}_L1C',
                'Raw metadata product, date, tile or sequence differs')
        eligibility = row['case_eligibility']
        require(len(eligibility) == 2 and all(
            all(item[key] == case[key] for key in ('event_id', 'group_id', 'pair_id', 'post_date'))
            and item['nominal_metadata_followup_eligible'] is True and not item['followup_blockers']
            and item['catalogue_footprint_covers'] == {'core': True, 'qa_context': True}
            and 5 <= item['interval_to_post_days'] <= 30 for item, case in zip(eligibility, planned)),
            'A nominated scene lacks the replayed per-group footprint or temporal scope')
        year, month, day_of_month = map(int, day.split('-'))
        stamp = f'{year}/{month}/{day_of_month}'
        selections.append({'item_id': row['item_id'], 'raw_item_sha256': row['raw_item_sha256'],
            'scene_id': scene, 'acquisition_date': day, 'sequence': sequence,
            'tile_prefix': f'tiles/36/U/XA/{stamp}/{sequence}/',
            'product_key': f'products/{stamp}/{scene}/metadata.xml', 'case_eligibility': eligibility})
    require([{key: row[key] for key in ('acquisition_date', 'scene_id', 'item_id')}
             for row in selections] == protocol['catalogue_roster'], 'V24 selected identities differ')
    return selections, features, planned


def parse_metadata(product_bytes, tile_bytes, selection, feature, planned):
    from annotation_scene_inventory import _grid
    from pixelqa_scene_metadata import _angle

    product = ET.fromstring(product_bytes)
    tile = ET.fromstring(tile_bytes)
    require(product.tag.rsplit('}', 1)[-1] == 'Level-1C_User_Product'
            and tile.tag.rsplit('}', 1)[-1] == 'Level-1C_Tile_ID', 'Provider XML must describe L1C')
    scene = selection['scene_id']
    identity = {name: single(product, name) for name in ('PRODUCT_URI', 'PROCESSING_LEVEL', 'PROCESSING_BASELINE',
                'SPACECRAFT_NAME', 'SENSING_ORBIT_NUMBER', 'QUANTIFICATION_VALUE', 'PRODUCT_START_TIME')}
    expected = {'PRODUCT_URI': scene+'.SAFE', 'PROCESSING_LEVEL': 'Level-1C',
                'PROCESSING_BASELINE': '02.06', 'SPACECRAFT_NAME': 'Sentinel-2'+scene[2]}
    require(all(identity[name] == value for name, value in expected.items()),
            'Provider product identity or baseline differs from the nominated scene')
    orbit = int(identity['SENSING_ORBIT_NUMBER'])
    quantification = float(identity['QUANTIFICATION_VALUE'])
    require(orbit == 64 and quantification == 10000
            and timestamp(identity['PRODUCT_START_TIME']).date().isoformat() == selection['acquisition_date'],
            'Product orbit, quantification or start date differs')
    offsets = {}
    for element in product.iter():
        if element.tag.rsplit('}', 1)[-1] != 'RADIO_ADD_OFFSET':
            continue
        key = element.attrib.get('band_id', element.attrib.get('bandId'))
        value = float(element.text)
        require(key is not None and key not in offsets and math.isfinite(value) and value == 0,
                'Unsupported, duplicated or nonzero radiometric offset')
        offsets[key] = value
    require(not offsets or set(offsets) == {str(index) for index in range(13)}, 'Incomplete XML offset list')

    granules = [element for element in product.iter() if element.tag.rsplit('}', 1)[-1] == 'Granule']
    require(len(granules) == 1, 'Exactly one product granule required')
    granule = granules[0]
    tile_id = single(tile, 'TILE_ID')
    datastrip = single(tile, 'DATASTRIP_ID')
    sensed = single(tile, 'SENSING_TIME')
    props = feature['properties']
    require(granule.attrib.get('granuleIdentifier') == tile_id == props.get('s2:granule_id')
            and granule.attrib.get('datastripIdentifier') == datastrip == props.get('s2:datastrip_id')
            and timestamp(sensed).date().isoformat() == selection['acquisition_date']
            and timestamp(sensed) == timestamp(props['datetime']),
            'Product/tile/catalogue identity or sensing time differs')
    orbit_code = re.search(r'_A(\d{6})_T36UXA_N02\.06$', tile_id)
    require(orbit_code is not None, 'Tile XML does not identify the native 36UXA N0206 granule')
    folder = f'L1C_T36UXA_A{orbit_code[1]}_{timestamp(sensed).strftime("%Y%m%dT%H%M%S")}'
    path_pattern = re.compile(rf'GRANULE/{folder}/IMG_DATA/T36UXA_{scene.split("_")[2]}_(B(?:0[1-9]|1[012]|8A))')
    image_paths = {}
    for path in texts(granule, 'IMAGE_FILE'):
        if path.endswith('_TCI'):
            continue
        match = path_pattern.fullmatch(path)
        require(match is not None and match[1] not in image_paths, 'Unsupported or repeated granule image path')
        image_paths[match[1]] = path+'.jp2'
    require(set(BANDS) <= image_paths.keys(), 'Product XML is missing required image bands')

    issues = []
    crs, grid = _grid(tile, issues)
    require(crs == 'EPSG:32636' and grid is not None and not issues, 'Invalid or unsupported native 10 m grid')
    covered = {}
    x0, y0, x1, y1 = grid['bounds']
    for case in planned:
        covered[case['event_id']] = {}
        for name, bounds in (('core', case['grid']['bounds']), ('qa_context', case['qa_context_bounds'])):
            covered[case['event_id']][name] = x0 <= bounds[0] < bounds[2] <= x1 and y0 <= bounds[1] < bounds[3] <= y1
            if not covered[case['event_id']][name]:
                issues.append('actual_grid_missing_full_'+name+':'+case['event_id'])
        shifts = [(actual-required)/10 for actual, required in
                  zip(grid['transform'][2:6:3], case['grid']['transform'][2:6:3])]
        if any(abs(shift-round(shift)) > 1e-6 for shift in shifts):
            issues.append('actual_grid_pixel_alignment_conflict:'+case['event_id'])
    angles = [element for element in tile.iter() if element.tag.rsplit('}', 1)[-1] == 'Mean_Sun_Angle']
    require(len(angles) == 1, 'Exactly one mean solar angle group required')
    zenith = _angle(angles[0], 'ZENITH_ANGLE', issues, maximum=90)
    azimuth = _angle(angles[0], 'AZIMUTH_ANGLE', issues, maximum=360)
    return {'identity': identity, 'actual_relative_orbit': orbit, 'quantification_value': quantification,
            'radio_add_offsets': offsets, 'xml_offset_elements_present': bool(offsets),
            'offset_interpretation': ('Explicit zero offsets in actual XML' if offsets
                else 'Absent list in actual N0206 XML; frozen-reader zero-offset convention'),
            'tile_id': tile_id, 'datastrip_id': datastrip, 'tile_sensing_time': sensed,
            'crs': crs, 'grid_10m': grid, 'fixed_event_coverage': covered,
            'solar_zenith_deg': zenith, 'solar_azimuth_deg': azimuth,
            'granule_folder': folder, 'image_paths': image_paths, 'issues': sorted(set(issues))}


def check_sources(roster, raw_features, planned, network, provider):
    results = []
    for index, selected in enumerate(roster, 1):
        feature = raw_features[selected['item_id']]
        row = {**selected, 'status': BLOCKED, 'issues': [], 'assets': [], 'required_assets': 58,
               'measured_payload_bytes': None, 'local_qa_ready': False, 'image_content_downloaded': False,
               'source_approved': False, 'catalogue_asset_href_conflicts': {
                   name: asset['href'] for name, asset in feature['assets'].items()
                   if 'sentinel-s2-l2a' in asset.get('href', '')}}
        prefix = f'source-{index:02d}/'
        try:
            listing = network.call('GET', listing_url(selected, provider), prefix+'listing.xml')
            listed = parse_listing(listing, selected['tile_prefix'], provider['listing_max_keys'])
            row['listed_objects'] = len(listed)
            product_pin = network.call('HEAD', BASE+selected['product_key'], None)
            tile_key = selected['tile_prefix']+'metadata.xml'
            require(tile_key in listed, 'Missing listed tile XML')
            product = network.call('GET', BASE+selected['product_key'], prefix+'product.xml', product_pin)
            tile = network.call('GET', BASE+tile_key, prefix+'tile.xml', listed[tile_key])
            metadata = parse_metadata(product, tile, selected, feature, planned)
            assets, issues = asset_manifest(selected, metadata, listed, product_pin)
            row.update(metadata=metadata, assets=assets, issues=metadata['issues']+issues)
            if len(assets) == 58:
                row['measured_payload_bytes'] = sum(asset['size'] for asset in assets)
                if not row['issues']:
                    row['status'] = READY
        except (HTTPException, OSError, ValueError, TypeError, KeyError, ET.ParseError) as error:
            row['issues'].append(f'{type(error).__name__}: {error}')
        results.append(row)
    return results


def evidence_hashes(directory):
    return {str(path.relative_to(directory)): digest(path)
            for path in sorted(directory.rglob('*')) if path.is_file() and path != directory/'metadata.json'}


def build_preflight(output, protocol, checksum, roster, features, planned, free_bytes=None, opener=None):
    provider = protocol['provider']
    network = MetadataNetwork(provider, roster, output, opener=opener)
    results = check_sources(roster, features, planned, network, provider)
    write(output/'requests.json', {'provider': network.requests})
    if free_bytes is None:
        free_bytes = shutil.disk_usage(ROOT/'Datasets/Sentinel-2').free
    artifact_bytes = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    ready = [row for row in results if row['status'] == READY]
    budget = disk_budget(ready, free_bytes, artifact_bytes)
    if not ready:
        budget.update(complete_payload_bytes=None, future_two_copy_staging_plus_temporary_bytes=None,
                      sufficient_for_conservative_future_acquisition=False)
    approved = bool(ready and budget['sufficient_for_conservative_future_acquisition'])
    status = ('metadata_preflight_passed_no_acquisition' if approved
              else 'metadata_preflight_blocked_no_acquisition')
    write(output/'preflight.json', {'schema_version': 24, 'status': status, 'protocol_sha256': checksum,
          'group_ids': [row['group_id'] for row in planned], 'results': results, 'disk_budget': budget,
          'metadata_request_count': len(network.requests), 'metadata_network_bytes': network.bytes})
    report = ['# Fresh 36UXA source preflight', '', status, '']
    for row in results:
        report.append(f'- {row["acquisition_date"]} · {row["status"]} · {len(row["assets"])}/58 pinned assets')
        report.extend(f'  - {issue}' for issue in row['issues'])
    report.extend(['', 'Actual product and tile XML were inspected. JP2 and GML availability is based on the provider listing.'])
    (output/'report.md').write_text('\n'.join(report)+'\n')
    return len(network.requests)


class ReplayResponse(BytesIO):
    def __init__(self, raw, receipt):
        super().__init__(raw)
        self.url = receipt['url']
        self.status = receipt['status']
        self.headers = Message()
        for name, value in receipt['response_headers'].items():
            self.headers[name] = value

    def geturl(self):
        return self.url


def verify(directory, output, protocol, checksum, roster, features, planned):
    directory = directory.resolve()
    metadata = load(directory/'metadata.json')
    require(metadata['mode'] == 'fresh_source_preflight' and metadata['execution_status'] == 'completed'
            and metadata['protocol_sha256'] == checksum
            and metadata['artifact_sha256'] == evidence_hashes(directory), 'V24 preflight evidence differs')
    checked_artifacts(directory, metadata['artifact_sha256'])
    previous = load(directory/'preflight.json')
    receipts = iter(load(directory/'requests.json')['provider'])

    def opener(request, timeout):
        receipt = next(receipts, None)
        require(receipt is not None and receipt['url'] == request.full_url
                and receipt['method'] == request.get_method() and receipt['verified'] is True,
                'Replay requires the exact successful provider request sequence')
        raw = b''
        if receipt['artifact'] is not None:
            path = directory/receipt['artifact']
            require(path.stat().st_size == receipt['bytes'] <= protocol['provider']['maximum_xml_bytes']
                    and digest(path) == receipt['sha256'], 'Archived provider response changed')
            raw = path.read_bytes()
        return ReplayResponse(raw, receipt)

    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        count = build_preflight(stage, protocol, checksum, roster, features, planned,
                                free_bytes=previous['disk_budget']['measured_free_bytes'], opener=opener)
        require(next(receipts, None) is None and evidence_hashes(stage) == metadata['artifact_sha256'],
                'Offline preflight reconstruction differs')
    write(output/'verification.json', {'status': 'passed', 'protocol_sha256': checksum,
          'preflight_metadata_sha256': digest(directory/'metadata.json'),
          'preflight_artifact_sha256': metadata['artifact_sha256'], 'replayed_requests': count,
          'ready_dates': [row['acquisition_date'] for row in previous['results'] if row['status'] == READY],
          'network_requests': 0})


def worker(args, output):
    checksum = digest(PROTOCOL)
    protocol = checked_protocol()
    roster, features, planned = inputs(protocol)
    if args.mode == 'preflight':
        requests = build_preflight(output, protocol, checksum, roster, features, planned)
    else:
        verify(args.preflight, output, protocol, checksum, roster, features, planned)
        requests = 0
    require(digest(PROTOCOL) == checksum and not {'torch', 'rasterio', 's2cloudless'} & sys.modules.keys(),
            'Protocol changed or an unexpected pixel/model library was imported')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {'execution_status': 'completed',
          'mode': 'fresh_source_preflight' if args.mode == 'preflight' else 'fresh_source_preflight_replay',
          'protocol_sha256': checksum, 'artifact_sha256': evidence_hashes(output), 'network_requests': requests,
          'image_asset_downloads': 0, 'satellite_pixel_reads': 0, 'model_executions': 0,
          'labels_approved': 0, 'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2)})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('preflight', 'verify'), default='preflight')
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    if (args.mode == 'verify') != (args.preflight is not None):
        parser.error('--preflight is required only for verify mode')
    for variable in THREAD_VARIABLES:
        os.environ[variable] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='fresh-source-preflight',
                    gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
