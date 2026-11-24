#!/usr/bin/env python3
"""Preflight the four V25 candidates using native Sentinel-2 listings and XML."""

import argparse
from http.client import HTTPException
import math
import os
from pathlib import Path
import re
import resource
import shutil
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

from inventory_2019_validation_sources import normalize_site_item, read_inputs as read_v25_inputs
from inventory_earlier_validation_sources import (
    PRODUCT, RESOURCES, ROOT, canonical, digest, require, rooted, sha, timestamp, write,
)
from preflight_earlier_validation_metadata import (
    BANDS, BASE, MetadataNetwork, asset_manifest, disk_budget, listing_url, parse_listing, single, texts,
)
from preflight_fresh_validation_sources import (
    PROVIDER_RULES, ReplayResponse, checked_artifacts, evidence_hashes, load,
)
from preflight_remaining_validation_metadata import BLOCKED, READY


PROTOCOL = ROOT/'Docs/Reviews/Validation2019PreflightV26/protocol.json'
IMPLEMENTATION = 'Solutions/preflight_2019_validation_sources.py'
MODE = 'validation_2019_native_metadata_preflight'
DATES = {'march': ('2019-03-03', '2019-03-08'), 'may': ('2019-05-07', '2019-05-12')}
PROVIDER = {**PROVIDER_RULES, 'maximum_total_response_bytes': 48*2**20,
            'maximum_requests': 16, 'timeout_seconds': 15}


def inputs(protocol_path):
    protocol = load(protocol_path)
    require(protocol['schema_version'] == 26 and protocol['resources'] == RESOURCES
            and protocol['provider'] == PROVIDER and protocol['required_files_per_scene'] == 58
            and protocol['acquisition_authorized'] is False, 'V26 preflight scope differs')
    files = {}
    for name, receipt in protocol['source_files'].items():
        path = rooted(receipt['path'])
        require(digest(path) == receipt['sha256'], 'Pinned V26 source differs: '+name)
        files[name] = path
    implementations = protocol['implementation_sha256']
    require(IMPLEMENTATION in implementations, 'Pin the preflight worker before use')
    for path, checksum in implementations.items():
        require(digest(rooted(path)) == checksum, 'Pinned V26 implementation differs: '+path)

    sites, evidence, v25_sha = read_v25_inputs(files['v25_protocol'])
    inventory, metadata, replay, integrity = (load(files[name]) for name in
        ('v25_inventory', 'v25_metadata', 'v25_replay', 'v25_integrity'))
    require(inventory['status'] == 'complete_2019_sentinel_catalogue_inventory'
            and metadata['execution_status'] == 'completed' and replay['status'] == integrity['status'] == 'passed'
            and replay['network_requests'] == 0 and replay['raw_pages_replayed'] == inventory['catalogue_pages'] == 4
            and all(item['protocol_sha256'] == v25_sha for item in (inventory, metadata, replay, integrity))
            and replay['inventory_metadata_sha256'] == digest(files['v25_metadata'])
            and replay['inventory_artifact_sha256'] == metadata['artifact_sha256']
            and metadata['artifact_sha256']['inventory.json'] == digest(files['v25_inventory'])
            and inventory['selected_roster_items'] == 0 and inventory['labels_approved'] == 0,
            'Preflight needs the verified V25 inventory and replay')
    inventory_root = files['v25_inventory'].parent
    checked_artifacts(inventory_root, metadata['artifact_sha256'])
    selections = []
    features = {}
    require(len(sites) == len(inventory['sites']) == 2, 'V25 site count differs')
    for site, catalogue in zip(sites, inventory['sites']):
        site_id = site['site_id']
        require(all(catalogue[key] == value for key, value in site.items())
                and catalogue['pagination_complete'] and catalogue['distinct_items'] == len(catalogue['items']),
                'V25 site identity, footprint or catalogue completeness differs: '+site_id)
        features_by_id = {}
        for page_index, page in enumerate(catalogue['pages'], 1):
            path = (inventory_root/page['response']).resolve()
            require(path.is_relative_to(inventory_root/site_id/'raw') and path.is_file()
                    and path.stat().st_size == page['bytes'] <= site['query']['maximum_response_bytes']
                    and digest(path) == page['sha256'], 'V25 raw page differs: '+site_id)
            for feature in load(path)['features']:
                item_id = feature['id']
                item = features_by_id.setdefault(item_id, {'feature': feature, 'sha256': sha(canonical(feature)),
                                                           'pages': []})
                require(feature['collection'] == 'sentinel-2-l1c'
                        and item['feature'] == feature,
                        'V25 raw item collection or duplicate differs: '+item_id)
                item['pages'].append(page_index)
        eligible = sorted((row for row in catalogue['items'] if row['nominal_metadata_followup_eligible']),
                          key=lambda row: row['acquisition_date'])
        require(tuple(row['acquisition_date'] for row in eligible) == DATES[site_id],
                'V26 must check every fixed V25 candidate: '+site_id)
        for row in eligible:
            item_id = row['item_id']
            item = features_by_id[item_id]
            feature = item['feature']
            require(normalize_site_item(item, site, evidence['prior_local_failures_by_site'][site_id]) == row
                    and row['catalogue_footprint_covers'] == {'core': True, 'qa_context': True}
                    and row['interval_to_post_days'] in (25, 30) and not row['followup_blockers'],
                    'V25 candidate does not match raw evidence: '+item_id)
            scene = row['product_uri'].removesuffix('.SAFE')
            identity = PRODUCT.fullmatch(row['product_uri'])
            sequence = feature['properties']['s2:sequence']
            day = row['acquisition_date']
            require(identity is not None and identity['tile'] == '36UYA' and identity['orbit'] == 'R064'
                    and identity['baseline'] == 'N0207' and sequence == '0'
                    and item_id == f'{scene[:3]}_36UYA_{day.replace("-", "")}_{sequence}_L1C'
                    and item_id not in features, 'V26 source prefix cannot be derived: '+item_id)
            year, month, dom = map(int, day.split('-'))
            stamp = f'{year}/{month}/{dom}'
            selections.append({'site_id': site_id, 'item_id': item_id, 'scene_id': scene,
                'acquisition_date': day, 'raw_item_sha256': row['raw_item_sha256'], 'sequence': sequence,
                'tile_prefix': f'tiles/36/U/YA/{stamp}/{sequence}/',
                'product_key': f'products/{stamp}/{scene}/metadata.xml',
                'pair_id': site['pair_id'], 'event_id': site['event_id'], 'group_id': site['group_id'],
                'post_date': site['post_date'], 'post_scene_id': site['post_scene_id'],
                'interval_to_post_days': row['interval_to_post_days']})
            features[item_id] = feature
    require(selections == protocol['metadata_candidates'], 'Pinned V26 candidate list differs')
    return protocol, selections, features, {site['site_id']: site for site in sites}


def native_timestamp(value):
    # Python 3.10 accepts three or six fractional digits. SAFE XML can use five.
    return timestamp(re.sub(r'(\.\d{1,6})(?=Z$)', lambda match: match[1].ljust(7, '0'), value))


def parse_metadata(product_bytes, tile_bytes, selection, feature, site):
    from annotation_scene_inventory import _grid
    from pixelqa_scene_metadata import _angle

    product = ET.fromstring(product_bytes)
    tile = ET.fromstring(tile_bytes)
    require(product.tag.rsplit('}', 1)[-1] == 'Level-1C_User_Product'
            and tile.tag.rsplit('}', 1)[-1] == 'Level-1C_Tile_ID', 'Provider XML must describe L1C')
    scene = selection['scene_id']
    identity = {name: single(product, name) for name in ('PRODUCT_URI', 'PROCESSING_LEVEL', 'PROCESSING_BASELINE',
                'SPACECRAFT_NAME', 'SENSING_ORBIT_NUMBER', 'QUANTIFICATION_VALUE', 'PRODUCT_START_TIME',
                'GENERATION_TIME')}
    expected = {'PRODUCT_URI': scene+'.SAFE', 'PROCESSING_LEVEL': 'Level-1C',
                'PROCESSING_BASELINE': '02.07', 'SPACECRAFT_NAME': 'Sentinel-2'+scene[2]}
    require(all(identity[name] == value for name, value in expected.items()),
            'Provider product identity or baseline differs from the catalogue product')
    orbit = int(identity['SENSING_ORBIT_NUMBER'])
    quantification = float(identity['QUANTIFICATION_VALUE'])
    day = selection['acquisition_date']
    product_start = native_timestamp(identity['PRODUCT_START_TIME'])
    generation = native_timestamp(identity['GENERATION_TIME'])
    acquisition = scene.split('_')[2]
    require(orbit == 64 and quantification == 10000
            and product_start.date().isoformat() == day
            and product_start.strftime('%Y%m%dT%H%M%S') == acquisition
            and generation.strftime('%Y%m%dT%H%M%S') == scene.split('_')[-1],
            'Product orbit, quantification, start time or generation differs')
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
    native_time = native_timestamp(sensed)
    catalogue_time = native_timestamp(props['datetime'])
    # Comparisons use UTC acquisition days. Exact IDs and generation bind the
    # native product and tile. Retain any intra-day catalogue time discrepancy.
    require(granule.attrib.get('granuleIdentifier') == tile_id == props.get('s2:granule_id')
            and granule.attrib.get('datastripIdentifier') == datastrip == props.get('s2:datastrip_id')
            and generation == native_timestamp(props['s2:generation_time'])
            and native_time.date().isoformat() == catalogue_time.date().isoformat() == day
            and product_start <= native_time < generation,
            'Product, tile, catalogue identity, generation or acquisition day differs')
    orbit_code = re.search(r'_A(\d{6})_T36UYA_N02\.07$', tile_id)
    datastrip_stamp = re.search(r'_S(\d{8}T\d{6})_N02\.07$', datastrip)
    require(orbit_code is not None and datastrip_stamp is not None
            and datastrip_stamp[1][:8] == day.replace('-', ''),
            'Tile XML does not identify the native 36UYA N0207 granule and datastrip')
    # Compact SAFE paths use the datastrip stamp even when tile sensing is later.
    folder = f'L1C_T36UYA_A{orbit_code[1]}_{datastrip_stamp[1]}'
    path_pattern = re.compile(rf'GRANULE/{folder}/IMG_DATA/T36UYA_{acquisition}_(B(?:0[1-9]|1[012]|8A)|TCI)')
    image_paths = {}
    for path in texts(granule, 'IMAGE_FILE'):
        match = path_pattern.fullmatch(path)
        require(match is not None and match[1] not in image_paths, 'Unsupported or repeated granule image path')
        image_paths[match[1]] = path+'.jp2'
    image_paths.pop('TCI', None)
    require(set(BANDS) <= image_paths.keys(), 'Product XML is missing required image bands')

    issues = []
    crs, grid = _grid(tile, issues)
    require(crs == site['grid']['crs'] == 'EPSG:32636' and grid is not None and not issues,
            'Invalid or unsupported native 10 m grid')
    x0, y0, x1, y1 = grid['bounds']
    covered = {}
    for name, bounds in (('core', site['grid']['bounds']), ('qa_context', site['qa_context_bounds'])):
        covered[name] = x0 <= bounds[0] < bounds[2] <= x1 and y0 <= bounds[1] < bounds[3] <= y1
        if not covered[name]:
            issues.append('actual_grid_missing_full_'+name)
    shifts = [(actual-required)/10 for actual, required in
              zip(grid['transform'][2:6:3], site['grid']['transform'][2:6:3])]
    if any(abs(shift-round(shift)) > 1e-6 for shift in shifts):
        issues.append('actual_grid_pixel_alignment_conflict')
    angles = [element for element in tile.iter() if element.tag.rsplit('}', 1)[-1] == 'Mean_Sun_Angle']
    require(len(angles) == 1, 'Exactly one mean solar angle group required')
    zenith = _angle(angles[0], 'ZENITH_ANGLE', issues, maximum=90)
    azimuth = _angle(angles[0], 'AZIMUTH_ANGLE', issues, maximum=360)
    return {'identity': identity, 'actual_relative_orbit': orbit, 'quantification_value': quantification,
            'radio_add_offsets': offsets, 'xml_offset_elements_present': bool(offsets),
            'offset_interpretation': ('Explicit zero offsets in actual XML' if offsets
                else 'Absent list in actual N0207 XML; frozen-reader zero-offset convention'),
            'tile_id': tile_id, 'datastrip_id': datastrip, 'tile_sensing_time': sensed,
            'catalogue_datetime': props['datetime'],
            'catalogue_minus_native_seconds': (catalogue_time-native_time).total_seconds(),
            'acquisition_day_authority': 'native Level-1C tile sensing time and exact product identity',
            'datastrip_timestamp': datastrip_stamp[1], 'crs': crs, 'grid_10m': grid,
            'fixed_event_coverage': {site['event_id']: covered},
            'solar_zenith_deg': zenith, 'solar_azimuth_deg': azimuth,
            'granule_folder': folder, 'image_paths': image_paths, 'issues': sorted(set(issues))}


def build_preflight(output, protocol, checksum, candidates, features, sites, *, free_bytes=None, opener=None):
    provider = protocol['provider']
    network = MetadataNetwork(provider, candidates, output, opener=opener)
    results = []
    for selected in candidates:
        feature = features[selected['item_id']]
        row = {**selected, 'status': BLOCKED, 'issues': [], 'assets': [], 'measured_payload_bytes': None,
               'catalogue_l2a_href_conflicts': {name: asset['href'] for name, asset in feature['assets'].items()
                                                if 'sentinel-s2-l2a' in asset.get('href', '')}}
        prefix = f'raw/{selected["item_id"]}/'
        try:
            listing = network.call('GET', listing_url(selected, provider), prefix+'listing.xml')
            listed = parse_listing(listing, selected['tile_prefix'], provider['listing_max_keys'])
            row['listed_objects'] = len(listed)
            product_pin = network.call('HEAD', BASE+selected['product_key'], None)
            tile_key = selected['tile_prefix']+'metadata.xml'
            require(tile_key in listed, 'Missing listed native tile XML')
            product = network.call('GET', BASE+selected['product_key'], prefix+'product.xml', product_pin)
            tile = network.call('GET', BASE+tile_key, prefix+'tile.xml', listed[tile_key])
            metadata = parse_metadata(product, tile, selected, feature, sites[selected['site_id']])
            assets, issues = asset_manifest(selected, metadata, listed, product_pin)
            row.update(metadata=metadata, issues=metadata['issues']+issues, assets=assets)
            if len(assets) == protocol['required_files_per_scene']:
                row['measured_payload_bytes'] = sum(asset['size'] for asset in assets)
                if not row['issues']:
                    row['status'] = READY
        except (HTTPException, OSError, ValueError, TypeError, KeyError, ET.ParseError) as error:
            row['issues'].append(f'{type(error).__name__}: {error}')
        results.append(row)

    write(output/'requests.json', {'provider': network.requests})
    free_bytes = shutil.disk_usage(ROOT/'Datasets/Sentinel-2').free if free_bytes is None else free_bytes
    evidence_bytes = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    ready = [row for row in results if row['status'] == READY]
    budget = disk_budget(ready, free_bytes, evidence_bytes)
    if not ready:
        budget.update(complete_payload_bytes=None, future_two_copy_staging_plus_temporary_bytes=None,
                      sufficient_for_conservative_future_acquisition=False)
    budget['included_ready_scenes'] = [row['scene_id'] for row in ready]
    write(output/'preflight.json', {'schema_version': 26, 'protocol_sha256': checksum,
          'results': results, 'disk_budget': budget, 'metadata_requests': len(network.requests),
          'metadata_network_bytes': network.bytes, 'candidate_count': len(candidates), 'ready_count': len(ready),
          'prior_rejected_dates': {key: site['prior_rejected_pre_dates'] for key, site in sites.items()},
          'acquisition_roster_selected': False, 'image_assets_downloaded': 0, 'labels_approved': 0})
    return len(network.requests)


def rebuild_archive(directory, output, protocol, checksum, candidates, features, sites, archive_checksum):
    directory = directory.resolve()
    metadata = load(directory/'metadata.json')
    require(metadata['mode'] == MODE and metadata['execution_status'] == 'completed'
            and metadata['protocol_sha256'] == archive_checksum
            and metadata['artifact_sha256'] == evidence_hashes(directory), 'V26 preflight archive differs')
    checked_artifacts(directory, metadata['artifact_sha256'])
    prior = load(directory/'preflight.json')
    receipts = iter(load(directory/'requests.json')['provider'])

    def opener(request, timeout):
        receipt = next(receipts, None)
        require(receipt is not None and receipt['url'] == request.full_url
                and receipt['method'] == request.get_method()
                and {key.lower(): value for key, value in request.header_items()}
                == {key.lower(): value for key, value in receipt['request_headers'].items()}
                and receipt['verified'] is True and timeout == protocol['provider']['timeout_seconds'],
                'Replay needs the exact successful metadata requests')
        raw = b''
        if receipt['artifact'] is not None:
            path = directory/receipt['artifact']
            require(path.resolve().is_relative_to(directory/'raw') and path.stat().st_size == receipt['bytes']
                    <= protocol['provider']['maximum_xml_bytes'] and digest(path) == receipt['sha256'],
                    'Replay response bytes differ')
            raw = path.read_bytes()
        return ReplayResponse(raw, receipt)

    count = build_preflight(output, protocol, checksum, candidates, features, sites,
                           free_bytes=prior['disk_budget']['measured_free_bytes'], opener=opener)
    require(next(receipts, None) is None, 'Archived metadata requests remain unused')
    return count, metadata


def verify(directory, output, protocol, checksum, candidates, features, sites):
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        count, metadata = rebuild_archive(directory, stage, protocol, checksum, candidates, features, sites,
                                          archive_checksum=checksum)
        require(evidence_hashes(stage) == metadata['artifact_sha256'], 'V26 preflight replay differs')
    write(output/'verification.json', {'status': 'passed', 'protocol_sha256': checksum,
          'preflight_metadata_sha256': digest(directory/'metadata.json'),
          'preflight_artifact_sha256': metadata['artifact_sha256'],
          'replayed_metadata_requests': count, 'ready_count': load(directory/'preflight.json')['ready_count'],
          'network_requests': 0})


def worker(args, output):
    checksum = digest(args.protocol)
    protocol, candidates, features, sites = inputs(args.protocol)
    requests = 0
    if args.preflight is not None:
        verify(args.preflight, output, protocol, checksum, candidates, features, sites)
    elif args.recheck:
        record = protocol['source_files']['initial_preflight_metadata']
        archive = rooted(record['path']).parent
        rebuild_archive(archive, output, protocol, checksum, candidates, features, sites,
                        archive_checksum=protocol['initial_protocol_sha256'])
    else:
        requests = build_preflight(output, protocol, checksum, candidates, features, sites)
    require(digest(args.protocol) == checksum and not {'torch', 'rasterio', 's2cloudless'} & sys.modules.keys(),
            'Protocol changed or a pixel/model library was imported')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {'mode': MODE if args.preflight is None else MODE+'_replay',
          'execution_status': 'completed', 'protocol_sha256': checksum, 'artifact_sha256': evidence_hashes(output),
          'network_requests': requests, 'rechecked_archived_provider_responses': args.recheck,
          'image_asset_downloads': 0, 'satellite_pixel_reads': 0,
          'model_executions': 0, 'labels_approved': 0,
          'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2)})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', type=Path, help='Replay a completed preflight byte for byte')
    mode.add_argument('--recheck', action='store_true', help='Recheck the pinned initial provider responses offline')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=RESOURCES['worker_rss_mib'],
                        min_available_gib=RESOURCES['available_memory_reserve_gib'],
                        max_seconds=RESOURCES['max_seconds_per_worker'])
    args = parser.parse_args()
    for variable in THREAD_VARIABLES:
        os.environ[variable] = '1'
    if args.worker_output:
        os.nice(RESOURCES['nice_increment'])
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='2019-native-preflight',
                    gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
