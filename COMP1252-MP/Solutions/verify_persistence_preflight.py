#!/usr/bin/env python3
"""Replay the V20 August metadata evidence offline."""

import argparse
from datetime import date
import hashlib
from pathlib import Path
import sys

from inventory_earlier_validation_sources import ROOT, digest, read_json, require, rooted
from preflight_earlier_validation_metadata import BASE, asset_manifest, disk_budget, listing_url, parse_listing, parse_metadata, pin
from preflight_persistence_source import DATES, ITEMS, PROTOCOL, catalogue_row, checked_protocol, fixed_roster
from preflight_remaining_validation_metadata import BLOCKED, READY


def read(path):
    return read_json(path.read_bytes())


def artifact_path(directory, name):
    path = (directory/name).resolve()
    require(path.is_relative_to(directory) and path.is_file(), 'Missing or escaping evidence: '+name)
    return path


def response_bytes(directory, entry, limit):
    path = artifact_path(directory, entry['artifact'])
    require(path.stat().st_size == entry['bytes'] <= limit and digest(path) == entry['sha256'],
            'Response size or hash differs: '+entry['artifact'])
    return path.read_bytes()


def header(receipt, name):
    return next((value for key, value in receipt['response_headers'].items()
                 if key.lower() == name.lower()), None)


def replay_xml(directory, receipt, expected_pin, limit):
    raw = response_bytes(directory, receipt, limit)
    md5 = hashlib.md5(raw).hexdigest()
    require(receipt['request_headers']['If-Match'] == header(receipt, 'ETag') == expected_pin['etag']
            and len(raw) == int(header(receipt, 'Content-Length')) == expected_pin['size']
            and md5 == receipt['md5'] == expected_pin['etag'][1:-1]
            and receipt['if_match_verified'] and receipt['content_length_verified'],
            'Product or tile XML does not match its provider pin')
    return raw


def replay_ready(directory, selection, feature, cover, row, receipts, requirements, provider):
    require(len(receipts) == 4 and all(entry['status'] == 200 and entry['verified'] for entry in receipts),
            'Ready source requires all four successful provider requests')
    listing, product_head, product_get, tile_get = receipts
    listed = parse_listing(response_bytes(directory, listing, provider['maximum_listing_bytes']),
                           selection['tile_prefix'], provider['listing_max_keys'])
    product_pin = pin(int(header(product_head, 'Content-Length')), header(product_head, 'ETag'))
    require(product_head['pin'] == product_pin, 'Product HEAD pin differs from saved headers')
    tile_entry = listed[selection['tile_prefix']+'metadata.xml']
    tile_pin = pin(tile_entry['size'], tile_entry['etag'])
    xml_limit = provider['maximum_xml_bytes']
    product = replay_xml(directory, product_get, product_pin, xml_limit)
    tile = replay_xml(directory, tile_get, tile_pin, xml_limit)
    metadata = parse_metadata(product, tile, selection, feature, requirements)
    assets, issues = asset_manifest(selection, metadata, listed, product_pin)
    href_conflicts = {name: asset['href'] for name, asset in feature['assets'].items()
                      if 'sentinel-s2-l2a' in asset.get('href', '')}
    require(row['metadata'] == metadata and not metadata['issues'] and not row['issues'] and not issues
            and row['assets'] == assets and len(assets) == row['required_assets'] == 58
            and row['measured_payload_bytes'] == sum(asset['size'] for asset in assets)
            and row['listed_objects'] == len(listed)
            and row['catalogue_asset_href_conflicts'] == href_conflicts
            and row['catalogue_footprint_covers'] == cover,
            'Listing, metadata or manifest replay differs: '+row['acquisition_date'])


def verify(directory):
    directory = directory.resolve()
    require(directory.is_relative_to(ROOT), 'Preflight evidence must be inside the project')
    protocol = checked_protocol()
    checksum = digest(PROTOCOL)
    roster = fixed_roster(protocol)
    requirements = read(rooted(protocol['source_files']['requirements']['path']))
    metadata = read(directory/'metadata.json')
    require(metadata['execution_status'] == 'completed' and metadata['mode'] == 'persistence_source_preflight'
            and metadata['protocol_sha256'] == checksum
            and all(metadata[key] == 0 for key in ('image_asset_downloads', 'satellite_pixel_reads', 'model_executions')),
            'Missing completed metadata-only preflight receipt')
    artifacts = metadata['artifact_sha256']
    require({'requests.json', 'preflight.json'} <= artifacts.keys(), 'Core preflight artifacts are missing')
    for name, expected_hash in artifacts.items():
        require(digest(artifact_path(directory, name)) == expected_hash, 'Artifact hash differs: '+name)

    preflight = read(directory/'preflight.json')
    requests = read(directory/'requests.json')
    require(preflight['schema_version'] == 20 and preflight['protocol_sha256'] == checksum
            and preflight['group_id'] == requirements['group_id']
            and [row['acquisition_date'] for row in preflight['results']] == list(DATES),
            'Preflight scope or date coverage differs')
    require(len(requests['catalogue']) <= len(roster), 'Catalogue receipt count exceeds roster')
    catalogue = {entry['url']: entry for entry in requests['catalogue']}
    require(len(catalogue) == len(requests['catalogue'])
            and set(catalogue) <= {ITEMS+row['item_id'] for row in roster}, 'Unexpected catalogue request')

    provider = protocol['provider']
    provider_requests = iter(requests['provider'])
    request = next(provider_requests, None)
    checked_artifacts = {'preflight.json', 'requests.json'}
    ready, blocked, catalogue_rows = [], [], 0
    provider_bytes = 0
    for expected, row in zip(roster, preflight['results']):
        day = expected['acquisition_date']
        require(all(row[key] == value for key, value in expected.items())
                and row['status'] in (READY, BLOCKED)
                and not any(row.get(key, False) for key in ('local_qa_ready', 'image_content_downloaded', 'source_approved')),
                'Preflight row identity, status or approval scope differs')
        receipt = catalogue.get(ITEMS+expected['item_id'])
        if receipt is None:
            require(row['status'] == BLOCKED and row['issues'] and not row['assets'], 'Missing catalogue evidence')
            blocked.append(day)
            continue
        catalogue_name = f'catalogue-{day}.json'
        require(receipt['status'] == 200 and receipt['artifact'] == catalogue_name, 'Catalogue receipt differs')
        checked_artifacts.add(catalogue_name)
        raw = response_bytes(directory, receipt, protocol['catalogue_response_bytes'])
        try:
            feature, selection, cover = catalogue_row(raw, expected, requirements)
        except (ValueError, TypeError, KeyError) as error:
            require(row['status'] == BLOCKED and not row['assets']
                    and row['issues'] == [f'{type(error).__name__}: {error}'], 'Catalogue failure replay differs')
            blocked.append(day)
            continue

        catalogue_rows += 1
        prefix = f'source-{catalogue_rows:02d}/'
        expected_requests = (
            ('GET', listing_url(selection, provider), prefix+'listing.xml', provider['maximum_listing_bytes']),
            ('HEAD', BASE+selection['product_key'], None, 0),
            ('GET', BASE+selection['product_key'], prefix+'product.xml', provider['maximum_xml_bytes']),
            ('GET', BASE+selection['tile_prefix']+'metadata.xml', prefix+'tile.xml', provider['maximum_xml_bytes']),
        )
        receipts = []
        for method, url, name, limit in expected_requests:
            if request is None or (request['method'], request['url']) != (method, url):
                break
            artifact = request['artifact']
            require(artifact == name or (artifact is None and not request.get('verified')),
                    'Unexpected provider response artifact')
            if artifact:
                response_bytes(directory, request, limit)
                checked_artifacts.add(artifact)
                provider_bytes += request['bytes']
            receipts.append(request)
            request = next(provider_requests, None)
        require(row['raw_item_sha256'] == selection['raw_item_sha256']
                and row['catalogue_footprint_covers'] == cover
                and all(row[key] == selection[key] for key in ('sequence', 'tile_prefix', 'product_key'))
                and row['days_after_july_31'] == (date.fromisoformat(day)-date(2018, 7, 31)).days,
                'Catalogue selection or interval differs')
        if row['status'] == READY:
            replay_ready(directory, selection, feature, cover, row, receipts, requirements, provider)
            ready.append(row)
        else:
            require(row['issues'], 'Blocked source must retain its reason')
            blocked.append(day)

    require(request is None and len(requests['provider']) <= provider['maximum_requests']
            and provider_bytes <= provider['maximum_total_response_bytes'] and set(artifacts) == checked_artifacts,
            'Unexpected provider request or unbound evidence artifact')
    budget = preflight['disk_budget']
    artifact_bytes = sum(artifact_path(directory, name).stat().st_size
                         for name in checked_artifacts if name != 'preflight.json')
    require(type(budget['measured_free_bytes']) is int and budget['measured_free_bytes'] >= 0,
            'Recorded free disk capacity is invalid')
    expected_budget = disk_budget(ready, budget['measured_free_bytes'], artifact_bytes)
    if not ready:
        expected_budget.update(complete_payload_bytes=None, future_two_copy_staging_plus_temporary_bytes=None,
                               sufficient_for_conservative_future_acquisition=False)
    status = ('metadata_preflight_passed_no_acquisition'
              if ready and expected_budget['sufficient_for_conservative_future_acquisition']
              else 'metadata_preflight_blocked_no_acquisition')
    require(budget == expected_budget and preflight['status'] == status, 'Disk budget or scope differs')
    require(not {'torch', 'rasterio', 's2cloudless'} & sys.modules.keys(), 'Unexpected pixel or model import')
    return {'status': 'passed', 'protocol_sha256': checksum, 'implementation_sha256': digest(Path(__file__)),
            'preflight_metadata_sha256': digest(directory/'metadata.json'),
            'preflight_sha256': artifacts['preflight.json'], 'artifacts_verified': len(artifacts),
            'catalogue_rows_replayed': catalogue_rows, 'manifest_rows_replayed': len(ready),
            'ready_dates': [row['acquisition_date'] for row in ready], 'blocked_dates': blocked,
            'provider_requests_verified': len(requests['provider']), 'disk_budget_replayed': True,
            'network_requests': 0, 'image_asset_downloads': 0, 'satellite_pixel_reads': 0, 'model_executions': 0}


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.preflight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()
