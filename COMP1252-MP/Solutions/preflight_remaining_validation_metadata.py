#!/usr/bin/env python3
"""Verify the remaining July sources using the frozen L1C metadata checks."""
import argparse
from collections import Counter
from datetime import date
from http.client import HTTPException
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import xml.etree.ElementTree as ET

from inventory_earlier_validation_sources import (
    ROOT, RESOURCES, canonical, digest, read_json, require, rooted, sha, write,
)
from preflight_earlier_validation_metadata import (
    BASE, DATES as CHECKED_DATES, MetadataNetwork, asset_manifest, disk_budget,
    listing_url, parse_listing, parse_metadata,
)

DATES = ('2018-07-11', '2018-07-06', '2018-07-01')
POST_DATE = date(2018, 7, 31)
READY = 'remote_metadata_and_asset_listing_passed'
BLOCKED = 'blocked_metadata_preflight'
PROTOCOL = ROOT/'Docs/Reviews/ValidationRemainingMetadataV16/protocol.json'


def select_roster(inventory, raw_page):
    raw_items = {item['id']: item for item in raw_page['features']}
    require(len(raw_items) == len(raw_page['features']), 'Duplicate raw catalogue ID')
    remaining = [row for row in inventory['items'] if row['nominal_metadata_followup_eligible']
                 and row['acquisition_date'] not in CHECKED_DATES]
    require(Counter(row['acquisition_date'] for row in remaining) == Counter(DATES),
            'Expected exactly one remaining eligible catalogue item for each fixed date')
    rows = {row['acquisition_date']: row for row in remaining}
    roster = []
    for day in DATES:
        row = rows[day]
        feature = raw_items.get(row['item_id'])
        require(feature is not None and sha(canonical(feature)) == row['raw_item_sha256'],
                'Raw catalogue item missing or hash differs')
        properties = feature['properties']
        scene = row['product_uri'].removesuffix('.SAFE')
        sequence = properties['s2:sequence']
        require(properties['s2:product_uri'] == row['product_uri']
                and re.fullmatch(r'S2[AB]_MSIL1C_\d{8}T\d{6}_N0206_R064_T36UXB_\d{8}T\d{6}', scene)
                and scene.split('_')[2][:8] == day.replace('-', '')
                and isinstance(sequence, str) and re.fullmatch(r'\d+', sequence),
                'Catalogue product or sequence differs from the fixed L1C identity')
        year, month, day_of_month = map(int, day.split('-'))
        stamp = f'{year}/{month}/{day_of_month}'
        roster.append({
            'item_id': row['item_id'], 'raw_item_sha256': row['raw_item_sha256'],
            'scene_id': scene, 'acquisition_date': day, 'sequence': sequence,
            'tile_prefix': f'tiles/36/U/XB/{stamp}/{sequence}/',
            'product_key': f'products/{stamp}/{scene}/metadata.xml',
        })
    return roster, raw_items


def inputs(protocol):
    require(protocol['schema_version'] == 16 and protocol['resources'] == RESOURCES,
            'V16 resource or scope mismatch')
    receipts = dict(protocol['source_files'])
    for section in ('frozen_implementation_sha256', 'implementation_sha256'):
        receipts.update({f'code/{path}': {'path': path, 'sha256': checksum}
                         for path, checksum in protocol[section].items()})
    require(str(Path(__file__).relative_to(ROOT)) in protocol['implementation_sha256'],
            'Preflight implementation is not pinned')
    for role, receipt in receipts.items():
        require(digest(rooted(receipt['path'])) == receipt['sha256'], f'Frozen input changed: {role}')

    def load(role):
        return read_json(rooted(receipts[role]['path']).read_bytes())

    roster, raw_items = select_roster(load('catalogue_inventory'), load('catalogue_raw_page'))
    selection = protocol['selection']
    require(roster == selection['roster'] and selection['count'] == len(DATES)
            and selection['post_date'] == POST_DATE.isoformat(), 'Roster differs from protocol')
    require(tuple(row['acquisition_date'] for row in load('prior_preflight')['results']) == CHECKED_DATES,
            'Prior preflight roster differs')
    prior = load('prior_earlier_screen')
    require(Counter(row['pre_date'] for row in prior['derived_pairs']) ==
            Counter({day: 2 for day in CHECKED_DATES})
            and len(prior['results']) == 4
            and all(row['quality_status'] == 'excluded_by_automatic_quality_screen'
                    for row in prior['results']), 'Prior QA exclusions differ')
    require(protocol['provider']['base_url'] == BASE
            and protocol['provider']['maximum_requests'] == 4*len(DATES)
            and protocol['provider']['maximum_listing_bytes'] == 2*2**20
            and protocol['provider']['maximum_xml_bytes'] == 5*2**20
            and protocol['provider']['maximum_total_response_bytes'] == 37*2**20
            and protocol['provider']['timeout_seconds'] == 20,
            'Fixed request limits differ')
    require(protocol['disk_budget']['minimum_free_reserve_bytes'] == 2*2**30
            and protocol['disk_budget']['allow_imagery_acquisition'] is False,
            'Metadata scope or disk reserve differs')
    return roster, raw_items, load('requirements'), receipts


def check_sources(roster, raw_items, requirements, network, provider):
    results = []
    for index, selected in enumerate(roster, 1):
        feature = raw_items[selected['item_id']]
        day = selected['acquisition_date']
        row = {
            'scene_id': selected['scene_id'], 'acquisition_date': day,
            'interval_days': (POST_DATE-date.fromisoformat(day)).days,
            'item_id': selected['item_id'], 'raw_item_sha256': selected['raw_item_sha256'],
            'catalogue_asset_href_conflicts': {
                name: asset['href'] for name, asset in feature['assets'].items()
                if 'sentinel-s2-l2a' in asset.get('href', '')
            },
            'status': BLOCKED, 'issues': [], 'assets': [], 'required_assets': 58,
            'measured_payload_bytes': None, 'local_qa_ready': False,
            'image_content_downloaded': False, 'source_approved': False,
        }
        prefix = f'source-{index:02d}/'
        try:
            listing = network.call('GET', listing_url(selected, provider), prefix+'listing.xml')
            listed = parse_listing(listing, selected['tile_prefix'], provider['listing_max_keys'])
            row['listed_objects'] = len(listed)
            product_pin = network.call('HEAD', BASE+selected['product_key'], None)
            tile_key = selected['tile_prefix']+'metadata.xml'
            require(tile_key in listed, 'Tile XML missing from provider listing')
            product = network.call('GET', BASE+selected['product_key'], prefix+'product.xml', product_pin)
            tile = network.call('GET', BASE+tile_key, prefix+'tile.xml', listed[tile_key])
            metadata = parse_metadata(product, tile, selected, feature, requirements)
            assets, issues = asset_manifest(selected, metadata, listed, product_pin)
            row.update(metadata=metadata, assets=assets, issues=metadata['issues']+issues)
            if len(assets) == row['required_assets']:
                row['measured_payload_bytes'] = sum(asset['size'] for asset in assets)
                if not row['issues']:
                    row['status'] = READY
        except (OSError, HTTPException, ValueError, TypeError, KeyError, ET.ParseError) as error:
            row['issues'].append(f'{type(error).__name__}: {error}')
        results.append(row)
    return results


def worker(protocol_path, output):
    protocol = read_json(protocol_path.read_bytes())
    roster, raw_items, requirements, receipts = inputs(protocol)
    disk_path = ROOT/'Datasets/Sentinel-2'
    free_before = shutil.disk_usage(disk_path).free
    provider = protocol['provider']
    network = MetadataNetwork(provider, roster, output)
    results = check_sources(roster, raw_items, requirements, network, provider)
    write(output/'requests.json', network.requests)
    artifact_bytes = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    budget = disk_budget(results, shutil.disk_usage(disk_path).free, artifact_bytes)
    budget['measured_filesystem_path'] = str(disk_path.relative_to(ROOT))
    all_passed = all(row['status'] == READY for row in results)
    status = ('metadata_preflight_passed_no_acquisition'
              if all_passed and budget['sufficient_for_conservative_future_acquisition']
              else 'metadata_preflight_blocked_no_acquisition')
    result = {
        'schema_version': 16, 'protocol_sha256': digest(protocol_path), 'status': status,
        'group_id': requirements['group_id'], 'results': results, 'disk_budget': budget,
        'disk_free_before_bytes': free_before, 'metadata_network_bytes': network.bytes,
        'metadata_request_count': len(network.requests), 'image_asset_downloads': 0,
        'gml_content_downloads': 0, 'satellite_pixel_reads': 0, 'qa_or_model_executions': 0,
        'labels_approved': 0, 'imagery_acquisition_roster_selected': False,
    }
    write(output/'preflight.json', result)
    report = ['# Remaining July source metadata', '', status, '']
    for row in results:
        report.append(f"- {row['acquisition_date']} | {row['status']} | {len(row['assets'])}/58 pinned assets")
        report.extend(f'  - {issue}' for issue in row['issues'])
    report.extend(['', 'Product and tile XML were checked against provider pins.',
                   'JP2 and GML availability is based on provider listings. Pixel quality remains untested.'])
    (output/'report.md').write_text('\n'.join(report)+'\n')
    require(not {'torch', 'rasterio', 's2cloudless'} & sys.modules.keys(), 'Unexpected pixel/model import')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'execution_status': 'completed', 'mode': 'remaining_source_metadata_preflight',
        'protocol_sha256': digest(protocol_path), 'input_receipts': receipts,
        'artifact_sha256': {str(path.relative_to(output)): digest(path)
                            for path in sorted(output.rglob('*')) if path.is_file()},
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2),
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args.protocol, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='remaining-source-preflight',
                    gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
