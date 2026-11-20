#!/usr/bin/env python3
"""Preflight the two fixed August sources using only catalogue records and XML."""

import argparse
from datetime import date, timedelta
import os
from pathlib import Path
import re
import resource
import shutil
import sys
from urllib.request import Request, build_opener

from inventory_earlier_validation_sources import (
    NoRedirect, RESOURCES, ROOT, canonical, digest, normalize, read_json, require, rooted, sha, write,
)
from preflight_earlier_validation_metadata import BASE, MetadataNetwork, disk_budget
from preflight_remaining_validation_metadata import BLOCKED, READY, check_sources

PROTOCOL = ROOT/'Docs/Reviews/ValidationPersistenceV20/protocol.json'
DATES = ('2018-08-05', '2018-08-10')
ITEMS = 'https://earth-search.aws.element84.com/v1/collections/sentinel-2-l1c/items/'


def checked_protocol():
    protocol = read_json(PROTOCOL.read_bytes())
    require(protocol['schema_version'] == 20 and protocol['resources'] == RESOURCES
            and tuple(protocol['candidate_dates']) == DATES and protocol['area'] == 'B'
            and protocol['catalogue_item_root'] == ITEMS
            and protocol['catalogue_response_bytes'] == 2**20
            and protocol['provider']['base_url'] == BASE
            and protocol['provider']['maximum_requests'] == 8, 'V20 scope or limits differ')
    hashes = {row['path']: row['sha256'] for row in protocol['source_files'].values()}
    hashes.update(protocol['frozen_implementation_sha256'])
    hashes.update(protocol['implementation_sha256'])
    require(str(Path(__file__).relative_to(ROOT)) in hashes, 'Pin the preflight runner first')
    for name, checksum in hashes.items():
        require(digest(rooted(name)) == checksum, 'Pinned persistence input changed: '+name)
    return protocol


def fixed_roster(protocol):
    prior = read_json(rooted(protocol['source_files']['v7_catalogue']['path']).read_bytes())
    roster = []
    for day in DATES:
        entries = [row for row in prior['all_earthsearch_timeline_products']
                   if row['acquisition_date'] == day and row['baseline_code'] == 'N0206'
                   and '_R064_T36UXB_' in row['product_uri'] and row['covers_core_union']
                   and row['covers_qa_context_union']]
        require(len(entries) == 1, f'V7 does not identify one nominal product on {day}')
        row = entries[0]
        roster.append({'acquisition_date': day, 'item_id': row['item_id'],
                       'scene_id': row['product_uri'].removesuffix('.SAFE')})
    require(roster == protocol['catalogue_roster'], 'Fixed catalogue identities differ')
    return roster


def catalogue_row(raw, expected, requirements):
    feature = read_json(raw)
    require(feature.get('type') == 'Feature' and feature.get('id') == expected['item_id'],
            'Catalogue response is not the expected item')
    day = date.fromisoformat(expected['acquisition_date'])
    query = {'start_inclusive': day.isoformat()+'T00:00:00Z',
             'end_exclusive': (day+timedelta(days=1)).isoformat()+'T00:00:00Z',
             'collection': 'sentinel-2-l1c', 'post_date': '2018-07-31', 'known_cloudy_date': '2018-07-26',
             'minimum_interval_days': 5, 'maximum_interval_days': 30,
             'target_tile': '36UXB', 'relative_orbit': 'R064', 'nominal_followup_baseline': 'N0206'}
    checksum = sha(canonical(feature))
    row = normalize({'feature': feature, 'sha256': checksum, 'pages': []}, query, requirements)
    require(not row['issues'] and row['acquisition_date'] == expected['acquisition_date']
            and row['product_uri'] == expected['scene_id']+'.SAFE'
            and all(value is True for value in row['catalogue_footprint_covers'].values()),
            f'Catalogue identity, footprint or issues differ: {row["issues"]}')
    sequence = feature['properties']['s2:sequence']
    require(isinstance(sequence, str) and re.fullmatch(r'[0-9]{1,3}', sequence)
            and expected['item_id'].endswith(f'_{sequence}_L1C'), 'Catalogue tile sequence differs')
    year, month, day = map(int, expected['acquisition_date'].split('-'))
    scene = expected['scene_id']
    selection = {**expected, 'sequence': sequence, 'raw_item_sha256': checksum,
                 'tile_prefix': f'tiles/36/U/XB/{year}/{month}/{day}/{sequence}/',
                 'product_key': f'products/{year}/{month}/{day}/{scene}/metadata.xml'}
    return feature, selection, row['catalogue_footprint_covers']


def fetch_item(selected, protocol, output, opener):
    item = selected['item_id']
    require(re.fullmatch(r'S2[AB]_36UXB_201808(?:05|10)_[0-9]{1,3}_L1C', item), 'Untrusted item ID')
    url = ITEMS+item
    cap = protocol['catalogue_response_bytes']
    request = Request(url, headers={'Accept': 'application/geo+json, application/json',
                                  'User-Agent': 'COMP1252-persistence-metadata/20'})
    with opener(request, timeout=protocol['provider']['timeout_seconds']) as response:
        require(response.status == 200 and response.geturl() == url, 'Non-200 catalogue response or redirect')
        raw = response.read(cap+1)
        require(len(raw) <= cap, 'Catalogue response exceeds byte cap')
    name = f'catalogue-{selected["acquisition_date"]}.json'
    (output/name).write_bytes(raw)
    return raw, {'url': url, 'status': 200, 'artifact': name, 'sha256': sha(raw), 'bytes': len(raw)}


def worker(output):
    protocol = checked_protocol()
    protocol_hash = digest(PROTOCOL)
    requirements = read_json(rooted(protocol['source_files']['requirements']['path']).read_bytes())
    raw_items, selections, coverage, catalogue_requests, failed = {}, [], {}, [], {}
    opener = build_opener(NoRedirect()).open
    for expected in fixed_roster(protocol):
        day = expected['acquisition_date']
        try:
            raw, receipt = fetch_item(expected, protocol, output, opener)
            catalogue_requests.append(receipt)
            feature, selection, covered = catalogue_row(raw, expected, requirements)
            selections.append(selection)
            coverage[day] = covered
            raw_items[expected['item_id']] = feature
        except (OSError, ValueError, TypeError, KeyError) as error:
            failed[day] = {**expected, 'status': BLOCKED, 'assets': [], 'required_assets': 58,
                           'issues': [f'{type(error).__name__}: {error}']}
    provider = protocol['provider']
    network = MetadataNetwork(provider, selections, output)
    results = dict(failed)
    for selection, result in zip(selections, check_sources(selections, raw_items, requirements, network, provider)):
        day = result['acquisition_date']
        result['days_after_july_31'] = -result.pop('interval_days')
        result['catalogue_footprint_covers'] = coverage[day]
        result.update({key: selection[key] for key in ('sequence', 'product_key', 'tile_prefix')})
        results[day] = result
    results = [results[day] for day in DATES]
    write(output/'requests.json', {'catalogue': catalogue_requests, 'provider': network.requests})
    ready = [row for row in results if row['status'] == READY]
    budget = disk_budget(ready, shutil.disk_usage(ROOT/'Datasets/Sentinel-2').free,
                         sum(path.stat().st_size for path in output.rglob('*') if path.is_file()))
    if not ready:
        budget.update(complete_payload_bytes=None, future_two_copy_staging_plus_temporary_bytes=None,
                      sufficient_for_conservative_future_acquisition=False)
    status = ('metadata_preflight_passed_no_acquisition'
              if ready and budget['sufficient_for_conservative_future_acquisition']
              else 'metadata_preflight_blocked_no_acquisition')
    write(output/'preflight.json', {'schema_version': 20, 'status': status, 'protocol_sha256': protocol_hash,
                                  'group_id': requirements['group_id'], 'results': results, 'disk_budget': budget})
    require(digest(PROTOCOL) == protocol_hash, 'Protocol changed during preflight')
    require(not {'torch', 'rasterio', 's2cloudless'} & sys.modules.keys(), 'Unexpected pixel or model import')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'execution_status': 'completed', 'mode': 'persistence_source_preflight', 'protocol_sha256': protocol_hash,
        'artifact_sha256': {str(path.relative_to(output)): digest(path)
                            for path in sorted(output.rglob('*')) if path.is_file()},
        'image_asset_downloads': 0, 'satellite_pixel_reads': 0, 'model_executions': 0,
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2),
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    for variable in THREAD_VARIABLES:
        os.environ[variable] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='persistence-source-preflight',
                    gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
