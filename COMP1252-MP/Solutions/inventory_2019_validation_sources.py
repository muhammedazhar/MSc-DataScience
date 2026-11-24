#!/usr/bin/env python3
"""Archive and replay bounded Sentinel-2 metadata for two fixed 2019 sites."""

import argparse
from datetime import date, timedelta
import os
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory

from inventory_earlier_validation_sources import (
    ENDPOINT, PRODUCT, RESOURCES, collect, digest, fetch, initial_request,
    normalize, read_json, require, rooted, sha, write,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'Docs/Reviews/Validation2019SourcesV25/protocol.json'
IMPLEMENTATION = 'Solutions/inventory_2019_validation_sources.py'
MODE = 'validation_2019_sentinel_inventory'
QUERY_RULES = {
    'url': ENDPOINT,
    'collection': 'sentinel-2-l1c',
    'page_limit': 100,
    'maximum_pages': 5,
    'maximum_response_bytes': 2*2**20,
    'timeout_seconds': 20,
    'target_tile': '36UYA',
    'relative_orbit': 'R064',
    'nominal_followup_baseline': 'N0207',
    'minimum_interval_days': 5,
    'maximum_interval_days': 30,
    'source_crs': 'EPSG:32636',
}
FIXED_SITES = {
    'march': {
        'pair_id': 'pair-f113871fa642d1296fbc',
        'event_id': 'event-09d8404da019897e1147',
        'group_id': 'group-a77532f92e6bf57d35aa',
        'annotation_date': '2019-03-28',
        'pre_date': '2019-03-18',
        'post_date': '2019-04-02',
        'post_scene_id': 'S2B_MSIL1C_20190402T083559_N0207_R064_T36UYA_20190402T111852',
        'prior_rejected_pre_dates': ['2019-03-13', '2019-03-18'],
    },
    'may': {
        'pair_id': 'pair-73bad06c9e0d3fe6b7fc',
        'event_id': 'event-011bc19e1e0f43eaa1ea',
        'group_id': 'group-01d258e68b5c4e02152b',
        'annotation_date': '2019-05-27',
        'pre_date': '2019-05-17',
        'post_date': '2019-06-06',
        'post_scene_id': 'S2A_MSIL1C_20190606T083601_N0207_R064_T36UYA_20190606T104250',
        'prior_rejected_pre_dates': ['2019-05-17'],
    },
}
SOURCE_ROLES = {'v23_plan', 'v24_screen', 'pairs', 'original_scene_inventory', 'catalogue', 'v22_review'}
REJECTED_QA = 'excluded_by_automatic_quality_screen'


def load(path):
    return read_json(path.read_bytes())


def artifact_hashes(directory):
    return {path.relative_to(directory).as_posix(): digest(path)
            for path in sorted(directory.rglob('*'))
            if path.is_file() and path != directory/'metadata.json'}


def indexed_jsonl(path, key, identifiers):
    records = {}
    with path.open('rb') as handle:
        for line in handle:
            record = read_json(line)
            identity = record[key]
            if identity in identifiers:
                require(identity not in records, 'Duplicate frozen identity: '+identity)
                records[identity] = record
    require(records.keys() == identifiers, 'Missing frozen source identity: '+key)
    return records


def requirements(site):
    return {'crs': site['grid']['crs'], 'core_union_bounds': site['grid']['bounds'],
            'qa_context_union_bounds': site['qa_context_bounds']}


def check_site(site, plan):
    site_id = site['site_id']
    require(all(site.get(key) == value for key, value in FIXED_SITES[site_id].items()),
            'Fixed site identity or prior dispositions differ: '+site_id)
    start = (date.fromisoformat(site['post_date'])-timedelta(days=30)).isoformat()
    last = (date.fromisoformat(site['pre_date'])-timedelta(days=1)).isoformat()
    expected_query = {
        **QUERY_RULES,
        'start_inclusive': start+'T00:00:00Z',
        'end_exclusive': site['pre_date']+'T00:00:00Z',
        'stac_datetime': start+'T00:00:00Z/'+last+'T23:59:59.999999Z',
        'post_date': site['post_date'],
        'known_cloudy_date': site['pre_date'],
    }
    query = site['query']
    require(all(query.get(key) == value for key, value in expected_query.items())
            and query.get('catalogue_cloud_percentage_used_for_selection', False) is False
            and query.get('select_roster', False) is False,
            'Fixed catalogue request differs: '+site_id)
    limits = requirements(site)
    require(all(key not in query or query[key] == limits[key]
                for key in ('core_union_bounds', 'qa_context_union_bounds')),
            'Catalogue geometry conflicts with fixed site bounds: '+site_id)
    grid = site['grid']
    x0, y0, x1, y1 = grid['bounds']
    require(grid['crs'] == query['source_crs'] and grid['width'] == grid['height'] == 64
            and x1-x0 == y1-y0 == 640
            and grid['transform'] == [10.0, 0.0, x0, 0.0, -10.0, y1]
            and site['qa_context_bounds'] == [x0-1200, y0-1200, x1+1200, y1+1200],
            'Frozen grid or its 1200 m QA context differs: '+site_id)
    rows = [row for row in plan['clear_post_alternatives'][site['group_id']]
            if all(row[key] == site[key]
                   for key in ('event_id', 'group_id', 'annotation_date', 'post_date', 'grid'))]
    require(sorted(row['pre_date'] for row in rows) == site['prior_rejected_pre_dates']
            and any(row['pair_id'] == site['pair_id'] and row['pre_date'] == site['pre_date']
                    and row['anchor_interior_pixels'] == site['eroded_support_pixels']
                    and row['later_usable_scene_ids'] == site['later_usable_scene_ids']
                    for row in rows)
            and all(row['split'] == 'validation' and row['footprint_isolated'] is True
                    and row['qa_status'] == REJECTED_QA
                    and row['endpoint_crop_unusable_pixels'][1] == 0
                    and row['endpoint_target_unusable_pixels'][1] == 0
                    and row['endpoint_crop_unusable_pixels'][0] > 0
                    and row['endpoint_target_unusable_pixels'][0] > 0 for row in rows),
            'Frozen earlier failures or clear post evidence differ: '+site_id)
    return sorted(rows, key=lambda row: row['pre_date'])


def read_inputs(protocol_path):
    raw_protocol = protocol_path.read_bytes()
    protocol = read_json(raw_protocol)
    require(protocol['schema_version'] == 25 and protocol['resources'] == RESOURCES
            and protocol['metadata_only'] is True and protocol['select_roster'] is False
            and protocol['labels_approved'] == 0,
            'V25 catalogue scope or resource budget differs')
    receipts = protocol['source_files']
    require(SOURCE_ROLES <= receipts.keys(), 'Missing pinned V25 source inputs')
    files = {role: rooted(receipt['path']) for role, receipt in receipts.items()}
    for role, path in files.items():
        require(digest(path) == receipts[role]['sha256'], 'Frozen input differs: '+role)
    frozen = protocol['frozen_implementation_sha256']
    implementation = protocol['implementation_sha256']
    require({'Solutions/inventory_earlier_validation_sources.py', 'Solutions/validate_sample.py'}
            <= frozen.keys() and IMPLEMENTATION in implementation,
            'Pin catalogue implementation and guard before operational requests')
    for name, checksum in (*frozen.items(), *implementation.items()):
        require(digest(rooted(name)) == checksum, 'Pinned implementation differs: '+name)

    plan = load(files['v23_plan'])
    screen = load(files['v24_screen'])
    area_b = load(files['v22_review'])
    manually_exposed = plan['excluded_previous_review_groups']
    screened_v24 = screen['groups_observed_by_automatic_qa']
    require(plan['schema_version'] == 23 and screen['schema_version'] == 24
            and area_b['schema_version'] == 22
            and len(set(manually_exposed)) == len(manually_exposed) == 7
            and manually_exposed == area_b['manual_exposed_group_ids'] == screen['prior_manual_exposed_group_ids']
            and manually_exposed == protocol['prior_manual_exposed_group_ids']
            and area_b['group_id'] in manually_exposed
            and len(set(screened_v24)) == 2 and screen['review_candidate_pair_ids'] == [],
            'Prior visual or automatic QA exposure evidence differs')
    sites = protocol['sites']
    require([site['site_id'] for site in sites] == list(FIXED_SITES),
            'V25 catalogue requests must retain two fixed sites in order')
    require(not {site['group_id'] for site in sites}.intersection(manually_exposed+screened_v24),
            'A prior review group was reused')
    prior_qa = {site['site_id']: check_site(site, plan) for site in sites}
    pair_ids = {row['pair_id'] for rows in prior_qa.values() for row in rows}
    pairs = indexed_jsonl(files['pairs'], 'pair_id', pair_ids)
    post_ids = {site['post_scene_id'] for site in sites}
    post_scenes = indexed_jsonl(files['original_scene_inventory'], 'scene_id', post_ids)
    for site in sites:
        post = post_scenes[site['post_scene_id']]
        tile_bounds = post['grid_10m']['bounds']
        x0, y0, x1, y1 = site['qa_context_bounds']
        require(post['metadata_usable'] is True and post['acquisition_date'] == site['post_date']
                and post['processing_level'] == 'L1C' and post['baseline_code'] == 'N0207'
                and post['tile'] == '36UYA' and post['crs'] == QUERY_RULES['source_crs']
                and tile_bounds[0] <= x0 <= x1 <= tile_bounds[2]
                and tile_bounds[1] <= y0 <= y1 <= tile_bounds[3],
                'Frozen local post metadata differs: '+site['site_id'])
        for row in prior_qa[site['site_id']]:
            pair = pairs[row['pair_id']]
            prior_product = PRODUCT.fullmatch(pair['pre_scene_id'])
            require(all(pair[key] == site[key] for key in ('group_id', 'grid', 'post_date', 'post_scene_id'))
                    and pair['anchor_event_ids'] == [site['event_id']]
                    and pair['pre_date'] == row['pre_date'] and pair['split'] == 'validation'
                    and prior_product is not None and prior_product['tile'] == '36UYA'
                    and prior_product['orbit'] == 'R064' and prior_product['baseline'] == 'N0207'
                    and prior_product['acquisition'][:8] == row['pre_date'].replace('-', '')
                    and (row['pair_id'] != site['pair_id'] or pair['pre_scene_id'] == site['pre_scene_id']),
                    'Frozen rejected local pair binding differs: '+row['pair_id'])
            row['pre_scene_id'] = pair['pre_scene_id']
    evidence = {'prior_manual_exposed_group_ids': manually_exposed,
                'prior_v24_automatically_screened_group_ids': screened_v24,
                'prior_local_failures_by_site': prior_qa}
    return sites, evidence, sha(raw_protocol)


def normalize_site_item(item, site, prior):
    row = normalize(item, site['query'], requirements(site))
    rejected_on_date = [case for case in prior if case['pre_date'] == row['acquisition_date']]
    uri = row['product_uri']
    scene_id = uri.removesuffix('.SAFE') if isinstance(uri, str) else None
    exact_failures = [case['pair_id'] for case in rejected_on_date if case['pre_scene_id'] == scene_id]
    row['prior_rejected_acquisition_date'] = bool(rejected_on_date)
    row['prior_rejected_pair_ids_for_date'] = [case['pair_id'] for case in rejected_on_date]
    row['exact_product_prior_rejected_pair_ids'] = exact_failures
    row['own_qa_status'] = 'excluded_in_prior_local_pair' if exact_failures else 'unexamined'
    if rejected_on_date:
        row['followup_blockers'] = sorted({*row['followup_blockers'], 'prior_rejected_pre_date_for_site'})
        row['nominal_metadata_followup_eligible'] = False
    return row


def build_inventory(output, sites, evidence, protocol_hash, site_fetchers=None):
    site_results = []
    for site in sites:
        site_id = site['site_id']
        directory = output/site_id
        directory.mkdir()
        query = site['query']
        limits = requirements(site)
        fetcher = fetch if site_fetchers is None else site_fetchers[site_id]
        items, pages, matched = collect(initial_request(limits, query), query, directory, fetcher=fetcher)
        prior = evidence['prior_local_failures_by_site'][site_id]
        rows = [normalize_site_item(item, site, prior) for item in items]
        compatible = [row for row in rows if row['nominal_metadata_followup_eligible']]
        site_results.append({
            **site,
            'catalogue_requirements': limits,
            'prior_local_failures': prior,
            'pagination_complete': True,
            'provider_number_matched': matched,
            'pages': [{**page, 'response': f'{site_id}/{page["response"]}'} for page in pages],
            'distinct_items': len(rows),
            'nominal_metadata_followup_items': len(compatible),
            'nominal_metadata_followup_dates': sorted({row['acquisition_date'] for row in compatible}),
            'prior_rejected_date_items_retained': sum(row['prior_rejected_acquisition_date'] for row in rows),
            'items': rows,
        })
    page_count = sum(len(site['pages']) for site in site_results)
    require(page_count <= 10, 'Combined page budget exceeded')
    result = {
        'schema_version': 25,
        'status': 'complete_2019_sentinel_catalogue_inventory',
        'protocol_sha256': protocol_hash,
        'prior_manual_exposed_group_ids': evidence['prior_manual_exposed_group_ids'],
        'prior_v24_automatically_screened_group_ids': evidence['prior_v24_automatically_screened_group_ids'],
        'sites': site_results,
        'catalogue_pages': page_count,
        'source_approvals': 0,
        'selected_roster_items': 0,
        'labels_approved': 0,
        'validation_eligible': False,
    }
    write(output/'inventory.json', result)
    report = ['# V25 Sentinel-2 catalogue inventory', '',
              '| Site | Query dates inclusive | Fixed post | Retained catalogue items | Metadata matches |',
              '| --- | --- | --- | ---: | --- |']
    for site in site_results:
        q = site['query']
        start, end = (value[:10] for value in q['stac_datetime'].split('/'))
        matches = ', '.join(site['nominal_metadata_followup_dates']) or 'None'
        report.append(f'| {site["site_id"].title()} | {start} to {end} | {site["post_date"]} | '
                      f'{site["distinct_items"]} | {matches} |')
    report += ['', 'The inventory retains all returned variants and the prior local QA exclusions. '
               'Metadata matches require native product verification and local QA before use. '
               'Crop clarity, class evidence and a first-visible interval remain unresolved.']
    (output/'report.md').write_text('\n'.join(report)+'\n')
    return result


def archived_fetcher(directory, site):
    receipts = iter(enumerate(site['pages'], start=1))

    def replay_fetch(request, query):
        entry = next(receipts, None)
        require(entry is not None, 'Missing archived response: '+site['site_id'])
        index, receipt = entry
        expected_response = f'{site["site_id"]}/raw/page-{index:02d}.json'
        require(receipt['response'] == expected_response and receipt['request'] == request,
                'Archived query sequence differs: '+site['site_id'])
        path = directory/expected_response
        require(path.is_file() and path.resolve().is_relative_to(directory.resolve())
                and path.stat().st_size == receipt['bytes'] <= query['maximum_response_bytes'],
                'Missing bounded archived response: '+expected_response)
        raw = path.read_bytes()
        require(sha(raw) == receipt['sha256'], 'Archived response changed: '+expected_response)
        return raw

    return replay_fetch


def replay_inventory(directory, output, sites, evidence, protocol_hash):
    metadata = load(directory/'metadata.json')
    previous = load(directory/'inventory.json')
    require(metadata['execution_status'] == 'completed' and metadata['mode'] == MODE
            and metadata['protocol_sha256'] == protocol_hash
            and metadata['artifact_sha256'] == artifact_hashes(directory)
            and [site['site_id'] for site in previous['sites']] == list(FIXED_SITES),
            'Replay requires complete unchanged V25 catalogue evidence')
    fetchers = {site['site_id']: archived_fetcher(directory, site) for site in previous['sites']}
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        result = build_inventory(stage, sites, evidence, protocol_hash, site_fetchers=fetchers)
        require(result == previous and artifact_hashes(stage) == metadata['artifact_sha256'],
                'Catalogue replay differs')
    write(output/'verification.json', {
        'status': 'passed',
        'protocol_sha256': protocol_hash,
        'inventory_metadata_sha256': digest(directory/'metadata.json'),
        'inventory_artifact_sha256': metadata['artifact_sha256'],
        'raw_pages_replayed': result['catalogue_pages'],
        'items_replayed_by_site': {site['site_id']: site['distinct_items'] for site in result['sites']},
        'network_requests': 0,
    })
    return result


def worker(args, output):
    sites, evidence, checksum = read_inputs(args.protocol)
    if args.mode == 'inventory':
        result = build_inventory(output, sites, evidence, checksum)
    else:
        result = replay_inventory(args.inventory, output, sites, evidence, checksum)
    require(not {'torch', 'rasterio', 's2cloudless'} & sys.modules.keys(), 'Unexpected pixel/model import')
    require(digest(args.protocol) == checksum, 'Protocol changed during catalogue work')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'execution_status': 'completed',
        'mode': MODE if args.mode == 'inventory' else MODE+'_replay',
        'protocol_sha256': checksum,
        'artifact_sha256': artifact_hashes(output),
        'catalogue_requests': result['catalogue_pages'] if args.mode == 'inventory' else 0,
        'satellite_pixel_reads': 0,
        'image_asset_downloads': 0,
        'model_executions': 0,
        'source_approvals': 0,
        'selected_roster_items': 0,
        'labels_approved': 0,
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2),
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
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='validation-2019-sentinel-metadata',
                    gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
