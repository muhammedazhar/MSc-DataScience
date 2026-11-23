#!/usr/bin/env python3
"""Archive the bounded V23 catalogue query and nominate earlier source metadata."""

import argparse
from collections import defaultdict
import os
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory

from inventory_earlier_validation_sources import (
    ENDPOINT, RESOURCES, collect, digest, initial_request, normalize, read_json,
    require, rooted, write,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'Docs/Reviews/ValidationFreshSourcesV23/protocol.json'
QUERY_RULES = {
    'url': ENDPOINT,
    'collection': 'sentinel-2-l1c',
    'start_inclusive': '2018-07-11T00:00:00Z',
    'end_exclusive': '2018-07-26T00:00:00Z',
    'stac_datetime': '2018-07-11T00:00:00Z/2018-07-25T23:59:59.999999Z',
    'page_limit': 100,
    'maximum_pages': 5,
    'maximum_response_bytes': 2*2**20,
    'timeout_seconds': 20,
    'target_tile': '36UXA',
    'relative_orbit': 'R064',
    'known_cloudy_date': '2018-07-26',
    'minimum_interval_days': 5,
    'maximum_interval_days': 30,
    'nominal_followup_baseline': 'N0206',
    'source_crs': 'EPSG:32636',
}


def load(path):
    return read_json(path.read_bytes())


def artifact_hashes(directory):
    return {str(path.relative_to(directory)): digest(path)
            for path in sorted(directory.rglob('*')) if path.is_file()
            and path != directory/'metadata.json'}


def read_inputs(protocol_path, plan_directory):
    protocol = load(protocol_path)
    checksum = digest(protocol_path)
    query = protocol['query']
    require(protocol['schema_version'] == 23 and protocol['resources'] == RESOURCES
            and all(query.get(key) == expected for key, expected in QUERY_RULES.items()),
            'Frozen V23 query or resource declaration differs')
    implementations = protocol['implementation_sha256']
    require('Solutions/inventory_fresh_validation_sources.py' in implementations,
            'Pin the catalogue implementation before querying')
    for receipt in protocol['source_files'].values():
        require(digest(rooted(receipt['path'])) == receipt['sha256'],
                'Pinned planning input differs: '+receipt['path'])
    for name, expected in {**protocol.get('frozen_implementation_sha256', {}), **implementations}.items():
        require(digest(rooted(name)) == expected, 'Pinned V23 implementation differs: '+name)

    metadata = load(plan_directory/'metadata.json')
    plan_path = plan_directory/'plan.json'
    plan_hash = digest(plan_path)
    require(metadata['execution_status'] == 'completed' and metadata['mode'] == 'fresh_source_planning'
            and metadata['protocol_sha256'] == checksum
            and metadata['artifact_sha256']['plan.json'] == plan_hash,
            'Plan does not bind the completed guarded V23 planning artifact')
    plan = load(plan_path)
    requirements = plan['catalogue_requirements']
    roster = plan['roster']
    require(plan['schema_version'] == 23 and plan['status'] == 'fresh_catalogue_plan_ready'
            and plan['protocol_sha256'] == checksum and len(roster) == 2
            and len({case['group_id'] for case in roster}) == 2
            and len({case['event_id'] for case in roster}) == 2
            and [{key: case[key] for key in ('pair_id', 'event_id', 'group_id', 'post_date')}
                 for case in roster] == protocol['fixed_parents']
            and sorted(case['pre_date'] for case in roster) == ['2018-07-26', '2018-07-31']
            and sorted(case['post_date'] for case in roster) == ['2018-07-31', '2018-08-10'],
            'V23 must retain the two planned independent groups and post-dates')
    require(requirements['crs'] == query['source_crs']
            and all(requirements[key] == query[key] for key in ('start_inclusive', 'end_exclusive',
                    'target_tile', 'relative_orbit', 'nominal_followup_baseline')),
            'Planning requirements differ from the predeclared query')
    require(all(case['grid']['crs'] == requirements['crs']
                and case['grid']['width'] == case['grid']['height'] == 64
                and case['qa_context_bounds'] == [coord+delta for coord, delta in
                    zip(case['grid']['bounds'], (-1200, -1200, 1200, 1200))]
                and case['pre_search_end_exclusive'][:10] == case['pre_date']
                and case['annotation_date'] <= case['post_date'] and case['post_metadata_usable'] is True
                for case in roster), 'Fixed grid or local post-source eligibility differs')

    for key, bounds in (('core_union_bounds', [case['grid']['bounds'] for case in roster]),
                        ('qa_context_union_bounds', [case['qa_context_bounds'] for case in roster])):
        expected = [min(box[0] for box in bounds), min(box[1] for box in bounds),
                    max(box[2] for box in bounds), max(box[3] for box in bounds)]
        require(requirements[key] == expected, 'Union bounds differ from the fixed crop roster')
    return query, plan, plan_hash, checksum


def normalize_cases(item, query, roster):
    cases = []
    for case in roster:
        requirements = {'crs': case['grid']['crs'], 'core_union_bounds': case['grid']['bounds'],
                        'qa_context_union_bounds': case['qa_context_bounds']}
        row = normalize(item, {**query, 'post_date': case['post_date']}, requirements)
        day = row['acquisition_date']
        blockers = row['followup_blockers']
        if day is not None and not (case['pre_search_start_inclusive'][:10] <= day
                < min(case['pre_search_end_exclusive'][:10], case['pre_date'], case['annotation_date'])):
            blockers = sorted(set(blockers+['outside_case_pre_window']))
        cases.append({
            'event_id': case['event_id'], 'group_id': case['group_id'], 'pair_id': case['pair_id'],
            'post_date': case['post_date'], 'interval_to_post_days': row['interval_to_post_days'],
            'catalogue_footprint_covers': row['catalogue_footprint_covers'],
            'nominal_metadata_followup_eligible': not blockers, 'followup_blockers': blockers,
        })
    identity = {key: value for key, value in row.items() if key not in (
        'interval_to_post_days', 'catalogue_footprint_covers', 'followup_blockers',
        'nominal_metadata_followup_eligible')}
    return {**identity, 'case_eligibility': cases,
            'nominal_metadata_followup_eligible': all(case['nominal_metadata_followup_eligible'] for case in cases)}


def nominate(items):
    by_day = defaultdict(list)
    for row in items:
        if row['nominal_metadata_followup_eligible']:
            by_day[row['acquisition_date']].append(row)
    ambiguous = []
    roster = []
    for day, rows in sorted(by_day.items(), reverse=True):
        if len({row['product_uri'] for row in rows}) > 1:
            ambiguous.append(day)
            continue
        if len(roster) < 2:
            row = min(rows, key=lambda row: row['item_id'])
            row['roster_selected'] = True
            roster.append({key: row[key] for key in ('item_id', 'product_uri', 'acquisition_date',
                           'raw_item_sha256', 'case_eligibility')})
    return roster, ambiguous


def build_inventory(output, plan, plan_hash, checksum, query, **options):
    request = initial_request(plan['catalogue_requirements'], query)
    items, pages, total = collect(request, query, output, **options)
    rows = [normalize_cases(item, query, plan['roster']) for item in items]
    roster, ambiguous = nominate(rows)
    result = {
        'schema_version': 23, 'status': 'complete_nominal_catalogue_inventory',
        'protocol_sha256': checksum, 'plan_sha256': plan_hash,
        'group_ids': [case['group_id'] for case in plan['roster']],
        'query': query, 'catalogue_requirements': plan['catalogue_requirements'],
        'pagination_complete': True, 'provider_number_matched': total, 'pages': pages,
        'distinct_items': len(rows), 'nominal_eligible_items': sum(row['nominal_metadata_followup_eligible'] for row in rows),
        'ambiguous_eligible_dates': ambiguous, 'nominal_roster': roster, 'items': rows,
        'roster_rule': 'Latest two unambiguous shared eligible pre-dates before 26 July UTC. '
                       'Only metadata is nominated; provider preflight is separate.',
    }
    write(output/'inventory.json', result)
    report = ['# V23 fresh-site source catalogue', '',
              f'All {len(rows)} items from {len(pages)} pages retained.', '', '## Nominal metadata roster', '']
    report += [f'- {row["acquisition_date"]} · `{row["product_uri"]}`' for row in roster]
    if not roster:
        report.append('No shared earlier date passes the metadata checks.')
    report += ['', 'Catalogue footprint coverage is a planning result. Local QA, visual class review and source compatibility remain pending.']
    (output/'report.md').write_text('\n'.join(report)+'\n')
    return result


def replay_inventory(directory, output, plan, plan_hash, checksum, query):
    metadata = load(directory/'metadata.json')
    previous = load(directory/'inventory.json')
    require(metadata['execution_status'] == 'completed' and metadata['mode'] == 'fresh_source_inventory'
            and metadata['protocol_sha256'] == checksum and metadata['plan_sha256'] == plan_hash
            and metadata['artifact_sha256'] == artifact_hashes(directory),
            'Replay requires a complete unchanged V23 catalogue result')
    receipts = iter(previous['pages'])

    def replay_fetch(request, _):
        receipt = next(receipts, None)
        require(receipt is not None and receipt['request'] == request, 'Archived query sequence differs')
        path = (directory/receipt['response']).resolve()
        require(path.is_relative_to((directory/'raw').resolve()) and path.is_file()
                and path.stat().st_size <= query['maximum_response_bytes'], 'Missing bounded raw page')
        require(path.stat().st_size == receipt['bytes'] and digest(path) == receipt['sha256'], 'Raw page changed')
        return path.read_bytes()

    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        result = build_inventory(stage, plan, plan_hash, checksum, query, fetcher=replay_fetch)
        require(next(receipts, None) is None and result == previous
                and artifact_hashes(stage) == metadata['artifact_sha256'], 'Catalogue replay differs')
    write(output/'verification.json', {'status': 'passed', 'protocol_sha256': checksum,
          'plan_sha256': plan_hash, 'inventory_metadata_sha256': digest(directory/'metadata.json'),
          'inventory_artifact_sha256': metadata['artifact_sha256'],
          'raw_pages_replayed': len(result['pages']), 'items_replayed': len(result['items']),
          'nominal_roster_items': len(result['nominal_roster']), 'network_requests': 0})
    return result


def worker(args, output):
    query, plan, plan_hash, checksum = read_inputs(args.protocol, args.plan)
    if args.mode == 'inventory':
        result = build_inventory(output, plan, plan_hash, checksum, query)
    else:
        result = replay_inventory(args.inventory, output, plan, plan_hash, checksum, query)
    require(not {'torch', 'rasterio', 's2cloudless'} & sys.modules.keys(), 'Unexpected pixel/model import')
    require(digest(args.protocol) == checksum and digest(args.plan/'plan.json') == plan_hash,
            'Protocol or input plan changed during catalogue work')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'execution_status': 'completed', 'mode': 'fresh_source_inventory' if args.mode == 'inventory' else 'fresh_source_inventory_replay',
        'protocol_sha256': checksum, 'plan_sha256': plan_hash, 'artifact_sha256': artifact_hashes(output),
        'catalogue_requests': len(result['pages']) if args.mode == 'inventory' else 0,
        'satellite_pixel_reads': 0, 'image_asset_downloads': 0, 'model_executions': 0,
        'labels_approved': 0, 'nominal_roster_items': len(result['nominal_roster']),
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2),
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--mode', choices=('inventory', 'verify'), default='inventory')
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    if (args.mode == 'verify') != (args.inventory is not None):
        parser.error('--inventory is required only for verify mode')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='fresh-validation-catalogue',
                    gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
