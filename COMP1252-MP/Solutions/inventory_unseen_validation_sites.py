#!/usr/bin/env python3
"""Plan and check a bounded search for fresh held-out validation sites."""

import argparse
from datetime import date, timedelta
from http.client import HTTPException
import os
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory

from inventory_earlier_validation_sources import (
    ENDPOINT, RESOURCES, ROOT, collect, coverage, digest, initial_request, read_json,
    require, rooted, timestamp, write,
)
from preflight_fresh_validation_sources import checked_artifacts, evidence_hashes


PROTOCOL = ROOT/'Docs/Reviews/ValidationUnseenSitesV28/protocol.json'
IMPLEMENTATION = 'Solutions/inventory_unseen_validation_sites.py'
SCENE_RULES = {'url': ENDPOINT, 'collection': 'sentinel-2-l1c', 'page_limit': 50,
               'maximum_pages': 4, 'maximum_response_bytes': 2*2**20, 'timeout_seconds': 12}


def load(path):
    return read_json(path.read_bytes())


def checked_protocol(path):
    protocol = load(path)
    require(protocol['schema_version'] == 28 and protocol['resources'] == RESOURCES
            and protocol['sentinel_rules'] == SCENE_RULES and protocol['maximum_sites'] == 3
            and protocol['imagery_and_labels_authorized'] is False, 'V28 scope or resources differ')
    files = {}
    for key, receipt in protocol['source_files'].items():
        files[key] = rooted(receipt['path'])
        require(digest(files[key]) == receipt['sha256'], 'Frozen source differs: '+key)
    pins = protocol['frozen_implementation_sha256'] | protocol['implementation_sha256']
    require(IMPLEMENTATION in protocol['implementation_sha256'], 'Pin V28 implementation before running')
    for name, checksum in pins.items():
        require(digest(rooted(name)) == checksum, 'Pinned V28 code differs: '+name)
    return protocol, files, digest(path)


def checked_phase(directory, filename, mode, protocol_sha):
    path = directory.resolve()
    receipt = load(path/'metadata.json')
    require(receipt['mode'] == mode and receipt['execution_status'] == 'completed'
            and receipt['protocol_sha256'] == protocol_sha
            and receipt['artifact_sha256'] == evidence_hashes(path)
            and filename in receipt['artifact_sha256'], 'Completed V28 phase differs: '+mode)
    checked_artifacts(path, receipt['artifact_sha256'])
    return receipt


def scene_row(item, site, limits):
    from bbox_sentinel_preflight import catalogue_source

    feature = item['feature']
    row = {'item_id': feature['id'], 'item_sha256': item['sha256'], 'occurrence_pages': item['pages'],
           'catalogue_cloud_percent': feature.get('properties', {}).get('eo:cloud_cover'), 'blockers': []}
    footprint, issue = coverage(feature.get('geometry'), limits)
    row['catalogue_footprint'] = footprint
    if not all(footprint.values()):
        row['blockers'].append(issue or 'catalogue_footprint_lacks_full_crop_or_qa_margin')
    try:
        source = catalogue_source(feature)
    except (ValueError, KeyError, TypeError) as error:
        row['blockers'].append(str(error))
    else:
        row.update(source)
        observed = timestamp(feature['properties']['datetime'])
        if not site['comparison_start_inclusive'] <= observed.date().isoformat() <= site['reference_end_inclusive']:
            row['blockers'].append('catalogue_time_outside_request')
        if source['tile'] != site['tile']:
            row['blockers'].append('different_native_tile')
    return row


def scene_compatibility(a, b):
    return all(a[key] == b[key] for key in ('tile', 'relative_orbit', 'processing_baseline'))


def cloud_score(value):
    return value if type(value) in (int, float) and 0 <= value <= 100 else 101


def options(site, rows):
    annotation = date.fromisoformat(site['annotation_date'])
    last = date.fromisoformat(site['comparison_end_inclusive'])
    eligible = [(date.fromisoformat(row['acquisition_date']), row) for row in rows if not row['blockers']]
    eligible.sort(key=lambda item: (item[0], item[1]['scene_id']))
    pairs = []
    for before_day, before in eligible:
        if before_day >= annotation:
            break
        for after_day, after in eligible:
            interval = (after_day-before_day).days
            if not (annotation <= after_day <= last and 5 <= interval <= 30 and scene_compatibility(before, after)):
                continue
            followups = [row['scene_id'] for day, row in eligible
                         if day > after_day and scene_compatibility(after, row)]
            stable = [{'date': day.isoformat(), 'scene_id': row['scene_id']} for day, row in eligible
                      if 5 <= (before_day-day).days <= 30 and scene_compatibility(before, row)]
            pairs.append({'pre_date': before_day.isoformat(), 'post_date': after_day.isoformat(),
                          'pre_scene_id': before['scene_id'], 'post_scene_id': after['scene_id'],
                          'interval_days': interval, 'relative_orbit': before['relative_orbit'],
                          'processing_baseline': before['processing_baseline'],
                          'catalogue_cloud_percent': [before['catalogue_cloud_percent'], after['catalogue_cloud_percent']],
                          'later_metadata_scene_ids': followups, 'stable_only_earlier_scene_options': stable})
    def rank(row):
        clouds = [cloud_score(value) for value in row['catalogue_cloud_percent']]
        return (not bool(row['later_metadata_scene_ids']), max(clouds), sum(clouds),
                row['post_date'], row['interval_days'], row['pre_scene_id'], row['post_scene_id'])

    return sorted(pairs, key=rank)


def build_sentinel(output, sites, protocol_sha, fetchers=None):
    archive = output/'catalogue'
    archive.mkdir()
    results = []
    for site in sites:
        site_id = site['site_id']
        directory = archive/site_id
        directory.mkdir()
        limits = {'crs': site['grid']['crs'], 'core_union_bounds': site['grid']['bounds'],
                  'qa_context_union_bounds': site['qa_context_bounds']}
        query = {**SCENE_RULES,
                 'stac_datetime': (site['comparison_start_inclusive']+'T00:00:00Z/'
                                   +site['reference_end_inclusive']+'T23:59:59.999999Z')}
        captured = {'site_id': site_id, 'query': query, 'limits': limits, 'provider_matched': None,
                    'pagination_complete': False, 'pages': [], 'items': [], 'comparison_options': []}
        try:
            fetch_args = {} if fetchers is None else {'fetcher': fetchers[site_id]}
            items, pages, matched = collect(initial_request(limits, query), query, directory, **fetch_args)
            rows = [scene_row(item, site, limits) for item in items]
            captured.update(provider_matched=matched, pagination_complete=True, pages=pages,
                            items=rows, comparison_options=options(site, rows))
        except (HTTPException, OSError, ValueError, KeyError, TypeError) as error:
            captured['error'] = f'{type(error).__name__}: {error}'
        results.append(captured)
    result = {'schema_version': 28, 'protocol_sha256': protocol_sha, 'sites': results,
              'rank_rule': ('Later compatible follow-up availability, then lowest maximum and total scene cloud '
                            'percent, earliest post, shorter interval and exact product IDs.'),
              'scope': 'Catalogue evidence only. Scene cloud and footprint do not establish crop clarity.',
              'acquisition_roster_selected': False, 'labels_approved': 0}
    write(output/'sentinel.json', result)
    return result


def verify_sentinel(directory, output, sites, protocol_sha):
    meta = checked_phase(directory, 'sentinel.json', 'sentinel', protocol_sha)
    prior = load(directory/'sentinel.json')
    require([row['site_id'] for row in prior['sites']] == [site['site_id'] for site in sites]
            and all(row['pagination_complete'] for row in prior['sites']),
            'Replay needs complete V28 Sentinel-2 pagination')

    def archived_fetch(row):
        pages = iter(row['pages'])

        def fetcher(request, query):
            receipt = next(pages, None)
            require(receipt is not None and receipt['request'] == request, 'Archived request sequence differs')
            base = (directory/'catalogue'/row['site_id']).resolve()
            path = (base/receipt['response']).resolve()
            require(path.is_relative_to(base/'raw') and path.stat().st_size == receipt['bytes']
                    <= query['maximum_response_bytes'] and digest(path) == receipt['sha256'],
                    'Archived STAC page differs')
            return path.read_bytes()

        return fetcher

    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        result = build_sentinel(stage, sites, protocol_sha,
                                fetchers={row['site_id']: archived_fetch(row) for row in prior['sites']})
        require(result == prior and evidence_hashes(stage) == meta['artifact_sha256'],
                'V28 Sentinel-2 replay differs')
    write(output/'verification.json', {'status': 'passed', 'protocol_sha256': protocol_sha,
          'replayed_archive_sha256': digest(directory/'metadata.json'), 'network_requests': 0,
          'items_replayed': sum(len(row['items']) for row in result['sites'])})


def worker(args, output):
    from datetime import datetime, timezone

    protocol, files, checksum = checked_protocol(args.protocol)
    if args.mode == 'plan':
        from plan_unseen_validation_sites import build_plan

        plan = build_plan(protocol, files)
        plan['protocol_sha256'] = checksum
        write(output/'plan.json', plan)
    else:
        checked_phase(args.plan, 'plan.json', 'plan', checksum)
        plan = load(args.plan/'plan.json')
        require(plan['protocol_sha256'] == checksum and len(plan['sites']) <= protocol['maximum_sites'],
                'V28 site plan differs from bounded inputs')
        sites = plan['sites']
        if args.mode == 'sentinel':
            build_sentinel(output, sites, checksum)
        elif args.mode == 'verify-sentinel':
            verify_sentinel(args.inventory, output, sites, checksum)
        else:
            from inventory_unseen_validation_references import build_references, verify_references

            if args.mode == 'references':
                build_references(output, sites, checksum, datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                 retry_from=args.retry_from)
            else:
                verify_references(args.inventory, output, sites, checksum)
    require(digest(args.protocol) == checksum and not {'torch', 's2cloudless'} & sys.modules.keys(),
            'Protocol changed or model library imported during metadata work')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {'schema_version': 28, 'mode': args.mode, 'execution_status': 'completed',
          'protocol_sha256': checksum, 'artifact_sha256': evidence_hashes(output),
          'satellite_pixel_reads': 0, 'image_asset_downloads': 0, 'model_executions': 0, 'labels_approved': 0,
          'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2)})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--mode', required=True,
                        choices=('plan', 'sentinel', 'references', 'verify-sentinel', 'verify-references'))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--retry-from', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=RESOURCES['worker_rss_mib'],
                        min_available_gib=RESOURCES['available_memory_reserve_gib'],
                        max_seconds=RESOURCES['max_seconds_per_worker'])
    args = parser.parse_args()
    if (args.mode != 'plan') != (args.plan is not None):
        parser.error('--plan is required for catalogue and replay modes')
    if args.mode.startswith('verify') != (args.inventory is not None):
        parser.error('--inventory is required for replay modes')
    if args.retry_from is not None and args.mode != 'references':
        parser.error('--retry-from is only for reference retries')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(RESOURCES['nice_increment'])
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='unseen-validation-metadata',
                    gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
