#!/usr/bin/env python3
"""Reconstruct unreviewed groups and replay their bounded reference search."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import sys

from inventory_unseen_validation_sites import (
    PROTOCOL as V28_PROTOCOL, RESOURCES, ROOT, build_sentinel, checked_phase,
    checked_protocol, digest, load, require, rooted, verify_sentinel, write,
)
from preflight_fresh_validation_sources import evidence_hashes


REPORT = ROOT/'Docs/Reviews/ValidationStage1SearchV35'
PROTOCOL = REPORT/'protocol.json'
MODES = ('roster', 'references', 'verify-references', 'sentinel', 'verify-sentinel')


def verified_protocol():
    protocol = load(PROTOCOL)
    require(protocol['schema_version'] == 35 and protocol['resources'] == RESOURCES
            and protocol['group_count'] == 14 and protocol['maximum_eligible_sites'] == 29,
            'Stage 1 protocol scope differs')
    paths = {}
    for role, receipt in protocol['input_sha256'].items():
        path = rooted(receipt['path'])
        require(digest(path) == receipt['sha256'], 'Stage 1 pinned input changed: '+role)
        paths[role] = path
    return paths


def worker(args, output):
    paths = verified_protocol()
    checksum = digest(PROTOCOL)
    if args.mode == 'roster':
        from plan_stage1_site_roster import build_roster

        protocol, v28_paths, _ = checked_protocol(V28_PROTOCOL)
        roster = build_roster(protocol, v28_paths, load(paths['v28_plan']), load(paths['exposure_history']))
        write(output/'roster.json', roster)
    else:
        checked_phase(args.roster, 'roster.json', 'roster', checksum)
        roster = load(args.roster/'roster.json')
        sites = roster['sites']
        require(len(sites) == 29 and len({site['group_id'] for site in sites}) == 14
                and len({site['site_id'] for site in sites}) == 29, 'Stage 1 roster differs')
        if args.mode in ('references', 'verify-references'):
            from search_stage1_landsat import build_search, verify_search

            if args.mode == 'references':
                now = datetime.now(timezone.utc).isoformat(timespec='seconds')
                build_search(output, sites, checksum, now, retry_from=args.retry_from)
            else:
                verify_search(args.inventory, output, sites, checksum)
        else:
            selected = select_sentinel_sites(args.subjects, sites, checksum)
            if args.mode == 'sentinel':
                build_sentinel(output, selected, checksum)
            else:
                verify_sentinel(args.inventory, output, selected, checksum)
    require(digest(PROTOCOL) == checksum and not {'torch', 's2cloudless'} & sys.modules.keys(),
            'Protocol changed or model runtime imported during stage 1 metadata work')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'schema_version': 35, 'mode': args.mode, 'execution_status': 'completed',
        'protocol_sha256': checksum, 'artifact_sha256': evidence_hashes(output),
        'satellite_pixel_reads': 0, 'image_asset_downloads': 0, 'model_executions': 0, 'labels_approved': 0,
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2),
    })


def select_sentinel_sites(subjects_path, sites, checksum):
    subjects = load(subjects_path)
    require(subjects['protocol_sha256'] == checksum, 'Selected subjects refer to a different protocol')
    by_id = {site['site_id']: site for site in sites}
    selected = [by_id[key] for key in subjects['sentinel_site_ids']]
    require(0 < len(selected) <= 3 and len({site['group_id'] for site in selected}) == len(selected),
            'Catalogue follow-up must use at most three distinct groups in a batch')
    return selected


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=MODES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--roster', type=Path)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--retry-from', type=Path)
    parser.add_argument('--subjects', type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    if (args.mode != 'roster') != (args.roster is not None):
        parser.error('--roster is required for catalogue and replay modes')
    if args.mode.startswith('verify-') != (args.inventory is not None):
        parser.error('--inventory is required for replay modes')
    if args.mode.endswith('sentinel') != (args.subjects is not None):
        parser.error('--subjects is required for Sentinel catalogue modes')
    if args.retry_from and args.mode != 'references':
        parser.error('--retry-from is only used for references')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='stage1-site-search',
                    gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
