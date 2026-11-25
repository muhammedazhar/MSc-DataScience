#!/usr/bin/env python3
"""Bind the three published site-03 observations to guarded crop QA and replay."""

import argparse
import os
from pathlib import Path
import re
import resource
import sys

from inventory_earlier_validation_sources import RESOURCES, ROOT, canonical, digest, require, rooted, sha, write
from preflight_fresh_validation_sources import checked_artifacts, evidence_hashes, load
from screen_repaired_validation import unchanged_assets


PROTOCOL = ROOT/'Docs/Reviews/ValidationSite03ScreenV30/protocol.json'
DATES = ('2018-05-10', '2018-05-30', '2018-06-04')
IMPLEMENTATION = 'Solutions/run_site03_quality.py'
QA_LIMITS = {'observation_count': 3, 'source_jp2_windows': 33, 'cloud_classifier_calls': 3}


def inputs(protocol_path):
    protocol = load(protocol_path)
    require(protocol['schema_version'] == 30 and protocol['resources'] == RESOURCES
            and protocol['fixed_site_id'] == 'site-03' and protocol['qa_limits'] == QA_LIMITS
            and protocol['visual_review_and_labels'] is False, 'V30 process scope differs')
    paths = {}
    for role, record in protocol['source_files'].items():
        path = rooted(record['path'])
        require(digest(path) == record['sha256'], 'Frozen V30 source differs: '+role)
        paths[role] = path
    require(IMPLEMENTATION in protocol['implementation_sha256'], 'Pin the V30 worker before use')
    for name, expected in (protocol['frozen_implementation_sha256'] | protocol['implementation_sha256']).items():
        require(digest(rooted(name)) == expected, 'Pinned V30 code differs: '+name)
    return protocol, paths


def read_publications(directory, plan, site):
    from bbox_sentinel_transfer import checked_scene
    from download import checked_report
    from pixelqa_scene_metadata import qa_scene_metadata

    require(tuple(row['date'] for row in plan['roster']) == DATES
            and plan['positive_pair'] == list(DATES[1:]) and plan['control_observations'] == list(DATES)
            and plan['listed_bytes'] == 1562638464 and plan['site'] == site, 'Fixed V30 acquisition plan differs')
    receipts, scene_records, publication_sha = {}, {}, {}
    for fixed in plan['roster']:
        day = fixed['date']
        preflight = rooted(str(Path(fixed['approved_preflight_path'])/'preflight.json'))
        source, source_sha = checked_report(preflight.parent, 'preflight', 'preflight.json')
        product = checked_scene(source)
        require(product['tile'] == '36UXA' and product['baseline'] == 'N0206' and product['orbit'] == 'R107'
                and source['scene_id'] == fixed['scene_id'] and source['acquisition_date'] == day
                and source_sha == fixed['preflight_sha256'], 'Pinned V29 preflight identity differs: '+day)
        run = (directory/day).resolve()
        meta = load(run/'metadata.json')
        require(meta['command'] == 'download' and meta['schema_version'] == 1 and meta['satellite_pixel_reads'] == 0
                and meta['artifact_sha256'] == evidence_hashes(run) and 'download.json' in meta['artifact_sha256'],
                'Completed V30 download report differs: '+day)
        checked_artifacts(run, meta['artifact_sha256'])
        report = load(run/'download.json')
        manifest = sha(canonical({'scene_id': source['scene_id'], 'assets': source['assets']}))
        require(report['transaction_state'] in ('complete', 'existing_verified') and report['preflight_sha256'] == source_sha
                and report['manifest_sha256'] == manifest == fixed['asset_manifest_sha256']
                and report['scene_id'] == source['scene_id'] and report['acquisition_date'] == day
                and report['payload_bytes'] == source['measured_payload_bytes'] == fixed['listed_bytes']
                and report['asset_count'] == len(source['assets']) == fixed['file_count'] == 58,
                'Published scene does not match the V29 manifest: '+day)
        folder = Path('Datasets/Sentinel-2')/source['scene_id']
        product_xml = rooted(str(folder/(source['scene_id']+'.SAFE')/'MTD_MSIL1C.xml'))
        local = product_xml.parent.parent
        require(Path(report['product_path']).resolve() == local, 'Published scene path differs: '+day)
        actual = qa_scene_metadata(local, ROOT)
        require(actual == report['final_metadata'] and actual['metadata_usable'] and actual['cloud_model_usable']
                and actual['shadow_geometry_usable'] and not actual['issues']
                and actual['grid_10m'] == source['metadata']['grid_10m'] and actual['crs'] == site['grid']['crs'],
                'Current SAFE metadata does not match published QA metadata: '+day)
        asset_receipts = report['source_receipts']
        require(asset_receipts.keys() == {asset['target_relpath'] for asset in source['assets']},
                'Published 58-file roster differs: '+day)
        for asset in source['assets']:
            relative = asset['target_relpath']
            receipt = asset_receipts[relative]
            name = str(folder/relative)
            path = rooted(name)
            stat = path.stat()
            require(path.is_relative_to(local) and name not in receipts and stat.st_size == receipt['bytes'] == asset['size']
                    and receipt['md5'] == asset['etag'][1:-1] and receipt['etag'] == asset['etag']
                    and receipt['source_url'] == asset['url'] and re.fullmatch('[0-9a-f]{64}', receipt['sha256']),
                    'Published file receipt differs: '+name)
            receipts[name] = {**receipt, 'stat': [stat.st_size, stat.st_mtime_ns]}
        publication_sha[day] = digest(run/'metadata.json')
        scene_records[day] = actual
    require(len(receipts) == 174, 'Expected all 174 provider-bound files')
    return scene_records, receipts, publication_sha


def prior_assets(paths):
    stats = {}
    for key in ('old_quality_summary', 'v24_screen', 'v27_screen'):
        record = load(paths[key])['source_file_stats']
        require(all(stats[name] == record[name] for name in stats.keys() & record.keys()),
                'Prior QA source fingerprints disagree: '+key)
        stats.update(record)
    result = {name: {'stat': value} for name, value in stats.items()}
    unchanged_assets(result)
    return result


def require_phase(directory, filename, mode, checksum):
    directory = directory.resolve()
    metadata = load(directory/'metadata.json')
    require(metadata['protocol_sha256'] == checksum and metadata['mode'] == mode
            and metadata['execution_status'] == 'completed' and filename in metadata['artifact_sha256']
            and metadata['artifact_sha256'] == evidence_hashes(directory), 'Completed V30 evidence differs: '+mode)
    checked_artifacts(directory, metadata['artifact_sha256'])
    return load(directory/filename)


def worker(args, output):
    from plan_site03_qa import prepare_geometry

    protocol, paths = inputs(args.protocol)
    checksum = digest(args.protocol)
    site = load(paths['v29_protocol'])['site']
    plan = load(paths['acquisition_plan'])
    old_assets = prior_assets(paths)
    scenes, receipts, publication_sha = read_publications(args.publication, plan, site)
    require(not receipts.keys() & old_assets.keys(), 'V30 inputs overlap earlier source assets')
    geometry = prepare_geometry(protocol, paths, scenes)
    bound = {**geometry, 'site': site, 'scenes': scenes, 'source_asset_receipts': receipts, 'protocol_sha256': checksum}
    release = {key: bound[key] for key in ('pair','geometry','control','site','source_asset_receipts','protocol_sha256')}
    release['publication_metadata_sha256_by_date'] = publication_sha
    if args.mode == 'prepare':
        write(output/'lineage.json', release)
        details = {'source_jp2_window_reads': 0, 'cloud_model_executions': 0,
                   'checks': {'fixed_geometry': True, 'cumulative_targets': True, 'split_isolation': True}}
    else:
        from screen_site03_quality import screen, verify

        require(require_phase(args.lineage, 'lineage.json', 'prepare', checksum) == release,
                'Published inputs differ from the frozen V30 lineage')
        if args.mode == 'screen':
            details = screen(output, bound)
        else:
            require_phase(args.screen, 'screen.json', 'screen', checksum)
            details = verify(output, args.screen, bound)
        require(details['source_jp2_window_reads'] <= QA_LIMITS['source_jp2_windows']
                and details['cloud_model_executions'] <= QA_LIMITS['cloud_classifier_calls'], 'V30 pixel QA bounds exceeded')
        if args.mode == 'verify':
            require(details['source_jp2_window_reads'] == details['cloud_model_executions'] == 0,
                    'V30 replay read original imagery or ran cloud classification')
    unchanged_assets(receipts)
    unchanged_assets(old_assets)
    require(digest(args.protocol) == checksum and 'torch' not in sys.modules, 'Protocol changed or forest runtime imported')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {'schema_version': 30, 'execution_status': 'completed', 'mode': args.mode,
          'protocol_sha256': checksum, 'artifact_sha256': evidence_hashes(output), 'labels_approved': 0,
          'forest_model_executions': 0, 'visually_reviewed': False, 'prior_source_file_stat_count': len(old_assets),
          'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2), **details})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('prepare', 'screen', 'verify'))
    parser.add_argument('--protocol', default=PROTOCOL, type=Path)
    parser.add_argument('--publication', required=True, type=Path)
    parser.add_argument('--lineage', type=Path)
    parser.add_argument('--screen', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=RESOURCES['worker_rss_mib'],
                        min_available_gib=RESOURCES['available_memory_reserve_gib'],
                        max_seconds=RESOURCES['max_seconds_per_worker'])
    args = parser.parse_args()
    if (args.mode != 'prepare') != (args.lineage is not None):
        parser.error('--lineage is required for screen and verify')
    if (args.mode == 'verify') != (args.screen is not None):
        parser.error('--screen is required only for verify')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output is not None:
        os.nice(RESOURCES['nice_increment'])
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='site03-frozen-quality', gdal_cache_mib=32, warp_memory_mib=16)


if __name__ == '__main__':
    main()
