#!/usr/bin/env python3
"""Find and acquire native Sentinel-2 scenes for plots in a GeoJSON file."""

import argparse
from datetime import date
import math
import os
from pathlib import Path
import resource
import sys

from inventory_earlier_validation_sources import RESOURCES, ROOT, digest, read_json, require, sha, write
from validate_sample import THREAD_VARIABLES, run_guarded


BOUNDING_BOX = ROOT / 'Datasets/BoundingBox'
COLLECTION = 'sentinel-2-l1c'


def load(path):
    return read_json(path.read_bytes())


def read_plots(path, codes):
    from shapely.errors import ShapelyError
    from shapely.geometry import shape

    data = load(path)
    require(isinstance(data, dict) and data.get('type') == 'FeatureCollection' and data.get('crs') is None
            and isinstance(data.get('features'), list) and data['features'],
            'Use a WGS84 GeoJSON FeatureCollection with named plot polygons')
    selected = set(codes or ())
    plots = []
    seen = set()
    for feature in data['features']:
        require(isinstance(feature, dict) and feature.get('type') == 'Feature'
                and isinstance(feature.get('properties'), dict), 'Malformed GeoJSON feature')
        properties = feature['properties']
        code = properties.get('code')
        require(isinstance(code, str) and code.strip() and code not in seen,
                'Plot codes must be nonempty, unique strings')
        seen.add(code)
        if selected and code not in selected:
            continue
        geometry = feature.get('geometry')
        require(isinstance(geometry, dict) and geometry.get('type') in ('Polygon', 'MultiPolygon'),
                f'{code} must have a Polygon or MultiPolygon geometry')
        try:
            polygon = shape(geometry)
        except (ShapelyError, TypeError, ValueError, IndexError) as error:
            raise ValueError(f'{code} has malformed coordinates') from error
        require(polygon.is_valid and not polygon.is_empty and all(map(math.isfinite, polygon.bounds)),
                f'{code} has invalid or empty coordinates')
        west, south, east, north = polygon.bounds
        require(-180 <= west < east <= 180 and -90 <= south < north <= 90 and east-west <= 180,
                f'{code} needs valid longitude/latitude bounds without an antimeridian crossing')
        plots.append({'code': code, 'status': properties.get('status'), 'bbox': [west, south, east, north]})
    require(selected <= seen, 'Unknown plot codes: '+', '.join(sorted(selected-seen)))
    return plots


def search_parameters(args):
    config = load(args.search_file)
    require(isinstance(config, dict) and config.get('collection') == COLLECTION,
            f'Search configuration must use the {COLLECTION} collection')
    start = args.start or date.fromisoformat(config['start'])
    end = args.end or date.fromisoformat(config['end'])
    count = config['count'] if args.count is None else args.count
    cloud = config.get('cloud_cover') if args.cloud_cover is None else args.cloud_cover
    require(start <= end, 'Search start date must be on or before the end date')
    require(type(count) is int and count > 0, 'Search count must be a positive integer')
    require(cloud is None or type(cloud) in (int, float) and 0 <= cloud <= 100,
            'Maximum scene cloud cover must be between 0 and 100 percent')
    return start, end, count, cloud


def checked_report(directory, command, filename):
    info = load(directory / 'metadata.json')
    raw = (directory / filename).read_bytes()
    checksum = sha(raw)
    require(info['schema_version'] == 1 and info['command'] == command
            and info['artifact_sha256'][filename] == checksum, f'Invalid or changed {command} report')
    return read_json(raw), checksum


def find_scene(plan, selected):
    feature = None
    plots = []
    for plot in plan['plots']:
        for candidate in plot['candidates']:
            uri = candidate['properties']['s2:product_uri']
            if selected not in (candidate['id'], uri, uri.removesuffix('.SAFE')):
                continue
            require(feature is None or feature == candidate, 'Scene identifier is ambiguous across plot results')
            feature = candidate
            row = {key: plot[key] for key in ('code', 'status', 'bbox')}
            if row not in plots:
                plots.append(row)
    require(feature is not None, f'Scene was not found in the search plan: {selected}')
    return feature, plots


def worker(args, output):
    if args.command == 'plan':
        from bbox_sentinel_catalogue import discover

        bbox_sha, config_sha = digest(args.bbox_file), digest(args.search_file)
        plots = read_plots(args.bbox_file, args.plot)
        start, end, count, cloud = search_parameters(args)
        plan = discover(plots, start, end, count, cloud, output)
        plan.update(collection=COLLECTION, start=start.isoformat(), end=end.isoformat(),
                    count=count, cloud_cover=cloud, bbox_file_sha256=bbox_sha, search_file_sha256=config_sha)
        require(digest(args.bbox_file) == bbox_sha and digest(args.search_file) == config_sha,
                'Search input changed during discovery')
        write(output/'plan.json', plan)
    elif args.command == 'preflight':
        from bbox_sentinel_preflight import preflight_scene

        plan, plan_sha = checked_report(args.plan, 'plan', 'plan.json')
        feature, plots = find_scene(plan, args.scene)
        scene = preflight_scene(feature, plots, output)
        scene['plan_sha256'] = plan_sha
        write(output/'preflight.json', scene)
    else:
        from bbox_sentinel_transfer import transfer_scene
        from preflight_remaining_validation_metadata import READY

        scene, preflight_sha = checked_report(args.preflight, 'preflight', 'preflight.json')
        require(scene['status'] == READY and not scene['issues'], 'Choose a passed preflight to download')
        destination = args.destination.absolute()
        transaction_root = destination.parent/'.sentinel-2-downloads'/destination.name
        result = transfer_scene(scene, destination, transaction_root)
        result['preflight_sha256'] = preflight_sha
        write(output/'download.json', result)

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    artifact_sha256 = {str(path.relative_to(output)): digest(path)
                       for path in sorted(output.rglob('*')) if path.is_file()}
    write(output/'metadata.json', {'schema_version': 1, 'command': args.command,
          'artifact_sha256': artifact_sha256, 'satellite_pixel_reads': 0,
          'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2)})


def describe(args):
    exit_code = 0
    if args.command == 'plan':
        for plot in load(args.output/'plan.json')['plots']:
            ids = ', '.join(candidate['id'] for candidate in plot['candidates'])
            print(f'{plot["code"]}: {plot["discovery_status"]}' + (f' ({ids})' if ids else ''))
            if plot['discovery_status'] in ('partial', 'failed'):
                print(f'  {plot.get("error", plot["stop_reason"])}')
                exit_code = 1
    elif args.command == 'preflight':
        result = load(args.output/'preflight.json')
        print(f'{result["scene_id"]}: {result["status"]}')
        for issue in result['issues']:
            print(issue)
        if result['issues']:
            exit_code = 1
        else:
            print(f'{len(result["assets"])} listed files, {result["measured_payload_bytes"]:,} bytes')
    else:
        print(f'Verified scene: {load(args.output/"download.json")["scene_id"]}')
    print(f'Report: {args.output.resolve()}')
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command')
    plan = commands.add_parser('plan', help='Search selected plot bounding boxes without downloading imagery')
    selection = plan.add_mutually_exclusive_group(required=True)
    selection.add_argument('--plot', action='append', help='Plot code; repeat for multiple plots')
    selection.add_argument('--all-plots', action='store_true', help='Search all plots within the run budget')
    plan.add_argument('--bbox-file', type=Path, default=BOUNDING_BOX/'bbox.geojson')
    plan.add_argument('--search-file', type=Path, default=BOUNDING_BOX/'search.json')
    plan.add_argument('--start', type=date.fromisoformat, help='Override the inclusive start date (YYYY-MM-DD)')
    plan.add_argument('--end', type=date.fromisoformat, help='Override the inclusive end date (YYYY-MM-DD)')
    plan.add_argument('--count', type=int, help='Override the maximum candidates per plot')
    plan.add_argument('--cloud-cover', type=float, help='Override the maximum scene-level cloud percentage')
    preflight = commands.add_parser('preflight', help='Verify the native metadata and file listing for one scene')
    preflight.add_argument('--plan', type=Path, required=True)
    preflight.add_argument('--scene', required=True, help='Exact catalogue item ID or SAFE scene ID from the plan')
    download = commands.add_parser('download', help='Stage, verify and publish one preflighted scene')
    download.add_argument('--preflight', type=Path, required=True)
    download.add_argument('--destination', type=Path, default=ROOT/'Datasets/Sentinel-2')
    for command in (plan, preflight, download):
        command.add_argument('--output', type=Path, required=True, help='New directory for this run report')
        command.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=RESOURCES['worker_rss_mib'],
                        min_available_gib=RESOURCES['available_memory_reserve_gib'],
                        max_seconds=RESOURCES['max_seconds_per_worker'])
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0
    for variable in THREAD_VARIABLES:
        os.environ[variable] = '1'
    try:
        if args.command == 'download':
            destination = args.destination.resolve()
            report = args.output.resolve()
            require(not report.is_relative_to(destination) and not destination.is_relative_to(report),
                    'Keep the run report separate from the scene destination')
        if args.worker_output is not None:
            os.nice(RESOURCES['nice_increment'])
            worker(args, args.worker_output)
            return 0
        run_guarded(args, script_path=__file__, operation='sentinel-bbox-'+args.command,
                    gdal_cache_mib=0, warp_memory_mib=0, quiet=True)
        return describe(args)
    except (KeyError, ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'Sentinel-2 {args.command}: {error}\n')


if __name__ == '__main__':
    sys.exit(main())
