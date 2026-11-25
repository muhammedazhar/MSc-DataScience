#!/usr/bin/env python3
"""Find a map-screened control inside each exact V28 crop."""

import argparse
import os
from pathlib import Path
import resource
import sys

from inventory_unseen_validation_sites import PROTOCOL as V28_PROTOCOL
from inventory_unseen_validation_sites import RESOURCES, ROOT, checked_protocol, digest, load, require, rooted, write
from preflight_fresh_validation_sources import evidence_hashes


REPORT = ROOT/'Docs/Reviews/ValidationRemainingSitesV32'
PROTOCOL = REPORT/'control_search_protocol.json'
SITES = ('site-01', 'site-02')


def plan_controls():
    from pyproj import Transformer
    from shapely import make_valid, union_all
    from shapely.geometry import box, shape
    from shapely.ops import transform
    from shapely.strtree import STRtree
    from plan_unseen_validation_sites import propose_control, records

    protocol = load(PROTOCOL)
    require(protocol['schema_version'] == 32 and tuple(protocol['site_ids']) == SITES
            and protocol['resources'] == RESOURCES and protocol['map_clearance_metres'] == 20,
            'V32 mapped-control search scope differs')
    for name, expected in protocol['input_sha256'].items():
        require(digest(rooted(name)) == expected, 'V32 control search input changed: '+name)
    _, paths, _ = checked_protocol(V28_PROTOCOL)
    v28_plan = load(V28_PROTOCOL.parent/'plan/plan.json')
    sites = {site['site_id']: site for site in v28_plan['sites']}
    observed = load(rooted(protocol['map_features_path']))
    require(observed['type'] == 'FeatureCollection', 'Expected bounded map features')
    project = Transformer.from_crs('EPSG:4326', 'EPSG:32636', always_xy=True).transform

    def project_shape(geometry):
        return transform(project, make_valid(shape(geometry)))

    catalogue = {feature['id']: project_shape(feature['geometry'])
                 for feature in load(paths['catalogue'])['features']}
    source_exclusions = [project_shape(row['geometry']) for row in records(paths['original_source_exclusions'])
                         if row.get('geometry') is not None]
    catalogue_geometries = list(catalogue.values()) + source_exclusions
    require(len(catalogue) == 3911 and source_exclusions, 'Frozen catalogue or footprint exclusions missing')
    catalogue_tree = STRtree(catalogue_geometries)
    results = []
    for site_id in SITES:
        site = sites[site_id]
        grid = site['grid']
        anchor = catalogue[site['event_id']]
        original = propose_control(grid, anchor, catalogue_tree)
        require(original == site['control_opportunity'], 'Original V28 control does not reproduce: '+site_id)

        mapped_features = [feature for feature in observed['features'] if feature['properties']['site_id'] == site_id]
        roads = [project_shape(feature['geometry']) for feature in mapped_features
                 if feature['properties']['role'] == 'road']
        wooded = [(feature['properties'], project_shape(feature['geometry']))
                  for feature in mapped_features if feature['properties']['role'] == 'wooded_polygon']
        open_areas = [project_shape(feature['geometry']) for feature in mapped_features
                      if feature['properties']['role'] == 'nonwooded_polygon']
        require(all(line.geom_type in ('LineString', 'MultiLineString') for line in roads)
                and all(polygon.geom_type in ('Polygon', 'MultiPolygon')
                        for polygon in [*(polygon for _, polygon in wooded), *open_areas]),
                'Unexpected map geometry type: '+site_id)
        map_extent = transform(project, box(*protocol['map_response_bounds_wgs84'][site_id]))
        core = box(*grid['bounds'])
        require(map_extent.covers(core), 'Map response envelope does not contain the full crop: '+site_id)
        # The existing proposal applies the same 20 m margin to each added
        # exclusion and keeps the complete 2 px context in the frozen crop.
        known_woodland = union_all([polygon for _, polygon in wooded]).intersection(map_extent)
        known_woodland = known_woodland.difference(union_all(open_areas))
        woodland_exclusion = core.buffer(20).difference(known_woodland)
        map_exclusions = [*roads, woodland_exclusion]
        alternative = propose_control(grid, anchor, STRtree(catalogue_geometries + map_exclusions))
        row = {
            'site_id': site_id, 'group_id': site['group_id'], 'grid': grid,
            'original_control': original, 'alternate_control': alternative,
            'road_features_checked': len(roads), 'wooded_polygons_checked': len(wooded),
            'nonwooded_polygons_checked': len(open_areas),
            'map_rule': 'Mapped road centerlines and areas outside mapped woodland excluded by 20 m.',
            'alternate_control_class_status': 'unverified',
            'status': 'mapped_geometry_candidate' if alternative else 'no_square_passed_mapped_geometry',
        }
        if alternative:
            context = box(*alternative['context_bounds'])
            woodland_margin = context.distance(known_woodland.boundary)
            require(alternative != original and known_woodland.covers(context)
                    and woodland_margin >= 20
                    and all(context.distance(road) >= 20 for road in roads),
                    'Alternate control must clear the entire mapped context: '+site_id)
            covering = [properties for properties, polygon in wooded if polygon.covers(context)]
            distances = [context.distance(road) for road in roads]
            row.update(
                wooded_osm_features_covering_context=covering,
                nearest_mapped_road_metres=round(min(distances), 3) if distances else None,
                mapped_woodland_margin_metres=round(woodland_margin, 3),
            )
        results.append(row)
    return results


def worker(output):
    checksum = digest(PROTOCOL)
    write(output/'controls.json', {
        'schema_version': 32, 'protocol_sha256': checksum, 'sites': plan_controls(),
        'satellite_pixel_reads': 0, 'image_asset_downloads': 0, 'labels_approved': 0,
    })
    require(checksum == digest(PROTOCOL) and not {'torch', 's2cloudless'} & sys.modules.keys(),
            'Protocol changed or model runtime imported')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'schema_version': 32, 'protocol_sha256': checksum, 'execution_status': 'completed',
        'artifact_sha256': evidence_hashes(output), 'satellite_pixel_reads': 0, 'model_executions': 0,
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2),
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='remaining-sites-control-metadata',
                    gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
