"""Expand the verified V28 geometry roster without changing its frozen gates."""

from datetime import date, timedelta

from audit_annotations import PROJECTED_CRS, crop_grid
from inventory_earlier_validation_sources import require
from plan_unseen_validation_sites import QA_PAD, build_plan, load, native_grids, positive_support, propose_control
from select_validation_cases import records


def _candidate_geometry(row, features, geometries, annotation_tree, control_tree, grids, unproject):
    from shapely.geometry import box

    event_id = row['event_id']
    index, properties = features[event_id]
    geometry = geometries[index]
    native_scene = grids[row['tile']]
    grid, reasons = crop_grid({'projected_geometry': geometry}, native_scene, 64)
    require(not reasons, f'Could not rebuild V28 grid for {event_id}: {reasons}')
    padded = [value + offset for value, offset in zip(grid['bounds'], (-QA_PAD, -QA_PAD, QA_PAD, QA_PAD))]
    core = box(*grid['bounds'])
    require(box(*native_scene['grid_10m']['bounds']).covers(box(*padded)),
            'Rebuilt QA source margin leaves the native tile: ' + event_id)

    hits = set(annotation_tree.query(core, predicate='intersects'))
    neighboring = [p for i, p in features.values() if i in hits]
    day = date.fromisoformat(row['annotation_date'])
    earlier = [p for p in neighboring if p['temporal_source_ids'] and p['observation_date'] < day.isoformat()]
    earlier_geometries = [geometries[features[p['event_id']][0]] for p in earlier]
    support = positive_support(geometry, earlier_geometries, grid)
    require(all(row[key] == value for key, value in support.items()),
            'Rebuilt target support differs from the V28 disposition: ' + event_id)
    control = propose_control(grid, geometry, control_tree)
    require(control == row['control_opportunity'],
            'Rebuilt full-context control geometry differs from V28: ' + event_id)
    require(support['positive_interior_pixels'] >= 4 and support['anchor_two_pixel_context_inside_crop'],
            'V28 support and full target context are required: ' + event_id)

    return {
        **row,
        'split': 'validation',
        'grid': grid,
        'qa_context_bounds': padded,
        'event_bounds': list(geometry.bounds),
        'frozen_group_footprint_bounds': properties['group_footprint_bounds'],
        'core_bounds_wgs84': list(unproject.transform_bounds(*grid['bounds'])),
        'qa_context_bounds_wgs84': list(unproject.transform_bounds(*padded)),
        'comparison_start_inclusive': (day - timedelta(days=30)).isoformat(),
        'comparison_end_inclusive': (day + timedelta(days=10)).isoformat(),
        'reference_start_inclusive': (day - timedelta(days=30)).isoformat(),
        'reference_end_inclusive': (day + timedelta(days=45)).isoformat(),
        'post_date': day.isoformat(),
        'post_date_role': 'reference release anchor only',
        'temporal_source_ids': properties['temporal_source_ids'],
        'native_grid_scene_id': native_scene['scene_id'],
        'native_grid_bounds': native_scene['grid_10m']['bounds'],
        'context_event_ids': sorted(p['event_id'] for p in neighboring),
        'prior_temporal_context_event_ids': sorted(p['event_id'] for p in earlier),
        'eroded_support_pixels': support['positive_interior_pixels'],
    }


def build_roster(v28_protocol, paths, plan, reviews):
    """Return every remaining eligible V28 event with its exact projected crop."""
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform
    from shapely.strtree import STRtree

    replayed = build_plan(v28_protocol, paths)
    archived = {key: value for key, value in plan.items() if key != 'protocol_sha256'}
    require(replayed == archived, 'Frozen V28 plan and all of its geometry dispositions must replay exactly')
    original_groups = set(plan['excluded_groups']['represented_in_original_pairs'])
    original_exposed = set(plan['excluded_groups']['prior_visual_exposure'])
    observed_by_qa = set(plan['excluded_groups']['prior_fresh_automatic_qa'])
    already_shortlisted = {site['group_id'] for site in plan['sites']}
    exposed = set(reviews['manual_exposed_group_ids'])
    require(reviews['schema_version'] == 34 and reviews['all_v28_shortlisted_groups_visually_exposed']
            and len(exposed) == len(reviews['manual_exposed_group_ids']) == 10
            and exposed == original_exposed | already_shortlisted,
            'All V34 exposure history, including the three earlier shortlist groups, is required')
    excluded = original_groups | exposed | observed_by_qa
    remaining = [row for row in plan['event_dispositions']
                 if not row['blockers'] and row['group_id'] not in excluded]
    require(len(remaining) == 29 and len({row['group_id'] for row in remaining}) == 14,
            'Expected the frozen set of 29 eligible events in 14 unexposed groups')

    project = Transformer.from_crs('EPSG:4326', PROJECTED_CRS, always_xy=True).transform
    unproject = Transformer.from_crs(PROJECTED_CRS, 'EPSG:4326', always_xy=True)
    catalogue = sorted(load(paths['catalogue'])['features'], key=lambda feature: feature['id'])
    features = {feature['id']: (index, feature['properties']) for index, feature in enumerate(catalogue)}
    geometries = [transform(project, shape(feature['geometry'])) for feature in catalogue]
    exclusions = [transform(project, shape(row['geometry'])) for row in records(paths['original_source_exclusions'])
                  if row.get('geometry') is not None]
    annotation_tree = STRtree(geometries)
    control_tree = STRtree(geometries + exclusions)
    grids = native_grids(paths['original_scene_inventory'])
    ordered = sorted(remaining, key=lambda row: (row['control_opportunity'] is None,
                                                 -row['positive_interior_pixels'], row['event_id'], row['group_id']))
    sites = []
    for sequence, row in enumerate(ordered, start=len(already_shortlisted) + 1):
        geometry = _candidate_geometry(row, features, geometries, annotation_tree, control_tree, grids, unproject)
        sites.append({'site_id': f'site-{sequence:02d}', **geometry})
    return {
        'schema_version': 35,
        'sites': sites,
        'exclusions': {
            'represented_in_original_pairs': sorted(original_groups),
            'prior_visual_exposure': sorted(exposed),
            'prior_fresh_automatic_qa': sorted(observed_by_qa),
        },
        'summary': {
            'frozen_v28_plan_reproduced': True,
            'v28_eligible_events': replayed['summary']['eligible_events'],
            'v28_eligible_groups': replayed['summary']['eligible_groups'],
            'remaining_eligible_events': len(sites),
            'remaining_eligible_groups': len({site['group_id'] for site in sites}),
            'control_opportunities': sum(site['control_opportunity'] is not None for site in sites),
            'source_image_reads': 0,
        },
    }
