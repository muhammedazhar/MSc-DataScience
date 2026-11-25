"""Select a bounded metadata roster from untouched, frozen validation groups."""

from collections import Counter
from datetime import date, timedelta
import hashlib

from audit_annotations import PROJECTED_CRS, crop_grid
from inventory_earlier_validation_sources import read_json, require
from select_validation_cases import grid_bounds, records


SOURCE_ROLES = {
    'catalogue', 'splits', 'pairs', 'training_manifest', 'original_scene_inventory',
    'annotation_manifest', 'original_source_exclusions', 'v23_plan', 'v24_screen', 'v27_screen',
}
MAX_SITES = 3
QA_PAD = 1200


def load(path):
    return read_json(path.read_bytes())


def frozen_groups(paths, features, manifest):
    splits = load(paths['splits'])
    require(splits['crs'] == PROJECTED_CRS and splits['patch_size_pixels'] == 64
            and splits['resolution_m'] == 10 and len(splits['groups']) == 601,
            'Frozen spatial split or crop geometry differs')
    by_event = {event: (group['group_id'], group['split']) for group in splits['groups']
                for event in group['event_ids']}
    require(len(features) == len(by_event) == len(manifest) == 3911
            and len({group['group_id'] for group in splits['groups'] if group['split'] == 'validation'}) == 91,
            'Frozen annotation or validation-group coverage differs')
    for row in features:
        event_id = row['id']
        properties = row['properties']
        require(event_id == properties['event_id']
                and (properties['group_id'], properties['split']) == by_event[event_id]
                and all(properties[key] == manifest[event_id][key] for key in (
                    'group_id', 'split', 'tile', 'observation_date', 'temporal_source_ids')),
                'Frozen annotation split or source membership differs: '+event_id)
    old_pairs = [row for row in records(paths['pairs']) if row['split'] == 'validation']
    require(len(old_pairs) == 1146 and len({row['pair_id'] for row in old_pairs}) == 1146,
            'Frozen validation pair coverage differs')
    old_groups = {row['group_id'] for row in old_pairs}
    for row in old_pairs:
        require(all(by_event[event] == (row['group_id'], 'validation')
                    and row['pair_id'] in manifest[event]['candidate_pair_ids']
                    for event in row['anchor_event_ids']), 'Frozen validation pair join differs')
    v23, v24, v27 = [load(paths[role]) for role in ('v23_plan', 'v24_screen', 'v27_screen')]
    exposed = set(v23['excluded_previous_review_groups'])
    require(len(exposed) == 7 and exposed == set(v24['prior_manual_exposed_group_ids'])
            == set(v27['prior_manual_exposed_group_ids']), 'Prior visual-exposure history differs')
    probed_24 = set(v24['groups_observed_by_automatic_qa'])
    probed_27 = set(v27['groups_observed_by_automatic_qa'])
    probed = probed_24 | probed_27
    require(len(probed_24) == len(probed_27) == 2 and len(probed) == 4 and not exposed & probed
            and probed == set(v23['clear_post_alternatives'])
            and not v24['processing_errors'] and not v27['processing_errors']
            and v24['review_candidate_pair_ids'] == v27['review_candidate_pair_ids'] == [],
            'Prior four-group QA disposition differs')
    return old_groups, exposed, probed


def native_grids(path):
    grids = {}
    for scene in sorted(records(path), key=lambda row: row['scene_id']):
        tile = scene['tile']
        if not scene['metadata_usable'] or scene['processing_level'] != 'L1C':
            continue
        grid = scene['grid_10m']
        require(scene['crs'] == PROJECTED_CRS and grid['xdim'] == -grid['ydim'] == 10
                and grid['width'] == grid['height'] == 10980,
                'Original native grid is outside the frozen 10 m study geometry')
        if tile in grids:
            require(grids[tile]['grid_10m'] == grid, 'Original tile grids disagree: '+tile)
        else:
            grids[tile] = scene
    return grids


def positive_support(geometry, earlier, grid):
    """Bound support using every known temporal loss before the annotation."""
    import numpy as np
    from affine import Affine
    from scipy.ndimage import binary_erosion
    from validate_sample import burn

    raster_grid = {**grid, 'transform': Affine(*grid['transform'])}
    anchor = burn([geometry], raster_grid).astype(bool)
    target = anchor & ~burn(earlier, raster_grid).astype(bool)
    interior = binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
    height, width = anchor.shape
    return {'anchor_pixels': int(anchor.sum()), 'positive_target_pixels': int(target.sum()),
            'positive_interior_pixels': int(interior.sum()),
            'anchor_two_pixel_context_inside_crop': int(anchor[2:height-2, 2:width-2].sum()) == int(anchor.sum()),
            'positive_interior_sha256': hashlib.sha256(interior.tobytes()).hexdigest()}


def propose_control(grid, anchor_geometry, catalogue_tree):
    """Return one prediction-blind control geometry using only annotation vectors."""
    from affine import Affine
    import numpy as np
    from scipy.ndimage import binary_dilation
    from shapely import make_valid
    from shapely.geometry import box
    from validate_sample import burn

    west, south, east, north = grid['bounds']
    cell_size, radius, margin = 10, 2, 20
    raster_grid = {**grid, 'transform': Affine(*grid['transform'][:6])}
    anchor = anchor_geometry if anchor_geometry.is_valid else make_valid(anchor_geometry)
    mask = burn([anchor], raster_grid).astype(bool)
    if not mask.any():
        return None
    target_context = binary_dilation(mask, structure=np.ones((5, 5), dtype=bool))
    query = box(west-margin, south-margin, east+margin, north+margin)
    buffered = []
    for index in catalogue_tree.query(query):
        geometry = catalogue_tree.geometries[index]
        repaired = geometry if geometry.is_valid else make_valid(geometry)
        grown = repaired.buffer(margin)
        if not grown.is_empty:
            buffered.append(grown if grown.is_valid else make_valid(grown))
    unavailable = target_context | burn(buffered, raster_grid, all_touched=True).astype(bool)

    def bounds(row, col, size):
        return [float(west+cell_size*col), float(north-cell_size*(row+size)),
                float(west+cell_size*(col+size)), float(north-cell_size*row)]

    for size in (16, 12, 8):
        for row in range(radius, grid['height']-size-radius+1):
            for col in range(radius, grid['width']-size-radius+1):
                top, left = row-radius, col-radius
                context_size = size+2*radius
                if unavailable[top:top+context_size, left:left+context_size].any():
                    continue
                return {'row': row, 'col': col, 'width': size, 'height': size,
                        'pixels': size**2, 'bounds': bounds(row, col, size), 'crs': grid['crs'],
                        'context_bounds': bounds(top, left, context_size),
                        'context_width': context_size, 'context_height': context_size,
                        'context_pixels': context_size**2, 'context_radius_pixels': radius,
                        'catalogue_exclusion_margin_metres': margin,
                        'control_class_status': 'unreviewed', 'pixel_qa_status': 'unassessed',
                        'independent_site': False, 'evaluation_approved': False}
    return None


def build_plan(protocol, paths):
    """Rebuild site ranks from the frozen metadata without catalogue or image I/O."""
    from pyproj import Transformer
    from shapely.geometry import box, shape
    from shapely.ops import transform
    from shapely.strtree import STRtree

    require(protocol['schema_version'] == 28 and SOURCE_ROLES <= paths.keys(), 'V28 planning inputs differ')
    features = sorted(load(paths['catalogue'])['features'], key=lambda feature: feature['id'])
    manifest = {row['event_id']: row for row in records(paths['annotation_manifest'])}
    old_groups, exposed, probed = frozen_groups(paths, features, manifest)
    grids = native_grids(paths['original_scene_inventory'])
    training = list(records(paths['training_manifest']))
    require(len(training) == 13 and all(row['split'] == 'train' for row in training),
            'All 13 frozen training grids are required')
    project = Transformer.from_crs('EPSG:4326', PROJECTED_CRS, always_xy=True).transform
    unproject = Transformer.from_crs(PROJECTED_CRS, 'EPSG:4326', always_xy=True)
    event_geometry = [transform(project, shape(feature['geometry'])) for feature in features]
    annotation_tree = STRtree(event_geometry)
    prohibited = [box(*feature['properties']['group_footprint_bounds']) for feature in features
                  if feature['properties']['split'] in ('train', 'test')]
    prohibited.extend(box(*grid_bounds(row['grid'], PROJECTED_CRS)) for row in training)
    barrier_tree = STRtree(prohibited)
    exclusions = [transform(project, shape(row['geometry'])) for row in records(paths['original_source_exclusions'])
                  if row.get('geometry') is not None]
    require(exclusions, 'Frozen unresolved-source footprints are required')
    exclusions_tree = STRtree(exclusions)
    control_tree = STRtree(event_geometry + exclusions)
    eligible, dispositions = [], []
    new_anchor_count = 0
    new_anchor_groups = set()
    for feature, geometry in zip(features, event_geometry):
        p = feature['properties']
        if p['split'] != 'validation':
            continue
        event_id, group_id, tile = (p[key] for key in ('event_id', 'group_id', 'tile'))
        day = date.fromisoformat(p['observation_date'])
        start, end = day-timedelta(days=30), day+timedelta(days=10)
        row = {'event_id': event_id, 'group_id': group_id, 'tile': tile, 'annotation_date': day.isoformat(),
               'annotation_flags': p['flags'], 'blockers': [], 'control_opportunity': None}
        dispositions.append(row)
        blockers = row['blockers']
        if group_id in old_groups or manifest[event_id]['candidate_pair_ids']:
            blockers.append('represented_in_frozen_candidate_pairs')
        if group_id in exposed or group_id in probed:
            blockers.append('previously_reviewed_or_screened_group')
        if not p['temporal_source_ids'] or p['flags']:
            blockers.append('anchor_has_unverified_temporal_provenance_or_flags')
        if tile not in grids:
            blockers.append('native_grid_unavailable')
        if blockers:
            continue
        new_anchor_count += 1
        new_anchor_groups.add(group_id)
        scene = grids[tile]
        grid, reasons = crop_grid({'projected_geometry': geometry}, scene, 64)
        core = box(*grid['bounds'])
        padded = [value+offset for value, offset in zip(grid['bounds'], (-QA_PAD, -QA_PAD, QA_PAD, QA_PAD))]
        blockers.extend(reasons)
        if not box(*scene['grid_10m']['bounds']).covers(box(*padded)):
            blockers.append('native_grid_does_not_cover_full_1200m_context')
        if not box(*p['group_footprint_bounds']).covers(core):
            blockers.append('crop_outside_frozen_grouping_envelope')
        if len(barrier_tree.query(core, predicate='intersects')):
            blockers.append('crop_overlaps_frozen_train_test_barrier')
        if len(exclusions_tree.query(core, predicate='intersects')):
            blockers.append('unresolved_original_annotation_in_crop')
        hits = sorted(int(index) for index in annotation_tree.query(core, predicate='intersects'))
        neighboring = [features[index]['properties'] for index in hits]
        if any(event['group_id'] != group_id for event in neighboring):
            blockers.append('crop_includes_another_group')
        if any('conflicting_observation_dates' in event['flags'] for event in neighboring):
            blockers.append('conflicting_annotation_in_crop')
        if any(not event['temporal_source_ids'] and start < date.fromisoformat(event['observation_date']) <= end
               for event in neighboring):
            blockers.append('unverified_temporal_label_in_proposed_window')
        if blockers:
            continue
        earlier = [index for index in hits if features[index]['properties']['temporal_source_ids']
                   and features[index]['properties']['observation_date'] < day.isoformat()]
        support = positive_support(geometry, [event_geometry[index] for index in earlier], grid)
        row.update(support)
        if support['positive_interior_pixels'] < 4:
            blockers.append('fewer_than_four_eroded_positive_pixels')
        if not support['anchor_two_pixel_context_inside_crop']:
            blockers.append('two_pixel_target_context_would_clip_the_crop')
        if blockers:
            continue
        row['control_opportunity'] = propose_control(grid, geometry, control_tree)
        eligible.append({**row, 'split': 'validation', 'grid': grid, 'qa_context_bounds': padded,
                         'event_bounds': list(geometry.bounds), 'frozen_group_footprint_bounds': p['group_footprint_bounds'],
                         'core_bounds_wgs84': list(unproject.transform_bounds(*grid['bounds'])),
                         'qa_context_bounds_wgs84': list(unproject.transform_bounds(*padded)),
                         'comparison_start_inclusive': start.isoformat(), 'comparison_end_inclusive': end.isoformat(),
                         'reference_start_inclusive': start.isoformat(),
                         'reference_end_inclusive': (day+timedelta(days=45)).isoformat(),
                         'post_date': day.isoformat(), 'post_date_role': 'reference release anchor only',
                         'temporal_source_ids': p['temporal_source_ids'], 'native_grid_scene_id': scene['scene_id'],
                         'native_grid_bounds': scene['grid_10m']['bounds'],
                         'context_event_ids': sorted(event['event_id'] for event in neighboring),
                         'prior_temporal_context_event_ids': sorted(features[index]['id'] for index in earlier),
                         'eroded_support_pixels': support['positive_interior_pixels']})
    sites, seen = [], set()
    for row in sorted(eligible, key=lambda item: (item['control_opportunity'] is None, -item['positive_interior_pixels'],
                                                  item['event_id'], item['group_id'])):
        if row['group_id'] not in seen and len(sites) < MAX_SITES:
            sites.append({'site_id': f'site-{len(sites)+1:02d}', **row})
            seen.add(row['group_id'])
    selected = {row['event_id'] for row in sites}
    for row in dispositions:
        row['disposition'] = ('blocked' if row['blockers'] else
                              'selected_for_metadata_search' if row['event_id'] in selected else
                              'eligible_outside_bounded_roster')
    return {'schema_version': 28, 'sites': sites, 'event_dispositions': dispositions,
            'excluded_groups': {'represented_in_original_pairs': sorted(old_groups),
                                'prior_visual_exposure': sorted(exposed), 'prior_fresh_automatic_qa': sorted(probed)},
            'selection_rule': ('Prioritize a geometry-only control opportunity, then rank by descending eroded '
                               'positive pixels after subtracting all earlier temporal annotations, then event ID '
                               'and group ID. Select up to three distinct groups.'),
            'summary': {'validation_events': len(dispositions), 'original_validation_pair_groups': len(old_groups),
                        'new_temporal_anchor_events_before_geometry': new_anchor_count,
                        'new_temporal_anchor_groups_before_geometry': len(new_anchor_groups),
                        'eligible_events': len(eligible), 'eligible_groups': len({row['group_id'] for row in eligible}),
                        'eligible_events_with_control_geometry': sum(row['control_opportunity'] is not None for row in eligible),
                        'sites': len(sites), 'blocker_counts': dict(sorted(Counter(
                            blocker for row in dispositions for blocker in row['blockers']).items()))}}
