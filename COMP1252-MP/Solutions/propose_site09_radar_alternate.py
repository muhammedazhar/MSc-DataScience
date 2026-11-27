"""Find a separate control within the fixed site-09 crop."""

from check_stage1_control_map import ROAD_MARGIN_METRES, _shape_snapshot
from inventory_earlier_validation_sources import require
from plan_unseen_validation_sites import load, propose_control, records


def propose_alternate(site, v28_paths, map_snapshot, stable_woodland, excluded_controls):
    """Use projected native radar-cell coverage to propose geometry for optical review."""
    from affine import Affine
    import numpy as np
    from pyproj import Transformer
    from scipy.ndimage import binary_dilation
    from shapely import make_valid, union_all
    from shapely.geometry import box, shape
    from shapely.ops import transform
    from shapely.strtree import STRtree
    from validate_sample import burn

    grid = site['grid']
    require(site['site_id'] == 'site-09' and grid['crs'] == 'EPSG:32636'
            and grid['width'] == grid['height'] == 64, 'Expected the fixed site-09 UTM crop')
    if stable_woodland.is_empty:
        return None
    require(stable_woodland.is_valid and stable_woodland.geom_type in ('Polygon', 'MultiPolygon'),
            'Expected a valid polygon union of stable radar cells in EPSG:32636')

    original = site['control_opportunity']
    radius = original['context_radius_pixels']
    margin = original['catalogue_exclusion_margin_metres']
    west, south, east, north = grid['bounds']
    require((radius, margin) == (2, ROAD_MARGIN_METRES) and grid['transform'] == [10, 0, west, 0, -10, north]
            and east-west == north-south == 640, 'Frozen control and grid geometry differs')
    project = Transformer.from_crs('EPSG:4326', grid['crs'], always_xy=True).transform
    features = load(v28_paths['catalogue'])['features']
    catalogue = {row['id']: transform(project, make_valid(shape(row['geometry']))) for row in features}
    exclusions = [transform(project, make_valid(shape(row['geometry'])))
                  for row in records(v28_paths['original_source_exclusions']) if row.get('geometry') is not None]
    require(len(catalogue) == len(features) == 3911 and exclusions, 'Frozen annotation barriers differ')
    anchor = catalogue[site['event_id']]
    frozen = list(catalogue.values()) + exclusions
    require(propose_control(grid, anchor, STRtree(frozen)) == original, 'Frozen V28 control does not replay')

    by_role, _ = _shape_snapshot(map_snapshot, project)
    roads = [geometry for _, geometry in by_role['road']]
    nonwood = union_all([geometry for _, geometry in by_role['nonwood']])
    woodland = union_all([geometry for _, geometry in by_role['woodland']]).difference(nonwood)
    obstacles = [*frozen, *roads, nonwood]
    if not woodland.is_empty:
        obstacles.append(woodland.boundary)
    obstacle_tree = STRtree([geometry for geometry in obstacles if not geometry.is_empty])
    search_bounds = box(west-margin, south-margin, east+margin, north+margin)
    buffered = [make_valid(make_valid(obstacle_tree.geometries[index]).buffer(margin))
                for index in obstacle_tree.query(search_bounds)]

    # Use the frozen raster barrier rule. Radar coverage and old-square identity
    # are separate context tests with no added buffer.
    raster_grid = {**grid, 'transform': Affine(*grid['transform'])}
    anchor_mask = burn([anchor], raster_grid).astype(bool)
    target_context = binary_dilation(anchor_mask, structure=np.ones((5, 5), dtype=bool))
    unavailable = target_context | burn(buffered, raster_grid, all_touched=True).astype(bool)
    old_contexts = {tuple(control['context_bounds']) for control in (original, *excluded_controls)}

    def bounds(row, col, size):
        return [west + 10*col, north - 10*(row+size), west + 10*(col+size), north - 10*row]

    for size in (16, 12, 8):
        context_size = size + 2*radius
        for row in range(radius, grid['height']-size-radius+1):
            for col in range(radius, grid['width']-size-radius+1):
                top, left = row-radius, col-radius
                if unavailable[top:top+context_size, left:left+context_size].any():
                    continue
                context_bounds = bounds(top, left, context_size)
                if tuple(context_bounds) in old_contexts or not stable_woodland.covers(box(*context_bounds)):
                    continue
                return {
                    **original,
                    'row': row, 'col': col, 'width': size, 'height': size, 'pixels': size**2,
                    'bounds': bounds(row, col, size), 'context_bounds': context_bounds,
                    'context_width': context_size, 'context_height': context_size,
                    'context_pixels': context_size**2,
                }
    return None
