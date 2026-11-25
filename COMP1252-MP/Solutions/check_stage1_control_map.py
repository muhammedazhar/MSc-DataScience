"""Bounded current-map checks for V35 geometry-only forest controls."""

from datetime import datetime, timezone
from math import ceil, floor
from pathlib import Path
import re
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from inventory_earlier_validation_sources import canonical, digest, read_json, require, sha


OSM_MAP = 'https://api.openstreetmap.org/api/0.6/map.json'
MAP_PAD_METRES = 400
ROAD_MARGIN_METRES = 20
MAX_RESPONSE_BYTES = 2 * 2**20
MAX_ELEMENTS = 20000
TIMEOUT_SECONDS = 12
ATTRIBUTION = '© OpenStreetMap contributors, https://www.openstreetmap.org/copyright'
OPEN_LAND = {
    'allotments', 'brownfield', 'cemetery', 'commercial', 'construction', 'farmland',
    'farmyard', 'grass', 'industrial', 'meadow', 'orchard', 'quarry', 'railway',
    'recreation_ground', 'reservoir', 'residential', 'retail', 'vineyard',
}
OPEN_NATURAL = {'bare_rock', 'grassland', 'heath', 'sand', 'scree', 'scrub', 'water', 'wetland'}


def map_request(site):
    from pyproj import Transformer

    grid = site['grid']
    require(grid['width'] == grid['height'] == 64 and grid['crs'] == 'EPSG:32636',
            'Expected the frozen V28 64 × 64 UTM grid')
    west, south, east, north = grid['bounds']
    bounds = Transformer.from_crs(grid['crs'], 'EPSG:4326', always_xy=True).transform_bounds(
        west - MAP_PAD_METRES, south - MAP_PAD_METRES, east + MAP_PAD_METRES, north + MAP_PAD_METRES)
    west, south = (floor(coord * 10**6) / 10**6 for coord in bounds[:2])
    east, north = (ceil(coord * 10**6) / 10**6 for coord in bounds[2:])
    require(-180 < west < east < 180 and -90 < south < north < 90
            and east - west < .04 and north - south < .04, 'OSM query outside the bounded crop extent')
    bbox = [west, south, east, north]
    return OSM_MAP + '?' + urlencode({'bbox': ','.join(f'{coord:.6f}' for coord in bbox)}), bbox


def _role(tags):
    if tags.get('highway'):
        return 'road'
    if tags.get('natural') == 'wood' or tags.get('landuse') == 'forest':
        return 'woodland'
    if tags.get('landuse') in OPEN_LAND or tags.get('natural') in OPEN_NATURAL:
        return 'nonwood'
    return None


def _snapshot(raw):
    data = read_json(raw)
    rows = data['elements']
    require(isinstance(rows, list) and len(rows) <= MAX_ELEMENTS, 'Unexpected or oversized OSM elements')
    nodes = {row['id']: row for row in rows if row['type'] == 'node'}
    result = {'ways': [], 'incomplete_way_ids': [], 'uninterpreted_relation_ids': []}
    for row in rows:
        role = _role(row.get('tags', {}))
        if role and row['type'] == 'relation':
            result['uninterpreted_relation_ids'].append(row['id'])
        if not role or row['type'] != 'way':
            continue
        node_ids = row.get('nodes', [])
        if len(node_ids) < 2 or not all(key in nodes for key in node_ids):
            result['incomplete_way_ids'].append(row['id'])
            continue
        coords = [[nodes[key]['lon'], nodes[key]['lat']] for key in node_ids]
        if role != 'road' and (len(coords) < 4 or coords[0] != coords[-1]):
            result['incomplete_way_ids'].append(row['id'])
            continue
        result['ways'].append({
            'id': row['id'], 'role': role, 'coords': coords,
            'mapped_tags': {key: row['tags'][key] for key in ('highway', 'landuse', 'natural') if key in row['tags']},
            'last_edit_utc': max([row.get('timestamp', ''), *(nodes[key].get('timestamp', '') for key in node_ids)]),
        })
    result['ways'].sort(key=lambda row: row['id'])
    for key in ('incomplete_way_ids', 'uninterpreted_relation_ids'):
        result[key].sort()
    return result


def _fetch(url):
    request = Request(url, headers={'User-Agent': 'COMP1252-MP/1.0 bounded research map check',
                                    'Accept': 'application/json'})
    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        destination = urlsplit(response.geturl())
        require(response.status == 200 and destination.scheme == 'https'
                and destination.netloc == 'api.openstreetmap.org', 'Unexpected OSM response')
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        require(0 < len(raw) <= MAX_RESPONSE_BYTES, 'OSM response exceeds checked byte limit')
        return raw


def _shape_snapshot(snapshot, project):
    from shapely import make_valid
    from shapely.geometry import LineString, Polygon
    from shapely.ops import transform

    by_role = {role: [] for role in ('road', 'woodland', 'nonwood')}
    incomplete = list(snapshot['incomplete_way_ids'])
    for way in snapshot['ways']:
        role, coords = way['role'], way['coords']
        raw = LineString(coords) if role == 'road' else Polygon(coords)
        geometry = transform(project, make_valid(raw))
        expected_types = ('LineString', 'MultiLineString') if role == 'road' else ('Polygon', 'MultiPolygon')
        if geometry.geom_type in expected_types and not geometry.is_empty:
            by_role[role].append((way, geometry))
        else:
            incomplete.append(way['id'])
    return by_role, sorted(incomplete)


def _map_disposition(site, paths, snapshot):
    from pyproj import Transformer
    from shapely import make_valid, union_all
    from shapely.geometry import box, shape
    from shapely.ops import transform
    from shapely.strtree import STRtree
    from plan_unseen_validation_sites import load, propose_control, records

    grid = site['grid']
    original = site['control_opportunity']
    project = Transformer.from_crs('EPSG:4326', grid['crs'], always_xy=True).transform
    by_role, incomplete = _shape_snapshot(snapshot, project)
    road, wood, open_land = (by_role[name] for name in ('road', 'woodland', 'nonwood'))
    roads = [geometry for _, geometry in road]
    nonwood = union_all([geometry for _, geometry in open_land])
    woodland = union_all([geometry for _, geometry in wood]).difference(nonwood)

    def describe(control):
        if control is None:
            return None
        context = box(*control['context_bounds'])
        near_road = [way for way, geometry in road if context.distance(geometry) < ROAD_MARGIN_METRES]
        return {
            'mapped_road_crosses_context': any(context.intersects(geometry) for geometry in roads),
            'nearest_mapped_road_metres': round(min(context.distance(g) for g in roads), 3) if roads else None,
            'mapped_roads_within_20m': [{key: row[key] for key in ('id', 'mapped_tags', 'last_edit_utc')}
                                       for row in near_road],
            'mapped_woodland_covers_context': woodland.covers(context),
            'mapped_woodland_boundary_within_20m': (not woodland.is_empty
                                                   and context.distance(woodland.boundary) < ROAD_MARGIN_METRES),
            'mapped_nonwood_within_20m': not nonwood.is_empty and context.distance(nonwood) < ROAD_MARGIN_METRES,
        }

    flags = describe(original)
    alternative = None
    if flags and (flags['mapped_roads_within_20m'] or flags['mapped_nonwood_within_20m']
                  or flags['mapped_woodland_boundary_within_20m']):
        features = load(paths['catalogue'])['features']
        require(len(features) == 3911, 'Frozen annotation inventory differs')
        catalogue = {feature['id']: transform(project, make_valid(shape(feature['geometry'])))
                     for feature in features}
        exclusions = [transform(project, make_valid(shape(row['geometry'])))
                      for row in records(paths['original_source_exclusions']) if row.get('geometry') is not None]
        require(bool(exclusions), 'Frozen unresolved-source footprints are missing')
        frozen = list(catalogue.values()) + exclusions
        anchor = catalogue[site['event_id']]
        require(propose_control(grid, anchor, STRtree(frozen)) == original, 'Frozen V28 control geometry differs')
        obstacles = [*roads, nonwood]
        if not woodland.is_empty:
            # First try the mapped woodland interior. Otherwise keep the same crop,
            # exclude the mapped woodland boundary and leave actual forest class open.
            outside_wood = box(*grid['bounds']).buffer(ROAD_MARGIN_METRES).difference(woodland)
            alternative = propose_control(grid, anchor, STRtree(frozen + obstacles + [outside_wood]))
            obstacles.append(woodland.boundary)
        if alternative is None:
            alternative = propose_control(grid, anchor, STRtree(frozen + obstacles))
        selected = alternative
        disposition = 'alternate_geometry' if alternative else 'no_map_screened_control'
    else:
        selected = original
        disposition = 'original_geometry_retained' if original else 'no_original_control'
    selected_evidence = describe(selected)
    if alternative:
        require(not selected_evidence['mapped_roads_within_20m']
                and not selected_evidence['mapped_nonwood_within_20m']
                and not selected_evidence['mapped_woodland_boundary_within_20m'],
                'Alternate control has a mapped exclusion in its full context margin')
    return {
        'disposition': disposition, 'original_control': original, 'original_map_evidence': flags,
        'alternate_control': alternative, 'selected_control': selected, 'selected_map_evidence': selected_evidence,
        'roads_checked': len(road), 'woodland_ways_checked': len(wood), 'nonwood_ways_checked': len(open_land),
        'incomplete_way_ids': incomplete, 'uninterpreted_relation_ids': snapshot['uninterpreted_relation_ids'],
        'control_class_status': 'unverified',
    }


def check_control_map(site, v28_paths, output, *, archived=None):
    """Fetch once or replay a compact OSM geometry snapshot without network access."""
    site_id = site['site_id']
    require(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', site_id) is not None, 'Unsafe map archive ID')
    url, bbox = map_request(site)
    receipt = {'url': url, 'bbox_wgs84': bbox, 'map_pad_metres': MAP_PAD_METRES,
               'max_response_bytes': MAX_RESPONSE_BYTES, 'timeout_seconds': TIMEOUT_SECONDS}
    relative = Path('raw') / f'{site_id}-osm-map.json'
    if archived is None:
        try:
            raw = _fetch(url)
            snapshot = _snapshot(raw)
            data = canonical(snapshot)
            require(len(data) <= MAX_RESPONSE_BYTES, 'OSM snapshot exceeds checked byte limit')
            receipt.update({
                'status': 'success', 'observed_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'provider_payload_bytes': len(raw), 'provider_payload_sha256': sha(raw),
                'snapshot': relative.as_posix(), 'snapshot_sha256': sha(data),
            })
        except (OSError, ValueError, KeyError, TypeError) as exc:
            receipt.update({'status': 'map_lookup_failed', 'error': type(exc).__name__})
    else:
        prior, prior_directory = archived
        saved = prior['map_receipt']
        require(prior['site_id'] == site_id and prior['event_id'] == site['event_id']
                and all(saved[key] == value for key, value in receipt.items()), 'Map replay request differs')
        receipt = saved
        if receipt['status'] == 'success':
            require(receipt['snapshot'] == relative.as_posix(), 'Map snapshot path differs')
            path = prior_directory / relative
            require(path.stat().st_size <= MAX_RESPONSE_BYTES and digest(path) == receipt['snapshot_sha256'],
                    'Archived map snapshot differs')
            data = path.read_bytes()
            snapshot = read_json(data)
    if receipt['status'] == 'success':
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        disposition = _map_disposition(site, v28_paths, snapshot)
    else:
        disposition = {
            'disposition': 'map_unavailable', 'original_control': site['control_opportunity'],
            'alternate_control': None, 'selected_control': site['control_opportunity'],
            'control_class_status': 'unverified',
        }
    return {
        'site_id': site_id, 'event_id': site['event_id'], 'map_receipt': receipt, **disposition,
        'scope': 'Current OSM vector screening. Incomplete ways and relations remain unassessed. '
                 'Absence of mapped exclusions and edit dates do not establish forest class or its history.',
        'attribution': ATTRIBUTION,
    }
