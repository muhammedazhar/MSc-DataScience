"""Verify one native L1C product and its required QA assets before transfer."""

from datetime import datetime
from http.client import HTTPException
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from annotation_scene_inventory import _grid
from inventory_earlier_validation_sources import PRODUCT, canonical, require, sha, write
from pixelqa_scene_metadata import _angle
from preflight_2019_validation_sources import native_timestamp
from preflight_earlier_validation_metadata import (
    BANDS, BASE, MetadataNetwork, asset_manifest, listing_url, parse_listing, single, texts,
)
from preflight_fresh_validation_sources import PROVIDER_RULES
from preflight_remaining_validation_metadata import BLOCKED, READY


PROVIDER = {**PROVIDER_RULES, 'maximum_requests': 4,
            'maximum_total_response_bytes': 12 * 2**20}
ASSET_COUNT = 58


def catalogue_source(feature):
    """Derive the public archive keys from compact product and STAC identities."""
    require(isinstance(feature, dict) and feature.get('collection') == 'sentinel-2-l1c'
            and isinstance(feature.get('properties'), dict), 'Expected a Sentinel-2 L1C item')
    props = feature['properties']
    uri = props.get('s2:product_uri')
    match = PRODUCT.fullmatch(uri) if isinstance(uri, str) else None
    require(match is not None, 'Missing or invalid compact Sentinel-2 L1C product URI')
    code = match['baseline'][1:]
    require(0 < int(code) < 400,
            'This QA reader requires a legacy Level-1C GML product below baseline 04.00')
    baseline = f'{code[:2]}.{code[2:]}'
    tile = match['tile']
    zone = int(tile[:2])
    require(1 <= zone <= 60 and re.fullmatch(r'\d{2}[C-HJ-NP-X][A-HJ-NP-Z]{2}', tile),
            'Invalid product MGRS tile')
    date = datetime.strptime(match['acquisition'], '%Y%m%dT%H%M%S').date()
    timestamp_fields = ('datetime', 's2:generation_time')
    require(all(isinstance(props.get(field), str) for field in timestamp_fields),
            'Missing catalogue acquisition or generation time')
    generation = native_timestamp(props['s2:generation_time'])
    require(native_timestamp(props['datetime']).date() == date
            and generation.strftime('%Y%m%dT%H%M%S') == match['generation'],
            'Catalogue acquisition date or generation differs from the L1C product')
    sequence = props.get('s2:sequence')
    require(isinstance(sequence, str) and re.fullmatch(r'0|[1-9][0-9]*', sequence),
            'Missing or invalid catalogue sequence')
    item_id = f'{match["satellite"]}_{tile}_{date:%Y%m%d}_{sequence}_L1C'
    require(feature.get('id') == item_id, 'Catalogue item ID disagrees with product tile, date or sequence')
    require(all(isinstance(props.get(field), str) and props[field]
                for field in ('s2:granule_id', 's2:datastrip_id')),
            'Missing catalogue granule or datastrip identifiers')

    scene = uri.removesuffix('.SAFE')
    stamp = f'{date.year}/{date.month}/{date.day}'
    return {'item_id': item_id, 'scene_id': scene, 'acquisition_date': date.isoformat(),
            'tile': tile, 'processing_baseline': baseline, 'relative_orbit': int(match['orbit'][1:]),
            'sequence': sequence, 'tile_prefix': f'tiles/{zone}/{tile[2]}/{tile[3:]}/{stamp}/{sequence}/',
            'product_key': f'products/{stamp}/{scene}/metadata.xml'}


def _image_paths(granule, folder, tile, acquisition):
    pattern = re.compile(rf'GRANULE/{folder}/IMG_DATA/T{tile}_{acquisition}_(B(?:0[1-9]|1[012]|8A)|TCI)')
    image_paths = {}
    for path in texts(granule, 'IMAGE_FILE'):
        match = pattern.fullmatch(path)
        require(match is not None and match[1] not in image_paths, 'Unsupported or repeated granule image path')
        image_paths[match[1]] = path + '.jp2'
    image_paths.pop('TCI', None)
    require(set(BANDS) <= image_paths.keys(), 'Native product XML is missing required image bands')
    return image_paths


def _plot_coverage(plots, geometry, crs, bounds):
    from pyproj import Transformer
    from shapely.geometry import box, shape

    footprint = shape(geometry) if geometry else None
    require(footprint is not None and footprint.geom_type in ('Polygon', 'MultiPolygon')
            and not footprint.is_empty and footprint.is_valid, 'Invalid catalogue footprint')
    project = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    covered = {}
    for plot in plots:
        bbox = plot['bbox']
        require(len(bbox) == 4 and all(type(value) in (int, float) and math.isfinite(value) for value in bbox)
                and -180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90,
                'Expected ordered geographic bounding-box coordinates')
        west, south, east, north = project.transform_bounds(*bbox, errcheck=True)
        covered[plot['code']] = {
            'catalogue_footprint': bool(footprint.covers(box(*bbox))),
            'native_grid': (bounds[0] <= west < east <= bounds[2]
                            and bounds[1] <= south < north <= bounds[3]),
        }
    return covered


def parse_metadata(product_bytes, tile_bytes, source, feature, plots):
    product = ET.fromstring(product_bytes)
    tile = ET.fromstring(tile_bytes)
    require(product.tag.rsplit('}', 1)[-1] == 'Level-1C_User_Product'
            and tile.tag.rsplit('}', 1)[-1] == 'Level-1C_Tile_ID', 'Provider XML must describe L1C')
    scene = source['scene_id']
    mgrs = source['tile']
    baseline = source['processing_baseline']
    identity = {name: single(product, name) for name in ('PRODUCT_URI', 'PROCESSING_LEVEL',
                'PROCESSING_BASELINE', 'SPACECRAFT_NAME', 'SENSING_ORBIT_NUMBER',
                'QUANTIFICATION_VALUE', 'PRODUCT_START_TIME', 'GENERATION_TIME')}
    expected = {'PRODUCT_URI': scene+'.SAFE', 'PROCESSING_LEVEL': 'Level-1C',
                'PROCESSING_BASELINE': baseline, 'SPACECRAFT_NAME': 'Sentinel-2'+scene[2]}
    require(all(identity[field] == value for field, value in expected.items()),
            'Provider product identity or baseline differs from the catalogue')
    orbit = int(identity['SENSING_ORBIT_NUMBER'])
    quantification = float(identity['QUANTIFICATION_VALUE'])
    product_start = native_timestamp(identity['PRODUCT_START_TIME'])
    generation = native_timestamp(identity['GENERATION_TIME'])
    acquisition = scene.split('_')[2]
    require(orbit == source['relative_orbit'] and quantification == 10000
            and product_start.strftime('%Y%m%dT%H%M%S') == acquisition
            and generation.strftime('%Y%m%dT%H%M%S') == scene.split('_')[-1],
            'Product orbit, quantification, start time or generation differs')

    offsets = {}
    for element in product.iter():
        if element.tag.rsplit('}', 1)[-1] == 'RADIO_ADD_OFFSET':
            key = element.attrib.get('band_id', element.attrib.get('bandId'))
            value = float(element.text)
            require(key is not None and key not in offsets and value == 0,
                    'Unsupported, duplicated or nonzero radiometric offset')
            offsets[key] = value
    require(not offsets or set(offsets) == {str(i) for i in range(13)}, 'Incomplete XML offset list')

    granules = [element for element in product.iter() if element.tag.rsplit('}', 1)[-1] == 'Granule']
    require(len(granules) == 1, 'Exactly one product granule is required')
    granule = granules[0]
    tile_id = single(tile, 'TILE_ID')
    datastrip = single(tile, 'DATASTRIP_ID')
    sensed = single(tile, 'SENSING_TIME')
    props = feature['properties']
    native_time = native_timestamp(sensed)
    catalogue_time = native_timestamp(props['datetime'])
    # The native product identifies the acquisition day. Keep any intra-day
    # discrepancy in the STAC time as evidence, as in the 2019 preflight.
    require(granule.attrib.get('granuleIdentifier') == tile_id == props.get('s2:granule_id')
            and granule.attrib.get('datastripIdentifier') == datastrip == props.get('s2:datastrip_id')
            and generation == native_timestamp(props['s2:generation_time'])
            and native_time.date() == catalogue_time.date() == product_start.date()
            and product_start <= native_time < generation,
            'Product, tile, catalogue identity, generation or acquisition day differs')
    suffix = re.escape(baseline)
    absolute_orbit = re.search(rf'_A(\d{{6}})_T{mgrs}_N{suffix}$', tile_id)
    datastrip_stamp = re.search(rf'_S(\d{{8}}T\d{{6}})_N{suffix}$', datastrip)
    require(absolute_orbit is not None and datastrip_stamp is not None
            and datastrip_stamp[1][:8] == acquisition[:8],
            'Tile XML does not identify the native granule and datastrip')
    folder = f'L1C_T{mgrs}_A{absolute_orbit[1]}_{datastrip_stamp[1]}'
    images = _image_paths(granule, folder, mgrs, acquisition)

    issues = []
    crs, grid = _grid(tile, issues)
    utm_prefix = '326' if mgrs[2] >= 'N' else '327'
    require(crs == f'EPSG:{utm_prefix}{mgrs[:2]}' and grid is not None and not issues,
            'Invalid or unsupported native 10 m grid for the product tile')
    covered = _plot_coverage(plots, feature.get('geometry'), crs, grid['bounds'])
    issues.extend(f'Plot {code} falls outside the catalogue footprint or native grid'
                  for code, coverage in covered.items() if not all(coverage.values()))
    angles = [element for element in tile.iter() if element.tag.rsplit('}', 1)[-1] == 'Mean_Sun_Angle']
    require(len(angles) == 1, 'Exactly one mean solar angle group is required')
    zenith = _angle(angles[0], 'ZENITH_ANGLE', issues, maximum=90)
    azimuth = _angle(angles[0], 'AZIMUTH_ANGLE', issues, maximum=360)
    return {'identity': identity, 'actual_relative_orbit': orbit, 'quantification_value': quantification,
            'radio_add_offsets': offsets, 'xml_offset_elements_present': bool(offsets),
            'tile_id': tile_id, 'datastrip_id': datastrip, 'datastrip_timestamp': datastrip_stamp[1],
            'tile_sensing_time': sensed, 'catalogue_datetime': props['datetime'],
            'catalogue_minus_native_seconds': (catalogue_time-native_time).total_seconds(),
            'crs': crs, 'grid_10m': grid, 'plot_coverage': covered, 'solar_zenith_deg': zenith,
            'solar_azimuth_deg': azimuth, 'granule_folder': folder, 'image_paths': images,
            'issues': issues}


def preflight_scene(feature, plots, output, *, opener=None):
    """Save the exact XML and listing receipts and return one scene manifest."""
    from pyproj.exceptions import ProjError
    from shapely.errors import ShapelyError

    source = catalogue_source(feature)
    require(plots and len({plot['code'] for plot in plots}) == len(plots), 'Select distinct plot codes')
    output = Path(output)
    require(not output.is_symlink() and (not output.exists() or (output.is_dir() and not any(output.iterdir()))),
            'Preflight output directory must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    write(output/'catalogue_item.json', feature)
    result = {**source, 'raw_item_sha256': sha(canonical(feature)), 'plots': plots,
              'status': BLOCKED, 'issues': [], 'assets': [], 'measured_payload_bytes': None}
    network = MetadataNetwork(PROVIDER, [source], output, opener=opener)
    try:
        listing = network.call('GET', listing_url(source, PROVIDER), 'raw/listing.xml')
        listed = parse_listing(listing, source['tile_prefix'], PROVIDER['listing_max_keys'])
        tile_key = source['tile_prefix']+'metadata.xml'
        require(tile_key in listed, 'Native tile XML is absent from the exact archive listing')
        product_pin = network.call('HEAD', BASE+source['product_key'], None)
        product = network.call('GET', BASE+source['product_key'], 'raw/product.xml', product_pin)
        tile = network.call('GET', BASE+tile_key, 'raw/tile.xml', listed[tile_key])
        metadata = parse_metadata(product, tile, source, feature, plots)
        assets, issues = asset_manifest(source, metadata, listed, product_pin)
        result.update(metadata=metadata, assets=assets, issues=metadata['issues']+issues)
        if len(assets) == ASSET_COUNT and not result['issues']:
            result.update(status=READY, measured_payload_bytes=sum(asset['size'] for asset in assets))
    except (HTTPException, OSError, ValueError, TypeError, KeyError, ET.ParseError, ProjError, ShapelyError) as error:
        result['issues'].append(f'{type(error).__name__}: {error}')
    write(output/'requests.json', {'provider': network.requests})
    return result
