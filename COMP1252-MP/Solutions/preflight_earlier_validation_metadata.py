#!/usr/bin/env python3
"""Check two fixed sources using listings/XML only; never construct a SAFE."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, build_opener
import xml.etree.ElementTree as ET

from inventory_earlier_validation_sources import (ROOT, RESOURCES, NoRedirect, canonical,
                                                digest, read_json, require, rooted, sha, timestamp, write)

BASE = 'https://sentinel-s2-l1c.s3.amazonaws.com/'
BANDS = ('B01', 'B02', 'B03', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12')
KINDS = ('NODATA', 'SATURA', 'DEFECT', 'TECQUA')
DATES = ('2018-07-21', '2018-07-16')


def texts(root, name):
    return [(element.text or '').strip() for element in root.iter()
            if element.tag.rsplit('}', 1)[-1] == name]


def single(root, name):
    values = texts(root, name)
    require(len(values) == 1 and values[0], 'Missing/ambiguous XML '+name)
    return values[0]


def pin(size, etag):
    require(type(size) is int and size > 0 and isinstance(etag, str)
            and re.fullmatch(r'"[0-9a-f]{32}"', etag), 'Positive size and simple MD5 ETag required')
    return {'size': size, 'etag': etag}


def listing_url(roster, provider):
    return BASE+'?'+urlencode({'list-type': '2', 'prefix': roster['tile_prefix'],
                               'max-keys': provider['listing_max_keys']})


def parse_listing(raw, prefix, maximum_keys):
    root = ET.fromstring(raw)
    require(root.tag.rsplit('}', 1)[-1] == 'ListBucketResult'
            and single(root, 'IsTruncated') == 'false'
            and single(root, 'Prefix') == prefix, 'Incomplete/mismatched S3 listing')
    contents = [element for element in root if element.tag.rsplit('}', 1)[-1] == 'Contents']
    require(len(contents) <= maximum_keys, 'Listing key bound exceeded')
    count = texts(root, 'KeyCount')
    require(not count or len(count) == 1 and int(count[0]) == len(contents), 'Listing count mismatch')
    result = {}
    for entry in contents:
        key = single(entry, 'Key')
        require(key.startswith(prefix) and key not in result, 'Escaping or duplicate listing key')
        # Unused provider objects may have multipart ETags/zero bytes; required assets are stricter.
        result[key] = {'key': key, 'size': int(single(entry, 'Size')), 'etag': single(entry, 'ETag'),
                       'url': BASE+quote(key, safe='/')}
    return result


def inputs(protocol):
    require(protocol['schema_version'] == 13 and protocol['resources'] == RESOURCES, 'V13/resource scope differs')
    provider = protocol['provider']
    expected = {'base_url': BASE, 'allow_automatic_redirects': False, 'listing_max_keys': 300,
                'maximum_listing_bytes': 2097152, 'require_listing_not_truncated': True,
                'maximum_xml_bytes': 5242880, 'maximum_total_response_bytes': 33554432,
                'maximum_requests': 12, 'timeout_seconds': 20,
                'require_xml_size_etag_content_length_md5_if_match': True,
                'anonymous_only_no_paid_access': True}
    require(all(provider.get(k) == v for k, v in expected.items()), 'Network limits differ')
    compatibility = protocol['compatibility']
    expected_compatibility = {'source_crs': 'EPSG:32636', 'tile': '36UXB', 'relative_orbit': 64,
        'processing_level': 'Level-1C', 'processing_baseline': '02.06', 'quantification_value': 10000,
        'require_absent_or_zero_xml_radio_add_offsets': True, 'required_jp2_bands': list(BANDS),
        'required_quality_types': list(KINDS), 'required_assets_per_source': 58,
        'source_qa_content_verified': False, 'local_qa_readiness_claimed': False}
    require(compatibility == expected_compatibility, 'Frozen compatibility scope differs')
    require(protocol['disk_budget']['minimum_free_reserve_bytes'] == 2*2**30
            and protocol['disk_budget']['allow_imagery_acquisition'] is False, 'Disk/acquisition scope differs')
    receipts = dict(protocol['source_files'])
    for name, checksum in {**protocol['frozen_implementation_sha256'], **protocol.get('implementation_sha256', {})}.items():
        receipts['code/'+name] = {'path': name, 'sha256': checksum}
    require('Solutions/preflight_earlier_validation_metadata.py' in protocol.get('implementation_sha256', {}),
            'Freeze tested implementation before operational requests')
    for receipt in receipts.values():
        require(digest(rooted(receipt['path'])) == receipt['sha256'], 'Frozen input changed: '+receipt['path'])
    inventory = read_json(rooted(receipts['catalogue_inventory']['path']).read_bytes())
    raw = read_json(rooted(receipts['catalogue_raw_page']['path']).read_bytes())
    raw_items = {f['id']: f for f in raw['features']}
    nominal = sorted([r for r in inventory['items'] if r['nominal_metadata_followup_eligible']],
                     key=lambda r: (r['acquisition_date'], r['item_id']), reverse=True)[:2]
    roster = protocol['selection']['roster']
    require(protocol['selection']['count'] == 2 and protocol['selection']['post_date'] == '2018-07-31'
            and tuple(r['acquisition_date'] for r in nominal) == DATES and len(roster) == 2,
            'Fixed nearest-two roster differs')
    for selected, row in zip(roster, nominal):
        feature = raw_items[row['item_id']]; day = row['acquisition_date']; y, m, d = map(int, day.split('-'))
        sequence = feature['properties']['s2:sequence']
        require(isinstance(sequence, str) and re.fullmatch(r'\d+', sequence), 'Invalid source sequence')
        scene = row['product_uri'].removesuffix('.SAFE')
        expected = {'item_id': row['item_id'], 'raw_item_sha256': sha(canonical(feature)), 'scene_id': scene,
                    'acquisition_date': day, 'sequence': sequence,
                    'tile_prefix': f'tiles/36/U/XB/{y}/{m}/{d}/{sequence}/',
                    'product_key': f'products/{y}/{m}/{d}/{scene}/metadata.xml'}
        require(selected == expected and row['raw_item_sha256'] == expected['raw_item_sha256'], 'Roster/source join differs')
        require(not (ROOT/'Datasets/Sentinel-2'/scene).exists(), 'Earlier source already exists; do not duplicate acquisition')
    requirements = read_json(rooted(receipts['requirements']['path']).read_bytes())
    return roster, raw_items, requirements, receipts


class MetadataNetwork:
    """An exact allowlist makes image/GML content requests impossible here."""
    def __init__(self, provider, roster, output, opener=None):
        self.provider = provider; self.output = output; self.requests = []; self.bytes = 0
        self.opener = opener or build_opener(NoRedirect()).open
        self.allowed = set()
        for row in roster:
            self.allowed.add(('GET', listing_url(row, provider)))
            self.allowed.add(('HEAD', BASE+row['product_key']))
            self.allowed.add(('GET', BASE+row['product_key']))
            self.allowed.add(('GET', BASE+row['tile_prefix']+'metadata.xml'))

    def call(self, method, url, name, asset=None):
        require((method, url) in self.allowed, 'Request outside metadata-only allowlist')
        require(len(self.requests) < self.provider['maximum_requests'], 'Request count bound exceeded')
        listing = method == 'GET' and url.startswith(BASE+'?')
        cap = self.provider['maximum_listing_bytes'] if listing else self.provider['maximum_xml_bytes']
        headers = {'Accept-Encoding': 'identity', 'User-Agent': 'COMP1252-source-metadata/13'}
        if method == 'GET' and not listing:
            pin(asset['size'], asset['etag']); require(asset['size'] <= cap, 'XML exceeds bound')
            headers['If-Match'] = asset['etag']
        entry = {'method': method, 'url': url, 'request_headers': headers, 'artifact': None}
        self.requests.append(entry)
        try:
            with self.opener(Request(url, method=method, headers=headers), timeout=self.provider['timeout_seconds']) as response:
                require(response.geturl() == url and response.status == 200, 'Redirect/non-200 metadata response')
                entry.update(status=response.status, response_headers=dict(response.headers))
                if method == 'HEAD':
                    result = pin(int(response.headers.get('Content-Length', '-1')), response.headers.get('ETag'))
                    require(result['size'] <= cap, 'Product XML exceeds bound')
                    entry.update(pin=result, verified=True)
                    return result
                raw = response.read(cap+1); self.bytes += len(raw)
                require(len(raw) <= cap and self.bytes <= self.provider['maximum_total_response_bytes'], 'Metadata byte cap exceeded')
                target = (self.output/name).resolve()
                require(target.is_relative_to(self.output.resolve()) and not target.exists(), 'Metadata artifact path exists/escapes')
                target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
                entry.update(artifact=name, bytes=len(raw), sha256=sha(raw))
                if not listing:
                    require(response.headers.get('ETag') == asset['etag']
                            and int(response.headers.get('Content-Length', '-1')) == asset['size']
                            and len(raw) == asset['size']
                            and hashlib.md5(raw).hexdigest() == asset['etag'][1:-1], 'XML source size/ETag/MD5 mismatch')
                    entry.update(md5=hashlib.md5(raw).hexdigest(), if_match_verified=True, content_length_verified=True)
                entry['verified'] = True
                return raw
        except (HTTPError, URLError, ValueError, OSError) as error:
            entry.update(error_type=type(error).__name__, error=str(error), status=getattr(error, 'code', entry.get('status')))
            raise


def parse_metadata(product_bytes, tile_bytes, roster, feature, requirements):
    from annotation_scene_inventory import _grid
    from pixelqa_scene_metadata import _angle

    product = ET.fromstring(product_bytes); tile = ET.fromstring(tile_bytes)
    issues = []
    if (product.tag.rsplit('}', 1)[-1] != 'Level-1C_User_Product'
            or tile.tag.rsplit('}', 1)[-1] != 'Level-1C_Tile_ID'):
        issues.append('non_l1c_xml_root')
    identity = {name: single(product, name) for name in ('PRODUCT_URI', 'PROCESSING_LEVEL', 'PROCESSING_BASELINE',
                 'SPACECRAFT_NAME', 'SENSING_ORBIT_NUMBER', 'QUANTIFICATION_VALUE', 'PRODUCT_START_TIME')}
    expected = {'PRODUCT_URI': roster['scene_id']+'.SAFE', 'PROCESSING_LEVEL': 'Level-1C',
                'PROCESSING_BASELINE': '02.06', 'SPACECRAFT_NAME': 'Sentinel-2'+roster['scene_id'][2]}
    for name, wanted in expected.items():
        if identity[name] != wanted:
            issues.append('product_identity_conflict:'+name)
    orbit = int(identity['SENSING_ORBIT_NUMBER']); quantification = float(identity['QUANTIFICATION_VALUE'])
    require(math.isfinite(quantification), 'Nonfinite quantification')
    if orbit != 64 or quantification != 10000:
        issues.append('orbit_or_quantification_incompatible')
    if timestamp(identity['PRODUCT_START_TIME']).date().isoformat() != roster['acquisition_date']:
        issues.append('product_start_date_conflict')
    offsets = {}
    for element in product.iter():
        if element.tag.rsplit('}', 1)[-1] == 'RADIO_ADD_OFFSET':
            key = element.attrib.get('band_id', element.attrib.get('bandId'))
            value = float(element.text)
            if key is None or key in offsets or not math.isfinite(value) or value != 0:
                issues.append('invalid_duplicate_or_nonzero_xml_offset')
            offsets[key] = value if math.isfinite(value) else None
    if offsets and set(offsets) != {str(i) for i in range(13)}:
        issues.append('incomplete_xml_offset_list')
    granules = [e for e in product.iter() if e.tag.rsplit('}', 1)[-1] == 'Granule']
    require(len(granules) == 1, 'Exactly one product granule required')
    granule = granules[0]; tile_id = single(tile, 'TILE_ID'); datastrip = single(tile, 'DATASTRIP_ID')
    if (granule.attrib.get('granuleIdentifier') != tile_id or tile_id != feature['properties'].get('s2:granule_id')
            or granule.attrib.get('datastripIdentifier') != datastrip
            or datastrip != feature['properties'].get('s2:datastrip_id')):
        issues.append('product_tile_catalogue_granule_or_datastrip_conflict')
    sensed = single(tile, 'SENSING_TIME')
    if (timestamp(sensed).date().isoformat() != roster['acquisition_date']
            or timestamp(sensed) != timestamp(feature['properties']['datetime'])):
        issues.append('tile_catalogue_sensing_time_conflict')
    crs, grid = _grid(tile, issues)
    if crs != requirements['crs']:
        issues.append('actual_grid_crs_conflict')
    covered = {}
    for event in requirements['events']:
        covered[event['event_id']] = {}
        for name, bounds in (('core', event['crop_grid']['bounds']), ('qa_context', event['required_qa_context_bounds'])):
            outer = grid['bounds'] if grid else None
            ok = bool(outer and outer[0] <= bounds[0] < bounds[2] <= outer[2]
                      and outer[1] <= bounds[1] < bounds[3] <= outer[3] and crs == requirements['crs'])
            covered[event['event_id']][name] = ok
            if not ok:
                issues.append('actual_grid_missing_full_'+name)
        if grid:
            for actual, required in zip(grid['transform'][2:6:3], event['crop_grid']['transform'][2:6:3]):
                if abs((actual-required)/10-round((actual-required)/10)) > 1e-6:
                    issues.append('actual_grid_pixel_alignment_conflict')
    groups = [e for e in tile.iter() if e.tag.rsplit('}', 1)[-1] == 'Mean_Sun_Angle']
    require(len(groups) == 1, 'Exactly one mean solar angle group required')
    zenith = _angle(groups[0], 'ZENITH_ANGLE', issues, maximum=90)
    azimuth = _angle(groups[0], 'AZIMUTH_ANGLE', issues, maximum=360)
    image_paths = {}; folders = set()
    pattern = re.compile(r'GRANULE/(L1C_T36UXB_A\d{6}_\d{8}T\d{6})/IMG_DATA/(T36UXB_\d{8}T\d{6})_(B(?:0[1-9]|1[012]|8A))')
    for path in texts(granule, 'IMAGE_FILE'):
        if path.endswith('_TCI'):
            continue
        match = pattern.fullmatch(path)
        require(match is not None, 'Unsupported/escaping product IMAGE_FILE')
        folder, _, band = match.groups()
        require(band not in image_paths, 'Duplicate product IMAGE_FILE band')
        image_paths[band] = path+'.jp2'; folders.add(folder)
        if match[2] != 'T36UXB_'+roster['scene_id'].split('_')[2]:
            issues.append('image_file_acquisition_name_conflict')
    require(len(folders) == 1 and all(b in image_paths for b in BANDS), 'Incomplete/ambiguous product image manifest')
    folder = next(iter(folders))
    if datetime.strptime(folder[-15:], '%Y%m%dT%H%M%S').date().isoformat() != roster['acquisition_date']:
        issues.append('granule_folder_date_conflict')
    absolute = re.search(r'_A(\d{6})_T36UXB_', tile_id)
    if (absolute is None or '_A'+absolute[1]+'_' not in folder
            or folder[-15:] != timestamp(sensed).strftime('%Y%m%dT%H%M%S')):
        issues.append('granule_folder_tile_identity_conflict')
    return {'identity': identity, 'actual_relative_orbit': orbit, 'quantification_value': quantification,
            'radio_add_offsets': offsets, 'xml_offset_elements_present': bool(offsets),
            'offset_interpretation': ('Absent list in actual N0206 XML; frozen-reader zero-offset convention'
                if not offsets and identity['PROCESSING_BASELINE'] == '02.06'
                else 'Explicit XML offset evidence; see issues, no source approval'),
            'tile_id': tile_id, 'datastrip_id': datastrip, 'tile_sensing_time': sensed,
            'crs': crs, 'grid_10m': grid, 'fixed_event_coverage': covered,
            'solar_zenith_deg': zenith, 'solar_azimuth_deg': azimuth,
            'granule_folder': folder, 'image_paths': image_paths, 'issues': sorted(set(issues))}


def asset_manifest(roster, metadata, listed, product_pin):
    safe = roster['scene_id']+'.SAFE'; folder = metadata['granule_folder']; prefix = roster['tile_prefix']
    required = [(roster['product_key'], 'product_xml', safe+'/MTD_MSIL1C.xml'),
                (prefix+'metadata.xml', 'tile_xml', safe+'/GRANULE/'+folder+'/MTD_TL.xml'),
                (prefix+'qi/MSK_CLOUDS_B00.gml', 'cloud_gml', safe+'/GRANULE/'+folder+'/QI_DATA/MSK_CLOUDS_B00.gml')]
    for band in BANDS:
        required.append((prefix+band+'.jp2', 'jp2', safe+'/'+metadata['image_paths'][band]))
        for kind in KINDS:
            filename=f'MSK_{kind}_{band}.gml'
            required.append((prefix+'qi/'+filename, 'quality_gml', safe+'/GRANULE/'+folder+'/QI_DATA/'+filename))
    assets = []; issues = []
    for key, role, target in required:
        source = {'key': key, 'url': BASE+key, **product_pin} if role == 'product_xml' else listed.get(key)
        if source is None:
            issues.append('missing_required_object:'+key); continue
        try:
            pin(source['size'], source['etag'])
            if role != 'jp2':
                require(source['size'] <= 5*2**20, 'Metadata/GML exceeds frozen 5 MiB bound')
        except ValueError as error:
            issues.append('unsupported_required_object:'+key+':'+str(error)); continue
        assets.append({**source, 'role': role, 'target_relpath': target,
                       'availability_evidence': 'HEAD_and_verified_XML_GET' if role == 'product_xml' else 'complete_provider_listing',
                       'payload_integrity_verified': role in ('product_xml', 'tile_xml')})
    require(len(required) == 58 and len({r[2] for r in required}) == 58, 'Required manifest cardinality/path conflict')
    return assets, issues


def disk_budget(scenes, free_bytes, metadata_bytes):
    complete = all(len(r.get('assets', [])) == 58 for r in scenes)
    assets = [a for r in scenes for a in r.get('assets', [])]
    payload = sum(a['size'] for a in assets) if complete else None
    largest = max((a['size'] for a in assets), default=0)
    peak = 2*payload+largest+metadata_bytes if complete else None
    return {'measured_free_bytes': free_bytes, 'complete_payload_bytes': payload, 'largest_asset_bytes': largest,
            'current_metadata_artifact_bytes': metadata_bytes, 'future_two_copy_staging_plus_temporary_bytes': peak,
            'minimum_free_reserve_bytes': 2*2**30,
            'sufficient_for_conservative_future_acquisition': peak is not None and free_bytes >= peak+2*2**30,
            'imagery_acquisition_executed': False}


def worker(args, output):
    import resource

    protocol = read_json(args.protocol.read_bytes()); roster, raw_items, requirements, receipts = inputs(protocol)
    disk_path = ROOT/'Datasets/Sentinel-2'
    free_before = shutil.disk_usage(disk_path).free
    network = MetadataNetwork(protocol['provider'], roster, output); results = []
    for index, selected in enumerate(roster, 1):
        feature = raw_items[selected['item_id']]
        row = {'scene_id': selected['scene_id'], 'acquisition_date': selected['acquisition_date'],
               'item_id': selected['item_id'], 'raw_item_sha256': selected['raw_item_sha256'],
               'catalogue_asset_href_conflicts': {name: asset['href'] for name, asset in feature['assets'].items()
                   if 'sentinel-s2-l2a' in asset.get('href', '')},
               'status': 'blocked_metadata_preflight', 'issues': [], 'assets': [],
               'local_qa_ready': False, 'image_content_downloaded': False, 'source_approved': False}
        try:
            prefix = f'source-{index:02d}/'
            listed = parse_listing(network.call('GET', listing_url(selected, protocol['provider']), prefix+'listing.xml'),
                                   selected['tile_prefix'], protocol['provider']['listing_max_keys'])
            product_pin = network.call('HEAD', BASE+selected['product_key'], None)
            tile_pin = listed.get(selected['tile_prefix']+'metadata.xml')
            require(tile_pin is not None, 'Tile XML missing from complete listing')
            product_bytes = network.call('GET', BASE+selected['product_key'], prefix+'product.xml', product_pin)
            tile_bytes = network.call('GET', BASE+tile_pin['key'], prefix+'tile.xml', tile_pin)
            metadata = parse_metadata(product_bytes, tile_bytes, selected, feature, requirements)
            assets, asset_issues = asset_manifest(selected, metadata, listed, product_pin)
            row.update(metadata=metadata, listed_objects=len(listed), assets=assets,
                       issues=metadata['issues']+asset_issues, required_assets=58,
                       measured_payload_bytes=sum(a['size'] for a in assets) if len(assets) == 58 else None)
            if not row['issues']:
                row['status'] = 'remote_metadata_and_asset_listing_passed'
        except (HTTPError, URLError, OSError, ValueError, TypeError, KeyError, ET.ParseError) as error:
            row['issues'].append(type(error).__name__+':'+str(error))
        results.append(row)
    write(output/'requests.json', network.requests)
    metadata_bytes = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    budget = disk_budget(results, shutil.disk_usage(disk_path).free, metadata_bytes)
    budget.update(measured_filesystem_path=str(disk_path.relative_to(ROOT)),
                  measured_filesystem_device=disk_path.stat().st_dev)
    passed = all(r['status'] == 'remote_metadata_and_asset_listing_passed' for r in results)
    result = {'schema_version': 13, 'protocol_sha256': digest(args.protocol),
              'status': 'metadata_preflight_passed_no_acquisition' if passed and budget['sufficient_for_conservative_future_acquisition']
                        else 'metadata_preflight_blocked_no_acquisition',
              'group_id': requirements['group_id'], 'results': results, 'disk_budget': budget,
              'disk_free_before_bytes': free_before, 'metadata_network_bytes': network.bytes,
              'metadata_request_count': len(network.requests), 'image_asset_downloads': 0,
              'gml_content_downloads': 0, 'satellite_pixel_reads': 0, 'qa_or_model_executions': 0,
              'labels_approved': 0, 'imagery_acquisition_roster_selected': False,
              'limitations': ['Listings pin presence/size/ETag; unread JP2/GML payloads are not integrity-verified or HEAD-tested.',
                  'Remote XML/asset evidence is not local frozen QA readiness, pixel clarity or source approval.',
                  'Catalogue L2A asset links were not fetched; explicit L1C provider XML was checked.',
                  'Fixed earlier source dates need new pair lineage and cumulative temporal targets before QA.']}
    write(output/'preflight.json', result); inputs(protocol)
    report = '# Earlier source metadata preflight\n\n'+result['status']+'\n\n'
    for row in results:
        report += '- '+row['acquisition_date']+': '+row['status']+'; '+str(len(row['assets']))+'/58 pinned assets; '+str(row['issues'])+'\n'
    report += '\n'+json.dumps(budget, indent=2)+'\n\n'+'\n'.join('- '+v for v in result['limitations'])+'\n'
    (output/'report.md').write_text(report)
    require(not any(name in sys.modules for name in ('torch', 'rasterio', 's2cloudless')), 'Pixel/model library imported')
    write(output/'metadata.json', {'execution_status': 'completed', 'mode': 'earlier_source_metadata_preflight',
          'protocol_sha256': digest(args.protocol), 'input_receipts': receipts,
          'artifact_sha256': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
          'worker_high_water_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                           (2**20 if sys.platform == 'darwin' else 1024), 2),
          'native_threads': 1, 'gpu_used': False, 'source_scene_publications': 0,
          'source_image_downloads': 0, 'model_executions': 0, 'labels_approved': 0})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationEarlierMetadataV13/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Retain 512 MiB/1.5 GiB/300 s capacity bounds')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10); worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='earlier-source-preflight', gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
