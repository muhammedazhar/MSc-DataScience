#!/usr/bin/env python3
"""Prepare one blinded validation review packet under the unchanged Mac guard.

Preparation never approves a label and never reads forest-model predictions.
Existing training-only preparation, review and export modules remain unchanged.
"""

import argparse
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BANDS = ('B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12')
BASE_CHANNELS = (*BANDS, 'NDVI', 'NDMI')
CHANNELS = tuple(f'{period}_{band}' for period in ('pre', 'post', 'difference') for band in BASE_CHANNELS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 32*2**20, 'Missing or oversized JSON metadata')
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def records(path):
    from build_curated_pilot import records as bounded_records
    yield from bounded_records(path)


def rooted(relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'A project-relative source path is required')
    path = (ROOT/relative).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.exists(), 'Source path is missing or escapes the project')
    return path


def validate_grid(pair):
    """Explicit validation-role grid check, also usable for derived pairs."""
    from affine import Affine

    require(pair.get('split') == 'validation', 'This path requires an explicit validation pair')
    require(all(isinstance(pair.get(k), str) and pair[k] for k in ('pair_id', 'group_id', 'pre_scene_id', 'post_scene_id')),
            'Frozen validation pair/group/scene identities are required')
    before, after = date.fromisoformat(pair['pre_date']), date.fromisoformat(pair['post_date'])
    require(5 <= (after-before).days <= 30, 'Validation observations must retain a forward 5–30 day interval')
    grid = pair['grid']; transform = Affine(*grid['transform'])
    require((grid['width'], grid['height']) == (64, 64)
            and (transform.a, transform.b, transform.d, transform.e) == (10, 0, 0, -10),
            'Validation features must retain their north-up 64x64, 10m grid')
    require(all(math.isfinite(v) for v in transform[:6])
            and grid['bounds'] == [transform.c, transform.f-640, transform.c+640, transform.f]
            and isinstance(grid['crs'], str) and grid['crs'], 'Validation transform/bounds/CRS disagree')
    return {**grid, 'transform': transform}


def validate_quality_summary(summary):
    """The frozen publication stores errors as a list, not an error count."""
    require(isinstance(summary.get('checks'), dict) and bool(summary['checks'])
            and all(value is True for value in summary['checks'].values())
            and summary.get('processing_errors') == [], 'Frozen QA is incomplete or has errors')


def validate_validation_pair(pair, event_id, catalogue, splits, training_records):
    """Reject cross-split footprints, including every actual training crop."""
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform

    validate_grid(pair)
    require(pair.get('status') == 'candidate_for_pixel_review' and not pair.get('blocking_reasons')
            and event_id in pair.get('anchor_event_ids', []), 'Selection must retain an unblocked frozen validation anchor/pair')
    features = {feature['id']: feature for feature in catalogue['features']}
    require(event_id in features, 'Selected validation anchor is absent from the frozen catalogue')
    properties = features[event_id]['properties']
    require(properties['split'] == pair['split'] == 'validation' and properties['group_id'] == pair['group_id'],
            'Validation anchor group/split differs from its exact frozen pair')
    group = [item for item in splits['groups'] if item['group_id'] == pair['group_id']]
    require(len(group) == 1 and group[0]['split'] == 'validation' and event_id in group[0]['event_ids'],
            'Validation group does not match the frozen split allocation')
    common_crs = splits['crs']

    def footprint(grid):
        polygon = box(*grid['bounds'])
        if grid['crs'] != common_crs:
            polygon = transform(Transformer.from_crs(grid['crs'], common_crs, always_xy=True).transform, polygon)
        require(polygon.is_valid and not polygon.is_empty, 'Feature footprint projection is invalid')
        return polygon

    core = footprint(pair['grid'])
    for feature in catalogue['features']:
        prop = feature['properties']
        if prop['split'] in ('train', 'test'):
            require(not core.intersects(box(*prop['group_footprint_bounds'])),
                    'Validation core intersects a frozen train/test conservative feature footprint')
    require(len(training_records) == 13 and all(row['split'] == 'train' for row in training_records),
            'The current thirteen training crops must be supplied for isolation checks')
    for row in training_records:
        require(row['group_id'] != pair['group_id'] and not core.intersects(footprint(row['grid'])),
                'Validation core overlaps an actual reviewed training crop/group')


def read_validation_features(pair, summary):
    """Return (27-channel float32 features, validity mask, source statistics).

    This function accepts explicit validation-role grids and source scenes.
    The caller separately verifies either original frozen-pair membership or
    predeclared derived-pair lineage; no pair is disguised as training data.
    """
    import numpy as np
    import rasterio
    from assess_pixel_quality import read_fused_band
    from pixelqa_scene_metadata import qa_scene_metadata
    from validate_sample import compute_features
    from curated_pilot_exports import stack_pair

    validate_grid(pair)
    validate_quality_summary(summary)
    metadata, paths = [], set()
    for side in ('pre', 'post'):
        scene = rooted(pair[side+'_path'])
        require(scene.is_dir(), 'Validation scene path must be a SAFE directory')
        item = qa_scene_metadata(scene, ROOT)
        require(item['metadata_usable'] is True and item['scene_id'] == pair[side+'_scene_id']
                and item['acquisition_date'] == pair[side+'_date']
                and item['feature_bands'] == list(BANDS), 'Validation source scene metadata differs from the declared pair')
        metadata.append(item)
        paths.update(item['cloud_band_paths'].values()); paths.update(item['feature_band_paths'].values())
        paths.add(item['quality_paths']['cloud'])
        for kind in ('nodata', 'satura', 'defect', 'tecqua'):
            paths.update(item['quality_paths'][kind])
        paths.update(str(p.relative_to(ROOT)) for p in scene.glob('GRANULE/*/MTD_TL.xml'))
        paths.add(str((scene/'MTD_MSIL1C.xml').relative_to(ROOT)))

    def source_stats():
        output = {}
        for name in sorted(paths):
            path = rooted(name)
            require(path.is_file(), 'A frozen validation source must be a file')
            stat = path.stat(); values = [stat.st_size, stat.st_mtime_ns]
            require(summary['source_file_stats'].get(name) == values,
                    'Validation source statistics differ from completed frozen QA')
            output[name] = values
        return output

    before = source_stats(); dates = []; valid = np.ones((64, 64), dtype=bool)
    with rasterio.Env(GDAL_CACHEMAX=32*2**20, GDAL_NUM_THREADS='1'):
        for item in metadata:
            raw = {}; date_valid = np.ones((64, 64), dtype=bool)
            for band in BANDS:
                _, _, values, band_valid = read_fused_band(rooted(item['feature_band_paths'][band]), pair['grid'], True)
                raw[band] = values; date_valid &= band_valid
            feature = compute_features(raw, item['quantification_value'])
            date_valid &= np.isfinite(feature).all(axis=0)
            dates.append(feature); valid &= date_valid
    require(before == source_stats(), 'Validation sources changed during feature extraction')
    for feature in dates:
        feature[:, ~valid] = np.nan
    features = stack_pair(*dates)
    require(features.shape == (27, 64, 64) and features.dtype == np.dtype('float32')
            and np.isfinite(features[:, valid]).all() and not np.isinf(features).any(),
            'Validation features lost their shape, dtype or reviewed-valid finite values')
    return features, valid, before


def catalogue_exclusion(pair, margin=20):
    """Conservatively repair relevant projected geometries without altering sources."""
    from shapely import make_valid
    from shapely.geometry import box
    from curated_pilot_exports import _projected_catalogue
    from validate_sample import burn

    grid = validate_grid(pair); core = box(*grid['bounds']); geometries = []
    for geometry in _projected_catalogue(grid['crs']):
        bounds = geometry.bounds
        require(len(bounds) == 4 and all(math.isfinite(v) for v in bounds), 'Catalogue projected bounds are invalid')
        if box(*bounds).distance(core) > margin:
            continue
        repaired = make_valid(geometry) if not geometry.is_valid else geometry
        buffered = repaired.buffer(margin)
        geometries.append(make_valid(buffered) if not buffered.is_valid else buffered)
    return burn(geometries, grid, all_touched=True).astype(bool)


def prepare_case(entry, pair, event, scenes, events, reader, summary, qa_row, output):
    import numpy as np
    import rasterio
    from pyproj import Transformer
    from scipy.ndimage import binary_erosion
    from shapely.ops import transform
    from assess_pixel_quality import MASK_NAMES, core_grid, label_target, render_pair
    from prepare_curated_pilot import select_timeline, unusable, propose_control, region_summary, render_timeline
    from validate_sample import burn, write_tif

    case = output/entry['event_id']; case.mkdir(); grid = core_grid(pair['grid'])
    if pair['grid']['crs'] != 'EPSG:32636':
        project = Transformer.from_crs('EPSG:32636', pair['grid']['crs'], always_xy=True).transform
        events = {key: {**value, 'geometry': transform(project, value['geometry'])} for key, value in events.items()}
        event = events[entry['event_id']]
    full_reference = label_target(pair, events).astype(bool)
    anchor = burn([event['geometry']], grid).astype(bool)
    target = anchor & full_reference
    interior = binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
    excluded = catalogue_exclusion(pair)
    pre = reader.observation(pair['pre_path'], pair['grid'])
    post = reader.observation(pair['post_path'], pair['grid'])
    endpoint_unusable = unusable(pre) | unusable(post)
    qa_path = rooted(qa_row['batch_path']+'/'+qa_row['qa_mask_file'])
    with rasterio.open(qa_path) as source:
        require(source.shape == (64, 64) and source.count == 6 and source.dtypes == ('uint8',)*6
                and source.transform == grid['transform']
                and source.crs == rasterio.crs.CRS.from_user_input(grid['crs']), 'Frozen published QA grid/type changed')
        published = source.read()
    require(np.isin(published, (0, 1)).all()
            and np.array_equal(published[:5].astype(bool), np.stack([pre[name] | post[name] for name in MASK_NAMES]))
            and np.array_equal(published[5].astype(bool), full_reference)
            and np.array_equal(endpoint_unusable, np.any(published[:5].astype(bool), axis=0)),
            'Frozen published endpoint QA/temporal reference differs from observation caches')
    control, control_metadata = propose_control(pre, post, excluded)
    features, feature_valid, feature_stats = read_validation_features(pair, summary)
    selected, window = select_timeline(scenes, pair['tile'], event['properties']['observation_date'])
    items, timeline = [], []
    for scene in selected:
        record = reader.observation(scene['path'], pair['grid']) if scene['metadata_usable'] else None
        metadata = {'date': scene['acquisition_date'], 'scene_id': scene['scene_id'], 'scene_path': scene['path'],
                    'metadata_usable': scene['metadata_usable'], 'metadata_issues': scene.get('issues', []),
                    'visual_event_state': 'unreviewed', 'visual_control_state': 'unreviewed',
                    'training_approved': False, 'evaluation_approved': False}
        if record is not None:
            metadata.update(observation=record['metadata'], event_metrics=region_summary(record, target),
                            control_metrics=region_summary(record, control),
                            crop_metrics=region_summary(record, np.ones((64, 64), dtype=bool)))
        else:
            metadata.update(event_metrics=None, control_metrics=None, crop_metrics=None)
        timeline.append(metadata); items.append({'observation': record, 'metadata': metadata})
    require({pair['pre_scene_id'], pair['post_scene_id']}.issubset({row['scene_id'] for row in timeline}),
            'Primary endpoints are absent from the complete declared timeline')
    render_pair(case/'pair.png', pre, post, target,
                'validation | '+entry['event_id']+' | '+pair['pre_date']+' -> '+pair['post_date'])
    render_timeline(case/'timeline.png', items, target, control,
                    'Validation review | '+entry['event_id']+' | '+entry.get('stratum', pair['tile']))
    arrays = {'target_mask.tif': target.astype('uint8'), 'positive_interior.tif': interior.astype('uint8'),
              'proposed_control.tif': control.astype('uint8'), 'catalogue_exclusion.tif': excluded.astype('uint8'),
              'endpoint_unusable.tif': endpoint_unusable.astype('uint8'), 'feature_valid.tif': feature_valid.astype('uint8'),
              'input_features.tif': features}
    for name, values in arrays.items():
        write_tif(case/name, values, grid, float('nan') if name == 'input_features.tif' else None,
                  CHANNELS if name == 'input_features.tif' else (name.removesuffix('.tif'),))
        with rasterio.open(case/name) as source:
            require(source.shape == (64, 64) and source.transform == grid['transform']
                    and source.crs == rasterio.crs.CRS.from_user_input(grid['crs'])
                    and np.array_equal(source.read(), values if values.ndim == 3 else values[None], equal_nan=True),
                    'Prepared validation artifact round-trip changed arrays/grid')
    artifacts = {path.name: digest(path) for path in case.iterdir() if path.is_file()}
    metadata = {'schema_version': 1, 'role': 'validation_review', 'event_id': entry['event_id'],
                'pair_id': pair['pair_id'], 'group_id': pair['group_id'], 'split': 'validation',
                'stratum': entry.get('stratum'), 'tile': pair['tile'], 'selection': entry, 'pair': pair,
                'annotation_observation_date': event['properties']['observation_date'], 'config_id': reader.config_id,
                'target_pixels': int(target.sum()), 'positive_interior_pixels': int(interior.sum()),
                'interior_rule': 'One-pixel 3x3 binary erosion of anchor-specific new-loss reference',
                'empty_interior_action': 'Explicit rejection or separately reviewed manual ROI; no automatic boundary fallback' if not interior.any() else None,
                'control_proposal': control_metadata, 'catalogue_exclusion_rule': 'Every cleaned catalogue geometry at every date/provenance, buffered20m and burned all-touched',
                'timeline_window': window, 'timeline': timeline,
                'primary_endpoint_cache_paths': {'pre': pre['metadata']['cache_path'], 'post': post['metadata']['cache_path']},
                'published_qa': {'index_row': qa_row, 'qa_file': str(qa_path.relative_to(ROOT)), 'qa_file_sha256': digest(qa_path)},
                'feature_source_file_stats': feature_stats, 'feature_channels': list(CHANNELS),
                'feature_valid_pixels': int(feature_valid.sum()), 'feature_nonfinite_values': int((~np.isfinite(features)).sum()),
                'features_nonfinite_policy': 'Only invalid context may be NaN; final reviewed ROIs must be fully finite before approval; no labels are approved here',
                'visual_review': 'unreviewed', 'temporal_label_review': 'unreviewed', 'control_label_review': 'unreviewed',
                'training_approved': False, 'evaluation_approved': False, 'model_predictions_read': False,
                'artifact_sha256': artifacts,
                'limitations': ['Preparation and numeric proposals do not approve labels.',
                                'Selected validation examples are a small reviewed-region development diagnostic, not a representative benchmark.',
                                'Timeline bounds are local observations, not exact felling dates.']}
    write_json(case/'case.json', metadata)
    return {'event_id': entry['event_id'], 'pair_id': pair['pair_id'], 'case_directory': case.name,
            'timeline_scene_count': len(timeline), 'target_pixels': int(target.sum()),
            'positive_interior_pixels': int(interior.sum()), 'control_proposal_pixels': int(control.sum()),
            'training_approved': False, 'evaluation_approved': False}


def prepare(args, output):
    import rasterio
    from assess_pixel_quality import ObservationReader, load_events, model_identity, validate_source_stats

    raw = read_json(args.selection)
    require(isinstance(raw, dict), 'Validation selection must be a JSON object')
    entries = raw.get('candidates')
    require(isinstance(entries, list) and 1 <= len(entries) <= 6 and all(isinstance(e, dict) for e in entries),
            'Selection requires one to six explicit validation candidates')
    require(len({e['pair_id'] for e in entries}) == len(entries)
            and len({e['event_id'] for e in entries}) == len(entries), 'Selected validation pair/event identities repeat')
    wanted = {e['pair_id'] for e in entries}
    pairs = {row['pair_id']: row for row in records(args.audit/'candidate_pairs.jsonl') if row['pair_id'] in wanted}
    require(set(pairs) == wanted, 'Selection contains an unknown frozen validation pair')
    catalogue = read_json(args.audit/'annotations_clean.geojson'); splits = read_json(args.audit/'splits.json')
    training_plan = read_json(ROOT/'Docs/Reviews/PixelPilotV1/experiment_plan.json')
    require(digest(args.training_dataset/'metadata.json') == training_plan['selection']['dataset_metadata_sha256']
            and digest(args.training_dataset/'training_manifest.jsonl') == training_plan['selection']['manifest_sha256'],
            'Isolation checks require the exact frozen dataset used by the completed model')
    training_records = list(records(args.training_dataset/'training_manifest.jsonl'))
    groups = set()
    for entry in entries:
        pair = pairs[entry['pair_id']]
        validate_validation_pair(pair, entry['event_id'], catalogue, splits, training_records)
        require(pair['group_id'] not in groups, 'Selection reuses a validation geographic group')
        groups.add(pair['group_id'])
    require(args.batch_index < len(entries), 'Selected validation batch is empty')
    summary = read_json(args.quality_root/'summary/metadata.json')
    validate_quality_summary(summary)
    validate_source_stats(summary)
    reader = ObservationReader(args.cache_root, model_identity())
    require(reader.config_id == summary['config_id'], 'Frozen QA policy changed')
    quality = {row['pair_id']: row for row in records(args.quality_root/'review/quality_result_index.jsonl') if row['pair_id'] in wanted}
    require(set(quality) == wanted, 'Validation pair is absent from frozen published QA')
    for identifier, row in quality.items():
        pair = pairs[identifier]
        require(all(row[key] == pair[key] for key in ('pair_id', 'group_id', 'split', 'tile', 'pre_date', 'post_date', 'anchor_event_ids'))
                and row['config_id'] == reader.config_id, 'Published validation QA identity differs from frozen candidate')
    source_paths = [args.selection, args.audit/'metadata.json', args.audit/'candidate_pairs.jsonl',
                    args.audit/'annotations_clean.geojson', args.audit/'splits.json', args.audit/'scene_inventory.jsonl',
                    args.training_dataset/'metadata.json', args.training_dataset/'training_manifest.jsonl',
                    args.quality_root/'summary/metadata.json', args.quality_root/'review/quality_result_index.jsonl']
    hashes = {str(path.resolve().relative_to(ROOT)): digest(path) for path in source_paths}
    implementation = digest(__file__)
    entry = entries[args.batch_index]; pair = pairs[entry['pair_id']]
    events = load_events(args.audit); scenes = list(records(args.audit/'scene_inventory.jsonl'))
    with rasterio.Env(GDAL_CACHEMAX=32*2**20, GDAL_NUM_THREADS='1'):
        result = prepare_case(entry, pair, events[entry['event_id']], scenes, events, reader, summary, quality[pair['pair_id']], output)
    validate_source_stats(summary)
    stats = {key: value for scene in reader.scenes.values() for key, value in scene['source_file_stats'].items()}
    validate_source_stats({'source_file_stats': stats})
    require(all(digest(ROOT/path) == checksum for path, checksum in hashes.items()) and digest(__file__) == implementation,
            'Validation preparation sources changed during execution')
    write_json(output/'metadata.json', {'schema_version': 1, 'mode': 'prepare_validation_review',
        'batch_index': args.batch_index, 'batch_size': 1, 'selection_count': len(entries), 'case_count': 1,
        'selection_sha256': digest(args.selection), 'audit_metadata_sha256': digest(args.audit/'metadata.json'),
        'candidate_manifest_sha256': digest(args.audit/'candidate_pairs.jsonl'),
        'annotation_catalogue_sha256': digest(args.audit/'annotations_clean.geojson'),
        'split_plan_sha256': digest(args.audit/'splits.json'), 'quality_summary_sha256': digest(args.quality_root/'summary/metadata.json'),
        'implementation_sha256': implementation, 'config_id': reader.config_id, 'source_identity_sha256': hashes,
        'new_observations': reader.misses, 'cache_hits': reader.hits, 'cases': [result], 'source_file_stats': stats,
        'training_approved': False, 'evaluation_approved': False, 'model_predictions_read': False,
        'checks': {'explicit_validation_only': True, 'exact_frozen_pair_join': True,
                   'no_train_test_conservative_footprint_overlap': True, 'no_actual_training_crop_overlap': True,
                   'complete_local_timeline': True, 'feature_grid_round_trip': True, 'frozen_endpoint_qa_exact': True}})
    print(f"[validation-review] prepared 1 case; new observations={reader.misses}; cache hits={reader.hits}", flush=True)


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--audit', type=Path, default=ROOT/'Datasets/AnnotationAudit/v1')
    parser.add_argument('--quality-root', type=Path, default=ROOT/'Datasets/PixelQA/v1')
    parser.add_argument('--cache-root', type=Path, default=ROOT/'Datasets/PixelQA/cache')
    parser.add_argument('--training-dataset', type=Path, default=ROOT/'Datasets/CuratedPilot/v1/dataset')
    parser.add_argument('--batch-index', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (args.batch_index >= 0 and args.batch_size == 1 and 128 <= args.max_rss_mib <= 512
            and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Use batch size 1, nonnegative index, 128–512 MiB RSS, >=1.5 GiB reserve and <=300s')
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        os.nice(10); prepare(args, args.worker_output)
        import resource
        metadata = read_json(args.worker_output/'metadata.json')
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata['worker_high_water_rss_mib'] = round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2)
        write_json(args.worker_output/'metadata.json', metadata)
    else:
        run_guarded(args, script_path=__file__, operation='validation-review', gdal_cache_mib=32, warp_memory_mib=16)


if __name__ == '__main__':
    main()
