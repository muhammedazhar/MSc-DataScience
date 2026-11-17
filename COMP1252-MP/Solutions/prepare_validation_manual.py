#!/usr/bin/env python3
"""Prepare one fixed manual-review timeline under the unchanged Mac guard.

The V3 automatic control result stays negative. This separate packet proposes
at most three geometry-ordered, QA-safe review squares without NDVI class rules.
Preparation never approves labels or reads a deforestation-model prediction.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MASK_NAMES = ('cloud', 'cloud_buffer', 'shadow_risk', 'haze_risk', 'invalid')
V3_SHAPES = {name: (64, 64) for name in ('pre_ndvi', 'post_ndvi', 'feature_valid',
    'target', 'positive_interior', 'target_context', 'positive_context',
    'catalogue_exclusion', 'control_exclusion', 'proposed_control', 'control_context')}
V3_SHAPES.update(pre_qa=(5, 64, 64), post_qa=(5, 64, 64),
                 pre_rgb=(3, 64, 64), post_rgb=(3, 64, 64), features=(27, 64, 64))
V3_FLOATS = ('pre_rgb', 'post_rgb', 'pre_ndvi', 'post_ndvi', 'features')


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
    require(path.is_file() and path.stat().st_size <= 32*2**20, 'Missing or oversized manual-review JSON')
    return json.loads(path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def rooted(relative, file=True):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            'Manual-review inputs must use project-relative paths')
    path = (ROOT/relative).resolve()
    require(path.is_relative_to(ROOT) and (path.is_file() if file else path.is_dir()),
            'Manual-review input escaped the project or is missing')
    return path


def _mask(value, name):
    import numpy as np
    require(isinstance(value, np.ndarray) and value.shape == (64, 64)
            and value.dtype == np.dtype(bool), name+' must be boolean64x64')
    return value


def control_roster(arrays, maximum=3):
    """Greedy first-safe squares with disjoint full input contexts, no NDVI test."""
    import numpy as np
    require(type(maximum) is int and 1 <= maximum <= 3, 'One to three review controls are allowed')
    features = arrays['features']
    require(features.shape == (27, 64, 64) and features.dtype == np.dtype('float32'),
            'Controls require all 27 original model channels')
    valid = _mask(arrays['feature_valid'], 'Feature validity')
    excluded = _mask(arrays['control_exclusion'], 'Control exclusion')
    allowed = valid & np.isfinite(features).all(axis=0) & ~excluded
    for side in ('pre', 'post'):
        qa, rgb, ndvi = (arrays[side+'_'+name] for name in ('qa', 'rgb', 'ndvi'))
        require(qa.shape == (5, 64, 64) and qa.dtype == np.dtype(bool), 'Endpoint QA must retain five boolean planes')
        require(rgb.shape == (3, 64, 64) and rgb.dtype == np.dtype('float32')
                and ndvi.shape == (64, 64) and ndvi.dtype == np.dtype('float32'),
                'Endpoint RGB/NDVI must retain float32 arrays')
        allowed &= ~np.any(qa, axis=0) & np.isfinite(rgb).all(axis=0) & np.isfinite(ndvi)
    reserved = np.zeros((64, 64), dtype=bool)
    metadata, rois, contexts = [], [], []
    for index in range(maximum):
        found = None
        for size in (16, 12, 8):
            for row in range(2, 65-size-2):
                for col in range(2, 65-size-2):
                    view = (slice(row-2, row+size+2), slice(col-2, col+size+2))
                    if allowed[view].all() and not reserved[view].any():
                        found = size, row, col
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            break
        size, row, col = found
        roi = np.zeros((64, 64), dtype=bool); roi[row:row+size, col:col+size] = True
        context = np.zeros_like(roi); context[row-2:row+size+2, col-2:col+size+2] = True
        reserved |= context
        metadata.append({'control_id': f'control-{index+1}', 'selection_order': index+1,
            'row': row, 'col': col, 'height': size, 'width': size, 'pixels': size*size,
            'context_radius_pixels': 2, 'context_pixels': (size+4)**2,
            'visual_review': 'unreviewed', 'training_approved': False, 'evaluation_approved': False})
        rois.append(roi); contexts.append(context)
    empty = np.zeros((0, 64, 64), dtype=bool)
    return metadata, np.stack(rois) if rois else empty.copy(), np.stack(contexts) if contexts else empty.copy()


def observation_metrics(observation, roi, context):
    """Per-image visual QA/finite evidence; never claims all 27 channels finite."""
    import numpy as np
    from audit_validation_context import context_mask
    from pixelqa_masks import region_metrics
    roi = _mask(roi, 'Review ROI'); context = _mask(context, 'Review context')
    require(roi.any() and np.array_equal(context, context_mask(roi, 2)),
            'Visual QA requires the full exact radius-two context, without clipping')
    for name in MASK_NAMES:
        _mask(observation[name], name)
    rgb, ndvi = observation['rgb'], observation['ndvi']
    require(rgb.shape == (3, 64, 64) and rgb.dtype == np.dtype('float32')
            and ndvi.shape == (64, 64) and ndvi.dtype == np.dtype('float32'),
            'Visual evidence requires float32 RGB and NDVI on the original grid')
    finite = np.isfinite(rgb).all(axis=0) & np.isfinite(ndvi)
    roi_metrics, context_metrics = region_metrics(observation, roi), region_metrics(observation, context)
    bad_roi, bad_context = int((roi & ~finite).sum()), int((context & ~finite).sum())
    return {'roi_pixels': int(roi.sum()), 'context_pixels': int(context.sum()),
        'full_context_inside_grid': True, 'roi': roi_metrics, 'context': context_metrics,
        'roi_nonfinite_rgb_ndvi_pixels': bad_roi,
        'context_nonfinite_rgb_ndvi_pixels': bad_context,
        'context_clear': bool(context_metrics['unusable_pixels'] == 0 and bad_context == 0)}


def _npz_headers(path, shapes, floats, metadata=False):
    """Validate allocation and types before loading any compressed image arrays."""
    import numpy as np
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 2*2**20, 'Manual-review NPZ compressed bound exceeded')
    with zipfile.ZipFile(path) as archive:
        names = [name+'.npy' for name in shapes]
        require(len(archive.namelist()) == len(names) and set(archive.namelist()) == set(names)
                and sum(item.file_size for item in archive.infolist()) <= 2*2**20,
                'Manual-review NPZ coverage or allocation bound changed')
        for name, expected_shape in shapes.items():
            with archive.open(name+'.npy') as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'Unsupported manual-review NPY format')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(stream)
                require(shape == expected_shape and not dtype.hasobject, 'Manual-review array shape/object type changed')
                if metadata and name == 'metadata':
                    require(dtype.kind == 'U' and dtype.itemsize <= 2**20, 'Cached metadata string bound exceeded')
                else:
                    require(dtype == np.dtype('float32' if name in floats else bool), 'Manual-review array dtype changed')


def load_v3_arrays(path, expected):
    import numpy as np
    require(digest(path) == expected, 'V3 context evidence hash changed')
    _npz_headers(path, V3_SHAPES, V3_FLOATS)
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in V3_SHAPES}
    require(digest(path) == expected, 'V3 evidence changed while loading')
    return values


def load_observation(path, policy):
    """Read the frozen cache safely without constructing or executing a model."""
    import numpy as np
    from verify_validation_dataset import read_cache
    shapes = {name: (64, 64) for name in (*MASK_NAMES, 'probability', 'product_cloud', 'ndvi')}
    shapes.update(rgb=(3, 64, 64), metadata=())
    _npz_headers(path, shapes, ('probability', 'rgb', 'ndvi'), metadata=True)
    before = digest(path)
    with np.load(path, allow_pickle=False) as archive:
        expected = json.loads(str(archive['metadata']))
    masks = read_cache(expected, policy)
    with np.load(path, allow_pickle=False) as archive:
        record = {**masks, 'rgb': archive['rgb'].copy(), 'ndvi': archive['ndvi'].copy(), 'metadata': expected}
    require(digest(path) == before, 'Observation cache changed while loading')
    return record, before


def validate_protocol(protocol):
    require(protocol.get('schema_version') == 4 and protocol.get('status')
            == 'predeclared_manual_review_before_new_imagery_or_validation_predictions',
            'The separately predeclared V4 manual-review protocol is required')
    rule = protocol['rule']
    for name, expected in {'context_radius_pixels': 2, 'control_sizes': [16, 12, 8],
            'maximum_control_candidates': 3, 'minimum_control_pixels': 64,
            'minimum_positive_interior_pixels': 4, 'catalogue_exclusion_buffer_metres': 20,
            'require_full_target_and_interior_context': True, 'require_full_halo_inside_core': True,
            'zero_all_five_qa_at_both_endpoints': True, 'finite_all27_endpoint_context': True,
            'finite_rgb_ndvi_each_clear_visual_context': True,
            'control_full_context_avoids_target_context_and_catalogue': True,
            'control_contexts_mutually_disjoint': True, 'no_ndvi_threshold_or_ranking': True,
            'roster_committed_before_images': True, 'no_roi_shrink_redraw_or_replacement': True,
            'v3_outcome_unchanged': True}.items():
        require(rule.get(name) == expected, 'Manual-review rule changed: '+name)
    require(protocol['resources'] == {'workers': 1, 'native_threads': 1, 'torch_threads': 1,
        'gpu_used': False, 'nice_increment': 10, 'worker_rss_mib': 512,
        'available_memory_reserve_gib': 1.5, 'max_seconds_per_worker': 300,
        'preparation_batch_size': 1}, 'Manual-review Mac capacity rules changed')
    require(len(protocol['timeline']) == 12 and len({row['scene_id'] for row in protocol['timeline']}) == 12,
            'The complete fixed twelve-scene timeline is required')
    require(sum(row['cache_exists'] is False for row in protocol['timeline']) == 4,
            'The four predeclared missing QA observations must remain explicit')


def _source_stats(statistics):
    for name, expected in statistics.items():
        stat = rooted(name).stat()
        require([stat.st_size, stat.st_mtime_ns] == expected, 'A manual-review source fingerprint changed: '+name)


@contextmanager
def count_quality_reads(receipt):
    """Count transparent calls to the frozen band reader without changing results."""
    import assess_pixel_quality
    original = assess_pixel_quality.read_fused_band
    def tracked(path, grid, include_feature):
        receipt.append({'source_path': str(Path(path).resolve().relative_to(ROOT)),
                        'include_feature_window': include_feature})
        return original(path, grid, include_feature)
    assess_pixel_quality.read_fused_band = tracked
    try:
        yield
    finally:
        assess_pixel_quality.read_fused_band = original


def _render_image(record, target, target_context, controls, contexts, mode):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    values = np.moveaxis(record['rgb'], 0, -1)
    rgb = (np.clip(np.nan_to_num(values, nan=0, posinf=0, neginf=0)/.30, 0, 1)*255).astype('uint8')
    if mode == 'qa':
        colors = {'cloud': (255, 255, 255), 'cloud_buffer': (255, 170, 35),
            'shadow_risk': (180, 70, 255), 'haze_risk': (40, 210, 255), 'invalid': (255, 30, 70)}
        for name, color in colors.items():
            rgb[record[name]] = ((rgb[record[name]].astype('float32')+np.asarray(color))/2).astype('uint8')
    elif mode == 'overlay':
        def outline(mask, color):
            image = Image.fromarray(mask.astype('uint8')*255)
            edge = np.asarray(image) > np.asarray(image.filter(ImageFilter.MinFilter(3)))
            rgb[edge] = color
        outline(target_context, (255, 180, 45))
        colors = ((80, 255, 90), (50, 220, 255), (255, 100, 240))
        for index, (control, halo) in enumerate(zip(controls, contexts)):
            outline(halo, tuple(int(channel*.6) for channel in colors[index]))
            outline(control, colors[index])
        outline(target, (255, 65, 65))
    return Image.fromarray(rgb)


def render_packet(output, observations, rows, arrays, roster):
    """Fixed nearest-neighbour views; the plain columns preserve unobscured pixels."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 17)
        small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 13)
    except OSError:
        font = small = ImageFont.load_default()
    target, halo, controls, contexts = (arrays[name] for name in ('target', 'target_context', 'controls', 'control_contexts'))
    modes = ('plain', 'overlay', 'qa')
    def panel(record, mode, size):
        image = _render_image(record, target, halo, controls, contexts, mode).resize((size, size), Image.Resampling.NEAREST)
        if mode == 'overlay':
            draw = ImageDraw.Draw(image)
            for control in roster:
                draw.text((int(control['col']*size/64)+3, int(control['row']*size/64)+3),
                    control['control_id'].replace('control-', 'C'), fill='white', font=small)
        return image
    def header(draw, title):
        draw.text((12, 8), title, fill='white', font=font)
        draw.text((12, 31), 'RGB reflectance 0-.30, nearest native pixels | red target; amber input margin; C1/C2/C3 review squares', fill='white', font=small)
        draw.text((12, 49), 'QA white cloud; orange buffer; purple shadow risk; cyan haze; red invalid. All labels unreviewed.', fill='white', font=small)
    def ticks(draw, x, y, size):
        for coordinate in (0, 16, 32, 48, 63):
            location = int((coordinate+.5)*size/64)
            draw.text((x+location-6, y-13), str(coordinate), fill='white', font=small)
            # Keep every glyph beyond the neighbouring plain panel's last
            # column. The 14px inter-panel gutter starts at x-14.
            draw.text((x-12, y+location-6), str(coordinate), fill='white', font=small)
    views = []
    for first in range(0, len(rows), 4):
        subset = list(zip(observations[first:first+4], rows[first:first+4]))
        panel_size, stride, top, gap = 320, 358, 91, 14
        canvas = Image.new('RGB', (3*panel_size+4*gap, top+len(subset)*stride+18), '#101822')
        draw = ImageDraw.Draw(canvas)
        header(draw, f'Manual validation timeline {first//4+1} | rows/cols0-63; each crop640m; no predictions')
        for col, label in enumerate(('Plain RGB', 'Target, full margin and control roster', 'Frozen QA flags')):
            draw.text((gap+col*(panel_size+gap), 72), label, fill='white', font=small)
        for row_index, (record, metadata) in enumerate(subset):
            y = top+row_index*stride
            text = metadata['date']+' | target context QA '+str(metadata['target_metrics']['context']['unusable_pixels'])
            draw.text((gap, y), text, fill='white', font=font)
            for col, mode in enumerate(modes):
                x = gap+col*(panel_size+gap)
                canvas.paste(panel(record, mode, panel_size), (x, y+25))
                if mode == 'overlay':
                    ticks(draw, x, y+25, panel_size)
        name = f'timeline-{first//4+1:02d}.png'; canvas.save(output/name)
        views.append({'file': name, 'scene_ids': [row['scene_id'] for _, row in subset],
                      'kinds': list(modes), 'width': canvas.width, 'height': canvas.height})
    # Larger endpoint panels make the four-pixel supervised interior inspectable
    # alongside its unmodified surrounding pixels; no enlargement adds detail.
    indices = [next(i for i, row in enumerate(rows) if row['scene_id'] == arrays['pair_'+side]) for side in ('pre', 'post')]
    panel_size, stride, top, gap = 384, 418, 91, 16
    canvas = Image.new('RGB', (2*panel_size+3*gap, top+3*stride+18), '#101822')
    draw = ImageDraw.Draw(canvas); header(draw, 'Fixed endpoint detail | same stretch and original64x64 pixels')
    for col, i in enumerate(indices):
        draw.text((gap+col*(panel_size+gap), 72), rows[i]['date'], fill='white', font=font)
        for r, mode in enumerate(modes):
            y = top+r*stride
            draw.text((gap+col*(panel_size+gap), y), mode, fill='white', font=font)
            canvas.paste(panel(observations[i], mode, panel_size), (gap+col*(panel_size+gap), y+24))
            if mode == 'overlay':
                ticks(draw, gap+col*(panel_size+gap), y+24, panel_size)
    canvas.save(output/'pair.png')
    views.append({'file': 'pair.png', 'scene_ids': [rows[i]['scene_id'] for i in indices],
                  'kinds': list(modes), 'width': canvas.width, 'height': canvas.height})
    yy, xx = np.nonzero(halo)
    side = min(64, max(int(yy.max()-yy.min()+1), int(xx.max()-xx.min()+1))+4)
    top_row, left = max(0, min(int(yy.min())-2, 64-side)), max(0, min(int(xx.min())-2, 64-side))
    bottom, right = top_row+side, left+side
    size = 384
    canvas = Image.new('RGB', (2*size+3*gap, 96+2*(size+32)+30), '#101822')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), 'Target detail | nearest enlargement, no additional image information', fill='white', font=font)
    draw.text((12, 33), f'Original grid rows[{top_row},{bottom}), cols[{left},{right}); red target, amber full2pixel margin', fill='white', font=small)
    for col, i in enumerate(indices):
        for r, mode in enumerate(('plain', 'overlay')):
            image = _render_image(observations[i], target, halo, controls, contexts, mode)
            image = image.crop((left, top_row, right, bottom)).resize((size, size), Image.Resampling.NEAREST)
            y = 96+r*(size+32)
            draw.text((gap+col*(size+gap), y-24), rows[i]['date']+' '+mode, fill='white', font=font)
            canvas.paste(image, (gap+col*(size+gap), y))
    canvas.save(output/'target-detail.png')
    views.append({'file': 'target-detail.png', 'scene_ids': [rows[i]['scene_id'] for i in indices],
        'kinds': ['plain', 'overlay'], 'width': canvas.width, 'height': canvas.height,
        'source_window': {'row_start': top_row, 'row_stop_exclusive': bottom,
                          'col_start': left, 'col_stop_exclusive': right}})
    return views


def prepare(args, output):
    import numpy as np
    import rasterio
    from assess_pixel_quality import ObservationReader, model_identity
    from prepare_curated_pilot import select_timeline
    from prepare_validation_review import (catalogue_exclusion, records, validate_quality_summary,
        validate_validation_pair)
    from verify_validation_context import feature_integrity, square_halo
    require('torch' not in sys.modules, 'Manual review must not import Torch')
    protocol_path = Path(args.protocol).resolve(); protocol = read_json(protocol_path)
    validate_protocol(protocol)
    source_hashes = {str(protocol_path.relative_to(ROOT)): digest(protocol_path)}
    sources = {}
    for role, item in protocol['source_files'].items():
        path = rooted(item['path']); require(digest(path) == item['sha256'], 'Manual-review input changed: '+role)
        sources[role] = path; source_hashes[item['path']] = item['sha256']
    for name, expected in protocol['immutable_sources_sha256'].items():
        require(digest(ROOT/'Solutions'/name) == expected, 'Frozen source changed: '+name)
    require(isinstance(protocol.get('implementation_sha256'), dict)
            and 'prepare_validation_manual.py' in protocol['implementation_sha256'],
            'Freeze the tested preparation implementation before processing')
    implementation = protocol['implementation_sha256']
    for name, expected in implementation.items():
        require(digest(ROOT/'Solutions'/name) == expected, 'Predeclared manual implementation changed: '+name)
    previous = read_json(sources['v3_audit_metadata']); verified = read_json(sources['v3_verification'])
    previous_protocol = read_json(sources['v3_protocol'])
    require(previous['execution_status'] == 'completed' and verified['verification_status'] == 'passed'
            and verified['audit_metadata_sha256'] == digest(sources['v3_audit_metadata'])
            and verified['protocol_sha256'] == previous['protocol_sha256'] == digest(sources['v3_protocol'])
            and verified['audit_artifact_sha256']['context_arrays.npz'] == digest(sources['v3_arrays'])
            and previous['artifact_sha256']['context_arrays.npz'] == digest(sources['v3_arrays'])
            and verified['numeric_context_passed'] is False and previous['numeric_context_passed'] is False
            and previous['proposal_metadata']['proposal_found'] is False,
            'V3 completed negative result or independent evidence changed')
    pair, candidate = protocol['pair'], protocol['candidate']
    require(pair == previous['pair'] and candidate == previous['candidate']
            and candidate == previous_protocol['selection']['candidate'], 'The single fixed candidate changed')
    matched = [row for row in records(sources['candidate_pairs']) if row['pair_id'] == pair['pair_id']]
    require(matched == [pair], 'Frozen pair manifest membership changed')
    catalogue, splits = read_json(sources['catalogue']), read_json(sources['splits'])
    training = list(records(sources['training_manifest']))
    validate_validation_pair(pair, candidate['event_id'], catalogue, splits, training)
    for role in ('catalogue', 'splits', 'training_manifest', 'training_metadata'):
        require(protocol['source_files'][role]['sha256'] == previous_protocol['source_files'][role]['sha256'],
                'V3 split/training/catalogue identity changed')
    quality = read_json(sources['quality_summary']); validate_quality_summary(quality)
    require(quality['config_id'] == protocol['qa_policy_id'], 'Frozen QA policy changed')
    arrays = load_v3_arrays(sources['v3_arrays'], protocol['source_files']['v3_arrays']['sha256'])
    for key in V3_SHAPES:
        require(hashlib.sha256(arrays[key].tobytes()).hexdigest() == previous['array_sha256'][key],
                'A bound V3 array changed: '+key)
    require(np.array_equal(arrays['catalogue_exclusion'], catalogue_exclusion(pair))
            and np.array_equal(arrays['control_exclusion'], arrays['catalogue_exclusion'] | arrays['target_context']),
            'Control exclusion differs from the unchanged catalogue/target context')
    for roi_name, context_name in (('target', 'target_context'), ('positive_interior', 'positive_context')):
        context, inside = square_halo(arrays[roi_name])
        require(inside and np.array_equal(context, arrays[context_name])
                and previous['contexts'][roi_name]['context_clear'] is True,
                'Both complete fixed target contexts must remain clear')
    roster, controls, contexts = control_roster(arrays)
    # Freeze the numerical roster before any image is rendered or inspected.
    # The parent commits this complete unreviewed packet before opening its views.
    scenes, window = select_timeline(list(records(sources['scene_inventory'])), pair['tile'], candidate['annotation_date'])
    require(window == protocol['timeline_window']
            and [(row['scene_id'], row['acquisition_date'], row['path'], row['metadata_usable']) for row in scenes]
            == [(row['scene_id'], row['date'], row['scene_path'], row['metadata_usable']) for row in protocol['timeline']],
            'The complete predeclared timeline changed')
    require(all(row['metadata_usable'] is True for row in scenes), 'All twelve declared scenes require usable metadata')
    stats = {}
    for row in protocol['timeline']:
        for name, values in row['source_file_stats'].items():
            require(name not in stats or stats[name] == values, 'Timeline source fingerprints disagree')
            stats[name] = values
        path = ROOT/row['cache_path']
        if row['cache_exists']:
            require(path.is_file() and digest(path) == row['cache_sha256'], 'A predeclared existing cache changed')
            record, _ = load_observation(path, protocol['qa_policy_id'])
            require(record['metadata']['source_id'] == row['source_id'], 'Existing cache source binding changed')
        else:
            require(not path.exists() and row.get('cache_sha256') is None,
                    'An unbound new cache already exists; record an explicit recovery checkpoint before processing')
    _source_stats(stats); _source_stats(previous['feature_source_file_stats'])
    reader = ObservationReader(ROOT/'Datasets/PixelQA/cache', model_identity())
    require(reader.config_id == protocol['qa_policy_id'], 'Runtime QA policy differs from the frozen policy')
    # Metadata preflight covers all sources before any new scene is processed.
    for row in protocol['timeline']:
        meta = reader.metadata(row['scene_path'])
        require(meta['source_file_stats'] == row['source_file_stats']
                and meta['metadata_uid'] == row['metadata_uid']
                and meta['cache_source_id'] == row['source_id'] and meta['scene_id'] == row['scene_id'],
                'Timeline cache prerequisites differ from the metadata-only plan')
    observations, timeline, cache_hashes, quality_reads = [], [], {}, []
    with count_quality_reads(quality_reads), rasterio.Env(GDAL_CACHEMAX=32*2**20, GDAL_NUM_THREADS='1'):
        for index, plan in enumerate(protocol['timeline']):
            print(f"[manual-review] {index+1}/12 {plan['date']} {'cache hit' if plan['cache_exists'] else 'bounded frozen QA fill'}", flush=True)
            actual = reader.observation(plan['scene_path'], pair['grid'])
            expected = actual['metadata']
            require(expected['scene_id'] == plan['scene_id'] and expected['metadata_uid'] == plan['metadata_uid']
                    and expected['source_id'] == plan['source_id'] and expected['cache_key'] == plan['cache_key']
                    and expected['cache_path'] == plan['cache_path'] and expected['grid'] == pair['grid']
                    and expected['config_id'] == protocol['qa_policy_id'], 'Actual timeline observation binding changed')
            record, checksum = load_observation(rooted(plan['cache_path']), protocol['qa_policy_id'])
            require(record['metadata'] == expected and all(np.array_equal(record[name], actual[name], equal_nan=True)
                    for name in (*MASK_NAMES, 'rgb', 'ndvi')), 'Published cache differs from the returned QA observation')
            if plan['cache_exists']:
                require(checksum == plan['cache_sha256'], 'An existing cache changed during preparation')
            cache_hashes[plan['cache_path']] = checksum
            metadata = {**plan, 'observation': expected, 'cache_sha256': checksum,
                'cache_created_this_run': not plan['cache_exists'],
                'target_metrics': observation_metrics(record, arrays['target'], arrays['target_context']),
                'interior_metrics': observation_metrics(record, arrays['positive_interior'], arrays['positive_context']),
                'control_metrics': [observation_metrics(record, roi, halo) for roi, halo in zip(controls, contexts)],
                'visual_event_state': 'unreviewed', 'visual_control_states': ['unreviewed']*len(roster),
                'training_approved': False, 'evaluation_approved': False}
            observations.append(record); timeline.append(metadata)
    require(reader.misses == 4 and reader.hits == 8, 'Actual QA/cache work differs from the fixed eight-hit/four-fill plan')
    require(len(quality_reads) == 44 and sum(row['include_feature_window'] for row in quality_reads) == 28,
            'Actual frozen QA band-reader requests differ from the declared44/28 bounds')
    endpoints = {side: observations[next(i for i, row in enumerate(timeline) if row['scene_id'] == pair[side+'_scene_id'])]
                 for side in ('pre', 'post')}
    feature_integrity(arrays['features'], arrays['feature_valid'], endpoints)
    for side in ('pre', 'post'):
        require(np.array_equal(np.stack([endpoints[side][name] for name in MASK_NAMES]), arrays[side+'_qa'])
                and np.array_equal(endpoints[side]['rgb'], arrays[side+'_rgb'], equal_nan=True)
                and np.array_equal(endpoints[side]['ndvi'], arrays[side+'_ndvi'], equal_nan=True),
                'Original endpoint arrays changed during manual preparation')
    saved = {'qa': np.stack([np.stack([record[name] for name in MASK_NAMES]) for record in observations]),
        'rgb': np.stack([record['rgb'] for record in observations]), 'ndvi': np.stack([record['ndvi'] for record in observations]),
        'controls': controls, 'control_contexts': contexts,
        **{name: arrays[name] for name in ('target', 'positive_interior', 'target_context', 'positive_context',
            'catalogue_exclusion', 'control_exclusion', 'feature_valid', 'features')}}
    np.savez_compressed(output/'observations.npz', **saved)
    _npz_headers(output/'observations.npz', {name: value.shape for name, value in saved.items()}, ('rgb', 'ndvi', 'features'))
    with np.load(output/'observations.npz', allow_pickle=False) as archive:
        require(set(archive.files) == set(saved) and all(np.array_equal(archive[name], value, equal_nan=True)
                for name, value in saved.items()), 'Manual evidence NPZ failed its exact round trip')
    render_arrays = {**saved, 'pair_pre': pair['pre_scene_id'], 'pair_post': pair['post_scene_id']}
    views = render_packet(output, observations, timeline, render_arrays, roster)
    _source_stats(stats); _source_stats(previous['feature_source_file_stats'])
    for name, checksum in {**source_hashes, **cache_hashes}.items():
        require(digest(rooted(name)) == checksum, 'A source/cache changed during manual preparation')
    for name, checksum in {**protocol['immutable_sources_sha256'], **implementation}.items():
        require(digest(ROOT/'Solutions'/name) == checksum, 'A frozen implementation changed during preparation')
    require('torch' not in sys.modules, 'Manual preparation unexpectedly imported Torch')
    artifacts = {path.name: digest(path) for path in output.iterdir() if path.is_file()}
    return {'schema_version': 4, 'mode': 'prepare_validation_manual', 'execution_status': 'completed',
        'scope': protocol['scope'], 'protocol_sha256': digest(protocol_path),
        'event_id': candidate['event_id'], 'pair_id': pair['pair_id'], 'group_id': pair['group_id'],
        'split': 'validation', 'pair': pair, 'candidate': candidate, 'qa_policy_id': protocol['qa_policy_id'],
        'architecture': previous['architecture'], 'v3_numeric_context_passed': False,
        'controls': roster, 'control_search_uses_ndvi_thresholds_or_ranking': False,
        'timeline_window': window, 'timeline': timeline, 'render_views': views,
        'source_input_sha256': source_hashes, 'observation_cache_sha256': cache_hashes,
        'implementation_sha256': implementation, 'source_file_stats': stats,
        'feature_source_file_stats': previous['feature_source_file_stats'],
        'array_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in saved.items()},
        'artifact_sha256': artifacts, 'new_observations': reader.misses, 'cache_hits': reader.hits,
        'cloud_model_executions': reader.misses, 'forest_model_executions': 0, 'model_executions': 0,
        'new_pair_feature_extractions': 0,
        'quality_fused_band_window_requests': len(quality_reads),
        'quality_include_feature_window_requests': sum(row['include_feature_window'] for row in quality_reads),
        'source_jp2_include_feature_window_requests_for_new_qa': sum(row['include_feature_window'] for row in quality_reads),
        'quality_band_read_receipt': quality_reads,
        'source_jp2_band_window_requests_for_new_qa': len(quality_reads),
        'source_jp2_read_count_scope': 'Instrumented transparent read_fused_band calls for the eleven distinct cloud/feature bands per new QA observation, including seven feature windows; not underlying GDAL read calls',
        'torch_imported': False, 'new_imagery_reviewed': 0, 'labels_approved': 0,
        'training_approved': False, 'evaluation_approved': False, 'model_predictions_read': False,
        'feature_finiteness_scope': 'All27 channels checked only for fixed endpoints; timeline visual claims require full-context QA and finite RGB/NDVI.',
        'limitations': ['Preparation and QA-safe squares do not approve forest or clearing labels.',
            'V3 automatic NDVI eligibility remains false; no threshold or old result changed.',
            'A 64x64 crop and nearest-neighbour enlargement do not add spatial information.',
            'Only the exact frozen tiny CNN has the two-pixel model-context contract.',
            'Four new QA cache observations use the unchanged cloud model; no forest model is executed.']}


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationManualV4/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Keep RSS <=512MiB, available-memory reserve >=1.5GiB and runtime <=300s')
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        os.nice(10)
        result = prepare(args, args.worker_output)
        import resource
        result['worker_high_water_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024), 2)
        write_json(args.worker_output/'metadata.json', result)
    else:
        run_guarded(args, script_path=__file__, operation='validation-manual-preparation', gdal_cache_mib=32, warp_memory_mib=16)


if __name__ == '__main__':
    main()
