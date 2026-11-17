#!/usr/bin/env python3
"""Predeclared one-case target/control context QA under the Mac resource guard."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
QA_NAMES = ('cloud', 'cloud_buffer', 'shadow_risk', 'haze_risk', 'invalid')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mask(value, name, nonempty=True):
    import numpy as np
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype(bool)
            and value.shape == (64, 64) and (not nonempty or bool(value.any())),
            name+' must be a '+('nonempty ' if nonempty else '')+'64x64 boolean mask')
    return value


def context_mask(roi, radius=2):
    import numpy as np
    from scipy.ndimage import binary_dilation
    roi = _mask(roi, 'ROI')
    require(type(radius) is int and 0 <= radius <= 8, 'Context radius must be a bounded nonnegative integer')
    yy, xx = np.nonzero(roi)
    require(yy.min() >= radius and xx.min() >= radius
            and yy.max() < 64-radius and xx.max() < 64-radius,
            'Full input context extends outside the frozen core')
    return binary_dilation(roi, structure=np.ones((2*radius+1, 2*radius+1), dtype=bool))


def _qa_union(observation):
    import numpy as np
    return np.logical_or.reduce([_mask(observation[n], n, nonempty=False) for n in QA_NAMES])


def qa_region_counts(region, pre, post):
    region = _mask(region, 'Quality region')
    result = {'pixels': int(region.sum())}
    for side, observation in (('pre', pre), ('post', post)):
        result[side+'_unusable_pixels'] = int((_qa_union(observation) & region).sum())
        result[side+'_components'] = {n: int((observation[n] & region).sum()) for n in QA_NAMES}
    return result


def _usable_context(observation):
    import numpy as np
    rgb, ndvi = observation['rgb'], observation['ndvi']
    require(isinstance(rgb, np.ndarray) and rgb.shape == (3, 64, 64) and rgb.dtype.kind == 'f'
            and isinstance(ndvi, np.ndarray) and ndvi.shape == (64, 64) and ndvi.dtype.kind == 'f',
            'Control proposal requires the frozen floating RGB/NDVI arrays')
    valid = _mask(observation['feature_valid'], 'Feature validity', nonempty=False)
    return ~_qa_union(observation) & valid & np.isfinite(rgb).all(axis=0) & np.isfinite(ndvi)


def propose_context_control(pre, post, excluded, radius=2, sizes=(16, 12, 8)):
    """First numerically eligible square; this function never approves a label."""
    import numpy as np
    excluded = _mask(excluded, 'Control exclusion', nonempty=False)
    require(type(radius) is int and radius == 2, 'This proposal is bound to the current radius-two model')
    require(isinstance(sizes, (tuple, list)) and sizes and all(type(s) is int and s in (16, 12, 8) for s in sizes)
            and list(sizes) == sorted(set(sizes), reverse=True), 'Use distinct declared control sizes in descending order')
    allowed = _usable_context(pre) & _usable_context(post) & ~excluded
    before, after = pre['ndvi'], post['ndvi']
    green = (before > .6) & (after > .6) & (np.abs(after-before) < .1)
    for size in sizes:
        for row in range(radius, 65-size-radius):
            for col in range(radius, 65-size-radius):
                core = (slice(row, row+size), slice(col, col+size))
                full = (slice(row-radius, row+size+radius), slice(col-radius, col+size+radius))
                if not green[core].all() or not allowed[full].all():
                    continue
                mask = np.zeros((64, 64), dtype=bool); mask[core] = True
                return mask, {'proposal_found': True, 'row': row, 'col': col,
                    'height': size, 'width': size, 'pixels': size*size,
                    'context_radius_pixels': radius, 'context_pixels': (size+2*radius)**2,
                    'minimum_endpoint_ndvi': float(np.minimum(before[core], after[core]).min()),
                    'maximum_absolute_ndvi_change': float(np.abs(after[core]-before[core]).max()),
                    'search_order': 'largest declared size, ascending row, ascending column',
                    'control_label_review': 'unreviewed', 'training_approved': False, 'evaluation_approved': False}
    return np.zeros((64, 64), dtype=bool), {'proposal_found': False, 'pixels': 0,
        'reason': 'No declared square has green stable endpoint ROI and fully clear, finite, unexcluded input context',
        'control_label_review': 'unreviewed', 'training_approved': False, 'evaluation_approved': False}


def architecture_contract(factory_path):
    """Bind the receptive-field statement to the actual simple model factory."""
    tree = ast.parse(Path(factory_path).read_text())
    factories = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_model']
    require(len(factories) == 1, 'Exactly one frozen model factory required')
    body = factories[0].body
    require(all(isinstance(n, (ast.Import, ast.ImportFrom, ast.Return)) for n in body)
            and sum(isinstance(n, ast.Return) for n in body) == 1, 'Model factory has unsupported nonlocal operations')
    ret = next(n.value for n in body if isinstance(n, ast.Return))
    require(isinstance(ret, ast.Call) and isinstance(ret.func, ast.Attribute) and ret.func.attr == 'to', 'Frozen CPU model conversion required')
    sequence = ret.func.value
    require(isinstance(sequence, ast.Call) and isinstance(sequence.func, ast.Attribute)
            and sequence.func.attr == 'Sequential' and not sequence.keywords, 'Simple sequential architecture required')
    layers = sequence.args
    require(len(layers) == 5 and [n.func.attr if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) else None
            for n in layers] == ['Conv2d', 'ReLU', 'Conv2d', 'ReLU', 'Conv2d'],
            'Architecture includes unsupported/global-context operations')
    expected = [((27, 8, 3), {'padding': 1}), ((8, 8, 3), {'padding': 1}), ((8, 1, 1), {})]
    for layer, (args, keywords) in zip(layers[::2], expected):
        require(tuple(ast.literal_eval(v) for v in layer.args) == args
                and {k.arg: ast.literal_eval(k.value) for k in layer.keywords} == keywords,
                'Frozen convolution kernel/stride/dilation/group contract changed')
    require(all(not layers[i].args and not layers[i].keywords for i in (1, 3)), 'Pointwise activations changed')
    return {'spatial_receptive_field_pixels': 5, 'context_radius_pixels': 2,
            'parameter_count': 2545, 'global_context_operations': False, 'factory_sha256': digest(factory_path)}


def validate_protocol(protocol):
    require(protocol['schema_version'] == 3
            and protocol['status'] == 'predeclared_context_quality_only_before_new_visual_review', 'Fixed V3 audit protocol required')
    rule = protocol['rule']
    require(rule['context_radius_pixels'] == 2 and rule['minimum_positive_interior_pixels'] == 4
            and rule['target_erosion_pixels'] == 1 and rule['control_sizes'] == [16, 12, 8]
            and rule['minimum_control_pixels'] == 64
            and rule['control_minimum_endpoint_ndvi_exclusive'] == .6
            and rule['control_maximum_absolute_ndvi_difference_exclusive'] == .1
            and rule['catalogue_exclusion_buffer_metres'] == 20,
            'Declared context/support/control thresholds changed')
    for key in ('require_full_halo_inside_core', 'require_full_target_and_interior_context',
                'require_zero_qa_at_both_dates', 'require_finite_all27_features_on_context',
                'retain_unchanged_full_crop_flags', 'control_full_halo_avoids_catalogue_buffer_and_target_context',
                'no_control_is_valid_negative_outcome'):
        require(rule[key] is True, 'Declared audit gate changed: '+key)
    require(rule['qa_components'] == list(QA_NAMES) and protocol['selection']['maximum_cases'] == 1,
            'QA components or fixed case bound changed')
    architecture = protocol['architecture']
    require(architecture['factory_file'] == 'Solutions/train_pilot_baseline.py'
            and architecture['factory_sha256'] == protocol['immutable_sources_sha256']['train_pilot_baseline.py']
            and architecture['spatial_receptive_field_pixels'] == 5
            and architecture['context_radius_pixels'] == 2 and architecture['parameter_count'] == 2545
            and architecture['global_context_operations'] is False, 'Declared model dependency contract changed')
    require(protocol['resources'] == {'workers': 1, 'native_threads': 1, 'gpu_used': False,
            'nice_increment': 10, 'worker_rss_mib': 512, 'available_memory_reserve_gib': 1.5,
            'max_seconds_per_worker': 300, 'maximum_cases': 1, 'feature_band_window_requests': 14},
            'Mac capacity or feature-reading bounds changed')


def write_report(path, metadata):
    lines = ['# Target and control input-context quality audit', '',
        '**Numeric eligibility only: no forest/clearing labels or model accuracy.**', '',
        f"Candidate `{metadata.get('event_id', '')}` / `{metadata.get('pair_id', '')}`.",
        f"Eligible for subsequent visual review: **{metadata.get('qa_eligible_for_visual_review')}**.", '',
        '| Region | ROI pixels | Context pixels | Full context inside grid | Clear and finite context |',
        '| --- | ---: | ---: | --- | --- |']
    for name, result in metadata['contexts'].items():
        lines.append(f"| {name} | {result['roi_pixels']} | {result['context_pixels']} | {result['full_context_inside_grid']} | {result['context_clear']} |")
    lines += ['', 'Full target and supervised-interior contexts are both required. No smaller ROI/halo fallback.',
        'A proposed forest control still needs explicit visual and temporal review. Missing controls remain a valid negative result.',
        'All original full-crop flags and V1/V2 decisions remain unchanged. Context radius two is bound to the frozen tiny CNN only.',
        'All 27 source features were checked in the one-case worker. Independent verification uses saved arrays/source fingerprints without repeating raw band extraction.',
        'No model inference, training, visual preview or label approval was performed.']
    Path(path).write_text('\n'.join(lines).rstrip()+'\n')


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationContextV3/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Use 128–512 MiB RSS, >=1.5 GiB reserve and <=300 seconds')
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    validate_protocol(json.loads(args.protocol.read_text()))
    if args.worker_output:
        os.nice(10)
        from validation_context_worker import run
        metadata = run(args, args.worker_output)
        write_report(args.worker_output/'report.md', metadata)
        metadata.setdefault('artifact_sha256', {})['report.md'] = digest(args.worker_output/'report.md')
        import resource
        metadata['worker_high_water_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024), 2)
        (args.worker_output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    else:
        run_guarded(args, script_path=__file__, operation='validation-context-quality', gdal_cache_mib=32, warp_memory_mib=16)


if __name__ == '__main__':
    main()
