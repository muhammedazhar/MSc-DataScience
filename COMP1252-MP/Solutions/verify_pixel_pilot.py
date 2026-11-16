#!/usr/bin/env python3
"""Verify the bounded all-example pilot from saved arrays, without Torch.

The verifier reconstructs reviewed-pixel losses and counts independently using
NumPy. Model gradient execution and saved-weight forward replay remain explicit
worker receipts; this process neither imports Torch nor executes a model.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

from summarize_pilot_overfit import (CHANNELS, WEIGHT_SHAPES, checked_npz, compare,
                                    digest, read_json, recompute_checkpoint,
                                    records, require, resource_receipt, within)

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SOURCES = {'train_reviewed_pilot.py', 'reviewed_pilot_worker.py',
                    'pilot_pixel_objective.py', 'pilot_dataset.py',
                    'pilot_objectives.py', 'train_pilot_baseline.py',
                    'pilot_overfit_worker.py', 'diagnose_pilot_overfit.py',
                    'run_pilot_checks.py', 'validate_sample.py'}


def aggregate_checkpoint(logits, probabilities, labels, event_ids, pos_weight, minimum=.95):
    """Rebuild per-example and pixel-pooled evidence, keeping failed cases."""
    import numpy as np
    from diagnose_pilot_overfit import fit_assessment

    z, probability, target = map(np.asarray, (logits, probabilities, labels))
    ids = list(event_ids)
    require(z.ndim == 3 and z.shape == probability.shape == target.shape
            and 0 < z.shape[0] <= 13 and 0 < z.shape[1]*z.shape[2] <= 4096,
            'Checkpoint arrays must cover a bounded list of matching image cores')
    require(len(ids) == z.shape[0] and all(isinstance(v, str) and v for v in ids)
            and len(set(ids)) == len(ids), 'Checkpoint event coverage must be unique and complete')
    examples = []
    for index, event_id in enumerate(ids):
        examples.append({'event_id': event_id,
                         **recompute_checkpoint(z[index], probability[index], target[index],
                                                pos_weight, minimum)})
    counts = {key: sum(row['metrics'][key] for row in examples)
              for key in ('tp', 'fp', 'fn', 'tn', 'ignored_pixels')}
    tp, fp, fn, tn = (counts[key] for key in ('tp', 'fp', 'fn', 'tn'))
    p, n, total = tp+fn, tn+fp, tp+fp+fn+tn
    precision = tp/(tp+fp) if tp+fp else None
    recall, specificity = tp/p, tn/n
    dice = 2*tp/(2*tp+fp+fn)
    metrics = {**counts, 'supervised_pixels': total, 'positive_pixels': p,
               'stable_pixels': n, 'updates': len(ids), 'empty_updates': 0,
               'probability_threshold': .5, 'precision': precision, 'recall': recall,
               'specificity': specificity, 'f1': dice, 'dice': dice,
               'iou': tp/(tp+fp+fn), 'accuracy': (tp+tn)/total,
               'balanced_accuracy': (recall+specificity)/2}
    weighted = sum(row['weighted_bce']*row['metrics']['supervised_pixels'] for row in examples)/total
    unweighted = sum(row['unweighted_bce']*row['metrics']['supervised_pixels'] for row in examples)/total
    aggregate = {'metrics': metrics, 'fit': fit_assessment(metrics, p, n, minimum),
                 'weighted_bce': weighted, 'unweighted_bce': unweighted,
                 'mean_weight_normalized_bce': weighted/((pos_weight*p+n)/total)}
    all_examples_fit = all(row['fit']['passed'] for row in examples)
    return {'aggregate': aggregate, 'examples': examples,
            'all_examples_fit': all_examples_fit,
            'fit_passed': aggregate['fit']['passed'] and all_examples_fit}


def source_dataset(plan, dataset):
    """Read only the small exported TIFFs and independently bind their tensors."""
    import numpy as np
    import rasterio
    from affine import Affine

    selection, objective = plan['selection'], plan['objective']
    require(digest(dataset/'metadata.json') == selection['dataset_metadata_sha256']
            and digest(dataset/'training_manifest.jsonl') == selection['manifest_sha256'],
            'Frozen dataset metadata or manifest changed')
    metadata = read_json(dataset/'metadata.json')
    require(metadata['qa_policy_id'] == selection['qa_policy_id']
            and metadata['output_sha256_excluding_this_metadata'] == selection['artifact_sha256'],
            'Frozen dataset publication identity changed')
    for name, checksum in selection['artifact_sha256'].items():
        require(digest(within(dataset, name)) == checksum, 'Dataset artifact changed: '+name)
    rows = list(records(dataset/'training_manifest.jsonl', maximum=13))
    require(len(rows) == objective['image_count'] == 13
            and [row['event_id'] for row in rows] == selection['event_ids']
            and [row['group_id'] for row in rows] == selection['group_ids']
            and len(set(selection['event_ids'])) == len(set(selection['group_ids'])) == 13
            and len({row['pair_id'] for row in rows}) == 13,
            'Exact thirteen-event/group manifest coverage changed')
    labels, tensor_hashes = [], []
    for row in rows:
        require(row['split'] == 'train' and row.get('training_approved') is True
                and row['qa_policy_id'] == selection['qa_policy_id']
                and row.get('visibility_evidence', {}).get('decision_status') == 'accepted'
                and row['visibility_evidence'].get('training_approved') is True,
                'Source must retain explicitly reviewed training-only evidence')
        grid, arrays = row['grid'], {}
        with rasterio.Env(GDAL_CACHEMAX=16*2**20, GDAL_NUM_THREADS='1'):
            for role in ('input_features', 'supervision_mask', 'valid_mask'):
                path = within(dataset, row['artifact_paths'][role])
                require(digest(path) == selection['artifact_sha256'][row['artifact_paths'][role]],
                        'Manifest artifact does not bind to frozen publication')
                with rasterio.open(path) as source:
                    require(source.shape == (64, 64) and source.transform == Affine(*grid['transform'])
                            and source.crs == rasterio.crs.CRS.from_user_input(grid['crs']),
                            'Source TIFF frozen grid changed')
                    if role == 'input_features':
                        require(source.count == 27 and source.dtypes == ('float32',)*27
                                and source.descriptions == CHANNELS, 'Source feature headers changed')
                    else:
                        require(source.count == 1 and source.dtypes == ('uint8',),
                                'Source label/validity headers changed')
                        if role == 'supervision_mask':
                            require(source.nodata == 255, 'Source ignore convention changed')
                    arrays[role] = source.read()
        features, target, valid = arrays['input_features'], arrays['supervision_mask'][0], arrays['valid_mask'][0]
        require(np.isfinite(features).all() and np.isin(target, (0, 1, 255)).all()
                and np.isin(valid, (0, 1)).all() and np.array_equal(valid.astype(bool), target != 255),
                'Source finite features or partial supervision changed')
        require(np.array_equal(features[18:], features[9:18]-features[:9]),
                'Source signed feature differences changed')
        for field, value in (('supervised_positive_pixels', 1), ('supervised_stable_pixels', 0),
                             ('ignored_pixels', 255)):
            require(type(row[field]) is int and int((target == value).sum()) == row[field],
                    'Source class counts differ from reviewed manifest')
        require(row['supervised_positive_pixels'] > 0 and row['supervised_stable_pixels'] > 0,
                'Every reviewed example must retain both classes')
        checksum = hashlib.sha256()
        for name, array, dtype in (('features', features, 'torch.float32'),
                                   ('labels', target, 'torch.uint8'),
                                   ('valid', valid.astype(bool), 'torch.bool')):
            checksum.update(name.encode())
            checksum.update(str(tuple(array.shape)).encode())
            checksum.update(dtype.encode())
            checksum.update(np.ascontiguousarray(array).tobytes())
        tensor_hashes.append(checksum.hexdigest())
        labels.append(target.copy())
    labels = np.stack(labels)
    p, n, ignored = (int((labels == value).sum()) for value in (1, 0, 255))
    require((p, n, ignored) == (selection['positive_pixels'], selection['stable_pixels'], selection['ignored_pixels'])
            == (151, 2640, 50457) and objective['total_reviewed_pixels'] == p+n
            and objective['positive_weight'] == n/p, 'Frozen aggregate support or class weighting changed')
    audit = {'example_count': 13, 'supervised_positive_pixels': p, 'supervised_stable_pixels': n,
             'ignored_pixels': ignored, 'nonfinite_feature_values_replaced': 0,
             'examples_with_nonfinite_replacements': 0, 'event_ids': selection['event_ids'],
             'group_ids': selection['group_ids'], 'splits': ['train'], 'feature_shape': [27, 64, 64],
             'channels': list(CHANNELS), 'qa_policy_id': selection['qa_policy_id'],
             'dataset_metadata_sha256': selection['dataset_metadata_sha256'],
             'manifest_sha256': selection['manifest_sha256'],
             'feature_handling': 'Exported finite values unchanged; non-finite ignored inputs replaced with zero only'}
    return labels, tensor_hashes, audit


def verify_steps(steps, labels, event_ids, objective, protocol):
    """Check exact exposures and finite per-step normalization/gradient receipts."""
    import numpy as np

    require(len(steps) == protocol['attempts'] == protocol['cycles']*len(event_ids),
            'Fixed training exposure changed')
    support = [int(np.count_nonzero(target != 255)) for target in labels]
    stalls = []
    for index, row in enumerate(steps):
        case = index % len(event_ids)
        compare(row['attempt'], index+1, 'Step attempt')
        compare(row['event_id'], event_ids[case], 'Step manifest order')
        compare(row['reviewed_pixels'], support[case], 'Step reviewed count')
        compare(row['step_scale'], len(event_ids)*support[case]/objective['total_reviewed_pixels'],
                'Step pixel scale')
        for key in ('scaled_loss_before_step', 'gradient_norm', 'parameter_delta_l2'):
            require(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0,
                    'Step loss, gradient and parameter delta must be finite/nonnegative')
        require(row['optimizer_step_called'] is True and row.get('updated', True) is True
                and row['ignored_gradient_max_abs'] == 0.
                and type(row['zero_parameter_delta']) is bool
                and row['zero_parameter_delta'] == (row['parameter_delta_l2'] == 0),
                'Step optimizer/ignored-gradient/stationary receipt mismatch')
        if row['zero_parameter_delta']:
            stalls.append(index+1)
    require(len(stalls) < len(steps), 'Training never changed parameters')
    return stalls


def verify(args, output):
    import numpy as np
    from train_reviewed_pilot import validate_plan

    require('torch' not in sys.modules, 'Independent verifier must not import Torch')
    plan_path, directory, dataset = args.plan.resolve(), args.run.resolve(), args.dataset.resolve()
    plan, metadata = read_json(plan_path), read_json(directory/'metadata.json')
    validate_plan(plan)
    selection, objective, protocol = plan['selection'], plan['objective'], plan['protocol']
    require(plan['schema_version'] == 1 and protocol['checkpoint_attempts'] == [0, 26, 416, 832, 1664]
            and protocol['attempts'] == 1664 and protocol['cycles'] == 128
            and protocol['probability_threshold'] == .5 and protocol['early_stopping'] is False
            and plan['fit_gate']['primary_attempt'] == 1664 and plan['fit_gate']['minimum'] == .95
            and plan['fit_gate']['require_each_example_and_aggregate'] is True,
            'Predeclared exposure, threshold or primary gate changed')
    require(objective['name'] == 'dataset_reviewed_pixel_mean_weighted_bce'
            and protocol['seed'] == 42 and protocol['parameter_count'] == 2545
            and protocol['batch_size'] == 1 and protocol['dtype'] == 'float32'
            and protocol['augmentation'] is False and protocol['order'] == 'fixed manifest order, no shuffle'
            and protocol['optimizer'] == {'optimizer': 'SGD', 'learning_rate': .05,
                                          'momentum': 0., 'weight_decay': 0.},
            'Predeclared objective, seed, architecture or optimizer changed')
    require(metadata['execution_status'] == 'completed' and metadata['plan_sha256'] == digest(plan_path),
            'Completed run or frozen plan identity changed')
    compare(metadata['objective'], objective, 'Run objective')
    compare(metadata['protocol'], protocol, 'Run protocol')
    compare(metadata['dataset_metadata_sha256'], selection['dataset_metadata_sha256'], 'Dataset metadata hash')
    compare(metadata['manifest_sha256'], selection['manifest_sha256'], 'Dataset manifest hash')
    require(metadata['attempted_batches'] == metadata['optimizer_steps'] == 1664
            and metadata['parameter_count'] == 2545, 'Run attempt or model coverage changed')
    implementation = metadata['implementation_sha256']
    require(isinstance(implementation, dict) and set(implementation) == REQUIRED_SOURCES
            and implementation == plan['implementation_sha256'],
            'Run implementation coverage or frozen pre-run implementation changed')
    for name, checksum in implementation.items():
        require(re.fullmatch(r'[A-Za-z0-9_]+\.py', name) is not None
                and digest(ROOT/'Solutions'/name) == checksum, 'Run implementation changed: '+name)
    labels, tensor_hashes, audit = source_dataset(plan, dataset)
    compare(metadata['tensor_sha256'], tensor_hashes, 'Exact exported tensors')
    compare(metadata['dataset_audit'], audit, 'Independent dataset audit')
    resource = resource_receipt(metadata)
    for field in ('all_ignored_no_forward_no_step', 'ignored_logit_gradients_zero_every_step',
                  'saved_weights_reproduce_final_logits', 'source_dataset_unchanged'):
        require(metadata[field] is True, 'Worker safety or replay receipt failed: '+field)
    compare(metadata['all_ignored_safety_check'],
            {'updated': False, 'reviewed_pixels': 0, 'step_scale': 0.,
             'scaled_loss_before_step': None, 'gradient_norm': 0., 'parameter_delta_l2': 0.,
             'zero_parameter_delta': True, 'ignored_gradient_max_abs': 0.,
             'optimizer_step_called': False}, 'All-ignored skip receipt')
    artifacts = metadata['artifact_sha256']
    expected_artifacts = {'steps.jsonl', 'model_weights.npz', 'report.md',
                          *(f'preview_{page}.png' for page in range(1, 5)),
                          *(f'checkpoint_{attempt:04d}.npz' for attempt in protocol['checkpoint_attempts'])}
    require(set(artifacts) == expected_artifacts, 'Run artifact coverage changed')
    for name, checksum in artifacts.items():
        require(digest(within(directory, name)) == checksum, 'Run artifact changed: '+name)
    steps = list(records(directory/'steps.jsonl', maximum=1664))
    stalls = verify_steps(steps, labels, selection['event_ids'], objective, protocol)
    compare(metadata['zero_parameter_delta_attempts'], stalls, 'Recorded stationary attempts')
    checkpoints = metadata['checkpoints']
    require(len(checkpoints) == 5 and [row['attempt'] for row in checkpoints] == protocol['checkpoint_attempts'],
            'All predeclared checkpoints must remain present')
    rebuilt = []
    for row in checkpoints:
        filename = f'checkpoint_{row["attempt"]:04d}.npz'
        require(row['arrays_file'] == filename, 'Checkpoint array identity changed')
        arrays = checked_npz(within(directory, filename),
                             {'logits': (13, 64, 64), 'probabilities': (13, 64, 64),
                              'labels': (13, 64, 64), 'valid': (13, 64, 64),
                              'event_ids': (13,), 'attempt': ()},
                             {'logits': 'float32', 'probabilities': 'float32', 'labels': 'uint8',
                              'valid': 'bool', 'event_ids': 'unicode', 'attempt': 'int64'})
        require(int(arrays['attempt']) == row['attempt']
                and arrays['event_ids'].tolist() == selection['event_ids']
                and np.array_equal(arrays['labels'], labels)
                and np.array_equal(arrays['valid'], labels != 255),
                'Saved checkpoint event order or supervision changed')
        computed = aggregate_checkpoint(arrays['logits'], arrays['probabilities'], labels,
                                         selection['event_ids'], objective['positive_weight'],
                                         plan['fit_gate']['minimum'])
        compare(row, computed, 'Checkpoint '+str(row['attempt']), tolerance=2e-6)
        rebuilt.append({'attempt': row['attempt'], 'parameter_sha256': row['parameter_sha256'], **computed})
    initial_hash, final_hash = metadata['initial_parameter_sha256'], metadata['final_parameter_sha256']
    require(re.fullmatch(r'[a-f0-9]{64}', initial_hash) is not None
            and initial_hash == checkpoints[0]['parameter_sha256']
            and final_hash == checkpoints[-1]['parameter_sha256'] and initial_hash != final_hash,
            'Initial/final parameter identity changed')
    weights = checked_npz(directory/'model_weights.npz', WEIGHT_SHAPES,
                          {key: 'float32' for key in WEIGHT_SHAPES})
    state_hash = hashlib.sha256()
    for name in sorted(weights):
        require(np.isfinite(weights[name]).all(), 'Final weights must remain finite')
        state_hash.update(name.encode())
        state_hash.update(weights[name].tobytes())
    require(state_hash.hexdigest() == final_hash, 'Safe final weights disagree with recorded parameter hash')
    require(metadata['fit_passed'] is rebuilt[-1]['fit_passed'], 'Primary final fit outcome changed')
    first_fit = next((row['attempt'] for row in rebuilt if row['fit_passed']), None)
    compare(metadata['first_passing_recorded_checkpoint'], first_fit, 'First passing recorded checkpoint')
    require(digest(dataset/'metadata.json') == selection['dataset_metadata_sha256']
            and digest(dataset/'training_manifest.jsonl') == selection['manifest_sha256'],
            'Dataset changed during verification')
    require('torch' not in sys.modules, 'Verifier unexpectedly imported Torch')
    result = {'verification_status': 'passed', 'scope': plan['scope'],
              'fit_passed': rebuilt[-1]['fit_passed'], 'all_fit_outcomes_retained': True,
              'first_passing_predeclared_checkpoint': first_fit,
              'plan_sha256': digest(plan_path), 'run_metadata_sha256': digest(directory/'metadata.json'),
              'dataset_metadata_sha256': selection['dataset_metadata_sha256'],
              'manifest_sha256': selection['manifest_sha256'], 'tensor_sha256': tensor_hashes,
              'objective': objective, 'protocol': protocol, 'checkpoints': rebuilt,
              'zero_parameter_delta_attempts': stalls, 'nonzero_parameter_updates': len(steps)-len(stalls),
              'run_resources': resource, 'artifact_sha256': artifacts,
              'run_implementation_sha256': implementation,
              'verification_implementation_sha256': {name: digest(ROOT/'Solutions'/name) for name in
                                                    ('verify_pixel_pilot.py', 'summarize_pilot_overfit.py',
                                                     'diagnose_pilot_overfit.py')},
              'model_executions': 0, 'source_jp2_pixel_reads': 0, 'torch_imported': False,
              'limitations': ['All scores use reviewed training regions, not unseen locations or ignored pixels.',
                              'Gradient execution and saved-weight forward replay are worker receipts.',
                              'Longer exposure and changed normalization do not isolate the old smoke failure cause.',
                              'No first-visible-date inference or threshold selection was performed.']}
    write_report(output/'report.md', result)
    return result


def write_report(path, result):
    def number(value):
        return 'undefined' if value is None else f'{value:.6f}' if isinstance(value, float) else str(value)
    lines = ['# Verified all-13 reviewed-pixel training diagnostic', '',
             '**Training diagnostics only. No held-out accuracy or temporal inference.**', '',
             f"Independent verification: **{result['verification_status']}**. Primary final fit: **{result['fit_passed']}**.",
             'Every predeclared checkpoint and finite no-fit outcome is retained.', '',
             '| Attempt | Weighted BCE | Normalized BCE | Dice | Precision | Recall | Specificity | Cases passing / 13 | Overall fit |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in result['checkpoints']:
        aggregate = row['aggregate']
        metrics = aggregate['metrics']
        values = [row['attempt'], aggregate['weighted_bce'], aggregate['mean_weight_normalized_bce'],
                  metrics['dice'], metrics['precision'], metrics['recall'], metrics['specificity'],
                  sum(case['fit']['passed'] for case in row['examples']), row['fit_passed']]
        lines.append('| '+' | '.join(map(number, values))+' |')
    lines += ['', 'Final per-example results:', '',
              '| Event | TP / FP / FN / TN | Dice | Precision | Recall | Specificity | Training AUC | Fit |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in result['checkpoints'][-1]['examples']:
        metrics = row['metrics']
        counts = ' / '.join(str(metrics[key]) for key in ('tp', 'fp', 'fn', 'tn'))
        values = [row['event_id'], counts, metrics['dice'], metrics['precision'], metrics['recall'],
                  metrics['specificity'], row['probability_diagnostics']['training_pairwise_auc'], row['fit']['passed']]
        lines.append('| '+' | '.join(map(number, values))+' |')
    resources = result['run_resources']
    lines += ['', f"Training worker: {resources['peak_worker_mib']:.2f} MiB peak, {resources['elapsed_seconds']:.2f}s; "
              f"minimum available memory {resources['minimum_available_gib']:.2f} GiB.",
              'The verifier reconstructs losses from logits with NumPy float64 logaddexp and uses logits >=0 for counts.',
              'Per-image step scale is 13 times the reviewed-pixel count divided by 2791. At fixed parameters, '
              'averaging scaled image losses/gradients yields the full reviewed-pixel objective. Sequential SGD '
              'updates are not equivalent to one full-batch update.', '', *result['limitations']]
    Path(path).write_text('\n'.join(lines).rstrip()+'\n')


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=ROOT/'Docs/Reviews/PixelPilotV1/experiment_plan.json')
    parser.add_argument('--dataset', type=Path, default=ROOT/'Datasets/CuratedPilot/v1/dataset')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=180)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Use 128–512 MiB RSS, >=1.5 GiB reserve and <=300 seconds')
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        os.nice(10)
        metadata = verify(args, args.worker_output)
        import resource
        metadata['worker_high_water_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                                    (2**20 if sys.platform == 'darwin' else 1024), 2)
        (args.worker_output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    else:
        run_guarded(args, script_path=__file__, operation='pixel-pilot-verification',
                    gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
