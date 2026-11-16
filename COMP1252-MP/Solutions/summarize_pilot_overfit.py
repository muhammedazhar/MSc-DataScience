#!/usr/bin/env python3
"""Verify saved one-example diagnostic artifacts without executing a model.

Counts and stable BCE are independently reconstructed from saved logits. A
valid completed arm may fail its fit gate; that outcome remains in the report.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARM_IDS = ('global_weight_sgd', 'local_weight_sgd', 'local_weight_adam')
SOURCE_NAMES = ('pilot_overfit_worker.py', 'diagnose_pilot_overfit.py', 'pilot_dataset.py',
                'pilot_objectives.py', 'train_pilot_baseline.py', 'run_pilot_checks.py',
                'validate_sample.py')
ARTIFACT_NAMES = {'checkpoint_arrays.npz', 'model_weights.npz', 'steps.jsonl',
                  'preview.png', 'report.md'}
CHANNELS = tuple(f'{period}_{band}' for period in ('pre', 'post', 'difference')
                 for band in ('B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12', 'NDVI', 'NDMI'))
WEIGHT_SHAPES = {'0.weight': (8, 27, 3, 3), '0.bias': (8,),
                 '2.weight': (8, 8, 3, 3), '2.bias': (8,),
                 '4.weight': (1, 8, 1, 1), '4.bias': (1,)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path, max_bytes=4*2**20):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= max_bytes, f'Missing/oversized metadata: {path}')
    value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {value}')))
    require(isinstance(value, dict), f'Metadata must be an object: {path}')
    return value


def records(path, maximum=20_000):
    with Path(path).open('rb') as stream:
        for index in range(maximum+1):
            line = stream.readline(512*1024+1)
            if not line:
                return
            require(index < maximum and 0 < len(line) <= 512*1024 and bool(line.strip()), 'JSONL record/line bound exceeded')
            value = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {value}')))
            require(isinstance(value, dict), 'JSONL records must be objects')
            yield value


def within(base, relative):
    require(isinstance(relative, str) and bool(relative) and not Path(relative).is_absolute(), 'Relative artifact path required')
    path = (Path(base)/relative).resolve()
    require(path.is_relative_to(Path(base).resolve()) and path.is_file(), 'Artifact path is missing or escaped its root')
    return path


def compare(actual, expected, label, tolerance=1e-10):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) >= set(expected), f'Missing fields: {label}')
        for key, value in expected.items():
            compare(actual[key], value, label+'.'+key, tolerance)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f'List coverage mismatch: {label}')
        for index, value in enumerate(expected):
            compare(actual[index], value, label+f'[{index}]', tolerance)
    elif type(expected) is float:
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), f'Numeric mismatch: {label}')
    else:
        require(type(actual) is type(expected) and actual == expected, f'Identity/count mismatch: {label}')


def checked_npz(path, shapes, types):
    """Check compressed member/header bounds before allocating small arrays."""
    import numpy as np

    path = Path(path)
    require(path.stat().st_size <= 2*2**20, 'Compressed array artifact exceeds its size bound')
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == {name+'.npy' for name in shapes}, 'Unexpected NPZ member set')
        require(sum(item.file_size for item in archive.infolist()) <= 2*2**20, 'Uncompressed array artifact exceeds its size bound')
        for name, shape in shapes.items():
            with archive.open(name+'.npy') as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'Unsupported NPY header version')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                found_shape, _, dtype = reader(stream)
                require(found_shape == shape and not dtype.hasobject, f'Unsafe NPZ shape/type: {name}')
                if types[name] == 'unicode':
                    require(dtype.kind == 'U' and dtype.itemsize <= 512, 'Unsafe event identity array')
                else:
                    require(dtype == np.dtype(types[name]), f'Unexpected NPZ dtype: {name}')
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in shapes}


def recompute_checkpoint(logits, probabilities, labels, pos_weight, minimum=.95):
    """Independent NumPy counts and stable float64 BCE; no Torch execution."""
    import numpy as np
    from diagnose_pilot_overfit import fit_assessment, probability_diagnostics

    z, probability, target = np.asarray(logits), np.asarray(probabilities), np.asarray(labels)
    require(z.ndim == 2 and z.shape == target.shape == probability.shape
            and 0 < target.size <= 4096, 'Checkpoint shapes must match one small core')
    require(z.dtype.kind == 'f' and probability.dtype.kind == 'f'
            and np.isfinite(z).all() and np.isfinite(probability).all(), 'Checkpoint predictions must remain finite')
    require(target.dtype.kind in 'ui' and np.isin(target, (0, 1, 255)).all(), 'Checkpoint labels must retain 0/1/255')
    require(type(pos_weight) in (int, float) and math.isfinite(pos_weight) and pos_weight > 0, 'Invalid positive weight')
    z64 = z.astype('float64')
    expected_probability = np.empty_like(z64)
    nonnegative = z64 >= 0
    expected_probability[nonnegative] = 1/(1+np.exp(-z64[nonnegative]))
    exp_value = np.exp(z64[~nonnegative])
    expected_probability[~nonnegative] = exp_value/(1+exp_value)
    require(((probability >= 0) & (probability <= 1)).all()
            and np.allclose(probability, expected_probability, rtol=0, atol=2e-7), 'Saved probabilities disagree with saved logits')
    positive, stable, ignored = target == 1, target == 0, target == 255
    p, n = int(positive.sum()), int(stable.sum())
    require(p > 0 and n > 0, 'Both reviewed classes are required')
    prediction = z >= 0
    tp, fp = int((prediction & positive).sum()), int((prediction & stable).sum())
    fn, tn = p-tp, n-fp
    def ratio(a, b):
        return a/b if b else None
    precision, recall, specificity = ratio(tp, tp+fp), tp/p, tn/n
    dice = 2*tp/(2*tp+fp+fn)
    metrics = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'supervised_pixels': p+n,
               'positive_pixels': p, 'stable_pixels': n, 'ignored_pixels': int(ignored.sum()),
               'updates': 1, 'empty_updates': 0, 'probability_threshold': .5,
               'precision': precision, 'recall': recall, 'specificity': specificity,
               'f1': dice, 'dice': dice, 'iou': ratio(tp, tp+fp+fn),
               'accuracy': (tp+tn)/(p+n), 'balanced_accuracy': (recall+specificity)/2}
    reviewed_z, reviewed_y = z64[~ignored], target[~ignored]
    unweighted = np.where(reviewed_y == 1, np.logaddexp(0., -reviewed_z), np.logaddexp(0., reviewed_z))
    weighted = np.where(reviewed_y == 1, pos_weight*np.logaddexp(0., -reviewed_z), np.logaddexp(0., reviewed_z))
    mean_weight = (pos_weight*p+n)/(p+n)
    return {'metrics': metrics, 'fit': fit_assessment(metrics, p, n, minimum),
            'weighted_bce': float(weighted.mean()), 'unweighted_bce': float(unweighted.mean()),
            'mean_weight_normalized_bce': float(weighted.mean()/mean_weight),
            'probability_diagnostics': probability_diagnostics(probability, target)}


def selected_source(plan):
    import numpy as np
    import rasterio
    from affine import Affine

    selection = plan['selection']
    dataset = (ROOT/selection['dataset_root']).resolve()
    require(dataset.is_relative_to(ROOT), 'Selected dataset escaped project')
    require(digest(dataset/'metadata.json') == selection['dataset_metadata_sha256']
            and digest(dataset/'training_manifest.jsonl') == selection['training_manifest_sha256'], 'Frozen dataset metadata/manifest changed')
    row = next(records(dataset/'training_manifest.jsonl', maximum=64))
    for key in ('event_id', 'pair_id', 'group_id', 'split'):
        require(row.get(key) == selection[key], 'First manifest selection changed: '+key)
    require(row['split'] == 'train' and row.get('training_approved') is True, 'Selected example must remain approved training data')
    require(row.get('visibility_evidence', {}).get('decision_status') == 'accepted', 'Selected review acceptance is missing')
    arrays, source_hashes = {}, {}
    grid = row['grid']
    for role, relative in row['artifact_paths'].items():
        path = within(dataset, relative)
        source_hashes[role] = digest(path)
        require(source_hashes[role] == selection['selected_artifacts_sha256'][role], 'Selected artifact hash changed: '+role)
        if role == 'metadata':
            continue
        with rasterio.open(path) as source:
            require(source.shape == (64, 64) and source.transform == Affine(*grid['transform'])
                    and source.crs == rasterio.crs.CRS.from_user_input(grid['crs']), 'Selected TIFF grid changed')
            if role == 'input_features':
                require(source.count == 27 and source.dtypes == ('float32',)*27 and source.descriptions == CHANNELS, 'Selected feature headers changed')
            else:
                require(source.count == 1 and source.dtypes == ('uint8',), 'Selected supervision headers changed')
                if role == 'supervision_mask':
                    require(source.nodata == 255, 'Selected ignore convention changed')
            arrays[role] = source.read()
    require(set(source_hashes) == set(selection['selected_artifacts_sha256']), 'Selected artifact roles changed')
    features, labels, valid = arrays['input_features'], arrays['supervision_mask'][0], arrays['valid_mask'][0]
    require(np.isfinite(features).all() and np.isin(labels, (0, 1, 255)).all()
            and np.isin(valid, (0, 1)).all() and np.array_equal(valid.astype(bool), labels != 255), 'Selected finite inputs or partial supervision changed')
    support = (int((labels == 1).sum()), int((labels == 0).sum()), int((labels == 255).sum()))
    require(support == (selection['positive_pixels'], selection['stable_pixels'], selection['ignored_pixels'])
            == (25, 256, 3815), 'Predeclared selected support changed')
    require(np.array_equal(features[18:], features[9:18]-features[:9]), 'Selected signed differences changed')
    checksum = hashlib.sha256()
    for name, array, torch_dtype in (('features', features, 'torch.float32'),
                                      ('labels', labels, 'torch.uint8'),
                                      ('valid', valid.astype(bool), 'torch.bool')):
        checksum.update(name.encode())
        checksum.update(str(tuple(array.shape)).encode())
        checksum.update(torch_dtype.encode())
        checksum.update(np.ascontiguousarray(array).tobytes())
    return row, dataset, labels, checksum.hexdigest(), source_hashes


def resource_receipt(metadata):
    resource = metadata['resources']
    require(metadata['device'] == 'cpu' and metadata['gpu_used'] is False
            and metadata['batch_size'] == 1 and metadata['dataloader_workers'] == 0
            and metadata['native_threads'] == metadata['torch_intraop_threads'] == metadata['torch_interop_threads'] == 1,
            'Arm device/thread bounds changed')
    require(resource['workers'] == resource['native_thread_limit'] == 1 and resource['gpu_used'] is False
            and resource['nice_increment'] == 10 and resource['worker_limit_mib'] <= 512
            and resource['minimum_available_gib_limit'] >= 1.5
            and 0 < resource['runtime_limit_seconds'] <= 300, 'Arm resource guard limits changed')
    peak = max(resource['worker_peak_rss_mib_sampled'], resource['worker_high_water_rss_mib'])
    require(math.isfinite(peak) and peak <= resource['worker_limit_mib']
            and resource['minimum_available_gib_sampled'] >= resource['minimum_available_gib_limit']
            and resource['elapsed_seconds'] <= resource['runtime_limit_seconds'], 'Arm resource receipt failed')
    return {'peak_worker_mib': peak, 'minimum_available_gib': resource['minimum_available_gib_sampled'],
            'elapsed_seconds': resource['elapsed_seconds']}


def verify_arm(plan, plan_path, root, arm, labels, tensor_hash, source_hashes):
    import numpy as np
    from diagnose_pilot_overfit import checkpoint_steps, constant_optimum

    directory = root/arm['arm_id']
    metadata_path = directory/'metadata.json'
    metadata = read_json(metadata_path)
    selection, protocol = plan['selection'], plan['fixed_protocol']
    require(metadata['execution_status'] == 'completed' and metadata['arm_id'] == arm['arm_id']
            and metadata['arm_configuration'] == arm, 'Completed arm/configuration mismatch')
    # Content hashes bind the plan. Keep the recorded execution path as
    # provenance without requiring a later agent to use the same worktree path.
    require(metadata['plan_sha256'] == digest(plan_path), 'Plan identity changed')
    for field in ('event_id', 'pair_id', 'group_id', 'split'):
        require(metadata[field] == selection[field], 'Selected arm identity changed: '+field)
    require(metadata['selected_manifest_index'] == 0 and metadata['selected_tensor_sha256'] == tensor_hash
            and metadata['selected_artifact_sha256'] == source_hashes
            and metadata['dataset_metadata_sha256'] == selection['dataset_metadata_sha256']
            and metadata['manifest_sha256'] == selection['training_manifest_sha256'], 'Arm data binding changed')
    require((metadata['selected_positive_pixels'], metadata['selected_stable_pixels'], metadata['selected_ignored_pixels']) == (25, 256, 3815), 'Arm support changed')
    require(metadata['seed'] == protocol['seed'] and metadata['model_specification'] == protocol['model']
            and metadata['parameter_count'] == 2545, 'Arm seed/model changed')
    implementation = metadata['implementation_sha256']
    require(set(implementation) == set(SOURCE_NAMES), 'Arm implementation coverage changed')
    for name, checksum in implementation.items():
        require(digest(ROOT/'Solutions'/name) == checksum, 'Implementation changed since arm: '+name)
    artifacts = metadata['artifact_sha256']
    require(set(artifacts) == ARTIFACT_NAMES, 'Arm artifact coverage changed')
    for name, checksum in artifacts.items():
        require(digest(within(directory, name)) == checksum, 'Arm artifact changed: '+name)
    schedule = checkpoint_steps(protocol['attempted_updates_per_arm'])
    require(schedule == protocol['checkpoint_attempts'] == [0, 26, 128, 256, 512], 'Frozen checkpoint schedule changed')
    require(metadata['attempted_batches'] == metadata['optimizer_steps'] == metadata['primary_fit_checkpoint'] == 512
            and metadata['early_stopping_used'] is False and protocol['early_stopping'] is False, 'Fixed attempt/primary gate changed')
    steps = list(records(directory/'steps.jsonl', maximum=512))
    require([row['attempt'] for row in steps] == list(range(1, 513)), 'Attempt log coverage changed')
    for row in steps:
        for field in ('weighted_bce_before_step', 'gradient_norm', 'parameter_delta_l2'):
            require(type(row[field]) in (int, float) and math.isfinite(row[field]) and row[field] >= 0, 'Nonfinite/negative step evidence')
        require(row['optimizer_step_called'] is True and row['ignored_gradient_max_abs'] == 0.
                and type(row['zero_parameter_delta']) is bool
                and row['zero_parameter_delta'] == (row['parameter_delta_l2'] == 0), 'Step/ignored-gradient receipt mismatch')
    stalls = [row['attempt'] for row in steps if row['zero_parameter_delta']]
    require(metadata['zero_parameter_delta_attempts'] == stalls
            and metadata['nonzero_parameter_updates'] == 512-len(stalls) > 0, 'Update/stall summary changed')
    for prefix, key in (('gradient_norm', 'gradient_norm'), ('parameter_delta_l2', 'parameter_delta_l2')):
        compare(metadata[prefix+'_min'], min(row[key] for row in steps), prefix+' minimum')
        compare(metadata[prefix+'_max'], max(row[key] for row in steps), prefix+' maximum')
    require(metadata['all_ignored_no_forward_no_step'] is True
            and metadata['ignored_logit_gradients_zero_every_step'] is True
            and metadata['saved_weights_reproduce_final_logits'] is True
            and metadata['source_dataset_unchanged'] is True and metadata['selected_arrays_read_count'] == 2,
            'Worker safety/source/reload receipts failed')
    compare(metadata['all_ignored_safety_check'], {'updated': False, 'loss': None, 'reviewed_pixels': 0,
                                                 'gradient_norm': 0., 'parameter_delta_l2': 0.}, 'All-ignored receipt')
    arrays = checked_npz(directory/'checkpoint_arrays.npz',
                         {'attempts': (5,), 'logits': (5, 64, 64), 'probabilities': (5, 64, 64),
                          'labels': (64, 64), 'valid': (64, 64), 'event_id': ()},
                         {'attempts': 'int64', 'logits': 'float32', 'probabilities': 'float32',
                          'labels': 'uint8', 'valid': 'bool', 'event_id': 'unicode'})
    require(arrays['attempts'].tolist() == schedule and str(arrays['event_id']) == selection['event_id']
            and np.array_equal(arrays['labels'], labels) and np.array_equal(arrays['valid'], labels != 255), 'Saved checkpoint selection/masks changed')
    checkpoints = metadata['checkpoints']
    require(len(checkpoints) == 5 and [row['attempt'] for row in checkpoints] == schedule, 'Checkpoint metadata coverage changed')
    minimums = [plan['fit_gate'][name] for name in ('requires_precision_at_least', 'requires_recall_at_least',
                                                   'requires_dice_at_least', 'requires_specificity_at_least')]
    require(len(set(minimums)) == 1 and protocol['probability_threshold'] == .5, 'Fit gate changed')
    analytic = constant_optimum(25, 256, arm['positive_weight'])
    compare(metadata['constant_prediction_optimum'], analytic, 'Analytic constant baseline')
    rebuilt = []
    for index, published in enumerate(checkpoints):
        computed = recompute_checkpoint(arrays['logits'][index], arrays['probabilities'][index], labels,
                                        arm['positive_weight'], minimums[0])
        for field in ('metrics', 'fit', 'probability_diagnostics'):
            compare(published[field], computed[field], f'{arm["arm_id"]}/{schedule[index]}/{field}')
        for field in ('weighted_bce', 'unweighted_bce', 'mean_weight_normalized_bce'):
            compare(published[field], computed[field], f'{arm["arm_id"]}/{schedule[index]}/{field}', tolerance=2e-6)
        rebuilt.append({'attempt': schedule[index], **computed, 'parameter_sha256': published['parameter_sha256']})
    initial_hash, final_hash = metadata['initial_parameter_sha256'], metadata['final_parameter_sha256']
    require(initial_hash == checkpoints[0]['parameter_sha256'] and final_hash == checkpoints[-1]['parameter_sha256']
            and final_hash != initial_hash, 'Initial/final parameter state receipts changed')
    weights = checked_npz(directory/'model_weights.npz', WEIGHT_SHAPES, {key: 'float32' for key in WEIGHT_SHAPES})
    state_hash = hashlib.sha256()
    for name in sorted(weights):
        require(np.isfinite(weights[name]).all(), 'Saved weights contain nonfinite values')
        state_hash.update(name.encode())
        state_hash.update(weights[name].tobytes())
    require(state_hash.hexdigest() == final_hash, 'Saved final parameter hash changed')
    first_fit = next((row['attempt'] for row in rebuilt if row['fit']['passed']), None)
    require(metadata['fit_passed'] is rebuilt[-1]['fit']['passed']
            and metadata['first_passing_predeclared_checkpoint'] == first_fit, 'Fit outcome/checkpoint selection changed')
    compare(metadata['final_metrics'], rebuilt[-1]['metrics'], 'Final confusion evidence')
    compare(metadata['final_fit'], rebuilt[-1]['fit'], 'Final fit evidence')
    result = {'arm_id': arm['arm_id'], 'arm_configuration': arm, 'fit_passed': rebuilt[-1]['fit']['passed'],
              'first_passing_predeclared_checkpoint': first_fit, 'checkpoints': rebuilt,
              'nonzero_parameter_updates': metadata['nonzero_parameter_updates'], 'zero_parameter_delta_attempts': stalls,
              'resources': resource_receipt(metadata), 'run_metadata_sha256': digest(metadata_path),
              'artifact_sha256': artifacts, 'initial_parameter_sha256': initial_hash,
              'final_parameter_sha256': final_hash, 'selected_tensor_sha256': tensor_hash,
              'implementation_sha256': implementation, 'torch_version': metadata['torch_version'],
              'torch_distribution_version': metadata['torch_distribution_version'],
              'torch_git_version': metadata['torch_git_version'], 'run_directory': str(directory.resolve())}
    return result, arrays['logits'][0].copy(), arrays['probabilities'][0].copy()


def summarize(args, output):
    import numpy as np

    require('torch' not in sys.modules, 'The summary process must not import or execute Torch')
    plan_path = args.plan.resolve()
    plan = read_json(plan_path)
    require([arm['arm_id'] for arm in plan['arms']] == list(ARM_IDS), 'Exactly the three predeclared arms are required')
    require(plan['selection']['split'] == 'train', 'Diagnostic selection is not training-only')
    row, dataset, labels, tensor_hash, source_hashes = selected_source(plan)
    results, initial_logits, initial_probabilities = [], None, None
    for arm in plan['arms']:
        result, logits, probability = verify_arm(plan, plan_path, args.runs_root.resolve(), arm, labels, tensor_hash, source_hashes)
        if results:
            require(result['initial_parameter_sha256'] == results[0]['initial_parameter_sha256']
                    and result['implementation_sha256'] == results[0]['implementation_sha256']
                    and result['selected_tensor_sha256'] == results[0]['selected_tensor_sha256'], 'Cross-arm initial/source identity changed')
            require(np.array_equal(logits, initial_logits) and np.array_equal(probability, initial_probabilities), 'Initial logits/probabilities differ across arms')
            for field in ('torch_version', 'torch_distribution_version', 'torch_git_version'):
                require(result[field] == results[0][field], 'Cross-arm runtime changed')
        else:
            initial_logits, initial_probabilities = logits, probability
        results.append(result)
    require(digest(dataset/'metadata.json') == plan['selection']['dataset_metadata_sha256']
            and digest(dataset/'training_manifest.jsonl') == plan['selection']['training_manifest_sha256'], 'Source dataset changed during summary')
    require('torch' not in sys.modules, 'Summary unexpectedly imported Torch')
    metadata = {'verification_status': 'passed', 'scope': plan['scope'], 'arm_count': len(results),
                'all_fit_outcomes_retained': True, 'fitted_arm_count': sum(r['fit_passed'] for r in results),
                'event_id': row['event_id'], 'pair_id': row['pair_id'], 'group_id': row['group_id'], 'split': 'train',
                'support': {'positive': 25, 'stable': 256, 'ignored': 3815}, 'selected_tensor_sha256': tensor_hash,
                'selected_artifacts_sha256': source_hashes, 'plan_sha256': digest(plan_path),
                'dataset_metadata_sha256': plan['selection']['dataset_metadata_sha256'],
                'manifest_sha256': plan['selection']['training_manifest_sha256'],
                'summary_implementation_sha256': digest(__file__), 'arms': results,
                'model_executions': 0, 'source_jp2_pixel_reads': 0, 'torch_imported': False,
                'checks': {'exact_three_completed_arms': True, 'identical_initial_states_and_predictions': True,
                           'frozen_source_plan_and_tensor_identities': True, 'all_512_step_receipts_retained': True,
                           'all_five_checkpoint_counts_and_losses_recomputed': True,
                           'fit_probability_distribution_and_auc_agree': True,
                           'finite_update_stall_and_ignored_gradient_receipts_consistent': True,
                           'source_tiff_grid_labels_validity_and_hashes_agree': True,
                           'safe_final_weight_arrays_match_recorded_parameter_hash': True,
                           'resource_limits_respected': True},
                'limitations': ['Model gradients and saved-weight forward replay are worker receipts, not re-executed by this verifier.',
                                'The selected training crop is not held-out evaluation or a globally valid forest-loss map.',
                                'Adam changes optimizer and learning rate together; differences cannot be assigned solely to optimizer.',
                                'A finite no-fit arm is retained as a valid negative diagnostic result.']}
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    write_report(output/'report.md', metadata, plan)


def write_report(path, metadata, plan):
    def number(value):
        return 'undefined' if value is None else f'{value:.6f}' if isinstance(value, float) else str(value)
    lines = ['# Verified single-example overfit diagnostic', '',
             f"Verified all **{metadata['arm_count']}** completed arms and retained every fit outcome; **{metadata['fitted_arm_count']}** pass the predeclared final-512 fit gate.",
             '**This is one-crop training memorization, not held-out accuracy, deployment readiness or first-visible-date inference.**', '',
             f"Selected event: `{metadata['event_id']}`; reviewed support **25 clearing / 256 forest / 3,815 ignored pixels**. All arms share identical initial tensors, weights and predictions.", '',
             '| Arm | Positive weight | Optimizer / rate | Final fit | Dice | Precision | Recall | Specificity | Normalized BCE | Training AUC | First passing checkpoint |',
             '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in metadata['arms']:
        final = arm['checkpoints'][-1]
        metric, config = final['metrics'], arm['arm_configuration']
        lines.append(f"| `{arm['arm_id']}` | {number(config['positive_weight'])} | {config['optimizer']} / {config['learning_rate']} | {arm['fit_passed']} | {number(metric['dice'])} | {number(metric['precision'])} | {number(metric['recall'])} | {number(metric['specificity'])} | {number(final['mean_weight_normalized_bce'])} | {number(final['probability_diagnostics']['training_pairwise_auc'])} | {number(arm['first_passing_predeclared_checkpoint'])} |")
    lines += ['', 'The primary result is the fixed final checkpoint. Earlier passing checkpoints are descriptive; no arm stopped early or substituted a best checkpoint.', '',
              '## Predeclared checkpoint curves', '',
              '| Arm | Attempt | Weighted BCE | Normalized BCE | Dice | Precision | Recall | Specificity | Training AUC | Fit |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for arm in metadata['arms']:
        for checkpoint in arm['checkpoints']:
            metric = checkpoint['metrics']
            lines.append(f"| `{arm['arm_id']}` | {checkpoint['attempt']} | {number(checkpoint['weighted_bce'])} | {number(checkpoint['mean_weight_normalized_bce'])} | {number(metric['dice'])} | {number(metric['precision'])} | {number(metric['recall'])} | {number(metric['specificity'])} | {number(checkpoint['probability_diagnostics']['training_pairwise_auc'])} | {checkpoint['fit']['passed']} |")
    lines += ['', '## Evidence and resources', '']
    for arm in metadata['arms']:
        directory = Path(arm['run_directory'])
        resource = arm['resources']
        lines += [f"- `{arm['arm_id']}`: [actual probability preview](<{directory/'preview.png'}>), [worker metadata](<{directory/'metadata.json'}>); peak {resource['peak_worker_mib']:.2f} MiB, {resource['elapsed_seconds']:.2f}s, {len(arm['zero_parameter_delta_attempts'])} stationary parameter attempts."]
    analytic = plan['analytic_constant_prediction_baselines']
    lines += ['', '## Interpretation', '',
              f"The selected-case constant-predictor optimum is {analytic['single_case_global_weight_p_star']:.5f} under the global weight and {analytic['single_case_local_weight_p_star']:.5f} under the local weight. The earlier full-pilot equal-image objective has optimum {analytic['full_pilot_equal_image_mean_objective_p_star']:.5f}, whereas its pixel-aggregated counterpart has optimum 0.5. This analytic positive-prior bias is context, not an empirical proof of the cause of the earlier all-positive outputs.",
              'The two SGD arms change only class weight for this fixed example. The Adam arm changes optimizer **and learning rate**, so an observed difference is not an isolated optimizer effect. Checkpoint exposure on one case does not directly reproduce the earlier 13-case smoke.',
              'BCE is independently recomputed from saved logits with stable float64 logaddexp. Counts use logits >=0, avoiding probability-rounding threshold ambiguity. Normalized BCE divides raw mean weighted BCE by its mean weight mass; no common raw-BCE cutoff is used.',
              'All 512 step receipts, zero ignored-logit gradients and allowed stationary deltas are retained. Gradient execution and Torch saved-weight replay remain validated worker receipts; this summary executes no model and reads no JP2 pixels.',
              'Unreviewed pixels remain ignored. Passing memorization proves neither generalization nor precise clearing dates. A finite completed no-fit arm is a valid negative result and is not removed.', '']
    Path(path).write_text('\n'.join(lines).rstrip()+'\n')


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'Docs/Reviews/OverfitDiagnosticV1/experiment_plan.json')
    parser.add_argument('--runs-root', type=Path, default=ROOT/'Docs/Reviews/OverfitDiagnosticV1/runs')
    parser.add_argument('--output', type=Path, default=ROOT/'Datasets/CuratedPilot/v1/overfit_diagnostic/summary')
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=180)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300, 'Unsafe resource settings')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        summarize(args, args.worker_output)
        import resource
        path = args.worker_output/'metadata.json'
        metadata = read_json(path)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata['worker_high_water_rss_mib'] = round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2)
        path.write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    else:
        run_guarded(args, script_path=__file__, operation='overfit-summary', gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
