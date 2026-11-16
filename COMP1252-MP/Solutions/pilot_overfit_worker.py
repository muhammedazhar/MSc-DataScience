"""Single guarded worker for the predeclared one-example overfit arms.

Only this module executes model operations. Importing it is lightweight; the
root CLI enforces process resources and performs atomic output publication.
"""

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tensor_hash(sample):
    checksum = hashlib.sha256()
    for name in ('features', 'labels', 'valid'):
        tensor = sample[name].detach().cpu().contiguous()
        checksum.update(name.encode())
        checksum.update(str(tuple(tensor.shape)).encode())
        checksum.update(str(tensor.dtype).encode())
        checksum.update(tensor.numpy().tobytes())
    return checksum.hexdigest()


class _CountingOptimizer:
    """Expose actual calls while retaining the ordinary Torch optimizer."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_calls = 0

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.step_calls += 1
        return self.optimizer.step()


def _optimizer(torch, model, arm):
    common = {'lr': arm['learning_rate'], 'weight_decay': arm.get('weight_decay', 0.),
              'foreach': False, 'fused': False}
    if arm['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), momentum=arm.get('momentum', 0.), **common)
    elif arm['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), betas=tuple(arm['betas']),
                                     eps=arm['epsilon'], **common)
    else:
        raise ValueError('Only the predeclared SGD and Adam arms are supported')
    return _CountingOptimizer(optimizer)


def _finite_optimizer(torch, optimizer):
    for state in optimizer.optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                _require(value.device.type == 'cpu' and bool(torch.isfinite(value).all()),
                         'Optimizer state must remain finite and on CPU')
            elif isinstance(value, (float, int)):
                _require(math.isfinite(value), 'Optimizer scalar state must remain finite')


def run_arm(args, output):
    """Return complete arm metadata; the caller writes RSS and report receipts.

    All 512 attempts run even if a checkpoint passes its memorization gate.
    A finite zero parameter delta near convergence is logged as a stall, not
    promoted to an update or treated as a fit/process failure.
    """
    from run_pilot_checks import configure_cpu
    from diagnose_pilot_overfit import (checkpoint_steps, constant_optimum, fit_assessment,
                                      probability_diagnostics, select_record)

    output = Path(output)
    plan_path, dataset_root = Path(args.plan).resolve(), Path(args.dataset).resolve()
    plan = json.loads(plan_path.read_text())
    selection, protocol = plan['selection'], plan['fixed_protocol']
    arm_matches = [item for item in plan['arms'] if item['arm_id'] == args.arm]
    _require(len(arm_matches) == 1, 'Arm must occur exactly once in the frozen plan')
    arm = arm_matches[0]
    limit = protocol['attempted_updates_per_arm']
    schedule = checkpoint_steps(limit)
    _require(limit == 512 and schedule == protocol['checkpoint_attempts'],
             'Attempt/checkpoint schedule differs from the predeclared diagnostic')
    _require(protocol['probability_threshold'] == .5 and protocol['early_stopping'] is False,
             'Diagnostic threshold and no-early-stop rule must remain frozen')
    torch = configure_cpu(protocol['seed'])
    import numpy as np
    from pilot_dataset import ReviewedPilotDataset
    from pilot_objectives import ConfusionCounts, masked_bce_with_logits
    from train_pilot_baseline import make_model, parameter_hash, render_preview, train_step

    source_names = ('pilot_overfit_worker.py', 'diagnose_pilot_overfit.py', 'pilot_dataset.py',
                    'pilot_objectives.py', 'train_pilot_baseline.py', 'run_pilot_checks.py',
                    'validate_sample.py')
    implementation = {name: _digest(ROOT/'Solutions'/name) for name in source_names}
    plan_sha = _digest(plan_path)
    dataset = ReviewedPilotDataset(dataset_root)
    _require(dataset.metadata_sha256 == selection['dataset_metadata_sha256']
             and dataset.manifest_sha256 == selection['training_manifest_sha256'],
             'The selected dataset/manifest changed after experiment declaration')
    index = select_record(dataset.records, selection['event_id'])
    _require(index == 0, 'The diagnostic must retain the first manifest case')
    record = dataset.records[index]
    for name in ('event_id', 'pair_id', 'group_id', 'split'):
        _require(record[name] == selection[name], 'Frozen selected identity changed: '+name)
    expected_artifacts = selection['selected_artifacts_sha256']
    artifact_hashes = {role: _digest(dataset_root/name) for role, name in record['artifact_paths'].items()}
    _require(artifact_hashes == expected_artifacts, 'Selected artifact hashes differ from the plan')
    raw = dataset[index]
    original_tensor_hash = _tensor_hash(raw)
    _require(raw['nonfinite_replacements'] == 0, 'The fixed diagnostic case must retain its exact finite inputs')
    sample = {'features': raw['features'].unsqueeze(0), 'labels': raw['labels'].unsqueeze(0),
              'valid': raw['valid'].unsqueeze(0)}
    _require(list(sample['features'].shape) == selection['features_shape'], 'Selected feature shape changed')
    _require(all(t.device.type == 'cpu' for t in sample.values()), 'Diagnostic tensors must remain on CPU')
    positives = int((sample['labels'] == 1).sum().item())
    negatives = int((sample['labels'] == 0).sum().item())
    ignored_pixels = int((sample['labels'] == 255).sum().item())
    _require((positives, negatives, ignored_pixels) ==
             (selection['positive_pixels'], selection['stable_pixels'], selection['ignored_pixels']),
             'Reviewed support differs from the predeclared case')
    weight = arm['positive_weight']
    analytic = constant_optimum(positives, negatives, weight)
    minimums = [plan['fit_gate'][key] for key in ('requires_precision_at_least',
                 'requires_recall_at_least', 'requires_dice_at_least', 'requires_specificity_at_least')]
    _require(len(set(minimums)) == 1, 'This diagnostic expects one fixed minimum for all fit scores')
    minimum = minimums[0]
    # No DataLoader/random augmentation is used. Reset immediately before model
    # creation so all independent arm workers produce identical initial state.
    torch.manual_seed(protocol['seed'])
    model = make_model()
    _require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()),
             'Diagnostic model must retain float32 CPU parameters')
    initial_hash = parameter_hash(model)
    optimizer = _optimizer(torch, model, arm)

    # Reuse the tested old skip implementation only for the explicit safety
    # exercise. Its strict nonzero-per-step rule is not used in the overfit loop.
    ignored = {**sample, 'labels': torch.full_like(sample['labels'], 255),
               'valid': torch.zeros_like(sample['valid'])}
    forward_calls = []
    handle = model.register_forward_hook(lambda *values: forward_calls.append(1))
    try:
        skip = train_step(model, optimizer, ignored, weight)
    finally:
        handle.remove()
    _require(not skip['updated'] and not forward_calls and optimizer.step_calls == 0
             and parameter_hash(model) == initial_hash, 'All-ignored skip changed model or optimizer execution')

    checkpoint_rows, checkpoint_logits, checkpoint_probabilities = [], [], []
    target_numpy = raw['labels'].numpy().copy()

    def checkpoint(attempt):
        model.eval()
        with torch.inference_mode():
            logits = model(sample['features'])
            _require(logits.device.type == 'cpu' and bool(torch.isfinite(logits).all()),
                     'Checkpoint logits must remain finite and on CPU')
            counter = ConfusionCounts(threshold=.5)
            count = counter.update(logits, sample['labels'], sample['valid'])
            _require(count == positives+negatives, 'Checkpoint reviewed support changed')
            metrics = counter.summary()
            weighted = float(masked_bce_with_logits(logits, sample['labels'], sample['valid'], weight))
            unweighted = float(masked_bce_with_logits(logits, sample['labels'], sample['valid']))
            _require(math.isfinite(weighted) and math.isfinite(unweighted), 'Checkpoint loss must be finite')
            probabilities = torch.sigmoid(logits[0, 0]).numpy().copy()
            logit_array = logits[0, 0].numpy().copy()
        fit = fit_assessment(metrics, positives, negatives, minimum=minimum)
        row = {'attempt': attempt, 'metrics': metrics, 'fit': fit,
               'weighted_bce': weighted, 'unweighted_bce': unweighted,
               'mean_weight_normalized_bce': weighted/analytic['mean_weight'],
               'probability_diagnostics': probability_diagnostics(probabilities, target_numpy),
               'parameter_sha256': parameter_hash(model)}
        checkpoint_rows.append(row)
        checkpoint_logits.append(logit_array)
        checkpoint_probabilities.append(probabilities)
        model.train()
        return row

    checkpoint(0)
    steps = []
    for attempt in range(1, limit+1):
        optimizer.zero_grad(set_to_none=True)
        before = [p.detach().clone() for p in model.parameters()]
        logits = model(sample['features'])
        _require(logits.device.type == 'cpu' and bool(torch.isfinite(logits).all()),
                 'Training logits must remain finite and on CPU')
        logits.retain_grad()
        loss = masked_bce_with_logits(logits, sample['labels'], sample['valid'], weight)
        _require(loss is not None and bool(torch.isfinite(loss)), 'Training loss must remain finite')
        loss.backward()
        _require(logits.grad is not None and bool(torch.isfinite(logits.grad).all()),
                 'Logit gradients must be finite')
        ignored_grad = logits.grad[:, 0][~sample['valid']]
        _require(bool((ignored_grad == 0).all()), 'Ignored pixels received logit gradients')
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        _require(bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients),
                 'Parameter gradients must be finite')
        gradient_norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients))
        _require(math.isfinite(gradient_norm), 'Gradient norm must be finite')
        optimizer.step()
        _finite_optimizer(torch, optimizer)
        after = list(model.parameters())
        _require(all(bool(torch.isfinite(p).all()) for p in after), 'Updated parameters must be finite')
        delta = math.sqrt(sum(float((p.detach()-old).double().square().sum()) for p, old in zip(after, before)))
        _require(math.isfinite(delta), 'Parameter delta must be finite')
        steps.append({'attempt': attempt, 'weighted_bce_before_step': float(loss.detach()),
                      'gradient_norm': gradient_norm, 'parameter_delta_l2': delta,
                      'zero_parameter_delta': delta == 0., 'ignored_gradient_max_abs': 0.0,
                      'optimizer_step_called': True})
        if attempt in schedule:
            row = checkpoint(attempt)
            print(f"[pilot-overfit] {args.arm} attempt {attempt}/{limit}; Dice={row['metrics']['dice']}; fit={row['fit']['passed']}", flush=True)

    final_hash = parameter_hash(model)
    _require(final_hash != initial_hash and any(s['parameter_delta_l2'] > 0 for s in steps),
             'The arm never changed its initial parameters')
    _require(optimizer.step_calls == limit and len(steps) == limit,
             'The fixed arm must retain every attempted optimizer step')
    _require([row['attempt'] for row in checkpoint_rows] == schedule, 'Checkpoint coverage changed')
    _require(_tensor_hash(raw) == original_tensor_hash, 'Cached selected tensors changed during training')
    reread = dataset[index]
    _require(_tensor_hash(reread) == original_tensor_hash and reread['nonfinite_replacements'] == 0,
             'Reread selected arrays differ from the original training sample')
    _require(_digest(dataset_root/'metadata.json') == selection['dataset_metadata_sha256']
             and _digest(dataset_root/'training_manifest.jsonl') == selection['training_manifest_sha256'],
             'Dataset metadata or manifest changed during the arm')
    _require({role: _digest(dataset_root/name) for role, name in record['artifact_paths'].items()} == artifact_hashes,
             'Selected artifact files changed during the arm')
    _require(_digest(plan_path) == plan_sha, 'Predeclared experiment plan changed during execution')
    _require({name: _digest(ROOT/'Solutions'/name) for name in source_names} == implementation,
             'Experiment implementation changed during the arm')

    np.savez_compressed(output/'checkpoint_arrays.npz', attempts=np.asarray(schedule, dtype='int64'),
                        logits=np.stack(checkpoint_logits), probabilities=np.stack(checkpoint_probabilities),
                        labels=target_numpy, valid=raw['valid'].numpy().copy(),
                        event_id=np.asarray(record['event_id']))
    weights = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    np.savez_compressed(output/'model_weights.npz', **weights)
    restored = make_model()
    with np.load(output/'model_weights.npz', allow_pickle=False) as archive:
        restored.load_state_dict({name: torch.from_numpy(archive[name].copy()) for name in archive.files})
    restored.eval()
    with torch.inference_mode():
        restored_logits = restored(sample['features'])[0, 0].numpy()
    _require(np.array_equal(restored_logits, checkpoint_logits[-1]),
             'Saved weights did not exactly reproduce final logits')
    _require(parameter_hash(restored) == final_hash, 'Restored parameter state changed')
    (output/'steps.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in steps))
    render_preview(output/'preview.png', [raw], [checkpoint_probabilities[-1]])
    final = checkpoint_rows[-1]
    grad_norms = [s['gradient_norm'] for s in steps]
    deltas = [s['parameter_delta_l2'] for s in steps]
    first_fit = next((r['attempt'] for r in checkpoint_rows if r['fit']['passed']), None)
    return {
        'execution_status': 'completed', 'fit_passed': final['fit']['passed'],
        'arm_id': args.arm, 'arm_configuration': arm,
        'scope': plan['scope'], 'event_id': record['event_id'], 'pair_id': record['pair_id'],
        'group_id': record['group_id'], 'split': 'train', 'selected_manifest_index': index,
        'selected_positive_pixels': positives, 'selected_stable_pixels': negatives,
        'selected_ignored_pixels': ignored_pixels, 'selected_tensor_sha256': original_tensor_hash,
        'selected_artifact_sha256': artifact_hashes,
        'dataset_metadata_sha256': dataset.metadata_sha256, 'manifest_sha256': dataset.manifest_sha256,
        'dataset_root': str(dataset_root), 'plan_path': str(plan_path), 'plan_sha256': plan_sha,
        'seed': protocol['seed'], 'device': 'cpu', 'gpu_used': False, 'batch_size': 1,
        'dataloader_workers': 0, 'native_threads': 1,
        'torch_intraop_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
        'model_specification': protocol['model'], 'parameter_count': sum(p.numel() for p in model.parameters()),
        'optimizer_implementation': 'Ordinary CPU for-loop optimizer; foreach=False,fused=False',
        'attempted_batches': len(steps), 'optimizer_steps': optimizer.step_calls,
        'nonzero_parameter_updates': sum(delta > 0 for delta in deltas),
        'zero_parameter_delta_attempts': [s['attempt'] for s in steps if s['zero_parameter_delta']],
        'gradient_norm_min': min(grad_norms), 'gradient_norm_max': max(grad_norms),
        'parameter_delta_l2_min': min(deltas), 'parameter_delta_l2_max': max(deltas),
        'all_ignored_no_forward_no_step': True, 'all_ignored_safety_check': skip,
        'ignored_logit_gradients_zero_every_step': True,
        'initial_parameter_sha256': initial_hash, 'final_parameter_sha256': final_hash,
        'constant_prediction_optimum': analytic,
        'checkpoints': checkpoint_rows, 'final_metrics': final['metrics'], 'final_fit': final['fit'],
        'first_passing_predeclared_checkpoint': first_fit,
        'primary_fit_checkpoint': limit, 'early_stopping_used': False,
        'saved_weights_reproduce_final_logits': True, 'selected_arrays_read_count': 2,
        'source_dataset_unchanged': True, 'implementation_sha256': implementation,
        'torch_version': str(torch.__version__), 'torch_distribution_version': importlib.metadata.version('torch'),
        'torch_git_version': torch.version.git_version, 'torch_module_path': str(Path(torch.__file__).resolve()),
        'numpy_version': np.__version__, 'rasterio_version': importlib.metadata.version('rasterio'),
        'python_version': sys.version, 'python_executable': sys.executable,
        'artifact_sha256': {path.name: _digest(path) for path in output.iterdir() if path.is_file()},
        'limitations': plan['interpretation_limits'],
    }
