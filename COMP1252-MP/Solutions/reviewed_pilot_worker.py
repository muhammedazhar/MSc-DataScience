"""One bounded CPU worker; all training is restricted to the reviewed exports."""

import importlib.metadata
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = ('reviewed_pilot_worker.py', 'train_reviewed_pilot.py', 'pilot_pixel_objective.py',
                'pilot_dataset.py', 'pilot_objectives.py', 'train_pilot_baseline.py',
                'pilot_overfit_worker.py', 'diagnose_pilot_overfit.py', 'run_pilot_checks.py',
                'validate_sample.py')


def train_step(model, optimizer, sample, pos_weight, *, image_count, total_reviewed_pixels):
    """Scale one image's loss; preserve skip safety and inspect ignored gradients."""
    import torch
    from diagnose_pilot_overfit import require
    from pilot_objectives import _reviewed
    from pilot_pixel_objective import pixel_aggregate_step_loss, pixel_step_scale
    from pilot_overfit_worker import _finite_optimizer

    features, labels, valid = (sample[k] for k in ('features', 'labels', 'valid'))
    require(all(isinstance(t, torch.Tensor) and t.device.type == 'cpu' for t in (features, labels, valid)),
            'Training requires CPU tensors')
    require(features.shape == (1, 27, 64, 64) and features.dtype == torch.float32,
            'Preserve the original single-image feature grid')
    require(all(p.device.type == 'cpu' and bool(torch.isfinite(p).all()) for p in model.parameters())
            and all(b.device.type == 'cpu' for b in model.buffers()), 'Model state must remain finite on CPU')
    _, _, mask = _reviewed(torch.zeros_like(labels, dtype=torch.float32), labels, valid)
    count = int(mask.sum().item())
    scale = pixel_step_scale(count, image_count, total_reviewed_pixels)
    optimizer.zero_grad(set_to_none=True)
    if not count:
        return {'updated': False, 'reviewed_pixels': 0, 'step_scale': scale,
                'scaled_loss_before_step': None, 'gradient_norm': 0., 'parameter_delta_l2': 0.,
                'zero_parameter_delta': True, 'ignored_gradient_max_abs': 0., 'optimizer_step_called': False}
    require(bool(torch.isfinite(features).all()), 'All input context must be finite')
    before = [p.detach().clone() for p in model.parameters()]
    logits = model(features)
    require(logits.device.type == 'cpu' and bool(torch.isfinite(logits).all()), 'Predictions must remain finite on CPU')
    logits.retain_grad()
    loss = pixel_aggregate_step_loss(logits, labels, valid, pos_weight,
                                   image_count=image_count, total_reviewed_pixels=total_reviewed_pixels)
    require(loss is not None and bool(torch.isfinite(loss)), 'Training loss must remain finite')
    loss.backward()
    require(logits.grad is not None and bool(torch.isfinite(logits.grad).all())
            and bool((logits.grad[:, 0][~valid] == 0).all()), 'Ignored logit gradient must be exactly zero')
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    require(bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients), 'Gradients must be finite')
    norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients))
    optimizer.step()
    _finite_optimizer(torch, optimizer)
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Updated parameters must be finite')
    delta = math.sqrt(sum(float((p.detach()-old).double().square().sum()) for p, old in zip(model.parameters(), before)))
    require(math.isfinite(norm) and math.isfinite(delta), 'Training diagnostics must be finite')
    return {'updated': True, 'reviewed_pixels': count, 'step_scale': scale,
            'scaled_loss_before_step': float(loss.detach()), 'gradient_norm': norm,
            'parameter_delta_l2': delta, 'zero_parameter_delta': delta == 0.,
            'ignored_gradient_max_abs': 0., 'optimizer_step_called': True}


def run(args, output):
    from run_pilot_checks import configure_cpu
    from diagnose_pilot_overfit import require, fit_assessment, probability_diagnostics
    from train_reviewed_pilot import validate_plan
    from pilot_overfit_worker import _digest, _tensor_hash, _optimizer

    plan_path, data = Path(args.plan).resolve(), Path(args.dataset).resolve()
    plan = json.loads(plan_path.read_text()); validate_plan(plan)
    selection, objective, protocol = (plan[k] for k in ('selection', 'objective', 'protocol'))
    implementation = {name: _digest(ROOT/'Solutions'/name) for name in SOURCE_NAMES}
    require(implementation == plan['implementation_sha256'], 'Implementation differs from the pre-run frozen plan')
    plan_sha = _digest(plan_path)
    torch = configure_cpu(protocol['seed'])
    import numpy as np
    from pilot_dataset import ReviewedPilotDataset
    from pilot_objectives import ConfusionCounts, masked_bce_with_logits
    from train_pilot_baseline import make_model, parameter_hash, render_preview

    dataset = ReviewedPilotDataset(data)
    require(dataset.metadata_sha256 == selection['dataset_metadata_sha256']
            and dataset.manifest_sha256 == selection['manifest_sha256'], 'Frozen dataset metadata/manifest changed')
    require([r['event_id'] for r in dataset.records] == selection['event_ids']
            and [r['group_id'] for r in dataset.records] == selection['group_ids'], 'Reviewed manifest coverage/order changed')
    require(dataset.qa_policy_id == selection['qa_policy_id']
            and dataset._hashes == selection['artifact_sha256'], 'Frozen QA/artifacts changed')
    raw = [dataset[i] for i in range(len(dataset))]
    require(all(r['nonfinite_replacements'] == 0 for r in raw), 'Retain exact finite exported tensors')
    tensors = [_tensor_hash(r) for r in raw]
    samples = [{key: r[key].unsqueeze(0) for key in ('features', 'labels', 'valid')} for r in raw]
    audit = dataset.audit()
    require((audit['supervised_positive_pixels'], audit['supervised_stable_pixels'], audit['ignored_pixels'])
            == (151, 2640, 50457), 'Reviewed support changed')
    # Reset directly before model construction; no DataLoader/RNG consumer.
    torch.manual_seed(protocol['seed'])
    model = make_model()
    require(sum(p.numel() for p in model.parameters()) == protocol['parameter_count'], 'Model architecture changed')
    initial_hash = parameter_hash(model)
    optimizer = _optimizer(torch, model, protocol['optimizer'])
    weight, total, count = objective['positive_weight'], objective['total_reviewed_pixels'], len(raw)
    minimum = plan['fit_gate']['minimum']
    kwargs = {'image_count': count, 'total_reviewed_pixels': total}
    ignored = {**samples[0], 'labels': torch.full_like(samples[0]['labels'], 255),
               'valid': torch.zeros_like(samples[0]['valid'])}
    forward_calls = []
    handle = model.register_forward_hook(lambda *values: forward_calls.append(1))
    try:
        skip = train_step(model, optimizer, ignored, weight, **kwargs)
    finally:
        handle.remove()
    require(not skip['updated'] and not forward_calls and optimizer.step_calls == 0
            and parameter_hash(model) == initial_hash, 'Ignored-only step changed execution/state')
    checkpoints = []
    labels = np.stack([r['labels'].numpy() for r in raw]); valid = labels != 255
    mean_weight = (weight*151+2640)/total

    def checkpoint(attempt):
        model.eval(); pooled = ConfusionCounts(); examples = []; logits_list = []; probabilities = []
        weighted, unweighted = 0., 0.
        with torch.inference_mode():
            for index, sample in enumerate(samples):
                logits = model(sample['features'])
                require(logits.device.type == 'cpu' and bool(torch.isfinite(logits).all()), 'Checkpoint predictions must be finite on CPU')
                n = pooled.update(logits, sample['labels'], sample['valid'])
                counter = ConfusionCounts(); counter.update(logits, sample['labels'], sample['valid'])
                metrics = counter.summary()
                bce = float(masked_bce_with_logits(logits, sample['labels'], sample['valid'], weight))
                plain = float(masked_bce_with_logits(logits, sample['labels'], sample['valid']))
                weighted += bce*n; unweighted += plain*n
                z = logits[0, 0].numpy().copy(); p = torch.sigmoid(logits[0, 0]).numpy().copy()
                logits_list.append(z); probabilities.append(p)
                examples.append({'event_id': selection['event_ids'][index], 'metrics': metrics,
                    'fit': fit_assessment(metrics, metrics['positive_pixels'], metrics['stable_pixels'], minimum),
                    'weighted_bce': bce, 'unweighted_bce': plain,
                    'mean_weight_normalized_bce': bce/((weight*metrics['positive_pixels']+metrics['stable_pixels'])/n),
                    'probability_diagnostics': probability_diagnostics(p, labels[index])})
        metrics = pooled.summary()
        require(metrics['supervised_pixels'] == total and metrics['ignored_pixels'] == selection['ignored_pixels'], 'Checkpoint support changed')
        aggregate = {'metrics': metrics, 'fit': fit_assessment(metrics, 151, 2640, minimum),
                     'weighted_bce': weighted/total, 'unweighted_bce': unweighted/total,
                     'mean_weight_normalized_bce': weighted/total/mean_weight}
        all_fit = all(e['fit']['passed'] for e in examples)
        filename = f'checkpoint_{attempt:04d}.npz'
        np.savez_compressed(output/filename, attempt=np.asarray(attempt, dtype='int64'),
                            logits=np.stack(logits_list), probabilities=np.stack(probabilities),
                            labels=labels, valid=valid, event_ids=np.asarray(selection['event_ids']))
        row = {'attempt': attempt, 'arrays_file': filename, 'parameter_sha256': parameter_hash(model),
               'aggregate': aggregate, 'examples': examples, 'all_examples_fit': all_fit,
               'fit_passed': all_fit and aggregate['fit']['passed']}
        checkpoints.append(row); model.train()
        print(f"[reviewed-pixel-pilot] {attempt}/{protocol['attempts']}; Dice={metrics['dice']:.4f}; examples passing={sum(e['fit']['passed'] for e in examples)}/13", flush=True)
        return logits_list, probabilities

    checkpoint(0); steps = []
    for attempt in range(1, protocol['attempts']+1):
        index = (attempt-1) % count
        result = train_step(model, optimizer, samples[index], weight, **kwargs)
        require(result['updated'], 'Every fixed reviewed example must produce a supervised step')
        steps.append({'attempt': attempt, 'event_id': selection['event_ids'][index], **result})
        if attempt in protocol['checkpoint_attempts']:
            final_logits, final_probabilities = checkpoint(attempt)
    final_hash = parameter_hash(model)
    require(final_hash != initial_hash and any(s['parameter_delta_l2'] > 0 for s in steps)
            and optimizer.step_calls == len(steps) == protocol['attempts'], 'Fixed training update coverage failed')
    require([c['attempt'] for c in checkpoints] == protocol['checkpoint_attempts'], 'Checkpoint coverage changed')
    require([_tensor_hash(r) for r in raw] == tensors, 'Cached arrays changed during training')
    after = ReviewedPilotDataset(data)
    require(after.audit() == audit and [_tensor_hash(after[i]) for i in range(len(after))] == tensors,
            'Exported data changed during training')
    require(_digest(plan_path) == plan_sha and {name: _digest(ROOT/'Solutions'/name) for name in SOURCE_NAMES} == implementation,
            'Plan or implementation changed during training')
    np.savez_compressed(output/'model_weights.npz', **{name: p.detach().cpu().numpy().copy() for name, p in model.state_dict().items()})
    restored = make_model()
    with np.load(output/'model_weights.npz', allow_pickle=False) as archive:
        restored.load_state_dict({name: torch.from_numpy(archive[name].copy()) for name in archive.files})
    restored.eval()
    with torch.inference_mode():
        require(all(np.array_equal(restored(s['features'])[0, 0].numpy(), z)
                    for s, z in zip(samples, final_logits)) and parameter_hash(restored) == final_hash,
                'Saved weights did not exactly reproduce every final logit map')
    (output/'steps.jsonl').write_text(''.join(json.dumps(s, allow_nan=False)+'\n' for s in steps))
    for index in range(0, count, 4):
        render_preview(output/f'preview_{index//4+1}.png', raw[index:index+4], final_probabilities[index:index+4])
    return {'execution_status': 'completed', 'scope': plan['scope'], 'fit_passed': checkpoints[-1]['fit_passed'],
            'plan_sha256': plan_sha, 'implementation_sha256': implementation,
            'dataset_metadata_sha256': dataset.metadata_sha256, 'manifest_sha256': dataset.manifest_sha256,
            'dataset_audit': audit, 'objective': objective, 'protocol': protocol, 'tensor_sha256': tensors,
            'initial_parameter_sha256': initial_hash, 'final_parameter_sha256': final_hash,
            'parameter_count': protocol['parameter_count'], 'device': 'cpu', 'gpu_used': False,
            'batch_size': 1, 'native_threads': 1, 'dataloader_workers': 0,
            'torch_intraop_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
            'attempted_batches': len(steps), 'optimizer_steps': optimizer.step_calls,
            'all_ignored_no_forward_no_step': True, 'all_ignored_safety_check': skip,
            'ignored_logit_gradients_zero_every_step': True, 'saved_weights_reproduce_final_logits': True,
            'source_dataset_unchanged': True, 'checkpoints': checkpoints,
            'first_passing_recorded_checkpoint': next((r['attempt'] for r in checkpoints if r['fit_passed']), None),
            'zero_parameter_delta_attempts': [s['attempt'] for s in steps if s['zero_parameter_delta']],
            'torch_version': str(torch.__version__), 'torch_distribution_version': importlib.metadata.version('torch'),
            'torch_git_version': torch.version.git_version, 'torch_module_path': str(Path(torch.__file__).resolve()),
            'numpy_version': np.__version__, 'rasterio_version': importlib.metadata.version('rasterio'),
            'python_version': sys.version, 'python_executable': sys.executable,
            'artifact_sha256': {p.name: _digest(p) for p in output.iterdir() if p.is_file()},
            'limitations': plan['interpretation_limits']}
