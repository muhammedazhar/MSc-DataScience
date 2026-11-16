#!/usr/bin/env python3
"""A small, CPU-only smoke run for reviewed partial-label training.

This is a development check on training data, not held-out evaluation or a
first-visible-date prediction model. The existing forest models are not loaded.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_model():
    import torch
    from torch import nn

    return nn.Sequential(nn.Conv2d(27, 8, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(8, 1, 1)).to(device='cpu', dtype=torch.float32)


class CountingOptimizer:
    def __init__(self, params, lr=.05, momentum=0., weight_decay=0.):
        import torch

        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.step_calls = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.step_calls += 1
        return self.optimizer.step()


def train_step(model, optimizer, sample, pos_weight=None):
    import torch
    from pilot_objectives import _reviewed, masked_bce_with_logits

    optimizer.zero_grad(set_to_none=True)
    features, labels, valid = sample['features'], sample['labels'], sample['valid']
    require(all(isinstance(x, torch.Tensor) and x.device.type == 'cpu' for x in (features, labels, valid)),
            'The smoke step requires CPU tensors')
    require(features.ndim == 4 and features.dtype == torch.float32, 'Features require float32 BxCxHxW tensors')
    require(all(p.device.type == 'cpu' for p in model.parameters())
            and all(b.device.type == 'cpu' for b in model.buffers()), 'The model must remain on CPU')
    # Validate labels before forward: even BatchNorm/RNG state must stay unchanged
    # when every pixel is ignored. No fake zero loss reaches an optimizer.
    _, _, reviewed = _reviewed(torch.zeros_like(labels, dtype=torch.float32), labels, valid)
    count = int(reviewed.sum().item())
    if not count:
        return {'updated': False, 'loss': None, 'reviewed_pixels': 0,
                'gradient_norm': 0.0, 'parameter_delta_l2': 0.0}
    require(bool(torch.isfinite(features).all()), 'All model inputs must be finite, including ignored context')
    before = [p.detach().clone() for p in model.parameters()]
    require(all(bool(torch.isfinite(p).all()) for p in before), 'Initial parameters must be finite')
    logits = model(features)
    require(logits.device.type == 'cpu', 'Model outputs must remain on CPU')
    loss = masked_bce_with_logits(logits, labels, valid, pos_weight)
    require(loss is not None and bool(torch.isfinite(loss)), 'Supervised loss must be finite')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    require(bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients), 'Gradients must be finite')
    norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients))
    require(math.isfinite(norm), 'Gradient norm must be finite')
    optimizer.step()
    after = list(model.parameters())
    require(all(bool(torch.isfinite(p).all()) for p in after), 'Updated parameters must be finite')
    delta = math.sqrt(sum(float((p.detach()-old).double().square().sum()) for p, old in zip(after, before)))
    require(math.isfinite(delta) and delta > 0, 'The supervised step did not update parameters')
    return {'updated': True, 'loss': float(loss.detach()), 'reviewed_pixels': count,
            'gradient_norm': norm, 'parameter_delta_l2': delta}


def parameter_hash(model):
    checksum = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        checksum.update(name.encode())
        checksum.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def evaluate(model, loader, pos_weight):
    import torch
    from pilot_objectives import ConfusionCounts, masked_bce_with_logits

    model.eval()
    total, weighted, unweighted, records, predictions = 0, 0., 0., [], []
    aggregate = ConfusionCounts()
    with torch.inference_mode():
        for sample in loader:
            logits = model(sample['features'])
            require(logits.device.type == 'cpu' and bool(torch.isfinite(logits).all()),
                    'Evaluation requires finite CPU predictions')
            labels, valid = sample['labels'], sample['valid']
            n = aggregate.update(logits, labels, valid)
            metrics = ConfusionCounts()
            metrics.update(logits, labels, valid)
            if n:
                weighted += float(masked_bce_with_logits(logits, labels, valid, pos_weight))*n
                unweighted += float(masked_bce_with_logits(logits, labels, valid))*n
                total += n
            records.append({'event_id': sample['event_id'][0], **metrics.summary()})
            predictions.append(torch.sigmoid(logits[0, 0]).numpy().copy())
    require(total > 0, 'Evaluation has no reviewed pixels')
    return {'weighted_bce': weighted/total, 'unweighted_bce': unweighted/total,
            **aggregate.summary()}, records, predictions


def render_preview(path, dataset, probabilities):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    panel, gap, row_height = 200, 16, 255
    count = min(4, len(dataset))
    canvas = Image.new('RGB', (3*(panel+gap)+gap, 55+row_height*count+28), '#101822')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((gap, 8), 'Training-only smoke predictions | red=reviewed clearing; green=reviewed forest', fill='white', font=font)
    for index in range(count):
        sample = dataset[index]
        values, labels = sample['features'].numpy(), sample['labels'].numpy()
        rgb = np.moveaxis(values[[11, 10, 9]], 0, -1)
        rgb = (np.clip(rgb/.30, 0, 1)*255).astype('uint8')
        supervision = np.full((64, 64, 3), 55, dtype='uint8')
        supervision[labels == 0] = [40, 195, 85]
        supervision[labels == 1] = [255, 65, 65]
        probability = np.clip(probabilities[index], 0, 1)
        probability_rgb = np.stack((probability*255, np.zeros_like(probability), (1-probability)*255), axis=-1).astype('uint8')
        y = 55+index*row_height
        draw.text((gap, y-17), sample['event_id'], fill='white', font=font)
        for column, (array, name) in enumerate(zip((rgb, supervision, probability_rgb),
                                                  ('Post RGB', 'Reviewed labels; gray=ignore', 'Predicted probability: blue=0 red=1'))):
            x = gap+column*(panel+gap)
            draw.text((x, y), name, fill='white', font=font)
            canvas.paste(Image.fromarray(array).resize((panel, panel), Image.Resampling.NEAREST), (x, y+20))
    draw.text((gap, canvas.height-24), 'Predictions outside reviewed ROIs are unvalidated; these are training diagnostics.', fill='white', font=font)
    canvas.save(path)


def run(args, output):
    os.nice(10)
    from run_pilot_checks import configure_cpu
    torch = configure_cpu(args.seed)
    import numpy as np
    from torch.utils.data import DataLoader
    from pilot_dataset import ReviewedPilotDataset

    dataset = ReviewedPilotDataset(args.dataset)
    audit = dataset.audit()
    require(audit['supervised_positive_pixels'] > 0 and audit['supervised_stable_pixels'] > 0,
            'The smoke run needs both reviewed classes')
    pos_weight = audit['supervised_stable_pixels']/audit['supervised_positive_pixels']
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    model = make_model()
    optimizer = CountingOptimizer(model.parameters(), lr=args.learning_rate)
    initial_hash = parameter_hash(model)
    initial, _, _ = evaluate(model, loader, pos_weight)
    # Exercise the actual skip path and prove it calls neither forward nor step.
    ignored = next(iter(loader))
    ignored['labels'] = torch.full_like(ignored['labels'], 255)
    ignored['valid'] = torch.zeros_like(ignored['valid'])
    calls_before = optimizer.step_calls
    forward_calls = []
    handle = model.register_forward_hook(lambda *values: forward_calls.append(1))
    skipped = train_step(model, optimizer, ignored, pos_weight)
    handle.remove()
    require(not skipped['updated'] and not forward_calls and optimizer.step_calls == calls_before
            and parameter_hash(model) == initial_hash, 'All-ignored skip changed model/optimizer execution')
    steps = []
    model.train()
    for _ in range(math.ceil(args.steps/len(dataset))):
        for sample in loader:
            if len(steps) >= args.steps:
                break
            result = train_step(model, optimizer, sample, pos_weight)
            steps.append({'attempt': len(steps)+1, 'event_id': sample['event_id'][0], **result})
    final, examples, predictions = evaluate(model, loader, pos_weight)
    final_hash = parameter_hash(model)
    require(final_hash != initial_hash and optimizer.step_calls == sum(s['updated'] for s in steps),
            'Optimizer/update checks failed')
    require(all(final[name] == audit[field] for name, field in
                (('positive_pixels', 'supervised_positive_pixels'), ('stable_pixels', 'supervised_stable_pixels'),
                 ('ignored_pixels', 'ignored_pixels'))), 'Training diagnostic support changed')
    after_audit = ReviewedPilotDataset(args.dataset).audit()
    require(audit == after_audit, 'Dataset changed during smoke training')
    weights = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    np.savez_compressed(output/'model_weights.npz', **weights)
    restored = make_model()
    with np.load(output/'model_weights.npz', allow_pickle=False) as archive:
        restored.load_state_dict({name: torch.from_numpy(archive[name].copy()) for name in archive.files})
    replay, _, replay_predictions = evaluate(restored, loader, pos_weight)
    require(replay == final and all(np.array_equal(a, b) for a, b in zip(predictions, replay_predictions)),
            'Saved weights did not reproduce predictions')
    np.savez_compressed(output/'predictions.npz', probabilities=np.stack(predictions),
                        event_ids=np.asarray(audit['event_ids']))
    (output/'steps.jsonl').write_text(''.join(json.dumps(s, allow_nan=False)+'\n' for s in steps))
    (output/'per_example_diagnostics.jsonl').write_text(''.join(json.dumps(s, allow_nan=False)+'\n' for s in examples))
    render_preview(output/'preview.png', dataset, predictions)
    sources = ('train_pilot_baseline.py', 'pilot_dataset.py', 'pilot_objectives.py', 'run_pilot_checks.py')
    metadata = {
        'smoke_status': 'passed', 'scope': 'Training-only pipeline smoke check; no independent accuracy or date-inference claim',
        'seed': args.seed, 'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
        'torch_distribution_version': importlib.metadata.version('torch'),
        'torch_git_version': torch.version.git_version,
        'torch_module_path': str(Path(torch.__file__).resolve()),
        'rasterio_version': importlib.metadata.version('rasterio'),
        'device': 'cpu', 'torch_intraop_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'batch_size': 1, 'dataloader_workers': 0, 'shuffle': False,
        'architecture': 'Conv27-8(3x3), ReLU, Conv8-8(3x3), ReLU, Conv8-1(1x1)',
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'optimizer': 'SGD; momentum=0; weight_decay=0', 'learning_rate': args.learning_rate,
        'positive_weight': pos_weight, 'attempted_batches': len(steps), 'optimizer_updates': optimizer.step_calls,
        'skipped_training_batches': sum(not s['updated'] for s in steps), 'all_ignored_safety_check': skipped,
        'all_ignored_no_forward_no_step': True, 'dataset_audit': audit,
        'initial_training_diagnostics': initial, 'final_training_diagnostics': final,
        'all_negative_accuracy': audit['supervised_stable_pixels']/(audit['supervised_positive_pixels']+audit['supervised_stable_pixels']),
        'initial_parameter_sha256': initial_hash, 'final_parameter_sha256': final_hash,
        'saved_weights_reproduce_predictions': True,
        'all_reviewed_predictions_positive': final['tp']+final['fp'] == final['supervised_pixels'],
        'smoke_gate': 'Finite updates, correct masking and preserved data; not useful detection accuracy',
        'implementation_sha256': {name: digest(ROOT/'Solutions'/name) for name in sources},
        'artifact_sha256': {path.name: digest(path) for path in output.iterdir() if path.is_file()},
        'limitations': ['All diagnostics use the same reviewed training groups.',
                        'Only 151 positive interior pixels are reviewed in the current pilot.',
                        'No threshold tuning, held-out evaluation or first-visible-date inference was performed.'],
    }
    write_report(output/'report.md', metadata)
    metadata['artifact_sha256']['report.md'] = digest(output/'report.md')
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metadata['worker_high_water_rss_mib'] = round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(f"[pilot-smoke] {optimizer.step_calls} updates; parameter and saved-prediction checks passed", flush=True)


def write_report(path, metadata):
    before, after = metadata['initial_training_diagnostics'], metadata['final_training_diagnostics']
    lines = ['# CPU pilot baseline smoke check', '',
             f"**Passed** {metadata['optimizer_updates']} tiny-model updates on {metadata['dataset_audit']['example_count']} reviewed training examples.",
             'All-ignored input skipped forward and optimizer execution. Saved weights exactly reproduced the final predictions. Dataset hashes/counts stayed unchanged.', '',
             '**These are training diagnostics. They are not independent accuracy or first-visible-date predictions.**', '',
             '| Diagnostic | Initial | Final |', '| --- | ---: | ---: |']
    for name in ('weighted_bce', 'unweighted_bce', 'tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'dice', 'iou', 'accuracy', 'balanced_accuracy'):
        values = [str(round(item[name], 6)) if isinstance(item[name], float) else str(item[name]) for item in (before, after)]
        lines.append(f"| {name} | {values[0]} | {values[1]} |")
    lines += ['', f"Reviewed support: {after['positive_pixels']} positive, {after['stable_pixels']} stable, {after['ignored_pixels']} ignored pixels.",
              f"An all-negative prediction already has {metadata['all_negative_accuracy']:.2%} reviewed-pixel accuracy; accuracy alone is misleading.",
              '', '![Actual model probabilities and reviewed labels](preview.png)', '',
              'The displayed probability images come from this tiny trained model. Gray supervision pixels are ignored; probabilities there are unvalidated.',
              '', 'One CPU worker, native/Torch threads one, batch size one, no data-worker processes or GPU. The adjacent metadata records the measured guard resources.', '']
    if metadata['all_reviewed_predictions_positive']:
        lines += ['At threshold 0.5 this short smoke model predicts every reviewed pixel as clearing. It passes the pipeline checks but has not learned useful discrimination.', '']
    lines += [f"Runtime PyTorch: {metadata['torch_version']} (git {metadata['torch_git_version']}); installed distribution metadata: {metadata['torch_distribution_version']}. The runtime identity is retained because these differ in the current environment.", '']
    Path(path).write_text('\n'.join(lines))


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'Datasets/CuratedPilot/v1/dataset')
    parser.add_argument('--output', type=Path, default=ROOT/'Datasets/CuratedPilot/v1/baseline_smoke')
    parser.add_argument('--steps', type=int, default=26)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=.05)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=180)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (1 <= args.steps <= 78 and 0 <= args.seed < 2**32
            and math.isfinite(args.learning_rate) and 0 < args.learning_rate <= .1):
        parser.error('Use 1–78 attempted steps, a uint32 seed and learning rate in (0, .1]')
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Use 128–512 MiB RSS, >=1.5 GiB reserve and <=300 seconds')
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        run(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='pilot-baseline', gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
