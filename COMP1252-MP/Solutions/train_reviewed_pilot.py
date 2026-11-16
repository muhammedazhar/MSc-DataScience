#!/usr/bin/env python3
"""Run the fixed all-13 reviewed-pixel diagnostic under the Mac resource guard."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def validate_plan(plan):
    """Reject accidental scope, threshold, support or resource expansion."""
    from diagnose_pilot_overfit import require

    require(plan['schema_version'] == 1, 'Unknown pilot protocol version')
    selection, objective, protocol = (plan[k] for k in ('selection', 'objective', 'protocol'))
    require(len(selection['event_ids']) == len(set(selection['event_ids'])) == 13
            and len(selection['group_ids']) == len(set(selection['group_ids'])) == 13,
            'The fixed diagnostic requires 13 unique reviewed training groups')
    require((selection['positive_pixels'], selection['stable_pixels'], selection['ignored_pixels'])
            == (151, 2640, 50457), 'Frozen reviewed support changed')
    require(objective['name'] == 'dataset_reviewed_pixel_mean_weighted_bce'
            and objective['positive_weight'] == 2640/151
            and objective['image_count'] == 13 and objective['total_reviewed_pixels'] == 2791,
            'The reviewed-pixel objective changed')
    require(protocol['seed'] == 42 and protocol['cycles'] == 128 and protocol['attempts'] == 1664
            and protocol['checkpoint_attempts'] == [0, 26, 416, 832, 1664], 'Fixed exposure changed')
    require(protocol['optimizer'] == {'optimizer': 'SGD', 'learning_rate': .05,
                                    'momentum': 0., 'weight_decay': 0.}, 'Fixed optimizer changed')
    require(protocol['probability_threshold'] == .5 and protocol['early_stopping'] is False
            and protocol['order'] == 'fixed manifest order, no shuffle'
            and protocol['parameter_count'] == 2545 and protocol['batch_size'] == 1
            and protocol['dtype'] == 'float32' and protocol['augmentation'] is False,
            'Fixed model/observation protocol changed')
    require(plan['fit_gate'] == {'primary_attempt': 1664, 'minimum': .95,
            'metrics': ['precision', 'recall', 'specificity', 'dice'],
            'require_each_example_and_aggregate': True, 'both_predicted_classes': True,
            'no_fit_is_valid_completed_result': True}, 'Fixed final fit gate changed')
    require(plan['resources'] == {'worker_rss_mib': 512, 'available_memory_reserve_gib': 1.5,
            'max_seconds': 300, 'workers': 1, 'native_threads': 1, 'torch_threads': 1,
            'nice_increment': 10, 'gpu_used': False, 'dataloader_workers': 0,
            'cached_examples': 13}, 'Mac resource declaration changed')


def write_report(path, metadata):
    lines = ['# All-13 reviewed-pixel training diagnostic', '',
             '**Training data only. These scores do not estimate unseen-location accuracy.**', '',
             f"Completed {metadata['attempted_batches']} fixed updates. Final fit on every reviewed example: **{metadata['fit_passed']}**.",
             '', '| Attempt | Pixel-weighted BCE | Normalized BCE | Dice | Precision | Recall | Specificity | Examples passing |',
             '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in metadata['checkpoints']:
        a = row['aggregate']; m = a['metrics']
        values = [row['attempt'], a['weighted_bce'], a['mean_weight_normalized_bce'],
                  m['dice'], m['precision'], m['recall'], m['specificity'],
                  f"{sum(e['fit']['passed'] for e in row['examples'])}/13"]
        lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)+' |')
    lines += ['', '## Final per-example reviewed scores', '',
              '| Event | Clearing / forest | Dice | Precision | Recall | Specificity | Fit |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in metadata['checkpoints'][-1]['examples']:
        m = row['metrics']
        values = [row['event_id'], f"{m['positive_pixels']} / {m['stable_pixels']}",
                  m['dice'], m['precision'], m['recall'], m['specificity'], row['fit']['passed']]
        lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)+' |')
    lines += ['', '## Actual predictions', '']
    for index in range(1, 5):
        lines += [f'![Reviewed labels and actual probabilities, page {index}](preview_{index}.png)', '']
    lines += ['Gray label pixels are ignored; predictions there remain unvalidated.', '',
              'The run changes normalization and exposure relative to the old 26-step smoke. It does not isolate an empirical cause.',
              'All checkpoints and no-fit outcomes are retained. No early stopping, threshold tuning or held-out selection.',
              'Gradient execution and saved-weight replay are worker receipts; the independent verifier reconstructs saved logits without executing a model.']
    Path(path).write_text('\n'.join(lines).rstrip()+'\n')


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'Docs/Reviews/PixelPilotV1/experiment_plan.json')
    parser.add_argument('--dataset', type=Path, default=ROOT/'Datasets/CuratedPilot/v1/dataset')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Use 128–512 MiB RSS, >=1.5 GiB reserve and <=300 seconds')
    validate_plan(json.loads(args.plan.read_text()))
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        os.nice(10)
        from reviewed_pilot_worker import run
        metadata = run(args, args.worker_output)
        write_report(args.worker_output/'report.md', metadata)
        metadata['artifact_sha256']['report.md'] = hashlib.sha256((args.worker_output/'report.md').read_bytes()).hexdigest()
        import resource
        metadata['worker_high_water_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024), 2)
        (args.worker_output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    else:
        run_guarded(args, script_path=__file__, operation='reviewed-pixel-pilot', gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
