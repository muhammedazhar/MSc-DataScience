#!/usr/bin/env python3
"""Predeclared, bounded single-example training diagnostics and fit gates."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_count(value, name):
    require(type(value) is int and value > 0, f'{name} must be a positive integer')
    return value


def constant_optimum(positives, negatives, pos_weight):
    """Analytic constant-logit optimum for mean weighted binary BCE."""
    p = _positive_count(positives, 'Positive count')
    n = _positive_count(negatives, 'Negative count')
    require(type(pos_weight) in (int, float) and math.isfinite(pos_weight) and pos_weight > 0,
            'Positive weight must be finite and positive')
    weighted_p = pos_weight*p
    require(math.isfinite(weighted_p+n), 'Weighted support exceeds finite range')
    logit = math.log(pos_weight)+math.log(p)-math.log(n)
    probability = 1/(1+math.exp(-logit)) if logit >= 0 else math.exp(logit)/(1+math.exp(logit))
    soft = math.log1p(math.exp(-abs(logit)))
    weighted_bce = (weighted_p*(soft+max(-logit, 0))+n*(soft+max(logit, 0)))/(p+n)
    return {'probability': probability, 'logit': logit, 'weighted_bce': weighted_bce,
            'mean_weight': (weighted_p+n)/(p+n)}


def fit_assessment(metrics, positives, negatives, minimum=.95):
    """Derive a nontrivial reviewed-ROI fit gate from counts, never claimed scores."""
    p = _positive_count(positives, 'Positive support')
    n = _positive_count(negatives, 'Negative support')
    require(type(minimum) in (int, float) and math.isfinite(minimum) and 0 < minimum <= 1,
            'Fit minimum must be finite and in (0,1]')
    require(isinstance(metrics, dict) and metrics.get('probability_threshold') == .5,
            'Diagnostic threshold must remain exactly 0.5')
    for name in ('tp', 'fp', 'fn', 'tn'):
        require(type(metrics.get(name)) is int and metrics[name] >= 0, 'Confusion counts must be nonnegative integers')
    tp, fp, fn, tn = (metrics[k] for k in ('tp', 'fp', 'fn', 'tn'))
    require(tp+fn == p and tn+fp == n, 'Reviewed class support changed')
    precision = tp/(tp+fp) if tp+fp else None
    scores = {'precision': precision, 'recall': tp/p, 'specificity': tn/n,
              'dice': 2*tp/(2*tp+fp+fn)}
    reasons = [f'{name}_below_minimum' for name, value in scores.items() if value is None or value < minimum]
    if not (tp+fp and tn+fn):
        reasons.append('only_one_predicted_class')
    return {'passed': not reasons, 'reasons': reasons, **scores, 'minimum': minimum}


def checkpoint_steps(limit):
    require(type(limit) is int and 1 <= limit <= 512, 'Use 1–512 attempted updates')
    return sorted({0, limit, *(step for step in (26, 128, 256, 512) if step <= limit)})


def select_record(records, event_id=None):
    require(isinstance(records, (list, tuple)) and bool(records), 'A nonempty reviewed manifest is required')
    identifiers = []
    for row in records:
        require(isinstance(row, dict) and row.get('split') == 'train' and row.get('training_approved') is True,
                'Only approved training records may enter this diagnostic')
        evidence = row.get('visibility_evidence', {})
        require(evidence.get('decision_status') == 'accepted' and evidence.get('training_approved') is True,
                'Explicit accepted review evidence is required')
        require(isinstance(row.get('event_id'), str) and bool(row['event_id']), 'Missing event identity')
        identifiers.append(row['event_id'])
    require(len(identifiers) == len(set(identifiers)), 'Reviewed event identities repeat')
    wanted = identifiers[0] if event_id is None else event_id
    require(wanted in identifiers, 'Selected event is absent from the reviewed training manifest')
    return identifiers.index(wanted)


def probability_diagnostics(probabilities, labels):
    """Class distributions and exact ranking AUC on reviewed pixels only."""
    import numpy as np

    probability, target = np.asarray(probabilities), np.asarray(labels)
    require(probability.ndim == 2 and probability.shape == target.shape and 0 < target.size <= 4096,
            'Diagnostic probability/label shapes must match inside one small core')
    require(target.dtype.kind in 'uif' and np.isin(target, (0, 1, 255)).all(), 'Labels must remain numeric 0/1/255')
    require(probability.dtype.kind in 'uif', 'Probabilities must be numeric')
    selected = probability[target != 255]
    require(np.isfinite(selected).all() and ((selected >= 0) & (selected <= 1)).all(), 'Reviewed probabilities must be finite in [0,1]')
    positive, negative = probability[target == 1].astype('float64'), probability[target == 0].astype('float64')
    require(positive.size > 0 and negative.size > 0, 'Both reviewed classes are needed')

    def stats(values):
        return {'count': int(values.size), 'mean': float(values.mean()), 'median': float(np.median(values)),
                'min': float(values.min()), 'max': float(values.max())}

    wins = positive[:, None] > negative[None, :]
    ties = positive[:, None] == negative[None, :]
    return {'positive': stats(positive), 'negative': stats(negative),
            'ignored_pixels': int((target == 255).sum()),
            'training_pairwise_auc': float(wins.mean()+.5*ties.mean())}


def write_arm_report(path, metadata):
    """The worker returns all evidence; this view stays small and readable."""
    lines = ['# Single-example overfit diagnostic', '',
             '**Training memorization diagnostic only; no held-out score or timing inference.**', '',
             f"Arm: `{metadata['arm_id']}`; event `{metadata['event_id']}`.",
             f"Completed {metadata['attempted_batches']} fixed attempts. Final reviewed-ROI fit gate: **{metadata['fit_passed']}**.",
             f"Support: {metadata['selected_positive_pixels']} clearing, {metadata['selected_stable_pixels']} forest, {metadata['selected_ignored_pixels']} ignored pixels.", '',
             'Actual prediction preview:', '', '![Actual probabilities and reviewed labels](preview.png)', '',
             'Gray labels are ignored. Predictions outside reviewed ROIs remain unvalidated.', '']
    checkpoints = metadata.get('checkpoints', [])
    if checkpoints:
        lines += ['| Attempt | Weighted BCE | Normalized BCE | Dice | Precision | Recall | Ranking AUC | Fit |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
        for row in checkpoints:
            metrics = row.get('metrics', {})
            fit = row.get('fit', row.get('fit_assessment', {}))
            loss = row.get('weighted_bce', metrics.get('weighted_bce'))
            def fmt(value):
                return f'{value:.6f}' if isinstance(value, float) else str(value)
            values = [row['attempt'], loss, row['mean_weight_normalized_bce'], metrics['dice'],
                      metrics['precision'], metrics['recall'],
                      row['probability_diagnostics']['training_pairwise_auc'], fit['passed']]
            lines.append('| '+' | '.join(fmt(value) for value in values)+' |')
    lines += ['', 'The primary fit decision uses the final attempt, never the best checkpoint. A completed fit=False result is retained as evidence.',
              'Global/local weighted losses have different scales; normalized BCE divides by mean class-weight mass. This does not make their objectives identical.',
              'Checkpoints and threshold 0.5 were fixed before execution. The adjacent metadata records actual resources, source hashes and stationary-step counts.', '']
    Path(path).write_text('\n'.join(lines)+'\n')


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'Docs/Reviews/OverfitDiagnosticV1/experiment_plan.json')
    parser.add_argument('--dataset', type=Path, default=ROOT/'Datasets/CuratedPilot/v1/dataset')
    parser.add_argument('--arm', required=True, choices=('global_weight_sgd', 'local_weight_sgd', 'local_weight_adam'))
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
    if args.worker_output:
        os.nice(10)
        from pilot_overfit_worker import run_arm
        metadata = run_arm(args, args.worker_output)
        write_arm_report(args.worker_output/'report.md', metadata)
        metadata.setdefault('artifact_sha256', {})['report.md'] = hashlib.sha256((args.worker_output/'report.md').read_bytes()).hexdigest()
        import resource
        metadata['worker_high_water_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024), 2)
        (args.worker_output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    else:
        run_guarded(args, script_path=__file__, operation='pilot-overfit', gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
