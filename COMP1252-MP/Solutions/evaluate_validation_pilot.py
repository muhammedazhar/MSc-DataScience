#!/usr/bin/env python3
"""Evaluate locked partial-ROI validation labels with the frozen final checkpoint.

The evaluate worker imports Torch only after verifying committed label locks.
The verify worker reconstructs saved predictions using NumPy and executes no
model. Neither mode trains, selects a checkpoint or adjusts a threshold.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = ('evaluate_validation_pilot.py', 'verify_validation_dataset.py',
                'train_pilot_baseline.py', 'pilot_objectives.py', 'run_pilot_checks.py',
                'summarize_pilot_overfit.py', 'validate_sample.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 16*2**20, 'Missing or oversized JSON metadata')
    value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    require(isinstance(value, dict), 'Metadata must be a JSON object')
    return value


def rooted(relative):
    require(isinstance(relative, str) and bool(relative) and not Path(relative).is_absolute(), 'Project-relative path required')
    path = (ROOT/relative).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), 'Missing or escaped project artifact')
    return path


def _git_bytes(commit, path):
    require(re.fullmatch(r'[a-f0-9]{7,40}|HEAD', commit) is not None, 'A fixed Git commit identity is required')
    repo = ROOT.parent
    require(Path(path).resolve().is_relative_to(repo), 'Git-bound artifact escaped repository')
    relative = Path(path).resolve().relative_to(repo).as_posix()
    result = subprocess.run(['git', 'show', f'{commit}:{relative}'], cwd=repo,
                            capture_output=True, timeout=20, check=False)
    require(result.returncode == 0 and len(result.stdout) <= 16*2**20,
            'Label artifact is absent from its required committed snapshot')
    return result.stdout


def verify_label_lock(dataset, protocol_path, lock_path, dataset_meta, protocol):
    """Refuse inference until exact labels/reviews/protocol have durable Git state."""
    lock = read_json(lock_path); evaluation = protocol['evaluation']
    require(protocol['status'] == 'predeclared_before_validation_predictions'
            and evaluation['checkpoint_attempt'] == 1664 and evaluation['probability_threshold'] == .5
            and evaluation['training_positive_weight'] == 2640/151
            and evaluation['retraining'] is False and evaluation['threshold_tuning'] is False
            and evaluation['best_checkpoint_selection'] is False, 'Frozen evaluation protocol changed')
    reviews = rooted(dataset_meta['reviews_file']); weights = rooted(evaluation['checkpoint'])
    actual = {'dataset_metadata_sha256': digest(dataset/'metadata.json'),
              'protocol_sha256': digest(protocol_path), 'reviews_sha256': digest(reviews),
              'weights_sha256': digest(weights)}
    require(all(lock.get(key) == value for key, value in actual.items()), 'Committed label lock hash binding failed')
    require(dataset_meta['protocol_sha256'] == actual['protocol_sha256']
            and dataset_meta['reviews_sha256'] == actual['reviews_sha256']
            and dataset_meta['model_weights_sha256'] == evaluation['weights_sha256'] == actual['weights_sha256'],
            'Dataset/review/protocol/model identities disagree')
    commit = lock.get('label_export_commit')
    require(isinstance(commit, str) and re.fullmatch(r'[a-f0-9]{7,40}', commit), 'Fixed label export commit required')
    ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT.parent,
                              capture_output=True, timeout=20, check=False)
    require(ancestor.returncode == 0, 'Label export snapshot must be an ancestor of the current branch')
    for path, expected in ((dataset/'metadata.json', actual['dataset_metadata_sha256']),
                           (reviews, actual['reviews_sha256']), (protocol_path, actual['protocol_sha256'])):
        require(hashlib.sha256(_git_bytes(commit, path)).hexdigest() == expected,
                'Current labels/reviews/protocol differ from the pre-inference export commit')
    require(hashlib.sha256(_git_bytes('HEAD', lock_path)).hexdigest() == digest(lock_path),
            'Label-lock file must itself be committed before validation inference')
    return {**actual, 'label_lock_sha256': digest(lock_path), 'label_export_commit': commit,
            'checkpoint_file': evaluation['checkpoint'], 'reviews_file': dataset_meta['reviews_file']}


def verify_dataset_payload(dataset, dataset_meta):
    for name, checksum in dataset_meta['output_sha256_excluding_this_metadata'].items():
        path=(dataset/name).resolve()
        require(path.is_relative_to(dataset) and path.is_file() and digest(path)==checksum,
                'Validation payload changed during inference: '+name)


def _ratio(a, b):
    return a/b if b else None


def _numpy_metrics(tp, fp, fn, tn, ignored, updates):
    precision, recall, specificity = _ratio(tp, tp+fp), _ratio(tp, tp+fn), _ratio(tn, tn+fp)
    overlap = _ratio(2*tp, 2*tp+fp+fn)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'supervised_pixels': tp+fp+fn+tn, 'positive_pixels': tp+fn,
            'stable_pixels': tn+fp, 'ignored_pixels': ignored, 'updates': updates,
            'empty_updates': 0, 'probability_threshold': .5,
            'precision': precision, 'recall': recall, 'specificity': specificity,
            'f1': overlap, 'dice': overlap, 'iou': _ratio(tp, tp+fp+fn),
            'accuracy': _ratio(tp+tn, tp+fp+fn+tn),
            'balanced_accuracy': (recall+specificity)/2 if recall is not None and specificity is not None else None,
            'reviewed_stable_false_positive_rate': _ratio(fp, fp+tn)}


def numpy_summary(logits, probabilities, labels, rows, pos_weight=2640/151):
    """Independent saved-logit counts/losses; stable-only undefined ratios persist."""
    import numpy as np

    z, prob, target = map(np.asarray, (logits, probabilities, labels))
    require(z.ndim == 3 and z.shape == prob.shape == target.shape and 1 <= len(rows) <= 6
            and z.shape[0] == len(rows) and 0 < z.shape[1]*z.shape[2] <= 4096,
            'Prediction arrays must match at most six small validation samples')
    require(z.dtype.kind == prob.dtype.kind == 'f' and target.dtype.kind in 'ui'
            and np.isin(target, (0, 1, 255)).all(), 'Prediction/label types or values changed')
    require(type(pos_weight) in (int, float) and math.isfinite(pos_weight) and pos_weight > 0,
            'Positive weight must remain finite and positive')
    require(len({r['sample_id'] for r in rows}) == len(rows)
            and all(r['kind'] in ('loss', 'stable_roi') for r in rows), 'Sample identity/kind coverage changed')
    examples = []
    for index, row in enumerate(rows):
        mask = target[index] != 255; selected_z = z[index][mask].astype('float64'); selected_p = prob[index][mask]
        require(selected_z.size > 0 and np.isfinite(selected_z).all() and np.isfinite(selected_p).all()
                and ((selected_p >= 0) & (selected_p <= 1)).all(), 'Reviewed logits/probabilities must be finite')
        expected = np.empty_like(selected_z); positive_z = selected_z >= 0
        expected[positive_z] = 1/(1+np.exp(-selected_z[positive_z]))
        exp_value = np.exp(selected_z[~positive_z]); expected[~positive_z] = exp_value/(1+exp_value)
        require(np.allclose(selected_p, expected, rtol=0, atol=2e-7), 'Saved probability disagrees with logits')
        actual = target[index][mask] == 1; predicted = selected_z >= 0
        tp, fp = int((predicted & actual).sum()), int((predicted & ~actual).sum())
        fn, tn = int((~predicted & actual).sum()), int((~predicted & ~actual).sum())
        require((row['kind'] == 'loss' and tp+fn >= 4) or (row['kind'] == 'stable_roi' and tp+fn == 0),
                'Loss/stable sample class support changed')
        plain = np.where(actual, np.logaddexp(0., -selected_z), np.logaddexp(0., selected_z))
        weighted = np.where(actual, pos_weight*np.logaddexp(0., -selected_z), np.logaddexp(0., selected_z))
        examples.append({'sample_id': row['sample_id'], 'event_id': row['event_id'],
                         'group_id': row['group_id'], 'kind': row['kind'],
                         'metrics': _numpy_metrics(tp, fp, fn, tn, int((~mask).sum()), 1),
                         'weighted_bce': float(weighted.mean()), 'unweighted_bce': float(plain.mean())})

    def aggregate(items):
        counts = [sum(r['metrics'][key] for r in items) for key in ('tp', 'fp', 'fn', 'tn', 'ignored_pixels')]
        metrics = _numpy_metrics(*counts, len(items)); total = metrics['supervised_pixels']
        return {'sample_count': len(items), 'group_count': len({r['group_id'] for r in items}),
                'metrics': metrics,
                'weighted_bce': sum(r['weighted_bce']*r['metrics']['supervised_pixels'] for r in items)/total if total else None,
                'unweighted_bce': sum(r['unweighted_bce']*r['metrics']['supervised_pixels'] for r in items)/total if total else None}
    return {'examples': examples, 'summaries': {
        'pooled': aggregate(examples), 'loss_pairs': aggregate([r for r in examples if r['kind'] == 'loss']),
        'stable_pairs': aggregate([r for r in examples if r['kind'] == 'stable_roi']),
        'groups': [{'group_id': group, **aggregate([r for r in examples if r['group_id'] == group])}
                   for group in sorted({r['group_id'] for r in examples})]}}


def _torch_results(torch, samples, rows, model, pos_weight):
    """Independent worker route: Torch counters/BCE, not the NumPy verifier helper."""
    import numpy as np
    from pilot_objectives import ConfusionCounts, masked_bce_with_logits

    examples, logits_out, probabilities_out, preview_samples = [], [], [], []
    kinds = ('pooled', 'loss_pairs', 'stable_pairs')
    counters = {name: ConfusionCounts() for name in kinds}; losses = {name: [0., 0., 0] for name in kinds}
    group_counters, group_losses = {}, {}
    with torch.inference_mode():
        for row, sample in zip(rows, samples):
            features = np.asarray(sample['features']); labels_np = np.asarray(sample['labels']); valid_np = np.asarray(sample['valid'])
            require(features.shape == (27, 64, 64) and features.dtype == np.dtype('float32')
                    and labels_np.shape == valid_np.shape == (64, 64)
                    and np.array_equal(valid_np, labels_np != 255), 'Verified validation tensor shape changed')
            require(np.isfinite(features[:, valid_np]).all(), 'Reviewed features must be finite')
            # Invalid context cannot reach convolution; no supervised value changes.
            finite_features = np.where(np.isfinite(features), features, 0).astype('float32', copy=False)
            x = torch.from_numpy(finite_features.copy()); labels = torch.from_numpy(labels_np.copy())
            valid = torch.from_numpy(valid_np.copy()).bool()
            logits = model(x[None]); require(logits.shape == (1, 1, 64, 64)
                and logits.device.type == 'cpu' and bool(torch.isfinite(logits).all()), 'Finite CPU predictions required')
            counter = ConfusionCounts(); n = counter.update(logits, labels, valid)
            require(n > 0, 'Reviewed validation sample cannot be empty')
            weighted = float(masked_bce_with_logits(logits, labels, valid, pos_weight))
            plain = float(masked_bce_with_logits(logits, labels, valid))
            require(math.isfinite(weighted) and math.isfinite(plain), 'Validation BCE must remain finite')
            metrics = counter.summary(); metrics['reviewed_stable_false_positive_rate'] = _ratio(metrics['fp'], metrics['fp']+metrics['tn'])
            examples.append({'sample_id': row['sample_id'], 'event_id': row['event_id'],
                             'group_id': row['group_id'], 'kind': row['kind'], 'metrics': metrics,
                             'weighted_bce': weighted, 'unweighted_bce': plain})
            for key in ('pooled', 'loss_pairs' if row['kind'] == 'loss' else 'stable_pairs'):
                counters[key].update(logits, labels, valid)
                losses[key][0] += weighted*n; losses[key][1] += plain*n; losses[key][2] += n
            group = row['group_id']; group_counters.setdefault(group, ConfusionCounts()).update(logits, labels, valid)
            entry = group_losses.setdefault(group, [0., 0., 0]); entry[0] += weighted*n; entry[1] += plain*n; entry[2] += n
            logits_out.append(logits[0, 0].numpy().copy()); probabilities_out.append(torch.sigmoid(logits[0, 0]).numpy().copy())
            preview_samples.append({'features': x, 'labels': labels, 'sample_id': row['sample_id'], 'kind': row['kind']})
    def package(counter, values, items):
        metrics = counter.summary(); metrics['reviewed_stable_false_positive_rate'] = _ratio(metrics['fp'], metrics['fp']+metrics['tn'])
        return {'sample_count': len(items), 'group_count': len({r['group_id'] for r in items}), 'metrics': metrics,
                'weighted_bce': values[0]/values[2] if values[2] else None,
                'unweighted_bce': values[1]/values[2] if values[2] else None}
    summaries = {key: package(counters[key], losses[key],
                 [r for r in rows if key == 'pooled' or r['kind'] == ('loss' if key == 'loss_pairs' else 'stable_roi')])
                 for key in kinds}
    summaries['groups'] = [{'group_id': group, **package(group_counters[group], group_losses[group],
                             [r for r in rows if r['group_id'] == group])} for group in sorted(group_counters)]
    return {'examples': examples, 'summaries': summaries}, np.stack(logits_out), np.stack(probabilities_out), preview_samples


def render_validation_preview(path, samples, probabilities):
    """Actual validation images/labels/probabilities with an explicit scope notice."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    panel,gap,stride=200,16,255
    canvas=Image.new('RGB',(3*(panel+gap)+gap,70+stride*len(samples)+32),'#101822')
    draw=ImageDraw.Draw(canvas)
    try:
        font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',13)
    except OSError:
        font=ImageFont.load_default()
    draw.text((gap,8),'VALIDATION | frozen final-1664 checkpoint | threshold 0.5',fill='white',font=font)
    draw.text((gap,28),'Reviewed partial ROIs only; red=clearing, green=stable forest, gray=ignore',fill='white',font=font)
    for index,(sample,probability) in enumerate(zip(samples,probabilities)):
        values=sample['features'].numpy();labels=sample['labels'].numpy()
        rgb=(np.clip(np.moveaxis(values[[11,10,9]],0,-1)/.30,0,1)*255).astype('uint8')
        supervision=np.full((64,64,3),55,dtype='uint8');supervision[labels==0]=[40,195,85];supervision[labels==1]=[255,65,65]
        probability=np.clip(probability,0,1)
        colors=np.stack((probability*255,np.zeros_like(probability),(1-probability)*255),axis=-1).astype('uint8')
        y=70+index*stride
        draw.text((gap,y-17),sample['sample_id'],fill='white',font=font)
        for column,(array,name) in enumerate(zip((rgb,supervision,colors),('Post RGB','Reviewed labels','Probability: blue=0 red=1'))):
            x=gap+column*(panel+gap);draw.text((x,y),name,fill='white',font=font)
            canvas.paste(Image.fromarray(array).resize((panel,panel),Image.Resampling.NEAREST),(x,y+20))
    draw.text((gap,canvas.height-27),'Unknown regions are unvalidated; samples sharing a group are not separate locations.',fill='white',font=font)
    canvas.save(path)


def write_report(path, metadata):
    def fmt(value):
        return 'undefined' if value is None else f'{value:.6f}' if isinstance(value, float) else str(value)
    lines = ['# Frozen-checkpoint partial-ROI validation', '',
             '**Small assistant-reviewed development validation; not a representative benchmark or expert-certified ground truth.**', '',
             'The final training checkpoint (update 1664) and threshold .5 were fixed before these predictions. No retraining or tuning occurred.',
             '', '| Subset | Samples | Groups | TP / FP / FN / TN | Precision | Recall | Specificity | Dice | Reviewed stable FPR | Weighted BCE |',
             '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in ('loss_pairs', 'stable_pairs', 'pooled'):
        row = metadata['summaries'][name]; m = row['metrics']
        values = [name, row['sample_count'], row['group_count'], ' / '.join(str(m[k]) for k in ('tp','fp','fn','tn')),
                  m['precision'], m['recall'], m['specificity'], m['dice'], m['reviewed_stable_false_positive_rate'], row['weighted_bce']]
        lines.append('| '+' | '.join(map(fmt, values))+' |')
    lines += ['', '## Per geographic group', '', '| Group | Samples | TP / FP / FN / TN | Dice | Weighted BCE |',
              '| --- | ---: | --- | ---: | ---: |']
    for row in metadata['summaries']['groups']:
        m = row['metrics']; values = [row['group_id'], row['sample_count'], ' / '.join(str(m[k]) for k in ('tp','fp','fn','tn')),m['dice'],row['weighted_bce']]
        lines.append('| '+' | '.join(map(fmt, values))+' |')
    lines += ['', '## Per sample', '', '| Sample | Kind | Group | TP / FP / FN / TN | Dice | Weighted BCE |',
              '| --- | --- | --- | --- | ---: | ---: |']
    for row in metadata['examples']:
        m=row['metrics']; values=[row['sample_id'],row['kind'],row['group_id'],' / '.join(str(m[k]) for k in ('tp','fp','fn','tn')),m['dice'],row['weighted_bce']]
        lines.append('| '+' | '.join(map(fmt,values))+' |')
    lines += ['', 'Stable temporal pairs share their reviewed parent geographic groups. Up to six temporal samples represent at most three locations, not six independent sites.',
              'All ratios use reviewed pixels only. Undefined precision/recall/Dice remain undefined; all-negative stable-only success does not create a clearing-recall score.',
              'Reviewed stable FPR is limited to these explicit forest ROIs, not a scene-wide or population false-alarm rate.',
              'Unknown pixels remain ignored. Clearing boundaries, total area and exact felling dates are not validated.',
              'The earlier perfect training fit is memorization; validation scores below describe different reviewed locations.',
              'No minimum accuracy gate removes poor results; completed finite outcomes are retained.']
    Path(path).write_text('\n'.join(lines).rstrip()+'\n')


def evaluate(args, output):
    read_json(args.label_lock)  # No label lock: stop before loading any validation arrays.
    from verify_validation_dataset import load_and_verify
    protocol = read_json(args.protocol); dataset_meta, rows, samples = load_and_verify(args.dataset, args.protocol)
    bindings = verify_label_lock(args.dataset, args.protocol, args.label_lock, dataset_meta, protocol)
    require('torch' not in sys.modules,'Label locks must be checked before importing Torch in a fresh worker')
    implementations = {name:digest(ROOT/'Solutions'/name) for name in SOURCE_NAMES}
    evaluation = protocol['evaluation']; weights_path = rooted(evaluation['checkpoint'])
    from run_pilot_checks import configure_cpu
    torch = configure_cpu(42)
    import numpy as np
    from train_pilot_baseline import make_model, parameter_hash
    from summarize_pilot_overfit import checked_npz, WEIGHT_SHAPES
    arrays = checked_npz(weights_path, WEIGHT_SHAPES, {key:'float32' for key in WEIGHT_SHAPES})
    require(all(np.isfinite(value).all() for value in arrays.values()), 'Frozen weights must remain finite')
    model = make_model(); model.load_state_dict({key:torch.from_numpy(value.copy()) for key,value in arrays.items()})
    model.eval(); before=parameter_hash(model)
    training_receipt=ROOT/'Docs/Reviews/PixelPilotV1/run/metadata.json'
    require(digest(training_receipt)==evaluation['run_metadata_sha256']
            and before==read_json(training_receipt)['final_parameter_sha256'], 'Loaded parameter state differs from frozen final training checkpoint')
    require(sum(p.numel() for p in model.parameters())==2545 and all(p.device.type=='cpu' for p in model.parameters()),
            'Frozen small CPU model architecture changed')
    result, logits, probabilities, previews = _torch_results(torch,samples,rows,model,evaluation['training_positive_weight'])
    after=parameter_hash(model); require(before==after and all(p.grad is None for p in model.parameters()), 'Inference altered model parameters/gradients')
    np.savez_compressed(output/'predictions.npz',logits=logits,probabilities=probabilities,
        labels=np.stack([s['labels'] for s in samples]),valid=np.stack([s['valid'] for s in samples]),
        sample_ids=np.asarray([r['sample_id'] for r in rows]))
    for index in range(0,len(rows),4):
        render_validation_preview(output/f'preview_{index//4+1}.png',previews[index:index+4],list(probabilities[index:index+4]))
    verify_dataset_payload(args.dataset,dataset_meta)
    require(verify_label_lock(args.dataset,args.protocol,args.label_lock,dataset_meta,protocol)==bindings,
            'Locked inputs changed during inference')
    require(all(digest(ROOT/'Solutions'/name)==sha for name,sha in implementations.items()),'Implementation changed during inference')
    metadata={'execution_status':'completed','mode':'evaluate','scope':protocol['scope'],**bindings,**result,
        'sample_ids':[r['sample_id'] for r in rows],'sample_count':len(rows),'group_ids':sorted({r['group_id'] for r in rows}),
        'label_lock_verified_before_torch_import':True,
        'source_implementation_sha256':implementations,'model_executions':len(rows),'training_steps':0,
        'optimizer_created':False,'threshold_tuning':False,'checkpoint_attempt':1664,'probability_threshold':.5,
        'training_positive_weight':evaluation['training_positive_weight'],'model_parameters_sha256_before':before,
        'model_parameters_sha256_after':after,'model_state_unchanged':True,'gradients_created':False,
        'parameter_count':2545,'device':'cpu','gpu_used':False,'batch_size':1,'dataloader_workers':0,'native_threads':1,
        'torch_intraop_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
        'torch_version':str(torch.__version__),'torch_distribution_version':importlib.metadata.version('torch'),
        'torch_git_version':torch.version.git_version,'torch_module_path':str(Path(torch.__file__).resolve()),
        'accuracy_success_gate':None,'source_jp2_pixel_reads':0}
    write_report(output/'report.md',metadata)
    metadata['artifact_sha256']={p.name:digest(p) for p in output.iterdir() if p.is_file()}
    return metadata


def verify(args, output):
    read_json(args.label_lock)
    import numpy as np
    from verify_validation_dataset import load_and_verify
    from summarize_pilot_overfit import checked_npz, compare, resource_receipt, WEIGHT_SHAPES
    require('torch' not in sys.modules,'Independent validation verifier must not import Torch')
    protocol=read_json(args.protocol); dataset_meta,rows,samples=load_and_verify(args.dataset,args.protocol)
    bindings=verify_label_lock(args.dataset,args.protocol,args.label_lock,dataset_meta,protocol)
    run=read_json(args.run/'metadata.json');require(run['execution_status']=='completed' and run['mode']=='evaluate','Completed frozen evaluation required')
    compare(run,bindings,'Locked validation bindings')
    count=len(rows);expected_artifacts={'predictions.npz','report.md',*(f'preview_{i+1}.png' for i in range(math.ceil(count/4)))}
    require(set(run['artifact_sha256'])==expected_artifacts,'Saved evaluation artifact coverage changed')
    for name,sha in run['artifact_sha256'].items():require(digest(args.run/name)==sha,'Evaluation artifact changed: '+name)
    require(set(run['source_implementation_sha256'])==set(SOURCE_NAMES),'Evaluation source coverage changed')
    for name,sha in run['source_implementation_sha256'].items():require(digest(ROOT/'Solutions'/name)==sha,'Evaluation source changed: '+name)
    for field,value in (('model_executions',count),('training_steps',0),('optimizer_created',False),
            ('threshold_tuning',False),('checkpoint_attempt',1664),('probability_threshold',.5),
            ('parameter_count',2545),('model_state_unchanged',True),('gradients_created',False),('accuracy_success_gate',None),
            ('source_jp2_pixel_reads',0),('sample_count',count),
            ('training_positive_weight',protocol['evaluation']['training_positive_weight']),
            ('sample_ids',[r['sample_id'] for r in rows]),('group_ids',sorted({r['group_id'] for r in rows}))):
        compare(run[field],value,'Frozen inference receipt '+field)
    require(run['label_lock_verified_before_torch_import'] is True,'Label-lock ordering receipt changed')
    require(run['model_parameters_sha256_before']==run['model_parameters_sha256_after'],'Model state receipt changed')
    weights=checked_npz(rooted(protocol['evaluation']['checkpoint']),WEIGHT_SHAPES,{name:'float32' for name in WEIGHT_SHAPES})
    parameter_hash=hashlib.sha256()
    for name in sorted(weights):
        require(np.isfinite(weights[name]).all(),'Frozen final parameters must remain finite')
        parameter_hash.update(name.encode());parameter_hash.update(weights[name].tobytes())
    require(parameter_hash.hexdigest()==run['model_parameters_sha256_before'],'Model parameter receipt differs from frozen safe weights')
    resource=resource_receipt(run)
    arrays=checked_npz(args.run/'predictions.npz',
        {'logits':(count,64,64),'probabilities':(count,64,64),'labels':(count,64,64),'valid':(count,64,64),'sample_ids':(count,)},
        {'logits':'float32','probabilities':'float32','labels':'uint8','valid':'bool','sample_ids':'unicode'})
    require(arrays['sample_ids'].tolist()==[r['sample_id'] for r in rows]
        and np.array_equal(arrays['labels'],np.stack([s['labels'] for s in samples]))
        and np.array_equal(arrays['valid'],np.stack([s['valid'] for s in samples])), 'Saved prediction labels/order changed')
    require(np.isfinite(arrays['logits']).all() and np.isfinite(arrays['probabilities']).all(),'Full saved maps must remain finite')
    full_logits=arrays['logits'].astype('float64');expected_probability=np.empty_like(full_logits)
    nonnegative=full_logits>=0
    expected_probability[nonnegative]=1/(1+np.exp(-full_logits[nonnegative]))
    exponent=np.exp(full_logits[~nonnegative]);expected_probability[~nonnegative]=exponent/(1+exponent)
    require(np.allclose(arrays['probabilities'],expected_probability,rtol=0,atol=2e-7),
            'Full saved probability maps disagree with logits')
    result=numpy_summary(arrays['logits'],arrays['probabilities'],arrays['labels'],rows,protocol['evaluation']['training_positive_weight'])
    compare(run,result,'Independent validation reconstruction',tolerance=2e-6)
    verify_dataset_payload(args.dataset,dataset_meta)
    require('torch' not in sys.modules,'Validation verifier unexpectedly imported Torch')
    metadata={'execution_status':'completed','verification_status':'passed','mode':'verify',**bindings,**result,
        'run_metadata_sha256':digest(args.run/'metadata.json'),'run_artifact_sha256':run['artifact_sha256'],
        'run_resources':resource,'model_executions':0,'source_jp2_pixel_reads':0,'torch_imported':False,
        'training_steps':0,'scope':protocol['scope'],'all_accuracy_outcomes_retained':True,
        'verification_implementation_sha256':{name:digest(ROOT/'Solutions'/name) for name in SOURCE_NAMES}}
    write_report(output/'report.md',metadata)
    return metadata


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('evaluate','verify'),default='evaluate')
    parser.add_argument('--dataset',type=Path,default=ROOT/'Datasets/ValidationPilot/v1/dataset')
    parser.add_argument('--protocol',type=Path,default=ROOT/'Docs/Reviews/ValidationPilotV1/protocol.json')
    parser.add_argument('--label-lock',type=Path,default=ROOT/'Docs/Reviews/ValidationPilotV1/label_lock.json')
    parser.add_argument('--run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-rss-mib',type=int,default=512)
    parser.add_argument('--min-available-gib',type=float,default=1.5)
    parser.add_argument('--max-seconds',type=int,default=180)
    parser.add_argument('--worker-output',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.mode=='verify' and args.run is None:parser.error('--run is required in verify mode')
    if not (128<=args.max_rss_mib<=512 and args.min_available_gib>=1.5 and 0<args.max_seconds<=300):
        parser.error('Use 128–512 MiB RSS, >=1.5 GiB reserve and <=300 seconds')
    args.dataset=args.dataset.resolve();args.protocol=args.protocol.resolve();args.label_lock=args.label_lock.resolve()
    require(args.dataset.is_relative_to(ROOT) and args.protocol.is_relative_to(ROOT) and args.label_lock.is_relative_to(ROOT),
            'Validation data, protocol and lock must stay inside the project')
    for key in THREAD_VARIABLES:os.environ[key]='1'
    if args.worker_output:
        os.nice(10);metadata=(evaluate if args.mode=='evaluate' else verify)(args,args.worker_output)
        import resource
        metadata['worker_high_water_rss_mib']=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform=='darwin' else 1024),2)
        (args.worker_output/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    else:run_guarded(args,script_path=__file__,operation='validation-'+args.mode,gdal_cache_mib=16,warp_memory_mib=0)


if __name__=='__main__':main()
