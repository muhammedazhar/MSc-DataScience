#!/usr/bin/env python3
"""Publish reviewed validation-only partial ROIs before any model inference."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {'pair.png', 'timeline.png', 'target_mask.tif', 'positive_interior.tif',
             'proposed_control.tif', 'catalogue_exclusion.tif', 'endpoint_unusable.tif',
             'feature_valid.tif', 'input_features.tif'}
IMPLEMENTATION_NAMES = ('build_validation_pilot.py', 'validation_review.py', 'prepare_validation_review.py',
                        'curated_pilot_exports.py', 'curated_pilot_review.py', 'prepare_curated_pilot.py',
                        'assess_pixel_quality.py', 'validate_sample.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 32*2**20, 'Missing or oversized JSON metadata')
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def rooted(relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'A project-relative input path is required')
    path = (ROOT/relative).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.is_file(), 'Input path is missing or escaped the project')
    return path


def records(path):
    from build_curated_pilot import records as bounded_records
    yield from bounded_records(path)


def bind_case_artifacts(review):
    """Bind raw case bytes, canonical metadata and every actual review artifact."""
    case_path = rooted(review['case_file']); case = read_json(case_path)
    require(review.get('case_sha256') == digest(case_path), 'Raw prepared case bytes changed after review')
    require(review.get('prepared_case_sha256') == canonical(case), 'Canonical prepared case changed after review')
    require(case.get('split') == case.get('pair', {}).get('split') == 'validation'
            and case.get('training_approved') is False and case.get('evaluation_approved') is False,
            'Only unapproved validation evidence may be exported')
    require(re.fullmatch(r'event-[A-Za-z0-9-]+', case.get('event_id', '')) is not None
            and case_path.parent.name == case['event_id'], 'Prepared case event/directory identity differs')
    require(set(case.get('artifact_sha256', {})) == ARTIFACTS
            and review.get('artifact_sha256') == case['artifact_sha256'], 'All nine prepared artifacts must be review-bound')
    hashes = {review['case_file']: digest(case_path)}
    for name, checksum in case['artifact_sha256'].items():
        path = case_path.parent/name
        require(path.is_file() and digest(path) == checksum, 'Prepared review artifact changed: '+name)
        hashes[str(path.relative_to(ROOT))] = checksum
    return case, case_path.parent, hashes


def enforce_selection_policy(candidates, dispositions):
    """Alternates are usable only after their predeclared primary is rejected."""
    require(isinstance(candidates, list) and 1 <= len(candidates) <= 6, 'Predeclared selection must contain one to six candidates')
    selected = {}
    for item in candidates:
        require(item.get('role') in ('primary', 'alternate') and isinstance(item.get('stratum'), str), 'Predeclared candidate role/stratum required')
        key = (item['stratum'], item['role'])
        require(key not in selected and item['event_id'] not in {r['event_id'] for r in selected.values()}, 'Selection roles/events repeat')
        selected[key] = item
    require(len(dispositions) == len({row['event_id'] for row in dispositions}), 'Every reviewed event must have one disposition')
    by_event = {row['event_id']: row for row in dispositions}
    require(set(by_event) <= {item['event_id'] for item in candidates}, 'Review event is absent from the predeclared selection')
    accepted = []
    for stratum in sorted({item['stratum'] for item in candidates}):
        primary = selected.get((stratum, 'primary'))
        require(primary is not None and primary['event_id'] in by_event, 'Every predeclared primary needs a retained review disposition')
        p = by_event[primary['event_id']]
        if p['decision_status'] == 'accepted':
            accepted.append(primary['event_id'])
        alternate = selected.get((stratum, 'alternate'))
        if alternate and alternate['event_id'] in by_event and by_event[alternate['event_id']]['decision_status'] == 'accepted':
            require(p['decision_status'] == 'rejected', 'An alternate may be accepted only after its primary is rejected')
            accepted.append(alternate['event_id'])
    require(len(accepted) <= 3, 'At most three loss cases may be accepted')
    return accepted


def validation_supervision(kind, positive, stable, unusable, features, feature_valid, excluded):
    """Unknown pixels stay ignored; zero-positive supervision is stable-ROI only."""
    import numpy as np
    from curated_pilot_exports import build_supervision

    require(kind in ('loss', 'stable_roi'), 'Unsupported validation sample kind')
    for name, mask in (('positive', positive), ('stable', stable), ('unusable', unusable),
                       ('feature_valid', feature_valid), ('excluded', excluded)):
        require(isinstance(mask, np.ndarray) and mask.shape == (64, 64) and mask.dtype == np.dtype(bool),
                'Validation '+name+' must be a frozen 64x64 boolean mask')
    require(isinstance(features, np.ndarray) and features.shape == (27, 64, 64)
            and features.dtype == np.dtype('float32') and not np.isinf(features).any(), 'Original signed float32 feature grid required')
    require(int(stable.sum()) >= 64 and not (stable & excluded).any(), 'Reviewed stable ROI is insufficient or intersects catalogue exclusion')
    require((kind == 'loss' and int(positive.sum()) >= 4) or (kind == 'stable_roi' and not positive.any()),
            'Loss needs reviewed positives; stable-only samples must contain no positive labels')
    labels = build_supervision(positive, stable, unusable)
    reviewed = labels != 255
    require(feature_valid[reviewed].all() and np.isfinite(features[:, reviewed]).all(), 'Reviewed validation features must remain finite and valid')
    require(np.array_equal(feature_valid, np.isfinite(features).all(axis=0)), 'Feature validity disagrees with actual finite input values')
    return labels


def read_case_arrays(directory, pair):
    import numpy as np
    import rasterio
    from prepare_validation_review import CHANNELS, validate_grid

    grid = validate_grid(pair); arrays = {}
    for name in sorted(ARTIFACTS - {'pair.png', 'timeline.png'}):
        with rasterio.open(directory/name) as source:
            require(source.shape == (64, 64) and source.transform == grid['transform']
                    and source.crs == rasterio.crs.CRS.from_user_input(grid['crs']), 'Prepared validation array grid changed')
            if name == 'input_features.tif':
                require(source.count == 27 and source.dtypes == ('float32',)*27 and source.descriptions == CHANNELS,
                        'Prepared feature header/channel identity changed')
                arrays[name] = source.read()
            else:
                require(source.count == 1 and source.dtypes == ('uint8',), 'Prepared ROI array type changed')
                values = source.read(1)
                require(np.isin(values, (0, 1)).all(), 'Prepared ROI masks must be binary')
                arrays[name] = values.astype(bool)
    return arrays


def export_sample(output, sample_id, kind, event_id, pair, parent, features, feature_valid, positive,
                  stable, qa, excluded, validation, review_sha, prepared_sha, source_stats, policy):
    import numpy as np
    import rasterio
    from prepare_validation_review import CHANNELS, validate_grid
    from validate_sample import write_tif

    labels = validation_supervision(kind, positive, stable, qa, features, feature_valid, excluded)
    grid = validate_grid(pair); directory = output/'examples'/sample_id; directory.mkdir()
    paths = {'input_features': f'examples/{sample_id}/input_features.tif',
             'supervision_mask': f'examples/{sample_id}/supervision_mask.tif',
             'valid_mask': f'examples/{sample_id}/valid_mask.tif',
             'metadata': f'examples/{sample_id}/metadata.json'}
    arrays = {'input_features.tif': features, 'supervision_mask.tif': labels,
              'valid_mask.tif': (labels != 255).astype('uint8'), 'endpoint_unusable.tif': qa.astype('uint8'),
              'feature_valid.tif': feature_valid.astype('uint8')}
    for name, values in arrays.items():
        nodata = float('nan') if name == 'input_features.tif' else 255 if name == 'supervision_mask.tif' else None
        descriptions = CHANNELS if name == 'input_features.tif' else (name.removesuffix('.tif'),)
        write_tif(directory/name, values, grid, nodata, descriptions)
        with rasterio.open(directory/name) as source:
            require(source.shape == (64, 64) and source.transform == grid['transform']
                    and source.crs == rasterio.crs.CRS.from_user_input(grid['crs'])
                    and np.array_equal(source.read(), values if values.ndim == 3 else values[None], equal_nan=True),
                    'Export round-trip changed validation arrays/grid')
    counts = {'supervised_positive_pixels': int((labels == 1).sum()),
              'supervised_stable_pixels': int((labels == 0).sum()), 'ignored_pixels': int((labels == 255).sum())}
    base = {'sample_id': sample_id, 'kind': kind, 'event_id': event_id, 'group_id': pair['group_id'],
            'split': 'validation', 'pair': pair, 'frozen_parent_pair_id': parent['pair_id'],
            'grid': pair['grid'], 'pre_date': pair['pre_date'], 'post_date': pair['post_date'],
            'qa_policy_id': policy, 'training_approved': False, 'evaluation_approved': True,
            **counts, 'artifact_paths': paths, 'review_sha256': review_sha, 'prepared_case_sha256': prepared_sha}
    metadata = {**base, 'channels': list(CHANNELS), 'feature_source_file_stats': source_stats,
                'feature_handling': 'Original signed exported scales; nonfinite invalid context remains unknown and must be zero-filled only after excluding it from metrics',
                'decision': validation, 'derived_pair': kind == 'stable_roi',
                'lineage': {'frozen_parent_pair': parent, 'reviewed_event_id': event_id,
                            'stable_pair_rule': validation.get('stable_pair') if kind == 'stable_roi' else None},
                'extra_artifact_paths': {'endpoint_unusable': f'examples/{sample_id}/endpoint_unusable.tif',
                                         'feature_valid': f'examples/{sample_id}/feature_valid.tif'},
                'output_sha256': {name: digest(directory/name) for name in arrays},
                'label_values': {'0': 'reviewed stable-forest control ROI', '1': 'reviewed clearing interior', '255': 'unknown/ignore'},
                'approval_scope': 'Assistant-reviewed partial validation regions only; no training permission or complete-mask ground truth'}
    write_json(directory/'metadata.json', metadata)
    return base


def build(args, output):
    import numpy as np
    import rasterio
    from scipy.ndimage import binary_erosion
    from assess_pixel_quality import ObservationReader, model_identity, validate_source_stats
    from prepare_curated_pilot import unusable
    from prepare_validation_review import (catalogue_exclusion, read_validation_features,
                                           validate_quality_summary, validate_validation_pair)
    from validation_review import validate_validation_review

    protocol_path = args.protocol.resolve(); protocol = read_json(protocol_path)
    implementation = {name: digest(ROOT/'Solutions'/name) for name in IMPLEMENTATION_NAMES}
    require(protocol['status'] == 'predeclared_before_validation_predictions', 'Validation protocol must remain predeclared')
    require(protocol['resources']['worker_rss_mib'] == 512 and protocol['resources']['available_memory_reserve_gib'] == 1.5,
            'Frozen Mac resource protocol changed')
    for name, checksum in protocol['immutable_sources_sha256'].items():
        require(digest(ROOT/'Solutions'/name) == checksum, 'An immutable archived source changed: '+name)
    evaluation = protocol['evaluation']
    require(evaluation['checkpoint_attempt'] == 1664 and evaluation['probability_threshold'] == .5
            and evaluation['retraining'] is False and evaluation['threshold_tuning'] is False
            and evaluation['best_checkpoint_selection'] is False,
            'The fixed validation checkpoint/threshold protocol changed')
    require(digest(rooted(evaluation['checkpoint'])) == evaluation['weights_sha256'], 'Frozen model weights changed before label publication')
    reviews_path = args.reviews.resolve(); raw = read_json(reviews_path)
    reviews = raw.get('reviews')
    require(isinstance(reviews, list) and 1 <= len(reviews) <= 6 and all(isinstance(r, dict) for r in reviews),
            'One to six explicit prepared review dispositions required')
    prepared_files = raw.get('prepared_case_files')
    require(isinstance(prepared_files, list) and len(prepared_files) == len(set(prepared_files))
            and set(prepared_files) == {review['case_file'] for review in reviews},
            'Every prepared case must have one retained review disposition')
    selection_path = rooted(raw['selection_file']); require(digest(selection_path) == raw['selection_sha256'], 'Predeclared selection changed')
    selection = read_json(selection_path); candidates = selection['candidates']
    selected = {item['event_id']: item for item in candidates}
    require(len(selected) == len(candidates), 'Selected event identities repeat')
    require({item['stratum'] for item in candidates} <= set(protocol['selection']['strata'])
            and {item['stratum'] for item in candidates if item['role'] == 'primary'} == set(protocol['selection']['strata']),
            'Every protocol stratum must retain its predeclared primary')
    audit = ROOT/'Datasets/AnnotationAudit/v1'; quality_root = ROOT/'Datasets/PixelQA/v1'
    summary_path = quality_root/'summary/metadata.json'; summary = read_json(summary_path)
    validate_quality_summary(summary)
    require(summary['config_id'] == protocol['qa_policy_id'], 'Completed frozen QA policy changed')
    validate_source_stats(summary)
    catalogue = read_json(audit/'annotations_clean.geojson'); splits = read_json(audit/'splits.json')
    training_path = ROOT/'Datasets/CuratedPilot/v1/dataset/training_manifest.jsonl'
    training = list(records(training_path))
    train_plan = read_json(ROOT/'Docs/Reviews/PixelPilotV1/experiment_plan.json')
    require(digest(training_path) == train_plan['selection']['manifest_sha256'], 'The exact completed training manifest changed')
    pairs = {row['pair_id']: row for row in records(audit/'candidate_pairs.jsonl') if row['pair_id'] in {c['pair_id'] for c in candidates}}
    require(set(pairs) == {item['pair_id'] for item in candidates}, 'A predeclared validation candidate is absent from the frozen manifest')
    selected_groups = set()
    for item in candidates:
        pair = pairs[item['pair_id']]
        validate_validation_pair(pair, item['event_id'], catalogue, splits, training)
        require(pair['group_id'] not in selected_groups, 'Predeclared validation candidates repeat a geographic group')
        selected_groups.add(pair['group_id'])
    identities = {str(path.relative_to(ROOT)): digest(path) for path in
                  (selection_path, protocol_path, reviews_path, summary_path, audit/'candidate_pairs.jsonl',
                   audit/'metadata.json', audit/'annotations_clean.geojson', audit/'splits.json', training_path)}
    all_hashes, source_stats, assembled, dispositions = {}, {}, [], []
    prepared_seen = set()
    for review in reviews:
        case, directory, hashes = bind_case_artifacts(review)
        event_id = case['event_id']; require(event_id in selected and event_id not in prepared_seen, 'Review event is unselected or duplicated')
        prepared_seen.add(event_id); item = selected[event_id]
        pair = pairs.get(item['pair_id'])
        require(pair is not None and case['pair'] == pair and case['pair_id'] == item['pair_id']
                and case.get('stratum') == item['stratum'], 'Prepared validation case differs from exact predeclared frozen pair')
        validate_validation_pair(pair, event_id, catalogue, splits, training)
        batch_file = review.get('prepared_batch_file', str((directory.parent/'metadata.json').relative_to(ROOT)))
        batch_path = rooted(batch_file); batch = read_json(batch_path)
        require(batch['mode'] == 'prepare_validation_review' and batch['config_id'] == protocol['qa_policy_id']
                and batch['implementation_sha256'] == digest(ROOT/'Solutions/prepare_validation_review.py')
                and any(row['event_id'] == event_id and row['pair_id'] == pair['pair_id'] for row in batch['cases']),
                'Prepared batch receipt does not bind this validation case/policy/implementation')
        expected = {'candidate_manifest_sha256': digest(audit/'candidate_pairs.jsonl'),
                    'annotation_catalogue_sha256': digest(audit/'annotations_clean.geojson'),
                    'split_plan_sha256': digest(audit/'splits.json'), 'audit_metadata_sha256': digest(audit/'metadata.json'),
                    'quality_summary_sha256': digest(summary_path), 'selection_sha256': digest(selection_path)}
        require(all(batch.get(key) == value for key, value in expected.items()), 'Prepared batch has stale frozen provenance')
        for path, checksum in batch['source_identity_sha256'].items():
            require(digest(rooted(path)) == checksum, 'Prepared source identity changed: '+path)
            identities[path] = checksum
        all_hashes.update(hashes); identities[batch_file] = digest(batch_path)
        for path, values in batch['source_file_stats'].items():
            require(path not in source_stats or source_stats[path] == values, 'Prepared batches disagree on source fingerprints')
            source_stats[path] = values
        decision = validate_validation_review(case, review)
        dispositions.append({'event_id': event_id, 'pair_id': pair['pair_id'], 'group_id': pair['group_id'],
                             'stratum': item['stratum'], 'role': item['role'], 'decision_status': decision['decision_status'],
                             'review': review, 'validation': decision})
        assembled.append((case, directory, pair, review, decision))
    accepted = enforce_selection_policy(candidates, dispositions)
    require(bool(accepted),
            'No validation labels were approved; use reviews-only verification instead of publishing an empty dataset')
    accepted_assembled = [row for row in assembled if row[0]['event_id'] in accepted]
    require(len({row[2]['group_id'] for row in accepted_assembled}) == len(accepted), 'Accepted validation groups repeat')
    validate_source_stats({'source_file_stats': source_stats})
    (output/'examples').mkdir(); rows = []; reader = None
    with rasterio.Env(GDAL_CACHEMAX=32*2**20, GDAL_NUM_THREADS='1'):
        for case, directory, pair, review, decision in accepted_assembled:
            arrays = read_case_arrays(directory, pair)
            target, positive, stable = (arrays[name] for name in ('target_mask.tif', 'positive_interior.tif', 'proposed_control.tif'))
            require(np.array_equal(positive, binary_erosion(target, structure=np.ones((3, 3), dtype=bool))),
                    'Reviewed clearing interior differs from exact target erosion')
            excluded = arrays['catalogue_exclusion.tif']
            require(np.array_equal(excluded, catalogue_exclusion(pair)), 'Prepared catalogue exclusion changed')
            features, valid, qa = (arrays[name] for name in ('input_features.tif', 'feature_valid.tif', 'endpoint_unusable.tif'))
            event_id = case['event_id']; review_sha = canonical(review); prepared_sha = canonical(case)
            rows.append(export_sample(output, 'loss-'+event_id, 'loss', event_id, pair, pair, features, valid,
                                      positive, stable, qa, excluded, decision, review_sha, prepared_sha,
                                      case['feature_source_file_stats'], protocol['qa_policy_id']))
            derived = decision['stable_pair']
            if derived is None:
                continue
            timeline = {row['scene_id']: row for row in case['timeline']}
            negative_pair = {**deepcopy(pair), **{key: derived[key] for key in ('pre_scene_id', 'post_scene_id', 'pre_date', 'post_date')}}
            negative_pair['pair_id'] = 'validation-stable-'+canonical({'parent_pair_id': pair['pair_id'], 'event_id': event_id,
                'pre_scene_id': derived['pre_scene_id'], 'post_scene_id': derived['post_scene_id'], 'grid': pair['grid']})[:20]
            negative_pair['interval_days'] = (date.fromisoformat(derived['post_date'])-
                                              date.fromisoformat(derived['pre_date'])).days
            for side in ('pre', 'post'):
                negative_pair[side+'_path'] = timeline[derived[side+'_scene_id']]['scene_path']
            negative_pair.update({'kind': 'derived_pre_event_stable_roi', 'frozen_parent_pair_id': pair['pair_id'],
                                  'not_an_original_frozen_candidate': True, 'label_event_ids_by_post_date': [],
                                  'status': 'derived_validation_pair_reviewed_stable_roi',
                                  'pixel_quality': 'reviewed_control_roi_only', 'training_approved': False,
                                  'label_interpretation': 'Reviewed stable control ROI only; unknown pixels ignored'})
            if reader is None:
                reader = ObservationReader(ROOT/'Datasets/PixelQA/cache', model_identity())
                require(reader.config_id == protocol['qa_policy_id'], 'Derived pair QA policy changed')
            before = reader.observation(negative_pair['pre_path'], negative_pair['grid'])
            after = reader.observation(negative_pair['post_path'], negative_pair['grid'])
            negative_qa = unusable(before) | unusable(after)
            require(not (negative_qa & stable).any(), 'Derived stable ROI intersects actual endpoint QA flags')
            negative_features, negative_valid, negative_stats = read_validation_features(negative_pair, summary)
            for name, values in negative_stats.items():
                require(name not in source_stats or source_stats[name] == values,
                        'Derived validation source fingerprints contradict prepared evidence')
                source_stats[name] = values
            rows.append(export_sample(output, 'stable-'+event_id, 'stable_roi', event_id, negative_pair, pair,
                negative_features, negative_valid, np.zeros((64, 64), dtype=bool), stable, negative_qa, excluded,
                decision, review_sha, prepared_sha, negative_stats, protocol['qa_policy_id']))
    require(sum(row['kind'] == 'loss' for row in rows) <= 3 and sum(row['kind'] == 'stable_roi' for row in rows) <= 3,
            'Predeclared validation size exceeded')
    (output/'validation_manifest.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))
    (output/'review_decisions.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in dispositions))
    validate_source_stats({'source_file_stats': source_stats}); validate_source_stats(summary)
    for path, checksum in {**identities, **all_hashes}.items():
        require(digest(rooted(path)) == checksum, 'Frozen validation evidence/source changed during export: '+path)
    for name, checksum in protocol['immutable_sources_sha256'].items():
        require(digest(ROOT/'Solutions'/name) == checksum, 'Archived source changed during validation export')
    require(all(digest(ROOT/'Solutions'/name) == checksum for name, checksum in implementation.items()),
            'Validation export implementation changed during execution')
    totals = Counter()
    for row in rows:
        for key in ('supervised_positive_pixels', 'supervised_stable_pixels', 'ignored_pixels'):
            totals[key] += row[key]
    (output/'report.md').write_text('# Reviewed partial-ROI validation dataset\n\n'
        f"Published {len(accepted)} loss cases and {sum(row['kind']=='stable_roi' for row in rows)} derived stable-ROI pairs "
        f"from {len(accepted)} distinct validation geographic groups. Retained {len(dispositions)} review dispositions.\n\n"
        'No model predictions were read, no training was run, and all unreviewed pixels remain 255. '
        'Assistant review is not expert-certified ground truth or complete clearing-boundary annotation.\n')
    metadata = {'schema_version': 1, 'ready_for_partial_validation': bool(rows), 'training_approved': False,
        'evaluation_approved': bool(rows), 'sample_count': len(rows), 'loss_case_count': len(accepted),
        'stable_pair_count': sum(row['kind']=='stable_roi' for row in rows),
        'group_ids': [row[2]['group_id'] for row in accepted_assembled], 'review_count': len(dispositions),
        'reviews_file': str(reviews_path.relative_to(ROOT)), 'reviews_sha256': digest(reviews_path),
        'protocol_file': str(protocol_path.relative_to(ROOT)), 'protocol_sha256': digest(protocol_path),
        'selection_file': str(selection_path.relative_to(ROOT)), 'selection_sha256': digest(selection_path),
        'model_weights_sha256': evaluation['weights_sha256'], 'qa_policy_id': protocol['qa_policy_id'],
        'supervision_pixel_counts': dict(totals), 'source_file_stats': source_stats,
        'source_identity_sha256': {**identities, **all_hashes},
        'implementation_sha256': implementation,
        'scope': protocol['scope'], 'model_predictions_read': False, 'forest_model_training_run': False,
        'checks': {'explicit_validation_only': True, 'all_prepared_review_dispositions_retained': True,
                   'exact_frozen_parent_pairs': True, 'primary_alternate_policy_preserved': True,
                   'no_train_test_conservative_footprint_overlap': True, 'no_actual_training_crop_overlap': True,
                   'unknown_pixels_ignored': True, 'stable_pair_lineage_explicit': True,
                   'source_statistics_unchanged': True, 'arrays_round_trip_equal': True},
        'output_sha256_excluding_this_metadata': {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob('*')) if path.is_file()}}
    write_json(output/'metadata.json', metadata)
    print(f'[validation-export] completed {len(rows)} samples; {len(accepted)} distinct groups', flush=True)


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationPilotV1/protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.reviews.resolve().is_relative_to(ROOT) and args.protocol.resolve().is_relative_to(ROOT), 'Review/protocol files must stay within project')
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Use 128–512 MiB RSS, >=1.5 GiB reserve and <=300s')
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        os.nice(10); build(args, args.worker_output)
        import resource
        metadata = read_json(args.worker_output/'metadata.json')
        metadata['worker_high_water_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024), 2)
        write_json(args.worker_output/'metadata.json', metadata)
    else:
        run_guarded(args, script_path=__file__, operation='validation-export', gdal_cache_mib=32, warp_memory_mib=16)


if __name__ == '__main__':
    main()
