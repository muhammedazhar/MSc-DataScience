#!/usr/bin/env python3
"""Audit every frozen validation QA anchor and lock a fresh review shortlist.

Only vector annotations, metadata and published QA masks are read. No satellite
bands, cloud models, deforestation predictions or new image previews are used.
"""

import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STRATA = ('36UXA_time-dependent', '36UYA_Summer_time-dependent', '36UYA_Spring_time-dependent')
PASSED = 'passed_automatic_screen_pending_visual_review'
SOURCE_NAMES = ('audit_validation_support_pool.py', 'audit_validation_geometry.py',
                'prepare_validation_review.py', 'select_validation_cases.py',
                'assess_pixel_quality.py', 'validate_sample.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    from prepare_validation_review import read_json as bounded_json
    return bounded_json(path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def validate_protocol(protocol):
    require(protocol['schema_version'] == 2 and protocol['status'] == 'predeclared_before_geometry_audit_and_new_imagery',
            'Predeclared broader audit protocol required')
    rule = protocol['rule']
    require(rule['erosion_pixels'] == 1 and rule['minimum_interior_pixels'] == 4
            and rule['followup_window_days'] == 45 and rule['followup_same_year'] is True
            and rule['primaries_per_stratum'] == rule['alternates_per_stratum_at_most'] == 1
            and rule['maximum_shortlist'] == 6 and rule['source_strata'] == list(STRATA)
            and rule['ranking'] == ['group_id', 'event_id', 'post_date_not_annotation_date', 'interval_days', 'pair_id'],
            'Declared support, follow-up, ranking or roster bounds changed')
    for name in ('require_automatic_pass', 'require_zero_endpoint_crop_and_target_qa',
                 'require_zero_actual_interior_qa', 'require_full_footprint_isolation',
                 'require_later_metadata_usable_scene', 'exclude_prior_events',
                 'exclude_prior_groups', 'no_fallback_when_slots_missing'):
        require(rule[name] is True, 'Declared eligibility gate changed: '+name)
    require(rule['ranking_uses_ndvi_or_model_scores'] is False
            and rule['shortlist_approves_labels'] is False and rule['new_imagery_reviewed'] is False,
            'This audit cannot approve labels, inspect imagery or use model rankings')
    resource = protocol['resources']
    require(resource == {'workers': 1, 'native_threads': 1, 'gpu_used': False, 'nice_increment': 10,
            'worker_rss_mib': 512, 'available_memory_reserve_gib': 1.5,
            'max_seconds_per_worker': 300, 'maximum_validation_pairs': 2000,
            'maximum_anchor_records': 10000}, 'Declared resource or coverage bounds changed')


def enumerate_validation_anchors(pairs, quality_rows, index_rows, policy):
    """Strict exact joins, without quality/history/size/proposal filtering."""
    def keyed(rows):
        result = {}
        for row in rows:
            require(row['pair_id'] not in result, 'Repeated frozen pair identity')
            result[row['pair_id']] = row
        return result
    frozen, quality, index = map(keyed, (pairs, quality_rows, index_rows))
    validation = {k: p for k, p in frozen.items() if p['split'] == 'validation'}
    wanted = (set(quality) | set(index)) & set(validation)
    wanted |= {r['pair_id'] for r in (*quality.values(), *index.values()) if r['split'] == 'validation'}
    require(wanted <= set(validation) and wanted <= set(quality) and wanted <= set(index),
            'Frozen validation QA/index/candidate coverage disagrees')
    result = []
    identity = ('pair_id', 'group_id', 'split', 'tile', 'pre_date', 'post_date', 'anchor_event_ids')
    for pid in sorted(wanted):
        pair, qa, receipt = validation[pid], quality[pid], index[pid]
        require(all(pair[k] == qa[k] == receipt[k] for k in identity)
                and qa['config_id'] == receipt['config_id'] == policy
                and qa['quality_status'] == receipt['quality_status']
                and qa['training_approved'] is False and receipt['training_approved'] is False,
                'Frozen validation QA/index identity or policy changed')
        anchors = pair['anchor_event_ids']
        require(isinstance(anchors, list) and anchors and len(anchors) == len(set(anchors)),
                'Frozen validation anchors must be nonempty and unique')
        for event_id in anchors:
            result.append({'event_id': event_id, 'pair': pair, 'qa': qa, 'index': receipt})
    return result


def followup_scene_ids(pair, annotation_date, scenes):
    observed, after = date.fromisoformat(annotation_date), date.fromisoformat(pair['post_date'])
    selected = [s for s in scenes if s['tile'] == pair['tile'] and s.get('metadata_usable') is True
                and after < date.fromisoformat(s['acquisition_date']) <= observed+timedelta(days=45)
                and date.fromisoformat(s['acquisition_date']).year == observed.year]
    return [s['scene_id'] for s in sorted(selected, key=lambda s: (s['acquisition_date'], s['scene_id']))]


def eligibility_reasons(row, prior_event_ids, prior_group_ids=()):
    require(row['split'] == 'validation', 'Only validation rows can enter this shortlist')
    reasons = []
    if row['qa_status'] != PASSED:
        reasons.append('not_automatic_quality_pass')
    if row['positive_interior_pixels'] < 4:
        reasons.append('anchor_interior_below_four_pixels')
    if any(row['endpoint_crop_unusable_pixels']):
        reasons.append('endpoint_crop_qa_flags')
    if any(row['endpoint_target_unusable_pixels']):
        reasons.append('endpoint_target_qa_flags')
    if row['interior_unusable_pixels']:
        reasons.append('actual_interior_qa_flags')
    if row['footprint_isolated'] is not True:
        reasons.append('full_feature_footprint_not_isolated')
    if not row['later_usable_scene_ids']:
        reasons.append('no_later_metadata_usable_observation')
    if row['stratum'] not in STRATA:
        reasons.append('source_stratum_missing_or_ambiguous')
    if row['event_id'] in prior_event_ids:
        reasons.append('previously_rejected_event')
    if row['group_id'] in prior_group_ids:
        reasons.append('previously_reviewed_geographic_group')
    return reasons


def choose_shortlist(rows, prior_event_ids, prior_group_ids=()):
    eligible = [r for r in rows if not eligibility_reasons(r, prior_event_ids, prior_group_ids)]
    eligible.sort(key=lambda r: (r['group_id'], r['event_id'], r['post_date'] != r['annotation_date'],
                                 r['interval_days'], r['pair_id']))
    fields = ('event_id', 'pair_id', 'group_id', 'stratum', 'split', 'pre_date', 'post_date',
              'interval_days', 'grid', 'annotation_date', 'positive_interior_pixels',
              'anchor_pixels', 'anchor_interior_pixels', 'target_pixels', 'temporal_union_pixels',
              'temporal_clipped_pixels', 'target_mask_sha256', 'positive_interior_sha256', 'qa_policy_id')
    candidates, missing, groups = [], [], set()
    for role in ('primary', 'alternate'):
        for stratum in STRATA:
            options = [r for r in eligible if r['stratum'] == stratum and r['group_id'] not in groups]
            if not options:
                missing.append({'stratum': stratum, 'role': role,
                                'reason': 'no_eligible_unreviewed_unused_geographic_group'})
                continue
            row = options[0]; groups.add(row['group_id'])
            candidates.append({**{k: row[k] for k in fields if k in row}, 'role': role,
                'rank': len(candidates)+1, 'observation_date': row['annotation_date'],
                'training_approved': False, 'evaluation_approved': False,
                'visual_review': 'unreviewed', 'temporal_label_review': 'unreviewed'})
    return {'candidates': candidates, 'missing_slots': missing,
            'eligible_anchor_pair_count': len(eligible),
            'eligible_event_count': len({r['event_id'] for r in eligible}),
            'eligible_group_count': len({r['group_id'] for r in eligible})}


def coverage(rows):
    return {'anchor_pair_count': len(rows), 'pair_count': len({r['pair_id'] for r in rows}),
            'event_count': len({r['event_id'] for r in rows}),
            'group_count': len({r['group_id'] for r in rows}),
            'anchor_pairs_with_interior_at_least_four': sum(r['positive_interior_pixels'] >= 4 for r in rows),
            'events_with_interior_at_least_four': len({r['event_id'] for r in rows if r['positive_interior_pixels'] >= 4})}


def raster_support(event_id, pair, events, published):
    import numpy as np
    from scipy.ndimage import binary_erosion
    from assess_pixel_quality import core_grid, label_target
    from validate_sample import burn
    # Reconstruct the temporal reference from the catalogue. The independent
    # verifier instead uses the published frozen target plane for this step.
    grid = core_grid(pair['grid'])
    require(pair['grid']['crs'] == 'EPSG:32636', 'Broader audit requires the frozen projected CRS')
    anchor = burn([events[event_id]['geometry']], grid).astype(bool)
    temporal = label_target(pair, events).astype(bool)
    require(np.array_equal(temporal, published[5].astype(bool)), 'Published temporal reference differs from frozen catalogue')
    target = anchor & temporal
    interior = binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
    unusable = np.any(published[:5].astype(bool), axis=0)
    return {'anchor_pixels': int(anchor.sum()),
            'anchor_interior_pixels': int(binary_erosion(anchor, structure=np.ones((3, 3), dtype=bool)).sum()),
            'temporal_union_pixels': int(temporal.sum()), 'target_pixels': int(target.sum()),
            'temporal_clipped_pixels': int(anchor.sum()-target.sum()),
            'positive_interior_pixels': int(interior.sum()),
            'interior_unusable_pixels': int((interior & unusable).sum()),
            'target_mask_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
            'positive_interior_sha256': hashlib.sha256(interior.tobytes()).hexdigest()}


def worker(args, output):
    os.nice(10)
    require('torch' not in sys.modules, 'Support audit must not import Torch')
    import numpy as np
    import rasterio
    from affine import Affine
    from assess_pixel_quality import load_events
    from prepare_validation_review import records, validate_grid, validate_quality_summary
    from select_validation_cases import grid_bounds, touches

    protocol_path = args.protocol.resolve(); protocol = read_json(protocol_path)
    validate_protocol(protocol)
    protocol_sha = digest(protocol_path)
    sources, source_hashes = {}, {str(protocol_path.relative_to(ROOT)): protocol_sha}
    for name, item in protocol['source_files'].items():
        path = (ROOT/item['path']).resolve()
        require(path.is_relative_to(ROOT) and digest(path) == item['sha256'], 'Frozen input changed: '+name)
        sources[name] = path; source_hashes[item['path']] = item['sha256']
    for name, sha in protocol['immutable_sources_sha256'].items():
        require(digest(ROOT/'Solutions'/name) == sha, 'Immutable archived source changed: '+name)
    implementation = {n: digest(ROOT/'Solutions'/n) for n in SOURCE_NAMES}
    summary = read_json(sources['quality_summary']); validate_quality_summary(summary)
    require(summary['config_id'] == protocol['qa_policy_id']
            and summary['candidate_manifest_sha256'] == digest(sources['pairs'])
            and summary['annotation_catalogue_sha256'] == digest(sources['catalogue'])
            and summary['audit_metadata_sha256'] == digest(sources['audit_metadata']), 'Frozen QA/audit identities disagree')
    candidates = [p for p in records(sources['pairs']) if p['split'] == 'validation']
    require(0 < len(candidates) <= protocol['resources']['maximum_validation_pairs'], 'Validation pair bound exceeded')
    quality = [q for q in records(sources['quality']) if q['split'] == 'validation']
    indices = [q for q in records(sources['quality_index']) if q['split'] == 'validation']
    joined = enumerate_validation_anchors(candidates, quality, indices, protocol['qa_policy_id'])
    require(0 < len(joined) <= protocol['resources']['maximum_anchor_records'], 'Validation anchor bound exceeded')
    catalogue, splits = read_json(sources['catalogue']), read_json(sources['splits'])
    props = {f['id']: f['properties'] for f in catalogue['features']}
    assignments = {eid: (g['group_id'], g['split']) for g in splits['groups'] for eid in g['event_ids']}
    require(len(assignments) == len(props), 'Frozen split/catalogue coverage differs')
    for eid, prop in props.items():
        require(assignments[eid] == (prop['group_id'], prop['split']), 'Frozen group allocation changed')
    train_plan = read_json(ROOT/'Docs/Reviews/PixelPilotV1/experiment_plan.json')
    require(digest(sources['training_manifest']) == train_plan['selection']['manifest_sha256']
            and digest(sources['training_metadata']) == train_plan['selection']['dataset_metadata_sha256'],
            'The exact trained thirteen-example reference changed')
    training = list(records(sources['training_manifest']))
    require(len(training) == 13, 'All thirteen actual training grids are required')
    barriers = [p['group_footprint_bounds'] for p in props.values() if p['split'] in ('train', 'test')]
    training_bounds = [grid_bounds(t['grid'], splits['crs']) for t in training]
    history = read_json(sources['prior_verification']); previous = read_json(sources['prior_reviews'])
    require(history['verification_status'] == 'passed' and history['accepted_loss_case_count'] == 0
            and history['model_executions'] == 0 and history['reviews_sha256'] == digest(sources['prior_reviews']),
            'Prior rejected review history changed')
    prior_events = {d['event_id'] for d in history['review_dispositions']}
    prior_groups = {d['group_id'] for d in history['review_dispositions']}
    require(prior_events == {r['event_id'] for r in previous['reviews']}, 'Prior event rejection coverage differs')
    scenes = list(records(sources['scenes'])); events = load_events(sources['catalogue'].parent)
    rows, mask_hashes, published_by_pair = [], {}, {}
    # One tiny QA file at a time; no full-scene, cloud-model or feature-band reads.
    with rasterio.Env(GDAL_CACHEMAX=16*2**20, GDAL_NUM_THREADS='1'):
        for joined_row in joined:
            event_id, pair, qa, index = (joined_row[k] for k in ('event_id', 'pair', 'qa', 'index'))
            validate_grid(pair)
            require(not pair['blocking_reasons'] and pair['status'] == 'candidate_for_pixel_review', 'A completed QA pair is blocked')
            prop = props[event_id]
            require((prop['group_id'], prop['split']) == (pair['group_id'], 'validation'), 'Frozen anchor group/split changed')
            observed = prop['observation_date']
            require(pair['pre_date'] < observed <= pair['post_date'], 'Frozen pair does not bracket its annotation date')
            bounds = grid_bounds(pair['grid'], splits['crs'])
            isolated = (not any(touches(bounds, b) for b in barriers)
                        and not any(touches(bounds, b) for b in training_bounds)
                        and all(t['group_id'] != pair['group_id'] for t in training))
            mask_relative = index['batch_path']+'/'+index['qa_mask_file']; mask_path = (ROOT/mask_relative).resolve()
            require(mask_path.is_relative_to(ROOT) and mask_path.is_file(), 'Published QA path escaped the project')
            mask_sha = digest(mask_path)
            if pair['pair_id'] in published_by_pair:
                require(published_by_pair[pair['pair_id']] == mask_relative, 'QA mask path changed between anchors')
            published_by_pair[pair['pair_id']] = mask_relative; mask_hashes[mask_relative] = mask_sha
            with rasterio.open(mask_path) as source:
                require(source.shape == (64, 64) and source.count == 6 and source.dtypes == ('uint8',)*6
                        and source.transform == Affine(*pair['grid']['transform'])
                        and source.crs == rasterio.crs.CRS.from_user_input(pair['grid']['crs']), 'Published QA grid/type changed')
                published = source.read()
            require(np.isin(published, (0, 1)).all(), 'Published QA mask must remain binary')
            target, unusable = published[5].astype(bool), np.any(published[:5].astype(bool), axis=0)
            require(int(target.sum()) == qa['target_metrics']['region_pixels']
                    and int(unusable.sum()) == qa['crop_metrics']['unusable_pixels']
                    and int((unusable & target).sum()) == qa['target_metrics']['unusable_pixels'],
                    'Published QA counts disagree with frozen numeric receipts')
            strata = [s for s in STRATA if s in prop['temporal_source_ids']]
            row = {'event_id': event_id, 'pair_id': pair['pair_id'], 'group_id': pair['group_id'],
                'stratum': strata[0] if len(strata) == 1 else None, 'split': 'validation',
                'annotation_date': observed, 'pre_date': pair['pre_date'], 'post_date': pair['post_date'],
                'interval_days': pair['interval_days'], 'grid': pair['grid'], 'qa_policy_id': protocol['qa_policy_id'],
                'qa_status': qa['quality_status'],
                'endpoint_crop_unusable_pixels': [qa['per_date'][s]['crop']['unusable_pixels'] for s in ('pre', 'post')],
                'endpoint_target_unusable_pixels': [qa['per_date'][s]['target']['unusable_pixels'] for s in ('pre', 'post')],
                'footprint_isolated': isolated, 'later_usable_scene_ids': followup_scene_ids(pair, observed, scenes),
                'published_qa_file': mask_relative, 'published_qa_sha256': mask_sha,
                **raster_support(event_id, pair, events, published),
                'training_approved': False, 'evaluation_approved': False}
            row['eligibility_reasons'] = eligibility_reasons(row, prior_events, prior_groups)
            row['eligible_for_fresh_review_shortlist'] = not row['eligibility_reasons']
            rows.append(row)
    rows.sort(key=lambda r: (r['event_id'], r['pair_id']))
    shortlist = choose_shortlist(rows, prior_events, prior_groups)
    screened_ids = {r['pair_id'] for r in rows}
    unprocessed = [{'pair_id': p['pair_id'], 'group_id': p['group_id'], 'anchor_event_ids': p['anchor_event_ids'],
                    'status': p['status'], 'blocking_reasons': p['blocking_reasons'],
                    'disposition': 'no_completed_published_qa', 'positive_interior_pixels': None}
                   for p in candidates if p['pair_id'] not in screened_ids]
    result = {'schema_version': 2, 'audit_status': 'passed', 'scope': protocol['scope'],
        'protocol_sha256': protocol_sha, 'records': rows, 'unscreened_candidates': unprocessed,
        'prior_rejected_event_ids': sorted(prior_events), 'prior_reviewed_group_ids': sorted(prior_groups),
        'coverage': coverage(rows), 'validation_candidate_pair_count': len(candidates),
        'unscreened_candidate_pair_count': len(unprocessed),
        'quality_status_counts': dict(Counter(r['quality_status'] for r in quality)),
        'by_stratum': {s: coverage([r for r in rows if r['stratum'] == s]) for s in STRATA},
        'eligible_before_history': coverage([r for r in rows if not eligibility_reasons(r, (), ())]),
        'eligible_after_history': coverage([r for r in rows if r['eligible_for_fresh_review_shortlist']]),
        'exclusion_reason_counts': dict(Counter(reason for r in rows for reason in r['eligibility_reasons'])),
        'source_sha256': {**source_hashes, **mask_hashes}, 'implementation_sha256': implementation,
        'source_jp2_pixel_reads': 0, 'qa_mask_files_read': len(mask_hashes),
        'model_executions': 0, 'new_imagery_reviewed': 0, 'labels_approved': 0}
    write_json(output/'audit.json', result)
    write_json(output/'selection.json', {'schema_version': 2, 'scope': protocol['scope'], 'rule': protocol['rule'],
        'protocol_sha256': protocol_sha, 'source_audit_sha256': digest(output/'audit.json'),
        'selection_uses_model_predictions': False, 'source_jp2_pixel_reads': 0, 'published_qa_masks_read': len(mask_hashes),
        'model_executions': 0, 'training_approved': False, 'evaluation_approved': False, **shortlist})
    lines = ['# Broader validation support audit and fixed shortlist', '', protocol['scope'], '',
        f"Audited {len(rows)} anchor-pairs from {len(quality)} QA-screened validation pairs; retained {len(unprocessed)} unscreened candidates separately.",
        f"Geometry support >=4 pixels: {result['coverage']['anchor_pairs_with_interior_at_least_four']} anchor-pairs; fresh-review eligible: {shortlist['eligible_anchor_pair_count']}.", '',
        '| Source | Audited anchor-pairs | Events | Interior >=4 |', '| --- | ---: | ---: | ---: |']
    for s, counts in result['by_stratum'].items():
        lines.append(f"| {s} | {counts['anchor_pair_count']} | {counts['event_count']} | {counts['anchor_pairs_with_interior_at_least_four']} |")
    lines += ['', '## Locked shortlist', '', '| Role | Source | Event | Pair | Interior pixels |', '| --- | --- | --- | --- | ---: |']
    for c in shortlist['candidates']:
        lines.append(f"| {c['role']} | {c['stratum']} | {c['event_id']} | {c['pair_id']} | {c['positive_interior_pixels']} |")
    lines += ['', 'Missing slots: '+str(len(shortlist['missing_slots']))+'. No fallback or approval was added.', '',
        'Counts distinguish raw anchors, temporal clipping and eroded targets. Final targets were reconstructed per pair, never cached by event alone.',
        'Prior reviewed events and entire groups are excluded from selection but retained in the audit.',
        'All exclusions are in audit.json; reasons can overlap. No area/bbox/top-three, NDVI or model-score selection.',
        'A later usable-metadata scene is only a review opportunity; no forest/clearing/persistence labels or model accuracy follow.']
    (output/'report.md').write_text('\n'.join(lines).rstrip()+'\n')
    require(all(digest(ROOT/n) == sha for n, sha in result['source_sha256'].items())
            and all(digest(ROOT/'Solutions'/n) == sha for n, sha in implementation.items()), 'Audit inputs or code changed during processing')
    require('torch' not in sys.modules, 'Support audit unexpectedly imported Torch')
    import resource
    write_json(output/'metadata.json', {'execution_status': 'completed', 'mode': 'validation_support_pool',
        'protocol_sha256': protocol_sha, 'coverage': result['coverage'], 'eligible_after_history': result['eligible_after_history'],
        'shortlist_count': len(shortlist['candidates']), 'implementation_sha256': implementation,
        'artifact_sha256': {n: digest(output/n) for n in ('audit.json', 'selection.json', 'report.md')},
        'source_jp2_pixel_reads': 0, 'model_executions': 0, 'new_imagery_reviewed': 0, 'labels_approved': 0,
        'torch_imported': False,
        'worker_high_water_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024), 2)})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationSupportV2/protocol.json')
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
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='validation-support-pool', gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
