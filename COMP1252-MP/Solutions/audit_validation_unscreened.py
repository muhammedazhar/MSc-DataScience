#!/usr/bin/env python3
"""Inventory blocked, unscreened validation geometry; never read satellite pixels."""
import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    require(Path(path).stat().st_size <= 32*2**20, 'Oversized inventory metadata')
    return json.loads(Path(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def keyed(rows, key):
    result = {}
    for row in rows:
        require(row[key] not in result, 'Duplicate identity: '+row[key])
        result[row[key]] = row
    return result


def complement(pairs, published, retained):
    """Preserve blocked pairs and every anchor, rather than filtering on status."""
    frozen = keyed(pairs, 'pair_id'); quality = keyed(published, 'pair_id')
    validation = {k: p for k, p in frozen.items() if p['split'] == 'validation'}
    require(set(quality) <= set(validation), 'Published validation QA has unknown pairs')
    missing = sorted(set(validation)-set(quality))
    old = keyed(retained, 'pair_id')
    require(set(old) == set(missing), 'V2 unscreened complement changed')
    for identifier in missing:
        pair = validation[identifier]
        require(pair['anchor_event_ids'] and len(pair['anchor_event_ids']) == len(set(pair['anchor_event_ids'])),
                'Unscreened anchors must be nonempty and unique')
        require(all(old[identifier][k] == pair[k] for k in ('group_id', 'anchor_event_ids', 'status', 'blocking_reasons')),
                'Retained V2 pair identity/blockers changed')
    return [validation[k] for k in missing]


def erode_numpy(mask):
    import numpy as np
    result = np.zeros_like(mask)
    result[1:-1, 1:-1] = np.logical_and.reduce(
        [mask[1+y:63+y, 1+x:63+x] for y in (-1, 0, 1) for x in (-1, 0, 1)])
    return result


def support(pair, event_id, events, independent=False):
    import numpy as np
    if independent:
        from affine import Affine
        from rasterio.features import rasterize
        def burn(geometries):
            return rasterize([(g, 1) for g in geometries], out_shape=(64, 64),
                transform=Affine(*pair['grid']['transform']), fill=0, dtype='uint8', all_touched=False).astype(bool) if geometries else np.zeros((64, 64), dtype=bool)
        before, after = [], []
        for identifier in pair['label_event_ids_by_post_date']:
            event = events[identifier]; properties = event['properties']
            require(properties['temporal_source_ids'], 'Baseline-only event in temporal reference')
            if properties['observation_date'] <= pair['post_date']:
                after.append(event['geometry'])
            if properties['observation_date'] <= pair['pre_date']:
                before.append(event['geometry'])
        reference = burn(after) & ~burn(before)
        anchor = burn([events[event_id]['geometry']])
        erosion = erode_numpy
    else:
        from assess_pixel_quality import core_grid, label_target
        from validate_sample import burn
        from scipy.ndimage import binary_erosion
        anchor = burn([events[event_id]['geometry']], core_grid(pair['grid'])).astype(bool)
        reference = label_target(pair, events).astype(bool)
        erosion = lambda value: binary_erosion(value, structure=np.ones((3, 3), dtype=bool))
    target = anchor & reference
    raw_interior, interior = erosion(anchor), erosion(target)
    masks = {'anchor': anchor, 'raw_interior': raw_interior, 'reference': reference,
             'target': target, 'interior': interior}
    return {'anchor_pixels': int(anchor.sum()), 'anchor_interior_pixels': int(raw_interior.sum()),
        'temporal_union_pixels': int(reference.sum()), 'target_pixels': int(target.sum()),
        'temporal_clipped_pixels': int((anchor & ~reference).sum()),
        'positive_interior_pixels': int(interior.sum()),
        'mask_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in masks.items()}}


def reasons(pair, geometry, isolated, followup, exposed, endpoints_usable):
    result = ['original:'+r for r in pair['blocking_reasons']]
    if pair['status'] != 'candidate_for_pixel_review':
        result.append('original_pair_status_blocked')
    if geometry['positive_interior_pixels'] < 4:
        result.append('insufficient_positive_interior')
    if not isolated:
        result.append('train_test_footprint_overlap')
    if not followup:
        result.append('no_later_metadata_usable_observation')
    if exposed:
        result.append('previously_inspected_group')
    if not endpoints_usable:
        result.append('endpoint_metadata_unusable')
    return result


def summarize(pairs, rows):
    sufficient = [r for r in rows if r['positive_interior_pixels'] >= 4]
    ready = [r for r in rows if r['passes_metadata_geometry_gates']]
    return {'pairs': len(pairs), 'anchor_pairs': len(rows),
        'events': len({r['event_id'] for r in rows}), 'groups': len({p['group_id'] for p in pairs}),
        'pair_status_counts': dict(sorted(Counter(p['status'] for p in pairs).items())),
        'pair_blocking_reason_counts_overlapping': dict(sorted(Counter(b for p in pairs for b in p['blocking_reasons']).items())),
        'anchor_pairs_with_interior_at_least_four': len(sufficient),
        'supported_events': len({r['event_id'] for r in sufficient}),
        'supported_groups': len({r['group_id'] for r in sufficient}),
        'anchor_pairs_with_temporal_clipping': sum(r['temporal_clipped_pixels'] > 0 for r in rows),
        'anchor_pairs_footprint_isolated': sum(r['footprint_isolated'] for r in rows),
        'anchor_pairs_with_later_metadata_opportunity': sum(bool(r['later_usable_scene_ids']) for r in rows),
        'anchor_pairs_in_exposed_groups': sum(r['previously_inspected_group'] for r in rows),
        'passes_metadata_geometry_gates': len(ready),
        'qa_status': 'unknown_no_published_qa', 'approved_labels': 0}


def calculate(protocol_path, independent=False):
    from prepare_validation_review import records, validate_grid
    from select_validation_cases import grid_bounds, touches
    protocol = read(protocol_path)
    require(protocol['schema_version'] == 5 and protocol['rule']['pairs'] == 606
            and protocol['rule']['anchor_pairs'] == 630
            and protocol['rule']['minimum_positive_interior_pixels'] == 4
            and protocol['rule']['no_selection'] is True, 'Fixed V5 scope changed')
    require(protocol['resources'] == {'workers': 1, 'native_threads': 1, 'nice_increment': 10,
        'gpu_used': False, 'worker_rss_mib': 512, 'available_memory_reserve_gib': 1.5,
        'max_seconds_per_worker': 300}, 'Mac resource contract changed')
    source = {}
    for role, item in protocol['source_files'].items():
        path = (ROOT/item['path']).resolve()
        require(path.is_relative_to(ROOT) and digest(path) == item['sha256'], 'Source changed: '+role)
        source[role] = path
    for name, checksum in {**protocol['immutable_sources_sha256'], **protocol['implementation_sha256']}.items():
        require(digest(ROOT/'Solutions'/name) == checksum, 'Frozen implementation changed: '+name)
    broad, broad_meta, proof = (read(source[k]) for k in ('v2_audit', 'v2_metadata', 'v2_verification'))
    require(proof['verification_status'] == 'passed' and proof['audit_sha256'] == digest(source['v2_audit'])
        and proof['audit_metadata_sha256'] == digest(source['v2_metadata'])
        and proof['protocol_sha256'] == broad['protocol_sha256'] == digest(source['v2_protocol'])
        and broad_meta['execution_status'] == 'completed', 'V2 evidence chain changed')
    all_pairs = list(records(source['pairs']))
    quality = [r for r in records(source['quality']) if r['split'] == 'validation']
    index = [r for r in records(source['quality_index']) if r['split'] == 'validation']
    require(set(keyed(quality, 'pair_id')) == set(keyed(index, 'pair_id')), 'Published QA/index coverage changed')
    pairs = complement(all_pairs, quality, broad['unscreened_candidates'])
    require(len(pairs) == 606 and sum(len(p['anchor_event_ids']) for p in pairs) == 630, 'Exact606/630coverage required')
    catalogue, splits = read(source['catalogue']), read(source['splits'])
    props = {f['id']: f['properties'] for f in catalogue['features']}
    allocation = {e: (g['group_id'], g['split']) for g in splits['groups'] for e in g['event_ids']}
    require(set(props) == set(allocation) and all(allocation[e] == (p['group_id'], p['split']) for e, p in props.items()), 'Split allocation changed')
    train = list(records(source['training_manifest']))
    require(len(train) == 13 and all(r['split'] == 'train' for r in train), 'All13training grids required')
    old_protocol = read(source['v2_protocol'])
    require(all(protocol['source_files'][k]['sha256'] == old_protocol['source_files'][k]['sha256']
            for k in old_protocol['source_files']), 'V2 frozen input identity changed')
    prior = read(source['prior_verification']); manual = read(source['manual_verification'])
    require(prior['verification_status'] == manual['verification_status'] == 'passed'
        and prior['reviews_sha256'] == digest(source['prior_reviews'])
        and manual['review_sha256'] == digest(source['manual_reviews'])
        and manual['prepared_metadata_sha256'] == digest(source['manual_metadata'])
        and manual['decision']['decision_status'] == 'rejected', 'Inspection history proof changed')
    exposed = {r['group_id'] for r in prior['review_dispositions']} | {read(source['manual_metadata'])['group_id']}
    barriers = [p['group_footprint_bounds'] for p in props.values() if p['split'] in ('train', 'test')]
    barriers += [grid_bounds(row['grid'], splits['crs']) for row in train]
    scenes = list(records(source['scenes'])); scene_map = keyed(scenes, 'scene_id')
    if independent:
        from shapely.geometry import shape
        from shapely.ops import transform
        from pyproj import Transformer
        projection = Transformer.from_crs('EPSG:4326', 'EPSG:32636', always_xy=True).transform
        events = {f['id']: {'properties': f['properties'], 'geometry': transform(projection, shape(f['geometry']))} for f in catalogue['features']}
    else:
        from assess_pixel_quality import load_events
        events = load_events(source['catalogue'].parent)
    rows = []
    for pair in pairs:
        validate_grid(pair); require(pair['grid']['crs'] == splits['crs'] == 'EPSG:32636', 'Original projected grid required')
        bounds = grid_bounds(pair['grid'], splits['crs'])
        isolated = not any(touches(bounds, b) for b in barriers) and all(t['group_id'] != pair['group_id'] for t in train)
        usable = all(scene_map[pair[side+'_scene_id']]['metadata_usable'] is True for side in ('pre', 'post'))
        for event_id in pair['anchor_event_ids']:
            prop = props[event_id]
            require((prop['group_id'], prop['split']) == (pair['group_id'], 'validation'), 'Anchor split/group differs')
            observed = date.fromisoformat(prop['observation_date'])
            followup = [s['scene_id'] for s in sorted(scenes, key=lambda s: (s['acquisition_date'], s['scene_id']))
                if s['tile'] == pair['tile'] and s['metadata_usable'] is True
                and pair['post_date'] < s['acquisition_date'] <= (observed+timedelta(days=45)).isoformat()
                and date.fromisoformat(s['acquisition_date']).year == observed.year]
            geometry = support(pair, event_id, events, independent)
            exclusion = reasons(pair, geometry, isolated, followup, pair['group_id'] in exposed, usable)
            rows.append({'event_id': event_id, 'pair_id': pair['pair_id'], 'group_id': pair['group_id'],
                'split': 'validation', 'temporal_source_ids': prop['temporal_source_ids'],
                'annotation_date': prop['observation_date'], 'pre_date': pair['pre_date'], 'post_date': pair['post_date'],
                'grid': pair['grid'], 'original_status': pair['status'], 'original_blocking_reasons': pair['blocking_reasons'],
                **geometry, 'footprint_isolated': isolated, 'endpoint_metadata_usable': usable,
                'later_usable_scene_ids': followup, 'previously_inspected_group': pair['group_id'] in exposed,
                'metadata_geometry_reasons': exclusion, 'passes_metadata_geometry_gates': not exclusion,
                'quality_status': 'unknown_no_published_qa', 'endpoint_unusable_pixels': None,
                'interior_unusable_pixels': None, 'training_approved': False, 'evaluation_approved': False})
    dispositions = [{k: p[k] for k in ('pair_id', 'group_id', 'anchor_event_ids', 'status', 'blocking_reasons')} for p in pairs]
    result = {'protocol_sha256': digest(protocol_path), 'pairs': dispositions, 'records': rows,
        'exposed_group_ids': sorted(exposed), 'summary': summarize(pairs, rows),
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in source.values()}}
    require(len({(r['pair_id'], r['event_id']) for r in rows}) == 630, 'Anchor coverage lost or duplicated')
    require(all(digest(source[k]) == item['sha256'] for k, item in protocol['source_files'].items()), 'Source changed during inventory')
    require('torch' not in sys.modules, 'Inventory must not import Torch')
    return result


def worker(args, output):
    result = calculate(args.protocol, independent=args.mode == 'verify')
    metadata = {'execution_status': 'completed', 'mode': args.mode, 'protocol_sha256': digest(args.protocol),
        'summary': result['summary'], 'source_jp2_pixel_reads': 0, 'qa_mask_files_read': 0,
        'cache_files_read': 0, 'model_executions': 0, 'cloud_model_executions': 0,
        'new_imagery_reviewed': 0, 'labels_approved': 0, 'torch_imported': False,
        'implementation_sha256': digest(__file__)}
    if args.mode == 'audit':
        write(output/'inventory.json', result)
        lines = ['# Unscreened validation inventory', '', json.dumps(result['summary'], indent=2), '',
            'Every original blocker is retained. Quality stays unknown. No shortlist or approved labels.',
            'Geometry can be measurable outside available scene coverage; support never overrides source/provenance blockers.']
        (output/'report.md').write_text('\n'.join(lines)+'\n')
        metadata['artifact_sha256'] = {n: digest(output/n) for n in ('inventory.json', 'report.md')}
    else:
        old = read(args.run/'metadata.json')
        from verify_validation_support_pool import verify_resources
        verify_resources(old)
        require(old['mode'] == 'audit' and old['protocol_sha256'] == metadata['protocol_sha256']
            and old['implementation_sha256'] == digest(__file__), 'Completed audit identity changed')
        require(set(old['artifact_sha256']) == {'inventory.json', 'report.md'}, 'Audit artifact coverage changed')
        for name, expected in old['artifact_sha256'].items():
            require(digest(args.run/name) == expected, 'Audit artifact changed')
        require(read(args.run/'inventory.json') == result, 'Independent geometry/erosion reconstruction differs')
        metadata.update(verification_status='passed', audit_metadata_sha256=digest(args.run/'metadata.json'),
            inventory_sha256=digest(args.run/'inventory.json'),
            independence_scope='Independent catalogue projection, cumulative temporal masks and NumPy erosion; metadata joins/gates and summary helpers shared, not an independent class/QA approval.')
        (output/'report.md').write_text('# Verified unscreened inventory\n\nAll606pairs/630anchors match independent geometry reconstruction.\nNo pixels, QA or models read/run. Metadata gate helpers are shared.\n')
    import resource
    metadata['worker_high_water_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024), 2)
    write(output/'metadata.json', metadata)


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('audit', 'verify'), default='audit')
    parser.add_argument('--protocol', type=Path, default=ROOT/'Docs/Reviews/ValidationUnscreenedV5/protocol.json')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=300)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == 'verify' and args.run is None:
        parser.error('Verification requires --run')
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Keep512MiB/1.5GiB/300s limits')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10); worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='unscreened-inventory', gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
