#!/usr/bin/env python3
"""Audit rasterized annotation support without reading satellite pixels.

This audits the locked metadata-eligible pool. It never changes the roster,
opens new imagery, approves labels, or executes a model.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


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
    require(path.is_file() and path.stat().st_size <= 32*2**20, 'Missing or oversized geometry evidence')
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite evidence')))


def geometry_support(event_id, pair, events):
    """Return actual anchor/temporal intersection and its 3x3 eroded interior."""
    import numpy as np
    from scipy.ndimage import binary_erosion
    from assess_pixel_quality import core_grid, label_target
    from prepare_validation_review import validate_grid
    from validate_sample import burn

    validate_grid(pair)
    require(event_id in pair.get('anchor_event_ids', []) and event_id in events,
            'Exact frozen validation anchor is required')
    projected = events
    if pair['grid']['crs'] != 'EPSG:32636':
        from pyproj import Transformer
        from shapely.ops import transform
        project = Transformer.from_crs('EPSG:32636', pair['grid']['crs'], always_xy=True).transform
        projected = {key: {**event, 'geometry': transform(project, event['geometry'])}
                     for key, event in events.items()}
    anchor = burn([projected[event_id]['geometry']], core_grid(pair['grid'])).astype(bool)
    temporal = label_target(pair, projected).astype(bool)
    target = anchor & temporal
    interior = binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
    return {'anchor_pixels': int(anchor.sum()), 'temporal_union_pixels': int(temporal.sum()),
            'target_pixels': int(target.sum()), 'positive_interior_pixels': int(interior.sum()),
            'passes_geometry_support_only': int(interior.sum()) >= 4}


def worker(args, output):
    os.nice(10)
    from assess_pixel_quality import load_events
    from prepare_validation_review import records, validate_validation_pair

    selection_path = args.selection.resolve()
    selection = read_json(selection_path)
    protocol_path = ROOT/'Docs/Reviews/ValidationPilotV1/protocol.json'
    protocol = read_json(protocol_path)
    require(selection['selection_uses_model_predictions'] is False
            and selection['pixel_arrays_read'] == selection['model_executions'] == 0
            and selection['frozen_policy_id'] == protocol['qa_policy_id']
            and digest(selection_path) == protocol['selection']['selection_sha256'],
            'Locked prediction-blind selection identity changed')
    for name, checksum in protocol['immutable_sources_sha256'].items():
        require(digest(ROOT/'Solutions'/name) == checksum, 'Archived source changed: '+name)
    audit, qa, training = (ROOT/path for path in ('Datasets/AnnotationAudit/v1',
                           'Datasets/PixelQA/v1', 'Datasets/CuratedPilot/v1/dataset'))
    sources = {'audit_metadata': audit/'metadata.json', 'splits': audit/'splits.json',
        'catalogue': audit/'annotations_clean.geojson', 'annotations': audit/'annotation_manifest.jsonl',
        'pairs': audit/'candidate_pairs.jsonl', 'scenes': audit/'scene_inventory.jsonl',
        'quality_summary': qa/'summary/metadata.json', 'quality': qa/'summary/quality_manifest.jsonl',
        'review_metadata': qa/'review/metadata.json', 'queue': qa/'review/review_queue.jsonl',
        'training_metadata': training/'metadata.json', 'training_manifest': training/'training_manifest.jsonl'}
    for name, path in sources.items():
        require(digest(path) == selection['source_files'][name]['sha256'], 'Frozen selection source changed: '+name)
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources.values()}
    source_hashes[str(selection_path.relative_to(ROOT))] = digest(selection_path)
    source_hashes[str(protocol_path.relative_to(ROOT))] = digest(protocol_path)
    implementation = digest(__file__)
    eligible = [(entry['event_id'], item['pair_id']) for entry in selection['dispositions']
                for item in entry['considered_pairs'] if item['eligible'] is True]
    require(0 < len(eligible) <= 160 and len(eligible) == len(set(eligible)),
            'Metadata-eligible geometry coverage is missing, repeated or oversized')
    wanted = {pair_id for _, pair_id in eligible}
    pairs = {row['pair_id']: row for row in records(sources['pairs']) if row['pair_id'] in wanted}
    require(set(pairs) == wanted, 'An eligible pair is absent from the frozen manifest')
    catalogue, splits = read_json(sources['catalogue']), read_json(sources['splits'])
    properties = {feature['id']: feature['properties'] for feature in catalogue['features']}
    training_rows = list(records(sources['training_manifest']))
    events = load_events(audit)
    selected = {(row['event_id'], row['pair_id']): row for row in selection['candidates']}
    rows = []
    for event_id, pair_id in sorted(eligible):
        pair = pairs[pair_id]; prop = properties[event_id]
        validate_validation_pair(pair, event_id, catalogue, splits, training_rows)
        require(prop['split'] == pair['split'] == 'validation'
                and prop['group_id'] == pair['group_id'], 'Frozen event group/role changed')
        sources_for_event = [name for name in selection['rule']['strata_in_order']
                             if name in prop['temporal_source_ids']]
        require(len(sources_for_event) == 1, 'Eligible source stratum changed')
        row = {'event_id': event_id, 'pair_id': pair_id, 'group_id': pair['group_id'],
               'stratum': sources_for_event[0], 'split': 'validation',
               'pre_date': pair['pre_date'], 'post_date': pair['post_date'],
               'anchor_area_m2_search_aid': prop['area_m2'],
               'minimum_bbox_width_height_m_search_aid': min(prop['width_m'], prop['height_m']),
               **geometry_support(event_id, pair, events),
               'in_locked_review_roster': (event_id, pair_id) in selected,
               'training_approved': False, 'evaluation_approved': False}
        if row['in_locked_review_roster']:
            entry = selected[(event_id, pair_id)]
            case_path = ROOT/'Docs/Reviews/ValidationPilotV1/prepared_cases'/event_id/'case.json'
            case = read_json(case_path)
            require(case['pair'] == pair and case['target_pixels'] == row['target_pixels']
                    and case['positive_interior_pixels'] == row['positive_interior_pixels'],
                    'Geometry-only reconstruction disagrees with the prepared selected case')
            row['review_role'] = entry['role']
            source_hashes[str(case_path.relative_to(ROOT))] = digest(case_path)
        rows.append(row)
    require(sum(row['in_locked_review_roster'] for row in rows) == len(selected),
            'Every selected review case must be cross-checked')

    def coverage(items):
        passed = [row for row in items if row['passes_geometry_support_only']]
        return {'anchor_pair_count': len(items), 'pair_count': len({row['pair_id'] for row in items}),
                'event_count': len({row['event_id'] for row in items}),
                'group_count': len({row['group_id'] for row in items}),
                'anchor_pairs_with_interior_at_least_4': len(passed),
                'pairs_with_any_anchor_interior_at_least_4': len({row['pair_id'] for row in passed}),
                'events_with_any_interior_at_least_4': len({row['event_id'] for row in passed}),
                'groups_with_any_interior_at_least_4': len({row['group_id'] for row in passed}),
                'anchor_pairs_below_4': len(items)-len(passed)}

    result = {'audit_status': 'passed', 'scope': 'Geometry support audit of the locked metadata-eligible pool; no new review roster or approved labels.',
        'coverage': coverage(rows), 'by_stratum': {name: coverage([row for row in rows if row['stratum'] == name])
                                                for name in selection['rule']['strata_in_order']},
        'locked_roster_coverage': coverage([row for row in rows if row['in_locked_review_roster']]),
        'records': rows, 'source_sha256': source_hashes, 'implementation_sha256': implementation,
        'satellite_pixel_reads': 0, 'source_jp2_pixel_reads': 0, 'model_executions': 0,
        'new_imagery_reviewed': 0, 'new_candidates_selected': 0,
        'training_approved': False, 'evaluation_approved': False,
        'limitations': ['Area and bounding dimensions do not guarantee a sufficiently wide pixel interior.',
            'Passing geometry support establishes neither visible clearing nor a usable temporal transition.',
            'The audited pool inherits the previous clear-image/geometry search-aid filters; it is not all validation events.',
            'A future review roster must be separately predeclared before new imagery inspection.']}
    require(all(digest(ROOT/name) == checksum for name, checksum in source_hashes.items())
            and digest(__file__) == implementation, 'Geometry audit sources changed during execution')
    (output/'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Validation geometry support audit', '', result['scope'], '',
        '| Stratum | Eligible anchor-pairs | Events | Groups | Anchor-pairs with interior >=4 | Events with any interior >=4 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, counts in result['by_stratum'].items():
        lines.append('| '+' | '.join(map(str, [name, counts['anchor_pair_count'], counts['event_count'],
            counts['group_count'], counts['anchor_pairs_with_interior_at_least_4'],
            counts['events_with_any_interior_at_least_4']]))+' |')
    lines += ['', 'Locked roster reconstruction:', '', '| Event | Role | Target pixels | Eroded interior pixels |',
              '| --- | --- | ---: | ---: |']
    for row in rows:
        if row['in_locked_review_roster']:
            lines.append(f"| {row['event_id']} | {row['review_role']} | {row['target_pixels']} | {row['positive_interior_pixels']} |")
    lines += ['', 'No satellite pixels, model predictions, new image reviews or label approvals.', '', *result['limitations']]
    (output/'report.md').write_text('\n'.join(lines).rstrip()+'\n')
    import resource
    metadata = {**{key: result[key] for key in ('audit_status', 'scope', 'coverage', 'by_stratum',
        'locked_roster_coverage', 'implementation_sha256', 'source_jp2_pixel_reads', 'model_executions',
        'new_imagery_reviewed', 'new_candidates_selected', 'training_approved', 'evaluation_approved')},
        'artifact_sha256': {name: digest(output/name) for name in ('audit.json', 'report.md')},
        'worker_high_water_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                        (2**20 if sys.platform == 'darwin' else 1024), 2)}
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=ROOT/'Docs/Reviews/ValidationPilotV1/selection/selection.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=120)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300):
        parser.error('Use 128–512 MiB RSS, >=1.5 GiB reserve and <=300 seconds')
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='validation-geometry-audit', gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
