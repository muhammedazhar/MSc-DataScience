#!/usr/bin/env python3
"""Find a bounded fresh-source plan from the frozen validation evidence."""

import argparse
from datetime import date, timedelta
from itertools import combinations
import os
from pathlib import Path
import resource
import sys

from inventory_earlier_validation_sources import RESOURCES, ROOT, canonical, digest, require, rooted, sha, write
from prepare_validation_review import read_json, records, validate_validation_pair


PROTOCOL = ROOT/'Docs/Reviews/ValidationFreshSourcesV23/protocol.json'
PARENTS = ('pair-4da0c1fed3e02525a046', 'pair-21bb6f74457c8d1631a5')
INPUT_ROLES = ('v2_audit', 'v5_inventory', 'v22_review', 'pairs', 'catalogue', 'splits',
               'training_manifest', 'training_metadata', 'original_scene_inventory')
QA_PAD = 1200


def pinned_inputs(protocol_path):
    protocol = read_json(protocol_path)
    require(protocol['schema_version'] == 23 and protocol['resources'] == RESOURCES,
            'Fresh validation scope or resource limits changed')
    implementation = protocol['implementation_sha256']
    require(str(Path(__file__).resolve().relative_to(ROOT)) in implementation,
            'Pin the fresh-source planner before use')
    paths = {}
    for name, receipt in protocol['source_files'].items():
        path = rooted(receipt['path'])
        require(digest(path) == receipt['sha256'], 'Pinned source changed: '+name)
        paths[name] = path
    require(set(INPUT_ROLES) <= paths.keys(), 'Missing frozen planning inputs')
    for name, checksum in {**protocol.get('frozen_implementation_sha256', {}), **implementation}.items():
        require(digest(rooted(name)) == checksum, 'Pinned source code changed: '+name)
    return paths


def indexed(rows, field):
    result = {}
    for row in rows:
        require(row[field] not in result, 'Repeated frozen identity: '+row[field])
        result[row[field]] = row
    return result


def frozen_pool(paths, pairs):
    screened = read_json(paths['v2_audit'])
    unscreened = read_json(paths['v5_inventory'])
    review = read_json(paths['v22_review'])
    published = indexed(screened['records'], 'pair_id')
    omitted = indexed(screened['unscreened_candidates'], 'pair_id')
    retained = indexed(unscreened['pairs'], 'pair_id')
    require(len(published) == 540 and len(retained) == len(omitted) == 606
            and not (published.keys() & omitted.keys()) and retained.keys() == omitted.keys()
            and pairs.keys() == (published.keys() | omitted.keys()), 'Frozen pool pairing or scope differs')
    keys = ('pair_id', 'group_id', 'anchor_event_ids', 'status', 'blocking_reasons')
    for ident, row in retained.items():
        require(all(row[key] == omitted[ident][key] == pairs[ident][key] for key in keys),
                'A frozen unscreened blocker or identity changed: '+ident)
    expected = {(pair_id, event) for pair_id, row in retained.items() for event in row['anchor_event_ids']}
    actual = {(row['pair_id'], row['event_id']) for row in unscreened['records']}
    require(len(expected) == len(actual) == len(unscreened['records']) == 630 and actual == expected,
            'An unscreened anchor is missing, repeated, or unexpected')
    for row in (*published.values(), *unscreened['records']):
        pair = pairs[row['pair_id']]
        require(row['event_id'] in pair['anchor_event_ids']
                and all(row[key] == pair[key] for key in ('group_id', 'split', 'pre_date', 'post_date', 'grid'))
                and not row['training_approved'] and not row['evaluation_approved'],
                'Frozen geometry, date, or approval differs: '+row['pair_id'])
    exposed = set(review['manual_exposed_group_ids'])
    july_group = 'group-96c27871459d03c08bcc'
    require(review['status'] == 'independent_reference_review_completed'
            and review['group_id'] == july_group and not review['validation_eligible']
            and review['labels_approved'] == 0 and len(exposed) == 7
            and exposed == (set(unscreened['exposed_group_ids']) | {july_group})
            and set(screened['prior_reviewed_group_ids']) <= exposed, 'Prior exposure history changed')
    return published.values(), unscreened['records'], exposed


def fresh_summary(rows, exposed, gate):
    fresh = [row for row in rows if row['group_id'] not in exposed]
    supported = [row for row in fresh if row['positive_interior_pixels'] >= 4]
    summary = {'anchor_pairs': len(rows), 'fresh_anchor_pairs': len(fresh),
               'supported_fresh_anchor_pairs': len(supported),
               'supported_fresh_groups': len({row['group_id'] for row in supported}),
               'eligible_fresh_anchor_pairs': sum(row[gate] for row in fresh)}
    require(summary['eligible_fresh_anchor_pairs'] == 0, 'A frozen review gate changed')
    return summary, supported


def pre_window(row):
    after = date.fromisoformat(row['post_date'])
    # Shared acquisition should precede every failed original pre date and annotation.
    end_exclusive = min(date.fromisoformat(row['pre_date']), date.fromisoformat(row['annotation_date']),
                        after-timedelta(days=4))
    return after-timedelta(days=30), end_exclusive


def ready_post_rows(supported, pairs):
    from audit_validation_support_pool import eligibility_reasons

    prior_pre_only = {'not_automatic_quality_pass', 'endpoint_crop_qa_flags',
                      'endpoint_target_qa_flags', 'actual_interior_qa_flags'}
    alternatives = {}
    for row in supported:
        pair = pairs[row['pair_id']]
        if (pair['status'] == 'candidate_for_pixel_review' and not pair['blocking_reasons']
                and pair['temporal_source_ids'] and not pair['conflicting_label_event_ids']
                and all(row[key][0] > 0 and row[key][1] == 0
                        for key in ('endpoint_crop_unusable_pixels', 'endpoint_target_unusable_pixels'))
                and set(eligibility_reasons(row, (), ())) == prior_pre_only
                and set(row['eligibility_reasons']) == prior_pre_only):
            alternatives.setdefault(row['group_id'], []).append(row)
    for rows in alternatives.values():
        rows.sort(key=lambda row: (row['post_date'] != row['annotation_date'], row['post_date'],
                                   row['interval_days'], row['pair_id']))
    require(len(alternatives) == 4 and sum(map(len, alternatives.values())) == 20,
            'Archived clear-post alternatives changed')
    return alternatives


def deployment_key(pair):
    return pair['tile'], pair['grid']['crs'], tuple(pair['post_scene_id'].split('_')[3:5])


def choose_shared_pair(alternatives, pairs):
    representatives = [rows[0] for _, rows in sorted(alternatives.items())]
    shared = []
    for left, right in combinations(representatives, 2):
        windows = pre_window(left), pre_window(right)
        start = max(window[0] for window in windows)
        end = min(window[1] for window in windows)
        if start < end and deployment_key(pairs[left['pair_id']]) == deployment_key(pairs[right['pair_id']]):
            shared.append(((left, right), start, end))
    require(len(shared) == 1 and tuple(row['pair_id'] for row in shared[0][0]) == PARENTS,
            'Expected one geographically distinct shared-source opportunity')
    return shared[0]


def union_bounds(bounds):
    return [min(box[0] for box in bounds), min(box[1] for box in bounds),
            max(box[2] for box in bounds), max(box[3] for box in bounds)]


def padded_bbox(bounds):
    return [bounds[0]-QA_PAD, bounds[1]-QA_PAD, bounds[2]+QA_PAD, bounds[3]+QA_PAD]


def post_record(pair, original):
    from pixelqa_scene_metadata import qa_scene_metadata
    from prepare_validation_review import rooted as scene_path

    path = scene_path(pair['post_path'])
    require(path.is_dir() and path.name.endswith('.SAFE'), 'Frozen post SAFE directory unavailable')
    post = qa_scene_metadata(path, ROOT)
    outer = post['grid_10m']['bounds']
    inner = padded_bbox(pair['grid']['bounds'])
    require(all(post[key] == original[key] for key in ('scene_id', 'path', 'tile', 'acquisition_date',
                 'processing_level', 'baseline_code', 'crs', 'grid_10m', 'metadata_sha256'))
            and post['scene_id'] == pair['post_scene_id'] and post['acquisition_date'] == pair['post_date']
            and post['metadata_usable'] and post['cloud_model_usable'] and post['shadow_geometry_usable']
            and post['crs'] == pair['grid']['crs']
            and outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3],
            'Original post metadata or padded coverage changed: '+pair['pair_id'])
    return {key: post[key] for key in ('scene_id', 'acquisition_date', 'crs', 'tile', 'baseline_code',
                                      'grid_10m', 'metadata_sha256', 'metadata_uid')}


def day_start(day):
    return day.isoformat()+'T00:00:00Z'


def build_plan(paths):
    pairs = indexed((row for row in records(paths['pairs']) if row['split'] == 'validation'), 'pair_id')
    screened, unscreened, exposed = frozen_pool(paths, pairs)
    screened_summary, support = fresh_summary(screened, exposed, 'eligible_for_fresh_review_shortlist')
    unscreened_summary, unscreened_support = fresh_summary(unscreened, exposed, 'passes_metadata_geometry_gates')
    require(len(unscreened_support) == 4 and all(
        {'baseline_only_provenance', 'unverified_temporal_label_in_crop'} <= set(row['original_blocking_reasons'])
        for row in unscreened_support), 'Supported unscreened provenance changed')
    alternatives = ready_post_rows(support, pairs)
    selected, start, end = choose_shared_pair(alternatives, pairs)
    catalogue, splits, training_metadata = (read_json(paths[role]) for role in ('catalogue', 'splits', 'training_metadata'))
    training = list(records(paths['training_manifest']))
    require(training_metadata['accepted_example_count'] == len(training) == 13
            and training_metadata['annotation_catalogue_sha256'] == digest(paths['catalogue'])
            and training_metadata['candidate_manifest_sha256'] == digest(paths['pairs']),
            'Frozen training and annotation provenance changed')
    scenes = indexed(records(paths['original_scene_inventory']), 'scene_id')
    roster = []
    for evidence in selected:
        pair = pairs[evidence['pair_id']]
        validate_validation_pair(pair, evidence['event_id'], catalogue, splits, training)
        post = post_record(pair, scenes[pair['post_scene_id']])
        before, until = pre_window(evidence)
        roster.append({**{key: pair[key] for key in ('pair_id', 'group_id', 'grid', 'pre_date', 'post_date',
                          'pre_scene_id', 'post_scene_id', 'post_path')},
                       'event_id': evidence['event_id'], 'annotation_date': evidence['annotation_date'],
                       'parent_pair_sha256': sha(canonical(pair)), 'qa_context_bounds': padded_bbox(pair['grid']['bounds']),
                       'post_metadata_usable': True, 'verified_post_metadata': post,
                       'pre_search_start_inclusive': day_start(before), 'pre_search_end_exclusive': day_start(until),
                       'evidence': evidence})
    tile, crs, (baseline, orbit) = deployment_key(pairs[roster[0]['pair_id']])
    return {'schema_version': 23, 'status': 'fresh_catalogue_plan_ready',
            'summary': {'screened': screened_summary, 'unscreened': unscreened_summary},
            'excluded_previous_review_groups': sorted(exposed),
            'clear_post_alternatives': alternatives,
            'supported_unscreened_original_rows': unscreened_support,
            'selection_rule': ('Keep the four fresh groups with an isolated supported target, complete original '
                               'post crop and target clear, retained pre QA failure and later metadata opportunity. '
                               'Rank each group by exact annotation-date post, earlier post, shorter interval, then pair ID. '
                               'Select the unique two-group window sharing tile, orbit and baseline.'),
            'roster': roster,
            'catalogue_requirements': {'crs': crs, 'target_tile': tile, 'relative_orbit': orbit,
                 'nominal_followup_baseline': baseline, 'start_inclusive': day_start(start), 'end_exclusive': day_start(end),
                 'required_post_dates': [row['post_date'] for row in roster],
                 'core_union_bounds': union_bounds([row['grid']['bounds'] for row in roster]),
                 'qa_context_union_bounds': union_bounds([row['qa_context_bounds'] for row in roster])},
            'labels_approved': 0, 'validation_eligible': False}


def worker(protocol_path, output):
    os.nice(10)
    protocol_sha = digest(protocol_path)
    paths = pinned_inputs(protocol_path)
    plan = build_plan(paths)
    require(protocol_sha == digest(protocol_path), 'Planning protocol changed during execution')
    require(not any(name in sys.modules for name in ('torch', 'rasterio', 's2cloudless')),
            'Metadata planning imported an image or model library')
    plan_path = output/'plan.json'
    write(plan_path, {**plan, 'protocol_sha256': protocol_sha})
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {'execution_status': 'completed', 'mode': 'fresh_source_planning',
          'protocol_sha256': protocol_sha, 'artifact_sha256': {'plan.json': digest(plan_path)},
          'image_asset_downloads': 0, 'satellite_pixel_reads': 0, 'model_executions': 0, 'labels_approved': 0,
          'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2)})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        worker(args.protocol, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='fresh-validation-planning',
                    gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == '__main__':
    main()
