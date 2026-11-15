#!/usr/bin/env python3
"""Independently check the small reviewed pilot without source-band reads."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(args, output):
    import numpy as np
    import rasterio
    from affine import Affine
    from shapely.geometry import box

    dataset = args.dataset.resolve()
    metadata = json.loads((dataset/'metadata.json').read_text())
    rows = [json.loads(line) for line in (dataset/'training_manifest.jsonl').read_text().splitlines()]
    decisions = [json.loads(line) for line in (dataset/'review_decisions.jsonl').read_text().splitlines()]
    catalogue = json.loads((ROOT/'Datasets/AnnotationAudit/v1/annotations_clean.geojson').read_text())
    held_out = [box(*f['properties']['group_footprint_bounds']) for f in catalogue['features']
                if f['properties']['split'] in ('validation', 'test')]
    frozen = {}
    with (ROOT/'Datasets/AnnotationAudit/v1/candidate_pairs.jsonl').open() as stream:
        wanted = {row['pair_id'] for row in rows}
        for line in stream:
            pair = json.loads(line)
            if pair['pair_id'] in wanted:
                frozen[pair['pair_id']] = pair
    for name, checksum in metadata['output_sha256_excluding_this_metadata'].items():
        require(digest(dataset/name) == checksum, f'Export hash changed: {name}')
    for name, checksum in metadata['prepared_evidence_sha256'].items():
        require(digest(ROOT/name) == checksum, f'Prepared evidence changed: {name}')
    for name, stats in metadata['source_file_stats'].items():
        current = (ROOT/name).stat()
        require([current.st_size, current.st_mtime_ns] == stats, 'Source statistics changed')
    accepted = {d['event_id']: d for d in decisions if d['accepted_for_partial_pilot']}
    require(len(accepted) == len(rows) == metadata['accepted_example_count'], 'Accepted coverage mismatch')
    require(len(decisions) == metadata['review_count'], 'Review dispositions missing')
    require(len({r['group_id'] for r in rows}) == len(rows), 'Accepted groups repeat')
    counts, interval_widths, cache_keys, dates_checked = Counter(), [], set(), 0
    expected_channels = tuple(f'{period}_{band}' for period in ('pre', 'post', 'difference')
                              for band in ('B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12', 'NDVI', 'NDMI'))
    for row in rows:
        pair = frozen[row['pair_id']]
        require(row['split'] == pair['split'] == 'train', 'Non-training example')
        require(row['grid'] == pair['grid'] and row['group_id'] == pair['group_id'], 'Frozen grid/group changed')
        footprint = box(*row['grid']['bounds'])
        require(not any(footprint.intersects(g) for g in held_out), 'Held-out footprint overlap')
        arrays = {}
        for key in ('input_features', 'supervision_mask', 'valid_mask'):
            with rasterio.open(dataset/row['artifact_paths'][key]) as source:
                require(source.transform == Affine(*row['grid']['transform']) and source.shape == (64, 64), 'Export grid mismatch')
                require(source.crs == rasterio.crs.CRS.from_user_input(row['grid']['crs']), 'Export CRS mismatch')
                if key == 'input_features':
                    require(source.count == 27 and source.dtypes == ('float32',)*27, 'Feature shape/type changed')
                    require(source.descriptions == expected_channels, 'Feature channel order changed')
                else:
                    require(source.count == 1 and source.dtypes == ('uint8',), 'Label/mask shape/type changed')
                arrays[key] = source.read()
        features, labels, valid = arrays['input_features'], arrays['supervision_mask'][0], arrays['valid_mask'][0]
        require(np.array_equal(features[18:], features[9:18]-features[:9], equal_nan=True), 'Signed differences changed')
        require((features[18:] < 0).any(), 'Expected negative change values were lost')
        require(np.isin(labels, (0, 1, 255)).all() and np.array_equal(valid, labels != 255), 'Ignore supervision mismatch')
        require(np.isfinite(features[:, labels != 255]).all(), 'Missing supervised features')
        for start in (0, 9):
            for left, right, index in ((3, 2, 7), (4, 5, 8)):
                a, b = features[start+left], features[start+right]
                expected = np.divide(a-b, a+b, out=np.zeros_like(a), where=np.isfinite(a+b) & ((a+b) != 0))
                require(np.allclose(features[start+index], expected, atol=2e-6, equal_nan=True), 'Index scaling/sign mismatch')
        decision = accepted[row['event_id']]['review']
        case_path = ROOT/decision['case_file']
        case = json.loads(case_path.read_text())
        require(case['pair'] == pair, 'Prepared pair differs from frozen manifest')
        with rasterio.open(case_path.parent/'positive_interior.tif') as source:
            positive = source.read(1).astype(bool)
        with rasterio.open(case_path.parent/'proposed_control.tif') as source:
            stable = source.read(1).astype(bool)
        require(np.array_equal(labels == 1, positive) and np.array_equal(labels == 0, stable), 'Approved ROI changed')
        require(positive.sum() >= 4 and stable.sum() >= 64 and not (positive & stable).any(), 'Insufficient/disjoint ROI support')
        observed = decision['observations']
        require([(d['scene_id'], d['date']) for d in observed] == [(t['scene_id'], t['date']) for t in case['timeline']], 'Timeline evidence coverage changed')
        clear_dates, forest_dates, stable_dates = [], [], []
        for statement, prepared in zip(observed, case['timeline']):
            if prepared.get('observation') is None:
                require(statement['target_state'] not in ('forest', 'clearing_visible') and statement['control_state'] != 'stable_forest', 'Unavailable observation claimed clear')
                continue
            path = ROOT/prepared['observation']['cache_path']
            with np.load(path, allow_pickle=False) as cache:
                unusable = np.logical_or.reduce([cache[name] for name in ('cloud', 'cloud_buffer', 'shadow_risk', 'haze_risk', 'invalid')])
                cache_meta = json.loads(str(cache['metadata']))
            require(cache_meta == prepared['observation'], 'Observation metadata changed')
            cache_keys.add(cache_meta['cache_key'])
            with rasterio.open(case_path.parent/'target_mask.tif') as source:
                target = source.read(1).astype(bool)
            if statement['target_state'] in ('forest', 'clearing_visible'):
                require(not (unusable & target).any(), 'Clear target claim intersects QA')
                (forest_dates if statement['target_state'] == 'forest' else clear_dates).append(statement['date'])
            if statement['control_state'] == 'stable_forest':
                require(not (unusable & stable).any(), 'Clear control claim intersects QA')
                stable_dates.append(statement['date'])
            dates_checked += 1
        require(len(clear_dates) >= 2 and len(stable_dates) >= 3, 'Persistence/control evidence missing')
        require(max(forest_dates) < min(clear_dates), 'Contradictory target chronology')
        bounds = row['visibility_evidence']['observed_change_interval']
        require(bounds['lower_bound_exclusive'] == max(forest_dates) and bounds['upper_bound_inclusive'] == min(clear_dates), 'Observed bounds changed')
        require(pair['pre_date'] in forest_dates and pair['post_date'] in clear_dates, 'Primary endpoint evidence missing')
        require(pair['pre_date'] in stable_dates and pair['post_date'] in stable_dates, 'Primary control evidence missing')
        interval_widths.append(bounds['width_days'])
        for name, value in (('supervised_positive_pixels', int(positive.sum())), ('supervised_stable_pixels', int(stable.sum())), ('ignored_pixels', int((labels == 255).sum()))):
            require(row[name] == value, 'Supervision count mismatch')
            counts[name] += value
    require(dict(counts) == metadata['supervision_pixel_counts'], 'Aggregate supervision counts differ')
    result = {'verification_status': 'passed', 'examples_checked': len(rows), 'reviews_retained': len(decisions),
              'timeline_observations_checked': dates_checked, 'unique_observation_caches_checked': len(cache_keys),
              'pixel_counts': dict(counts), 'visibility_interval_width_days': interval_widths,
              'source_jp2_pixel_reads': 0, 'model_predictions': 0,
              'dataset_metadata_sha256': digest(dataset/'metadata.json'),
              'verification_implementation_sha256': digest(__file__),
              'scope': 'Artifact/evidence consistency only; assistant labels are not expert-certified ground truth.'}
    (output/'metadata.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'report.md').write_text(f"# Curated pilot verification\n\nPassed all {len(rows)} exported examples, {len(decisions)} retained decisions, and {dates_checked} timeline observations. Signed features, indices, grids, ROI labels, QA evidence, source fingerprints, split isolation and date bounds agree. No JP2 pixels or models were read.\n")


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'Datasets/CuratedPilot/v1/dataset')
    parser.add_argument('--output', type=Path, default=ROOT/'Datasets/CuratedPilot/v1/verification')
    parser.add_argument('--max-rss-mib', type=int, default=512)
    parser.add_argument('--min-available-gib', type=float, default=1.5)
    parser.add_argument('--max-seconds', type=int, default=180)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300, 'Unsafe resource settings')
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    if args.worker_output:
        os.nice(10)
        verify(args, args.worker_output)
        import resource
        path = args.worker_output/'metadata.json'
        metadata = json.loads(path.read_text())
        metadata['worker_high_water_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024)
        path.write_text(json.dumps(metadata, indent=2)+'\n')
    else:
        run_guarded(args, script_path=__file__, operation='curated-pilot-verification', gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
