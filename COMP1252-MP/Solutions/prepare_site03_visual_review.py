#!/usr/bin/env python3
"""Render a fixed model-blind site-03 packet from the verified V30 archives."""

import argparse
import os
from pathlib import Path
import resource
import sys
import tempfile

from prepare_july_visual_review import array_hash, board, colour_map, detail_box, panel, rgb
from preflight_fresh_validation_sources import checked_artifacts, evidence_hashes
from run_site03_quality import DATES, RESOURCES, ROOT, digest, inputs, load, require, require_phase, rooted, write


REVIEW = ROOT/'Docs/Reviews/ValidationSite03ReviewV31'
V30 = ROOT/'Docs/Reviews/ValidationSite03ScreenV30'
PROTOCOL = REVIEW/'visual_protocol.json'
PAIR_ID = 'derived-site03-891e388f6c6aaee1a9ea'
GROUP_ID = 'group-a85fa2022a25e1a72e2c'
DISPLAY = {
    'rgb_reflectance': [0, .30], 'ndvi': [-.1, .9], 'ndvi_difference': [-.5, .5],
    'resampling': 'nearest', 'control_display_box_col_row': [0, 0, 26, 26],
}
DAYS = ('10 May 2018', '30 May 2018', '4 June 2018')
FULL = (0, 0, 64, 64)
NDVI_PALETTE = ('#973b26', '#d3a96e', '#eae7ca', '#83a77a', '#23513f')
CHANGE_PALETTE = ('#9d2f28', '#e3b08a', '#f3f1e8', '#89b9ab', '#246554')


def verified_packet_inputs():
    from plan_site03_qa import prepare_geometry, site03_masks
    from screen_site03_quality import archived_observation, bound_inputs

    protocol = load(PROTOCOL)
    require(protocol['schema_version'] == 31 and protocol['site_id'] == 'site-03'
            and protocol['pair_id'] == PAIR_ID and protocol['group_id'] == GROUP_ID
            and protocol['target_dates'] == list(DATES[1:])
            and protocol['control_dates'] == list(DATES)
            and protocol['display'] == DISPLAY and protocol['resources'] == RESOURCES,
            'V31 review protocol differs from fixed site, dates or display')
    for name, expected in protocol['input_and_code_sha256'].items():
        require(digest(rooted(name)) == expected, 'V31 pinned evidence differs: '+name)
    v30_path = V30/'protocol.json'
    v30, paths = inputs(v30_path)
    checksum = digest(v30_path)
    lineage = require_phase(V30/'lineage', 'lineage.json', 'prepare', checksum)
    screen = require_phase(V30/'screen', 'screen.json', 'screen', checksum)
    replay = require_phase(V30/'verification', 'verification.json', 'verify', checksum)
    require(screen['review_candidate_pair_ids'] == replay['review_candidate_pair_ids'] == [PAIR_ID]
            and screen['control_qa_status'] == 'clear_on_all_three_dates'
            and len(replay['checks']) == 8 and all(replay['checks'].values())
            and screen['processing_errors'] == [],
            'V30 QA/replay did not retain the exact target and control')
    require(replay['verified_screen_metadata_sha256'] == digest(V30/'screen/metadata.json')
            and screen['pair'] == lineage['pair']
            and screen['source_asset_receipts'] == lineage['source_asset_receipts'],
            'V30 QA/replay does not bind to the published lineage')
    scenes = {}
    for day in DATES:
        publication = V30/'publication'/day
        meta = load(publication/'metadata.json')
        require(digest(publication/'metadata.json') == lineage['publication_metadata_sha256_by_date'][day],
                'V30 publication metadata differs: '+day)
        checked_artifacts(publication, meta['artifact_sha256'])
        scenes[day] = load(publication/'download.json')['final_metadata']
    geometry = prepare_geometry(v30, paths, scenes)
    require(all(geometry[key] == lineage[key] for key in ('pair', 'geometry', 'control'))
            and geometry['pair']['group_id'] == GROUP_ID, 'Rebuilt V30 review geometry differs')
    masks = site03_masks(geometry['pair'], geometry['events'], geometry['control'])
    require({key: array_hash(mask) for key, mask in masks.items()} == lineage['geometry']['mask_sha256'],
            'V31 review mask hashes differ from V30')
    _, bindings = bound_inputs({**geometry, 'scenes': scenes, 'site': lineage['site'],
                                'source_asset_receipts': lineage['source_asset_receipts'],
                                'protocol_sha256': checksum})
    observations = {}
    for day in DATES:
        record = screen['observations'][day]
        observations[day] = archived_observation(V30/'screen', record['cache_receipt'],
                                                 record['observation'], bindings[day])
    return lineage, masks, observations, screen


def columns(days, values, box, scale, overlays=()):
    return [(day, panel(value, box, scale, overlays)) for day, value in zip(days, values, strict=True)]


def rgb_views(output, observations, masks):
    rgbs = [rgb(observations[day]['rgb']) for day in DATES]
    control_box = tuple(DISPLAY['control_display_box_col_row'])
    target_box = detail_box(masks['anchor_target'])
    model_blind = 'Matched reflectance 0–0.30 · north up · 10 m source pixels'
    board(output/'target-rgb-plain.png', 'Site-03 | Target comparison', model_blind,
          columns(DAYS[1:], rgbs[1:], FULL, 8), '30 May and 4 June 2018')
    board(output/'control-rgb-plain.png', 'Site-03 | Control comparison', model_blind,
          columns(DAYS, rgbs, control_box, 12), 'Same northwest window on each date')
    target_lines = [(masks['anchor_context'], '#70eff3'), (masks['anchor_target'], '#ffdb63'),
                    (masks['anchor_interior'], '#ffffff'), (masks['cumulative_pre'], '#ff7ac8')]
    board(output/'target-rgb-outlined.png', 'Site-03 | Target detail', model_blind,
          columns(DAYS[1:], rgbs[1:], target_box, 12, target_lines),
          'Yellow target · white interior · cyan context · pink earlier catalogue polygons')
    control_lines = [(masks['control_context'], '#70eff3'), (masks['control'], '#ffdb63')]
    board(output/'control-rgb-outlined.png', 'Site-03 | Fixed control', model_blind,
          columns(DAYS, rgbs, control_box, 12, control_lines), 'Yellow control · cyan context')


def ndvi_views(output, observations, masks):
    import numpy as np

    def palette(colours):
        return [tuple(bytes.fromhex(item[1:])) for item in colours]

    arrays = [observations[date]['ndvi'] for date in DATES]
    paired_change = arrays[2] - arrays[1]
    bands = [colour_map(array, DISPLAY['ndvi'], palette(NDVI_PALETTE)) for array in arrays]
    change = colour_map(paired_change, DISPLAY['ndvi_difference'], palette(CHANGE_PALETTE))
    board(output/'target-ndvi.png', 'Site-03 | Target vegetation index',
          'NDVI −0.1 to 0.9 · change −0.5 to +0.5',
          columns((*DAYS[1:], 'Change, 4 June − 30 May'), [*bands[1:], change],
                  detail_box(masks['anchor_target']), 12, [(masks['anchor_target'], '#ffdb63')]),
          'Change panel: rust = decrease, green = increase · yellow target')
    board(output/'control-ndvi.png', 'Site-03 | Control vegetation index', 'NDVI −0.1 to 0.9 on each date',
          columns(DAYS, bands, tuple(DISPLAY['control_display_box_col_row']), 12,
                  [(masks['control'], '#ffdb63')]), 'Yellow control')
    summaries = {}
    for name in ('anchor_target', 'anchor_interior', 'anchor_context', 'control', 'control_context'):
        mask = masks[name]
        series = arrays if name.startswith('control') else arrays[1:]
        dates = DATES if name.startswith('control') else DATES[1:]
        values = [array[mask] for array in series]
        require(all(np.isfinite(sample).all() for sample in values), 'Nonfinite cached NDVI: '+name)
        change_values = values[-1] - values[-2]
        summaries[name] = {
            'pixels': int(mask.sum()),
            'median_by_date': {day: float(np.median(sample)) for day, sample in zip(dates, values, strict=True)},
            'pair_median_change': float(np.median(change_values)),
            'pair_pixels_decreased': int(np.count_nonzero(change_values < 0)),
        }
    return summaries


def packet(output):
    lineage, masks, observations, screen = verified_packet_inputs()
    rgb_views(output, observations, masks)
    stats = ndvi_views(output, observations, masks)
    write(output/'packet.json', {
        'site_id': 'site-03', 'pair_id': PAIR_ID, 'group_id': GROUP_ID, 'grid': lineage['pair']['grid'],
        'target_dates': list(DATES[1:]), 'control_dates': list(DATES),
        'scene_ids': {day: screen['observations'][day]['scene_id'] for day in DATES},
        'source_cache_sha256_by_date': {day: screen['observations'][day]['cache_receipt']['sha256'] for day in DATES},
        'mask_sha256': lineage['geometry']['mask_sha256'], 'ndvi_descriptive_only': stats,
        'source_jp2_reads': 0, 'forest_model_executions': 0, 'cloud_classifier_calls': 0,
        'labels_approved': 0, 'validation_eligible': False,
    })
    return evidence_hashes(output)


def worker(args, output):
    expected = digest(PROTOCOL)
    if args.mode == 'prepare':
        artifacts = packet(output)
        payload = {'artifact_sha256': artifacts}
    else:
        original = args.packet.resolve()
        require(original.is_relative_to(REVIEW), 'Review packet must be in the V31 report directory')
        metadata = load(original/'metadata.json')
        require(metadata['execution_status'] == 'completed' and metadata['mode'] == 'prepare'
                and metadata['protocol_sha256'] == expected and metadata['artifact_sha256'] == evidence_hashes(original),
                'Prepared V31 packet binding differs')
        with tempfile.TemporaryDirectory(dir=output) as directory:
            artifacts = packet(Path(directory))
        require(artifacts == metadata['artifact_sha256'], 'V31 packet replay differs')
        payload = {'verified_artifact_sha256': artifacts, 'verified_packet_metadata_sha256': digest(original/'metadata.json')}
    require(digest(PROTOCOL) == expected and 'torch' not in sys.modules,
            'Protocol changed or forest runtime imported')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'schema_version': 31, 'protocol_sha256': expected, 'execution_status': 'completed', 'mode': args.mode,
        'source_jp2_reads': 0, 'model_executions': 0, 'labels_approved': 0,
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2), **payload,
    })


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'verify'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(max_rss_mib=512, min_available_gib=1.5, max_seconds=300)
    args = parser.parse_args()
    if (args.mode == 'verify') != (args.packet is not None):
        parser.error('Only verify mode uses --packet')
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation='site03-model-blind-packet',
                    gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
