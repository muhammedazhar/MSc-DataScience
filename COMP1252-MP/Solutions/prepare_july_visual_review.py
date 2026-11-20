#!/usr/bin/env python3
"""Render the two July endpoint comparisons from verified V18 caches."""
import argparse
import hashlib
import os
from pathlib import Path
import resource
import sys
import tempfile

from screen_repaired_validation import (
    GROUP, POLICY, RESOURCES, ROOT, digest, load_safe_cache, read, require, rooted, write,
)

REVIEW = ROOT/'Docs/Reviews/ValidationJulyVisualV19'
PROTOCOL = REVIEW/'protocol.json'
STATUS = 'passed_automatic_screen_pending_visual_review'
DATES = ('2018-07-06', '2018-07-31')
DISPLAY = {'rgb_reflectance_range': [0, .30], 'ndvi_range': [-.1, .9],
           'ndvi_difference_range': [-.5, .5], 'detail_padding_pixels': 6, 'resampling': 'nearest'}


def verified_inputs():
    protocol = read(PROTOCOL)
    require(protocol['schema_version'] == 19 and protocol['resources'] == RESOURCES
            and protocol['display'] == DISPLAY, 'Fixed V19 review scope differs')
    paths = {}
    for role, receipt in protocol['source_files'].items():
        path = rooted(receipt['path'])
        require(digest(path) == receipt['sha256'], 'Pinned evidence changed: '+role)
        paths[role] = path
    implementation = protocol['implementation_sha256']
    require(str(Path(__file__).relative_to(ROOT)) in implementation, 'Pin the packet renderer first')
    for name, checksum in {**protocol['frozen_implementation_sha256'], **implementation}.items():
        require(digest(rooted(name)) == checksum, 'Pinned implementation changed: '+name)
    screen, screen_meta, replay = (read(paths[key]) for key in ('screen', 'screen_metadata', 'qa_replay'))
    integrity = read(paths['qa_integrity'])
    require(screen_meta['execution_status'] == replay['execution_status'] == 'completed'
            and screen['qa_policy_id'] == POLICY and not screen['processing_errors']
            and all(replay['checks'].values()) and integrity['status'] == 'passed'
            and screen_meta['artifact_sha256']['screen.json'] == integrity['screen_sha256'] == digest(paths['screen'])
            and integrity['verification_metadata_sha256'] == digest(paths['qa_replay'])
            and integrity['cache_index_sha256'] == digest(paths['cache_index'])
            and screen['derived_pairs'] == read(paths['lineage'])['derived_pairs'], 'V18 replay or screen differs')
    picks = [(index, pair, row) for index, (pair, row) in enumerate(zip(screen['derived_pairs'], screen['results']))
             if row['quality_status'] == STATUS]
    roster = [{'area': area, 'pair_id': pair['pair_id'], 'event_id': pair['anchor_event_ids'][0],
               'pre_date': pair['pre_date'], 'post_date': pair['post_date']}
              for area, (_, pair, _) in zip(('A', 'B'), picks)]
    require(len(picks) == 2 and roster == protocol['roster']
            and [r['pair_id'] for r in roster] == integrity['passed_pairs_pending_visual_review'],
            'Fixed review roster differs')
    return paths, screen, screen_meta, list(zip(roster, picks))


def array_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def detail_box(target):
    import numpy as np

    rows, cols = np.nonzero(target)
    pad = DISPLAY['detail_padding_pixels']
    return (max(0, int(cols.min())-pad), max(0, int(rows.min())-pad),
            min(64, int(cols.max())+pad+1), min(64, int(rows.max())+pad+1))


def rgb(array):
    import numpy as np
    return np.rint(255*np.clip(np.nan_to_num(array.transpose(1, 2, 0))/.30, 0, 1)).astype('uint8')


def colour_map(array, limits, colours):
    import numpy as np

    steps = np.linspace(*limits, len(colours))
    image = np.stack([np.interp(array, steps, channel) for channel in np.array(colours).T], axis=-1)
    return np.rint(np.nan_to_num(image)).astype('uint8')


def outline(draw, mask, scale, colour):
    import numpy as np

    padded = np.pad(mask, 1)
    for row, col in np.argwhere(mask):
        x, y = int(col)*scale, int(row)*scale
        for adjacent, line in (
            (padded[row, col+1], (x, y, x+scale, y)),
            (padded[row+2, col+1], (x, y+scale-1, x+scale, y+scale-1)),
            (padded[row+1, col], (x, y, x, y+scale)),
            (padded[row+1, col+2], (x+scale-1, y, x+scale-1, y+scale)),
        ):
            if not adjacent:
                draw.line(line, fill=colour, width=2)


def panel(values, box, scale, overlays=()):
    from PIL import Image, ImageDraw

    x0, y0, x1, y1 = box
    result = Image.fromarray(values[y0:y1, x0:x1]).resize(
        ((x1-x0)*scale, (y1-y0)*scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(result)
    for mask, colour in overlays:
        outline(draw, mask[y0:y1, x0:x1], scale, colour)
    return result


def board(path, title, subtitle, columns, footer):
    from PIL import Image, ImageDraw, ImageFont

    width, height = columns[0][1].size
    margin, top = 20, 86
    image = Image.new('RGB', (max(675, margin+(width+margin)*len(columns)), top+height+50), '#f4f5f1')
    draw = ImageDraw.Draw(image)
    draw.text((margin, 12), title, fill='#172320', font=ImageFont.load_default(size=20))
    draw.text((margin, 42), subtitle, fill='#314640', font=ImageFont.load_default(size=13))
    for index, (label, pixels) in enumerate(columns):
        x = margin+index*(width+margin)
        draw.text((x, 65), label, fill='#172320', font=ImageFont.load_default(size=14))
        image.paste(pixels, (x, top))
    draw.text((margin, top+height+14), footer, fill='#314640', font=ImageFont.load_default(size=13))
    image.save(path)


def visual_views(output, area, observations, masks, box):
    pre, post = observations
    dates = ('6 July 2018', '31 July 2018')
    images = [rgb(record['rgb']) for record in observations]
    prefix = f'area-{area.lower()}'
    filenames = [prefix+'-'+name+'.png' for name in ('plain', 'detail', 'ndvi')]
    board(output/filenames[0], f'Area {area} | Full crop', 'Matched reflectance 0–0.30',
          [(day, panel(image, (0, 0, 64, 64), 8)) for day, image in zip(dates, images)],
          'North up · 64 × 64 source pixels · 10 m per pixel')
    overlays = [(masks['context'], '#70eff3'), (masks['target'], '#ffdb63'), (masks['interior'], '#ffffff')]
    details = [(day, panel(image, box, 12, overlays+[(masks['shadow'][index], '#ff7ac8')]))
               for index, (day, image) in enumerate(zip(dates, images))]
    board(output/filenames[1], f'Area {area} | Target detail', 'Matched reflectance 0–0.30 · native pixels enlarged',
          details, 'Yellow target · white eroded interior · cyan context · pink shadow-risk pixels')
    greens = ['#973b26', '#d3a96e', '#eae7ca', '#83a77a', '#23513f']
    changes = ['#9d2f28', '#e3b08a', '#f3f1e8', '#89b9ab', '#246554']

    def palette(codes):
        return [tuple(bytes.fromhex(code[1:])) for code in codes]

    bands = [colour_map(record['ndvi'], (-.1, .9), palette(greens)) for record in observations]
    bands.append(colour_map(post['ndvi']-pre['ndvi'], (-.5, .5), palette(changes)))
    board(output/filenames[2], f'Area {area} | Vegetation index',
          'NDVI −0.1 to 0.9 on both dates · change −0.5 to +0.5',
          [(label, panel(values, box, 12, [(masks['target'], '#ffdb63')]))
           for label, values in zip((*dates, 'Change, 31 July − 6 July'), bands)],
          'Change panel: rust = decrease, green = increase · yellow = target')
    return filenames


def build_packet(output):
    import numpy as np
    import rasterio
    from scipy.ndimage import binary_erosion
    from assess_pixel_quality import MASK_NAMES, core_grid, label_target, load_events
    from audit_validation_context import context_mask
    from validate_sample import burn

    paths, screen, meta, picks = verified_inputs()
    archived = {r['cache_key']: r for r in read(paths['cache_index'])}
    events = load_events(paths['catalogue'].parent)
    cases = []
    for entry, (index, pair, row) in picks:
        require(pair['group_id'] == GROUP and pair['split'] == 'validation'
                and (pair['pre_date'], pair['post_date']) == DATES and row['pair_id'] == pair['pair_id'],
                'Endpoint crop identity differs')
        grid = core_grid(pair['grid'])
        temporal = label_target(pair, events).astype(bool)
        anchor = burn([events[entry['event_id']]['geometry']], grid).astype(bool)
        target = anchor & temporal
        interior = binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
        context = context_mask(target, radius=2)
        checked = screen['anchor_contexts'][index]
        require(array_hash(target) == checked['target_mask_sha256']
                and array_hash(interior) == checked['interior_mask_sha256']
                and array_hash(context) == checked['target']['context_mask_sha256'], 'Target or QA context differs')
        mask_name = row['qa_mask_file']
        mask_path = paths['screen'].parent/mask_name
        require(digest(mask_path) == meta['artifact_sha256'][mask_name], 'Published QA mask changed')
        with rasterio.open(mask_path) as tif:
            require(tif.shape == (64, 64) and tif.count == 6 and tif.transform == grid['transform']
                    and tif.crs.to_string() == pair['grid']['crs']
                    and tif.descriptions == (*MASK_NAMES, 'reference_new_loss')
                    and np.array_equal(tif.read(6), temporal), 'Published temporal plane differs')
        observations, caches = [], []
        for side, receipt in zip(('pre', 'post'), screen['cache_receipts'][index*2:index*2+2]):
            known = archived[receipt['cache_key']]
            snapshot = paths['cache_index'].parent.parent/known['snapshot']
            require(all(known[key] == receipt[key] for key in receipt)
                    and digest(snapshot) == receipt['sha256']
                    and {'pair_id': pair['pair_id'], 'side': side} in known['bindings'], 'Endpoint snapshot differs')
            record = load_safe_cache(receipt, row['per_date'][side]['observation'])
            require(record['metadata']['scene_id'] == pair[side+'_scene_id']
                    and record['metadata']['grid'] == pair['grid'], 'Cached source or crop grid differs')
            require(np.isfinite(record['rgb'][:, context]).all() and np.isfinite(record['ndvi'][context]).all(),
                    'Nonfinite endpoint values in full target context')
            observations.append(record)
            caches.append(receipt)
        shadow = [record['shadow_risk'] & context for record in observations]
        shadow_points = [mask.nonzero() for mask in shadow]
        masks = {'target': target, 'interior': interior, 'context': context, 'shadow': shadow}
        box = detail_box(target)
        files = visual_views(output, entry['area'], observations, masks, box)
        stats = {}
        for label, mask in (('target', target), ('eroded_interior', interior)):
            pre, post = (record['ndvi'][mask] for record in observations)
            stats[label] = {'pixels': int(mask.sum()), 'median_pre': float(np.median(pre)),
                            'median_post': float(np.median(post)), 'median_change': float(np.median(post-pre)),
                            'decreased_pixels': int(np.count_nonzero(post<pre))}
        cases.append({**entry, 'grid': pair['grid'], 'detail_col_row_bounds': box,
                      'cache_receipts': caches, 'qa_mask_sha256': digest(mask_path),
                      'target_mask_sha256': array_hash(target), 'interior_mask_sha256': array_hash(interior),
                      'shadow_context_row_col': [list(zip(rr.tolist(), cc.tolist())) for rr, cc in shadow_points],
                      'ndvi_descriptive_only': stats, 'files': files})
    write(output/'packet.json', {'schema_version': 19, 'protocol_sha256': digest(PROTOCOL), 'cases': cases})
    return {file.name: digest(file) for file in sorted(output.iterdir()) if file.is_file()}


def worker(args, output):
    if args.mode == 'prepare':
        artifacts = build_packet(output)
    else:
        packet = args.packet.resolve()
        require(packet.is_relative_to(ROOT), 'Review packet must be inside the project')
        prior = read(packet/'metadata.json')
        require(prior['execution_status'] == 'completed' and prior['mode'] == 'prepare'
                and prior['protocol_sha256'] == digest(PROTOCOL), 'Incorrect review packet')
        with tempfile.TemporaryDirectory(dir=output) as directory:
            artifacts = build_packet(Path(directory))
        require(artifacts == prior['artifact_sha256'] and
                all(digest(packet/name) == checksum for name, checksum in artifacts.items()), 'Packet replay differs')
    require('torch' not in sys.modules, 'Unexpected model import in packet renderer')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(output/'metadata.json', {
        'execution_status': 'completed', 'mode': args.mode, 'protocol_sha256': digest(PROTOCOL),
        'artifact_sha256': artifacts, 'pair_count': 2, 'cache_bindings_checked': 4,
        'source_jp2_reads': 0, 'model_executions': 0, 'labels_approved': 0,
        'worker_high_water_rss_mib': round(peak/(2**20 if sys.platform == 'darwin' else 1024), 2),
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
        run_guarded(args, script_path=__file__, operation='july-visual-packet',
                    gdal_cache_mib=16, warp_memory_mib=0)


if __name__ == '__main__':
    main()
