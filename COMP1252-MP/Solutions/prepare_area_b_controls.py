"""Freeze and render prediction-blind control candidates on the Area B crop."""

from audit_validation_context import context_mask
from prepare_july_visual_review import array_hash, board, panel, rgb
from prepare_validation_manual import MASK_NAMES, control_roster, observation_metrics, require
from prepare_validation_review import catalogue_exclusion
from verify_validation_manual import independent_roster

DATES = ('2018-07-06', '2018-07-31', '2018-08-10')
CONTROL_COLOURS = ('#ff92d4', '#b4afff', '#9ef787')


def _control_inputs(pair, target, target_context, features, feature_valid, observations):
    import numpy as np

    require(tuple(pair[side+'_date'] for side in ('pre', 'post')) == DATES[:2],
            'Area B control selection requires the frozen July endpoints')
    require(len(observations) == len(DATES), 'Control review requires the three fixed dates')
    for date, record in zip(DATES, observations):
        meta = record['metadata']
        require(meta['grid'] == pair['grid'] and meta['scene_id'].split('_')[2].startswith(date.replace('-', '')+'T'),
                'Control observation date or grid differs')
    require(all(record['metadata']['scene_id'] == pair[side+'_scene_id']
                and record['metadata']['context_incomplete'] is False
                for side, record in zip(('pre', 'post'), observations[:2])),
            'Control endpoints must use the exact pair and full padded QA source')
    require(np.array_equal(target_context, context_mask(target, radius=2))
            and target_context.dtype == np.dtype(bool), 'Target exclusion requires the exact full context')
    require(features.shape == (27, 64, 64) and features.dtype == np.dtype('float32')
            and feature_valid.shape == (64, 64) and feature_valid.dtype == np.dtype(bool)
            and np.array_equal(feature_valid, np.isfinite(features).all(axis=0)),
            'Control eligibility requires the exact feature arrays and validity mask')

    catalogue = catalogue_exclusion(pair, margin=20)
    control_exclusion = catalogue | target_context
    arrays = {'features': features, 'feature_valid': feature_valid, 'control_exclusion': control_exclusion}
    for side, record in zip(('pre', 'post'), observations[:2]):
        arrays.update({side+'_qa': np.stack([record[name] for name in MASK_NAMES]),
                       side+'_rgb': record['rgb'], side+'_ndvi': record['ndvi']})
    return catalogue, arrays


def _bounds(grid, row, col, size):
    x_origin, y_origin = grid['transform'][2], grid['transform'][5]
    return [x_origin+10*col, y_origin-10*(row+size),
            x_origin+10*(col+size), y_origin-10*row]


def _records(grid, roster, rois, contexts, observations, arrays):
    import numpy as np

    assigned = np.zeros((64, 64), dtype=bool)
    records = []
    for control, roi, context in zip(roster, rois, contexts):
        row, col, size = (control[key] for key in ('row', 'col', 'width'))
        require(np.array_equal(context, context_mask(roi, radius=2))
                and int(context.sum()) == control['context_pixels']
                and not (context & (assigned | arrays['control_exclusion'])).any()
                and arrays['feature_valid'][context].all(), 'Control context is ineligible or overlaps')
        assigned |= context
        observation_rows = []
        for date, observation in zip(DATES, observations):
            metrics = observation_metrics(observation, roi, context)
            metadata = observation['metadata']
            full_source_context = metadata['context_incomplete'] is False
            observation_rows.append({'date': date, 'scene_id': metadata['scene_id'],
                                     'metrics': metrics, 'full_source_context': full_source_context,
                                     'eligible_for_visual_review': metrics['context_clear'] and full_source_context})
        require(all(entry['eligible_for_visual_review'] for entry in observation_rows[:2]),
                'A selected control fails the frozen July QA rule')
        records.append({**control,
                        'roi_bounds': _bounds(grid, row, col, size),
                        'context_bounds': _bounds(grid, row-2, col-2, size+4),
                        'roi_sha256': array_hash(roi), 'context_sha256': array_hash(context),
                        'finite_model_channels_at_endpoints': 27, 'observations': observation_rows})
    return records


def _render(output, target, target_context, observations, records, rois, contexts):
    images = [rgb(record['rgb']) for record in observations]
    crop = (0, 0, 64, 64)
    footer = 'North up | 64 x 64 source pixels | 10 m per pixel'
    display = 'Matched RGB reflectance 0 to 0.30 | native pixels enlarged'
    plain, outlined = 'controls-plain.png', 'controls-outlined.png'
    board(output/plain, 'Area B | Control review crop', display,
          [(date, panel(image, crop, 7)) for date, image in zip(DATES, images)], footer)
    overlays = [(target_context, '#70eff3'), (target, '#ffdb63')]
    overlays.extend(zip(rois, CONTROL_COLOURS))
    legend = 'Cyan: target context | Yellow: target'
    names = ('pink', 'lavender', 'green')
    for entry, name in zip(records, names):
        legend += f' | {name}: {entry["control_id"]}'
    board(output/outlined, 'Area B | Fixed control locations', display,
          [(date, panel(image, crop, 7, overlays)) for date, image in zip(DATES, images)], legend)
    views = [plain, outlined]

    for record, roi, context in zip(records, rois, contexts):
        row, col, size = (record[key] for key in ('row', 'col', 'width'))
        x_min, y_min = max(0, col-4), max(0, row-4)
        x_max, y_max = min(64, col+size+4), min(64, row+size+4)
        box = x_min, y_min, x_max, y_max
        label = record['control_id']
        detail_plain, detail_outlined = f'{label}-plain.png', f'{label}-outlined.png'
        position = f'ROI row {row}, column {col} | {size} x {size} source pixels | zero-based north-up grid'
        dates = [entry['date'] if entry['eligible_for_visual_review'] else entry['date']+' (QA obscured)'
                 for entry in record['observations']]
        board(output/detail_plain, f'Area B | {label}', display,
              [(date, panel(image, box, 14)) for date, image in zip(dates, images)], position)
        board(output/detail_outlined, f'Area B | {label} bounds', display,
              [(date, panel(image, box, 14, [(context, '#70eff3'), (roi, '#ffdb63')]))
               for date, image in zip(dates, images)], 'Yellow: control square | Cyan: full two-pixel context')
        record['detail_window'] = {'rows': [y_min, y_max], 'columns': [x_min, x_max], 'stop_exclusive': True}
        record['views'] = {'plain': detail_plain, 'outlined': detail_outlined}
        views.extend((detail_plain, detail_outlined))
    return views


def prepare_controls(output, pair, target, target_context, features, feature_valid, observations, *, verify=False):
    """Use the frozen V4 July-only roster, then assess the same squares in August."""
    catalogue, arrays = _control_inputs(pair, target, target_context, features, feature_valid, observations)
    if verify:
        rois, contexts, roster = independent_roster(arrays)
    else:
        roster, rois, contexts = control_roster(arrays)
    records = _records(pair['grid'], roster, rois, contexts, observations, arrays)
    views = _render(output, target, target_context, observations, records, rois, contexts)
    return {'grid': pair['grid'], 'controls': records, 'views': views,
            'catalogue_exclusion_pixels': int(catalogue.sum()), 'catalogue_exclusion_sha256': array_hash(catalogue),
            'target_context_sha256': array_hash(target_context),
            'combined_exclusion_pixels': int(arrays['control_exclusion'].sum()),
            'combined_exclusion_sha256': array_hash(arrays['control_exclusion']),
            'selection_dates': list(DATES[:2]), 'followup_date': DATES[2],
            'selection': {'sizes': [16, 12, 8], 'order': 'largest then ascending row and column',
                          'maximum_controls': 3, 'context_radius_pixels': 2, 'catalogue_buffer_metres': 20,
                          'require_mutually_disjoint_contexts': True, 'used_ndvi_values_for_ranking': False,
                          'used_visual_class_or_model_output': False, 'replace_failed_followup': False}}
