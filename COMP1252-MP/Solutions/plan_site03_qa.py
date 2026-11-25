"""Derive the site-03 QA geometry from the frozen annotations and scene identities."""

from datetime import date
import hashlib
from pathlib import Path

from screen_repaired_validation import contains, object_hash, read, require


SITE_ID = 'site-03'
CONTROL_DATE, PRE_DATE, POST_DATE = '2018-05-10', '2018-05-30', '2018-06-04'
SCENE_ROLES = {
    CONTROL_DATE: 'earlier_control_observation',
    PRE_DATE: 'before_observation',
    POST_DATE: 'after_observation',
}
SOURCE_ROLES = {'v28_plan', 'v29_protocol', 'catalogue', 'splits', 'training_manifest',
                'original_source_exclusions', 'pairs'}


def _checked_site(paths, selected_scenes):
    from audit_annotations import PROJECTED_CRS
    from plan_unseen_validation_sites import QA_PAD

    v28, v29 = read(paths['v28_plan']), read(paths['v29_protocol'])
    sites = [row for row in v28['sites'] if row['site_id'] == SITE_ID]
    require(v28['schema_version'] == 28 and v29['schema_version'] == 29 and len(sites) == 1,
            'The exact V28 and V29 site plans are required')
    site = sites[0]
    site_keys = ('site_id', 'event_id', 'group_id', 'grid', 'qa_context_bounds', 'control_opportunity')
    require(v29['site'] == {key: site[key] for key in site_keys}
            and site['split'] == 'validation' and site['tile'] == '36UXA'
            and site['annotation_date'] == '2018-06-01' and site['blockers'] == []
            and site['annotation_flags'] == []
            and all(site['group_id'] not in groups for groups in v28['excluded_groups'].values()),
            'V28 and V29 site lineage or validation eligibility differs')
    grid = site['grid']
    padded = [value + offset for value, offset in zip(grid['bounds'], (-QA_PAD, -QA_PAD, QA_PAD, QA_PAD))]
    require(grid['crs'] == PROJECTED_CRS and site['qa_context_bounds'] == padded,
            'The full frozen 1200 m QA context is required')

    v29_roster = {row['acquisition_date']: row for row in v29['roster']}
    require(len(v29['roster']) == len(v29_roster) == len(SCENE_ROLES)
            and v29_roster.keys() == selected_scenes.keys() == SCENE_ROLES.keys(),
            'The three frozen V29 dates are required')
    for day, role in SCENE_ROLES.items():
        expected = v29_roster[day]
        actual = selected_scenes[day]
        safe = Path(actual['path'])
        native = actual.get('grid_10m') or {}
        require(expected['role'] == role and expected['tile'] == actual.get('tile') == site['tile']
                and actual.get('acquisition_date') == day and actual.get('scene_id') == expected['scene_id']
                and not safe.is_absolute() and '..' not in safe.parts and safe.name == expected['scene_id']+'.SAFE'
                and actual.get('crs') == PROJECTED_CRS and actual.get('processing_level') == 'L1C'
                and actual.get('processing_baseline') == expected['processing_baseline'] == '02.06'
                and actual.get('nonzero_offset') is False and actual.get('quantification_value') == 10000.0
                and all(actual.get(key) is True for key in ('metadata_usable', 'cloud_model_usable',
                                                           'shadow_geometry_usable')) and not actual.get('issues')
                and native.get('bounds') == site['native_grid_bounds']
                and (native.get('width'), native.get('height'), native.get('xdim'), native.get('ydim'))
                == (10980, 10980, 10, -10) and contains(native['bounds'], padded),
                'Scene does not match the frozen V29 QA source: '+day)
    return site


def _catalogue_geometry(site, catalogue, source_exclusions_path, before_scene):
    from pyproj import Transformer
    from shapely.geometry import box, shape
    from shapely.ops import transform
    from shapely.strtree import STRtree
    from audit_annotations import PROJECTED_CRS, crop_grid
    from plan_unseen_validation_sites import propose_control
    from prepare_validation_review import records

    project = Transformer.from_crs('EPSG:4326', PROJECTED_CRS, always_xy=True).transform
    features = sorted(catalogue['features'], key=lambda feature: feature['id'])
    positions = {feature['id']: index for index, feature in enumerate(features)}
    require(len(features) == len(positions) == 3911 and site['event_id'] in positions,
            'The complete frozen event catalogue is required')
    projected = [transform(project, shape(feature['geometry'])) for feature in features]
    anchor_index = positions[site['event_id']]
    anchor, properties = projected[anchor_index], features[anchor_index]['properties']
    grid = site['grid']
    core = box(*grid['bounds'])
    rebuilt_grid, crop_blockers = crop_grid({'projected_geometry': anchor}, before_scene, 64)
    require(rebuilt_grid == grid and not crop_blockers and list(anchor.bounds) == site['event_bounds']
            and properties['event_id'] == site['event_id'] and properties['tile'] == site['tile']
            and properties['group_id'] == site['group_id'] and properties['split'] == site['split']
            and properties['observation_date'] == site['annotation_date']
            and properties['temporal_source_ids'] == site['temporal_source_ids'] and site['temporal_source_ids']
            and properties['flags'] == site['annotation_flags']
            and properties['group_footprint_bounds'] == site['frozen_group_footprint_bounds']
            and box(*properties['group_footprint_bounds']).covers(core),
            'Selected catalogue anchor, crop or frozen grouping envelope differs')

    exclusions = [transform(project, shape(row['geometry'])) for row in records(source_exclusions_path)
                  if row.get('geometry') is not None]
    require(exclusions and not any(core.intersects(geometry) for geometry in exclusions),
            'Frozen unresolved-source footprints are missing or overlap the selected crop')
    hits = sorted(STRtree(projected).query(core, predicate='intersects'))
    events = {features[index]['id']: {'geometry': projected[index], 'properties': features[index]['properties']}
              for index in hits}
    start, end = site['comparison_start_inclusive'], site['comparison_end_inclusive']
    require(list(events) == site['context_event_ids'] and start <= CONTROL_DATE < PRE_DATE
            < site['annotation_date'] <= POST_DATE <= end, 'The frozen crop context or dates differ')
    for event_id, event in events.items():
        row = event['properties']
        require(row['event_id'] == event_id and row['group_id'] == site['group_id'] and row['split'] == 'validation'
                and 'conflicting_observation_dates' not in row['flags']
                and (row['temporal_source_ids'] or not start < row['observation_date'] <= end),
                'Crop context contains another group or unverified temporal label: '+event_id)

    control = propose_control(grid, anchor, STRtree(projected + exclusions))
    require(control is not None and control == site['control_opportunity']
            and (control['row'], control['col'], control['width'], control['height']) == (2, 2, 16, 16)
            and (control['pixels'], control['context_pixels'], control['context_radius_pixels']) == (256, 400, 2)
            and control['control_class_status'] == 'unreviewed' and control['pixel_qa_status'] == 'unassessed'
            and control['independent_site'] is False and control['evaluation_approved'] is False,
            'The frozen geometry-only control or its full context differs')
    return events, control, len(exclusions)


def site03_masks(pair, events, control):
    """Return rasterized catalogue reference masks and geometric context regions."""
    import numpy as np
    from scipy.ndimage import binary_erosion
    from assess_pixel_quality import core_grid, label_target
    from audit_validation_context import context_mask
    from validate_sample import burn, temporal_masks

    grid = core_grid(pair['grid'])
    pre_date, post_date = map(date.fromisoformat, (pair['pre_date'], pair['post_date']))
    pre_ids, post_ids = (pair['label_event_ids_by_'+side+'_date'] for side in ('pre', 'post'))
    dated = [(date.fromisoformat(events[key]['properties']['observation_date']), events[key]['geometry'])
             for key in post_ids]
    before, after, reference = temporal_masks(dated, pre_date, post_date, grid)
    pre_burn = burn([events[key]['geometry'] for key in pre_ids], grid)
    post_burn = burn([events[key]['geometry'] for key in post_ids], grid)
    require(np.array_equal(before, pre_burn) and np.array_equal(after, post_burn)
            and np.array_equal(reference, (post_burn > 0) & (pre_burn == 0))
            and np.array_equal(reference, label_target(pair, events)), 'Cumulative date masks disagree')

    anchor = burn([events[pair['anchor_event_ids'][0]]['geometry']], grid).astype(bool)
    target = anchor & reference.astype(bool)
    interior = binary_erosion(target, structure=np.ones((3, 3), dtype=bool))
    radius = control['context_radius_pixels']
    require(int(interior.sum()) >= 4, 'The eroded target has fewer than four pixels')
    anchor_full_context = context_mask(anchor, radius)
    control_roi = np.zeros(anchor.shape, dtype=bool)
    row, col = control['row'], control['col']
    control_roi[row:row+control['height'], col:col+control['width']] = True
    control_context = context_mask(control_roi, radius)
    require(int(control_roi.sum()) == control['pixels'] and int(control_context.sum()) == control['context_pixels']
            and not (anchor_full_context & control_context).any(), 'The positive and control contexts differ')
    return {'cumulative_pre': before, 'cumulative_post': after, 'reference_new_loss': reference,
            'anchor': anchor, 'anchor_target': target, 'anchor_interior': interior,
            'control': control_roi, 'control_context': control_context,
            'anchor_context': context_mask(target, radius), 'interior_context': context_mask(interior, radius)}


def prepare_geometry(protocol, paths, selected_scenes):
    """Build the exact derived pair, proof of geometry, and unreviewed control."""
    from audit_annotations import PROJECTED_CRS
    from prepare_validation_review import records, validate_validation_pair

    require(protocol['schema_version'] == 30 and protocol['fixed_site_id'] == SITE_ID and SOURCE_ROLES <= paths.keys(),
            'V30 site geometry inputs differ')
    site = _checked_site(paths, selected_scenes)
    catalogue, splits = read(paths['catalogue']), read(paths['splits'])
    require(splits['crs'] == PROJECTED_CRS and splits['patch_size_pixels'] == 64 and splits['resolution_m'] == 10,
            'Frozen split grid differs')
    events, control, exclusion_count = _catalogue_geometry(site, catalogue, paths['original_source_exclusions'],
                                                           selected_scenes[PRE_DATE])
    dated = {event_id: event['properties']['observation_date'] for event_id, event in events.items()
             if event['properties']['temporal_source_ids']}
    pre_ids = [event_id for event_id, day in dated.items() if day <= PRE_DATE]
    post_ids = [event_id for event_id, day in dated.items() if day <= POST_DATE]
    require(pre_ids == site['prior_temporal_context_event_ids'] and set(post_ids)-set(pre_ids) == {site['event_id']},
            'Actual source dates no longer bracket the frozen anchor and its prior context')
    lineage = {'site_id': SITE_ID, 'group_id': site['group_id'], 'anchor_event_ids': [site['event_id']],
               'grid': site['grid'], 'context_event_ids': list(events),
               'label_event_ids_by_pre_date': pre_ids, 'label_event_ids_by_post_date': post_ids}
    for side, day in (('control', CONTROL_DATE), ('pre', PRE_DATE), ('post', POST_DATE)):
        scene = selected_scenes[day]
        lineage.update({side+'_date': day, side+'_scene_id': scene['scene_id'], side+'_path': scene['path']})
    pair = {**lineage, 'pair_id': 'derived-site03-'+object_hash(lineage)[:20], 'split': 'validation',
            'tile': site['tile'], 'annotation_date': site['annotation_date'],
            'interval_days': (date.fromisoformat(POST_DATE)-date.fromisoformat(PRE_DATE)).days,
            'qa_context_bounds': site['qa_context_bounds'], 'temporal_source_ids': site['temporal_source_ids'],
            'status': 'candidate_for_pixel_review', 'blocking_reasons': [], 'conflicting_label_event_ids': [],
            'training_approved': False, 'evaluation_approved': False}
    require(all(row['group_id'] != pair['group_id'] and row['pair_id'] != pair['pair_id']
                for row in records(paths['pairs'])), 'Site group or pair is represented in the original candidates')
    validate_validation_pair(pair, site['event_id'], catalogue, splits, list(records(paths['training_manifest'])))

    masks = site03_masks(pair, events, control)
    pixels = {name: int(mask.sum()) for name, mask in masks.items()}
    hashes = {name: hashlib.sha256(mask.tobytes()).hexdigest() for name, mask in masks.items()}
    require(pixels['anchor'] == site['anchor_pixels'] and pixels['anchor_target'] == site['positive_target_pixels']
            and pixels['anchor_interior'] == site['positive_interior_pixels']
            and hashes['anchor_interior'] == site['positive_interior_sha256'], 'Frozen V28 positive support differs')
    geometry = {'pair_id': pair['pair_id'], 'grid': site['grid'], 'qa_context_bounds': site['qa_context_bounds'],
                'context_event_ids': list(events), 'temporal_events_by_annotation_image_date': dated,
                'pre_event_ids': pre_ids, 'post_event_ids': post_ids, 'pixels': pixels, 'mask_sha256': hashes,
                'unresolved_source_footprints_checked': exclusion_count,
                'unresolved_source_footprints_intersecting_crop': 0,
                'reference_rule': 'Cumulative temporal footprints on or before each image date, with pre removed from post',
                'control_reference_rule': 'Geometry-only ROI outside buffered catalogue and unresolved source footprints',
                'control_class_status': 'unreviewed', 'class_labels_approved': 0}
    return {'pair': pair, 'geometry': geometry, 'control': control, 'events': events}
