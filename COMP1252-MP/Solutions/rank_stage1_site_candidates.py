"""Rank dated reference leads and compatible Sentinel-2 metadata options."""

from datetime import date
from itertools import product

from inventory_earlier_validation_sources import require
from inventory_unseen_validation_sites import cloud_score, options


def _days(from_day, to_day):
    return (date.fromisoformat(to_day) - date.fromisoformat(from_day)).days


def _reference_score(site, before, after):
    before_cloud, after_cloud = (cloud_score(scene['scene_cloud_percent']) for scene in (before, after))
    return [
        max(before_cloud, after_cloud),
        before_cloud,
        sum(scene['collection_tier'] == 'T2' for scene in (before, after)),
        _days(before['capture_date_utc'], after['capture_date_utc']),
        -site['positive_interior_pixels'],
        site['site_id'], site['group_id'],
        before['scene_id'], before['object_id'], after['scene_id'], after['object_id'],
    ]


def rank_reference_candidates(roster, inventory):
    """Return each complete crop's best dated Landsat pair in metadata rank order."""
    sites = {site['site_id']: site for site in roster['sites']}
    queries = {query['site_id']: query for query in inventory['queries']}
    require(roster['sites'] == inventory['sites'] and sites.keys() == queries.keys()
            and len(sites) == len(roster['sites']) == len(inventory['queries']),
            'Landsat queries do not match the complete Stage 1 roster')
    candidates = []
    for site_id, site in sites.items():
        query = queries[site_id]
        if query['status'] != 'complete':
            continue
        before_options = query['before_options']
        after_options = query['after_options']
        require(all(scene['capture_date_utc'] < site['annotation_date'] for scene in before_options)
                and all(scene['capture_date_utc'] >= site['annotation_date'] for scene in after_options),
                'Reference dates are on the wrong side of the annotation date: '+site_id)
        if not before_options or not after_options:
            continue
        before, after = min(product(before_options, after_options), key=lambda pair: _reference_score(site, *pair))
        candidates.append({
            'site_id': site_id, 'group_id': site['group_id'], 'before': before, 'after': after,
            'score': _reference_score(site, before, after),
        })
    return sorted(candidates, key=lambda candidate: candidate['score'])


def _date_relation(delta):
    if delta == 0:
        return 'same_day'
    return 'landsat_earlier' if delta > 0 else 'landsat_later'


def rank_sentinel_options(site, sentinel_site, reference_pair):
    """Rank the unchanged V28-compatible pairs by cloud and reference timing."""
    require(site['site_id'] == sentinel_site['site_id'] == reference_pair['site_id']
            and site['group_id'] == reference_pair['group_id'], 'Sentinel and reference sites differ')
    if not sentinel_site['pagination_complete']:
        return []
    compatible = options(site, sentinel_site['items'])
    require(compatible == sentinel_site['comparison_options'], 'V28-compatible Sentinel-2 options differ on replay')
    nominees = []
    for pair in compatible:
        before_gap = _days(reference_pair['before']['capture_date_utc'], pair['pre_date'])
        after_gap = _days(reference_pair['after']['capture_date_utc'], pair['post_date'])
        before_cloud, after_cloud = (cloud_score(value) for value in pair['catalogue_cloud_percent'])
        score = [
            max(before_cloud, after_cloud), int(before_gap > 0), abs(before_gap), abs(after_gap),
            before_cloud, pair['interval_days'], pair['pre_scene_id'], pair['post_scene_id'],
        ]
        nominees.append({
            **pair,
            'site_id': site['site_id'],
            'group_id': site['group_id'],
            'sentinel_minus_landsat_before_days': before_gap,
            'sentinel_minus_landsat_after_days': after_gap,
            'date_gap_notes': {'before': _date_relation(before_gap), 'after': _date_relation(after_gap),
                               'basis': 'UTC date comparison. Same-day endpoints may differ in acquisition time.'},
            'score': score,
        })
    return sorted(nominees, key=lambda pair: pair['score'])
