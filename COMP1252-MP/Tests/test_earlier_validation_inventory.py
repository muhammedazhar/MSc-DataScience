"""Offline catalogue regressions; simulated metadata only, no imagery/network."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from inventory_earlier_validation_sources import (ENDPOINT, KNOWN_CLOUDY, ROOT, canonical,
    collect, coverage, initial_request, next_request, normalize, sha, summarize, trusted)


class EarlierValidationInventoryTests(unittest.TestCase):
    def setUp(self):
        self.q = json.loads((ROOT/'Docs/Reviews/ValidationEarlierCatalogueV12/protocol.json').read_text())['query']
        self.requirements = json.loads((ROOT/'Docs/Reviews/ValidationCoverageV6/source_requirements.json').read_text())
        self.request = initial_request(self.requirements, self.q)

    def geographic(self, geometry):
        from pyproj import Transformer
        from shapely.geometry import mapping
        from shapely.ops import transform
        return mapping(transform(Transformer.from_crs('EPSG:32636', 'EPSG:4326', always_xy=True).transform, geometry))

    def feature(self, day='21', baseline='0206', identifier=None):
        from shapely.geometry import box
        uri = f'S2B_MSIL1C_201807{day}T084009_N{baseline}_R064_T36UXB_201807{day}T104227.SAFE'
        return {'type': 'Feature', 'id': identifier or uri, 'collection': 'sentinel-2-l1c',
                'properties': {'s2:product_uri': uri, 'datetime': f'2018-07-{day}T08:45:01Z',
                               'grid:code': 'MGRS-36UXB', 's2:processing_baseline': baseline[:2]+'.'+baseline[2:],
                               'sat:relative_orbit': 64, 'platform': 'sentinel-2b', 'eo:cloud_cover': 99},
                'geometry': self.geographic(box(660000, 5600000, 668000, 5609000)), 'assets': {}}

    def row(self, feature):
        return normalize({'feature': feature, 'sha256': sha(canonical(feature)), 'pages': [1]}, self.q, self.requirements)

    def page(self, features, links=None, matched=None):
        page = {'type': 'FeatureCollection', 'features': features, 'links': links or [], 'numberReturned': len(features)}
        if matched is not None:
            page['numberMatched'] = matched
        return canonical(page)

    def run_pages(self, pages, query=None):
        iterator = iter(pages)
        with tempfile.TemporaryDirectory() as directory:
            return collect(self.request, query or self.q, Path(directory), lambda *_: next(iterator))

    def test_mixed_baselines_keep_variants_order_independent_and_offsets_unknown(self):
        rows = [self.row(self.feature(baseline=b)) for b in ('0500', '0206')]
        self.assertEqual(summarize(rows), summarize(rows[::-1]))
        summary = summarize(rows)
        self.assertEqual(summary['distinct_product_uris'], 2)
        self.assertEqual(summary['baselines'], ['N0206', 'N0500'])
        self.assertEqual(summary['nominal_metadata_followup_dates'], ['2018-07-21'])
        self.assertFalse(rows[0]['nominal_metadata_followup_eligible'])
        self.assertTrue(rows[1]['nominal_metadata_followup_eligible'])  # even with 99% catalogue cloud
        for row in rows:
            self.assertIsNone(row['supplied_offset_properties'])
            self.assertIsNone(row['supplied_asset_band_offsets'])
            self.assertFalse(row['source_approved'])
            self.assertEqual(row['reader_compatibility'], 'unverified')
        supplied = self.feature(); supplied['assets']['blue'] = {'raster:bands': [{'offset': -1000}]}
        self.assertEqual(self.row(supplied)['supplied_asset_band_offsets'], {'blue': [{'band_index': 0, 'offset': -1000}]})

    def test_date_boundaries_and_known_cloudy_identity(self):
        self.assertTrue(self.row(self.feature(day='01'))['nominal_metadata_followup_eligible'])
        cloudy = self.feature(day='26'); cloudy['properties']['s2:product_uri'] = KNOWN_CLOUDY+'.SAFE'
        row = self.row(cloudy)
        self.assertTrue(row['known_cloudy_acquisition_date'])
        self.assertEqual(row['own_qa_status'], 'excluded_by_V11_automatic_quality_screen')
        self.assertFalse(row['nominal_metadata_followup_eligible'])
        variant = self.row(self.feature(day='26', baseline='0500'))
        self.assertTrue(variant['known_cloudy_acquisition_date'])
        self.assertEqual(variant['own_qa_status'], 'unexamined')
        self.assertIn('outside_forward_interval', self.row(self.feature(day='27'))['followup_blockers'])
        self.assertIn('item_datetime_outside_query', self.row(self.feature(day='31'))['issues'])

    def test_missing_and_conflicting_identity_metadata_retained_with_blockers(self):
        for key, value, issue in [('grid:code', 'MGRS-36UXA', 'provider_tile_conflict'),
                                  ('s2:processing_baseline', '05.00', 'provider_baseline_conflict'),
                                  ('sat:relative_orbit', 65, 'provider_orbit_conflict'),
                                  ('datetime', '2018-07-22T00:00:00Z', 'product_and_item_date_conflict'),
                                  ('s2:product_uri', None, 'missing_or_invalid_product_uri')]:
            with self.subTest(key=key):
                item = self.feature(); item['properties'][key] = value
                row = self.row(item)
                self.assertIn(issue, row['issues'])
                self.assertFalse(row['nominal_metadata_followup_eligible'])
        item = self.feature(); del item['properties']['s2:processing_baseline']
        self.assertIn('missing_provider_baseline', self.row(item)['issues'])
        item = self.feature(); item['properties']['s2:product_uri'] = ['malformed']
        self.assertEqual(summarize([self.row(item)])['items_with_issues'], 1)
        item = self.feature(); del item['properties']['grid:code']
        item['properties'].update({'mgrs:utm_zone': 36, 'mgrs:latitude_band': 'U', 'mgrs:grid_square': 'XB'})
        self.assertTrue(self.row(item)['nominal_metadata_followup_eligible'])

    def test_complete_projected_coverage_respects_holes_and_multipolygons(self):
        from shapely.geometry import MultiPolygon, Polygon, box
        outer = box(660000, 5600000, 668000, 5609000)
        hole = box(662300, 5604000, 662500, 5604200)  # context, outside core
        geometry = self.geographic(Polygon(outer.exterior.coords, [hole.exterior.coords]))
        covered, issue = coverage(geometry, self.requirements)
        self.assertIsNone(issue)
        self.assertTrue(covered['core']); self.assertFalse(covered['qa_context'])
        multi = self.geographic(MultiPolygon([outer, box(670000, 5600000, 671000, 5601000)]))
        self.assertEqual(coverage(multi, self.requirements), ({'core': True, 'qa_context': True}, None))
        invalid = {'type': 'Polygon', 'coordinates': [[[34,50],[35,51],[34,51],[35,50],[34,50]]]}
        self.assertEqual(coverage(invalid, self.requirements)[1], 'invalid_footprint')
        self.assertEqual(coverage(None, self.requirements), ({'core': None, 'qa_context': None}, 'missing_footprint'))

    def test_pagination_merge_preserves_search_and_get_keeps_query(self):
        next_link = {'rel': 'next', 'href': ENDPOINT, 'method': 'POST', 'merge': True, 'body': {'next': 'token'}}
        request = next_request({'links': [next_link]}, self.request)
        self.assertEqual(request['body'], {**self.request['body'], 'next': 'token'})
        request = next_request({'links': [{'rel': 'next', 'href': '?token=x'}]}, self.request)
        self.assertEqual(request, {'url': ENDPOINT+'?token=x', 'method': 'GET', 'body': None})
        for url in ('http://earth-search.aws.element84.com/v1/search', 'https://evil.example/v1/search',
                    'https://earth-search.aws.element84.com/v1/collections', 'https://user@earth-search.aws.element84.com/v1/search'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                trusted(url)

    def test_exact_duplicates_coalesce_with_pages_but_variants_stay_and_raw_bytes_survive(self):
        old = self.feature(); new = self.feature(baseline='0500')
        link = {'rel': 'next', 'href': ENDPOINT, 'method': 'POST', 'merge': True, 'body': {'next': 'two'}}
        pages = [self.page([old], [link], matched=2), self.page([old, new], matched=2)]
        iterator = iter(pages)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            items, receipts, total = collect(self.request, self.q, output, lambda *_: next(iterator))
            self.assertEqual((len(items), len(receipts), total), (2, 2, 2))
            self.assertEqual(sorted(len(i['pages']) for i in items), [1, 2])
            for i, raw in enumerate(pages, 1):
                self.assertEqual((output/f'raw/page-{i:02d}.json').read_bytes(), raw)
                self.assertEqual(receipts[i-1]['sha256'], sha(raw))

    def test_conflicting_items_or_truncated_matched_count_fail(self):
        old = self.feature(); changed = deepcopy(old); changed['properties']['eo:cloud_cover'] = 1
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            self.run_pages([self.page([old, changed])])
        with self.assertRaisesRegex(ValueError, 'matched count'):
            self.run_pages([self.page([old], matched=2)])

    def test_loops_page_and_byte_caps_or_malformed_links_fail(self):
        loop = {'rel': 'next', 'href': ENDPOINT, 'method': 'POST', 'merge': True, 'body': {}}
        with self.assertRaisesRegex(ValueError, 'loop'):
            self.run_pages([self.page([], [loop])])
        next_link = {**loop, 'body': {'next': 'two'}}
        with self.assertRaisesRegex(ValueError, 'page cap'):
            self.run_pages([self.page([], [next_link])], {**self.q, 'maximum_pages': 1})
        with self.assertRaisesRegex(ValueError, 'cap'):
            self.run_pages([b'x'*100], {**self.q, 'maximum_response_bytes': 99})
        with self.assertRaisesRegex(ValueError, 'links'):
            self.run_pages([canonical({'type': 'FeatureCollection', 'features': []})])


if __name__ == '__main__':
    unittest.main()
