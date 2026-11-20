"""Offline checks for the fixed three-date metadata preflight."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, call, patch
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Solutions'))
import preflight_remaining_validation_metadata as preflight


class RemainingMetadataPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        evidence = ROOT / 'Docs/Reviews/ValidationEarlierCatalogueV12/inventory'
        cls.inventory = json.loads((evidence / 'inventory.json').read_bytes())
        cls.raw_page = json.loads((evidence / 'raw/page-01.json').read_bytes())

    def test_frozen_catalogue_selects_only_the_three_remaining_products(self):
        roster, _ = preflight.select_roster(self.inventory, self.raw_page)
        expected = (
            ('11', 'S2A_36UXB_20180711_0_L1C',
             'S2A_MSIL1C_20180711T083601_N0206_R064_T36UXB_20180711T122613',
             'aa2d938f1fabc22496e6f834a6b1a8bf6ef1a55314db8963fd9d963b65029ac6'),
            ('06', 'S2B_36UXB_20180706_0_L1C',
             'S2B_MSIL1C_20180706T083559_N0206_R064_T36UXB_20180706T123246',
             '240e23af9e7dad88c77e0457ab01e7c6d7cb66b13f65a8d90ae80906728c4af0'),
            ('01', 'S2A_36UXB_20180701_0_L1C',
             'S2A_MSIL1C_20180701T083601_N0206_R064_T36UXB_20180701T104347',
             '3215101ad83527fd9b743fcb676e67038a27ef8e894874a26a2f53e2e283733d'),
        )
        self.assertEqual(roster, [
            {'item_id': item, 'raw_item_sha256': checksum, 'scene_id': scene,
             'acquisition_date': f'2018-07-{day}', 'sequence': '0',
             'tile_prefix': f'tiles/36/U/XB/2018/7/{int(day)}/0/',
             'product_key': f'products/2018/7/{int(day)}/{scene}/metadata.xml'}
            for day, item, scene, checksum in expected
        ])

    def test_missing_and_duplicate_eligible_dates_are_rejected(self):
        items = self.inventory['items']
        target = next(row for row in items if row['item_id'] == 'S2B_36UXB_20180706_0_L1C')
        missing = [row for row in items if row is not target]
        variant = next(row for row in items if row['item_id'] == 'S2B_36UXB_20180706_1_L1C')
        duplicate = [dict(row, nominal_metadata_followup_eligible=True) if row is variant else row for row in items]
        for name, changed in (('missing', missing), ('duplicate', duplicate)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                preflight.select_roster({**self.inventory, 'items': changed}, self.raw_page)

    def test_changed_raw_catalogue_item_is_rejected(self):
        raw_page = deepcopy(self.raw_page)
        target = next(row for row in raw_page['features'] if row['id'] == 'S2A_36UXB_20180701_0_L1C')
        target['properties']['s2:granule_id'] = 'changed granule'
        with self.assertRaises(ValueError):
            preflight.select_roster(self.inventory, raw_page)

    def test_first_source_timeout_retains_failure_and_checks_both_later_roster_entries(self):
        roster, raw_items = preflight.select_roster(self.inventory, self.raw_page)
        base_url = 'https://sentinel-s2-l1c.s3.amazonaws.com/'
        provider = {'listing_max_keys': 300}
        product_pin = {'size': 12, 'etag': '"' + '1' * 32 + '"'}
        assets = [{'size': 10} for _ in range(58)]
        listings = []
        expected_calls = []
        responses = []
        for index, source in enumerate(roster, 1):
            prefix = source['tile_prefix']
            product_url = base_url + source['product_key']
            tile_key = prefix + 'metadata.xml'
            tile_pin = {'key': tile_key, **product_pin}
            listings.append({tile_key: tile_pin})
            artifact = f'source-{index:02d}/'
            query = urlencode({'list-type': '2', 'prefix': prefix, 'max-keys': 300})
            listing_url = base_url + '?' + query
            expected_calls.extend([
                call('GET', listing_url, artifact + 'listing.xml'),
                call('HEAD', product_url, None),
                call('GET', product_url, artifact + 'product.xml', product_pin),
                call('GET', base_url + tile_key, artifact + 'tile.xml', tile_pin),
            ])
            tile_response = TimeoutError('tile XML timed out') if index == 1 else b'tile XML'
            responses.extend([b'listing XML', product_pin, b'product XML', tile_response])
        network = Mock()
        network.call.side_effect = responses
        with patch.object(preflight, 'parse_listing', side_effect=listings), \
             patch.object(preflight, 'parse_metadata', return_value={'issues': []}), \
             patch.object(preflight, 'asset_manifest', return_value=(assets, [])):
            results = preflight.check_sources(roster, raw_items, {}, network, provider)
        self.assertEqual(network.call.call_args_list, expected_calls)
        self.assertEqual([row['scene_id'] for row in results], [row['scene_id'] for row in roster])
        self.assertEqual([row['status'] for row in results], [
            'blocked_metadata_preflight',
            'remote_metadata_and_asset_listing_passed',
            'remote_metadata_and_asset_listing_passed',
        ])
        self.assertIn('TimeoutError', results[0]['issues'][0])
        self.assertEqual([len(row['assets']) for row in results], [0, 58, 58])
        self.assertTrue(all(len(row['catalogue_asset_href_conflicts']) == 17 for row in results))


if __name__ == '__main__':
    unittest.main()
