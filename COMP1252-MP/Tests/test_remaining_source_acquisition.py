"""Offline checks of the V16 manifests pinned for source acquisition."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Solutions'))
import acquire_remaining_validation_sources as acquisition


class RemainingSourceAcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        evidence = ROOT / 'Docs/Reviews/ValidationRemainingMetadataV16/preflight/preflight.json'
        cls.preflight = json.loads(evidence.read_bytes())
        cls.protocol = {
            'roster': [
                {
                    'scene_id': scene['scene_id'],
                    'acquisition_date': scene['acquisition_date'],
                    'asset_count': len(scene['assets']),
                    'payload_bytes': scene['measured_payload_bytes'],
                }
                for scene in cls.preflight['results']
            ],
            'stage_allowance_bytes': 3_454_603_931,
        }

    def test_pinned_manifests_match_all_three_dates_and_payload_totals(self):
        scenes = acquisition.validate_roster(self.protocol, self.preflight)
        self.assertEqual(tuple(scene['acquisition_date'] for scene in scenes),
                         ('2018-07-11', '2018-07-06', '2018-07-01'))
        self.assertEqual([len(scene['assets']) for scene in scenes], [58, 58, 58])
        totals = [sum(asset['size'] for asset in scene['assets']) for scene in scenes]
        self.assertEqual(totals, [562_911_967, 529_319_399, 572_015_183])
        self.assertEqual(sum(totals), 1_664_246_549)

    def test_missing_or_duplicate_scene_is_rejected(self):
        first, second, last = self.preflight['results']
        for name, scenes in (('missing', [first, last]), ('duplicate', [first, second, second])):
            changed = {**self.preflight, 'results': scenes}
            with self.subTest(name=name), self.assertRaises(ValueError):
                acquisition.validate_roster(self.protocol, changed)

    def test_duplicate_and_foreign_asset_mappings_are_rejected(self):
        first, second, _ = self.preflight['results']
        product, tile = first['assets'][:2]
        other_product, other_tile = second['assets'][:2]
        alternatives = {
            'duplicate target': {**tile, 'target_relpath': product['target_relpath']},
            'duplicate provider object': {**tile, 'key': product['key'], 'url': product['url']},
            'foreign scene target': {**tile, 'target_relpath': other_product['target_relpath']},
            'foreign scene provider object': {**tile, 'key': other_tile['key'], 'url': other_tile['url']},
            'foreign provider': {**tile, 'url': 'https://example.invalid/' + tile['key']},
        }
        for name, asset in alternatives.items():
            changed = deepcopy(self.preflight)
            changed['results'][0]['assets'][1] = asset
            with self.subTest(name=name), self.assertRaises(ValueError):
                acquisition.validate_roster(self.protocol, changed)

    def test_asset_sizes_must_sum_to_the_pinned_source_totals(self):
        changed = deepcopy(self.preflight)
        changed['results'][1]['assets'][0]['size'] += 1
        with self.assertRaises(ValueError):
            acquisition.validate_roster(self.protocol, changed)


if __name__ == '__main__':
    unittest.main()
