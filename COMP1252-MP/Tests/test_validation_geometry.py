"""Geometry size alone cannot guarantee a conservative reviewed pixel interior."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from audit_validation_geometry import geometry_support


class ValidationGeometryTests(unittest.TestCase):
    def test_equal_area_thin_annotation_loses_interior_while_square_retains_support(self):
        from shapely.geometry import box

        grid = {'crs': 'EPSG:32636', 'width': 64, 'height': 64,
                'bounds': [500000., 5599360., 500640., 5600000.],
                'transform': [10., 0., 500000., 0., -10., 5600000.]}
        pair = {'pair_id': 'pair-validation', 'group_id': 'group-validation', 'split': 'validation',
                'pre_scene_id': 'before', 'post_scene_id': 'after', 'pre_date': '2019-06-01',
                'post_date': '2019-06-11', 'anchor_event_ids': ['event-validation'],
                'label_event_ids_by_post_date': ['event-validation'], 'grid': grid}
        properties = {'observation_date': '2019-06-11', 'temporal_source_ids': ['source']}
        # Both geometries have 1600m2 area, but one is only two pixels wide.
        thin = {'event-validation': {'properties': properties,
                                    'geometry': box(500100., 5599800., 500180., 5599820.)}}
        square = {'event-validation': {'properties': properties,
                                      'geometry': box(500100., 5599800., 500140., 5599840.)}}
        narrow = geometry_support('event-validation', pair, thin)
        broad = geometry_support('event-validation', pair, square)
        self.assertEqual(narrow['target_pixels'], broad['target_pixels'])
        self.assertEqual(narrow['target_pixels'], 16)
        self.assertEqual(narrow['positive_interior_pixels'], 0)
        self.assertEqual(broad['positive_interior_pixels'], 4)
        self.assertFalse(narrow['passes_geometry_support_only'])
        self.assertTrue(broad['passes_geometry_support_only'])

    def test_other_temporal_annotations_cannot_supply_anchor_interior(self):
        from shapely.geometry import box

        grid = {'crs': 'EPSG:32636', 'width': 64, 'height': 64,
                'bounds': [500000., 5599360., 500640., 5600000.],
                'transform': [10., 0., 500000., 0., -10., 5600000.]}
        pair = {'pair_id': 'pair-validation', 'group_id': 'group-validation', 'split': 'validation',
                'pre_scene_id': 'before', 'post_scene_id': 'after', 'pre_date': '2019-06-01',
                'post_date': '2019-06-11', 'anchor_event_ids': ['anchor'],
                'label_event_ids_by_post_date': ['anchor', 'other'], 'grid': grid}
        properties = {'observation_date': '2019-06-11', 'temporal_source_ids': ['source']}
        events = {'anchor': {'properties': properties, 'geometry': box(500100., 5599800., 500180., 5599820.)},
                  'other': {'properties': properties, 'geometry': box(500300., 5599600., 500400., 5599700.)}}
        result = geometry_support('anchor', pair, events)
        self.assertGreater(result['temporal_union_pixels'], result['target_pixels'])
        self.assertEqual(result['target_pixels'], 16)
        self.assertEqual(result['positive_interior_pixels'], 0)


if __name__ == '__main__':
    unittest.main()
