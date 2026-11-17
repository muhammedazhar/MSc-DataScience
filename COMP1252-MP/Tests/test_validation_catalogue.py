"""Catalogue search tests: geometry transformation, compatibility and bounded audit."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Solutions'))
from audit_validation_catalogue import (
    assess_reader_compatibility,
    bounds_to_polygon_points,
    parse_product_info,
    point_in_polygon,
    polygon_contains_bounds_wgs84,
    utm_to_latlon,
)


class CatalogueAuditTests(unittest.TestCase):
    def test_utm_to_latlon_precision(self):
        # EPSG:32636 coordinates for the fixed context bounds
        lon, lat = utm_to_latlon(663380.0, 5604740.0)
        self.assertAlmostEqual(lon, 35.307303, places=5)
        self.assertAlmostEqual(lat, 50.571703, places=5)

    def test_point_in_polygon_ray_casting(self):
        poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
        self.assertTrue(point_in_polygon(5.0, 5.0, poly))
        self.assertFalse(point_in_polygon(15.0, 5.0, poly))
        self.assertFalse(point_in_polygon(-1.0, 5.0, poly))

    def test_polygon_contains_bounds_wgs84(self):
        # A large polygon covering UTM zone 36 around northing 5600000
        big_poly = [(34.0, 50.0), (37.0, 50.0), (37.0, 52.0), (34.0, 52.0), (34.0, 50.0)]
        core_bounds = [663380.0, 5604740.0, 664280.0, 5605640.0]
        self.assertTrue(polygon_contains_bounds_wgs84(big_poly, core_bounds))

        # A polygon outside the crop
        outside_poly = [(30.0, 48.0), (32.0, 48.0), (32.0, 49.0), (30.0, 49.0), (30.0, 48.0)]
        self.assertFalse(polygon_contains_bounds_wgs84(outside_poly, core_bounds))

    def test_parse_product_info_standard_and_malformed(self):
        name = 'S2B_MSIL1C_20180726T084009_N0206_R064_T36UXB_20180726T104227'
        info = parse_product_info(name)
        self.assertIsNotNone(info)
        self.assertEqual(info['satellite'], 'S2B')
        self.assertEqual(info['processing_level'], 'L1C')
        self.assertEqual(info['acquisition_date'], '2018-07-26')
        self.assertEqual(info['baseline_code'], 'N0206')
        self.assertEqual(info['relative_orbit'], 'R064')
        self.assertEqual(info['tile_id'], '36UXB')

        # Malformed name returns None
        self.assertIsNone(parse_product_info('not_a_valid_s2_name'))

    def test_assess_reader_compatibility_n0206_vs_n0500(self):
        # N0206 in SAFE package is fully compatible with frozen reader
        c_n0206_safe = assess_reader_compatibility('N0206', 'SAFE')
        self.assertTrue(c_n0206_safe['compatible_baseline'])
        self.assertFalse(c_n0206_safe['radiometric_offset_present'])
        self.assertTrue(c_n0206_safe['overall_reader_compatible'])

        # N0500 has -1000 DN offset and triggers reader issue
        c_n0500 = assess_reader_compatibility('N0500', 'SAFE')
        self.assertFalse(c_n0500['compatible_baseline'])
        self.assertTrue(c_n0500['radiometric_offset_present'])
        self.assertEqual(c_n0500['radiometric_offset_value'], -1000)
        self.assertEqual(c_n0500['reader_issue'], 'nonzero_radiometric_offset_unsupported')
        self.assertFalse(c_n0500['overall_reader_compatible'])

        # N0206 in COG format requires reassembly
        c_n0206_cog = assess_reader_compatibility('N0206', 'COG/S3')
        self.assertTrue(c_n0206_cog['compatible_baseline'])
        self.assertFalse(c_n0206_cog['overall_reader_compatible'])


if __name__ == '__main__':
    unittest.main()
