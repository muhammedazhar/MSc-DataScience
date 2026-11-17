"""Metadata footprint tests: containment, outside gaps and bounded discovery."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Solutions'))
from audit_validation_coverage import discover, parse_tile, relation


class CoverageMetadataTests(unittest.TestCase):
    def test_overlap_and_edge_contact_do_not_imply_full_crop_coverage(self):
        tile = [0., 0., 100., 100.]
        self.assertTrue(relation([0., 0., 100., 100.], tile)['fully_covered'])
        partial = relation([90., 90., 110., 110.], tile)
        self.assertFalse(partial['fully_covered']);self.assertEqual(partial['positive_area_overlap_m2'], 100.)
        touch = relation([0., 100., 10., 110.], tile)
        self.assertFalse(touch['fully_covered']);self.assertEqual(touch['positive_area_overlap_m2'], 0.)

    def test_exact_fixed_crop_north_gaps_are_not_small_edge_clips(self):
        tile = [600000., 5490240., 709800., 5600040.]
        for crop, gap in [([663640., 5605000., 664280., 5605640.], 4960.),
                          ([663380., 5604740., 664020., 5605380.], 4700.)]:
            result = relation(crop, tile)
            self.assertFalse(result['fully_covered']);self.assertEqual(result['north_gap_m'], gap)
            self.assertEqual(result['positive_area_overlap_m2'], 0.)

    def test_invalid_bounds_are_not_silently_classified_uncovered(self):
        for bounds in ([1., 0., 0., 1.], [0., 0., float('nan'), 1.], [0., 0., 0., 1.]):
            with self.assertRaises(ValueError):relation(bounds, [0., 0., 10., 10.])

    def tile_xml(self):
        return '<Root><HORIZONTAL_CS_CODE>EPSG:32636</HORIZONTAL_CS_CODE><Size resolution="10"><NCOLS>10980</NCOLS><NROWS>10980</NROWS></Size><Geoposition resolution="10"><ULX>600000</ULX><ULY>5600040</ULY><XDIM>10</XDIM><YDIM>-10</YDIM></Geoposition></Root>'

    def test_tile_xml_derives_signed_north_up_bounds_from_origin_and_size(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'tile.xml';path.write_text(self.tile_xml())
            grid=parse_tile(path)
            self.assertEqual(grid['bounds'], [600000.,5490240.,709800.,5600040.])
            self.assertEqual(grid['width'],10980)

    def test_tile_xml_rejects_nonfinite_ambiguous_or_wrong_spacing(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'tile.xml'
            for text in [self.tile_xml().replace('5600040','NaN'),self.tile_xml().replace('<XDIM>10','<XDIM>20'),
                         self.tile_xml().replace('</Root>','<Size resolution="10"/></Root>')]:
                path.write_text(text)
                with self.assertRaises(ValueError):parse_tile(path)

    def test_discovery_retains_direct_and_nested_safes_and_refuses_incomplete_product(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'direct.SAFE').mkdir();(root/'S2product').mkdir();(root/'S2product'/'nested.SAFE').mkdir()
            self.assertEqual(len(discover(root)),2)
            (root/'S2missing').mkdir()
            with self.assertRaises(ValueError):discover(root)


if __name__ == '__main__':
    unittest.main()
