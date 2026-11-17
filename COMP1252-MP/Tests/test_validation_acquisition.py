"""Acquisition feasibility tests: manifest construction, SAFE assembly and reader verification."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Solutions'))
from adapt_validation_scenes import (
    adapt_baseline_offset,
    assemble_safe_skeleton,
    build_scene_manifest,
    verify_safe_metadata,
)


class AcquisitionFeasibilityTests(unittest.TestCase):
    def test_build_scene_manifest_contains_required_assets(self):
        scene_id = "S2B_MSIL1C_20180726T084009_N0206_R064_T36UXB_20180726T104227"
        m = build_scene_manifest(scene_id, 2018, 7, 26, '0')
        self.assertEqual(len(m['metadata_assets']), 3)
        self.assertEqual(len(m['band_assets']), 7)
        band_names = [b['band'] for b in m['band_assets']]
        for req in ('B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12'):
            self.assertIn(req, band_names)

    def test_adapt_baseline_offset_detects_zero_and_nonzero_offsets(self):
        with tempfile.TemporaryDirectory() as d:
            # Baseline 02.06 XML without RADIO_ADD_OFFSET
            p_zero = Path(d) / 'zero_offset.xml'
            p_zero.write_text('<Root><PROCESSING_BASELINE>02.06</PROCESSING_BASELINE></Root>')
            res_zero = adapt_baseline_offset(p_zero)
            self.assertFalse(res_zero['has_nonzero_offset'])
            self.assertFalse(res_zero['adaptation_required'])

            # Baseline 05.00 XML with RADIO_ADD_OFFSET = -1000
            p_offset = Path(d) / 'nonzero_offset.xml'
            p_offset.write_text(
                '<Root><PROCESSING_BASELINE>05.00</PROCESSING_BASELINE>'
                '<RADIO_ADD_OFFSET band_id="B02">-1000</RADIO_ADD_OFFSET></Root>'
            )
            res_offset = adapt_baseline_offset(p_offset)
            self.assertTrue(res_offset['has_nonzero_offset'])
            self.assertTrue(res_offset['adaptation_required'])
            self.assertEqual(res_offset['offsets']['B02'], -1000.0)

    def test_safe_assembly_and_reader_verification_on_mock_scene(self):
        tile_xml = (
            '<Root><HORIZONTAL_CS_CODE>EPSG:32636</HORIZONTAL_CS_CODE>'
            '<Size resolution="10"><NCOLS>10980</NCOLS><NROWS>10980</NROWS></Size>'
            '<Geoposition resolution="10"><ULX>600000</ULX><ULY>5700000</ULY>'
            '<XDIM>10</XDIM><YDIM>-10</YDIM></Geoposition></Root>'
        )
        prod_xml = (
            '<Root><PROCESSING_BASELINE>02.06</PROCESSING_BASELINE>'
            '<QUANTIFICATION_VALUE>10000</QUANTIFICATION_VALUE></Root>'
        )
        cloud_qa = '<Mask></Mask>'

        scene_id = "S2B_MSIL1C_20180726T084009_N0206_R064_T36UXB_20180726T104227"
        m = build_scene_manifest(scene_id, 2018, 7, 26, '0')

        with tempfile.TemporaryDirectory() as d:
            safe_dir = assemble_safe_skeleton(m, d, {
                'product_metadata': prod_xml,
                'tile_metadata': tile_xml,
                'cloud_qa': cloud_qa
            })
            self.assertTrue((safe_dir / 'MTD_MSIL1C.xml').exists())

            # Verify reader inspection
            v = verify_safe_metadata(safe_dir, Path(d))
            self.assertEqual(v['scene_id'], scene_id)
            self.assertEqual(v['processing_baseline'], '02.06')
            self.assertEqual(v['quantification_value'], 10000.0)
            self.assertFalse(v['nonzero_offset'])
            self.assertFalse(v['cloud_qa_missing'])
            self.assertTrue(v['cloud_qa_well_formed'])
            self.assertEqual(v['crs'], 'EPSG:32636')


if __name__ == '__main__':
    unittest.main()
