"""Tiny SAFE fixtures for pixel-QA metadata; raster contents are never read."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SOLUTIONS = Path(__file__).resolve().parents[1] / "Solutions"
if str(SOLUTIONS) not in sys.path:
    sys.path.insert(0, str(SOLUTIONS))
SPEC = importlib.util.spec_from_file_location("pixelqa_scene_metadata", SOLUTIONS / "pixelqa_scene_metadata.py")
metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metadata)
PRODUCT = "S2A_MSIL1C_20180731T083601_N0206_R064_T36UXA_20180731T110233"


class PixelQAMetadataTests(unittest.TestCase):
    def fixture(self, directory, solar=None, offsets=""):
        safe = Path(directory) / PRODUCT / f"{PRODUCT}.SAFE"
        granule = safe / "GRANULE/L1C_T36UXA_A000001_20180731T083601"
        bands, qa = granule / "IMG_DATA", granule / "QI_DATA"
        bands.mkdir(parents=True)
        qa.mkdir()
        for band in metadata.QUALITY_BANDS:
            (bands / f"T36UXA_20180731T083601_{band}.jp2").write_bytes(b"do not read")
            for kind in metadata.QUALITY_TYPES:
                (qa / f"MSK_{kind}_{band}.gml").write_text(
                    '<eop:Mask xmlns:eop="http://www.opengis.net/eop/2.0"><eop:maskMembers/></eop:Mask>'
                )
        (qa / "MSK_CLOUDS_B00.gml").write_text(
            '<eop:Mask xmlns:eop="http://www.opengis.net/eop/2.0"><eop:maskMembers/></eop:Mask>'
        )
        (safe / "MTD_MSIL1C.xml").write_text(
            '<Product><PROCESSING_BASELINE>02.06</PROCESSING_BASELINE>'
            '<QUANTIFICATION_VALUE>10000</QUANTIFICATION_VALUE>'
            f'<Cloud_Coverage_Assessment>1.4</Cloud_Coverage_Assessment>{offsets}</Product>'
        )
        solar = solar if solar is not None else (
            '<Mean_Sun_Angle><ZENITH_ANGLE unit="deg">28.5</ZENITH_ANGLE>'
            '<AZIMUTH_ANGLE unit="deg">156.8</AZIMUTH_ANGLE></Mean_Sun_Angle>'
        )
        (granule / "MTD_TL.xml").write_text(
            '<Tile><HORIZONTAL_CS_CODE>EPSG:32636</HORIZONTAL_CS_CODE>'
            '<Size resolution="10"><NROWS>10</NROWS><NCOLS>12</NCOLS></Size>'
            '<Geoposition resolution="10"><ULX>600000</ULX><ULY>5600040</ULY>'
            '<XDIM>10</XDIM><YDIM>-10</YDIM></Geoposition>'
            f'<CLOUDY_PIXEL_PERCENTAGE>1.4</CLOUDY_PIXEL_PERCENTAGE>{solar}</Tile>'
        )
        return safe, granule

    def test_complete_metadata_preserves_cloud_band_order_angles_and_qa_paths_without_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            safe, _ = self.fixture(directory)
            original = Path.read_bytes

            def refuse_rasters(path):
                self.assertNotEqual(path.suffix.lower(), ".jp2")
                return original(path)

            with patch.object(Path, "read_bytes", refuse_rasters):
                record = metadata.qa_scene_metadata(safe, root_dir=directory)
        self.assertTrue(record["metadata_usable"], record["issues"])
        self.assertTrue(record["cloud_model_usable"])
        self.assertTrue(record["shadow_geometry_usable"])
        self.assertEqual(record["cloud_bands"], ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"])
        self.assertEqual(record["feature_bands"], list(metadata.FEATURE_BANDS))
        self.assertEqual(record["solar_azimuth_deg"], 156.8)
        self.assertEqual(record["solar_zenith_deg"], 28.5)
        self.assertEqual(record["quantification_value"], 10000)
        self.assertEqual(record["missing_cloud_bands"], [])
        self.assertEqual(set(record["quality_paths"]), {"cloud", "nodata", "satura", "defect", "tecqua"})
        self.assertEqual(len(record["quality_paths"]["nodata"]), 11)
        self.assertEqual(len(record["metadata_uid"]), 64)
        self.assertFalse(Path(record["cloud_band_paths"]["B01"]).is_absolute())
        json.dumps(record, allow_nan=False)

    def test_missing_cloud_band_and_quality_mask_are_flagged_without_implicit_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            safe, granule = self.fixture(directory)
            next((granule / "IMG_DATA").glob("*_B01.jp2")).unlink()
            (granule / "QI_DATA/MSK_NODATA_B03.gml").unlink()
            record = metadata.qa_scene_metadata(safe, root_dir=directory)
        self.assertFalse(record["metadata_usable"])
        self.assertFalse(record["cloud_model_usable"])
        self.assertIn("B01", record["missing_cloud_bands"])
        self.assertIn("B03", record["missing_quality_masks"]["NODATA"])
        self.assertIsNone(record["quality_band_paths"]["NODATA"]["B03"])
        self.assertIn("missing_cloud_band_B01", record["issues"])
        self.assertIn("missing_quality_mask_NODATA_B03", record["issues"])

    def test_viewing_angles_are_not_mistaken_for_missing_mean_sun_angles(self):
        solar = '<Mean_Viewing_Incidence_Angle><ZENITH_ANGLE>2</ZENITH_ANGLE><AZIMUTH_ANGLE>3</AZIMUTH_ANGLE></Mean_Viewing_Incidence_Angle>'
        with tempfile.TemporaryDirectory() as directory:
            safe, _ = self.fixture(directory, solar=solar)
            record = metadata.qa_scene_metadata(safe)
        self.assertFalse(record["metadata_usable"])
        self.assertFalse(record["shadow_geometry_usable"])
        self.assertIsNone(record["solar_azimuth_deg"])
        self.assertIsNone(record["solar_zenith_deg"])
        self.assertIn("missing_or_ambiguous_mean_sun_angle", record["issues"])

    def test_nonfinite_or_horizon_angles_are_rejected_for_shadow_geometry(self):
        for value in ("NaN", "90", "-1"):
            solar = f'<Mean_Sun_Angle><ZENITH_ANGLE>{value}</ZENITH_ANGLE><AZIMUTH_ANGLE>150</AZIMUTH_ANGLE></Mean_Sun_Angle>'
            with self.subTest(zenith=value), tempfile.TemporaryDirectory() as directory:
                safe, _ = self.fixture(directory, solar=solar)
                record = metadata.qa_scene_metadata(safe)
                self.assertFalse(record["metadata_usable"])
                self.assertIsNone(record["solar_zenith_deg"])
                self.assertIn("invalid_solar_zenith", record["issues"])

    def test_metadata_fingerprint_tracks_content_and_not_root_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            safe, granule = self.fixture(directory)
            first = metadata.qa_scene_metadata(safe, root_dir=directory)
            same = metadata.qa_scene_metadata(safe.parent, root_dir=safe.parent)
            self.assertEqual(first["metadata_uid"], same["metadata_uid"])
            path = granule / "MTD_TL.xml"
            path.write_text(path.read_text().replace(">28.5<", ">30.5<"))
            changed = metadata.qa_scene_metadata(safe, root_dir=directory)
        self.assertNotEqual(first["metadata_uid"], changed["metadata_uid"])
        self.assertEqual(changed["solar_zenith_deg"], 30.5)

    def test_ambiguous_product_directory_and_nonzero_offsets_remain_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            safe, _ = self.fixture(directory, offsets='<RADIO_ADD_OFFSET band_id="3">-1000</RADIO_ADD_OFFSET>')
            offset_record = metadata.qa_scene_metadata(safe)
            (safe.parent / "another.SAFE").mkdir()
            ambiguous = metadata.qa_scene_metadata(safe.parent)
        self.assertFalse(offset_record["metadata_usable"])
        self.assertFalse(offset_record["cloud_model_usable"])
        self.assertTrue(offset_record["nonzero_offset"])
        self.assertIn("ambiguous_safe_directory", ambiguous["issues"])
        self.assertFalse(ambiguous["metadata_usable"])


if __name__ == "__main__":
    unittest.main()
