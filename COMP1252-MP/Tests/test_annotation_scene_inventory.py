"""Tiny metadata-only fixtures; satellite pixels are never opened."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "Solutions/annotation_scene_inventory.py"
SPEC = importlib.util.spec_from_file_location("annotation_scene_inventory", MODULE_PATH)
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)
PRODUCT = "S2A_MSIL1C_20180731T083601_N0206_R064_T36UXA_20180731T110233"


class InventoryTests(unittest.TestCase):
    def fixture(self, directory, name=PRODUCT, direct=False, offsets=""):
        directory = Path(directory)
        safe = directory / f"{name}.SAFE" if direct else directory / name / f"{name}.SAFE"
        granule = safe / "GRANULE/L1C_T36UXA_A000001_20180731T083601"
        bands, quality = granule / "IMG_DATA", granule / "QI_DATA"
        bands.mkdir(parents=True)
        quality.mkdir()
        for band in inventory.REQUIRED_BANDS:
            (bands / f"T36UXA_20180731T083601_{band}.jp2").write_bytes(b"not a raster")
        (quality / "MSK_CLOUDS_B00.gml").write_text(
            '<eop:Mask xmlns:eop="http://www.opengis.net/eop/2.0"><eop:maskMembers/></eop:Mask>'
        )
        (safe / "MTD_MSIL1C.xml").write_text(
            '<n:Product xmlns:n="urn:test"><PROCESSING_BASELINE>02.06</PROCESSING_BASELINE>'
            '<QUANTIFICATION_VALUE>10000</QUANTIFICATION_VALUE>'
            f'<Cloud_Coverage_Assessment>4.2</Cloud_Coverage_Assessment>{offsets}</n:Product>'
        )
        (granule / "MTD_TL.xml").write_text(
            '<n:Tile xmlns:n="urn:test"><HORIZONTAL_CS_CODE>EPSG:32636</HORIZONTAL_CS_CODE>'
            '<Size resolution="20"><NROWS>5</NROWS><NCOLS>6</NCOLS></Size>'
            '<Size resolution="10"><NROWS>10</NROWS><NCOLS>12</NCOLS></Size>'
            '<Geoposition resolution="10"><ULX>600000</ULX><ULY>5600040</ULY>'
            '<XDIM>10</XDIM><YDIM>-10</YDIM></Geoposition>'
            '<CLOUDY_PIXEL_PERCENTAGE>3.1</CLOUDY_PIXEL_PERCENTAGE></n:Tile>'
        )
        return safe, granule

    def test_complete_inventory_reads_metadata_but_never_jp2_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            safe, _ = self.fixture(directory)
            read_bytes = Path.read_bytes

            def forbid_raster_contents(path):
                self.assertNotEqual(path.suffix.lower(), ".jp2")
                return read_bytes(path)

            with patch.object(Path, "read_bytes", forbid_raster_contents):
                records = inventory.inventory_scenes(directory, root_dir=directory)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertTrue(record["metadata_usable"], record["issues"])
        self.assertEqual(record["scene_id"], PRODUCT)
        self.assertEqual(record["tile"], "36UXA")
        self.assertEqual(record["acquisition_date"], "2018-07-31")
        self.assertEqual(record["acquisition_datetime"], "2018-07-31T08:36:01Z")
        self.assertEqual(record["processing_baseline"], "02.06")
        self.assertEqual(record["baseline_code"], "N0206")
        self.assertEqual(record["quantification_value"], 10000)
        self.assertEqual(record["cloud_cover_percent"], 3.1)
        self.assertEqual(record["cloud_cover_source"], "tile_metadata")
        self.assertEqual(record["grid_10m"]["bounds"], [600000, 5599940, 600120, 5600040])
        self.assertEqual(record["grid_10m"]["width"], 12)
        self.assertEqual(record["grid_10m"]["height"], 10)
        self.assertEqual(record["missing_bands"], [])
        self.assertFalse(record["cloud_qa_missing"])
        self.assertTrue(record["cloud_qa_well_formed"])
        self.assertFalse(record["nonzero_offset"])
        self.assertEqual(set(record["metadata_sha256"]), {"product_metadata", "tile_metadata", "cloud_qa"})
        self.assertFalse(Path(record["path"]).is_absolute())

    def test_missing_band_cloud_and_malformed_tile_are_retained_as_issues(self):
        with tempfile.TemporaryDirectory() as directory:
            safe, granule = self.fixture(directory)
            next((granule / "IMG_DATA").glob("*_B11.jp2")).unlink()
            (granule / "QI_DATA/MSK_CLOUDS_B00.gml").unlink()
            (granule / "MTD_TL.xml").write_text("<broken>")
            records = inventory.inventory_scenes(directory)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertFalse(record["metadata_usable"])
        self.assertEqual(record["missing_bands"], ["B11"])
        self.assertTrue(record["cloud_qa_missing"])
        self.assertIsNone(record["grid_10m"])
        self.assertIn("malformed_tile_metadata", record["issues"])
        self.assertEqual(record["cloud_cover_percent"], 4.2)
        self.assertEqual(record["cloud_cover_source"], "product_metadata")

    def test_nonzero_offsets_are_reported_as_unsupported_by_current_crop_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory, offsets='<RADIO_ADD_OFFSET band_id="3">-1000</RADIO_ADD_OFFSET>')
            record = inventory.inventory_scenes(directory)[0]
        self.assertTrue(record["nonzero_offset"])
        self.assertEqual(record["radiometric_offsets"], {"3": -1000})
        self.assertFalse(record["metadata_usable"])
        self.assertIn("nonzero_radiometric_offset_unsupported", record["issues"])

    def test_direct_safe_layout_and_missing_safe_directory_are_both_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory, direct=True)
            missing = PRODUCT.replace("S2A_", "S2B_", 1)
            (Path(directory) / missing).mkdir()
            records = inventory.inventory_scenes(directory)
        self.assertEqual(len(records), 2)
        complete = next(record for record in records if record["satellite"] == "S2A")
        incomplete = next(record for record in records if record["satellite"] == "S2B")
        self.assertTrue(complete["metadata_usable"])
        self.assertFalse(incomplete["metadata_usable"])
        self.assertIn("missing_safe_directory", incomplete["issues"])

    def test_invalid_dates_and_nonfinite_numeric_values_are_flagged(self):
        invalid_name = PRODUCT.replace("20180731T083601", "20180231T083601")
        with tempfile.TemporaryDirectory() as directory:
            safe, _ = self.fixture(directory, name=invalid_name)
            product_xml = safe / "MTD_MSIL1C.xml"
            product_xml.write_text(product_xml.read_text().replace(">10000<", ">NaN<"))
            record = inventory.inventory_scenes(directory)[0]
        self.assertFalse(record["metadata_usable"])
        self.assertIsNone(record["acquisition_date"])
        self.assertIsNone(record["quantification_value"])
        self.assertIn("invalid_acquisition_datetime", record["issues"])
        self.assertIn("invalid_quantification_value", record["issues"])

    def test_multiple_required_band_files_are_flagged_not_chosen_arbitrarily(self):
        with tempfile.TemporaryDirectory() as directory:
            _, granule = self.fixture(directory)
            (granule / "IMG_DATA/duplicate_B02.jp2").write_bytes(b"not a raster")
            record = inventory.inventory_scenes(directory)[0]
        self.assertFalse(record["metadata_usable"])
        self.assertIn("B02", record["ambiguous_bands"])
        self.assertIsNone(record["band_paths"]["B02"])
        self.assertIn("ambiguous_band_B02", record["issues"])


if __name__ == "__main__":
    unittest.main()
