"""Tests for scene ingestion, SAFE directory assembly, and registration."""
import hashlib
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Solutions'))
from ingest_validation_scenes import (
    check_bounds_containment,
    download_file,
    ingest_scene,
)
from annotation_scene_inventory import inventory_scenes


class ValidationIngestionTests(unittest.TestCase):
    def test_check_bounds_containment(self):
        outer = [100.0, 100.0, 200.0, 200.0]
        # Fully inside
        self.assertTrue(check_bounds_containment([120.0, 120.0, 180.0, 180.0], outer))
        # Exact match
        self.assertTrue(check_bounds_containment([100.0, 100.0, 200.0, 200.0], outer))
        # Partially outside left
        self.assertFalse(check_bounds_containment([90.0, 120.0, 180.0, 180.0], outer))
        # Partially outside bottom
        self.assertFalse(check_bounds_containment([120.0, 90.0, 180.0, 180.0], outer))
        # Partially outside right
        self.assertFalse(check_bounds_containment([120.0, 120.0, 210.0, 180.0], outer))
        # Partially outside top
        self.assertFalse(check_bounds_containment([120.0, 120.0, 180.0, 210.0], outer))
        # Completely outside
        self.assertFalse(check_bounds_containment([300.0, 300.0, 400.0, 400.0], outer))

    def test_download_file_mocked_http(self):
        sample_data = b"Hello, Sentinel-2 Open Data test content!"
        expected_hash = hashlib.sha256(sample_data).hexdigest()

        mock_resp = MagicMock()
        mock_resp.read.side_effect = [sample_data, b""]
        mock_resp.__enter__.return_value = mock_resp

        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "subdir" / "test_download.jp2"
            with patch("urllib.request.urlopen", return_value=mock_resp):
                res = download_file("https://example.com/asset.jp2", target, expected_size=len(sample_data))
                self.assertEqual(res["bytes"], len(sample_data))
                self.assertEqual(res["sha256"], expected_hash)
                self.assertTrue(target.exists())
                self.assertEqual(target.read_bytes(), sample_data)

    def test_ingest_scene_dry_run(self):
        manifest = {
            "scene_id": "S2B_MSIL1C_TEST",
            "safe_directory": "S2B_MSIL1C_TEST.SAFE",
            "metadata_assets": [
                {"role": "product_metadata", "url": "https://example.com/meta.xml", "target_relpath": "S2B_MSIL1C_TEST.SAFE/MTD_MSIL1C.xml"}
            ],
            "band_assets": [
                {"band": "B02", "url": "https://example.com/B02.jp2", "target_relpath": "S2B_MSIL1C_TEST.SAFE/GRANULE/G1/IMG_DATA/B02.jp2"}
            ]
        }
        with tempfile.TemporaryDirectory() as d:
            res = ingest_scene(manifest, d, dry_run=True)
            self.assertEqual(res["total_bytes"], 0)
            self.assertEqual(len(res["assets"]), 2)
            self.assertTrue(all(a["status"] == "planned" for a in res["assets"]))

    def test_ingest_scene_with_mock_source_and_existing_cache(self):
        manifest = {
            "scene_id": "S2B_MSIL1C_TEST",
            "safe_directory": "S2B_MSIL1C_TEST.SAFE",
            "metadata_assets": [
                {"role": "product_metadata", "url": "https://example.com/meta.xml", "target_relpath": "S2B_MSIL1C_TEST.SAFE/MTD_MSIL1C.xml"}
            ],
            "band_assets": [
                {"band": "B02", "url": "https://example.com/B02.jp2", "target_relpath": "S2B_MSIL1C_TEST.SAFE/GRANULE/G1/IMG_DATA/B02.jp2"}
            ]
        }
        with tempfile.TemporaryDirectory() as d_src, tempfile.TemporaryDirectory() as d_target:
            # Create mock source assets
            (Path(d_src) / "MTD_MSIL1C.xml").write_bytes(b"<xml>metadata</xml>")
            (Path(d_src) / "B02.jp2").write_bytes(b"JP2_PIXEL_DATA")

            # First run: mock copied
            res1 = ingest_scene(manifest, d_target, mock_source_dir=d_src)
            self.assertEqual(len(res1["assets"]), 2)
            self.assertTrue(all(a["status"] == "mock_copied" for a in res1["assets"]))
            self.assertEqual(res1["total_bytes"], len(b"<xml>metadata</xml>") + len(b"JP2_PIXEL_DATA"))

            # Second run: existing cache detected
            res2 = ingest_scene(manifest, d_target, mock_source_dir=d_src)
            self.assertEqual(len(res2["assets"]), 2)
            self.assertTrue(all(a["status"] == "existing" for a in res2["assets"]))
            self.assertEqual(res2["total_bytes"], res1["total_bytes"])

    def test_safe_registration_and_event_containment(self):
        scene_id = "S2B_MSIL1C_20180726T084009_N0206_R064_T36UXB_20180726T104227"
        granule_id = "L1C_T36UXB_A007240_20180726T084437"

        prod_xml = (
            "<Root><PROCESSING_BASELINE>02.06</PROCESSING_BASELINE>"
            "<QUANTIFICATION_VALUE>10000</QUANTIFICATION_VALUE></Root>"
        )
        tile_xml = (
            "<Root><HORIZONTAL_CS_CODE>EPSG:32636</HORIZONTAL_CS_CODE>"
            "<Size resolution=\"10\"><NCOLS>10980</NCOLS><NROWS>10980</NROWS></Size>"
            "<Geoposition resolution=\"10\"><ULX>600000</ULX><ULY>5700000</ULY>"
            "<XDIM>10</XDIM><YDIM>-10</YDIM></Geoposition>"
            "<CLOUDY_PIXEL_PERCENTAGE>0.00</CLOUDY_PIXEL_PERCENTAGE></Root>"
        )
        cloud_qa = "<Mask></Mask>"

        with tempfile.TemporaryDirectory() as d:
            safe_parent = Path(d) / scene_id
            safe_dir = safe_parent / f"{scene_id}.SAFE"
            granule_dir = safe_dir / "GRANULE" / granule_id
            (granule_dir / "IMG_DATA").mkdir(parents=True, exist_ok=True)
            (granule_dir / "QI_DATA").mkdir(parents=True, exist_ok=True)

            (safe_dir / "MTD_MSIL1C.xml").write_text(prod_xml)
            (granule_dir / "MTD_TL.xml").write_text(tile_xml)
            (granule_dir / "QI_DATA" / "MSK_CLOUDS_B00.gml").write_text(cloud_qa)

            for band in ("B02", "B03", "B04", "B08", "B8A", "B11", "B12"):
                (granule_dir / "IMG_DATA" / f"T36UXB_20180726T084009_{band}.jp2").write_bytes(b"JP2")

            scenes = inventory_scenes(d)
            self.assertEqual(len(scenes), 1)
            s = scenes[0]
            self.assertEqual(s["scene_id"], scene_id)
            self.assertEqual(s["tile"], "36UXB")
            self.assertTrue(s["metadata_usable"])
            self.assertFalse(s["nonzero_offset"])
            self.assertEqual(s["missing_bands"], [])
            self.assertFalse(s["cloud_qa_missing"])
            self.assertTrue(s["cloud_qa_well_formed"])

            # Verify containment of both validation events
            tile_bounds = s["grid_10m"]["bounds"]  # [600000, 5590200, 709800, 5700000]
            event_1_crop = [663640.0, 5605000.0, 664280.0, 5605640.0]
            event_1_qa = [662440.0, 5603800.0, 665480.0, 5606840.0]
            event_2_crop = [663380.0, 5604740.0, 664020.0, 5605380.0]
            event_2_qa = [662180.0, 5603540.0, 665220.0, 5606580.0]

            self.assertTrue(check_bounds_containment(event_1_crop, tile_bounds))
            self.assertTrue(check_bounds_containment(event_1_qa, tile_bounds))
            self.assertTrue(check_bounds_containment(event_2_crop, tile_bounds))
            self.assertTrue(check_bounds_containment(event_2_qa, tile_bounds))


if __name__ == '__main__':
    unittest.main()
