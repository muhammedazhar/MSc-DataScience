"""Small synthetic regressions for the single-sample geospatial validator.

These tests never open the satellite dataset. All raster/QA fixtures are tiny
temporary files and native libraries are restricted to one thread.
"""

from datetime import date
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "GDAL_NUM_THREADS",
    "OPJ_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box


MODULE_PATH = Path(__file__).resolve().parents[1] / "Solutions/validate_sample.py"
SPEC = importlib.util.spec_from_file_location("validate_sample", MODULE_PATH)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class FeatureTests(unittest.TestCase):
    def test_unsigned_inputs_preserve_negative_indices_and_independent_scaling(self):
        raw = {name: np.array([[2000]], dtype=np.uint16) for name in validator.BANDS}
        raw.update(
            B04=np.array([[2051]], dtype=np.uint16),
            B08=np.array([[2048]], dtype=np.uint16),
            B8A=np.array([[1000]], dtype=np.uint16),
            B11=np.array([[3000]], dtype=np.uint16),
        )

        features = validator.compute_features(raw, 10000)

        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(features.shape, (9, 1, 1))
        self.assertAlmostEqual(float(features[3, 0, 0]), 0.2048, places=7)
        self.assertAlmostEqual(float(features[7, 0, 0]), -3 / 4099, places=9)
        self.assertAlmostEqual(float(features[8, 0, 0]), -0.5, places=7)
        # Feature computation must not modify original source DNs.
        self.assertEqual(int(raw["B08"][0, 0]), 2048)

    def test_zero_denominators_do_not_create_nan_or_infinity(self):
        raw = {name: np.zeros((2, 2), dtype=np.uint16) for name in validator.BANDS}
        features = validator.compute_features(raw, 10000)
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(features, np.zeros((9, 2, 2)))


class TemporalTargetTests(unittest.TestCase):
    def test_target_includes_all_interval_changes_and_excludes_existing_or_future_loss(self):
        grid = {"height": 4, "width": 5, "transform": from_origin(0, 4, 1, 1)}
        pre, post = date(2018, 7, 26), date(2018, 8, 5)
        records = [
            (pre, box(0, 0, 2, 4)),
            (date(2018, 7, 31), box(1, 0, 3, 4)),
            (post, box(4, 3, 5, 4)),
            (date(2018, 8, 6), box(3, 0, 4, 4)),
        ]

        before, after, change = validator.temporal_masks(records, pre, post, grid)

        expected = np.zeros((4, 5), dtype=np.uint8)
        expected[:, 2] = 1
        expected[0, 4] = 1
        np.testing.assert_array_equal(change, expected)
        self.assertEqual(int(before.sum()), 8)
        self.assertEqual(int(after.sum()), 13)
        self.assertFalse(change[:, :2].any())
        self.assertFalse(change[:, 3].any())

    def test_reversed_observations_are_rejected(self):
        grid = {"height": 2, "width": 2, "transform": from_origin(0, 2, 1, 1)}
        with self.assertRaises(ValueError):
            validator.temporal_masks([], date(2018, 8, 5), date(2018, 7, 26), grid)


class QualityGeometryTests(unittest.TestCase):
    def write_gml(self, directory, polygon="", crs=None):
        envelope = f'<gml:boundedBy><gml:Envelope srsName="{crs}"/></gml:boundedBy>' if crs else ""
        text = (
            '<eop:Mask xmlns:eop="http://www.opengis.net/eop/2.0" '
            'xmlns:gml="http://www.opengis.net/gml/3.2">'
            f'{envelope}<eop:maskMembers>{polygon}</eop:maskMembers></eop:Mask>'
        )
        path = Path(directory) / "quality.gml"
        path.write_text(text)
        return path

    def test_empty_quality_mask_without_crs_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_gml(directory)
            geometries, repairs = validator.quality_geometries(path, "EPSG:32636")
        self.assertEqual(geometries, [])
        self.assertEqual(repairs, 0)

    def test_holes_survive_projection_and_coordinates_are_mapped_in_xy_order(self):
        polygon = """
          <gml:Polygon>
            <gml:exterior><gml:LinearRing><gml:posList srsDimension="2">
              0 0 0.01 0 0.01 0.02 0 0.02 0 0
            </gml:posList></gml:LinearRing></gml:exterior>
            <gml:interior><gml:LinearRing><gml:posList srsDimension="2">
              0.002 0.005 0.008 0.005 0.008 0.015 0.002 0.015 0.002 0.005
            </gml:posList></gml:LinearRing></gml:interior>
          </gml:Polygon>
        """
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_gml(directory, polygon, "EPSG:4326")
            geometries, repairs = validator.quality_geometries(path, "EPSG:3857")
        self.assertEqual(len(geometries), 1)
        geometry = geometries[0]
        self.assertEqual(repairs, 0)
        self.assertEqual(len(geometry.interiors), 1)
        self.assertAlmostEqual(geometry.bounds[2], 1113.1949079, places=3)
        self.assertAlmostEqual(geometry.bounds[3], 2226.3898611, places=3)
        self.assertTrue(geometry.contains(Point(100, 100)))
        self.assertFalse(geometry.contains(Point(500, 1000)))

    def test_nonempty_quality_mask_without_crs_is_rejected(self):
        polygon = """
          <gml:Polygon><gml:exterior><gml:LinearRing><gml:posList>
            0 0 1 0 1 1 0 1 0 0
          </gml:posList></gml:LinearRing></gml:exterior></gml:Polygon>
        """
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_gml(directory, polygon)
            with self.assertRaises(ValueError):
                validator.quality_geometries(path, "EPSG:32636")

    def test_three_dimensional_quality_ring_discards_nonzero_altitudes(self):
        polygon = """
          <gml:Polygon><gml:exterior><gml:LinearRing><gml:posList srsDimension="3">
            100 200 9 120 200 12 120 220 15 100 220 18 100 200 9
          </gml:posList></gml:LinearRing></gml:exterior></gml:Polygon>
        """
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_gml(directory, polygon, "EPSG:32636")
            geometries, repairs = validator.quality_geometries(path, "EPSG:32636")
        self.assertEqual(len(geometries), 1)
        self.assertEqual(repairs, 0)
        self.assertEqual(geometries[0].bounds, (100, 200, 120, 220))
        self.assertEqual(geometries[0].area, 400)
        self.assertFalse(geometries[0].has_z)


class RasterAlignmentTests(unittest.TestCase):
    def write_raster(self, path, data, transform):
        with rasterio.open(
            path, "w", driver="GTiff", count=1, width=data.shape[1], height=data.shape[0],
            dtype=data.dtype, crs="EPSG:32636", transform=transform, nodata=0,
        ) as destination:
            destination.write(data, 1)

    def test_10m_and_20m_bands_land_on_identical_physical_crop(self):
        # The two source rasters sample the same physical plane at different
        # resolutions. A crop away from image edges should agree after warping.
        fine_x, fine_y = np.meshgrid(np.arange(5, 80, 10), np.arange(75, 0, -10))
        coarse_x, coarse_y = np.meshgrid(np.arange(10, 80, 20), np.arange(70, 0, -20))
        fine = (1000 + fine_x + 2 * fine_y).astype(np.uint16)
        coarse = (1000 + coarse_x + 2 * coarse_y).astype(np.uint16)
        grid = {"height": 4, "width": 4, "transform": from_origin(20, 60, 10, 10), "crs": "EPSG:32636"}
        with tempfile.TemporaryDirectory() as directory, rasterio.Env(GDAL_CACHEMAX=8 * 2**20, GDAL_NUM_THREADS="1"):
            fine_path, coarse_path = Path(directory) / "fine.tif", Path(directory) / "coarse.tif"
            self.write_raster(fine_path, fine, from_origin(0, 80, 10, 10))
            self.write_raster(coarse_path, coarse, from_origin(0, 80, 20, 20))
            fine_crop, fine_valid = validator.read_band_on_grid(fine_path, grid)
            coarse_crop, coarse_valid = validator.read_band_on_grid(coarse_path, grid)
        np.testing.assert_allclose(fine_crop, fine[2:6, 2:6], rtol=0, atol=1e-4)
        np.testing.assert_allclose(coarse_crop, fine[2:6, 2:6], rtol=0, atol=1e-4)
        self.assertTrue(fine_valid.all())
        self.assertTrue(coarse_valid.all())

    def test_nodata_and_saturation_are_not_valid_training_pixels(self):
        grid = {"height": 2, "width": 2, "transform": from_origin(0, 20, 10, 10), "crs": "EPSG:32636"}
        with tempfile.TemporaryDirectory() as directory, rasterio.Env(GDAL_CACHEMAX=8 * 2**20, GDAL_NUM_THREADS="1"):
            path = Path(directory) / "quality.tif"
            self.write_raster(path, np.array([[0, 65535], [100, 200]], dtype=np.uint16), grid["transform"])
            _, valid = validator.read_band_on_grid(path, grid)
        np.testing.assert_array_equal(valid, [[False, False], [True, True]])


class ResourceGuardTests(unittest.TestCase):
    def args(self, output):
        return SimpleNamespace(
            output=output, min_available_gib=1.5, max_rss_mib=512, max_seconds=180,
        )

    def fake_psutil(self, available_gib):
        fake = mock.MagicMock()
        fake.virtual_memory.return_value = SimpleNamespace(
            total=18 * 2**30, available=available_gib * 2**30,
        )

        class NoSuchProcess(Exception):
            pass

        fake.NoSuchProcess = NoSuchProcess
        return fake

    def test_low_available_memory_refuses_before_starting_or_creating_output(self):
        fake = self.fake_psutil(1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created" / "sample"
            with mock.patch.dict(sys.modules, {"psutil": fake}), \
                    mock.patch.object(validator.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "Insufficient available memory"):
                    validator.run_guarded(self.args(output))
            popen.assert_not_called()
            self.assertFalse(output.parent.exists())

    def test_excess_worker_memory_terminates_worker_and_discards_only_staging(self):
        fake = self.fake_psutil(4)
        fake.Process.return_value.memory_info.return_value = SimpleNamespace(rss=513 * 2**20)
        worker = mock.MagicMock()
        worker.pid = 12345
        worker.poll.return_value = None
        worker.wait.return_value = 0
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "sample"
            existing = parent / "existing.txt"
            existing.write_text("keep existing data")
            with mock.patch.dict(sys.modules, {"psutil": fake}), \
                    mock.patch.object(validator.subprocess, "Popen", return_value=worker) as popen, \
                    mock.patch.object(validator.shutil, "disk_usage", return_value=SimpleNamespace(free=2**30)):
                with self.assertRaisesRegex(RuntimeError, "Worker exceeded its memory budget"):
                    validator.run_guarded(self.args(output))

            worker.terminate.assert_called_once_with()
            worker.wait.assert_called_once_with(timeout=3)
            command = popen.call_args.args[0]
            staging = Path(command[command.index("--worker-output") + 1])
            self.assertFalse(staging.exists())
            self.assertFalse(output.exists())
            self.assertEqual(existing.read_text(), "keep existing data")
            self.assertEqual(list(parent.iterdir()), [existing])


if __name__ == "__main__":
    unittest.main()
