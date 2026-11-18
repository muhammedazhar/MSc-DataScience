"""Offline integrity and interrupted-publication regressions; no raster or model I/O."""
from io import BytesIO
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Solutions"))
import complete_validation_qa_inputs as repair


def asset(data, **extra):
    return {"size": len(data), "etag": '"' + hashlib.md5(data).hexdigest() + '"',
            "url": "https://example.invalid/source", **extra}


class Response(BytesIO):
    status = 200

    def __init__(self, data, declared=None, etag=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if declared is None else declared),
                        "ETag": etag or asset(data)["etag"]}


def fake_qa(product):
    return {"scene_id": product.name, "crs": "EPSG:32636", "quantification_value": 10000.0,
            "nonzero_offset": False, "metadata_usable": True, "issues": [],
            "cloud_model_usable": True, "shadow_geometry_usable": True, "missing_cloud_bands": [],
            "missing_quality_masks": {kind: [] for kind in repair.QUALITY_TYPES},
            "quality_band_paths": {kind: {band: "source.gml" for band in repair.QUALITY_BANDS}
                                   for kind in repair.QUALITY_TYPES}}


class QACompletionTests(unittest.TestCase):
    def test_seven_band_inventory_identity_does_not_replace_frozen_qa_gate(self):
        original = fake_qa(Path("scene-a"))
        self.assertIsNone(repair.verify_qa(copy.deepcopy(original), original))
        incomplete = []
        record = copy.deepcopy(original)
        record["metadata_usable"] = False
        record["issues"] = ["missing_cloud_band_B01"]
        incomplete.append(("QA metadata rejects incomplete scene", record))
        record = copy.deepcopy(original)
        record["missing_cloud_bands"] = ["B01", "B05", "B09", "B10"]
        incomplete.append(("all four cloud inputs are mandatory", record))
        record = copy.deepcopy(original)
        record["missing_quality_masks"]["NODATA"] = ["B01"]
        incomplete.append(("source masks cannot be assumed clear", record))
        record = copy.deepcopy(original)
        record["quality_band_paths"]["TECQUA"]["B12"] = None
        incomplete.append(("all 44 discovered mask paths are mandatory", record))
        for reason, record in incomplete:
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                repair.verify_qa(record, original)

    def fixture(self, root):
        sentinel = root / "Datasets/Sentinel-2"
        sentinel.mkdir(parents=True)
        scenes = []
        for name in ("scene-a", "scene-b"):
            product = sentinel / name
            product.mkdir()
            old = product / "original.xml"
            old.write_bytes(b"original " + name.encode())
            old_asset = {"path": str(old.relative_to(root)), "bytes": old.stat().st_size,
                         "sha256": hashlib.sha256(old.read_bytes()).hexdigest(),
                         "etag": asset(old.read_bytes())["etag"], "source_url": "https://example.invalid/original"}
            scenes.append({"scene_id": name, "old_assets": [old_asset],
                           "missing_assets": [asset(b"supplement", target_relpath="added.gml")]})
        return {"scenes": scenes}, sentinel, root / "Datasets/ValidationPilot/v10/transaction"

    def downloader(self, source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"supplement")
        return {**repair.file_digests(target), "path": str(target)}

    def test_download_pins_if_match_and_records_content_identity(self):
        data = b"actual source data"
        source = asset(data)
        seen = []
        def opener(request, timeout):
            seen.append((request.get_header("If-match"), timeout))
            return Response(data)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "asset.gml"
            receipt = repair.download_asset(source, target, opener=opener)
            self.assertEqual(seen, [(source["etag"], 30)])
            self.assertEqual(receipt["sha256"], hashlib.sha256(data).hexdigest())
            self.assertTrue(receipt["content_length_verified"])
            self.assertTrue(receipt["if_match_verified"])

    def test_download_rejects_size_md5_and_etag_failures_without_final_asset(self):
        for data, declared, response_etag in ((b"short", 10, None),
                                               (b"wrong-data", 10, None),
                                               (b"sourceDATA", 10, '"' + "0" * 32 + '"'),
                                               (b"sourceDATA", 9, None)):
            source = asset(b"sourceDATA")
            with self.subTest(data=data, declared=declared, etag=response_etag), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "asset.gml"
                def opener(request, timeout):
                    return Response(data, declared, response_etag or source["etag"])
                with self.assertRaises(ValueError):
                    repair.download_asset(source, target, opener=opener)
                self.assertFalse(target.exists())
                self.assertFalse(target.with_name("asset.gml.download").exists())

    def test_nonempty_original_with_source_md5_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, sentinel, transaction = self.fixture(root)
            old = sentinel / "scene-a/original.xml"
            old.write_bytes(b"CORRUPT" + b"x" * (old.stat().st_size - 7))
            with patch.object(repair, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                    repair.prepare_transaction(protocol, transaction, sentinel, qa_reader=fake_qa, downloader=self.downloader)
            self.assertEqual(old.read_bytes()[:7], b"CORRUPT")
            self.assertFalse((sentinel / "scene-a/added.gml").exists())

    def test_staging_failure_never_changes_either_discoverable_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, sentinel, transaction = self.fixture(root)
            originals = {name: repair.identity(sentinel / name) for name in ("scene-a", "scene-b")}
            calls = []
            def fail_second(source, target):
                calls.append(target)
                if len(calls) == 2:
                    raise OSError("simulated second scene transfer failure")
                return self.downloader(source, target)
            with patch.object(repair, "ROOT", root):
                with self.assertRaises(OSError):
                    repair.prepare_transaction(protocol, transaction, sentinel, qa_reader=fake_qa, downloader=fail_second)
            self.assertEqual(repair.read(transaction / "journal.json")["state"], "rolled_back")
            for name in originals:
                self.assertEqual(repair.identity(sentinel / name), originals[name])
                self.assertFalse((sentinel / name / "added.gml").exists())

    def test_second_scene_publication_failure_restores_both_original_inodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, sentinel, transaction = self.fixture(root)
            originals = {name: repair.identity(sentinel / name) for name in ("scene-a", "scene-b")}
            def fail(index):
                if index == 1:
                    raise OSError("simulated two-scene publication interruption")
            with patch.object(repair, "ROOT", root):
                journal = repair.prepare_transaction(protocol, transaction, sentinel, qa_reader=fake_qa, downloader=self.downloader)
                with self.assertRaises(OSError):
                    repair.publish_transaction(journal, after_replace=fail, qa_reader=fake_qa)
                self.assertEqual(repair.recover_transaction(transaction)["state"], "rolled_back")
            for name in originals:
                self.assertEqual(repair.identity(sentinel / name), originals[name])
                self.assertFalse((sentinel / name / "added.gml").exists())
            generation = Path(journal["generation"])
            self.assertTrue((generation / "rolled_back_new/scene-a/added.gml").exists())
            self.assertTrue((generation / "rolled_back_new/scene-b/added.gml").exists())

    def test_crash_between_original_backup_and_publish_is_recovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, sentinel, transaction = self.fixture(root)
            with patch.object(repair, "ROOT", root):
                journal = repair.prepare_transaction(protocol, transaction, sentinel, qa_reader=fake_qa, downloader=self.downloader)
                item = journal["scenes"][0]
                backup = Path(item["backup"])
                backup.parent.mkdir(parents=True)
                Path(item["final"]).rename(backup)
                # Simulate process death before any post-rename journal write.
                recovered = repair.recover_transaction(transaction)
                self.assertEqual(recovered["state"], "rolled_back")
                self.assertEqual(repair.identity(item["final"]), item["original_identity"])
                self.assertFalse(backup.exists())

    def test_recovery_preserves_unexpected_user_tree_and_original_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, sentinel, transaction = self.fixture(root)
            with patch.object(repair, "ROOT", root):
                journal = repair.prepare_transaction(protocol, transaction, sentinel, qa_reader=fake_qa, downloader=self.downloader)
                item = journal["scenes"][0]
                final, backup = Path(item["final"]), Path(item["backup"])
                backup.parent.mkdir(parents=True)
                final.rename(backup)
                final.mkdir()
                (final / "user-file.txt").write_text("retain me")
                with self.assertRaisesRegex(ValueError, "unexpected final tree"):
                    repair.recover_transaction(transaction)
                self.assertEqual((final / "user-file.txt").read_text(), "retain me")
                self.assertEqual(repair.identity(backup), item["original_identity"])


if __name__ == "__main__":
    unittest.main()
