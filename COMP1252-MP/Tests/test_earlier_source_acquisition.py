"""Tiny offline acquisition, ownership, and interrupted-publication regressions."""
from io import BytesIO
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Solutions"))
import acquire_earlier_validation_sources as acquisition

DATA = b"tiny source"


class Response(BytesIO):
    status = 200

    def __init__(self, asset, data=DATA, **headers):
        super().__init__(data)
        self.url = asset["url"]
        self.headers = {"ETag": asset["etag"], "Content-Length": str(asset["size"]), **headers}

    def geturl(self):
        return self.url


class EarlierSourceAcquisitionTests(unittest.TestCase):
    def setUp(self):
        disk = patch.object(acquisition.shutil, "disk_usage",
                            return_value=SimpleNamespace(free=acquisition.RESERVE + 1024))
        disk.start()
        self.addCleanup(disk.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sentinel = self.root / "sentinel"
        self.sentinel.mkdir()
        self.scenes = [{"scene_id": name, "acquisition_date": date, "metadata": {},
                        "assets": [{"target_relpath": name + ".SAFE/tiny.xml", "size": len(DATA),
                                    "etag": '"' + hashlib.md5(DATA).hexdigest() + '"',
                                    "url": "https://example.invalid/" + name}]}
                       for name, date in (("scene-a", "2018-07-21"), ("scene-b", "2018-07-16"))]
        self.transaction = acquisition.open_transaction(self.root / "transaction", self.sentinel,
                                                       "protocol-pin", self.scenes)
        for index, item in enumerate(self.transaction["scenes"]):
            staged, _ = acquisition.locations(self.transaction, index)
            staged.mkdir()
            item["identity"] = acquisition.identity(staged)
        acquisition.save(self.transaction)

    def target(self, index=0):
        staged, _ = acquisition.locations(self.transaction, index)
        return staged / self.scenes[index]["assets"][0]["target_relpath"]

    def prepare(self):
        for index, item in enumerate(self.transaction["scenes"]):
            target = self.target(index)
            target.parent.mkdir(parents=True)
            target.write_bytes(DATA)
            item["receipts"] = {self.scenes[index]["assets"][0]["target_relpath"]:
                                {"sha256": hashlib.sha256(DATA).hexdigest()}}
            item["state"] = "prepared"
        self.transaction["state"] = "ready"
        acquisition.save(self.transaction)

    def test_exclusive_move_preserves_an_existing_empty_directory(self):
        source, destination = self.root / "source", self.root / "destination"
        source.mkdir()
        destination.mkdir()
        original = acquisition.identity(destination)
        with self.assertRaises(OSError):
            acquisition.exclusive_move(source, destination)
        self.assertTrue(source.is_dir())
        self.assertEqual(acquisition.identity(destination), original)

    def test_open_transaction_rejects_changed_protocol_binding(self):
        with self.assertRaisesRegex(ValueError, "binding/ownership changed"):
            acquisition.open_transaction(self.root / "transaction", self.sentinel,
                                         "different-protocol", self.scenes)

    def test_atomic_journal_preserves_foreign_temporary_symlinks(self):
        target = self.root / 'foreign.txt'
        target.write_bytes(b'preserve user bytes')
        journal = self.root / 'transaction/journal.json'
        legacy_temporary = journal.with_name(journal.name + '.tmp')
        legacy_temporary.symlink_to(target)
        acquisition.save(self.transaction)
        self.assertEqual(target.read_bytes(), b'preserve user bytes')
        self.assertTrue(legacy_temporary.is_symlink())
        destination = self.root / 'symlink-destination.json'
        destination.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            acquisition.atomic_json(destination, {'state': 'never write'})
        self.assertEqual(target.read_bytes(), b'preserve user bytes')

    def test_stream_checks_request_pins_and_records_content_identity(self):
        asset = self.scenes[0]["assets"][0]
        requests = []
        def opener(request, timeout):
            requests.append((request.full_url, request.get_header("If-match"),
                             request.get_header("Accept-encoding"), timeout))
            return Response(asset)
        receipt = acquisition.stream_asset(asset, self.target(), self.transaction, 0, opener)
        self.assertEqual(requests, [(asset["url"], asset["etag"], "identity", 30)])
        self.assertEqual(self.target().read_bytes(), DATA)
        self.assertEqual(receipt["bytes"], len(DATA))
        self.assertEqual(receipt["md5"], hashlib.md5(DATA).hexdigest())
        self.assertEqual(receipt["sha256"], hashlib.sha256(DATA).hexdigest())
        self.assertTrue(receipt["if_match_verified"] and receipt["content_length_verified"])
        self.assertIsNone(self.transaction["scenes"][0]["partial"])

    def test_failed_stream_pins_and_payloads_remove_only_owned_partial(self):
        asset = self.scenes[0]["assets"][0]
        cases = [(DATA, {"ETag": '"' + "0" * 32 + '"'}),
                 (DATA, {"Content-Length": str(len(DATA) + 1)}),
                 (DATA[:-1], {}), (DATA + b"!", {}), (b"evil source", {})]
        for payload, headers in cases:
            with self.subTest(payload=payload, headers=headers), self.assertRaises(ValueError):
                acquisition.stream_asset(asset, self.target(), self.transaction, 0,
                                         lambda request, timeout: Response(asset, payload, **headers))
            self.assertFalse(self.target().exists())
            self.assertFalse(self.target().with_suffix(".xml.download").exists())
            self.assertIsNone(self.transaction["scenes"][0]["partial"])
            self.assertEqual(self.transaction["scenes"][0]["receipts"], {})
        removed = [event for event in self.transaction["events"] if event["kind"] == "owned_partial_removed"]
        self.assertEqual(len(removed), len(cases))

    def test_recover_partial_refuses_unowned_inode_and_unknown_target(self):
        staged, _ = acquisition.locations(self.transaction, 0)
        item = self.transaction["scenes"][0]
        for name, owned in ((item["targets"][0] + ".download", False), ("unknown.download", True)):
            partial = staged / name
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b"preserve")
            actual = acquisition.identity(partial)
            item["partial"] = {"path": name, "identity": actual if owned else [actual[0], -1]}
            with self.subTest(name=name), self.assertRaises(ValueError):
                acquisition.recover_partial(self.transaction, 0)
            self.assertEqual(partial.read_bytes(), b"preserve")

    def test_reused_nonempty_asset_is_reverified_and_corruption_is_refused(self):
        target = self.target()
        target.parent.mkdir(parents=True)
        target.write_bytes(b"evil source")
        with patch.object(acquisition, "stream_asset") as transfer:
            with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                acquisition.acquire_scene(self.transaction, self.scenes, 0)
        transfer.assert_not_called()
        self.assertEqual(target.read_bytes(), b"evil source")

    def test_publish_preserves_existing_user_tree(self):
        self.prepare()
        _, final = acquisition.locations(self.transaction, 0)
        final.mkdir()
        marker = final / "user.txt"
        marker.write_bytes(b"user data")
        original = acquisition.identity(final)
        with patch.object(acquisition, "validate_local", return_value={}):
            with self.assertRaisesRegex(ValueError, "unowned final tree"):
                acquisition.publish(self.transaction, self.scenes)
        self.assertEqual(marker.read_bytes(), b"user data")
        self.assertEqual(acquisition.identity(final), original)
        self.assertEqual(self.transaction["state"], "ready")

    def test_publish_callback_failure_rolls_owned_move_back(self):
        self.prepare()
        def fail_after_move(index):
            self.assertEqual(index, 0)
            self.assertTrue(acquisition.locations(self.transaction, 0)[1].is_dir())
            raise RuntimeError("injected publication interruption")
        with patch.object(acquisition, "validate_local", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "injected publication interruption"):
                acquisition.publish(self.transaction, self.scenes, after_move=fail_after_move)
        for index, item in enumerate(self.transaction["scenes"]):
            staged, final = acquisition.locations(self.transaction, index)
            self.assertTrue(acquisition.matches_identity(staged, item["identity"]))
            self.assertFalse(final.exists())
            self.assertEqual(item["state"], "prepared")
        self.assertEqual(self.transaction["state"], "ready")
        self.assertFalse((self.root / "transaction/completion.json").exists())

    def test_publish_resumes_rename_before_journal_update(self):
        self.prepare()
        self.transaction["state"] = "publishing"
        acquisition.save(self.transaction)
        acquisition.exclusive_move(*acquisition.locations(self.transaction, 0))
        recovered = acquisition.open_transaction(self.root / "transaction", self.sentinel,
                                                 "protocol-pin", self.scenes)
        self.assertEqual(recovered["scenes"][0]["state"], "prepared")
        with patch.object(acquisition, "validate_local", side_effect=lambda product, scene, *args:
                          {"scene_id": scene["scene_id"]}):
            result = acquisition.publish(recovered, self.scenes)
        self.assertEqual((recovered["state"], result["new_scenes"], result["new_assets"]),
                         ("complete", 2, 2))
        for index, item in enumerate(recovered["scenes"]):
            staged, final = acquisition.locations(recovered, index)
            self.assertFalse(staged.exists())
            self.assertTrue(acquisition.matches_identity(final, item["identity"]))
        self.assertTrue((self.root / "transaction/completion.json").is_file())

    def test_validate_local_rejects_unusable_metadata_after_asset_verification(self):
        self.prepare()
        staged, _ = acquisition.locations(self.transaction, 0)
        item = self.transaction["scenes"][0]
        seen = []
        def reader(product):
            seen.append(product)
            return {"metadata_usable": False}
        with self.assertRaisesRegex(ValueError, "frozen QA metadata"):
            acquisition.validate_local(staged, self.scenes[0], item["receipts"], reader)
        self.assertEqual(seen, [staged])
        self.assertEqual(item["receipts"][item["targets"][0]]["stat"][0], len(DATA))


if __name__ == "__main__":
    unittest.main()
