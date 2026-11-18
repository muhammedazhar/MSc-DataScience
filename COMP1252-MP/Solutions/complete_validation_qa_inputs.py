#!/usr/bin/env python3
"""Complete the fixed V9 source pair without changing frozen QA implementations.

Network assets are source-pinned by simple S3 ETags (MD5), exact byte counts,
and If-Match. A persistent transaction outside scene discovery preserves the
original products and makes interrupted pair publication recoverable.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCENE_IDS = (
    "S2B_MSIL1C_20180726T084009_N0206_R064_T36UXB_20180726T104227",
    "S2A_MSIL1C_20180731T083601_N0206_R064_T36UXB_20180731T110233",
)
EXTRA_BANDS = ("B01", "B05", "B09", "B10")
QUALITY_BANDS = ("B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12", "B03")
QUALITY_TYPES = ("NODATA", "SATURA", "DEFECT", "TECQUA")
SUPPLEMENT_BYTES = 74541437
MIN_DISK_BYTES = 2 * 2**30
RESOURCES = {"workers": 1, "native_threads": 1, "nice_increment": 10,
             "gpu_used": False, "worker_rss_mib": 512,
             "available_memory_reserve_gib": 1.5, "max_seconds_per_worker": 300}
STABLE_METADATA = ("scene_id", "satellite", "processing_level", "tile", "acquisition_date",
                   "acquisition_datetime", "baseline_code", "processing_baseline",
                   "quantification_value", "radiometric_offsets", "nonzero_offset",
                   "crs", "grid_10m", "metadata_sha256")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.stat().st_size <= 32 * 2**20, "Metadata exceeds size bound")
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_digests(path):
    sha256, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            size += len(chunk)
            sha256.update(chunk)
            md5.update(chunk)
    return {"bytes": size, "sha256": sha256.hexdigest(), "md5": md5.hexdigest()}


def sha(path):
    return file_digests(path)["sha256"]


def expected_md5(asset):
    require(re.fullmatch(r'"[0-9a-f]{32}"', asset["etag"]) is not None, "Only simple quoted MD5 ETags are supported")
    return asset["etag"][1:-1]


def verify_asset(path, asset, known_sha256=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing or symlink asset: {path}")
    result = file_digests(path)
    require(result["bytes"] == asset.get("size", asset.get("bytes")), f"Source size mismatch: {path}")
    require(result["md5"] == expected_md5(asset), f"Source MD5 mismatch: {path}")
    checksum = known_sha256 or asset.get("sha256")
    if checksum:
        require(result["sha256"] == checksum, f"Recorded SHA256 mismatch: {path}")
    return result


def download_asset(asset, target, opener=None):
    """One bounded HTTP request; failures retain no purported complete asset."""
    target = Path(target)
    require(not target.exists(), f"Refusing to overwrite asset: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    require(not temporary.exists(), f"Unexpected download temporary: {temporary}")
    expected_md5(asset)
    request = urllib.request.Request(asset["url"], headers={"If-Match": asset["etag"], "Accept-Encoding": "identity",
                                                         "User-Agent": "DeforestationPilot-SourceRepair/10"})
    opener = opener or urllib.request.urlopen
    size, sha256, md5 = 0, hashlib.sha256(), hashlib.md5()
    try:
        with opener(request, timeout=30) as response:
            require(response.status == 200, "Full HTTP 200 response required")
            require(response.headers.get("ETag") == asset["etag"], "Response ETag changed")
            require(int(response.headers.get("Content-Length", "-1")) == asset["size"], "Response Content-Length mismatch")
            with temporary.open("xb") as stream:
                while True:
                    chunk = response.read(2**20)
                    if not chunk:
                        break
                    size += len(chunk)
                    require(size <= asset["size"], "Response exceeded predeclared byte bound")
                    stream.write(chunk)
                    sha256.update(chunk)
                    md5.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        require(size == asset["size"], "Truncated HTTP asset")
        require(md5.hexdigest() == expected_md5(asset), "Downloaded source MD5 mismatch")
        temporary.rename(target)
        sync_directory(target.parent)
        return {"bytes": size, "sha256": sha256.hexdigest(), "md5": md5.hexdigest(),
                "etag": asset["etag"], "source_url": asset["url"], "path": str(target),
                "if_match_verified": True, "content_length_verified": True}
    finally:
        temporary.unlink(missing_ok=True)


def contained(path, parent):
    path, parent = Path(path), Path(parent).resolve()
    require(not path.is_symlink(), f"Symlink is not supported: {path}")
    resolved = path.resolve()
    require(resolved.is_relative_to(parent) and resolved != parent, f"Path escaped parent: {path}")
    return resolved


def identity(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino]


def matches_identity(path, expected):
    path = Path(path)
    return path.is_dir() and not path.is_symlink() and identity(path) == expected


def qa_record(product):
    from pixelqa_scene_metadata import qa_scene_metadata
    return qa_scene_metadata(product, ROOT)


def verify_qa(record, original):
    require(all(record.get(key) == original.get(key) for key in STABLE_METADATA), "Original metadata identity/grid/scale changed")
    require(record["metadata_usable"] is True and record["issues"] == [], "Frozen QA prerequisites are incomplete")
    require(record["cloud_model_usable"] is True and record["shadow_geometry_usable"] is True, "Cloud/shadow prerequisites are incomplete")
    require(record["missing_cloud_bands"] == [], "Missing cloud-model band")
    require(all(record["missing_quality_masks"][kind] == [] for kind in QUALITY_TYPES), "Missing source quality mask")
    require(all(len(record["quality_band_paths"][kind]) == 11 and all(record["quality_band_paths"][kind].values())
                for kind in QUALITY_TYPES), "Exactly 44 source quality masks are required")
    require(record["crs"] == "EPSG:32636" and record["quantification_value"] == 10000.0
            and record["nonzero_offset"] is False, "Frozen CRS/scale/offset contract changed")


def validate_protocol(protocol):
    require(protocol["schema_version"] == 10 and protocol["resources"] == RESOURCES, "Protocol/resource contract changed")
    scenes = protocol["scenes"]
    require([scene["scene_id"] for scene in scenes] == list(SCENE_IDS), "Fixed pair/order changed")
    require(sum(asset["size"] for scene in scenes for asset in scene["missing_assets"]) == SUPPLEMENT_BYTES,
            "Supplement byte budget changed")
    for scene, day in zip(scenes, (26, 31)):
        require(len(scene["old_assets"]) == 10 and len(scene["missing_assets"]) == 48, "Asset count changed")
        prefix = f"tiles/36/U/XB/2018/7/{day}/0/"
        expected = {prefix + band + ".jp2" for band in EXTRA_BANDS}
        expected |= {prefix + f"qi/MSK_{kind}_{band}.gml" for kind in QUALITY_TYPES for band in QUALITY_BANDS}
        require({asset["key"] for asset in scene["missing_assets"]} == expected, "Supplement asset matrix changed")
        old_paths = [asset["path"] for asset in scene["old_assets"]]
        require(len(set(old_paths)) == 10, "Duplicate original asset")
        targets = []
        for asset in scene["missing_assets"]:
            expected_md5(asset)
            require(isinstance(asset["size"], int) and asset["size"] > 0, "Invalid source size")
            require(asset["url"] == "https://sentinel-s2-l1c.s3.amazonaws.com/" + asset["key"], "Source URL changed")
            target = Path(asset["target_relpath"])
            require(not target.is_absolute() and ".." not in target.parts and target.parts[0] == scene["scene_id"] + ".SAFE", "Unsafe target path")
            require(target.name == Path(asset["key"]).name or (asset["key"].endswith(".jp2")
                    and target.name.endswith("_" + Path(asset["key"]).name)), "Source/target filename mismatch")
            targets.append(str(target))
        require(len(set(targets)) == 48, "Duplicate supplemental target")
        for asset in scene["old_assets"]:
            expected_md5(asset)
            require(re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]) is not None, "Invalid original SHA256")


def save_journal(transaction_root, journal):
    atomic_json(Path(transaction_root) / "journal.json", journal)


def recover_transaction(transaction_root):
    """Restore incomplete publication using inode identities, without deleting trees."""
    transaction_root = Path(transaction_root).resolve()
    journal_path = transaction_root / "journal.json"
    if not journal_path.exists():
        require(not any(transaction_root.iterdir()) if transaction_root.exists() else True, "Unrecognized transaction directory")
        return None
    journal = read(journal_path)
    require(journal["schema_version"] == 1 and journal["transaction_root"] == str(transaction_root), "Unrecognized transaction journal")
    require(journal.get("production") is True, "Transaction is not bound to the fixed Sentinel-2 source root")
    require(journal["state"] in ("preparing", "prepared", "publishing", "published", "complete", "rolled_back"), "Unrecognized transaction state")
    if journal["state"] in ("complete", "rolled_back"):
        return journal
    generation = contained(journal["generation"], transaction_root)
    for item in reversed(journal["scenes"]):
        final = contained(item["final"], ROOT / "Datasets/Sentinel-2")
        backup = contained(item["backup"], generation)
        quarantine = contained(item["quarantine"], generation)
        if matches_identity(final, item.get("prepared_identity")):
            require(not quarantine.exists(), "Rollback quarantine already occupied")
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            final.rename(quarantine)
            sync_directory(final.parent)
            sync_directory(quarantine.parent)
        if backup.exists():
            require(matches_identity(backup, item["original_identity"]), "Original backup identity changed")
            require(not final.exists(), "Refusing to replace an unexpected final tree during rollback")
            backup.rename(final)
            sync_directory(backup.parent)
            sync_directory(final.parent)
        require(matches_identity(final, item["original_identity"]), "Original tree unavailable or changed; manual recovery required")
    journal["state"] = "rolled_back"
    save_journal(transaction_root, journal)
    return journal


def prepare_transaction(protocol, transaction_root, sentinel_dir, qa_reader=None, downloader=None):
    qa_reader, downloader = qa_reader or qa_record, downloader or download_asset
    transaction_root, sentinel_dir = Path(transaction_root).resolve(), Path(sentinel_dir).resolve()
    require(sentinel_dir == (ROOT / "Datasets/Sentinel-2").resolve(), "Only the fixed Sentinel-2 source root is supported")
    previous = recover_transaction(transaction_root)
    require(previous is None or previous["state"] != "complete", "Transaction already completed")
    transaction_root.mkdir(parents=True, exist_ok=True)
    require(transaction_root.stat().st_dev == sentinel_dir.stat().st_dev, "Transaction and scenes must use the same filesystem")
    generation = transaction_root / uuid.uuid4().hex
    generation.mkdir()
    journal = {"schema_version": 1, "transaction_root": str(transaction_root), "generation": str(generation),
               "state": "preparing", "production": sentinel_dir == (ROOT / "Datasets/Sentinel-2").resolve(),
               "scenes": [], "download_receipts": [], "original_receipts": []}
    for scene in protocol["scenes"]:
        final = contained(sentinel_dir / scene["scene_id"], sentinel_dir)
        require(final.is_dir(), f"Original product is missing: {final}")
        journal["scenes"].append({"scene_id": scene["scene_id"], "final": str(final),
                                  "prepared": str(generation / "prepared" / scene["scene_id"]),
                                  "backup": str(generation / "backups" / scene["scene_id"]),
                                  "quarantine": str(generation / "rolled_back_new" / scene["scene_id"]),
                                  "original_identity": identity(final)})
    save_journal(transaction_root, journal)
    try:
        # Check every original against V9 SHA256 AND independently pinned source MD5.
        for scene, item in zip(protocol["scenes"], journal["scenes"]):
            final, prepared = Path(item["final"]), Path(item["prepared"])
            item["original_metadata"] = qa_reader(final)
            for asset in scene["old_assets"]:
                source = contained(ROOT / asset["path"], final)
                result = verify_asset(source, asset)
                journal["original_receipts"].append({**asset, **result, "scene_id": scene["scene_id"]})
            # Refuse symlinks; retain any unrelated user files without rewriting them.
            require(not any(path.is_symlink() for path in final.rglob("*")), "Symlink in original product")
            prepared.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(final, prepared, copy_function=os.link)
            item["prepared_identity"] = identity(prepared)
            save_journal(transaction_root, journal)
            for asset in scene["missing_assets"]:
                target = contained(prepared / asset["target_relpath"], prepared)
                require(not target.exists(), f"Supplement target already exists; refusing overwrite: {target}")
                require(shutil.disk_usage(transaction_root).free >= MIN_DISK_BYTES + asset["size"],
                        "Supplement would breach the 2 GiB free disk reserve")
                receipt = downloader(asset, target)
                verify_asset(target, asset, receipt["sha256"])
                journal["download_receipts"].append({**receipt, "scene_id": scene["scene_id"], "target_relpath": asset["target_relpath"]})
                save_journal(transaction_root, journal)
            for asset in scene["old_assets"]:
                original = contained(ROOT / asset["path"], final)
                verify_asset(prepared / original.relative_to(final), asset)
            item["prepared_metadata"] = qa_reader(prepared)
            verify_qa(item["prepared_metadata"], item["original_metadata"])
        journal["state"] = "prepared"
        save_journal(transaction_root, journal)
        return journal
    except BaseException:
        recover_transaction(transaction_root)
        raise


def publish_transaction(transaction, after_replace=None, qa_reader=None):
    qa_reader = qa_reader or qa_record
    transaction_root = Path(transaction["transaction_root"])
    require(transaction["state"] == "prepared", "Only a fully prepared pair may publish")
    transaction["state"] = "publishing"
    save_journal(transaction_root, transaction)
    try:
        for index, item in enumerate(transaction["scenes"]):
            final, prepared, backup = Path(item["final"]), Path(item["prepared"]), Path(item["backup"])
            require(matches_identity(final, item["original_identity"]), "Original tree identity changed before publication")
            require(matches_identity(prepared, item["prepared_identity"]), "Prepared tree identity changed before publication")
            require(not backup.exists(), "Refusing to replace backup")
            backup.parent.mkdir(parents=True, exist_ok=True)
            final.rename(backup)
            sync_directory(final.parent)
            sync_directory(backup.parent)
            prepared.rename(final)
            sync_directory(prepared.parent)
            sync_directory(final.parent)
            if after_replace:
                after_replace(index)
        for item in transaction["scenes"]:
            item["final_metadata"] = qa_reader(Path(item["final"]))
            verify_qa(item["final_metadata"], item["original_metadata"])
        transaction["state"] = "published"
        save_journal(transaction_root, transaction)
        return transaction
    except BaseException:
        recover_transaction(transaction_root)
        raise


def verify_final_assets(protocol, transaction):
    receipts = {(r["scene_id"], r["target_relpath"]): r for r in transaction["download_receipts"]}
    for scene, item in zip(protocol["scenes"], transaction["scenes"]):
        final, backup = Path(item["final"]), Path(item["backup"])
        require(matches_identity(final, item["prepared_identity"]) and matches_identity(backup, item["original_identity"]), "Published/backup tree identity changed")
        for asset in scene["old_assets"]:
            original = contained(ROOT / asset["path"], final)
            verify_asset(original, asset)
            verify_asset(backup / original.relative_to(final), asset)
        for asset in scene["missing_assets"]:
            receipt = receipts[(scene["scene_id"], asset["target_relpath"])]
            verify_asset(contained(final / asset["target_relpath"], final), asset, receipt["sha256"])
        verify_qa(qa_record(final), item["original_metadata"])


def worker(args, output):
    protocol = read(args.protocol)
    validate_protocol(protocol)
    for role, source in protocol["source_files"].items():
        path = contained(ROOT / source["path"], ROOT)
        require(sha(path) == source["sha256"], f"Source receipt changed: {role}")
    for name, checksum in {**protocol["immutable_sources_sha256"], **protocol.get("implementation_sha256", {})}.items():
        require(sha(ROOT / "Solutions" / name) == checksum, f"Frozen implementation changed: {name}")
    transaction_root = contained(args.transaction_root, ROOT / "Datasets/ValidationPilot")
    require(not transaction_root.is_relative_to(ROOT / "Datasets/Sentinel-2"), "Transaction must remain outside scene discovery")
    disk_parent = transaction_root.parent
    while not disk_parent.exists():
        disk_parent = disk_parent.parent
    free_before = shutil.disk_usage(disk_parent).free
    require(free_before >= MIN_DISK_BYTES + SUPPLEMENT_BYTES, "Supplement requires 2 GiB free disk space plus its download budget")
    # Lock is adjacent: recovery can distinguish an owned transaction from an arbitrary directory.
    transaction_root.parent.mkdir(parents=True, exist_ok=True)
    with (transaction_root.parent / (transaction_root.name + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = recover_transaction(transaction_root)
        if previous and previous["state"] == "complete":
            require(previous["protocol_sha256"] == sha(args.protocol), "Completed transaction belongs to a different protocol")
            transaction = previous
            verify_final_assets(protocol, transaction)
            completion = read(Path(transaction["generation"]) / "completion.json")
        else:
            transaction = prepare_transaction(protocol, transaction_root, ROOT / "Datasets/Sentinel-2")
            transaction["protocol_sha256"] = sha(args.protocol)
            save_journal(transaction_root, transaction)
            try:
                publish_transaction(transaction)
                verify_final_assets(protocol, transaction)
                require(len(transaction["download_receipts"]) == 96 and sum(r["bytes"] for r in transaction["download_receipts"]) == SUPPLEMENT_BYTES,
                        "Actual network asset/byte budget differs from predeclared supplement")
                require(shutil.disk_usage(transaction_root).free >= MIN_DISK_BYTES, "Free disk reserve fell below 2 GiB")
                completion = {"protocol_sha256": transaction["protocol_sha256"], "source_checks": transaction["download_receipts"],
                              "original_checks": transaction["original_receipts"], "new_assets": 96, "original_assets_preserved": 20,
                              "bytes_transferred": SUPPLEMENT_BYTES, "disk_free_bytes_before": free_before,
                              "disk_free_bytes_after": shutil.disk_usage(transaction_root).free,
                              "final_scenes": [item["final_metadata"] for item in transaction["scenes"]],
                              "original_backups": [item["backup"] for item in transaction["scenes"]],
                              "satellite_pixel_reads": 0, "model_executions": 0, "labels_approved": 0}
                atomic_json(Path(transaction["generation"]) / "completion.json", completion)
                transaction["state"] = "complete"
                save_journal(transaction_root, transaction)
            except BaseException:
                recover_transaction(transaction_root)
                raise
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "completion.json", completion)
    (output / "report.md").write_text("# Fixed validation source repair\n\nVerified 96 source-pinned supplemental assets (74,541,437 bytes), preserved all 20 original assets and retained original product backups. Both final scenes pass unchanged frozen QA metadata prerequisites. No pixel QA, model execution, labels or accuracy claims.\n")
    import resource
    require(not any(name in sys.modules for name in ("torch", "rasterio", "numpy", "s2cloudless")), "Source completion must remain metadata-only")
    atomic_json(output / "metadata.json", {"execution_status": "completed", "protocol_sha256": sha(args.protocol),
                "implementation_sha256": sha(__file__), "transaction_journal": str(transaction_root / "journal.json"),
                "artifact_sha256": {name: sha(output / name) for name in ("completion.json", "report.md")},
                "satellite_pixel_reads": 0, "model_executions": 0, "labels_approved": 0,
                "worker_high_water_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024), 2)})


def main():
    from validate_sample import THREAD_VARIABLES, run_guarded
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "Docs/Reviews/ValidationSourceRepairV10/protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transaction-root", type=Path, default=ROOT / "Datasets/ValidationPilot/v10/source_transaction")
    parser.add_argument("--max-rss-mib", type=int, default=512)
    parser.add_argument("--min-available-gib", type=float, default=1.5)
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(128 <= args.max_rss_mib <= 512 and args.min_available_gib >= 1.5 and 0 < args.max_seconds <= 300,
            "Retain 512 MiB/1.5 GiB/300 s resource bounds")
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    if args.worker_output:
        os.nice(10)
        worker(args, args.worker_output)
    else:
        run_guarded(args, script_path=__file__, operation="source-repair", gdal_cache_mib=0, warp_memory_mib=0)


if __name__ == "__main__":
    main()
