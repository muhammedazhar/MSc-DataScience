"""Load the reviewed partial-supervision pilot without altering its features.

The exported TIFFs already contain scaled reflectance, signed vegetation
indices and signed post-minus-pre differences. Only non-finite values at
unreviewed pixels may be replaced with zero before a convolution sees them.
"""

from collections import Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT/"Datasets/CuratedPilot/v1/dataset"
BASE_CHANNELS = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12", "NDVI", "NDMI")
CHANNELS = tuple(f"{period}_{name}" for period in ("pre", "post", "difference") for name in BASE_CHANNELS)
ARTIFACTS = {"input_features": "input_features.tif", "supervision_mask": "supervision_mask.tif",
             "valid_mask": "valid_mask.tif", "metadata": "metadata.json"}
REQUIRED_CHECKS = ("explicit_hash_bound_reviews_only", "all_rejections_retained",
                   "exact_frozen_pair_joins", "train_split_only", "accepted_events_and_groups_unique",
                   "no_held_out_feature_footprint_overlap", "all_supervision_partial_and_reviewed",
                   "source_statistics_unchanged")
COUNT_FIELDS = ("supervised_positive_pixels", "supervised_stable_pixels", "ignored_pixels")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path, limit=4*2**20):
    path = Path(path)
    _require(path.is_file() and path.stat().st_size <= limit, f"Missing or oversized JSON metadata: {path}")
    value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Non-finite JSON: {value}")))
    _require(isinstance(value, dict), "Dataset metadata must contain a JSON object")
    return value


def _digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _count(value, name):
    _require(type(value) is int and value >= 0, f"{name} must be a non-negative integer")
    return value


def _identifier(value, prefix):
    _require(isinstance(value, str) and re.fullmatch(prefix+r"-[A-Za-z0-9-]+", value), f"Invalid {prefix} identifier")
    return value


def _validate_grid(grid):
    _require(isinstance(grid, dict), "A frozen geographic grid is required")
    _require(type(grid.get("width")) is int and type(grid.get("height")) is int
             and (grid["width"], grid["height"]) == (64, 64), "The reviewed grid must remain 64 by 64")
    transform = grid.get("transform")
    _require(isinstance(transform, list) and len(transform) == 6
             and all(type(v) in (int, float) and math.isfinite(v) for v in transform), "Invalid frozen transform")
    a, b, west, d, e, north = transform
    _require((a, b, d, e) == (10, 0, 0, -10), "The reviewed grid must remain north-up at 10 metres")
    _require(grid.get("bounds") == [west, north-640, west+640, north], "Grid bounds and transform disagree")
    _require(isinstance(grid.get("crs"), str) and bool(grid["crs"]), "Frozen CRS is missing")


class ReviewedPilotDataset(Dataset):
    """Strict, lazy TIFF reader for the accepted training-only pilot.

    ``labels`` retain uint8 0/1/255. ``valid`` is exactly labels != 255; callers
    must use it for loss and metrics. No normalization or augmentation is
    performed. Artifact hashes and metadata are checked before arrays are read.
    """

    def __init__(self, dataset_root=DEFAULT_DATASET):
        self.dataset_root = Path(dataset_root).resolve()
        self.metadata_path = self.dataset_root/"metadata.json"
        self.metadata = _json(self.metadata_path)
        self.metadata_sha256 = _digest(self.metadata_path)
        checks = self.metadata.get("checks", {})
        _require(isinstance(checks, dict) and all(checks.get(key) is True for key in REQUIRED_CHECKS), "Dataset publication checks are incomplete")
        _require(self.metadata.get("ready_for_partial_pilot_training") is True, "The dataset is not approved for partial pilot training")
        self.qa_policy_id = self.metadata.get("qa_policy_id")
        _require(_sha(self.qa_policy_id), "The frozen QA policy identity is invalid")
        self._hashes = self.metadata.get("output_sha256_excluding_this_metadata")
        _require(isinstance(self._hashes, dict) and bool(self._hashes), "Published artifact hashes are missing")
        for name, checksum in self._hashes.items():
            self._path(name)
            _require(_sha(checksum), "Published artifact checksum is invalid")
        manifest = self._checked_path("training_manifest.jsonl")
        self.manifest_sha256 = _digest(manifest)
        rows = []
        with manifest.open("rb") as stream:
            for index in range(65):
                line = stream.readline(256*1024+1)
                if not line:
                    break
                _require(index < 64 and 0 < len(line) <= 256*1024 and bool(line.strip()), "Manifest record/line bound exceeded")
                row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Non-finite JSON: {value}")))
                _require(isinstance(row, dict), "Manifest records must be JSON objects")
                rows.append(row)
        _require(bool(rows) and len(rows) == self.metadata.get("accepted_example_count"), "Accepted example coverage disagrees with the manifest")
        _require(self.metadata.get("accepted_event_count") == len(rows)
                 and self.metadata.get("accepted_geographic_group_count") == len(rows), "Accepted event/group counts disagree")
        self.records = tuple(rows)
        self._paths, self._examples = [], []
        events, groups, pairs = set(), set(), set()
        declared = Counter()
        for row in self.records:
            _require(row.get("split") == "train" and row.get("training_approved") is True, "Only explicitly approved training examples can be loaded")
            for field, prefix, seen in (("event_id", "event", events), ("group_id", "group", groups), ("pair_id", "pair", pairs)):
                identifier = _identifier(row.get(field), prefix)
                _require(identifier not in seen, f"Repeated {field} in the reviewed pilot")
                seen.add(identifier)
            _require(row.get("qa_policy_id") == self.qa_policy_id, "Manifest combines different QA policies")
            _validate_grid(row.get("grid"))
            _require(date.fromisoformat(row["pre_date"]) < date.fromisoformat(row["post_date"]), "Pair observations are reversed")
            evidence = row.get("visibility_evidence", {})
            _require(isinstance(evidence, dict) and evidence.get("decision_status") == "accepted"
                     and evidence.get("training_approved") is True, "Manifest lacks accepted visual/temporal evidence")
            paths = row.get("artifact_paths")
            _require(isinstance(paths, dict) and set(paths) == set(ARTIFACTS), "Manifest artifact roles differ from the reviewed export")
            expected_paths = {key: f"examples/{row['event_id']}/{name}" for key, name in ARTIFACTS.items()}
            _require(paths == expected_paths, "An artifact escaped its exact event export directory")
            resolved = {key: self._checked_path(name) for key, name in paths.items()}
            example = _json(resolved["metadata"])
            self._validate_example(row, example)
            self._paths.append(resolved)
            self._examples.append(example)
            for field in COUNT_FIELDS:
                declared[field] += _count(row.get(field), field)
            _require(row["supervised_positive_pixels"] > 0 and row["supervised_stable_pixels"] > 0
                     and sum(row[field] for field in COUNT_FIELDS) == 4096, "A reviewed example has inconsistent partial-label counts")
        _require(dict(declared) == self.metadata.get("supervision_pixel_counts"), "Published aggregate supervision counts disagree")

    def _path(self, name):
        _require(isinstance(name, str) and bool(name), "Artifact paths must be relative strings")
        pure = PurePosixPath(name)
        _require(not pure.is_absolute() and ".." not in pure.parts and "\\" not in name, "Artifact paths must stay inside the dataset")
        path = (self.dataset_root/name).resolve()
        _require(path.is_relative_to(self.dataset_root) and path.is_file(), "Artifact path is missing or escapes the dataset")
        return path

    def _checked_path(self, name):
        path = self._path(name)
        _require(name in self._hashes and _digest(path) == self._hashes[name], f"Published artifact hash changed: {name}")
        return path

    def _validate_example(self, row, example):
        for field in ("pair_id", "group_id", "split", "tile", "pre_date", "post_date", "grid", "qa_policy_id"):
            _require(example.get(field) == row.get(field), f"Example/manifest {field} mismatch")
        _require(row["event_id"] in example.get("anchor_event_ids", []), "Reviewed event is absent from its source anchors")
        _require(tuple(example.get("channels", [])) == CHANNELS, "Feature channel order differs from the reviewed export")
        _require(example.get("training_approved") is True and example.get("difference_direction") == "post minus pre, without clipping", "Example approval or signed-difference convention is missing")
        decision = example.get("decision", {})
        _require(isinstance(decision, dict) and decision.get("visual_review") == decision.get("temporal_label_review") == "accepted"
                 and decision.get("qa_policy_id") == self.qa_policy_id, "Example does not bind accepted visual/temporal reviews")
        local_hashes = example.get("output_sha256", {})
        for role in ("input_features", "supervision_mask", "valid_mask"):
            name = row["artifact_paths"][role]
            _require(local_hashes.get(ARTIFACTS[role]) == self._hashes[name], "Example and dataset artifact hashes disagree")
        for field in COUNT_FIELDS:
            _require(example.get(field) == row[field], "Example/manifest supervision counts disagree")

    def __len__(self):
        return len(self.records)

    def _read_arrays(self, index):
        import numpy as np
        import rasterio
        from affine import Affine

        row = self.records[index]
        paths = {key: self._checked_path(name) for key, name in row["artifact_paths"].items()}
        before = {key: (path.stat().st_size, path.stat().st_mtime_ns) for key, path in paths.items()}
        grid, arrays = row["grid"], {}
        with rasterio.Env(GDAL_CACHEMAX=16*2**20, GDAL_NUM_THREADS="1"):
            for role in ("input_features", "supervision_mask", "valid_mask"):
                with rasterio.open(paths[role]) as source:
                    _require(source.shape == (64, 64) and source.transform == Affine(*grid["transform"])
                             and source.crs == rasterio.crs.CRS.from_user_input(grid["crs"]), "TIFF geographic grid differs from the frozen manifest")
                    if role == "input_features":
                        _require(source.count == 27 and source.dtypes == ("float32",)*27
                                 and source.descriptions == CHANNELS, "TIFF feature shape/type/channel order changed")
                    else:
                        _require(source.count == 1 and source.dtypes == ("uint8",), "TIFF label/valid type or dimensions changed")
                        if role == "supervision_mask":
                            _require(source.nodata == 255, "Supervision TIFF must retain the 255 ignore convention")
                    arrays[role] = source.read()
        _require(before == {key: (path.stat().st_size, path.stat().st_mtime_ns) for key, path in paths.items()}, "Artifact files changed during a sample read")
        features, labels, raw_valid = arrays["input_features"], arrays["supervision_mask"][0], arrays["valid_mask"][0]
        _require(np.isin(labels, (0, 1, 255)).all(), "Supervision labels must remain 0, 1 or 255")
        _require(np.isin(raw_valid, (0, 1)).all(), "The supervision-valid mask must be binary")
        valid = labels != 255
        _require(np.array_equal(raw_valid.astype(bool), valid), "Valid mask must equal labels != 255 exactly")
        actual_counts = {"supervised_positive_pixels": int((labels == 1).sum()),
                         "supervised_stable_pixels": int((labels == 0).sum()), "ignored_pixels": int((labels == 255).sum())}
        _require(all(actual_counts[field] == row[field] for field in COUNT_FIELDS), "TIFF label counts differ from the reviewed manifest")
        finite = np.isfinite(features)
        _require(finite[:, valid].all(), "Supervised pixels contain non-finite source features")
        with np.errstate(invalid="ignore", over="ignore"):
            expected = features[9:18]-features[:9]
        comparable = np.isfinite(expected) & finite[18:]
        _require(np.array_equal(features[18:][comparable], expected[comparable]), "Finite feature differences are not post minus pre")
        replacements = int((~finite).sum())
        if replacements:
            features[~finite] = 0
        return features, labels, valid, replacements

    def __getitem__(self, index):
        import torch

        features, labels, valid, replacements = self._read_arrays(index)
        row = self.records[index]
        return {"features": torch.from_numpy(features), "labels": torch.from_numpy(labels),
                "valid": torch.from_numpy(valid), "event_id": row["event_id"],
                "pair_id": row["pair_id"], "group_id": row["group_id"],
                "nonfinite_replacements": replacements}

    def audit(self):
        """Read samples one at a time and compare actual aggregate supervision."""
        import numpy as np

        totals, replacements, affected = Counter(), 0, 0
        for index in range(len(self)):
            _, labels, _, count = self._read_arrays(index)
            for name, value in (("supervised_positive_pixels", 1), ("supervised_stable_pixels", 0), ("ignored_pixels", 255)):
                totals[name] += int(np.count_nonzero(labels == value))
            replacements += count
            affected += count > 0
        _require(dict(totals) == self.metadata["supervision_pixel_counts"], "Actual aggregate supervision changed")
        return {"example_count": len(self), **dict(totals),
                "nonfinite_feature_values_replaced": replacements,
                "examples_with_nonfinite_replacements": affected,
                "event_ids": [row["event_id"] for row in self.records],
                "group_ids": [row["group_id"] for row in self.records],
                "splits": ["train"], "feature_shape": [27, 64, 64], "channels": list(CHANNELS),
                "qa_policy_id": self.qa_policy_id, "dataset_metadata_sha256": self.metadata_sha256,
                "manifest_sha256": self.manifest_sha256,
                "feature_handling": "Exported finite values unchanged; non-finite ignored inputs replaced with zero only"}
