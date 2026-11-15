"""Small synthetic regressions for annotation identities and geographic splits.

Fixtures contain a handful of geometries and metadata dictionaries. The suite
never opens satellite imagery or the real annotation catalogue.
"""

from datetime import date, timedelta
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import unittest


for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "GDAL_NUM_THREADS",
    "OPJ_NUM_THREADS",
):
    os.environ[name] = "1"

from pyproj import Transformer
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, box, mapping
from shapely.ops import transform


SOLUTIONS = Path(__file__).resolve().parents[1] / "Solutions"
sys.path.insert(0, str(SOLUTIONS))
SPEC = importlib.util.spec_from_file_location("annotation_audit_under_test", SOLUTIONS / "audit_annotations.py")
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)

TO_WGS84 = Transformer.from_crs(32636, 4326, always_xy=True).transform


def feature(x=650000, y=5530000, observed="2018-08-05", tile="36UXA", width=40, multi=False):
    geometry = transform(TO_WGS84, box(x-width/2, y-20, x+width/2, y+20))
    if multi:
        geometry = MultiPolygon([geometry])
    return {
        "type": "Feature", "properties": {"tile": tile, "img_date": observed},
        "geometry": mapping(geometry),
    }


def event_identity(event):
    return (event["tile"], event["observation_date"], audit.canonical_geometry(event["geometry"]).wkb)


def temporal_events(features):
    references = {}
    for index, item in enumerate(features):
        properties = item["properties"]
        key = (properties["tile"], properties["img_date"],
               hashlib.sha256(audit.canonical_geometry(item["geometry"]).wkb).hexdigest())
        references[key] = [{"source_id": "fixture-temporal", "kind": "temporal",
                            "source_feature_index": index}]
    events, _ = audit.clean_events(features, references)
    audit.assign_spatial_splits(events, patch_size=64, seed=42)
    return events


def scene(observed, usable=True):
    return {
        "scene_id": f"fixture-{observed}", "tile": "36UXA",
        "acquisition_date": observed, "path": f"fixtures/{observed}.SAFE",
        "metadata_usable": usable, "issues": [] if usable else ["missing_band_B11"],
        "crs": "EPSG:32636",
        "grid_10m": {"ulx": 600000, "uly": 5600040, "width": 10980, "height": 10980,
                     "bounds": [600000, 5490240, 709800, 5600040]},
    }


class AnnotationIdentityTests(unittest.TestCase):
    def test_roundoff_matching_accepts_numeric_noise_but_rejects_real_shift_even_for_tiny_area(self):
        original = box(650000, 5530000, 650000.1, 5530000.1)
        roundoff = translate(original, xoff=1e-7, yoff=-1e-7)
        shifted = translate(original, xoff=.002)
        # An area-only comparison would accept this real 2 mm displacement.
        self.assertLess(original.symmetric_difference(shifted).area, .01)
        self.assertTrue(audit.roundoff_match(original, roundoff))
        self.assertTrue(audit.roundoff_match(roundoff, original))
        self.assertFalse(audit.roundoff_match(original, shifted))
        self.assertFalse(audit.roundoff_match(shifted, original))

    def test_single_polygon_and_singlepart_multipolygon_have_same_canonical_geometry(self):
        polygon = Polygon([(0, 0, 10), (1, 0, 20), (1, 1, 30), (0, 1, 40), (0, 0, 10)])
        reverse = Polygon(list(polygon.exterior.coords)[::-1])
        first = audit.canonical_geometry(mapping(polygon))
        second = audit.canonical_geometry(MultiPolygon([reverse]))
        self.assertEqual(first.geom_type, "Polygon")
        self.assertFalse(first.has_z)
        self.assertEqual(first.wkb, second.wkb)

    def test_duplicate_events_keep_every_source_index_and_stable_ids(self):
        original = feature()
        duplicate = feature(multi=True)
        different = feature(x=655000)
        events, _ = audit.clean_events([original, duplicate, different])
        reordered, _ = audit.clean_events([different, duplicate, original])
        self.assertEqual(len(events), 2)
        self.assertEqual({event_identity(e): e["event_id"] for e in events},
                         {event_identity(e): e["event_id"] for e in reordered})
        duplicate_event = next(e for e in events if len(e["source_indices"]) == 2)
        self.assertEqual(sorted(duplicate_event["source_indices"]), [0, 1])
        self.assertEqual(sorted(i for e in events for i in e["source_indices"]), [0, 1, 2])

    def test_conflicting_observation_dates_are_preserved_at_both_locations(self):
        features = [
            feature(observed="2017-09-19"),
            feature(observed="2018-09-19", multi=True),
            feature(x=655000, observed="2019-08-10"),
            feature(x=655000, observed="2019-08-30", multi=True),
        ]
        events, issues = audit.clean_events(features)
        self.assertEqual(len(events), 4)
        self.assertEqual(len({e["location_id"] for e in events}), 2)
        self.assertTrue(all("conflicting_observation_dates" in e["flags"] for e in events))
        self.assertEqual(issues, [])

    def test_null_geometry_and_properties_are_preserved_as_issues_instead_of_crashing(self):
        missing_geometry = feature()
        missing_geometry["geometry"] = None
        missing_properties = feature()
        missing_properties["properties"] = None
        events, issues = audit.clean_events([missing_geometry, missing_properties, None])
        self.assertEqual(events, [])
        self.assertEqual({issue["source_feature_index"] for issue in issues}, {0, 1, 2})


class GeographicSplitTests(unittest.TestCase):
    def test_transitive_crop_overlap_groups_dates_and_different_tiles_together(self):
        # 640 m crops overlap A-B and B-C. A and C do not touch directly.
        features = [
            feature(x=650000, observed="2018-07-31"),
            feature(x=650600, observed="2018-08-05", tile="36UYA"),
            feature(x=651200, observed="2018-08-10"),
            feature(x=658000),
        ]
        events, _ = audit.clean_events(features)
        audit.assign_spatial_splits(events, patch_size=64, seed=42)
        chain = [e for e in events if e["source_indices"][0] < 3]
        self.assertEqual(len({e["group_id"] for e in chain}), 1)
        self.assertEqual(len({e["split"] for e in chain}), 1)
        distant = next(e for e in events if e["source_indices"] == [3])
        self.assertNotEqual(chain[0]["group_id"], distant["group_id"])

    def test_full_annotation_extent_prevents_leakage_between_disjoint_crops(self):
        # The large annotation extends into B's crop even though the two crop
        # centres are 900 m apart. The whole event must stay in a single group.
        events, _ = audit.clean_events([feature(width=1600), feature(x=650900)])
        audit.assign_spatial_splits(events, patch_size=64, seed=42)
        self.assertEqual(events[0]["group_id"], events[1]["group_id"])
        self.assertEqual(events[0]["split"], events[1]["split"])

    def test_group_ids_and_split_assignments_are_stable_when_input_is_reordered(self):
        features = [feature(x=640000+2500*(i % 5), y=5520000+2500*(i // 5)) for i in range(20)]
        first, _ = audit.clean_events(features)
        second, _ = audit.clean_events(list(reversed(features)))
        audit.assign_spatial_splits(first, patch_size=64, seed=42)
        audit.assign_spatial_splits(second, patch_size=64, seed=42)
        snapshot = lambda events: {e["event_id"]: (e["group_id"], e["split"]) for e in events}
        self.assertEqual(snapshot(first), snapshot(second))


class CandidateManifestTests(unittest.TestCase):
    def test_only_5_to_30_calendar_day_pairs_are_enumerated_including_both_boundaries(self):
        observed = date(2018, 8, 5)
        for interval, expected in ((4, 0), (5, 1), (30, 1), (31, 0)):
            with self.subTest(interval_days=interval):
                events = temporal_events([feature(observed=observed.isoformat())])
                before = (observed-timedelta(days=interval)).isoformat()
                pairs, manifest = audit.enumerate_candidates(
                    events, [scene(before), scene(observed.isoformat())], patch_size=64,
                )
                self.assertEqual(len(pairs), expected)
                self.assertEqual(len(manifest[0]["candidate_pair_ids"]), expected)
                if expected:
                    self.assertEqual(pairs[0]["interval_days"], interval)
                    self.assertEqual(pairs[0]["status"], "candidate_for_pixel_review")
                    self.assertEqual(pairs[0]["blocking_reasons"], [])
                    self.assertFalse(pairs[0]["training_approved"])
                else:
                    self.assertIn("no_5_to_30_day_bracketing_pair", manifest[0]["review_reasons"])

    def test_missing_scene_metadata_is_explicit_and_does_not_silently_remove_the_pair(self):
        events = temporal_events([feature()])
        pairs, manifest = audit.enumerate_candidates(
            events, [scene("2018-07-31"), scene("2018-08-05", usable=False)], patch_size=64,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["status"], "blocked_for_temporal_training")
        self.assertIn("scene_metadata_incomplete", pairs[0]["blocking_reasons"])
        self.assertEqual(manifest[0]["candidate_pair_ids"], [pairs[0]["pair_id"]])
        self.assertFalse(pairs[0]["training_approved"])

    def test_missing_observations_remain_explicit_in_annotation_manifest(self):
        events = temporal_events([feature()])
        pairs, manifest = audit.enumerate_candidates(events, [], patch_size=64)
        self.assertEqual(pairs, [])
        self.assertEqual(manifest[0]["candidate_pair_ids"], [])
        self.assertIn("missing_pre_observation", manifest[0]["review_reasons"])
        self.assertIn("missing_post_observation", manifest[0]["review_reasons"])
        self.assertFalse(manifest[0]["training_approved"])

    def test_unverified_neighbouring_interval_label_blocks_a_temporal_anchor(self):
        for neighbour_date, should_block in (("2018-08-05", True), ("2018-07-25", False)):
            with self.subTest(neighbour_date=neighbour_date):
                events = temporal_events([feature(), feature(x=650200, observed=neighbour_date)])
                anchor = next(e for e in events if e["source_indices"] == [0])
                neighbour = next(e for e in events if e["source_indices"] == [1])
                neighbour["temporal_source_ids"] = []
                neighbour["source_memberships"] = [{"source_id": "fixture-baseline", "kind": "baseline"}]
                neighbour["flags"] = ["baseline_only_provenance"]
                pairs, _ = audit.enumerate_candidates(
                    events, [scene("2018-07-31"), scene("2018-08-05")], patch_size=64,
                )
                pair = next(p for p in pairs if anchor["event_id"] in p["anchor_event_ids"])
                self.assertEqual("unverified_temporal_label_in_crop" in pair["blocking_reasons"], should_block)
                self.assertIn(neighbour["event_id"], pair["context_event_ids"])
                self.assertNotIn(neighbour["event_id"], pair["label_event_ids_by_post_date"])
                self.assertFalse(pair["training_approved"])

    def test_unresolved_original_annotation_blocks_only_intersecting_candidate_crops(self):
        events = temporal_events([feature()])
        for unresolved_geometry, should_block in (
            (box(650050, 5529950, 650080, 5529980), True),
            (box(655000, 5529950, 655030, 5529980), False),
        ):
            with self.subTest(intersects_crop=should_block):
                pairs, _ = audit.enumerate_candidates(
                    events, [scene("2018-07-31"), scene("2018-08-05")], patch_size=64,
                    source_exclusions=[{"_projected_geometry": unresolved_geometry}],
                )
                self.assertEqual(len(pairs), 1)
                self.assertEqual("unresolved_original_annotation_in_crop" in pairs[0]["blocking_reasons"], should_block)


if __name__ == "__main__":
    unittest.main()
