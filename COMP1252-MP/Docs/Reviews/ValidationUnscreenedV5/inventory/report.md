# Unscreened validation inventory

{
  "pairs": 606,
  "anchor_pairs": 630,
  "events": 67,
  "groups": 9,
  "pair_status_counts": {
    "blocked_for_temporal_training": 606
  },
  "pair_blocking_reason_counts_overlapping": {
    "baseline_only_provenance": 79,
    "crop_crosses_tile_edge": 347,
    "post_scene_does_not_cover_crop": 347,
    "unverified_temporal_label_in_crop": 259
  },
  "anchor_pairs_with_interior_at_least_four": 37,
  "supported_events": 6,
  "supported_groups": 4,
  "anchor_pairs_with_temporal_clipping": 111,
  "anchor_pairs_footprint_isolated": 630,
  "anchor_pairs_with_later_metadata_opportunity": 566,
  "anchor_pairs_in_exposed_groups": 203,
  "passes_metadata_geometry_gates": 0,
  "qa_status": "unknown_no_published_qa",
  "approved_labels": 0
}

Every original blocker is retained. Quality stays unknown. No shortlist or approved labels.
Geometry can be measurable outside available scene coverage; support never overrides source/provenance blockers.
