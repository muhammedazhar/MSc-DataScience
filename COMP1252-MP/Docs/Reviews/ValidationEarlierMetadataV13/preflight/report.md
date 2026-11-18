# Earlier source metadata preflight

metadata_preflight_passed_no_acquisition

- 2018-07-21: remote_metadata_and_asset_listing_passed; 58/58 pinned assets; []
- 2018-07-16: remote_metadata_and_asset_listing_passed; 58/58 pinned assets; []

{
  "measured_free_bytes": 25819086848,
  "complete_payload_bytes": 1050527541,
  "largest_asset_bytes": 120797363,
  "current_metadata_artifact_bytes": 1086316,
  "future_two_copy_staging_plus_temporary_bytes": 2222938761,
  "minimum_free_reserve_bytes": 2147483648,
  "sufficient_for_conservative_future_acquisition": true,
  "imagery_acquisition_executed": false,
  "measured_filesystem_path": "Datasets/Sentinel-2",
  "measured_filesystem_device": 16777233
}

- Listings pin presence/size/ETag; unread JP2/GML payloads are not integrity-verified or HEAD-tested.
- Remote XML/asset evidence is not local frozen QA readiness, pixel clarity or source approval.
- Catalogue L2A asset links were not fetched; explicit L1C provider XML was checked.
- Fixed earlier source dates need new pair lineage and cumulative temporal targets before QA.
