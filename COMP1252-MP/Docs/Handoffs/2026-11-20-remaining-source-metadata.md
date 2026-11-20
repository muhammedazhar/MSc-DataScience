# Remaining July source metadata

V16 implementation is tested and pinned. Live checks and verification are pending.
Continue on `codex/curated-deforestation-pilot` and push verified checkpoints to
the same branch. Read the linked V15 handoff for the original grids, existing
cloud exclusions, and Mac capacity limits.

The fixed roster is 11, 6, and 1 July 2018, with intervals of 20, 25, and 30 days
to the 31 July post-image. These are the three remaining nominal dates from
V12. The protocol pins their catalogue IDs and genuine L1C provider keys.
Catalogue cloud percentages play no part in selection. Record an outcome for
each fixed date and stop there.

Use up to four requests per source for an S3 listing, product XML headers,
product XML, and tile XML. Reuse the frozen V13 checks for source identity,
grid and context coverage, radiometry, solar angles, and all 58 required files.
The disk estimate allows two copies, the largest temporary file, metadata,
and a separate 2 GiB reserve. This phase covers metadata only.

The existing guard allows one nice +10 CPU worker, one native thread, 512 MiB
RSS, 1.5 GiB available system memory, and a maximum of 300 seconds. Use 120
seconds for tests. All 384 tests passed in 4.84 seconds. Peak sampled RSS was
370.6 MiB. Available system memory stayed at or above 3.34 GiB. The four added
tests cover the exact dates and identities, missing and duplicate eligible dates,
raw catalogue hash binding, and retaining all outcomes after a request failure.

See `Docs/Reviews/ValidationRemainingMetadataV16/protocol.json` for the fixed
inputs and request limits.
