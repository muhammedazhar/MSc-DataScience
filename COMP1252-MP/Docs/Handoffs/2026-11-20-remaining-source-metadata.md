# Remaining July source metadata

V16 is complete. All three remaining July 2018 candidates passed the source XML
and required-file listing checks. Continue on `codex/curated-deforestation-pilot`
and push verified checkpoints to the same branch. The V15 handoff records the
prior cloud exclusions, unchanged grids, geographic group, and frozen QA rules.

| Pre-image | Days before 31 July | Required files listed | Listed payload |
| --- | ---: | ---: | ---: |
| 11 July 2018 | 20 | 58 | 562,911,967 bytes |
| 6 July 2018 | 25 | 58 | 529,319,399 bytes |
| 1 July 2018 | 30 | 58 | 572,015,183 bytes |

All three product and tile XML pairs match the V12 identities for L1C, N0206,
orbit R064, and tile 36UXB. Grid coverage and alignment include both original
crops and their full QA contexts. Quantification is 10000. The product XML
omits offsets, which the frozen reader handles as zero for this baseline.
Every required file has a size and simple MD5 ETag recorded. Each manifest has
11 JP2 bands, 44 per-band quality masks, one cloud mask, and two XML files.
JP2 and GML availability comes from listings. Their readability and pixel
quality remain untested. The 17 L2A asset links in each L1C catalogue entry are
recorded as conflicts. All 12 requests used the explicit L1C bucket.

The live preflight downloaded 1,616,389 bytes of metadata and completed in
4.32 seconds. Peak RSS was 44.64 MiB and available memory stayed above 3.59 GiB
at the sampled precision. The guarded worker ran at nice +10 with one native
thread, a 512 MiB RSS cap, a 1.5 GiB memory reserve, and a 300-second time limit.
The listed payload totals 1,664,246,549 bytes (1.55 GiB). Allow 3.22 GiB for
two copies, the largest temporary file, and metadata, plus a separate 2 GiB
reserve. The source filesystem had 29.05 GiB free at preflight.

All 384 tests passed under the guard in 4.84 seconds. The four added tests
cover exact roster selection, missing and duplicate eligible dates, raw
catalogue hash binding, and preservation of later outcomes after a request
failure. Offline verification replayed all three listings and six XML files
against the saved HTTP receipts and reproduced all 174 manifest entries and
the disk calculation. It checked 60 pinned inputs and the size/mtime receipts
for 232 previous source files. Both prior transactions are complete, all six
prior QA exclusions remain recorded, and zero workers are active.

Protocol checkpoint `e3a5da0` establishes the whole roster before live requests.
Checkpoint `27810c8` pins the tested wrapper and frozen V13 parsing helpers.
The completion checkpoint stores the raw provider responses, request receipts,
manifests, and final integrity check under
`Docs/Reviews/ValidationRemainingMetadataV16/`. The local run is at
`Datasets/ValidationPilot/v16/preflight/`. The two evidence copies match byte for
byte.

V16 ends at metadata compatibility. The next distinct phase would acquire
these three pinned manifests and verify actual local QA prerequisites before
pixel screening. Recheck available disk before that phase and use the guarded
transaction pattern from V14. Derive any new pairs from the original
annotations and recalculate cumulative targets at each pre-date, including the
30-day boundary and the earlier baseline for 1 July. Pixel QA and visual
evidence are still needed to establish a first-visible interval.
