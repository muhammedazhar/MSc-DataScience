# Remaining July source acquisition

The V17 runner and protocol are tested and pinned. Download and publication
are pending. All 388 tests passed in 4.58 seconds, with a sampled peak of
346.4 MiB and at least 3.07 GiB available memory. The four new offline tests
cover the complete roster, missing and duplicate scenes, duplicate and foreign
asset mappings, and payload totals.
Continue on `codex/curated-deforestation-pilot`. See the V16 metadata handoff
for the three pinned manifests and the V15 handoff for frozen QA constraints.

The acquisition roster is 11, 6, and 1 July 2018. It contains 174 files and
1,664,246,549 bytes. Initial capacity was 28.98 GiB free on the source
filesystem and 3.39 GiB system memory available. Recheck capacity for each
guarded run. The disk allowance is 3.22 GiB plus a separate 2 GiB reserve.

Use the existing V14 transfer, recovery, and publication functions with a
separate journal under `Datasets/ValidationPilot/v17/source_transaction`.
Run one low-priority worker at a time with a single native thread, 512 MiB
RSS, 1.5 GiB available memory reserve, and a 300-second cap. Download and
validate each scene in staging. Keep reports and verified receipts after each
scene. Publish the roster after all three scenes pass local metadata checks.

Protocol evidence is under `Docs/Reviews/ValidationRemainingAcquisitionV17/`.

- 2018-07-11 is staged with all 58 files verified. The guarded run took 52.74 seconds and peaked at 63.03 MiB RSS. It downloaded 58 files and reused 0 verified files. See the `download_11` receipt and journal snapshot.

- 2018-07-06 is staged with all 58 files verified. The guarded run took 47.42 seconds and peaked at 59.58 MiB RSS. It downloaded 58 files and reused 0 verified files. See the `download_06` receipt and journal snapshot.

- 2018-07-01 is staged with all 58 files verified. The guarded run took 47.75 seconds and peaked at 66.94 MiB RSS. It downloaded 58 files and reused 0 verified files. See the `download_01` receipt and journal snapshot.
