# Earlier source acquisition

V14 acquires the two exact V13 manifests, 21 July then 16 July 2018. No alternate
dates or catalogue cloud ranking. Start with
`Docs/Handoffs/2026-11-18-earlier-source-acquisition.md` for current progress;
the implementation is not proof that an operation completed.

`Solutions/acquire_earlier_validation_sources.py` uses one guarded low-priority
worker at a time. Download mode requires scene index 0 or 1 and a new report
output path. Publish mode requires both fully staged scenes. Verify mode is
read-only and available only after a complete transaction, particularly if a
guard/report interruption followed successful persistent publication.

Durable state is `Datasets/ValidationPilot/v14/source_transaction/journal.json`,
outside report staging and scene discovery. Scene files remain there until both
pass actual unchanged frozen QA metadata prerequisites. Each object is pinned
by explicit L1C URL, size and MD5 ETag, downloaded with If-Match and verified
using MD5/SHA256. Never follow the L2A hrefs in the saved L1C STAC records.

On interruption, retain verified complete files and the journal. Resume using
the SAME protocol, transaction and scene index, with a NEW report output path.
Complete files are rehashed against source pins and saved SHA; an owned partial
is removed only when its exact allowed path and device/inode match. Unknown
staged files, symlinks, altered journals or ambiguous ownership cause refusal.
Never delete the transaction or source files just to bypass those checks.

Publication uses native no-replace directory renames, inode ownership and an
exclusive lock. An ordinary failure restores owned published trees to staging;
a process crash can resume the owned publication. Existing user/old source trees
are never replaced. This is recoverable publication, not an atomic pair swap.
The persistent completion receipt is written before the final journal state.

Use unchanged 512 MiB RSS / 1.5 GiB available memory reserve / 300 s guard, nice
+10, one native thread, no GPU; tests use 120 s. Require 2 GiB free disk reserve,
conservative live staging capacity before operations and space before each file/
stream chunk. Source data stays local; GitHub stores code and bounded receipts.
Local metadata readiness does not establish pixel clarity, forest labels, full
27-channel eligibility or unseen accuracy. New temporal pairs/QA come later.
