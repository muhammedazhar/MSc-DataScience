# Earlier pair QA continuation

## Current checkpoint: V15 predeclared; implementation and execution pending

Continue on `codex/curated-deforestation-pilot`; update this handoff, commit and
push each verified logical change to the SAME branch. Read
`2026-11-18-earlier-source-acquisition.md` and its linked V13/V12/V11/older
handoffs for provenance, source integrity and frozen constraints. Last complete
checkpoint `028cbd6` is pushed; both earlier sources are acquired and metadata-ready.

The user approved the next step. Protocol:
`Docs/Reviews/ValidationEarlierScreenV15/protocol.json`. Exactly FOUR new derived
pairs: pre21 July and pre16 July 2018 x the two existing events, fixed31 July post,
original grids/group/split/anchors. One geographic group, not independent locations.
Preserve original blocked36UXA parents and both V11 cloudy26 July source pairs.
No new source dates, replacements, labels, forest predictions, training, accuracy,
manual imagery or 27-channel approval. No V15 pixel work has run at this checkpoint.
Implementation, tests, lineage/geometry, screen and replay remain INCOMPLETE.

Derive new identities with explicit parent hashes, date changes and source lineage.
Update interval_days to10/15. Recompute intersecting context/post/pre temporal
event sets and expanded-interval provenance blockers. Preserve annotation source
tile/source IDs; replacement imagery is36UXB, annotations still originate36UXA.
Recalculate cumulative post-minus-pre masks with frozen helpers, independently
check boolean difference, and retain counts/hashes even if unchanged. Do not reuse
V11's hardcoded dates/derivation/masks or burn only events inside the interval.
Validate entire original train/test conservative footprints and actual13 training
grids before decoding any satellite pixels. Newly introduced provenance blockers
must not be silently removed.

Three source scenes /174 files are used: earlier21/16 July plus original31 July.
Hash actual files against V14/V11 receipts and check actual frozen QA prerequisites.
Preserve all116 original V11 source stats/excluded outcomes, prior manual exposure
and automatic-QA history. Source preparation/geometry reads no satellite pixels.
Then screen all four pairs with unchanged QA, no previews. Eight observation calls,
at most six distinct observations/cloud calls and66 fused-band window requests;
existing post caches may be safely reused. Replay saved caches/QA TIFFs and targets;
this is shared-helper replay, not independent cloud or class truth.

Use unchanged `validate_sample.run_guarded`, ONE nice +10 CPU worker, native
threads1, no GPU,512 MiB RSS,1.5 GiB available-system-memory reserve,300 s max
(tests120 s). Never overlap workers, overwrite completed output or rerun source
acquisition. Initial process check found0 active guarded workers; disk25.54 GiB.
Capacity is dynamic.369 tests passed at V14. V10/V14 journals/backups remain intact.

Next: implement/test separate date-aware lineage/QA worker, freeze hashes, run
geometry/lineage preparation and commit it BEFORE pixel QA. Then bounded screen,
replay, durable evidence and final state. Retain every outcome; never shrink masks,
change QA thresholds or call an automatic pass a class/label approval.

GitHub HTTPS pushes have timed out. SSH port443 works with the same account:
`ssh://git@ssh.github.com:443/muhammedazhar/MSc-DataScience.git`, BatchMode and
StrictHostKeyChecking enabled. Push same branch, fetch tracking and compare HEADs;
no force push/global remote changes. Raw sources/caches stay local; code and bounded
evidence are backed up. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Project `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`;
Git root is its parent. Preserve model1664 at0.5 and masks0/1/255.
