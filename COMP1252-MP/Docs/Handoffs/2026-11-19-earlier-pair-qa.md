# Earlier pair QA continuation

## Current checkpoint: four lineages/targets verified; automatic QA pending

Continue on `codex/curated-deforestation-pilot`; update this handoff, commit and
push each verified logical change to the SAME branch. Read
`2026-11-18-earlier-source-acquisition.md` and its linked V13/V12/V11/older
handoffs for provenance, source integrity and frozen constraints. Last complete
checkpoint `028cbd6` is pushed; both earlier sources are acquired and metadata-ready.

The user approved the next step. Protocol:
`Docs/Reviews/ValidationEarlierScreenV15/protocol.json`. Exactly FOUR new derived
pairs: pre 21 July and pre 16 July 2018 x the two existing events, fixed 31 July post,
original grids/group/split/anchors. One geographic group, not independent locations.
Preserve original blocked 36UXA parents and both V11 cloudy 26 July source pairs.
No new source dates, replacements, labels, forest predictions, training, accuracy,
manual imagery or 27-channel approval. No V15 pixel work has run at this checkpoint.
Implementation and 11 offline tests are complete/frozen. Lineage/geometry preparation is now complete; screen and replay remain
INCOMPLETE. Protocol checkpoint 48af981 and tested code c63618c are pushed.

Derive new identities with explicit parent hashes, date changes and source lineage.
Update interval_days to 10/15. Recompute intersecting context/post/pre temporal
event sets and expanded-interval provenance blockers. Preserve annotation source
tile/source IDs; replacement imagery is 36UXB, annotations still originate 36UXA.
Recalculate cumulative post-minus-pre masks with frozen helpers, independently
check boolean difference, and retain counts/hashes even if unchanged. Do not reuse
V11's hardcoded dates/derivation/masks or burn only events inside the interval.
Validate entire original train/test conservative footprints and actual 13 training
grids before decoding any satellite pixels. Newly introduced provenance blockers
must not be silently removed.

Three source scenes / 174 files are used: earlier 21/16 July plus original 31 July.
Hash actual files against V14/V11 receipts and check actual frozen QA prerequisites.
Preserve all 116 original V11 source stats/excluded outcomes, prior manual exposure
and automatic-QA history. Source preparation/geometry reads no satellite pixels.
Then screen all four pairs with unchanged QA, no previews. Eight observation calls,
at most six distinct observations/cloud calls and 66 fused-band window requests;
existing post caches may be safely reused. Replay saved caches/QA TIFFs and targets;
this is shared-helper replay, not independent cloud or class truth.

Use unchanged `validate_sample.run_guarded`, ONE nice +10 CPU worker, native
threads 1, no GPU, 512 MiB RSS, 1.5 GiB available-system-memory reserve, 300 s max
(tests 120 s). Never overlap workers, overwrite completed output or rerun source
acquisition. Initial process check found 0 active guarded workers; disk 25.54 GiB.
Capacity is dynamic. 380 tests pass (369 prior plus 11 new), 4.84 s guarded time,
333.52 MiB high-water / 334.0 MiB sampled, minimum 3.44 GiB available. V10/V14
journals/backups remain intact. Read-only review confirmed date-aware lineage,
174-file joins and replay; its full-roster error-retention finding was corrected
and tested. Processing-error runs keep all four outcomes with explicit incomplete
metadata; replay refuses them. Existing V11 caches are hash-pinned before reuse.

Next: run geometry/lineage preparation and commit it BEFORE pixel QA. Then bounded screen,
replay, durable evidence and final state. Retain every outcome; never shrink masks,
change QA thresholds or call an automatic pass a class/label approval.

GitHub HTTPS pushes have timed out. SSH port 443 works with the same account:
`ssh://git@ssh.github.com:443/muhammedazhar/MSc-DataScience.git`, BatchMode and
StrictHostKeyChecking enabled. Push same branch, fetch tracking and compare HEADs;
no force push/global remote changes. Raw sources/caches stay local; code and bounded
evidence are backed up. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Project `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`;
Git root is its parent. Preserve model 1664 at 0.5 and masks 0/1/255.

## Completed pre-pixel lineage checkpoint

Four distinct derived IDs, intervals 10/15 days, exact parent hashes, recomputed
context/post/pre catalogue sets and full train/test barriers passed. Both earlier
dates produce the SAME cumulative reference as the original 26 July parent
(0 changed pixels), because relevant temporal dates are 21 June, 6 July and 31 July;
future 5 August remains context only. This equality was recalculated, not copied.
Anchor target/interior counts remain 55/8 and 92/25 for both dates. No mask or margin
was shrunk and no class label approved. 174 actual source files were SHA-checked
and their actual frozen metadata passed. V11 exclusions/exposure are retained.
Preparation used 2.3 s, RSS 167.03 MiB, minimum 3.23 GiB available.
Completed local output `Datasets/ValidationPilot/v15/lineage/`; byte-identical
tracked copy `Docs/Reviews/ValidationEarlierScreenV15/lineage/`. Never overwrite.
Next: screen all four using this exact completed lineage, NEW `v15/screen/`.
No satellite pixels decoded or QA models run in preparation.
