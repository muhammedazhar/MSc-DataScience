# Remaining July pair QA

V18 lineage is complete. All six pairs have distinct IDs, and cumulative
references, full source hashes, and train/test isolation are verified.
Pixel QA and saved-evidence verification are pending. All 391 tests passed,
including the 30-day boundary, unique IDs, and cumulative date cutoffs.
Lineage preparation took 2.79 seconds and peaked at 239.48 MiB. Continue on `codex/curated-deforestation-pilot`
and push verified checkpoints to the same branch.

The roster covers six pairs. Each of the 11, 6, and 1 July 2018 scenes is
paired with the fixed 31 July scene at both original event grids. Both
events belong to the same geographic group. Recompute cumulative annotation
references for all six pairs and save them before screening. The 1 July
baseline predates the 6 July annotations. The 6 July baseline includes them.
The 30-day interval from 1 to 31 July is within the frozen bounds.

Use the V17 acquisition receipts and V15 handoff for source integrity, split
isolation, and QA constraints. Use one nice +10 CPU worker, a single native
thread, a 512 MiB RSS cap, a 1.5 GiB available memory reserve, and a 300-second
maximum per run. The protocol is under
`Docs/Reviews/ValidationRemainingScreenV18/`.
