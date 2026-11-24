# Fresh validation acquisition and QA

V24 follows the completed V23 source search. Continue on
`codex/curated-deforestation-pilot` and push verified checkpoints to that branch.
The two exact 36UXA/R064/N0206 scenes are 21 and 16 July 2018. The two
original validation groups, post dates, and 64 × 64 crop grids stay fixed.
Earlier rejected pair outcomes and the seven exposed groups are retained.

Check provider metadata and all 58 listed files per scene before acquisition.
Stage each verified source separately and use the existing transaction and
guard. Derive four comparison identities, recompute cumulative reference
geometry before decoding new imagery, and retain every QA result. Use one
nice +10 CPU worker, one native thread, 512 MiB RSS cap, 1.5 GiB available
memory reserve, and at least 2 GiB disk reserve during download.

Checkpoint `62b7b76` pins the protocol and three workers. Live preflight and
offline replay passed for both real 36UXA XML pairs, complete padded footprints,
and 58 listed asset pins per scene. The listed total is 1,143,754,335 bytes.
Working memory peaked at 102.83 MiB in preflight and 90.39 MiB in replay.
Checkpoint `6bc801e` saves preflight and exact replay. The 21 July source
passed staging with 58 files and 596,775,942 bytes verified. The download
took 79.64 seconds and peaked at 101.56 MiB RSS, recorded at `5b2e405`.
The 16 July source also passed staging with 58 files and 546,978,393 bytes
verified. It took 76.94 seconds and peaked at 100.03 MiB RSS. Both sources
are ready for transaction publication. Publication and crop QA are pending.
