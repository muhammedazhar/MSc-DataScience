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

Current status is pinned source and QA preparation. Preflight, source
publication, and the four crop checks are pending. Class approvals and model
evaluation remain separate from this QA milestone.
