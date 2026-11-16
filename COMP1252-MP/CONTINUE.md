# Continue this project with any coding agent

This is the tool-neutral entry point for Codex, Gemini Antigravity, or another
agent. Work in this project on **`codex/curated-deforestation-pilot`**. Read
`AGENTS.md`, check `git status`, and read the latest handoff linked below before
editing. Preserve other work and commit each completed, verified logical change
on this same branch with an updated handoff.

Current completed audit/state: **`Docs/Handoffs/2026-11-16-validation-support-pool.md`**.
Previous completed review: `Docs/Handoffs/2026-11-16-validation-pilot.md`.
Previous completed milestone: `Docs/Handoffs/2026-11-16-pixel-pilot.md`.
Previous completed milestone: `Docs/Handoffs/2026-11-16-overfit-diagnostic.md`.
Earlier pipeline: `Docs/Handoffs/2026-11-16-pilot-baseline.md`.
Dataset/provenance: `Docs/Handoffs/2026-11-15-curated-pilot.md`.

Project root:
`/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Git root is its parent MSc-DataScience directory. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.

The project locates forest loss in paired Sentinel-2 images and aims to estimate
when it first becomes visible. Annotation image dates and observed visibility
intervals are not exact felling dates. A small reviewed training dataset and a
working partial-label pipeline are committed. The short smoke predicted all
reviewed pixels as clearing; the completed controlled single-example experiment
shows all three predeclared configurations can learn the reviewed regions of
one crop. The completed reviewed-pixel objective and bounded all-13 experiment
now also fit every reviewed training region, with independent saved-logit
verification and 206 passing tests. No unseen-location accuracy, complete-mask
correctness, clearing area or date-inference claim follows. The first bounded
validation roster is now fully reviewed and independently verified: all five
cases were rejected, so no validation dataset or accuracy score exists. The
geometry audit explains insufficient pixel interiors in most eligible pairs.
The broader support audit is also complete and independently verified: all
540 screened validation pairs are accounted for, but no fresh group meets
every current gate. The empty shortlist and all reasons are saved; 263 tests
pass. Next is an explicitly predeclared review of quality screening on the
reviewed ROIs and their model context. Do not silently relax the completed
protocol, repeat training, or force labels from rejected cases.
The handoff identifies proposed next work and what has already finished.

Use one low-priority CPU worker, native/Torch threads one, no GPU, monitored
512 MiB worker RSS and 1.5 GiB available-system-memory reserve. Use the existing
resource guard unchanged. Read the handoff for the frozen QA files and data
locations. Labels are **0** reviewed forest, **1** reviewed clearing, **255**
ignore; preserve this mask in every loss and metric.

If the previous agent stopped mid-run, check active Python commands and the
handoff's completed output directories before launching another worker.
Completed guarded output has a metadata.json; unfinished workers publish no
final output. Never duplicate an active pixel/model job or overwrite completed
results. Read current files and receipts rather than assuming a previous
agent's chat history or tool state is available.
