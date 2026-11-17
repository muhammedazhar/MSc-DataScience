# Continue this project with any coding agent

This is the tool-neutral entry point for Codex, Gemini Antigravity, or another
agent. Work in this project on **`codex/curated-deforestation-pilot`**. Read
`AGENTS.md`, check `git status`, and read the latest handoff linked below before
editing. Preserve other work and commit each completed, verified logical change
on this same branch with an updated handoff.

Current in-progress check/state: **`Docs/Handoffs/2026-11-17-validation-catalogue.md`**.
Previous completed check/state: `Docs/Handoffs/2026-11-17-validation-coverage.md`.
Previous completed inventory: `Docs/Handoffs/2026-11-17-validation-unscreened.md`.
Previous completed review: `Docs/Handoffs/2026-11-17-validation-manual.md`.
Previous completed audit: `Docs/Handoffs/2026-11-17-validation-context.md`.
Previous completed audit: `Docs/Handoffs/2026-11-16-validation-support-pool.md`.
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
pass. The separate context audit is now also complete and independently
verified: target/interior plus two-pixel margins pass both-date QA and all 27
feature checks, but the fixed NDVI control search yields no proposal. Overall
numeric eligibility remains false; no validation labels or accuracy exist.
The separate manual review is now complete and independently verified: all
12 dates and three precommitted control areas were inspected, but the target's
prior forest state and control classes remain uncertain. The case is rejected,
with no labels, observed change interval or accuracy. All 299 tests pass.
This validation group is now exposed and must not be called fresh in a future
roster. The V5 inventory now covers all 606 unscreened pairs/630 anchors:
all pairs already have source-coverage or provenance blockers, and zero passes
all metadata gates. Only 12 supported temporal anchor-pairs (two events in one
fresh group) remain of interest, but all lack post-scene coverage. QA stays
unknown; no roster or labels. The coverage check is now complete: fresh XML
and local directory discovery confirm neither crop intersects any of the 50
local Sentinel-2 scenes. They lie entirely 4.7–5.0km north of source coverage.
All 311 tests pass; the metadata worker used41.31MiB in0.51s. Next proposed work
is a bounded catalogue metadata search for compatible covering imagery, with
exact requirements saved. No download or new QA experiment has been executed.
Do not rerun completed processing, silently retune protocols or force
labels from numeric proxies or rejected cases.
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
