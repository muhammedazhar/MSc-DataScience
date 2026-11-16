# Broader validation support audit continuation

## Current state: approved audit milestone complete; fresh shortlist empty

Branch: `codex/curated-deforestation-pilot`. Commit every completed, verified
logical change with this handoff updated and push to the SAME GitHub branch.
Root `CONTINUE.md` is the tool-neutral entry point for Codex, Gemini Antigravity
or another agent. No chat history or model/pixel job needs recovering.

Milestone checkpoints: `a4b3ef0` predeclared broad protocol; `546ec81` tested
coverage/ranking worker; `24ca554` complete raw audit and explicit empty shortlist.
The following completion commit contains independent verification, 263-test
proof and final documentation. Completed checkpoints are backed up on origin.

The authorized broader support audit is DONE. It does not produce validation
labels, a validation dataset or model accuracy. All six fresh shortlist slots
are explicitly missing under the predeclared gates; do not relax criteria,
recycle a rejected group or add fallback cases to make the list nonempty.

## Complete universe and finding

Protocol: `Docs/Reviews/ValidationSupportV2/protocol.json`, committed before
processing, binding all input/source hashes, resource limits and selection rules.
All 1146 frozen validation candidate pairs are accounted for:

- 540 completed QA pairs/anchors from 53 events in 14 geographic groups audited.
- 606 without completed published QA retained separately, support count=None.
- All 540 anchor records remain, including quality/history/geometry failures.
- No old area/bounding-size or top-three proposal prefilter is used.

100 anchor-pairs from 10 events retain >=4 one-pixel-eroded interior pixels.
440 do not. Only 8 pairs from 2 events in 2 groups satisfy all quality,
footprint and later-observation gates before review-history exclusions. Those
are prior-exposed groups, leaving ZERO eligible fresh cases. Six missing slots
are saved; no new imagery was opened and no labels were approved.

The measured target is the selected anchor intersected with the pair-specific
catalogue temporal reference, eroded by a 3x3 one-pixel margin. Raw anchor,
raw interior, temporal union, clipped pixels, target/interior and mask hashes
are recorded separately. 34 pairs had temporal clipping, but none changed
from adequate raw interior to inadequate final interior at the four-pixel
cutoff in this pool. Final targets are NOT cached by event/grid alone.

Automatic QA statuses: 229 pass, 248 excluded, 63 visual-review. Selection
additionally requires zero endpoint crop/target QA, zero actual interior QA,
full feature-core separation from EVERY frozen train/test conservative footprint
and all 13 actual training grids, and a later metadata-usable observation in the
same year and within annotation+45 days. Availability is only a persistence
review opportunity, not a clear-label claim.

All five prior EVENTS and GROUPS are excluded from the fresh shortlist while
remaining in the audit. Rank is group/event, exact annotation-date post preferred,
shorter interval and pair ID; all primaries precede alternates and no geographic
group repeats. Size/NDVI/model scores are never ranking criteria. This remains
a deliberately curated diagnostic pool, not population-representative sampling.

## Independent verification and tests

Independent verification PASSED every record, target/interior hash, source/QA
join, full coverage, exclusion, count and complete deterministic empty shortlist.
The worker uses catalogue-date temporal masks and SciPy erosion; the verifier
instead burns anchors against the published sixth QA plane and erodes with
nine NumPy slices. It never calls the worker's gate or shortlist functions.
Both use frozen metadata/QA files, no satellite JP2 pixels, no Torch import,
no model/cloud-model execution, new image review, label approval or training.

**263 tests pass** (previous 250 + 10 broad-rule/protocol regressions + 3 independent
verification regressions). Full suite: 1.254s tests in 2.80s guarded worker,
peak 312.58 MiB, minimum 3.11 GiB available. Do not rerun finished tests or audits
without a new change, failure or explicit recheck need.

Audit worker: 4.59s / 198.77 MiB, minimum 4.17 GiB available. Independent verification:
2.04s / 284.67 MiB, minimum 2.55 GiB available. All workers were sequential nice+ 10
CPU, native threads 1, no GPU, unchanged 512 MiB RSS and 1.5 GiB memory reserve.
Every original source/QA/model/review fingerprint remained unchanged.

## Diagnostic evidence for the next task (no rule change)

The verifier counts one-gate-at-a-time omissions for planning only:

| Only omitted gate | Pairs | Events | Groups |
| --- | ---: | ---: | ---: |
| Entire-crop-zero-QA | 1 | 1 | 1 |
| Previous-group exclusion | 3 | 1 | 1 |

The first retains ALL other gates, including fresh group, positive/target QA,
minimum support and follow-up. It suggests one fresh event is blocked solely
by quality flags elsewhere in the core. It does not establish that the model's
local context or any proposed forest control is clear. The second is event-only
history exclusion, not permission to call prior-exposed group imagery fresh.
The original protocol and empty shortlist are unchanged in both diagnostics.

Next proposed bounded task (NOT executed): predeclare and audit QA on reviewed
ROIs AND the model context they depend on, using this diagnostic evidence.
The current tiny CNN has two 3x3 convolutions followed by 1x1 output: radius 2
pixels of input context. A possible separate rule must explicitly screen that
halo at BOTH dates, preserve original flags/unknown 255 pixels, and separately
review forest controls and temporal evidence. This is architecture-specific;
a larger model could depend on wider/global context and need a different rule.
Do not silently relax this completed whole-core protocol or infer labels from QA.
Only a separately locked, reviewed dataset could precede final 1664 inference at .5.

## Durable files and safe reproduction

Authoritative tracked evidence: `Docs/Reviews/ValidationSupportV2/`:

- `protocol.json`: predeclared rules and input/source identities.
- `audit/{audit.json,selection.json,report.md,metadata.json}`: all 540 measured
  records, 606 unscreened dispositions, source hashes, empty roster and resources.
- `verification/`: independent proof and diagnostic gate counts/report.
- `checks/`: focused and full-suite receipts/logs.

Guide: `Docs/Guides/ValidationSupportPool.md`. New code:
`Solutions/audit_validation_support_pool.py`,
`Solutions/verify_validation_support_pool.py`; corresponding new tests in Tests.
Original frozen files are unchanged. Local ignored outputs:
`Datasets/ValidationPilot/v2/`. Full source/audit/QA inputs remain on this Mac;
GitHub holds code and bounded evidence, not complete source imagery/cache.

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Git root is its parent. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
Check branch/status and active jobs before new work. Use one low-priority CPU
worker, native threads 1, no GPU, existing unchanged guard 512 MiB / 1.5 GiB reserve,
max 300s. Preserve train/test split boundaries and labels 0/1/255 semantics.
Frozen QA policy and Torch runtime/distribution discrepancy remain as documented
in the previous `2026-11-16-validation-pilot.md` and linked training handoffs.

If a verification recheck is specifically required, use a NEW output directory:

```bash
PILOT_PYTHON=/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python
"$PILOT_PYTHON" -B Solutions/verify_validation_support_pool.py --run Docs/Reviews/ValidationSupportV2/audit --output Datasets/ValidationPilot/v2/verification-recheck
```

No worker remains active. No training, previous five-case review, broad audit
or independent verification needs repeating to continue. Current test groups
remain untouched; first-visible-date inference remains later work.
