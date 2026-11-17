# Validation context quality audit continuation

## Current state: approved work in progress

Continue on `codex/curated-deforestation-pilot`; commit every completed verified
logical change with this handoff updated and push to the same GitHub branch.
Previous broader audit is complete at `41fb657`; do not repeat it or training.
Root `CONTINUE.md` works for Codex, Gemini Antigravity or another agent.

The approved next task is a separate predeclared quality/context audit of the
one fresh V2 candidate blocked only by whole-crop QA. Protocol:
`Docs/Reviews/ValidationContextV3/protocol.json`. Candidate:
`event-bf3d28f31dcf19938962`, pair `pair-430ec0ecae32ce98efcb`, validation group
`group-45f0c39e787ca8622918`, 2019-04-02 -> 2019-04-27, annotation 2019-04-22.
Its target has36 pixels/interior4; prior endpoint full-crop QA counts56/0.
The worker/helpers and architecture checks are implemented. Fourteen focused
tests pass, including the actual factory's synthetic gradient proof of its
5x5 dependency, full-halo failure cases, finite inputs and protocol bounds.
All twelve worker source hashes are now frozen in the protocol before execution.
No candidate context calculation, feature extraction or image review has run yet.
The independent verifier is being finalized; no result is approved yet.

## Fixed rules and boundaries

The frozen tiny CNN has two3x3 convolutions and1x1 output: a5x5 receptive field,
radius2 input halo. Static source checks and a synthetic gradient test must
prove this without using trained weights or real validation data in the test.
This bound cannot be transferred to larger/global-context models.

Audit BOTH the full36-pixel anchor-temporal target +halo2 AND the4-pixel
supervised interior +halo2. Require the full halos inside64x64, all five frozen
QA components zero at both dates, and all27 actual model input features finite.
Do not clip a halo, shrink a ROI or remove flagged pixels to make it pass.
Keep the original full-crop flags and all old V1/V2 results unchanged.

Propose at most one forest control using sizes16/12/8, largest first then first
row/column; no best-NDVI/model score. Its ROI must have NDVI>.6 at both endpoints
and abs change<.1. Its FULLhalo must be clear/finite and avoid the all-catalogue
20m exclusion plus full target context. The NDVI rule is a search aid on the ROI,
not a forest label. No control is a valid retained negative audit outcome.

Actual27-channel features will be extracted from fourteen bounded source band
windows for this ONE fixed pair, with source stats before/after. Cached RGB/NDVI
alone cannot establish finite input values in every channel. No deforestation
model inference, new visual previews, training, labels or accuracy in this task.
Only independently verified numeric eligibility can precede separate visual review.

## Remaining work

1. Finish/test new helpers, one-case worker and independent verifier, preserving
   every frozen source file. No archived QA/training/validation modules modified.
2. Run one low-priority guarded audit and independently reconstruct all halos,
   per-date QA, finite inputs and deterministic control proposal from saved arrays.
3. Save complete results/test proof, update guide and this handoff, commit/push.
   Record failure/no-control outcomes without fallback or automatic labels.

Project: `/Users/muhammedazhar/Developer/Learning/MSc-DataScience/COMP1252-MP`.
Git root is its parent. Python:
`/Users/muhammedazhar/.miniconda3/envs/COMP1252-MP/bin/python`.
One nice+10 CPU worker, native threads1, no GPU. Unchanged guard512MiB RSS,
1.5GiB available-memory reserve,300s maximum; never concurrent pixel/model jobs.
Local working outputs: `Datasets/ValidationPilot/v3/` (ignored). Durable evidence:
`Docs/Reviews/ValidationContextV3/`. Never overwrite completed outputs. Preserve
0/1/255 labels and frozen train/test footprint isolation. Earlier provenance,
QA policy and runtime discrepancy are in linked prior handoffs/protocol sources.
