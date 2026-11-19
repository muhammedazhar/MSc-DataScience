# Earlier-date pair QA

V15 is complete: all four fixed pairs (pre 21/pre 16 July 2018 x two events,
post 31 July) are excluded by frozen automatic QA. Cloud/buffer flags cover every
target/context pixel in both pre-images. No labels, forest predictions, manual
imagery, first-visible interval or accuracy were produced. 380 tests pass.

Start with `Docs/Handoffs/2026-11-19-earlier-pair-qa.md`. Separate implementation
`Solutions/screen_earlier_validation.py` supports prepare, screen and verify.
Preparation was committed before source pixels: four new IDs, 10/15-day intervals,
full parent/group/split/grid/annotation provenance, actual 174 source hashes,
train/test barriers and freshly recomputed cumulative post-minus-pre references.
The masks match the parent 26 July geometry because relevant annotation dates
are unchanged; this was recalculated and hash-bound, not copied.

QA retains all four outcomes with no previews. Eight requests / six unique
observations produced 4 new cloud calls and 4 cache hits; original V11 post caches
are byte-identical. Replay verified all four six-plane masks, safe cache bindings,
quality outcomes and anchor contexts without source JP2 windows/model reruns.
Processing errors would retain the entire roster as explicitly incomplete and
cannot replay/pass. Automatic QA never approves class labels or all 27 features.

Evidence: `Docs/Reviews/ValidationEarlierScreenV15/{lineage,screen,verification,cache_evidence,checks}/`.
Local completed `Datasets/ValidationPilot/v15/` results must not be overwritten
or rerun due to an agent change. Four TIFFs and six bounded NPZ snapshots are
backed up with code; full source imagery stays local. QA used 360.02 MiB in 11.2 s,
one nice +10 CPU worker, native threads 1, no GPU and unchanged 512 MiB/1.5 GiB
reserve guard. Original parents, V11 failures and V10/V14 transactions persist.

Next proposed task: separately predeclare metadata-only checks of all three
remaining V12 nominal dates (11, 6, 1 July). V15 acquired/screened no extra dates.
Any later labels need clear temporal class evidence, controls, persistence,
full 27-channel eligibility and a label lock. Masks 0/1/255/model 1664 stay frozen.
