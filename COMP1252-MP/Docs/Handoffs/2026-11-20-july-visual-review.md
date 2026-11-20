# Visual review of the 6–31 July pairs

V19 packet preparation and exact replay are complete. Inspection of both
areas and visual dispositions are pending.
Continue on `codex/curated-deforestation-pilot` and push verified checkpoints
to the same branch.

The fixed roster is both 6 July to 31 July 2018 pairs that passed automatic QA.
The packet uses the original 64 × 64 grids and four hash-verified V18 cached
observations. Use matched fixed-scale RGB first, then exact contours,
nearest-neighbour target detail, QA flags, and NDVI diagnostics. Record
the appearance and uncertainty for both areas, including the four 31 July
shadow-risk pixels near area A.

See `Docs/Reviews/ValidationJulyVisualV19/protocol.json` for the predeclared
roster, inspection rule, and resource limits. The two areas share one
geographic group. Use a single nice +10 worker and the existing 512 MiB cap,
1.5 GiB available memory reserve, and 300-second limit.

The archived `packet/` includes six comparison panels, the per-area inputs
and descriptive statistics, and the guarded worker receipt. `packet_verification/`
records the exact replay of all seven packet artifacts. Preparation and replay
peaked at 161.98 and 159.02 MiB RSS respectively. All 391 offline tests passed at
340.61 MiB peak worker RSS. The packet used four cache bindings and zero raw
JP2 reads. The inspection should open the plain panels before the overlays.
