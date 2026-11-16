# Rejected validation review verification

Verified all 5 rejected candidates and every prepared observation.
**No validation dataset was published and no model evaluation or accuracy estimate was produced.**

| Event | Role | Target pixels | Eroded interior pixels | Rejection reasons |
| --- | --- | ---: | ---: | --- |
| event-9993d3bc382d3405a7ad | primary | 21 | 1 | positive_interior_1_below_minimum_4 |
| event-4ee5bdeb33ddec684474 | primary | 14 | 0 | positive_interior_0_below_minimum_4 |
| event-4a34368ee9483ed8f7a7 | primary | 36 | 10 | pre_event_forest_not_unambiguous_at_10m |
| event-9495d4e4197219fc88a6 | alternate | 22 | 0 | positive_interior_0_below_minimum_4 |
| event-3304b611a63af8c10ebc | alternate | 21 | 0 | positive_interior_0_below_minimum_4 |

Checked frozen selection/pair/group joins, complete geographic footprints, review-time artifact hashes,
actual target erosion, local timeline coverage, cached QA, source fingerprints and retained rejection decisions.
No Torch, model execution, raw JP2 pixel reads or training.
