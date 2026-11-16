# Review of final training prediction previews

Date: 2026-11-16. All four committed pages `run/preview_1.png` through
`run/preview_4.png` were opened and visually inspected after training and
independent numerical verification. They display all 13 manifest examples,
post-image RGB, explicit reviewed labels and actual final predicted probability.

The clearing-interior labels lie within high-probability regions and the
stable-forest control ROIs within low-probability regions, consistent with the
verified counts. Gray label pixels are explicitly identified as ignored.
The readable previews retain their training-only and unvalidated-region notices.

Several predictions extend beyond the small reviewed clearing interiors,
including other visible features in the same crop. These pixels have unknown
labels: visual resemblance does not establish correct boundaries, clearing
areas, false-positive rates or complete-mask accuracy. No new supervision was
created, and these predictions were not used to change training, choose a
checkpoint or tune a threshold. This review is of presentation and qualitative
consistency, not new ground-truth annotation or held-out evaluation.
