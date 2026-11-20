# Area B input audit and control review

All 27 frozen model-input channels are finite at all 4,096 crop pixels for
the 6 July to 31 July 2018 pair. The full target context (274 pixels) and
eroded interior context (104 pixels) have zero input-validity or QA failures.
The tensor includes seven reflectance bands, NDVI and NDMI per date, and nine
signed post-minus-pre channels. Replay checked channel identities and the
cached RGB and NDVI, plus the original grid and source fingerprints.

## Fixed control squares

The 16 × 16 squares were fixed from July QA, finite inputs and geometry
before the detailed views were opened. Each full 400-pixel context avoids
the target context, the catalogue annotations buffered by 20 m and the other
control contexts. All three contexts are clear on 6 July, 31 July and
10 August. August was checked at the fixed locations after July placement.
All three squares lie within the same geographic group west of the target.

| Control | Zero-based row, column | Three-date observations |
| --- | --- | --- |
| 1 | 2, 2 | Continuous dark blue-green cover with faint mottling and no visible opening or crossing line. |
| 2 | 22, 2 | Continuous dark cover with subtle lower-right tonal variation. The square shows no distinct boundary or new bright patch. |
| 3 | 42, 2 | Smooth dark cover west of the stronger southeastern boundary. The cover appears stable, with the least visible texture to support class identification. |

Both reviewers opened the plain RGB boards before examining the outlines.
They found stable, forest-like cover in all three squares. The August image
is darker across the three squares together, with no localized change
consistent with the opening visible to their east. The dark, low-contrast
10 m imagery leaves some uncertainty about the forest class across each
whole square. All three remain provisional control candidates.

[Three-date control locations](audit/controls-outlined.png) · [Control 1 detail](audit/control-1-outlined.png) · [Control 2 detail](audit/control-2-outlined.png) · [Control 3 detail](audit/control-3-outlined.png)

The July input checks apply to the model tensor. The August check uses the
stored RGB, NDVI and QA on the same grid for follow-up evidence. The result
retains the original Area B observation that the lower opening expanded
after 6 July and by 31 July and remained visible on 10 August. The onset and
full class of that clearing, class approval for the controls, and validation
eligibility remain unresolved. [Recorded decisions](reviews.json)
