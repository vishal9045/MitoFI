# Mask checks after scoring (2026-10-28)

Done after `results_summary.txt` was written, on the investigator's question
("the numbers look odd for fragmented"). Nothing here changes a scored number;
the results stand as scored. Figures in `diagnosis/` (PNG, on disk, not in git).

## Cape Town FCCP: fragmented fields read elongated

FCCP beat control (0.017 vs 0.002, p = 0.0008, every repeat in direction), but
fields that look plainly fragmented read 0.00–0.45, mostly in the elongated band.

- **The images are fragmented**: separate round puncta, packed around the
  nucleus, on a bright, grainy cytoplasm (MitoTracker leaks out once FCCP
  collapses the membrane potential). `fccp_zoom.png`.
- **The masks are not**: the pixel classifier labels the haze between packed
  puncta as mitochondria and fuses a cluster into one long, thick ribbon.
- **So the shape features read elongated** (medians, `cape_town_features.csv`):

  | feature | Cape Town FCCP | Cape Town control | Olympus elongated | Olympus fragmented |
  |---|---|---|---|---|
  | p90 aspect | 3.31 | 4.04 | 3.17 | 2.54 |
  | skeleton length per object (um) | 5.25 | 9.09 | 5.84 | 3.56 |
  | fraction of foreground in largest object | 0.156 | 0.207 | 0.130 | 0.044 |
  | P(elongated) | 0.94 | 1.00 | | |

- **Not mainly the projection**: on the single sharpest plane the clusters still
  merge into blobs (and those dimmer planes are refused by the gate);
  `fccp_plane.png`. The max projection over 10–19 planes adds to it.
- **The gate does not catch it**: the fused masks are clean and pass.

Same weakness as the Allen dim cells: textured or hazy cytoplasm between
mitochondria is called mitochondria. Named limitation of MitoFI 1.0.0: densely packed
puncta on a bright cytoplasmic background (e.g. depolarised cells stained with a
potential-dependent dye) are read as more elongated than they are; the direction
held here, the magnitude is understated. The dim-cytoplasm scribbles (fields
33–44, for the next version) target this; using these images for training would end their
use as a test.

## Mito-Hacker crops

Drp1-/- networks are segmented as connected networks (0.00); one-copy cells
are punctate (0.24–0.63). Refused crops are mostly blurry, enlarged cells,
suggesting some images were taken at a different zoom from the rest, so one
assumed pixel size does not fit every image. `mh_masks.png`.

## BBBC053

At 0.0201 um/px, DMSO networks break into short pieces and FCCP shows swollen
ring-shaped mitochondria picked up as rings and short pieces. `bbbc_masks.png`.
As published, 17 of 29 DMSO fields were refused (speckle) against 2 of 29 FCCP,
so the scored DMSO fields are a selected subset; the 4x copy scores all 58 and
the prediction still holds there (p = 0.041).
