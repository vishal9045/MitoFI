# Mask check after scoring (2026-10-28)

Done after `results_summary.txt` was written (`775cc6d`); no number changes.
Figure `diagnosis/g_masks.png` (on disk, not in git).

**G (MitoGFP-RFP, 27 Dec 2023): every field reads ~0.00, mock and infected
alike (mock median 0.016, 31 of 38 elongated).** In these fields the GFP signal
fills the cytoplasm as dense, fine granules; the pixel classifier labels most of
it, so the mask is one connected mesh over the whole cell and the index reads
elongated whatever the cells look like. The gate passes these masks. This is the
whole-cell mesh failure already known from stretched Allen cells and the public
Cape Town FCCP fields (dense puncta / texture on bright cytoplasm), here on a
reporter label. The three G comparisons (two firm) therefore test segmentation,
not morphology. CCCP fields (A, 6 Nov 2023, MitoTracker) are segmented as
separate puncta and read fragmented, as predicted.
