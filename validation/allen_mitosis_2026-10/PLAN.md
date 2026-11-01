# Allen hiPSC TOMM20: mitosis vs interphase, plan frozen before scoring

Written 2026-10-23 and committed with `inputs.csv` and `run.py` before any cell
was scored. First test on public data: none of it shaped the tool.

## Data

Allen Institute WTC-11 hiPSC Single-Cell Image Dataset (Viana et al., Nature
2023, doi:10.1038/s41586-022-05563-7), line AICS-11, endogenous mEGFP-TOMM20.
Public S3: `allencell/aics/hipsc_single_cell_image_dataset`, `metadata.csv`
filtered to `structure_name == TOMM20`: 24,426 cells, 1,963 fields, 52
plates. 3D spinning-disk, 100x, voxel 0.108 um, 16-bit single-cell crops
(channels dna, membrane, structure). Research use under the Allen terms; cite
the paper; no images redistributed beyond figures.

Mitotic stage from the dataset's `cell_stage` column (Allen's rule-based
criteria plus a deep-learning classifier): M0 interphase, M1M2, M3, M4M5,
M6M7_single / M6M7_complete.

## Sample (`inputs.csv`, seed 20261023)

- 150 cells at random from each of M1M2, M3, M4M5 and M6M7 (single and
  complete pooled): 600.
- For each M1M2/M3/M4M5 cell, one interphase (M0) cell from the same field,
  without replacement: 450 (all matched within the field).

## What is run, per cell

The TOMM20 channel through the tool's reader (`to_field`: maximum projection
over all Z, bit depth as the tool handles it). The files record no pixel size,
so 0.108333 um from the metadata is given, as a user would with `--px-um`.

Two analyses, each with the tool's rescue chain for refused fields (as is,
then background removal, then black point at the cytoplasm peak; stop at the
first pass; the rescue used is recorded):

- **whole crop**: the crop scored as a user would, neighbours included.
- **inside the cell** (primary): MitoFI's own mitochondria mask, kept only
  inside Allen's membrane segmentation of this cell (projected over Z).
  MitoFI's segmentation and classifier are unchanged; Allen's mask only says
  which mitochondria belong to this cell.

## Predictions

1. **Primary (inside the cell):** median index of mitotic cells (M1M2 + M3 +
   M4M5) is above that of interphase cells; one-sided Mann-Whitney, alpha
   0.05.
2. The same holds for the whole crop (expected smaller: neighbours are
   interphase).
3. Among stages, the median is highest at M3 or M4M5.

M6M7 is reported, not tested: mitochondria re-fuse as division completes.

## What is reported

Per group and analysis: cells, scored, refused by reason, rescue used,
median and IQR, band counts. Predictions 1-2 also without black-point rows
(D24). Field-matched pairs: paired difference (mitotic minus its interphase
mate) with a Wilcoxon signed-rank test, secondary. A random outline check of
20 cells (seed 20261023). Download or read errors are listed, never dropped.

## Known limits, stated now

- hiPSC mitochondria are less networked than in differentiated cells, so
  interphase may already sit high and leave less room.
- The maximum projection spans the whole cell (about 10 um); rounded mitotic
  cells pile mitochondria up, which in other data has read as elongated. This
  works against prediction 1.
- Intensities are low (camera offset ~420, mitochondria to ~700-900 counts);
  many fields may need a rescue.
- Mitotic stage labels are Allen's, not ours, and are themselves predictions.
