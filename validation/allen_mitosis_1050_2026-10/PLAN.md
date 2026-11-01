# Allen TOMM20 mitosis vs interphase, scored with MitoFI 1.0.0 — plan frozen before scoring

Written 2026-10-27 and committed with `inputs.csv`, `run.py` and `summarise.py`
before any cell is scored with MitoFI 1.0.0. Asked for by the investigator, to add the
1,050 Allen cells of the October study to the evaluation of MitoFI 1.0.0.

## What these cells are, and what they are not

- The same 1,050 cells as `validation/allen_mitosis_2026-10/` (`inputs.csv` copied
  unchanged, seed 20261023): 150 each of M1M2, M3, M4M5, M6M7, and 450 interphase
  (M0) cells, each matched to an M1M2/M3/M4M5 cell in the same field.
- **Never trained on.** None is among the 60 Allen cells the scribble fields came
  from, nor the 238 fresh cells; no index model saw any Allen cell.
- **Not untouched.** Scored once before with an earlier build (v5 pixel classifier,
  v7 index): mitotic 0.305 vs interphase 0.257, p = 0.056, not supported. That run
  showed the black point setting the score (open item 10), which fed the work that
  became MitoFI 1.0.0. This is therefore a re-test of a known set with a new tool, weaker
  than the fresh-cell test (`e6cf77f`), and is reported as such.

## Tool

MitoFI **1.0.0** (tag `v1.0.0`): pixel classifier v6 (sha `820cd485…`), index model
v8 (sha `4dfdd1c6…`), gate haze 0.06 / local SNR 1.16. `run.py` checks both model hashes before scoring.

## What is run, per cell (as the October study, unchanged)

`run.py` reuses `validation/allen_mitosis_2026-10/run.py`: TOMM20 channel through
`to_field` (maximum projection), 0.108333 um/px, the tool's rescue chain for
refused fields (as is, then background removal, then black point; stop at the
first pass; the rescue is recorded). Two analyses: **inside the cell** (primary,
MitoFI's mask kept within Allen's membrane segmentation) and **whole crop**.

## Predictions (same as October)

1. **Primary (inside the cell):** median index of mitotic cells (M1M2 + M3 + M4M5)
   above interphase; one-sided Mann-Whitney, alpha 0.05. **This decides pass/fail.**
2. The same for the whole crop.
3. Among stages, the median is highest at M3 or M4M5.

M6M7 reported, not tested.

## Added for MitoFI 1.0.0 (black point reads ~0.2 high, D24 / `fa2891a`)

- **Rescue mix by group** (as is / background / black point), with a chi-square
  test. If mitotic and interphase cells were rescued in different proportions,
  prediction 1 is confounded by the rescue and is reported as such.
- Prediction 1 again **within black-point rows only** and **without black-point
  rows**, so the comparison is made between cells treated alike.
- Field-matched pairs (Wilcoxon signed-rank), both cells needing the same rescue
  and regardless; secondary.

## Reported with every number

Cells, scored, refused by reason, rescue used, median and IQR, band counts; read
or download errors listed, never dropped. Random outline check of 20 cells (seed
20261023). Results committed unedited.

## Known limits

As in October: hiPSC interphase mitochondria may already read fragmented; the
whole-cell projection piles up rounded mitotic mitochondria; low intensities make
most cells need a rescue; stage labels are Allen's predictions.
