# Retrain the pixel classifier on the condition-chosen scribbles: plan, frozen before fitting

Written 2026-10-25 and committed with `retrain.py`, `scribbles_summary.txt`
and `allen_inputs.csv` before any model was fitted (D25). DEVELOPMENT for
Olympus and Leica (every in-house image is now training or development data);
the Allen part is a fresh frozen test.

## Training pixels

- **Original scribbles**: the 43,873 Labkit pixels (`data/labelings/montage.tif.labeling`),
  mapped back to their 8 tiles (`data/montage_tiles.csv`). Each tile is
  byte-identical to its source image, so features are computed on the tile alone,
  never on the montage canvas.
- **New scribbles**: every stroke in `data/scribbles_v2/mitofi_scribbles.ilp`
  (labels read by name: `Mitochondria`, `Background`), each lane matched to its
  field by file name, features computed on that field's own image
  (`data/scribbles_v2/fields/`). Field 08 (not a mitochondria image) is excluded
  even if strokes are found on it. Counts and the ilp hash: `scribbles_summary.txt`.
- **Sources**: olympus (8 montage tiles + the new Olympus fields), leica, allen.
- **Weights** (RandomForest `sample_weight`): within a field, each class's pixels
  share weight min(n, 2000), so a field cannot outweigh others by stroke volume
  (fields 03 and 14 carry 15,580 and 10,803 background pixels; one montage tile
  carried 76% of the old set). Then every (source, class) pair is scaled to the same
  total weight, so classes and sources are balanced.
- **Forest**: 200 trees, scikit-learn defaults otherwise, seed 20260905 (as v5).

## Candidates

- **(a) raw**: the shipped 25-plane stack, `mitofi.segment.feature_stack`, sigmas 1, 2, 4, 8.
- **(b) ratio**: `stack_invariant` from `scripts/invariant_features.py` (28 planes,
  each filter response divided by a Gaussian local mean at 4 sigma). Known limit,
  stated now: the ratios are invariant to gain, not to an additive camera offset
  (Allen ~420 counts); the Allen strokes are what is meant to teach that.
- **v5**, the shipped model, is the reference throughout.

## Step 1 — pixel check on the scribbles (both candidates)

Leave one source out: train on two sources, predict the third's strokes; report
balanced accuracy (mean of the two class recalls, each field weighted equally)
for (a), (b) and v5. Descriptive; not a pass criterion.

## Step 2 — Olympus development pool (both candidates)

For each candidate: segment all 500 development images, compute the shipped
features (`mitofi.features`), refit the index model exactly as
`scripts/stage7_index.py` does (same 9 features, standardise + multinomial
logistic regression, GroupKFold by session, 5 folds, seed 20260905, the 444
three-rater majority labels, 90%-purity bands) -> v8 candidate. Re-derive the gate
limits on the new masks with the rule of `scripts/gate_thresholds.py` (haze: p99
of rater-1-readable fields rounded up; large-object SNR: p1 rounded down). v7 is
recomputed by the same code from `data/features_v5.csv` as a check that the
baseline reproduces (0.721, kappa 0.548, AUC 0.951).

**Pass** (all three): out-of-fold accuracy on the 444 >= 0.691 (v7 minus 0.03,
about 1.5 standard errors at n = 444); fragmented-vs-elongated AUC >= 0.931 (v7
minus 0.02); refusals among the 444 under the re-derived gate <= 26. Kappa and a
session-bootstrap 95% interval on (candidate - v7) accuracy are reported.

## Step 3 — Leica, no black point (both candidates)

All 496 series of `validation/all_lif_2026-10/inventory.csv`, read as that run
did (channel per series, max projection, bit depth), scored with the candidate's
pixel model, its v8 index model and its re-derived gate; a refused field is
scored again after background removal (rolling ball 2.5 um). **No black point.**
Fields coarser than 0.2 um/px stay refused. Conditions via `run.condition`.

Comparisons (median index, group A vs group B, within one file):

| # | file | A | B | expected |
|---|---|---|---|---|
| 1 | CDKi treatment + CoV-2 | `cdki inf` | `veh cdki inf` + `veh inf for cdki` | A < B |
| 2 | huh7 mfn1 ko pool / Huh-7 WT for MFN1 KO | pool `mark_and_find` | WT `mark_and_find` | A > B |
| 3 | colony 9 mfn1 ko mito / Huh-7 WT for MFN1 KO | `col mfn1 ko` | WT `mark_and_find` | A > B |
| 4 | mef all cells mavs localizatio test | `mfn1 ko mock` | `wt mock` | A > B |
| 5 | mef wt mfn2 ko mavs localization | `mfn2 ko mock` | `wt mock` | A > B |
| 6 | 2nd feb 2023 mef wt and ko | `mef ko mock` | `mef wt 24h mock` + `mef wt mock` | A > B |
| 7 | 2nd feb 2023 mef wt and ko | `mef 48h mavs ko mock` | same WT as 6 | A > B |

(2 and 3 pair two files of the same cell line, as the rescue-chain table did.)

**Pass** (both): all 7 medians in the expected direction (a comparison with an
empty group fails); more than 256 of 496 series scored. Two-sided Mann-Whitney p
is reported, not required. Plus an outline check: two scored series per file at
random (seed 20261025), image beside mask, judged by eye and described.

## Selection

The candidate that passes steps 2 and 3; if both pass, the one scoring more Leica
series; on a tie, (a) as the smaller change. If neither passes, the retrain has
failed: the Allen test below is **not** run (it stays unspent) and the next step is
a pretrained network fine-tuned on masks corrected from these fields.

## Step 4 — leave one source out (selected candidate only)

Retrain the selected candidate without Leica and without Allen (same weighting);
each gets its own Olympus refit (step 2). The no-Leica model runs step 3; the
no-Allen model runs step 5 alongside the full model. Reported: refusals, the 7
comparisons, outline checks. Descriptive: shows what a source not in training gets.

## Step 5 — Allen fresh test (selected candidate, run once)

`allen_inputs.csv` (seed 20261025): from the TOMM20 metadata, mitotic cells
(M1M2, M3, M4M5) never sampled before -- not in `allen_mitosis_2026-10/inputs.csv`,
not among the 60 cells of `data/scribbles_v2/allen_src/`, and not in a field (FOV)
of those 60, since crops include neighbours. Up to 50 per stage; for each, one
never-sampled interphase (M0) cell from the same field, without replacement.

Per cell, as `allen_mitosis_2026-10/run.py`: TOMM20 channel through `to_field`,
0.108333 um/px, the tool's mask kept inside Allen's membrane segmentation
(primary) and the whole crop (secondary); as is, then background removal; **no
black point**. Scored with the selected full model, the no-Allen model and v5 (v5
with the shipped v7 index and gate), same downloads.

**Pass** (both, selected full model, inside the cell): mitotic median index above
the field-matched interphase median, one-sided Mann-Whitney p < 0.05; and more than
half of all cells pass the gate as is (no rescue). Matched-pair Wilcoxon reported.

## If it passes

Ship the model (pixel classifier v6, index model v8, new gate limits); retire the
black point (D24); README/CLI help, CHANGES doc + PDF; consider a release. If it fails,
results are committed as they come and the paper is scoped to images that pass the gate.

## Reported with every number

Fields scored, refused by reason, rescued (background), and for v5 rows the
rescue used.
