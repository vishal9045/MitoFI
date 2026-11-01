# MitoFI 1.0.0 and the manuscript

Where every number, table and figure in the manuscript comes from.

## Version

**MitoFI 1.0.0** is the published release: pixel classifier v6
(`mitofi/models/pixel_classifier_v6.joblib`, SHA-256 `820cd485…`) and index model
v8 (`mitofi/models/index_model_v8.json`, SHA-256 `4dfdd1c6…`). All tests and
figures were produced with these models; the scoring scripts check the version
and both model hashes before scoring, and the `.run.json` beside every results
file records the model hashes.

Every table is described in `DATA_DICTIONARY.md`. Development numbers below can be regenerated with the repository's Python
environment (`requirements.txt`). Images, the blinding key and the ilastik
scribble project are not in the repository (see the manuscript's Data
availability statement).

## Results: development

| Manuscript | Produced by |
|---|---|
| Pilot composition, observer counts, agreement (κ, Fleiss κ) | `scripts/rater_agreement.py`; labels in `data/labels_*.csv` |
| Feature screen (Methods) and its rerun on the release inputs | `scripts/rank_features.py` (original); `scripts/paper_numbers.py` (`feature_screen_release`) |
| Table 1 (index model coefficients) | `mitofi/models/index_model_v8.json`; printed by `scripts/paper_numbers.py` |
| Inputs for the development numbers | `data/features_v6.csv` (features on v6 masks), `data/index_v8_oof.csv` (out-of-fold predictions), `data/labels_*.csv`, `data/covariates.csv` |
| Session-held-out accuracy, balanced accuracy, macro F1, AUC, medians, disagreements, high-confidence subset, bands, QC refusals, pixel-size and intensity association, gate on the 500-field pool | `scripts/paper_numbers.py` |
| Held-out observer versus model | `scripts/paper_numbers.py` (`held_out_observer_majority_style`) |
| Figures 1–3 | `scripts/paper_figures.py` |
| Pixel classifier v6 and index model v8: training, leave-one-source-out pixel check | `validation/scribbles_v2_retrain/retrain.py`; plan `PLAN.md` (committed `917eac5`), results `results_summary.txt`, `step1_pixelcheck.txt` |

## Results: locked tests

Each test was scored once under a plan committed before scoring. Every folder
holds the plan, the file selection, the scoring and summary scripts, the results
as scored and `results_summary.txt`.

| Test (Table 2) | Folder | Plan committed |
|---|---|---|
| Allen hiPSC, fresh 238 cells | `validation/scribbles_v2_retrain/` (step 5; `allen_inputs.csv`, `allen_a_v5.csv`) | `917eac5` |
| Allen hiPSC, 1,050 cells | `validation/allen_mitosis_1050_2026-10/` (reuses the reader in `validation/allen_mitosis_2026-10/`) | `0771022` |
| BBBC053, Mito-Hacker, MitoTracker (Cape Town) | `validation/public_sets_2026-10/` | `8d73e92` |
| In-house Olympus archive, 828 fields | `validation/olympus_holdout_2026-10/` (conversion: `OirInfo.java`, `OirToOme.java`, `convert.py`; channel check: `contact_sheet.py`; comparisons: `groups.py`) | `fb0935f` |
| Figures 4 and 5 | `validation/paper_figures_2026-10/test_figures.py`; values in `Figures_4_5_values.txt` | — |

Leica images (development data, not a locked test): `validation/all_lif_2026-10/`
(inventory and reader), `validation/scribbles_v2_retrain/leica_v5_a_b.csv`
(scores without the black point) and `validation/rescue_options_2026-10/`
(scores with the full rescue chain: 411 of 458 series).

## Development record

Approaches tried and rejected, intermediate models and the decisions behind them
are available from the authors on request.

## Runtime

Scoring took about 1.9 s per field (512 × 512 or 1,024 × 1,024; 30 pilot fields in
58 s) on a 12-core laptop CPU (Intel Core Ultra 7 255HX), about 7 s per field on
one core, with about 1.5 s start-up per run and peak memory near 1 GB.
