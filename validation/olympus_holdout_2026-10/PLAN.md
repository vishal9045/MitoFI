# Olympus images never used in development, scored with MitoFI 1.0.0 — plan frozen before scoring

Written 2026-10-28, committed with `groups.py`, `inputs.csv`, `run.py`,
`summarise.py` and the conversion tools before any image is scored. Asked for by
the investigator after training was complete: test the tool on the in-house
Olympus FV3000 archive files that never entered the blinded pool.

## Which files (selection done before any scoring)

- Archive: `F:\microscopy VS`, 1,807 `.oir` files.
- **Excluded: every file in the blind key** (630: the 500-image pool, 127 removed
  in manual QC, 3 never converted), matched to the archive by path or name
  (630/630 matched). The key opened only now, with training complete
  (investigator, 2026-10-28). Key-derived tables stay outside the repository
  (`D:\My ideas\mitofi\olympus_holdout\`).
- Of the 1,177 others (metadata by Bio-Formats, `OirInfo.java`): a channel the
  acquisition preset names MitoTracker (1,103); one image series, no time-lapse;
  pixels <= 0.2 um (980); **acquired on a day that is not a training-pool day**
  (939) nor any key day (938; local time, UTC+5:30). Nile Red sets dropped
  (no mitochondrial stain): 923 files, 107 folders.
- **Channel check by eye** (`contact_sheet.py`, two sample files per folder, every
  channel): the preset names "Mito Tracker Red" even where the stain is absent,
  and in some files sits on the 650 nm laser while the mitochondria are on 570 nm.
  Kept: 92 folders, **831 files**; dropped: 15 folders (no visible mitochondria,
  actin stain, LC3 puncta only, or hazy). Channel by laser line, not name:
  **570 nm** (MitoTracker) or, for MitoGFP-RFP / mTS-GFP sets, **488 nm GFP**
  (recorded as 500 nm), as the investigator chose.
- Conversion: `.oir` -> OME-TIFF by Bio-Formats (`OirToOme.java`, all channels and
  planes, original metadata), via a local copy (`convert.py`).

## Tool

MitoFI **1.0.0** (tag `v1.0.0`; pixel classifier v6 sha `820cd485…`, index v8
sha `4dfdd1c6…`). Command line as a user runs it, one source folder at a time:
`mitofi score <folder> --channel-ex <570|500> --rescue` (max projection over Z;
background removal then black point for refused fields only, marked).
`run.py` checks version and model hashes first.

## Predictions (the investigator's, given before scoring)

One-sided Mann-Whitney on fields, first group vs second; alpha 0.05 each.
"Firm" = stated without hedging; "tentative" = the investigator said unsure / "I
think". Exact file selections in `groups.py`.

| id | first vs second | prediction | confidence |
|---|---|---|---|
| A | CCCP 1:200 vs mock (Huh-7, 6 Nov 2023) | more fragmented | firm |
| B | EV + CCCP vs EV mock (MitoGFP, 19 Sep 2024) | more fragmented | firm |
| C12, C24, C48 | infected vs mock, 12 / 24 / 48 hpi (17 Jul 2023) | more fragmented, growing with time | firm |
| D12, D24, D48 | infected vs mock (29–30 Dec 2022) | more fragmented | tentative |
| E | vehicle infected vs vehicle mock (Mdivi experiment) | more fragmented | firm |
| F_CoV2, F_JEV | CoV-2 / JEV vs mock (LC3 set, 27 Nov 2023) | more fragmented | firm |
| F_DENV | DENV vs mock (same) | more elongated | tentative |
| G_CoV2, G_JEV | CoV-2 / JEV vs mock, 24 hpi (MitoGFP, 27 Dec 2023) | more fragmented | firm |
| G_DENV | DENV vs mock, 24 hpi (same) | more elongated | tentative |
| I | MAVS KO vs WT, mock (MitoGFP, 9 May 2024) | more fragmented | firm |
| K_mock, K_CoV2 | MAVS KO vs WT, mock / CoV-2 (22 Jan 2024) | more fragmented | firm |
| L_Dec, L_Apr | IFN vs vehicle (18 Dec 2023; 29 Apr 2024) | more elongated | firm |
| M | poly I:C vs PBS, 2.5x zoom (19 Sep 2022) | more elongated | firm |
| N_CoV2, N_LC3 | MAVS oe vs PUNO empty vector (29 Jan 2024; 18 Dec 2023) | more elongated | firm |
| O | FL-MAVS + JEV vs EV + JEV (MitoGFP) | more elongated | firm |
| J, P, Q | A549 MAVS KO vs WT; MFN1 oe vs EV; A2a vs mock | no prediction (reported) | — |

Duplicates removed in the selection: M uses the 2.5x zoom fields only (the same
fields were re-imaged at 4x; reported as M_4x, secondary); K excludes "F2 Zoom
bunch" (re-image of F2); N_CoV2 excludes "F10 diff conc" (different concentration).
C: "growing with time" is reported as the infected-minus-mock median difference
at 12, 24 and 48 hpi, not tested.

**Overall test (primary):** of the 19 firm predictions, the number whose medians
differ in the predicted direction, against 50% by chance (one-sided binomial).
The tool **passes** if that is significant at 0.05 **and** at least half of the
firm comparisons reach p < 0.05 individually. Tentative predictions are reported
the same way, separately. Comparisons share control groups (F, G mocks; C, D by
day), so they are not independent; the binomial is descriptive of that.

## Reported with every number

Per group: fields, scored, refused by reason, rescue used (as is / background /
black point), median and IQR, bands; whether the rescue mix differs between the
two groups (chi-square) and, if it does, the comparison within fields given the
same rescue. Pool-level: of all 831 files, how many pass the gate as is, after
each rescue, and refused. Results committed unedited.

## Known limits, stated now

- Same lab, microscope and staining as training (other days only), so this tests
  generalisation across experiments and days, not instruments.
- Many groups are small (3–5 fields: K, L, M, N_LC3, C, D).
- MitoGFP-RFP (mitophagy reporter) and mTS-GFP are scored on GFP, a different
  label from the MitoTracker the index was trained on.
- Infection is not confirmed per field; "infected" means the well was infected.
- Known limitation of MitoFI 1.0.0: densely packed puncta on bright cytoplasm read too
  elongated (public sets, `public_sets_2026-10/DIAGNOSIS.md`).
