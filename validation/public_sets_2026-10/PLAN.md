# Three public datasets scored with MitoFI 1.0.0 — plan frozen before scoring

Written 2026-10-28, committed with `prepare.py`, `inputs.csv`, `run.py` and
`summarise.py` before any image is scored. Asked for by the investigator: test the
released tool on public images it has never seen, from other labs, microscopes
and markers. None of these images, labs or microscopes shaped any part of MitoFI.

## Tool

MitoFI **1.0.0** (tag `v1.0.0`): pixel classifier v6 (sha `820cd485…`), index v8
(sha `4dfdd1c6…`), gate haze 0.06 / local SNR 1.16. Run through the command line exactly as a
user would: `mitofi score <folder> --rescue` (background removal, then black
point, for refused fields only; marked in the `rescue` column). `run.py` checks
version and model hashes first.

## Data and preparation (`prepare.py`, format changes only)

| set | source | cells, marker | microscope | pixel size | groups (n) |
|---|---|---|---|---|---|
| **BBBC053** | Broad BBBC053 (Edwards et al. 2021, *Patterns*), CC BY-NC-SA 3.0 | mouse CAD cells, TOM20 | Nikon confocal 60x/1.49, max projections, deconvolved | 0.0201 um (file) | DMSO 29, FCCP 29 |
| **Mito-Hacker** | GitHub Mitogenie/Mito-Hacker `Sample_images` (Rohani et al. 2020, *Sci Rep*), MIT | mouse pancreatic cancer lines with 0 / 1 / 2 copies of Drp1 (KPDC145 / 143 / 253), mito-YFP, fixed | Zeiss LSM 710 63x, single slice, 8-bit | not recorded; see below | single-cell crops 33 / 34 / 26; fields 7 / 4 / 6 |
| **Cape Town** | Figshare doi:10.25375/uct.28531679, CC BY 4.0 | cell type not stated; MitoTracker (CMXRos per description, CMTMRos per file metadata) | Zeiss LSM780 63x/1.4, z-stacks | 0.0659 um (file) | per repeat A / C / D: control 6 / 5 / 3, FCCP 6 / 5 / 5, propionic acid 24 h 6 / 5 / 5, 72 h 5 / 5 / 5 |

- BBBC053: published files scored as they are; a 4x block-mean copy (0.0803 um,
  inside the classifier's training range 0.046–0.196 um) scored as a secondary run,
  because 0.0201 um is finer than anything the classifier was trained on.
- Mito-Hacker: RGB composites are refused by the tool by design, so the green
  (mito-YFP) channel is saved on its own, as the README tells users. The single-cell
  crops are the 2048 px fields shrunk 2x by Mito-Hacker and centred on one cell
  (checked by registration, r = 0.995), other cells set to 0.
- Cape Town: `.czi` is not read by the tool; the channel excited at 561 nm
  (MitoTracker) is saved with its z-stack, pixel size and z-step; the tool
  max-projects it. Two "slow" files are second scans of fields already in the
  set and are left out. `AP72_a.czi` is 0 bytes in the deposited zip
  (unreadable, reported). Repeat D has three files labelled "CT" beside the "C"
  controls; their meaning is not stated, so they are not used in the test and are
  reported on their own.

**Mito-Hacker pixel size (an assumption, stated now).** The paper gives the
microscope and objective but not the zoom. Primary: fields 0.066 um (LSM 710,
63x, zoom 1, 2048 px), so crops 0.132 um. Every image gets the same value, so
the comparison between lines does not depend on it; the absolute index does.
Sensitivity runs with fields 0.044 / 0.099 um (crops 0.088 / 0.198 um), x2/3 and
x1.5 (x2 would put crops beyond the 0.2 um refusal).

## Predictions (one-sided Mann-Whitney on images, alpha 0.05)

1. **BBBC053 (primary: published files):** FCCP above DMSO.
2. BBBC053, 4x copy: FCCP above DMSO.
3. **Mito-Hacker (primary: single-cell crops at 0.132 um):** Drp1-/- (KPDC145)
   below Drp1+/+ (KPDC253) -- loss of fission, more elongated.
4. Mito-Hacker crops: index rises with Drp1 copy number 0 < 1 < 2 (one-sided
   Spearman correlation of index with copy number).
5. Mito-Hacker fields (0.066 um): prediction 3 again (n 7 vs 6, low power).
6. **Cape Town (primary):** FCCP above control, pooled over the three repeats
   (controls: A "AC", C "CC", D "C").
7. Cape Town, per repeat: FCCP median above control median in each of A, C, D.

Each set passes on its primary prediction (1, 3, 6). Propionic acid: no
prediction; reported. Sensitivity: prediction 3 at the other two pixel sizes;
prediction 6 with the repeat D "CT" files counted as controls.

## Reported with every number

Per group: images, scored, refused by reason, rescue used (as is / background /
black point), median and IQR, bands. The rescue mix is compared between the
groups of each test (chi-square); if it differs, the test is repeated within
images given the same rescue. Unreadable files are listed, never dropped. Results
committed unedited.

## Known limits, stated now

- **BBBC053 brightness:** 16-bit files whose values reach only ~1,700; the tool
  scales 16-bit data by 1/16, so the classifier sees them as very dim. Many may be
  refused or need the black point; that is what a user would get.
- **BBBC053 finer pixels** than any training image (handled by the 4x copy).
- **Mito-Hacker:** cells come from few fields (7 / 4 / 6), so cells are not fully
  independent; the field-level test (5) is the check. The crops were resampled by
  another tool; the pixel size is assumed.
- **Cape Town:** MitoTracker loading depends on membrane potential, which FCCP
  collapses; dim FCCP fields may be refused. Cell type not stated; few fields per
  repeat.
