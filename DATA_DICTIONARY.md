# Data dictionary

Every table in this repository that the manuscript draws on. Field identifiers
(`img_xxxxxx`) are the random six-character IDs assigned at blinding; no file here
links an identifier to an experimental condition.

## Pilot images and labels (`data/`)

The 500 pilot fields are maximum-intensity projections of the mitochondrial
channel (Olympus FV3000), one OME-TIFF per field, named `<image_id>.ome.tif`
(deposited separately; see the manuscript's Data availability statement).

### `labels_blind.csv`, `labels_rater2.csv`, `labels_rater3.csv`
One row per field and observer (observers 1, 2 and 3).

| column | meaning |
|---|---|
| `image_id` | blinded field identifier |
| `blinded_call` | `elongated`, `normal` (the intermediate category), `fragmented`, or `uncertain` (signal not readable as mitochondrial morphology) |
| `date_rated` | date of the rating |
| `notes` | optional free-text or tags (`donut`, `clump`); descriptive only, not model targets |

### `labels_consensus.csv`
| column | meaning |
|---|---|
| `image_id` | blinded field identifier |
| `blinded_call` | reference label: two-of-three majority (`elongated`, `normal`, `fragmented`); `uncertain` if any observer said uncertain; `split` for a three-way disagreement |
| `r1`, `r2`, `r3` | the three observers' calls |

### `covariates.csv`
| column | meaning |
|---|---|
| `image_id` | blinded field identifier |
| `session` | imaging session (`S01`–`S36`; one acquisition day) |
| `px_um` | lateral pixel size, µm |
| `sizeZ` | number of Z planes projected |
| `z_step_um`, `z_range_um` | Z step and range, µm, where recorded |
| `px_mean` | mean intensity of the source stack, where recorded |

### `features_v6.csv`
Mask features for all 500 fields, measured on masks from the released pixel
classifier (v6). One row per field. The nine model features are in **bold**.

| column | meaning |
|---|---|
| `image_id`, `px_um` | identifier; pixel size (µm) |
| `frame_px` | pixels in the field |
| `fg_frac`, `fg_area_um2` | foreground fraction; foreground area (µm²) |
| `n_objects`, `obj_per_um2_fg` | retained objects (area ≥ 0.05 µm²); objects per µm² of foreground |
| `med_area_um2`, `p90_area_um2` | median and 90th-percentile object area |
| `med_perim_um`, `p90_perim_um` | median and 90th-percentile object perimeter |
| `med_major_um`, `p90_major_um` | median and 90th-percentile major-axis length |
| **`med_aspect`**, **`p90_aspect`** | median and 90th-percentile aspect ratio (major / max(minor, 1 px)) |
| `med_solidity`, `p90_solidity` | object solidity |
| **`med_ecc`**, **`p90_ecc`** | median and 90th-percentile eccentricity |
| `med_extent`, `p90_extent` | object extent |
| `med_formfactor`, `p90_formfactor` | form factor 4πA/P² |
| **`areawt_aspect`**, **`areawt_formfactor`** | area-weighted mean aspect ratio and form factor |
| **`frac_fg_largest`** | fraction of retained foreground in the largest object |
| `frac_fg_over_2um2`, `frac_obj_over_2um2` | foreground and object fractions above 2 µm² |
| `n_holes`, `frac_obj_with_hole` | enclosed holes; fraction of objects with a hole |
| **`hole_area_frac`** | area added by hole filling / foreground area |
| `med_hole_um2`, `med_hole_roundness`, `n_holes_round` | hole size and shape |
| `skel_len_um`, `skel_len_per_um2_fg` | total skeleton length; per µm² of foreground |
| `endpoints_per_100um`, `junctions_per_100um` | skeleton endpoints and junctions per 100 µm |
| **`mean_skel_len_per_obj_um`** | skeleton length per retained object, µm |
| `fg_mean_int`, `img_mean_int` | mean intensity under the mask; of the field |
| `haze`, `local_contrast`, `local_snr` | quality-gate measurements (see the manuscript, Quality gate) |

`features_v5.csv` holds the same measurements on masks from the earlier
classifier, used for the original feature screen.

### `index_v8_oof.csv`
Session-held-out predictions of index model v8 for the 444 majority-reference fields.

| column | meaning |
|---|---|
| `image_id`, `session` | identifier; imaging session |
| `blinded_call` | majority reference label |
| `index` | out-of-fold MitoFI index |
| `pred` | class with the highest predicted probability |
| `refused` | whether the current quality gate refuses the field |

### Segmentation annotations
| file | content |
|---|---|
| `labelings/montage.tif.labeling` | Labkit scribbles (JSON; coordinates `[x, y]`) on the montage of eight pilot fields |
| `montage_tiles.csv` | where each pilot field sits in the montage: `tile_index`, `col`, `row`, `x`, `y`, `width`, `height`, `image_file` |

The ilastik scribble project used for the released classifier is deposited with
the images.

### Other tables
| file | content |
|---|---|
| `feature_ranking.csv` | original feature screen: `feature`, effect size `eps2`, `p`, Spearman with pixel size `r_px` and intensity `r_int`, `monotone`, class medians `med_F`, `med_N`, `med_E` |
| `cell_counts.csv` | cells per field: `n_cells`, `n_cells_unclipped`, `frac_area_largest_cell`, `frac_area_clipped` |

## MitoFI output (every `results*.csv` under `validation/`)

| column | meaning |
|---|---|
| `image_id` | file name without extension |
| `qc_pass`, `qc_reason` | whether the field passed the quality gate; refusal reason |
| `index` | MitoFI index, 0 (elongated) to 1 (fragmented); empty when refused |
| `band` | `elongated` (< 0.108), `mixed`, `fragmented` (> 0.869) |
| `confidence` | highest class probability |
| `index_unvalidated` | the index computed for a refused field, reported separately |
| `fg_frac`, `p90_major_um`, `n_objects` | mask summaries |
| `local_contrast`, `haze`, `local_snr` | quality-gate measurements |
| `px_um` | pixel size used |
| `rescue` | empty, `background` or `black point` |
| `reduction` | how the file was reduced to one field (axes, channel, projection, bit-depth scaling) |

`*_summary.csv` gives counts scored and refused (by reason), rescues used, and
the mean, median, SD and range of the index. `*.run.json` records the tool
version, settings, input folders and both model hashes.

## Locked-test inputs (`validation/`)

| file | columns |
|---|---|
| `allen_mitosis_1050_2026-10/inputs.csv` | Allen cell metadata: `CellId`, `group`, `cell_stage`, `FOVId`, plate and well, imaging mode, `scale_micron`, crop paths, `matched_to` (same-field interphase partner) |
| `allen_mitosis_1050_2026-10/results.csv` | per cell, for the whole crop (`whole_*`) and inside the cell outline (`cell_*`): pass, reason, index, band, rescue, haze, local SNR |
| `scribbles_v2_retrain/allen_inputs.csv`, `allen_a_v5.csv` | fresh 238-cell Allen sample and its results (same layout) |
| `public_sets_2026-10/inputs.csv` | `dataset`, `group`, `source` file, `prepared` file, `px_um`, `shape`, `note` |
| `olympus_holdout_2026-10/inputs.csv` | `rel` (archive path), `folder`, `name`, acquisition `day`, `px`, `Z`, `C`, `channel_ex` (nm), `comparison`, `side` (first/second group), `direction`, `confidence` |
