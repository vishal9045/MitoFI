# MitoFI

A continuous, calibrated score for how fragmented the mitochondria in a 2D
confocal fluorescence image are.

```
0.0 ─────────────────────────────────────── 1.0
elongated network                     fragmented
```

It is an **instrument for measuring morphology**, calibrated against the
majority call of three trained observers who each rated the same 500 images
blind. It is not an assay for a treatment, and was never trained on any
experimental condition; it was tested on held-out images against expected
effects (see `PAPER.md`).

The score comes from a model with no tunable threshold: no watershed, no
object-splitting parameter, nothing a user can adjust until the answer looks
right.

## Install

```bash
pip install mitofi
```

The trained models ship with the package. Nothing else to download, and no Fiji.
This build carries `index_model_v8.json` (nine features, SHA-256
`4dfdd1c6…`) and `pixel_classifier_v6.joblib` (SHA-256 `820cd485…`). Results are written beside a
`.run.json` recording both model hashes, because a version string alone does
not identify which model produced a number.

Python 3.10 or newer, with scikit-learn 1.5 or newer; verified on scikit-learn
1.5.2, 1.7.2 and 1.9.0. The pixel classifier is a pickled random forest and so
does depend on that library, but the index model is stored as plain numbers, so
the number this tool reports does not change with the version installed.

If `mitofi` is not found afterwards, pip put it in a scripts directory that is
not on your PATH — common on Windows. This always works instead:

```bash
python -m mitofi score images/ --suffix .tif
```

To build a wheel for a colleague, from a checkout:

```bash
pip wheel . --no-deps -w dist/      # one file, models included
```

## Use

### A window, if you prefer

```bash
mitofi gui
```

Choose a folder and press Run. Everything else is read from the files: it lists
the TIFFs, reports how many carry a pixel size, and offers the channels **by
name** in a dropdown, preselecting one that looks mitochondrial. You only type
something to override the metadata, or when a file records no pixel size.

The channel is matched by name in each file, not by position, so a folder whose
files export the same dye at different indices still scores the right stain --
and the window says so when it finds more than one layout.

Results are written to `mitofi.csv` in the image folder as the run goes, so a
window closed halfway leaves nothing to redo: reopen the folder and **Resume**
is offered, already ticked, saying how many are done. The status line carries
the count, the time spent and an estimate of the time left. **Save a copy…**
writes the same two files anywhere else.

Results appear as a table, and the preview follows the run, showing each
segmentation as it is scored; clicking a row stops it there. Scroll to zoom,
drag to pan, double-click to fit -- at 1:1 and above you are looking at the
mask's own pixels, which is the only way to see the two failures that matter: a
mask that fuses neighbouring tubules, and one that traces noise. The panel grows
with the window.

Look at a few before trusting the numbers: every failure found while developing
this index was caught by looking at a mask, not by reading a metric.

Save CSV writes the same two files as the command line, into the image folder.

### Command line

```bash
# raw OME-TIFF stacks: pixel size and layout come from the file
mitofi score images/ --suffix .ome.tif --channel 2

# a plain 2D image with the pixel size given
mitofi score images/ --px-um 0.0829

# pixel size differs per image and is not in the files
mitofi score images/ --px-um-csv sizes.csv       # columns: image_id,px_um

# you already have masks: give the images too, so the gate stays whole
mitofi score images/ --masks masks/ --mask-suffix _mask.tif

# masks only
mitofi segment images/ --channel 2

# a long run: pick up where an interrupted one stopped, and leave cores free
mitofi score images/ --resume --jobs 4
```

A folder of 500 takes an hour or so. While it works, a bar on stderr shows how
many are done, how long it has taken and how long is left; the results table on
stdout stays clean, so piping it to a file is unaffected. Results are written
after every image -- through a temporary file, so an interruption cannot damage
what is already there -- and `--resume` continues rather than repeats.

`--resume` will not carry over rows it cannot vouch for. Beside the results CSV
the tool writes `<results>.csv.run.json`, recording the version, the channel,
the timepoint, the input folders and the SHA-256 of both models. Resuming after
any of those changed stops with a message naming what changed, rather than
mixing two analyses in one file; rows for images that are no longer in the input
folder are dropped, and rows whose pixel size no longer matches are re-scored.
Delete the CSV to start over.

### What it does with a raw acquisition

The index is defined on one 2D field: a maximum-intensity projection of the
mitochondrial channel. A raw OME-TIFF is usually more than that, so the tool
reduces it and records what it did in a `reduction` column:

```
axes ZCYX; channel 2 of 4; max projection over 9 planes
```

- **Z is maximum-projected**, matching how the pilot's images were made.
- **A channel must be named** with `--channel` when the file holds more than
  one. The tool will not pick for you: choosing the wrong channel produces a
  confident, entirely wrong number.
- **Pixel size is read from the OME metadata** when present, in whatever unit
  the file declares. `--px-um` overrides it. If neither exists the tool stops —
  the index is in physical units and is meaningless without a scale.
- **Give it raw images, not contrast-adjusted ones.** Do not press Auto or
  Apply in Fiji's Brightness/Contrast first. Stretching a hazy image without
  removing the haze makes the cytoplasm look like mitochondria: the whole cell
  is masked as one mesh, scores near 0 (fully elongated), and the quality gate
  cannot tell that mesh from a dense network, so it passes. On 238 stem-cell
  images, 163 were scored that way. If a field is refused, use the tool's own
  rescues, which remove the haze before anything is stretched.

From Python:

```python
from mitofi import segment, score_mask
import tifffile

mask = segment(tifffile.imread('field.tif'))
print(score_mask(mask, px_um=0.0829))
# {'qc_pass': True, 'index': 0.812, 'band': 'mixed', 'confidence': 0.71, ...}
```

## Output

Two files, written into the folder the images came from unless `-o` says
otherwise: `mitofi.csv` has one row per image, `mitofi_summary.csv` has one row
for the whole run — count scored, count rejected, and the mean, median, standard
deviation and range of the index.

**On the bands.** They describe where the number falls, not a proven morphology.
The two ends are high-purity zones: in the development pool, over 90% of images
beyond each cut carried the matching majority call of the three raters. `mixed` is the
remainder, and it is not homogeneous — it contained fragmented, normal and
elongated fields alike. Read the index; treat the band as a coarse summary of it.

**Averages cover the images that were scored.** A rejected image is counted in
`n_rejected` but not averaged: the tool declined to score it, so folding in a
number it refused to report would be inventing data.

| column | meaning |
|---|---|
| `index` | 0 elongated to 1 fragmented. **Empty when the image fails QC.** |
| `band` | `fragmented`, `mixed`, or `elongated` |
| `confidence` | the model's probability for its top class |
| `qc_pass`, `qc_reason` | whether the image was scorable, and why not |
| `index_unvalidated` | the raw score for a rejected image, reported separately |

## The quality gate, and why you cannot skip it

The index cannot tell a genuinely fragmented cell from an image too poor to
read. Both produce many small disconnected objects, and the model scores the
second one **confidently**.

So a gate runs first, and a rejected image gets no index at all. The rules apply
in this order:

| rule | meaning |
|---|---|
| pixels coarser than 0.2 µm | coarser than any training image |
| no objects in the mask | nothing segmented |
| more than 6% of the signal under the mask is smooth background | too much diffuse background |
| foreground below 1% | the segmentation collapsed |
| objects at least 2.25 µm long that stand less than 1.16× their own surrounding noise | the mask looks like noise texture |
| objects both under 2.25 µm **and** under 4 background SD above their surroundings | the mask is speckle |

**Diffuse background.** Images processed at acquisition, as the development
images were, carry almost none; the 6% limit was set on readable development
fields. Images
straight off some microscopes carry a lot: mitochondria riding on bright
cytoplasmic haze. The classifier then segments the cytoplasm as one textured
object, which the size rule used to wave through and score as confidently
elongated. Such fields are now refused with that reason. After a run, the tool
offers to remove the background from **every field the gate refused**, whatever
the reason, and score them again -- a dialog in the GUI, a prompt in a terminal.
Fields that passed are never touched. Or remove it for every field up front:

```bash
mitofi score images/ --channel 0 --subtract-background        # rolling ball, 2.5 µm
```

The radius is in micrometres and converted with each field's own pixel size.
Rows scored this way say so in their `reduction` column. Removing background is
a choice about your images, so it is offered, never applied silently.

**Black point, the second rescue.** Some unprocessed images carry grainy
cytoplasmic signal rather than smooth haze, which a rolling ball leaves in
place. For fields still refused after background removal, the tool then offers
to set a black point at the cytoplasm level -- what setting the minimum in
Fiji's Brightness/Contrast does, with the level read from each image (the most
common brightness of the cell area). To accept both rescues in advance:

```bash
mitofi score images/ --channel 0 --rescue
```

Every row says which rescue it had in the `rescue` column (empty, `background`
or `black point`). The black point keeps the order between fields that all had
it, but reads about 0.2 higher than untouched fields (median +0.21 to +0.24 on
fields that already pass), so compare black-point rows with each other, or check
that a difference holds without them. When a run mixes rescued and untouched
fields the tool says so at the end, and the summary counts them
(`n_rescued_background`, `n_rescued_black_point`).

**Bit depth** is handled for you. The classifier was trained on 12-bit data;
8-, 14-, 16-bit and floating-point fields are put on that scale first, using
the depth the file records or, when it records none, the smallest standard
depth that holds the data. 12-bit data is left exactly as it is.

On the development pool, the current gate rejected 14 of 35 images the
first observer marked unreadable and 12 of the 465 they could read. Those figures were
measured on the same pool used to choose the thresholds, so they describe
development performance rather than independent validation.

**Small objects alone are not grounds for rejection.** An earlier version cut on
size, and on MFN1 knockout cells — where losing a fusion protein makes
mitochondria genuinely small — it discarded three of five images that scored 0.93
to 0.98 when allowed through. It was rejecting the phenotype the tool exists to
measure.

Size cannot separate the two cases: in that comparison, genuinely fragmented
images spanned 1.57–2.51 µm and unreadable ones 1.51–2.92 µm. Local contrast
can: 10.4 against 2.0. So a mask is called speckle only when its objects are
both small and barely above the background touching them.

`--masks` on its own supplies no image, so contrast cannot be measured and the
size rule applies alone -- the rule that was rejecting the phenotype. On the
MFN1-null fields that costs three images of five. Pass the image folder as well
as `--masks` and the gate works as it does everywhere else:

```bash
mitofi score images/ --masks masks/ --mask-suffix _mask.tif   # gated on contrast
mitofi score --masks masks/ --px-um 0.0829                    # size-only gate
```

The tool warns on screen when it is scoring masks alone, and warns again if the
gate rejected more images than it scored -- an average over the survivors of a
mass rejection describes the survivors, not the experiment.

**What the gate is not.** It applies the rules above and nothing else.
There is no saturation, blur, wrong-channel, out-of-focus or
not-actually-mitochondria detector, and a synthetic all-foreground field passes
it with an index of 1.0. `qc_pass` means "passed these rules", not "this is a
readable image of mitochondria". Look at the masks.

### Choosing the channel by laser line

Channel names are typed at acquisition and can be wrong. On one Leica export
the channel named "Mito Tracker Red" was excited at 514 nm while the
mitochondria were in another channel, excited at 594 nm and named "EGFP",
"DAPI" or nothing. Choose by laser line instead:

```bash
mitofi score images/ --channel-ex 594
```

It finds the channel excited at that wavelength in each file, so files that
order their channels differently still have the same stain scored, and it
refuses a file with no such channel, or two. In the GUI the channel list shows
each named channel's laser line and offers "excitation ... nm" entries
alongside the names.

### What it refuses to read

Rather than guess, the tool stops and says why:

| input | why it is refused |
|---|---|
| an RGB or multi-sample TIFF | a colour composite is not a channel stack; export the mitochondrial channel on its own |
| a file recording non-square pixels (X 0.1 µm, Y 0.2 µm) | every feature here assumes square pixels; resample first |
| a pixel size in a unit it does not know | it used to be multiplied by 1.0 and treated as micrometres |
| a pixel size that is zero, negative, or infinite | the index is in physical units; infinity used to pass the gate and return `NaN` |
| a file holding several image series | it will not pick a series for you |
| a mask holding more than two distinct values | a multi-class label image is not a binary mask |
| more than one channel, with no `--channel` | position is not identity: the same experiment exported CH1/CH3/CH4 in one file and CH1/CH2/CH3 in another |

## What it is validated on, and what that means for you

**Development** (500 fields, Olympus FV3000, 36 imaging sessions, pixel sizes
0.046–0.196 µm; three blinded raters):

- Held out by imaging session, it reproduces the raters' two-of-three call with
  **0.723 accuracy** against a 0.482 majority-class baseline, on the 444 fields
  every rater could read and at least two agreed on (balanced accuracy 0.683).
- Between the two ends of the scale it separates their calls with **AUC 0.978**.
- Calls made with probability 0.85 or above were right in 140 of 152 fields
  (92.1%; mean stated probability 92.9%).
- Held out in turn, each rater matched the other two's agreed call on
  77.7–80.5% of fields; the model, refitted without that rater, on 74.5–78.2%.
  The differences (1.6–3.1 points) are within the session-bootstrap intervals.
- The index is weakly associated with pixel size (Spearman ρ = +0.13) and mean
  image intensity (ρ = −0.15).

**Locked tests**, each scored once under a plan committed before scoring (plans,
selections and unedited results under `validation/`; summary in `PAPER.md`):

| set | comparison | result |
|---|---|---|
| Allen Institute hiPSC, TOMM20, 1,050 cells | mitotic > interphase | 0.246 vs 0.097, P = 2 × 10⁻¹⁶ |
| BBBC053, TOM20, 58 fields | FCCP > vehicle | 0.309 vs 0.107, P = 0.0007 |
| Mito-Hacker, mito-YFP, 93 cells | Drp1−/− < Drp1+/+ | 0.002 vs 0.253, P = 4 × 10⁻⁶ |
| MitoTracker (Zeiss LSM780), 64 fields | FCCP > no FCCP | 0.017 vs 0.002, P = 0.0008 |
| In-house Olympus archive, 828 fields, 86 new days | 9 directional predictions | 9 with P < 0.05 |

Limits worth knowing before you trust a number:

- **Dense puncta and granular signal can fuse.** Densely packed puncta on bright
  cytoplasm (e.g. depolarised cells stained with a potential-dependent dye) and
  granular reporter signal can be segmented as one connected network, which
  reads as elongated and passes the gate. Look at the masks in such images.
- **It scores the whole field.** An observer may judge one cell; in crowded
  fields, crop or mask single cells before scoring.
- **Rescued fields read differently.** The black point raises the index by about
  0.2; compare rescued fields with fields given the same rescue.
- **The index model and band cuts were fitted on one microscope**, and the
  features, gate thresholds and band cuts were chosen on the development pool.
  On a new instrument, check masks and, if possible, a comparison with a known expected effect.

## Segmentation

The bundled pixel classifier (v6) is a random forest of 200 trees on 25
image-feature planes, trained on hand-drawn scribbles: the original Labkit
montage of Olympus fields, plus ilastik scribbles on 30 fields chosen by image
condition (brightness, haze, noise, density) and morphology from Olympus, Leica
and Allen Institute images, with background drawn inside cells. Training pixels
are weighted so that no single field or image source dominates. An earlier
classifier, a Python port of the Labkit model (median IoU 0.931 against the
Labkit masks), responded to absolute brightness and failed on dim or hazy
images from other microscopes; v6 replaced it.

## What is in this repository

The tool is `mitofi/` and its two trained models. Everything else is the record
behind the paper, and none of it is needed to run the tool:

```
mitofi/        the package: CLI, GUI, segmentation, features, gate, index
tests/         59 checks, including one that scores a committed image end to end
scripts/       how the models were fitted and the paper's numbers and figures made
data/          labels, features, covariates and the original scribbles
validation/    every locked test: frozen plan, file selection, scripts, results
notebooks/     a worked example, and the Labkit-to-Python port
notes/         the rating rubric and the blinded pilot protocol
```

`DATA_DICTIONARY.md` describes every table. `PAPER.md` maps every number, table and figure in the manuscript to the file or
script that produces it. The full development record, including approaches that
were tried and rejected, is available from the authors on request.

## Acknowledgement

MitoFI was designed by Vishal Sah. Anthropic's AI assistant Claude helped write and test the code, check it
against real and synthetic data, and package it as a Python tool. Every analysis choice (what is measured, thresholds
and defaults, and how results are reported) was reviewed and decided by the author, who takes responsibility for it.
If you find a problem, please open an issue.
