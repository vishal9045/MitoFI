# All .lif files: score everything, fixed before scoring

Written 2026-10-23 and committed with `inventory.csv`, `lif_sha256.txt`,
`OmeDump.java` and `run.py` before any series was scored. Asked for by the
investigator: every series in every file, no selection, to see how the tool
behaves across many stimuli.

## Data

21 `.lif` files in `testing_data/test lifs`, 496 series. All 8-bit, Leica
SP8/DMI6000 confocal, 0.045-0.24 um/px, 1-128 Z planes, no mosaics, no
time-lapses. Several sessions were seen before (MEF WT/MFN1/MFN2, Mdivi-1,
set 2); they are included like the rest.

## Channel, fixed per series in `inventory.csv`

- **laser line** (376): the one channel excited at 550-600 nm (561 or 594 nm),
  from Bio-Formats OME-XML (`OmeDump.java`).
- **by eye, per file** (119): five files record no 550-600 nm line (CDKi +
  CoV-2, NEM 2 quant, nem quant 3, mavs ko veh mdivi, the 3-channel series of
  mdivi mef wt and mavs ko). Contact sheets of two series per file showed
  mitochondria in channel 1 and nuclei in channel 2 in all five; channel 1 is
  used for every series in them.
- **single channel** (1).
- Contact sheets of every file confirmed the laser-line pick where it applies.
  One picked channel is blank (CDKi s0); it is scored like the rest.

## Conditions

`run.condition`: the series name with numbering, zoom, replicate, batch and
position tags removed and spelling variants unified ("cdk i"/"cdki",
"220 mg"/"220mg", "1ko"/"mfn1 ko"). Word-order variants and unclear names
("mark_and_find", "series001", "mef ko") are kept verbatim. Results are
reported per file and condition; nothing is pooled across files.

## What is run

The tool at this commit (index model v7), per series: maximum projection over
Z at timepoint 0, scaled from the true bit depth, segmented and scored; a field
the gate refuses is re-scored after the background rescue (2.5 um rolling
ball). No other correction. Every series is reported, including refusals,
read errors and blank or nuclei-only series.

## What is reported

Per file and condition: series, scored as is, scored after the rescue,
refusals by reason, median and IQR of the index, band counts. Then an outline
check on a random sample (seed 20261023, two scored series per file) so the
reliability of the numbers is judged on series not chosen by us.

## Known limits

- All data are 8-bit Leica, the image type on which the tool has so far
  segmented bright cytoplasm and swollen bodies as mitochondria (set 2,
  Mdivi-1); values may be compressed toward elongated.
- Many conditions are infected or drug-treated; fields are not biological
  replicates.
