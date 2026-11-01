# Stage 3 rating rubric — v1.0 FROZEN, 05 September 2026

Reconstructed from rater 1's 162 free-text notes and the images they refer to,
then corrected by rater 1.

**This file is frozen.** Do not edit it while any rating is in progress. If it
must change, close the round, version the file, and start a new round, because a
rubric that shifts mid-round makes the two raters incomparable.

---

## What you are looking at

One 2D maximum-intensity projection of a confocal stack, mitochondrial channel
only. Contrast is scaled per image, so absolute brightness carries no meaning —
do not read "brighter" as "more mitochondria".

A field may contain one cell, several cells, or cells clipped by the frame edge.

**There is no scale bar, and pixel size varies about four-fold across the pool.**
Judge object shape and the relation of objects to each other *within the image
you are looking at*. Never judge "long" or "short" against a remembered image.

Mitochondria are normally excluded from the nuclear region, so most fields show
a ring or crescent of signal around a dark centre. That is ordinary cell
anatomy. It is not a morphology phenotype and it is not a "donut".

---

## The call — exactly one per image, mandatory

Describe the population that **dominates** the field.

| Call | Meaning |
|---|---|
| `fragmented` | Majority of objects are small, discrete, rounded or short rods. Little or no connected tubular network. |
| `elongated` | Majority of objects are long and tubular, frequently branched or joined into a continuous network. |
| `normal` | Both types are present and neither dominates. A genuinely mixed population. |
| `uncertain` | The signal cannot be read as mitochondrial morphology. |

Two points that generate most disagreement, so read them twice:

**`normal` means mixed, not healthy.** It is a morphology class, not a
condition, not a control, not a judgement of quality. Rater 1 wrote this class
as "mixed" throughout the notes. If a field sits between fragmented and
elongated, it is `normal`.

**`uncertain` is about unreadable signal, not about intermediate morphology.**
Use it in two situations: the signal is too dim or too low in contrast for object
shape to be made out at all, or the signal does not look mitochondrial (saturated
undifferentiated blobs, diffuse haze, staining that could be something else).

A field whose morphology is clearly visible but sits between fragmented and
elongated is **`normal`**, never `uncertain`. `uncertain` means you cannot see
well enough to judge; `normal` means you can see perfectly well and the answer is
"both".

Both readings are kept deliberately. Rater 1's notes on uncertain images all say
some version of "uncertain if this is mito signal", which is the identity
reading, while rater 1 later described the class as the dimmest fields where
morphology cannot be made out, which is the legibility reading. Measurement
supports keeping both: those 35 images share one signature after segmentation --
the mask is speckle, with a 90th-percentile object length of 1.90 um against
3.79 um for readable images. Whether the cause was dimness or a signal that was
not mitochondrial, the consequence is the same and the call is the same.

---

## Decision rules

1. **Zoom before you call.** Use at least 2x on a representative region. At fit
   view on a 512-pixel field the difference between a short rod and a punctum is
   not resolvable, and a rater who calls from the fit view is not performing the
   same task as one who zooms.
2. **Judge objects, not brightness or area.** A few large bright structures do
   not outweigh many small ones.
3. **Tags never drive the call.** Decide the call first, then tag.
4. **Dominance is majority by object count.** Count, not area and not brightness.
   A field of many small objects and a few long ones is `fragmented`, however much
   of the cell the long ones cover.
5. **One cell per field.** Judge the dominant cell, which is usually the central
   one. Do not average across cells when a field contains several.
6. **Ignore cells clipped by the frame edge.** They are not the subject of the
   call even when they occupy more of the frame than the central cell.
7. **Too dim to read is `uncertain`.** If object shape cannot be made out, the
   call is `uncertain` regardless of how much signal is present.

---

## Tags — optional, any number, do not advance the image

| Tag | Meaning |
|---|---|
| `donut` | Individual mitochondria forming a closed ring with a visible dark lumen. An object-level shape. |
| `clump` | The brightest signal in the image: a dense, saturated aggregate in which individual objects cannot be separated. |

`donut` is **not** the ring the whole population forms around the nucleus. That
arrangement appears in most fields, including ones rater 1 did not tag.

`clump` is judged on brightness relative to the rest of the same image, not against
other images — display contrast is scaled per image, so absolute grey values are
not comparable between fields. These structures are taken to be aggregated
mitochondria on the basis of a separate study by the same group; the tag records
the appearance, and the interpretation sits outside this pilot.

Tags record **presence, not majority**. A handful of donuts in an otherwise
fragmented field is tagged `donut` and called `fragmented`. This is the pattern
throughout rater 1's notes: the call is the majority, the tag is the minority
subpopulation.

Tags are descriptive. They are not labels, they are not scored, and they do not
feed the classifier. They mark fields a later per-object analysis will have to
handle.

---

## Notes

Free text, optional. Rater 1's convention was `"<tag> in <call>"`, e.g.
"Donut in fragmented", "Clumps among elongated". Useful, keep it.

**Press the tag button; do not only type the word.** On 11 images rater 1 typed
"Donut in fragmented" without pressing the tag, so the flag is missing while the
prose says otherwise. The exported CSV records tags as a prefix in the notes
field, so an unpressed tag is invisible to analysis.

---

## Worked examples

To be added. The 127 QC-rejected images cannot serve as anchors: they were
rejected for unusable signal and show nothing worth learning from.

Anchors must therefore come from the pool, so they are handled in two ways that
keep the second rating clean:

- **Tag anchors are zoomed crops**, not whole fields. A crop showing one donut or
  one clump teaches the object shape without revealing the field's call, so the
  image it came from can still be rated blind.
- **Call anchors are whole fields, held out.** Twelve images, three per call, are
  shown to the second rater as training and then removed from their rating set.
  Agreement is computed on the remaining 488. Rater 1's calls on the twelve remain
  valid for every other purpose.

Each anchor carries one sentence saying why it is that class and not the
neighbouring one.

---

## Procedure for a second rater

- Use Chrome or Firefox. Safari blocks browser storage on local files and you
  will lose everything.
- Keep `blind_rater.html` in the same folder as `rating_png/`, or no images load.
- Export the CSV every ~100 images, not only at the end.
- Return the CSV unmodified. Do not open it in Excel and re-save — Excel mangles
  the identifiers and the dates.
- Do not discuss individual images with rater 1 until both sets are complete.
