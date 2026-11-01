# Blind rating — instructions

Thank you for doing this. It takes roughly two to three hours for 500 images,
and it does not have to be done in one sitting.

## What this is for

We are building a tool that scores mitochondrial fragmentation automatically.
It was trained on one person's judgement. Your ratings are not used to train it.
They are used to measure how well two trained observers agree with each other,
which sets the ceiling on how well any automatic method could possibly do.

So please rate as you see it. Do not try to guess what the first rater said, and
do not try to be consistent with anyone but yourself.

## Setup

1. Unzip everything into one folder. Keep `blind_rater_r2.html` in the **same
   folder as** `rating_png/`, or no images will load.
2. Open `blind_rater_r2.html` by double-clicking it.
3. **Use Chrome or Firefox. Not Safari** — Safari blocks local storage on files
   opened from disk, and you would lose everything you had done.
4. Read `rating_rubric.md` before you start. It defines the four calls and the
   two tags. It matters more than it looks: most disagreement between raters
   comes from different readings of the words, not from different eyes.

## Rating

- One call per image, by key or click: **F**ragmented, **N**ormal/mixed,
  **E**longated, **U**ncertain. The call advances to the next image.
- **D** and **C** tag donuts and clumps. Tags do not advance — press a call
  afterwards.
- Zoom with **Z**, **+**, **-**, or the scroll wheel, up to 4x. Drag to pan.
  Please zoom at least once per image; at fit view a short rod and a dot look
  the same.
- **Backspace** undoes the previous image.
- The note field is optional. Typing in it pauses the keyboard shortcuts, and
  the footer turns amber to tell you so; press Esc or click a call to resume.

## Your progress is saved automatically

Progress lives in your browser, so you can close the page and come back.

Two warnings:

- **Do not use a private/incognito window,** and do not clear site data.
- The tool downloads a backup CSV automatically every 50 images. Keep those
  files. If anything goes wrong, the most recent one is your recovery point.

## When you finish

Press **Export CSV** and send back the file it saves.

Please return it **unchanged**. Do not open it in Excel and re-save it — Excel
reformats the identifiers and the dates and the file becomes unusable.

## Please do not

- Discuss individual images with the other rater until both sets are complete.
  Once we have compared them, disagreements are exactly what we want to talk
  about.
- Rate the same image twice deliberately. The tool prevents it anyway.
