# Rescue options, one at a time: plan written before scoring

2026-10-26. DEVELOPMENT, not validation: every image used here has been scored
before. Asked for by the investigator: try each processing option the tool
could offer a refused field, on its own, and see whether it improves results.
Tool as shipped at `3ea8346` (pixel classifier v6, index v8, gate 0.06 / 1.16).

## Options (each applied to the original field, alone)

| name | what | fixed settings |
|---|---|---|
| `background` | current rescue 1: rolling ball | 2.5 um |
| `black point` | current rescue 2: floor at cytoplasm peak, stretched | as shipped |
| `offset` | camera dark level removed, no stretch | 0.5th percentile of the field subtracted, clipped at 0 |
| `ball 1um` | rolling ball, smaller | 1.0 um |
| `denoise` | non-local means | skimage `denoise_nl_means`, fast mode, patch 5, distance 6, h = 0.8 x noise SD (median absolute deviation of horizontal neighbour differences / sqrt 2) |
| `deconv` | Richardson-Lucy, 2D, Gaussian PSF | 20 iterations; PSF sigma = 0.21 x emission / NA, in pixels; Allen 510 nm / NA 1.25; Leica and Olympus 600 nm / NA 1.4 (assumed: not recorded) |
| `slab` | thin projection instead of the full one (3D data only) | max over planes within 1 um of the sharpest plane (largest variance of the Laplacian of Gaussian, sigma 2 px); Leica and Allen only |

Cell cropping (Cellpose) is left out: installing it pulls numpy 2 into this
environment. Allen's own cell outline already shows the effect of cropping
(the "inside the cell" analysis).

## Data

- **Safety (D23)**: fields that pass as is. Olympus: 100 drawn at random (seed
  20261026) from the development fields that pass under v6. Leica: every series
  that passes as is (px <= 0.2 um).
- **Rescue**: Leica, every series refused as is (px <= 0.2 um); Allen, the 238
  cells of `scribbles_v2_retrain/allen_inputs.csv` (now development data),
  inside Allen's cell outline as the primary view, whole crop as well.

## Reported, per option

1. **Safety**: fields still passing; index shift against as is (median, 90th
   percentile of |shift|); band changes. **An option passes D23 if the median
   |shift| <= 0.03 and the band changes in <= 10% of fields**, on Olympus and on
   Leica separately. (The cytoplasm floor failed at +0.15, the black point at +0.12.)
2. **Leica rescue**: refused series it lets through; the 7 within-file
   comparisons of `scribbles_v2_retrain/PLAN.md` using as-is fields plus fields
   this option rescued; and the same with the option in the black point's place
   (as is, then background, then the option).
3. **Allen**: cells passing with this option alone; mitotic vs interphase
   (one-sided Mann-Whitney), same-field pairs (Wilcoxon), median by stage.

No option is wired into the tool by this run. What to offer is decided after,
from the table, and recorded in the CHANGES document.

## Step 2, added 2026-10-26 after step 1's Allen result, before scoring it

Step 1 on Allen: every option that removes the floor without stretching
(background, offset, 1 um ball) leaves the cells too dim ("almost nothing
segmented"); denoise, deconvolution and slab leave the haze ("diffuse
background"). Only the black point, which also stretches, rescues (211/238).
Three options test whether the stretch, not the cut at the cytoplasm peak, is
what works:

| name | what | fixed settings |
|---|---|---|
| `imagej auto` | ImageJ Brightness/Contrast "Auto" + Apply | `ContrastAdjuster.autoAdjust`, first click: 256 bins min..max, bins over 10% of pixels skipped, first bin from each end over pixels/5000 sets min and max; linear map to 0..4095, clipped |
| `offset+stretch` | camera dark level cut, then stretched | min = 0.5th percentile; max = the Auto maximum; same map |
| `slab+black point` | thin projection, then the shipped black point | slab as in step 1, then `subtract_floor(peak=True)`; Leica and Allen only |

Same data, same reports, same D23 rule. Run: `run.py olympus|leica|allen set2`.

**Change, 2026-10-26 15:30, after the step-1 Olympus safety result and before any
Leica scoring:** step 1 on Leica stopped (offset, denoise, deconvolution and slab
alone were dead ends on Allen; the investigator asked why run them). Step 2 carries
the baselines it needs instead: as is, background, black point, 1 um ball.

## Slab thickness, added 2026-10-27 before scoring

Slab + black point gave the strongest Allen result (mitotic 0.427 vs
interphase 0.118). Risk: mitotic cells are rounded, and a thin slab through a
rounded cell may cut tubules running in Z into short pieces, reading as
fragmentation. `slab_thickness.py`: the 238 Allen cells, slab 0.5, 1, 2 and 4 um
thick around the sharpest plane, and the full stack, each followed by the
shipped black point, inside the cell outline.

**Rule, fixed now:** if the mitotic median AND the interphase median each move by
<= 0.05 between the 2 um and 4 um slabs, the effect has plateaued (overlap
removed, not tubules cut) and slab + black point may be offered as a third rescue
for Z stacks. Otherwise it is not offered. All thicknesses reported.

## Step 3, added 2026-10-27 before scoring

`background+stretch`: rolling ball 2.5 um, then the ImageJ Auto stretch (as in
`rescue_recovery_2026-10`, where it was unbiased on dim fields with a camera
offset at every strength, -0.007 to -0.001, and on light-to-medium haze).
Proposed as an additional rescue, not a replacement for the black point. Same
data, same reports, same D23 rule. Run: `run.py olympus|leica|allen set3`.
