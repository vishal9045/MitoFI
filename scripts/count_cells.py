"""Estimate how many cells are in each field, from the mitochondrial mask alone.

The rating rubric says the call describes the dominant, usually central cell and
ignores cells clipped by the frame edge. The features, by contrast, are computed
over every object in the frame. On a multi-cell field the label and the features
therefore describe different things. This script measures how often that happens
and whether the model's errors concentrate there.

Method: mitochondria of one cell are close together and separated from the next
cell by empty cytoplasm-free space, so closing the mask with a disk of a few
micrometres merges each cell's organelles into a single blob. Everything is done
at a fixed physical scale, not in pixels, so it behaves the same at every
magnification in the pool.

    .venv/bin/python scripts/count_cells.py
"""
import csv, os, sys
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import binary_closing, disk, remove_small_objects
from skimage.transform import rescale

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASKS = os.path.join(ROOT, 'data', 'masks_batch_v5')
OUT = os.path.join(ROOT, 'data', 'cell_counts.csv')

TARGET_UM = 0.5     # work at this pixel size regardless of the source
CLOSE_UM = 4.0      # disk radius: bridges gaps within a cell, not between cells
MIN_CELL_UM2 = 100  # a blob smaller than this is debris, not a cell


def cells(mask, px_um):
    scale = px_um / TARGET_UM
    small = rescale(mask.astype(float), scale, order=0, anti_aliasing=False) > 0.5
    closed = binary_closing(small, disk(int(round(CLOSE_UM / TARGET_UM))))
    closed = ndi.binary_fill_holes(closed)          # the nucleus is a hole
    closed = remove_small_objects(closed, int(MIN_CELL_UM2 / TARGET_UM ** 2))
    lab, n = ndi.label(closed)
    if n == 0:
        return 0, 0, 0.0, 0.0
    areas = ndi.sum(closed, lab, range(1, n + 1)) * TARGET_UM ** 2
    border = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    clipped_area = sum(a for i, a in enumerate(areas, 1) if i in border)
    return n, n - len(border), float(areas.max() / areas.sum()), \
        float(clipped_area / areas.sum())


def main():
    cov = {r['image_id']: r for r in csv.DictReader(open(os.path.join(ROOT, 'data', 'covariates.csv')))}
    rows = []
    for k, i in enumerate(sorted(cov), 1):
        p = os.path.join(MASKS, i + '.ome_segmentation.tif')
        if not os.path.exists(p):
            continue
        n, nfull, frac_largest, frac_clipped = cells(tifffile.imread(p).astype(bool),
                                                    float(cov[i]['px_um']))
        rows.append(dict(image_id=i, n_cells=n, n_cells_unclipped=nfull,
                         frac_area_largest_cell=round(frac_largest, 4),
                         frac_area_clipped=round(frac_clipped, 4)))
        if k % 100 == 0:
            print(f"  {k}/{len(cov)}", flush=True)
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import collections
    print("cells per field:", dict(sorted(collections.Counter(r['n_cells'] for r in rows).items())))
    print(f"wrote {len(rows)} rows -> {OUT}")


def audit():
    """Do the model's errors concentrate on fields containing several cells?"""
    import pandas as pd
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    MAIN = ['fragmented', 'normal', 'elongated']
    CLEAN = ['p90_aspect', 'p90_ecc', 'areawt_formfactor', 'med_aspect', 'med_ecc',
             'hole_area_frac', 'areawt_aspect', 'frac_fg_largest']
    f = pd.read_csv(os.path.join(ROOT, 'data', 'features_v5.csv'))
    l = pd.read_csv(os.path.join(ROOT, 'data', 'labels_blind.csv'))
    c = pd.read_csv(OUT)
    d = f.merge(l[['image_id', 'blinded_call']], on='image_id').merge(c, on='image_id')
    m = d[d.blinded_call.isin(MAIN)].reset_index(drop=True)
    y, g = m.blinded_call.values, m.session.values
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=20260905))
    pred = cross_val_predict(pipe, m[CLEAN], y, cv=list(GroupKFold(n_splits=5).split(m, y, g)))
    m = m.assign(correct=(pred == y))

    print("accuracy by number of cells in the field")
    for n, grp in m.groupby('n_cells'):
        if len(grp) >= 10:
            print(f"  {n} cell(s)   n {len(grp):>4}   accuracy {grp.correct.mean():.3f}")
    single, multi = m[m.n_cells == 1], m[m.n_cells > 1]
    t = [[int(single.correct.sum()), int((~single.correct).sum())],
         [int(multi.correct.sum()), int((~multi.correct).sum())]]
    chi = stats.chi2_contingency(t)
    print(f"\n  single cell  n {len(single):>4}  accuracy {single.correct.mean():.3f}")
    print(f"  multi cell   n {len(multi):>4}  accuracy {multi.correct.mean():.3f}")
    print(f"  chi2 {chi.statistic:.2f}  p {chi.pvalue:.3f}")

    print("\naccuracy by the fraction of signal in cells clipped by the frame edge")
    for lo, hi in [(0, .01), (.01, .2), (.2, .5), (.5, 1.01)]:
        sel = m[(m.frac_area_clipped >= lo) & (m.frac_area_clipped < hi)]
        if len(sel) >= 10:
            print(f"  {lo:.2f}-{hi:.2f}   n {len(sel):>4}   accuracy {sel.correct.mean():.3f}")

    print("\ncall distribution by cell count (is the rater's label itself shifting?)")
    for n in sorted(m.n_cells.unique()):
        grp = m[m.n_cells == n]
        if len(grp) >= 10:
            v = grp.blinded_call.value_counts(normalize=True)
            print(f"  {n} cell(s)  " + "  ".join(f"{k} {100*v.get(k,0):.0f}%" for k in MAIN))

    # Is cell count merely a proxy for magnification? Coarser pixels blur thin
    # structures towards apparent tubules, which would produce the same shift.
    print("\nis cell count a proxy for magnification?")
    print("  n_cells vs px_um    rho %+.2f  p %.2g" % stats.spearmanr(m.n_cells, m.px_um)[:2])
    print("  n_cells vs frame    rho %+.2f  p %.2g" % stats.spearmanr(m.n_cells, m.frame_px)[:2])
    for n, grp in m.groupby('n_cells'):
        if len(grp) >= 10:
            print(f"  {n} cell(s)  n {len(grp):>4}  median px_um {grp.px_um.median():.4f}"
                  f"  frame {int(grp.frame_px.median() ** 0.5)}px")
    print("\n  elongated rate split at the median pixel size, within each cell count")
    med = m.px_um.median()
    for n in sorted(m.n_cells.unique()):
        grp = m[m.n_cells == n]
        for lab, sel in [('fine px', grp[grp.px_um <= med]), ('coarse px', grp[grp.px_um > med])]:
            if len(sel) >= 10:
                print(f"    {n} cell(s) {lab:<10} n {len(sel):>4}"
                      f"   elongated {100 * (sel.blinded_call == 'elongated').mean():.0f}%")
    print("  A shift that survives this split is not a magnification effect.")


def selftest():
    px = 0.2
    # two blobs 30 um apart: must read as two cells, neither touching the border
    m = np.zeros((400, 400), bool)
    yy, xx = np.mgrid[:400, :400]
    m |= np.hypot(yy - 150, xx - 120) < 40
    m |= np.hypot(yy - 250, xx - 300) < 40
    n, nfull, _, clipped = cells(m, px)
    assert (n, nfull) == (2, 2), (n, nfull)
    assert clipped == 0.0, clipped
    # one blob running off the edge: counted, and flagged as clipped
    m2 = np.zeros((400, 400), bool); m2[:120, 100:260] = True
    n2, nfull2, _, clipped2 = cells(m2, px)
    assert (n2, nfull2) == (1, 0), (n2, nfull2)
    assert clipped2 == 1.0, clipped2
    print("selftest ok")


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    elif '--audit' in sys.argv:
        audit()
    else:
        main()
