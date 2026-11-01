"""Stage 5: per-image morphology features from the v5 masks.

Physical units throughout, using each image's own px_um. No watershed and no
instance-splitting parameter: objects are connected components of the mask as
segmented, nothing is split or merged after the fact.

Writes data/features_v5.csv, one row per image.
"""
import csv, os, sys

import numpy as np
import tifffile

# The feature code lives in the installed package, which is the canonical
# implementation the tool ships. Keeping a second copy here meant the research
# record and the tool could drift apart silently.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitofi.features import features                            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASKS = os.path.join(ROOT, 'data', 'masks_batch_v5')
IMAGES = os.path.join(ROOT, 'data', 'images')
COVAR = os.path.join(ROOT, 'data', 'covariates.csv')
OUT = os.path.join(ROOT, 'data', 'features_v5.csv')


def main():
    cov = {r['image_id']: r for r in csv.DictReader(open(COVAR))}
    rows, empty = [], []
    ids = sorted(cov)
    for k, i in enumerate(ids, 1):
        mp = os.path.join(MASKS, i + '.ome_segmentation.tif')
        ip = os.path.join(IMAGES, i + '.ome.tif')
        if not os.path.exists(mp):
            empty.append((i, 'mask missing')); continue
        mask = tifffile.imread(mp).astype(bool)
        img = tifffile.imread(ip) if os.path.exists(ip) else None
        px = float(cov[i]['px_um'])
        f = features(mask, img, px)
        if f.get('n_objects', 0) == 0:
            empty.append((i, 'no objects'))
        f['image_id'] = i
        f['session'] = cov[i]['session']
        f['px_um'] = px
        f['sizeZ'] = cov[i]['sizeZ']
        rows.append(f)
        if k % 50 == 0:
            print(f"  {k}/{len(ids)}", flush=True)

    cols = ['image_id', 'session', 'px_um', 'sizeZ'] + [c for c in rows[0] if c not in
            ('image_id', 'session', 'px_um', 'sizeZ')]
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval='')
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows, {len(cols)} columns -> {OUT}")
    if empty:
        print("images with no usable objects:", empty)


def selftest():
    px = 0.1
    # one 40x4 px bar: aspect must be ~10, form factor low, no holes
    m = np.zeros((80, 80), bool); m[20:60, 38:42] = True
    f = features(m, None, px)
    assert f['n_objects'] == 1, f['n_objects']
    assert 8 < f['med_aspect'] < 13, f['med_aspect']
    assert f['med_formfactor'] < 0.6, f['med_formfactor']
    assert f['n_holes'] == 0
    assert abs(f['fg_area_um2'] - 40 * 4 * px * px) < 1e-9

    # one ring: exactly one hole, roundness high
    yy, xx = np.mgrid[:80, :80]
    r = np.hypot(yy - 40, xx - 40)
    m = (r < 20) & (r > 12)
    f = features(m, None, px)
    assert f['n_holes'] == 1, f['n_holes']
    assert f['med_hole_roundness'] > 0.7, f['med_hole_roundness']
    assert f['frac_obj_with_hole'] == 1.0

    # physical units must not depend on pixel size: same object, pixels half the size
    a1 = features(np.pad(np.ones((40, 4), bool), 20), None, px)['med_area_um2']
    a2 = features(np.pad(np.ones((80, 8), bool), 40), None, px / 2)['med_area_um2']
    assert abs(a1 - a2) < 1e-9, (a1, a2)
    print("selftest ok")


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main()
