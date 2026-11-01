"""Does normalising each image before feature extraction remove the classifier's
dependence on brightness?

The two annotation tasks in this project saw different things. The rater judged
8-bit renderings stretched per image, so their calls are largely blind to
absolute brightness. The pixel classifier trained on raw 16-bit values, so it can
use absolute intensity as a cue -- and the masks correlate +0.84 with image
brightness, of which -0.24 survives into the index.

This trains two classifiers on the same scribbles and differs in one thing:

    raw     features from raw values          (reproduces current behaviour)
    norm    features from per-image scaled    (what the rater effectively saw)

and reports, for each: agreement with the Labkit masks, and the correlation
between image brightness and how much foreground the classifier finds. The
second is the number the experiment exists to move.

The montage scribbles are split back into per-image annotations using
montage_tiles.csv, so each tile can be scaled on its own statistics rather than
the whole canvas's.

    .venv/bin/python scripts/norm_experiment.py --selftest
    .venv/bin/python scripts/norm_experiment.py --run 40

RESULT: the hypothesis is not supported. Over 40 images, per-image scaling moved
the brightness correlation from +0.87 to +0.83 and cost 16 points of agreement
with the Labkit masks (median IoU 0.924 -> 0.762), while finding MORE foreground
in dim images (median 0.0785 -> 0.0890).

The likely reason: a dim image does not merely look dim, it contains less
resolvable structure. Scaling the numbers up does not create signal, it amplifies
noise, and the local texture the filters respond to changes. Brightness
dependence here is a property of the images, not a cue the classifier chose to
use, and it cannot be removed by rescaling the input.
"""
import argparse, csv, os, sys
import numpy as np
import tifffile
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit_port import feature_stack, spec, scribbles, iou   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260905


def normalise(img):
    """Scale to the 1st-99.9th percentile of this image, as the display rule did.

    The rating PNGs were stretched per image (135 distinct stretches over the
    pool), so this is the transform that puts the classifier's input on the same
    footing as what the rater actually looked at.
    """
    a = img.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.9)
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def tiles():
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'data', 'montage_tiles.csv'))))
    for r in rows:
        for k in ('x', 'y', 'width', 'height', 'tile_index'):
            r[k] = int(r[k])
    return rows


def per_image_training_data(sigmas, norm):
    """Split the montage scribbles by tile so each source image stands alone."""
    montage = tifffile.imread(os.path.join(ROOT, 'data', 'montage.tif'))
    lab, _ = scribbles()
    classes = spec()[2]
    X, y, used = [], [], 0
    for t in tiles():
        x0, y0, w, h = t['x'], t['y'], t['width'], t['height']
        sub = montage[y0:y0 + h, x0:x0 + w]
        sub = normalise(sub) if norm else sub
        for ci, name in enumerate(classes):
            key = next(k for k in lab if k.lower() == name.lower())
            c = lab[key]
            m = (c[:, 0] >= x0) & (c[:, 0] < x0 + w) & (c[:, 1] >= y0) & (c[:, 1] < y0 + h)
            if not m.any():
                continue
            local = c[m] - np.array([x0, y0])
            r = local[:, 1], local[:, 0]
            f = np.stack([p[r] for _, p in feature_stack(sub, sigmas)], axis=1)
            X.append(f); y.append(np.full(len(f), ci))
        used += 1
    return np.concatenate(X), np.concatenate(y), used


def fit(norm):
    sigmas = spec()[0]
    X, y, n = per_image_training_data(sigmas, norm)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED).fit(X, y)
    return rf, sigmas, X.shape, n


def segment(img, rf, sigmas, norm):
    im = normalise(img) if norm else img
    planes = [p.ravel() for _, p in feature_stack(im, sigmas)]
    return rf.predict(np.stack(planes, axis=1)).reshape(img.shape).astype(bool)


def run(n):
    imgs = os.path.join(ROOT, 'data', 'images')
    masks = os.path.join(ROOT, 'data', 'masks_batch_v5')
    ids = sorted(f[:-len('.ome_segmentation.tif')] for f in os.listdir(masks)
                 if f.endswith('.ome_segmentation.tif'))
    ids = list(np.random.default_rng(SEED).choice(ids, min(n, len(ids)), replace=False))

    models = {}
    for norm in (False, True):
        rf, sigmas, shape, ntiles = fit(norm)
        models['norm' if norm else 'raw'] = (rf, sigmas, norm)
        print(f"{'norm' if norm else 'raw ' } : trained on {shape[0]:,} pixels "
              f"x {shape[1]} features from {ntiles} tiles")

    rows = []
    for k, i in enumerate(ids, 1):
        img = tifffile.imread(os.path.join(imgs, i + '.ome.tif'))
        ref = tifffile.imread(os.path.join(masks, i + '.ome_segmentation.tif')).astype(bool)
        r = dict(image_id=i, mean_int=float(img.mean()), fg_labkit=float(ref.mean()))
        for name, (rf, sigmas, norm) in models.items():
            got = segment(img, rf, sigmas, norm)
            r[f'fg_{name}'] = float(got.mean())
            r[f'iou_{name}'] = iou(got, ref)
        rows.append(r)
        if k % 10 == 0:
            print(f'  {k}/{len(ids)}', flush=True)

    out = os.path.join(ROOT, 'data', 'norm_experiment.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    li = np.log([r['mean_int'] for r in rows])
    print(f"\n{'model':<8}{'median IoU':>12}{'median fg':>12}{'brightness corr':>18}")
    for name in ('labkit', 'raw', 'norm'):
        fg = np.array([r[f'fg_{name}'] for r in rows])
        rho = stats.pearsonr(li, fg).statistic
        io_ = np.median([r[f'iou_{name}'] for r in rows]) if name != 'labkit' else 1.0
        print(f"{name:<8}{io_:>12.3f}{np.median(fg):>12.4f}{rho:>+18.2f}")
    print("\nbrightness corr = Pearson r between log mean image intensity and")
    print("foreground fraction. Closer to zero is better: it means the classifier")
    print("is not deciding how much mitochondrion it sees from how bright the image is.")
    print(f"-> {out}")


def selftest():
    a = np.zeros((50, 50), np.uint16); a[10:40, 10:40] = 1000
    b = (a.astype(np.float32) * 4).astype(np.uint16)      # same scene, four times brighter
    na, nb = normalise(a), normalise(b)
    assert np.allclose(na, nb, atol=1e-5), np.abs(na - nb).max()
    assert 0.0 <= na.min() and na.max() <= 1.0
    flat = normalise(np.zeros((8, 8), np.uint16))
    assert flat.max() == 0 and np.isfinite(flat).all()
    t = tiles()
    assert len(t) >= 2 and all(x['image_file'].endswith('.tif') for x in t)
    print(f'selftest ok: normalisation is scale-invariant, {len(t)} tiles mapped')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--run', type=int, metavar='N')
    a = ap.parse_args()
    if a.selftest: selftest()
    elif a.run: run(a.run)
    else: ap.print_help()
