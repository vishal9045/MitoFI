"""Make the pixel classifier's features dimensionless, and see what it buys.

The index works because its features are dimensionless: aspect ratio and
eccentricity do not change when the lamp is brighter. The pixel classifier's
features are the opposite -- every Gaussian, difference-of-Gaussians, gradient
and Hessian response scales linearly with intensity, which is why its masks
correlate +0.84 with image brightness.

Rescaling the image was tried and failed (norm_experiment.py): a dim field
contains less resolvable structure, and stretching it amplifies noise. This is a
different intervention. Instead of changing the input, divide each filter
response by a local intensity estimate, so the feature is a *ratio* and carries
no units. Multiply the whole image by two and these features do not move.

    ratio    filter(I, s) / (gauss(I, 4s) + eps)
    plus     I / (gauss(I, 4s) + eps)

Evaluated on two axes that do not reference the v5 masks as truth:

    held-out tile accuracy   train on 6 montage tiles, test on the scribbles of
                             the 2 held out. Non-circular: it scores against the
                             human annotation, not against Labkit's output.
    brightness dependence    correlation of foreground fraction with log mean
                             image intensity. Lower is better.

Agreement with v5 is reported too, but as a sanity check, not a target -- v5 is
the thing being improved on.

    .venv/bin/python scripts/invariant_features.py --selftest
    .venv/bin/python scripts/invariant_features.py --run 40
"""
import argparse, csv, os, sys
import numpy as np
import tifffile
from scipy import ndimage as ndi, stats
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit_port import spec, scribbles, iou            # noqa: E402
from norm_experiment import tiles                        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260905
EPS = 1e-3          # on a 0..1 scale after the local estimate; keeps flat background finite


def stack_raw(img, sigmas):
    """The Labkit-equivalent stack: every response proportional to intensity."""
    im = img.astype(np.float32)
    yield 'identity', im
    for s in sigmas:
        g = ndi.gaussian_filter(im, s)
        yield f'gauss{s:g}', g
        yield f'dog{s:g}', g - ndi.gaussian_filter(im, s / 2)
        yield f'gradmag{s:g}', ndi.gaussian_gradient_magnitude(im, s)
        yield f'log{s:g}', ndi.gaussian_laplace(im, s)
        e = hessian_matrix_eigvals(hessian_matrix(im, sigma=s, order='rc',
                                                  use_gaussian_derivatives=True))
        yield f'hess{s:g}_1', e[0]
        yield f'hess{s:g}_2', e[1]


def stack_invariant(img, sigmas):
    """The same responses, each divided by a local intensity estimate.

    The divisor is a wide Gaussian at four times the feature's own scale, so it
    estimates the local background level without cancelling the structure the
    feature is measuring. Every output is a ratio and therefore unitless.
    """
    im = img.astype(np.float32)
    im = im / max(np.percentile(im, 99.9), 1.0)      # put values on a common order
    for s in sigmas:
        local = ndi.gaussian_filter(im, 4 * s) + EPS
        g = ndi.gaussian_filter(im, s)
        yield f'rel_int{s:g}', im / local
        yield f'rel_gauss{s:g}', g / local
        yield f'rel_dog{s:g}', (g - ndi.gaussian_filter(im, s / 2)) / local
        yield f'rel_gradmag{s:g}', ndi.gaussian_gradient_magnitude(im, s) / local
        yield f'rel_log{s:g}', ndi.gaussian_laplace(im, s) / local
        e = hessian_matrix_eigvals(hessian_matrix(im, sigma=s, order='rc',
                                                  use_gaussian_derivatives=True))
        yield f'rel_hess{s:g}_1', e[0] / local
        yield f'rel_hess{s:g}_2', e[1] / local


STACKS = {'raw': stack_raw, 'invariant': stack_invariant}


def training_rows(stack, sigmas, keep_tiles):
    assert keep_tiles, 'no tiles selected'
    montage = tifffile.imread(os.path.join(ROOT, 'data', 'montage.tif'))
    lab, _ = scribbles()
    classes = spec()[2]
    X, y = [], []
    for t in tiles():
        if t['tile_index'] not in keep_tiles:
            continue
        x0, y0, w, h = t['x'], t['y'], t['width'], t['height']
        sub = montage[y0:y0 + h, x0:x0 + w]
        planes = [p for _, p in stack(sub, sigmas)]
        for ci, name in enumerate(classes):
            key = next(k for k in lab if k.lower() == name.lower())
            c = lab[key]
            m = (c[:, 0] >= x0) & (c[:, 0] < x0 + w) & (c[:, 1] >= y0) & (c[:, 1] < y0 + h)
            if not m.any():
                continue
            loc = c[m] - np.array([x0, y0])
            r = loc[:, 1], loc[:, 0]
            X.append(np.stack([p[r] for p in planes], axis=1))
            y.append(np.full(m.sum(), ci))
    assert X, f'no scribbles found in tiles {sorted(keep_tiles)}'
    return np.concatenate(X), np.concatenate(y)


def segment(img, rf, stack, sigmas):
    planes = [p.ravel() for _, p in stack(img, sigmas)]
    return rf.predict(np.stack(planes, axis=1)).reshape(img.shape).astype(bool)


def run(n):
    sigmas = spec()[0]
    tl = tiles()
    idx = [t['tile_index'] for t in tl]
    rng = np.random.default_rng(SEED)
    held = set(rng.choice(idx, 2, replace=False).tolist())
    train_t = set(idx) - held
    print(f'{len(tl)} tiles: training on {sorted(train_t)}, holding out {sorted(held)}\n')

    imgs = os.path.join(ROOT, 'data', 'images')
    masks = os.path.join(ROOT, 'data', 'masks_batch_v5')
    ids = sorted(f[:-len('.ome_segmentation.tif')] for f in os.listdir(masks)
                 if f.endswith('.ome_segmentation.tif'))
    ids = list(np.random.default_rng(SEED).choice(ids, min(n, len(ids)), replace=False))

    out, fitted = {}, {}
    for name, stack in STACKS.items():
        Xtr, ytr = training_rows(stack, sigmas, train_t)
        Xte, yte = training_rows(stack, sigmas, held)
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
        acc = rf.score(Xte, yte)
        full = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                      random_state=SEED).fit(*training_rows(stack, sigmas, set(idx)))
        fitted[name] = (full, stack)
        out[name] = dict(held_out_acc=acc, n_features=Xtr.shape[1])
        print(f'{name:<10} {Xtr.shape[1]:>3} features   held-out tile accuracy {acc:.4f}')

    rows = []
    for k, i in enumerate(ids, 1):
        img = tifffile.imread(os.path.join(imgs, i + '.ome.tif'))
        ref = tifffile.imread(os.path.join(masks, i + '.ome_segmentation.tif')).astype(bool)
        r = dict(image_id=i, mean_int=float(img.mean()), fg_v5=float(ref.mean()))
        for name, (rf, stack) in fitted.items():
            got = segment(img, rf, stack, sigmas)
            r[f'fg_{name}'] = float(got.mean())
            r[f'iou_{name}'] = iou(got, ref)
        rows.append(r)
        if k % 10 == 0:
            print(f'  {k}/{len(ids)}', flush=True)

    dst = os.path.join(ROOT, 'data', 'invariant_features.csv')
    with open(dst, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    li = np.log([r['mean_int'] for r in rows])
    print(f"\n{'model':<12}{'held-out acc':>14}{'brightness r':>15}{'median fg':>12}{'IoU vs v5':>12}")
    fg5 = np.array([r['fg_v5'] for r in rows])
    print(f"{'labkit v5':<12}{'-':>14}{stats.pearsonr(li, fg5).statistic:>+15.2f}"
          f"{np.median(fg5):>12.4f}{1.0:>12.3f}")
    for name in STACKS:
        fg = np.array([r[f'fg_{name}'] for r in rows])
        print(f"{name:<12}{out[name]['held_out_acc']:>14.4f}"
              f"{stats.pearsonr(li, fg).statistic:>+15.2f}{np.median(fg):>12.4f}"
              f"{np.median([r[f'iou_{name}'] for r in rows]):>12.3f}")
    print("\nheld-out acc scores the classifier against human scribbles on images it")
    print("never trained on. brightness r should fall towards zero. IoU against v5")
    print("is a sanity check, not a target -- v5 is what is being improved on.")
    print(f"-> {dst}")


def selftest():
    sigmas = spec()[0]
    a = np.zeros((80, 80), np.float32); a[30:50, 36:44] = 800.0
    a += 20.0                                    # a background pedestal
    b = a * 3.0                                  # identical scene, three times brighter

    ra = {k: v for k, v in stack_raw(a, sigmas)}
    rb = {k: v for k, v in stack_raw(b, sigmas)}
    assert not np.allclose(ra['gauss2'], rb['gauss2']), 'raw stack should scale with intensity'

    ia = {k: v for k, v in stack_invariant(a, sigmas)}
    ib = {k: v for k, v in stack_invariant(b, sigmas)}
    for k in ia:
        d = np.abs(ia[k] - ib[k]).max()
        assert d < 1e-3, (k, d)                  # the whole point: unitless
    assert len(ia) == 7 * len(sigmas), len(ia)
    assert np.isfinite(np.stack(list(ia.values()))).all()

    flat = {k: v for k, v in stack_invariant(np.zeros((32, 32), np.float32), sigmas)}
    assert np.isfinite(np.stack(list(flat.values()))).all(), 'must survive an empty image'
    print(f'selftest ok: {len(ia)} invariant features, unchanged under a 3x intensity scaling')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--run', type=int, metavar='N')
    a = ap.parse_args()
    if a.selftest: selftest()
    elif a.run: run(a.run)
    else: ap.print_help()
