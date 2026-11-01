"""A Python reimplementation of the Labkit pixel classifier.

Labkit stores its feature specification and its scribbles as plain JSON. Only
the trained forest is closed -- a Java-serialised Weka FastRandomForest. So
rather than parse that, this retrains an equivalent forest in Python from the
same scribbles and the same feature stack, and measures how far the result
drifts from the Labkit masks.

The benchmark to judge that drift against is already measured: classifiers v4
and v5, made by one person from the same images with the same tool and differing
only in where the scribbles went, agree on 67% of segmented pixels. A port that
lands well above that is disagreeing with Labkit by less than annotation alone
already does.

    .venv/bin/python scripts/labkit_port.py --selftest
    .venv/bin/python scripts/labkit_port.py --train
    .venv/bin/python scripts/labkit_port.py --compare 40
"""
import argparse, json, os, sys

import numpy as np
import tifffile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFIER = os.path.join(ROOT, 'data', 'classifiers', 'classifier_v5_combined.classifier')
MONTAGE = os.path.join(ROOT, 'data', 'montage.tif')
LABELING = os.path.join(ROOT, 'data', 'labelings', 'montage.tif.labeling')
# written straight into the package: the model the tool serves and the model
# this script fits are the same file, so a retrain cannot leave the two apart
MODEL_OUT = os.path.join(ROOT, 'mitofi', 'models', 'pixel_classifier_v5.joblib')


def spec(path=CLASSIFIER):
    """Read Labkit's own feature specification rather than assuming it."""
    d = json.load(open(path))
    g = d['features']['globals']
    ops = [o['class'].rsplit('.', 1)[-1] for o in d['features']['ops']]
    return g['sigmas'], ops, d['classNames']


# feature_stack lives in the installed package, which is the implementation the
# tool ships. Two copies of the same filter bank could drift apart without either
# failing a test.
sys.path.insert(0, ROOT)
from mitofi.segment import feature_stack  # noqa: E402


def sample_at(img, coords, sigmas):
    """Feature vectors at the given (x, y) pixels, one filter plane at a time."""
    r = coords[:, 1], coords[:, 0]          # Labkit writes [x, y]
    cols, names = [], []
    for name, plane in feature_stack(img, sigmas):
        cols.append(plane[r]); names.append(name)
    return np.stack(cols, axis=1), names


def scribbles(path=LABELING):
    d = json.load(open(path))
    return {k: np.array(v, dtype=int) for k, v in d['labels'].items()}, d['interval']


def train():
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    sigmas, ops, classes = spec()
    print('sigmas', sigmas, '| classes', classes)
    print('ops   ', ', '.join(ops))

    img = tifffile.imread(MONTAGE)
    lab, interval = scribbles()
    print(f'montage {img.shape}, labeling interval {interval["max"]}')
    for k, v in lab.items():
        print(f'  {k:<14} {len(v):>7} scribbled pixels')

    X, y = [], []
    for cls_i, name in enumerate(classes):
        key = next(k for k in lab if k.lower() == name.lower())
        f, names = sample_at(img, lab[key], sigmas)
        X.append(f); y.append(np.full(len(f), cls_i))
    X = np.concatenate(X); y = np.concatenate(y)
    print(f'{X.shape[0]} training pixels, {X.shape[1]} features')

    # Weka's FastRandomForest defaults: 200 trees, sqrt(n) features per split.
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=20260905)
    rf.fit(X, y)
    joblib.dump({'rf': rf, 'sigmas': sigmas, 'classes': classes, 'features': names}, MODEL_OUT)
    print(f'out-of-sample is not meaningful on scribbles; train accuracy '
          f'{rf.score(X, y):.4f}')
    print(f'-> {MODEL_OUT}')


def segment(img, M):
    planes = [p for _, p in feature_stack(img, M['sigmas'])]
    X = np.stack([p.ravel() for p in planes], axis=1)
    pred = M['rf'].predict(X).reshape(img.shape)
    return pred.astype(bool)          # class 1 = Mitochondria


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 1.0


def compare(n):
    import joblib
    M = joblib.load(MODEL_OUT)
    imgs = os.path.join(ROOT, 'data', 'images')
    masks = os.path.join(ROOT, 'data', 'masks_batch_v5')
    ids = sorted(f[:-len('.ome_segmentation.tif')] for f in os.listdir(masks)
                 if f.endswith('.ome_segmentation.tif'))
    rng = np.random.default_rng(20260905)
    ids = list(rng.choice(ids, min(n, len(ids)), replace=False))

    rows = []
    for k, i in enumerate(ids, 1):
        img = tifffile.imread(os.path.join(imgs, i + '.ome.tif'))
        ref = tifffile.imread(os.path.join(masks, i + '.ome_segmentation.tif')).astype(bool)
        got = segment(img, M)
        rows.append(dict(image_id=i, iou=iou(got, ref),
                         fg_labkit=ref.mean(), fg_python=got.mean()))
        if k % 10 == 0:
            print(f'  {k}/{len(ids)}', flush=True)
    import csv
    out = os.path.join(ROOT, 'data', 'port_iou.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    v = np.array([r['iou'] for r in rows])
    fl = np.array([r['fg_labkit'] for r in rows]); fp = np.array([r['fg_python'] for r in rows])
    print(f'\nIoU against Labkit v5 over {len(v)} images')
    print(f'  median {np.median(v):.3f}   IQR {np.percentile(v,25):.3f}-{np.percentile(v,75):.3f}'
          f'   min {v.min():.3f}')
    print(f'  above 0.80: {(v>0.8).mean():.0%}    above 0.67 (the v4/v5 benchmark): {(v>0.67).mean():.0%}')
    print(f'foreground fraction: Labkit median {np.median(fl):.4f}, Python median {np.median(fp):.4f}')
    print(f'-> {out}')


def write_masks(n, outdir):
    """Segment N images with the Python classifier and save the masks, so the
    downstream index can be computed from them and compared."""
    import joblib
    M = joblib.load(MODEL_OUT)
    imgs = os.path.join(ROOT, 'data', 'images')
    masks = os.path.join(ROOT, 'data', 'masks_batch_v5')
    ids = sorted(f[:-len('.ome_segmentation.tif')] for f in os.listdir(masks)
                 if f.endswith('.ome_segmentation.tif'))
    rng = np.random.default_rng(20260905)          # same draw as --compare
    ids = list(rng.choice(ids, min(n, len(ids)), replace=False))
    os.makedirs(outdir, exist_ok=True)
    for k, i in enumerate(ids, 1):
        got = segment(tifffile.imread(os.path.join(imgs, i + '.ome.tif')), M)
        tifffile.imwrite(os.path.join(outdir, i + '.ome_segmentation.tif'),
                         got.astype(np.uint8), compression='zlib')
        if k % 10 == 0:
            print(f'  {k}/{len(ids)}', flush=True)
    print(f'{len(ids)} masks -> {outdir}')


def selftest():
    sigmas, ops, classes = spec()
    assert sigmas == [1.0, 2.0, 4.0, 8.0], sigmas
    assert classes == ['background', 'Mitochondria'], classes
    assert len(ops) == 6, ops

    img = np.zeros((64, 64), np.uint16); img[20:44, 30:34] = 4000
    names = [n for n, _ in feature_stack(img, sigmas)]
    assert len(names) == 1 + 6 * len(sigmas), (len(names), names)
    assert len(set(names)) == len(names)

    # A bright bar curves sharply across its width and not along its length, so
    # the across-bar second derivative is strongly negative while the along-bar
    # one is flat. skimage sorts eigenvalues descending, so that response lands
    # in the SECOND plane, not the first -- checking the first passes trivially
    # on an all-zero background and tests nothing.
    planes = dict(feature_stack(img, sigmas))
    assert planes['hess1_2'][32, 32] < -100, planes['hess1_2'][32, 32]
    assert abs(planes['hess1_1'][32, 32]) < 1e-6, planes['hess1_1'][32, 32]
    for k in ('log1', 'gradmag1', 'dog1'):
        assert abs(planes[k][32, 32]) > 100 and planes[k][5, 5] == 0, k

    lab, interval = scribbles()
    assert set(lab) == {'background', 'Mitochondria'}, set(lab)
    assert lab['Mitochondria'].shape[1] == 2
    assert lab['background'].max() <= interval['max'][0]
    print('selftest ok:', len(names), 'features,',
          sum(len(v) for v in lab.values()), 'scribbled pixels')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--compare', type=int, metavar='N')
    ap.add_argument('--write-masks', type=int, metavar='N')
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'masks_python_v5'))
    a = ap.parse_args()
    if a.selftest: selftest()
    elif a.train: train()
    elif a.compare: compare(a.compare)
    elif a.write_masks: write_masks(a.write_masks, a.out)
    else: ap.print_help()
