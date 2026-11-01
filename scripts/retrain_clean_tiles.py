"""Retrain the pixel classifier without the dim, unusable training tiles.

Two of the eight montage tiles are images the rater could not read or that carry
too little signal to annotate reliably: tile 2 at mean intensity 3, and tile 5 at
mean 8, which the rater marked uncertain. Their scribbles taught the classifier
what mitochondria look like in fields where that could not be judged.

This trains with and without them and compares, on data neither classifier has
seen: MEF wild-type against MFN1 knockout, where fusion loss should fragment.

    .venv/bin/python scripts/retrain_clean_tiles.py
"""
import glob, os, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from invariant_features import training_rows, stack_raw       # noqa: E402
from labkit_port import spec                                  # noqa: E402
from mitofi.score import load as load_index, predict_proba     # noqa: E402
from norm_experiment import tiles                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DROP = {2, 5}
SEED = 20260905


def fit(keep):
    X, y = training_rows(stack_raw, spec()[0], keep)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED).fit(X, y)
    return rf, X.shape


def segment_with(img, rf, sigmas):
    planes = [p.ravel() for _, p in stack_raw(img, sigmas)]
    return rf.predict(np.stack(planes, axis=1)).reshape(img.shape).astype(bool)


def main():
    sys.path.insert(0, ROOT)
    from mitofi import features
    from mitofi.io import to_field

    sigmas = spec()[0]
    allt = {t['tile_index'] for t in tiles()}
    models = {}
    for name, keep in [('all 8 tiles', allt), ('6 tiles, dim dropped', allt - DROP)]:
        rf, shape = fit(keep)
        models[name] = rf
        print(f'{name:<22} {shape[0]:>6,} training pixels from {len(keep)} tiles')

    M = load_index()                     # the shipped model, as plain numbers
    CLEAN, cls, LO, HI = M['features'], M['classes'], M['lo'], M['hi']

    print(f"\n{'image':<32}" + ''.join(f'{n:>24}' for n in models))
    out = {n: {'MFN1': [], 'WT': []} for n in models}
    for f in sorted(glob.glob(os.path.join(ROOT, 'data/test/*/*.tif'))):
        img, px, _ = to_field(f)
        base = os.path.basename(f)[:30]
        cond = 'MFN1' if 'MFN1' in base else 'WT'
        line = f'{base:<32}'
        for n, rf in models.items():
            m = segment_with(img, rf, sigmas)
            fe = features(m, img, px)
            p = predict_proba(M, [fe[k] for k in CLEAN])
            idx = float(p[cls.index('normal')] * 0.5 + p[cls.index('fragmented')] * 1.0)
            out[n][cond].append(idx)
            line += f"{idx:>15.3f}  p90 {fe['p90_major_um']:>4.2f}"
        print(line)

    print()
    for n in models:
        a, b = np.array(out[n]['MFN1']), np.array(out[n]['WT'])
        gap = a.min() - b.max()
        print(f'{n:<22} MFN1 {a.mean():.3f}   WT {b.mean():.3f}   '
              f'gap {gap:+.3f}   {"separated" if gap > 0 else "OVERLAP"}')


if __name__ == '__main__':
    main()
