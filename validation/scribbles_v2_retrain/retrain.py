"""Retrain the pixel classifier on the condition-chosen scribbles, as PLAN.md fixes.

    retrain.py inputs                 scribble counts + Allen fresh sample (before fitting)
    retrain.py fit a|b [--without leica|allen]
    retrain.py pixelcheck             step 1, leave one source out on the strokes
    retrain.py olympus NAME...        step 2: segment 500, refit index, re-derive gate
    retrain.py leica NAME...          step 3 (or step 4 with a no-Leica model)
    retrain.py outline NAME           two scored series per file, image beside mask
    retrain.py allen NAME...          step 5
    retrain.py summary

NAME is a fitted model (a, b, a_noleica, ...) or v5 (the shipped model).
Per-image Olympus tables carry pilot IDs, so they stay in data/scribbles_v2/
(gitignored); only summaries are written here.
"""
import csv
import hashlib
import json
import math
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
DATA = os.path.join(ROOT, 'data', 'scribbles_v2')
MODELS = os.path.join(DATA, 'models')
ILP = os.path.join(DATA, 'mitofi_scribbles.ilp')
LIFS = os.path.join(os.path.dirname(ROOT), 'testing_data', 'test lifs')
ALLEN_BASE = 'https://allencell.s3.amazonaws.com/aics/hipsc_single_cell_image_dataset/'
SEED, CAP, SIGMAS = 20260905, 2000, [1.0, 2.0, 4.0, 8.0]
CLASSES = ['background', 'Mitochondria']        # v5's order: class 1 is the mask
SOURCES = ['olympus', 'leica', 'allen']
EXCLUDED_FIELDS = {'field_08_olympus.tif'}       # not a mitochondria image
PX_V5 = 0.2


# ---------------------------------------------------------------- scribbles

def _ilp_strokes():
    """{field file: {class index: (ys, xs)}} from the ilastik project."""
    import h5py
    out = {}
    with h5py.File(ILP, 'r') as f:
        names = [n.decode() for n in f['PixelClassification/LabelNames'][()]]
        val = {names.index('Mitochondria') + 1: 1,
               next(i for i, n in enumerate(names) if n.lower() == 'background') + 1: 0}
        infos = f['Input Data/infos']
        lanes = sorted(infos)
        sets = sorted(f['PixelClassification/LabelSets'])
        assert len(lanes) == len(sets), (len(lanes), len(sets))
        for lane, ls in zip(lanes, sets):
            assert lane[-3:] == ls[-3:], (lane, ls)
            fp = infos[lane]['Raw Data/filePath'][()]
            fp = fp.decode() if isinstance(fp, bytes) else fp
            name = re.split(r'[\\/]', fp)[-1]
            ys, xs, cs = [], [], []
            for blk in f['PixelClassification/LabelSets'][ls].values():
                a = blk[()][..., 0]
                (y0, _), (x0, _) = [map(int, s.split(':')) for s in
                                    str(blk.attrs['blockSlice']).strip('[]').split(',')[:2]]
                yy, xx = np.nonzero(a)
                ys.append(yy + y0); xs.append(xx + x0); cs.append(a[yy, xx])
            ys, xs, cs = (np.concatenate(v) if v else np.array([], int) for v in (ys, xs, cs))
            out[name] = {val[v]: (ys[cs == v], xs[cs == v]) for v in val}
    return out


def fields():
    """Every labelled field: dict(name, source, path, strokes={class: (ys, xs)})."""
    from labkit_port import scribbles
    from norm_experiment import tiles
    lab, _ = scribbles()
    out = []
    for t in tiles():
        x0, y0, w, h = t['x'], t['y'], t['width'], t['height']
        st = {}
        for ci, cname in enumerate(CLASSES):
            c = lab[next(k for k in lab if k.lower() == cname.lower())]
            m = (c[:, 0] >= x0) & (c[:, 0] < x0 + w) & (c[:, 1] >= y0) & (c[:, 1] < y0 + h)
            st[ci] = (c[m, 1] - y0, c[m, 0] - x0)
        out.append(dict(name=f"montage_tile_{t['tile_index']}", source='olympus',
                        path=os.path.join(ROOT, 'data', 'images', t['image_file']), strokes=st))
    meta = {r['field']: r for r in csv.DictReader(open(os.path.join(DATA, 'fields.csv')))}
    for name, st in sorted(_ilp_strokes().items()):
        if name in EXCLUDED_FIELDS or not sum(len(v[0]) for v in st.values()):
            continue
        out.append(dict(name=name, source=meta[name]['source'],
                        path=os.path.join(DATA, 'fields', name), strokes=st))
    return out


def inputs():
    """Scribble counts and the Allen fresh sample. Fits nothing."""
    F = fields()
    L = [f'ilp sha256 {hashlib.sha256(open(ILP, "rb").read()).hexdigest()}', '',
         f'{"field":<26}{"source":<9}{"Mito":>7}{"Bg":>7}']
    tot = {}
    for f in F:
        n = [len(f['strokes'][c][0]) for c in (1, 0)]
        L.append(f'{f["name"]:<26}{f["source"]:<9}{n[0]:>7}{n[1]:>7}')
        for c, k in zip((1, 0), n):
            tot[(f['source'], c)] = tot.get((f['source'], c), 0) + k
    L += ['', 'by source: ' + ', '.join(f'{s} Mito {tot.get((s, 1), 0)} Bg {tot.get((s, 0), 0)}'
                                        for s in SOURCES)]
    open(os.path.join(HERE, 'scribbles_summary.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))

    meta = pd.read_csv(os.path.join(DATA, 'allen_tomm20_metadata.csv'))
    old = set(pd.read_csv(os.path.join(ROOT, 'validation', 'allen_mitosis_2026-10',
                                       'inputs.csv')).CellId)
    src = {int(p.split('_')[0]) for p in os.listdir(os.path.join(DATA, 'allen_src'))}
    bad_fov = set(meta[meta.CellId.isin(src)].FOVId)
    free = meta[~meta.CellId.isin(old | src) & ~meta.FOVId.isin(bad_fov)]
    rng = np.random.default_rng(20261025)
    rows, used = [], set()
    for stage in ('M1M2', 'M3', 'M4M5'):
        pool = free[free.cell_stage == stage]
        pick = pool.iloc[rng.permutation(len(pool))[:50]]
        for _, r in pick.iterrows():
            mates = free[(free.FOVId == r.FOVId) & (free.cell_stage == 'M0')
                         & ~free.CellId.isin(used)]
            if not len(mates):
                continue                         # no unused interphase mate in this field
            m = mates.iloc[rng.integers(len(mates))]
            used.add(m.CellId)
            rows.append(dict(r, group=stage, matched_to=''))
            rows.append(dict(m, group='M0', matched_to=r.CellId))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, 'allen_inputs.csv'), index=False)
    d = pd.DataFrame(rows)
    print(f'\nAllen fresh sample: {len(d)} cells', d.group.value_counts().to_dict(),
          f'| available mitotic: {free.cell_stage.isin(["M1M2", "M3", "M4M5"]).sum()}')


# ---------------------------------------------------------------- features + models

def stack_fn(kind):
    if kind == 'ratio':
        from invariant_features import stack_invariant
        return stack_invariant
    from mitofi.segment import feature_stack
    return feature_stack


def read_img(path):
    import tifffile
    a = tifffile.imread(path)
    assert a.ndim == 2, (path, a.shape)
    return a.astype(np.float32)


def training(kind):
    """(X, y, field, source), cached per stack."""
    cache = os.path.join(MODELS, f'train_{kind}.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z['X'], z['y'], z['field'], z['source']
    X, y, fl, so = [], [], [], []
    for f in fields():
        planes = [p for _, p in stack_fn(kind)(read_img(f['path']), SIGMAS)]
        for c, (ys, xs) in f['strokes'].items():
            if len(ys):
                X.append(np.stack([p[ys, xs] for p in planes], 1)); y.append(np.full(len(ys), c))
                fl += [f['name']] * len(ys); so += [f['source']] * len(ys)
        print(f'  {f["name"]}', flush=True)
    X, y, fl, so = np.concatenate(X), np.concatenate(y), np.array(fl), np.array(so)
    os.makedirs(MODELS, exist_ok=True)
    np.savez(cache, X=X, y=y, field=fl, source=so)
    return X, y, fl, so


def weights(y, fl, so):
    """min(n, CAP) per (field, class); then equal total per (source, class)."""
    w = np.zeros(len(y))
    for key in set(zip(fl, y)):
        m = (fl == key[0]) & (y == key[1])
        w[m] = min(m.sum(), CAP) / m.sum()
    for key in set(zip(so, y)):
        m = (so == key[0]) & (y == key[1])
        w[m] /= w[m].sum()
    return w * len(y) / w.sum()


def fit_rf(X, y, w):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED).fit(
        X, y, sample_weight=w)


def fit(kind, without=None):
    import joblib
    X, y, fl, so = training(kind)
    keep = so != without
    rf = fit_rf(X[keep], y[keep], weights(y[keep], fl[keep], so[keep]))
    name = {'raw': 'a', 'ratio': 'b'}[kind] + (f'_no{without}' if without else '')
    names = [n for n, _ in stack_fn(kind)(np.zeros((32, 32), np.float32), SIGMAS)]
    joblib.dump(dict(rf=rf, sigmas=SIGMAS, classes=CLASSES, features=names, stack=kind,
                     trained_on=sorted(set(so[keep])), n_pixels=int(keep.sum())),
                os.path.join(MODELS, f'pixel_{name}.joblib'))
    print(f'{name}: {keep.sum()} pixels from {len(set(fl[keep]))} fields -> pixel_{name}.joblib')


def load_pixel(name):
    import joblib
    if name == 'v5':
        from mitofi.segment import PIXEL_MODEL
        return joblib.load(PIXEL_MODEL)
    return joblib.load(os.path.join(MODELS, f'pixel_{name}.joblib'))


def segment_with(M, img):
    planes = [p.ravel() for _, p in stack_fn(M.get('stack', 'raw'))(img, M['sigmas'])]
    return M['rf'].predict(np.stack(planes, 1)).reshape(img.shape).astype(bool)


def pixelcheck():
    """Leave one source out on the strokes; field-weighted balanced accuracy."""
    import joblib
    from mitofi.segment import PIXEL_MODEL
    v5 = joblib.load(PIXEL_MODEL)['rf']
    L = ['Step 1: leave one source out on the strokes (balanced accuracy, fields equal)', '',
         f'{"held out":<10}{"fields":>7}{"(a) raw":>10}{"(b) ratio":>11}{"v5":>8}']

    def bacc(pred, y, fl):
        rec = []
        for c in (0, 1):
            per = [np.mean(pred[(fl == f) & (y == c)] == c) for f in set(fl[y == c])]
            rec.append(np.mean(per))
        return np.mean(rec)

    res = {}
    for kind in ('raw', 'ratio'):
        X, y, fl, so = training(kind)
        for s in SOURCES:
            tr, te = so != s, so == s
            rf = fit_rf(X[tr], y[tr], weights(y[tr], fl[tr], so[tr]))
            res[(kind, s)] = bacc(rf.predict(X[te]), y[te], fl[te])
            if kind == 'raw':
                res[('v5', s)] = bacc(v5.predict(X[te]), y[te], fl[te])
    X, y, fl, so = training('raw')
    for s in SOURCES:
        L.append(f'{s:<10}{len(set(fl[so == s])):>7}{res[("raw", s)]:>10.3f}'
                 f'{res[("ratio", s)]:>11.3f}{res[("v5", s)]:>8.3f}')
    L.append('v5 was trained on the montage tiles, which are part of the olympus row.')
    open(os.path.join(HERE, 'step1_pixelcheck.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


# ---------------------------------------------------------------- Olympus (step 2)

MAIN = ['fragmented', 'normal', 'elongated']


def fit_index(d):
    """stage7_index.py on a features table: OOF probabilities, bands, a JSON spec."""
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from stage7_index import MODEL_FEATURES, band_edges, PURITY
    m = d[d.blinded_call.isin(MAIN)].reset_index(drop=True)
    y, g = m.blinded_call.values, m.session.values
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    cv = list(GroupKFold(n_splits=5).split(m, y, g))
    X = m[MODEL_FEATURES].fillna(0)
    prob = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')
    cls = list(np.unique(y))
    idx = prob[:, cls.index('normal')] * 0.5 + prob[:, cls.index('fragmented')]
    lo, hi = band_edges(idx, y, PURITY)
    # Deviation from PLAN.md (2026-10-25, found on the v5 row, before any candidate
    # was scored): when no cut reaches 90% purity that end gets no band -- nothing
    # is called fragmented (hi 1.0) or elongated (lo 0.0). Bands are not a pass
    # criterion; the index and the gate do not depend on them.
    lo = 0.0 if lo is None else lo
    hi = 1.0 if hi is None else hi
    fitm = pipe.fit(X, y)
    sc, lr = fitm[0], fitm[-1]
    spec = dict(kind='standardise + multinomial logistic regression',
                features=list(MODEL_FEATURES), classes=cls, mean=sc.mean_.tolist(),
                scale=sc.scale_.tolist(), coef=lr.coef_.tolist(),
                intercept=lr.intercept_.tolist(), lo=float(lo), hi=float(hi),
                purity=float(PURITY), seed=SEED, fitted_with=f'scikit-learn {sklearn.__version__}')
    return m.assign(index=idx, pred=np.array(cls)[prob.argmax(1)]), spec


def olympus_metrics(m):
    from sklearn.metrics import cohen_kappa_score, roc_auc_score
    y = m.blinded_call.values
    ends = np.isin(y, ['fragmented', 'elongated'])
    return dict(acc=float((m.pred == y).mean()), kappa=float(cohen_kappa_score(y, m.pred)),
                auc=float(roc_auc_score(y[ends] == 'fragmented', m['index'].values[ends])))


def _measure(args):
    name, iid, px = args
    import tifffile
    from mitofi.features import features, haze_fraction, local_contrast
    from mitofi.io import to_field
    img, _, _ = to_field(os.path.join(ROOT, 'data', 'images', f'{iid}.ome.tif'))
    img = np.asarray(img, np.float32)
    cache = (os.path.join(ROOT, 'data', 'masks_python_v5') if name == 'v5'
             else os.path.join(DATA, f'masks_{name}'))
    p = os.path.join(cache, f'{iid}.ome_segmentation.tif')
    if os.path.exists(p):
        m = tifffile.imread(p) > 0
    else:
        m = segment_with(_MODEL[name], img)
        os.makedirs(cache, exist_ok=True)
        tifffile.imwrite(p, m.astype(np.uint8), compression='zlib')
    f = features(m, img, px)
    f.update(image_id=iid, px_um=px)
    if f.get('n_objects', 0):
        f.update(haze=haze_fraction(img, m, px), local_contrast=local_contrast(img, m, px),
                 local_snr=local_contrast(img, m, px, local=True))
    return f


_MODEL = {}


def _init(names):
    for n in names:
        if n not in _MODEL:
            _MODEL[n] = load_pixel(n)
            _MODEL[n]['rf'].n_jobs = 1


def derive_gate(d):
    """gate_thresholds.py's rule on these masks."""
    from mitofi.score import MIN_P90_MAJOR_UM
    read = d[d.rater1 != 'uncertain']
    haze = math.ceil(np.percentile(read.haze.dropna(), 99) * 100) / 100
    big = read[read.p90_major_um >= MIN_P90_MAJOR_UM].local_snr.dropna()
    snr = math.floor(np.percentile(big, 1) * 100) / 100
    return haze, snr


def qc_with(f, haze, snr):
    from mitofi import score
    old = score.MAX_HAZE, score.MIN_LARGE_SNR
    score.MAX_HAZE, score.MIN_LARGE_SNR = haze, snr
    try:
        return score.qc({k: v for k, v in f.items() if not (isinstance(v, float) and np.isnan(v))})
    finally:
        score.MAX_HAZE, score.MIN_LARGE_SNR = old


def olympus(names):
    from multiprocessing import Pool
    cov = pd.read_csv(os.path.join(ROOT, 'data', 'covariates.csv'))
    sess = pd.read_csv(os.path.join(ROOT, 'data', 'features_v5.csv'))[['image_id', 'session']]
    lab = pd.read_csv(os.path.join(ROOT, 'data', 'labels_consensus.csv'))[['image_id', 'blinded_call']]
    r1 = pd.read_csv(os.path.join(ROOT, 'data', 'labels_blind.csv'))[['image_id', 'blinded_call']]
    for name in names:
        out = os.path.join(DATA, f'olympus_{name}.csv')
        if not os.path.exists(out):
            jobs = [(name, i, p) for i, p in zip(cov.image_id, cov.px_um)]
            with Pool(10, initializer=_init, initargs=([name],)) as pool:
                rows = []
                for k, r in enumerate(pool.imap(_measure, jobs, chunksize=4), 1):
                    rows.append(r)
                    if k % 50 == 0:
                        print(f'  {name} {k}/{len(jobs)}', flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
        d = (pd.read_csv(out).merge(sess, on='image_id')
             .merge(lab, on='image_id').merge(r1.rename(columns={'blinded_call': 'rater1'}),
                                               on='image_id'))
        m, spec = fit_index(d)
        haze, snr = derive_gate(d)
        spec.update(max_haze=haze, min_large_snr=snr, pixel_model=name)
        q = d.set_index('image_id').loc[m.image_id]
        refused = [not qc_with(r.to_dict(), haze, snr)[0] for _, r in q.iterrows()]
        m['refused'] = refused
        m[['image_id', 'session', 'blinded_call', 'index', 'pred', 'refused']].to_csv(
            os.path.join(DATA, f'olympus_{name}_oof.csv'), index=False)
        os.makedirs(MODELS, exist_ok=True)
        json.dump(spec, open(os.path.join(MODELS, f'index_{name}.json'), 'w'), indent=1)
        mt = olympus_metrics(m)
        print(f'{name}: acc {mt["acc"]:.3f} kappa {mt["kappa"]:.3f} AUC {mt["auc"]:.3f} '
              f'refused {sum(refused)}/444 gate haze {haze} snr {snr} band {spec["lo"]}-{spec["hi"]}')


def olympus_summary():
    from stage7_index import MODEL_FEATURES  # noqa: F401 (same features as fit_index)
    f5 = pd.read_csv(os.path.join(ROOT, 'data', 'features_v5.csv'))
    lab = pd.read_csv(os.path.join(ROOT, 'data', 'labels_consensus.csv'))[['image_id', 'blinded_call']]
    base, _ = fit_index(f5.merge(lab, on='image_id'))
    b = olympus_metrics(base)
    L = ['Step 2: Olympus development pool (444 three-rater fields, out of fold, GroupKFold by session)',
         '', f'v7 recomputed from data/features_v5.csv (Labkit v5 masks): acc {b["acc"]:.3f}, '
         f'kappa {b["kappa"]:.3f}, AUC {b["auc"]:.3f}', '',
         f'{"pixel model":<14}{"acc":>7}{"kappa":>7}{"AUC":>7}{"refused":>9}{"haze":>6}{"snr":>6}'
         f'{"band":>14}  acc - v7 (session bootstrap 95%)  pass']
    rng = np.random.default_rng(SEED)
    bc = base.set_index('image_id')
    for name in ['v5', 'a', 'b', 'a_noleica', 'a_noallen', 'b_noleica', 'b_noallen']:
        p = os.path.join(DATA, f'olympus_{name}_oof.csv')
        if not os.path.exists(p):
            continue
        m = pd.read_csv(p)
        spec = json.load(open(os.path.join(MODELS, f'index_{name}.json')))
        mt = olympus_metrics(m)
        diff = (m.pred == m.blinded_call).values.astype(float) - \
               (bc.loc[m.image_id, 'pred'] == m.blinded_call.values).values.astype(float)
        ss = m.session.values
        us = np.unique(ss)
        boot = []
        for _ in range(2000):
            pick = rng.choice(us, len(us))
            sel = np.concatenate([np.nonzero(ss == s)[0] for s in pick])
            boot.append(diff[sel].mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ok = mt['acc'] >= 0.691 and mt['auc'] >= 0.931 and m.refused.sum() <= 26
        L.append(f'{name:<14}{mt["acc"]:>7.3f}{mt["kappa"]:>7.3f}{mt["auc"]:>7.3f}'
                 f'{int(m.refused.sum()):>6}/444{spec["max_haze"]:>6}{spec["min_large_snr"]:>6}'
                 f'{spec["lo"]:>7.3f}-{spec["hi"]:.3f}  {diff.mean():+.3f} ({lo:+.3f} to {hi:+.3f})'
                 f'{"     PASS" if ok else "     fail"}')
    L.append('pass = acc >= 0.691, AUC >= 0.931, refused <= 26 (PLAN step 2). v5 row: the shipped '
             'pixel model with the index refit on its Python masks.')
    return L


# ---------------------------------------------------------------- scoring with a candidate

def score_with(name, img, px, mask_within=None):
    """score_mask with the candidate's pixel model, index model and gate."""
    from mitofi import score
    M = _INDEX[name]
    old = score.load, score.MAX_HAZE, score.MIN_LARGE_SNR
    score.load = lambda: M
    score.MAX_HAZE, score.MIN_LARGE_SNR = M['max_haze'], M['min_large_snr']
    try:
        seg = segment_with(_MODEL[name], img)
        return score.score_mask(seg if mask_within is None else seg & mask_within, px, img=img)
    finally:
        score.load, score.MAX_HAZE, score.MIN_LARGE_SNR = old


_INDEX = {}


def _init_full(names):
    _init(names)
    from mitofi import score
    for n in names:
        if n == 'v5':
            M = dict(score.load()); M.update(max_haze=score.MAX_HAZE, min_large_snr=score.MIN_LARGE_SNR)
        else:
            M = json.load(open(os.path.join(MODELS, f'index_{n}.json')))
            for k in ('mean', 'scale', 'coef', 'intercept'):
                M[k] = np.array(M[k], float)
        _INDEX[n] = M


def chain(name, img, px, cellmask=None, black=False):
    """As is, then background removal; the black point only when asked.

    black=True was added after the retrain failed its plan (D26), so the Allen
    rows show what each rescue the tool offers does. Pass criteria ignore those rows.
    """
    from mitofi.cli import _rescuable
    from mitofi.io import subtract_background, subtract_floor
    r = score_with(name, img, px, cellmask)
    r['rescue'] = ''
    if _rescuable(r):
        r2 = score_with(name, subtract_background(img, px), px, cellmask)
        if r2['qc_pass']:
            r, r['rescue'] = r2, 'background'
        elif black:
            r3 = score_with(name, subtract_floor(img, px, peak=True)[0], px, cellmask)
            r, r['rescue'] = (r3, 'black point') if r3['qc_pass'] else (r, 'refused')
        else:
            r['rescue'] = 'refused'
    return r


def _leica_one(args):
    names, r = args
    from readlif.reader import LifFile
    from mitofi.io import to_trained_scale
    from run import condition
    out = []
    try:
        im = LifFile(os.path.join(LIFS, r['lif'])).get_image(int(r['series']))
        c = int(float(r['mito_channel']))
        a = np.max([np.asarray(im.get_frame(z=z, t=0, c=c)) for z in range(int(r['z']))], 0)
        a = np.asarray(to_trained_scale(a, int(r['bits']))[0], np.float32)
        px = float(r['px_um'])
        for n in names:
            q = chain(n, a, px)
            out.append(dict(model=n, lif=r['lif'], series=r['series'], condition=condition(r['name']),
                            px_um=px, qc_pass=q['qc_pass'], reason=q['qc_reason'], rescue=q['rescue'],
                            index=q.get('index'), band=q.get('band'), haze=q.get('haze'),
                            local_snr=q.get('local_snr'), fg_frac=q.get('fg_frac')))
    except Exception as e:
        out += [dict(model=n, lif=r['lif'], series=r['series'], condition=condition(r['name']),
                     qc_pass=False, reason=f'could not be read: {e!r}') for n in names]
    return out


def leica(names):
    from multiprocessing import Pool
    sys.path.insert(0, os.path.join(ROOT, 'validation', 'all_lif_2026-10'))
    inv = list(csv.DictReader(open(os.path.join(ROOT, 'validation', 'all_lif_2026-10', 'inventory.csv'))))
    out = os.path.join(HERE, f'leica_{"_".join(names)}.csv')
    rows = []
    with Pool(6, initializer=_init_full, initargs=(names,)) as pool:
        for k, o in enumerate(pool.imap_unordered(_leica_one, [(names, r) for r in inv]), 1):
            rows += o
            if k % 25 == 0:
                print(f'  {k}/{len(inv)}', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'-> {out}')


COMPARISONS = [
    ('CDKi < vehicle', 'CDKi treatment', ['cdki inf'], 'CDKi treatment', ['veh cdki inf', 'veh inf for cdki'], -1),
    ('Huh7 MFN1 KO pool > WT', 'huh7 mfn1 ko pool', ['mark_and_find'], 'Huh-7 WT', ['mark_and_find'], 1),
    ('Huh7 colony 9 > WT', 'colony 9 mfn1 ko', ['col mfn1 ko'], 'Huh-7 WT', ['mark_and_find'], 1),
    ('MEF MFN1 KO > WT, all-cells file', 'mef all cells', ['mfn1 ko mock'], 'mef all cells', ['wt mock'], 1),
    ('MEF MFN2 KO > WT, localization', 'mef wt mfn2 ko mavs', ['mfn2 ko mock'], 'mef wt mfn2 ko mavs', ['wt mock'], 1),
    ('MEF MFN1 KO > WT, 2 Feb', '2nd feb 2023', ['mef ko mock'], '2nd feb 2023', ['mef wt 24h mock', 'mef wt mock'], 1),
    ('MEF MAVS KO > WT, 2 Feb', '2nd feb 2023', ['mef 48h mavs ko mock'], '2nd feb 2023', ['mef wt 24h mock', 'mef wt mock'], 1),
]


def leica_summary(path):
    from scipy.stats import mannwhitneyu
    from mitofi.score import refusal_kind
    d = pd.read_csv(path)
    d['f'] = d.lif.str.split('/').str[-1]
    L = [f'Step 3: Leica, all 496 series, no black point ({os.path.basename(path)})', '']
    for n, g in d.groupby('model', sort=False):
        ok = g[g.qc_pass]
        why = pd.Series([refusal_kind(r)[0] for r in g[~g.qc_pass].reason]).value_counts()
        b = ok.band.value_counts()
        L += [f'== {n}: scored {len(ok)}/{len(g)} (as is {int((ok.rescue.fillna("") == "").sum())}, '
              f'background {int((ok.rescue == "background").sum())}); refused '
              + ', '.join(f'{k} {v}' for k, v in why.items()),
              f'   bands: elongated {b.get("elongated", 0)}, mixed {b.get("mixed", 0)}, '
              f'fragmented {b.get("fragmented", 0)}']
        held = 0
        for lab_, fa, ca, fb, cb, sign in COMPARISONS:
            A = ok[ok.f.str.contains(fa, regex=False) & ok.condition.isin(ca)]['index']
            B = ok[ok.f.str.contains(fb, regex=False) & ok.condition.isin(cb)]['index']
            if len(A) and len(B):
                good = np.sign(A.median() - B.median()) == sign
                p = mannwhitneyu(A, B).pvalue if len(A) > 1 or len(B) > 1 else float('nan')
                held += good
                L.append(f'   {lab_:<34} {A.median():.2f} ({len(A)}) vs {B.median():.2f} ({len(B)})'
                         f'  p {p:.2g}  {"holds" if good else "REVERSED"}')
            else:
                L.append(f'   {lab_:<34} not testable ({len(A)} vs {len(B)} scored)')
        ok_all = held == len(COMPARISONS) and len(ok) > 256
        L += [f'   {held}/{len(COMPARISONS)} comparisons hold; scored {len(ok)} vs 256 needed -> '
              f'{"PASS" if ok_all else "fail"}', '']
    return L


def outline(name):
    """Two scored series per file at random: image beside mask."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from readlif.reader import LifFile
    from mitofi.io import subtract_background, to_trained_scale
    path = next(p for p in sorted(os.listdir(HERE)) if p.startswith('leica_') and
                p.endswith('.csv') and name in p[6:-4].split('_'))
    d = pd.read_csv(os.path.join(HERE, path))
    d = d[(d.model == name) & d.qc_pass]
    inv = pd.read_csv(os.path.join(ROOT, 'validation', 'all_lif_2026-10', 'inventory.csv'), dtype=str)
    d = d.assign(series=d.series.astype(str)).merge(inv[['lif', 'series', 'z', 'bits', 'mito_channel']],
                                                    on=['lif', 'series'])
    rng = np.random.default_rng(20261025)
    pick = pd.concat([g.iloc[rng.permutation(len(g))[:2]] for _, g in d.groupby('lif')])
    _init_full([name])
    per = 6
    for k in range(0, len(pick), per):
        chunk = pick.iloc[k:k + per]
        fig, ax = plt.subplots(len(chunk), 2, figsize=(10, 5 * len(chunk)))
        ax = np.atleast_2d(ax)
        for i, (_, r) in enumerate(chunk.iterrows()):
            im = LifFile(os.path.join(LIFS, r.lif)).get_image(int(r.series))
            a = np.max([np.asarray(im.get_frame(z=z, t=0, c=int(float(r.mito_channel))))
                        for z in range(int(r.z))], 0)
            a = np.asarray(to_trained_scale(a, int(r.bits))[0], np.float32)
            if r.rescue == 'background':
                a = subtract_background(a, float(r.px_um))
            m = segment_with(_MODEL[name], a)
            ax[i, 0].imshow(a, cmap='gray', vmax=np.percentile(a, 99.7))
            ax[i, 1].imshow(m, cmap='gray')
            ax[i, 0].set_title(f'{r.lif.split("/")[-1][:30]} s{r.series} {r.condition[:20]}', fontsize=8)
            ax[i, 1].set_title(f'{name} index {r["index"]:.2f} {r.rescue or "as is"}', fontsize=8)
            for x in ax[i]:
                x.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f'outline_{name}_{k // per + 1}.png'), dpi=70)
        plt.close(fig)
    print(f'{len(pick)} series in outline_{name}_*.png')


# ---------------------------------------------------------------- Allen (step 5)

def _allen_one(args):
    names, r = args
    import tempfile
    import urllib.request
    import tifffile
    from mitofi.io import to_field
    out = []
    tmp = tempfile.mkdtemp()
    raw, seg = os.path.join(tmp, 'raw.ome.tif'), os.path.join(tmp, 'seg.ome.tif')
    try:
        nd = eval(r['name_dict'])
        for url, p in ((r['crop_raw'], raw), (r['crop_seg'], seg)):
            for t in range(3):
                try:
                    urllib.request.urlretrieve(ALLEN_BASE + url, p); break
                except Exception:
                    if t == 2:
                        raise
        img, _, _ = to_field(raw, channel=nd['crop_raw'].index('structure'))
        img = np.asarray(img, np.float32)
        cell = tifffile.imread(seg)[:, nd['crop_seg'].index('membrane_segmentation')].max(0) > 0
        px = 0.108333
        for n in names:
            o = dict(model=n, CellId=r['CellId'], group=r['group'], FOVId=r['FOVId'],
                     matched_to=r['matched_to'])
            for a, cm in (('whole', None), ('cell', cell)):
                q = chain(n, img, px, cm, black=True)
                o.update({f'{a}_pass': q['qc_pass'], f'{a}_reason': q['qc_reason'],
                          f'{a}_index': q.get('index'), f'{a}_rescue': q['rescue'],
                          f'{a}_band': q.get('band'), f'{a}_fg': q.get('fg_frac')})
            out.append(o)
    except Exception as e:
        out = [dict(model=n, CellId=r['CellId'], group=r['group'], error=repr(e)) for n in names]
    finally:
        for p in (raw, seg):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(tmp)
    return out


def allen(names):
    from multiprocessing import Pool
    rows_in = pd.read_csv(os.path.join(HERE, 'allen_inputs.csv')).fillna('').to_dict('records')
    rows = []
    with Pool(8, initializer=_init_full, initargs=(names,)) as pool:
        for k, o in enumerate(pool.imap_unordered(_allen_one, [(names, r) for r in rows_in]), 1):
            rows += o
            if k % 25 == 0:
                print(f'  {k}/{len(rows_in)}', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, f'allen_{"_".join(names)}.csv'), index=False)


def allen_summary(path):
    from scipy.stats import mannwhitneyu, wilcoxon
    from mitofi.score import refusal_kind
    d = pd.read_csv(path)
    L = [f'Step 5: Allen fresh test, no black point ({os.path.basename(path)})', '']
    for n, g in d.groupby('model', sort=False):
        L.append(f'== {n}  ({len(g)} cells, errors {int(g.get("error", pd.Series()).notna().sum())})')
        for a in ('cell', 'whole'):
            withbp = g[g[f'{a}_pass'] == True]                                  # noqa: E712
            bp = withbp[withbp[f'{a}_rescue'] == 'black point']
            ok = withbp[withbp[f'{a}_rescue'] != 'black point']
            if len(bp):
                bm, bi = bp[bp.group != 'M0'][f'{a}_index'], bp[bp.group == 'M0'][f'{a}_index']
                L.append(f'   {a:<6} black point rescued {len(bp)} more (not in the test): '
                         f'mitotic {bm.median():.3f} ({len(bm)}), interphase {bi.median():.3f} ({len(bi)})')
            asis = (ok[f'{a}_rescue'].fillna('') == '').sum()
            why = pd.Series([refusal_kind(r)[0] for r in g[g[f'{a}_pass'] != True][f'{a}_reason']
                             .fillna('')]).value_counts()
            mit = ok[ok.group != 'M0'][f'{a}_index']
            inter = ok[ok.group == 'M0'][f'{a}_index']
            p = mannwhitneyu(mit, inter, alternative='greater').pvalue if len(mit) and len(inter) else float('nan')
            L.append(f'   {a:<6} scored {len(ok)}/{len(g)} (as is {asis}, background {len(ok) - asis}); '
                     f'refused ' + ', '.join(f'{k} {v}' for k, v in why.items()))
            for grp in ('M0', 'M1M2', 'M3', 'M4M5'):
                v = ok[ok.group == grp][f'{a}_index']
                if len(v):
                    L.append(f'          {grp:<5} n {len(v):>3}  median {v.median():.3f}  '
                             f'IQR {v.quantile(.25):.2f}-{v.quantile(.75):.2f}')
            L.append(f'          mitotic {mit.median():.3f} ({len(mit)}) vs interphase '
                     f'{inter.median():.3f} ({len(inter)}), one-sided p {p:.2g}')
            by = ok.set_index('CellId')[f'{a}_index']
            pairs = [(by[c], by[m]) for c, m in zip(ok.matched_to, ok.CellId)
                     if c != '' and pd.notna(c) and int(float(c)) in by.index]
            if len(pairs) > 5:
                dd = np.array([x - y for x, y in pairs])
                L.append(f'          pairs {len(dd)}: median mitotic - interphase {np.median(dd):+.3f}, '
                         f'Wilcoxon p {wilcoxon(dd).pvalue:.2g}')
            if a == 'cell':
                good = (len(mit) and mit.median() > inter.median() and p < 0.05
                        and (g['cell_pass'] == True).sum() and asis > len(g) / 2)  # noqa: E712
                L.append(f'          -> {"PASS" if good else "fail"} (needs mitotic > interphase, '
                         f'p < 0.05, as-is passes > {len(g) // 2})')
        L.append('')
    return L


def summary():
    L = olympus_summary() + ['']
    for p in sorted(os.listdir(HERE)):
        if p.startswith('leica_') and p.endswith('.csv'):
            L += leica_summary(os.path.join(HERE, p))
        if p.startswith('allen_') and p.endswith('.csv') and p != 'allen_inputs.csv':
            L += allen_summary(os.path.join(HERE, p))
    open(os.path.join(HERE, 'results_summary.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def selftest():
    y = np.array([0] * 10 + [1] * 4 + [0] * 3000 + [1] * 5)
    fl = np.array(['a'] * 14 + ['b'] * 3005)
    so = np.array(['olympus'] * 14 + ['leica'] * 3005)
    w = weights(y, fl, so)
    for s in ('olympus', 'leica'):
        tot = [w[(so == s) & (y == c)].sum() for c in (0, 1)]
        assert abs(tot[0] - tot[1]) < 1e-6, tot
    assert abs(w[so == 'olympus'].sum() - w[so == 'leica'].sum()) < 1e-6
    assert abs(w.sum() - len(y)) < 1e-6
    print('selftest ok')


if __name__ == '__main__':
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == 'fit':
        fit({'a': 'raw', 'b': 'ratio'}[rest[0]],
            rest[rest.index('--without') + 1] if '--without' in rest else None)
    else:
        dict(inputs=inputs, pixelcheck=pixelcheck, summary=summary, selftest=selftest)[cmd]() \
            if cmd in ('inputs', 'pixelcheck', 'summary', 'selftest') else \
            dict(olympus=olympus, leica=leica, allen=allen, outline=lambda n: outline(n[0]))[cmd](rest)
