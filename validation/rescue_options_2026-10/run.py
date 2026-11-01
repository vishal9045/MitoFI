"""Each rescue option alone, on Olympus, Leica and Allen, as PLAN.md fixes.

    .venv/bin/python validation/rescue_options_2026-10/run.py olympus|leica|allen
    .venv/bin/python validation/rescue_options_2026-10/run.py summary

Olympus rows carry pilot IDs, so they are written to data/scribbles_v2/
(gitignored); Leica and Allen rows are written here.
"""
import csv
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'validation', 'all_lif_2026-10'))
sys.path.insert(0, os.path.join(ROOT, 'validation', 'scribbles_v2_retrain'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
LIFS = os.path.join(os.path.dirname(ROOT), 'testing_data', 'test lifs')
OLY_OUT = os.path.join(ROOT, 'data', 'scribbles_v2', 'rescue_options_olympus{}.csv')
OPTIONS = ['as is', 'background', 'black point', 'offset', 'ball 1um', 'denoise', 'deconv', 'slab',
           'imagej auto', 'offset+stretch', 'slab+black point', 'background+stretch']
SET2 = OPTIONS[8:]          # step 2, planned after step 1's Allen result (PLAN.md)
SET = 3 if 'set3' in sys.argv else 2 if 'set2' in sys.argv else 1
SUFFIX = {1: '', 2: '_set2', 3: '_set3'}[SET]
OPTICS = {'allen': (510, 1.25), 'leica': (600, 1.4), 'olympus': (600, 1.4)}


def deconv(a, px, src):
    from skimage.restoration import richardson_lucy
    lam, na = OPTICS[src]
    s = 0.21 * lam / 1000 / na / px
    r = int(np.ceil(3 * s))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    psf = np.exp(-(yy ** 2 + xx ** 2) / (2 * s * s)); psf /= psf.sum()
    m = max(float(a.max()), 1.0)
    return np.clip(richardson_lucy(a / m, psf, num_iter=20, clip=False), 0, None) * m


def denoise(a):
    from skimage.restoration import denoise_nl_means
    d = np.diff(a, axis=1)                   # noise SD from neighbour differences (MAD)
    s = max(1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2), 1e-3)
    return denoise_nl_means(a, h=0.8 * s, sigma=s, fast_mode=True, patch_size=5,
                            patch_distance=6, preserve_range=True).astype(np.float32)


def slab(stack, zstep, half_um=1.0):
    sharp = [ndi.gaussian_laplace(p.astype(np.float32), 2).var() for p in stack]
    z0 = int(np.argmax(sharp))
    keep = [z for z in range(len(stack)) if abs(z - z0) * zstep <= half_um]
    return stack[keep].max(0)


def auto_bounds(a):
    """ImageJ's Brightness/Contrast "Auto" (ContrastAdjuster.autoAdjust, first click):
    256 bins from min to max; skip bins holding > 10% of pixels; the first bin from
    each end holding > pixels/5000 sets the display minimum and maximum."""
    lo0, hi0 = float(a.min()), float(a.max())
    if hi0 <= lo0:
        return lo0, hi0
    h, _ = np.histogram(a, bins=256, range=(lo0, hi0))
    n = a.size
    c = np.where(h > n // 10, 0, h) > n // 5000
    if not c.any():
        return lo0, hi0
    b = (hi0 - lo0) / 256
    return lo0 + np.argmax(c) * b, lo0 + (255 - np.argmax(c[::-1])) * b


def stretch(a, lo, hi):
    from mitofi.io import TRAINED_MAX
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32) * TRAINED_MAX


def variants(mip, px, src, stack=None, zstep=None):
    from mitofi.io import subtract_background, subtract_floor
    a = np.asarray(mip, np.float32)
    if SET == 3:
        b = subtract_background(a, px)
        yield 'background+stretch', stretch(b, *auto_bounds(b))
        return
    if SET == 2:
        # baselines step 2 needs, since step 1 on Leica was stopped (PLAN.md)
        yield 'as is', a
        yield 'background', subtract_background(a, px)
        yield 'black point', subtract_floor(a, px, peak=True)[0]
        yield 'ball 1um', subtract_background(a, px, radius_um=1.0)
        yield 'imagej auto', stretch(a, *auto_bounds(a))
        yield 'offset+stretch', stretch(a, float(np.percentile(a, 0.5)), auto_bounds(a)[1])
        if stack is not None and len(stack) > 1 and zstep:
            yield 'slab+black point', subtract_floor(np.asarray(slab(stack, zstep), np.float32),
                                                     px, peak=True)[0]
        return
    yield 'as is', a
    yield 'background', subtract_background(a, px)
    yield 'black point', subtract_floor(a, px, peak=True)[0]
    yield 'offset', np.clip(a - np.percentile(a, 0.5), 0, None)
    yield 'ball 1um', subtract_background(a, px, radius_um=1.0)
    yield 'denoise', denoise(a)
    yield 'deconv', deconv(a, px, src)
    if stack is not None and len(stack) > 1 and zstep:
        yield 'slab', np.asarray(slab(stack, zstep), np.float32)


def scored(img, px, cell=None):
    from mitofi.score import score_mask
    from mitofi.segment import segment
    seg = segment(img, jobs=1)
    out = {'whole': score_mask(seg, px, img=img)}
    if cell is not None:
        out['cell'] = score_mask(seg & cell, px, img=img)
    return out


def rows_for(base, imgs, px, cell=None):
    rows = []
    for name, img in imgs:
        for view, q in scored(img, px, cell).items():
            rows.append(dict(base, option=name, view=view, qc_pass=q['qc_pass'],
                             reason=q['qc_reason'], index=q.get('index'), band=q.get('band')))
    return rows


# ---------------------------------------------------------------- per source

def _olympus_one(args):
    k, iid = args
    from mitofi.io import to_field
    img, px, _ = to_field(os.path.join(ROOT, 'data', 'images', f'{iid}.ome.tif'))
    px = float(pd.read_csv(os.path.join(ROOT, 'data', 'covariates.csv'))
               .set_index('image_id').loc[iid, 'px_um'])
    return rows_for(dict(source='olympus', id=iid, n=k), variants(img, px, 'olympus'), px)


def _leica_one(r):
    from readlif.reader import LifFile
    from mitofi.io import to_trained_scale
    from run import condition
    base = dict(source='leica', lif=r['lif'], series=r['series'], condition=condition(r['name']),
                px_um=float(r['px_um']))
    try:
        im = LifFile(os.path.join(LIFS, r['lif'])).get_image(int(r['series']))
        c = int(float(r['mito_channel']))
        st = np.stack([np.asarray(im.get_frame(z=z, t=0, c=c)) for z in range(int(r['z']))])
        st = np.asarray(to_trained_scale(st, int(r['bits']))[0], np.float32)
        zs = im.scale[2]
        zstep = 1.0 / zs if zs else None
        return rows_for(base, variants(st.max(0), base['px_um'], 'leica', st, zstep), base['px_um'])
    except Exception as e:
        return [dict(base, option='as is', view='whole', qc_pass=False, reason=f'could not be read: {e!r}')]


def _allen_one(r):
    import tempfile
    import urllib.request
    import tifffile
    from mitofi.io import _significant_bits, to_field, to_trained_scale
    from retrain import ALLEN_BASE
    base = dict(source='allen', CellId=r['CellId'], group=r['group'], FOVId=r['FOVId'],
                matched_to=r['matched_to'])
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
        ch = nd['crop_raw'].index('structure')
        mip, _, _ = to_field(raw, channel=ch)
        st = tifffile.imread(raw)[:, ch]
        st = np.asarray(to_trained_scale(st, _significant_bits(raw))[0], np.float32)
        cell = tifffile.imread(seg)[:, nd['crop_seg'].index('membrane_segmentation')].max(0) > 0
        px = 0.108333
        return rows_for(base, variants(mip, px, 'allen', st, px), px, cell)
    except Exception as e:
        return [dict(base, option='as is', view='cell', qc_pass=False, reason=f'error {e!r}')]
    finally:
        for p in (raw, seg):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(tmp)


def run(src):
    if src == 'olympus':
        oof = pd.read_csv(os.path.join(ROOT, 'data', 'scribbles_v2', 'olympus_a_oof.csv'))
        ids = oof[~oof.refused].image_id.tolist()
        rng = np.random.default_rng(20261026)
        jobs, fn, out = list(enumerate(rng.choice(ids, 100, replace=False))), _olympus_one, OLY_OUT.format(SUFFIX)
    elif src == 'leica':
        inv = list(csv.DictReader(open(os.path.join(ROOT, 'validation', 'all_lif_2026-10', 'inventory.csv'))))
        jobs, fn, out = [r for r in inv if float(r['px_um']) <= 0.2], _leica_one, os.path.join(HERE, f'leica{SUFFIX}.csv')
    else:
        jobs = pd.read_csv(os.path.join(ROOT, 'validation', 'scribbles_v2_retrain', 'allen_inputs.csv')
                           ).fillna('').to_dict('records')
        fn, out = _allen_one, os.path.join(HERE, f'allen{SUFFIX}.csv')
    rows = []
    with Pool(11) as pool:
        for k, o in enumerate(pool.imap_unordered(fn, jobs), 1):
            rows += o
            if k % 25 == 0:
                print(f'  {src} {k}/{len(jobs)}', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'-> {out}')


# ---------------------------------------------------------------- summary

def safety(d, key):
    L = []
    w = d[d.view == 'whole']
    base = w[w.option == 'as is'].set_index(key)
    ok_ids = base[base.qc_pass].index
    for o in OPTIONS[1:]:
        x = w[(w.option == o)].set_index(key)
        x = x[x.index.isin(ok_ids)]
        if not len(x):
            continue
        both = x[x.qc_pass]
        sh = both['index'] - base.loc[both.index, 'index']
        bc = (both.band != base.loc[both.index, 'band']).mean() if len(both) else float('nan')
        good = len(both) and abs(sh.median()) <= 0.03 and bc <= 0.10
        L.append(f'   {o:<12} still pass {len(both):>3}/{len(x):<3} shift median {sh.median():+.3f}  '
                 f'p90 |shift| {sh.abs().quantile(.9):.3f}  band changed {bc:.0%}  '
                 f'{"SAFE" if good else "fails D23"}')
    return L


def leica_rescue(d):
    from scipy.stats import mannwhitneyu
    from retrain import COMPARISONS
    w = d[d.view == 'whole'].copy()
    w['f'] = w.lif.str.split('/').str[-1]
    key = ['lif', 'series']
    piv = {o: g.set_index(key) for o, g in w.groupby('option')}
    asis = piv['as is']
    refused = asis[~asis.qc_pass].index
    L = [f'   refused as is: {len(refused)} of {len(asis)} (px <= 0.2 um)']

    def table(label, pick):
        ok = pick[pick.qc_pass]
        out, held = [], 0
        for lab_, fa, ca, fb, cb, sign in COMPARISONS:
            A = ok[ok.f.str.contains(fa, regex=False) & ok.condition.isin(ca)]['index']
            B = ok[ok.f.str.contains(fb, regex=False) & ok.condition.isin(cb)]['index']
            if len(A) and len(B):
                good = np.sign(A.median() - B.median()) == sign
                held += good
                p = mannwhitneyu(A, B).pvalue if len(A) + len(B) > 2 else float('nan')
                out.append(f'{A.median():.2f}/{B.median():.2f}{"" if good else "!"} p{p:.1g}')
            else:
                out.append('n/a')
        return f'   {label:<26} scored {len(ok):>3}  holds {held}/7  ' + ' | '.join(out)

    for o in OPTIONS[1:]:
        if o not in piv:
            continue
        x = piv[o]
        n = int(x.loc[x.index.intersection(refused)].qc_pass.sum())
        alone = asis.copy()
        r = x.index.intersection(refused)
        alone.loc[r] = x.loc[r]
        L.append(f'   {o:<12} rescues {n:>3}/{len(refused)}')
        L.append(table(f'as is + {o}', alone.reset_index()))
        if o not in ('background', 'black point'):
            ch = asis.copy()
            bg = piv['background']
            for i in refused:
                if i in bg.index and bg.loc[i, 'qc_pass']:
                    ch.loc[i] = bg.loc[i]
                elif i in x.index:
                    ch.loc[i] = x.loc[i]
            L.append(table(f'as is, background, {o}', ch.reset_index()))
    ch = asis.copy()
    for i in refused:
        for o in ('background', 'black point'):
            if piv[o].loc[i, 'qc_pass']:
                ch.loc[i] = piv[o].loc[i]; break
    L.append(table('shipped chain (bg, black pt)', ch.reset_index()))
    L.append('   comparisons, in order: ' + '; '.join(c[0] for c in __import__('retrain').COMPARISONS)
             + ' ("!" = wrong direction)')
    return L


def allen_table(d):
    from scipy.stats import mannwhitneyu, wilcoxon
    L = []
    for view in ('cell', 'whole'):
        L.append(f'   -- {view}')
        for o in OPTIONS:
            g = d[(d.view == view) & (d.option == o)]
            ok = g[g.qc_pass == True]                                  # noqa: E712
            if not len(ok):
                L.append(f'   {o:<12} scored   0/{len(g)}'); continue
            mi, it = ok[ok.group != 'M0']['index'], ok[ok.group == 'M0']['index']
            p = mannwhitneyu(mi, it, alternative='greater').pvalue if len(mi) and len(it) else float('nan')
            by = ok.set_index('CellId')['index']
            pr = [by[int(float(c))] - by[i] for c, i in zip(ok.matched_to, ok.CellId)
                  if str(c) not in ('', 'nan') and int(float(c)) in by.index]
            pw = f'pairs {len(pr)} {np.median(pr):+.3f} p {wilcoxon(pr).pvalue:.2g}' if len(pr) > 5 else ''
            st = ' '.join(f'{s} {ok[ok.group == s]["index"].median():.2f}' for s in ('M0', 'M1M2', 'M3', 'M4M5'))
            L.append(f'   {o:<12} scored {len(ok):>3}/{len(g)}  mitotic {mi.median():.3f} vs interphase '
                     f'{it.median():.3f} p {p:.2g}  {pw}  [{st}]')
    return L


def summary():
    L = ['Rescue options, one at a time (DEVELOPMENT). PLAN.md fixes the settings and the D23 rule.', '']
    def both(p):
        d = pd.concat([pd.read_csv(x) for x in (p, p.replace('.csv', '_set2.csv'), p.replace('.csv', '_set3.csv')) if os.path.exists(x)])
        return d.drop_duplicates([c for c in ('id', 'lif', 'series', 'CellId', 'option', 'view') if c in d])
    if os.path.exists(OLY_OUT.format('')):
        L += ['== Safety, Olympus: 100 development fields that pass as is'] + safety(both(OLY_OUT.format('')), 'id') + ['']
    lp = os.path.join(HERE, 'leica.csv')
    if os.path.exists(lp) or os.path.exists(lp.replace('.csv', '_set2.csv')):
        d = both(lp)
        d['key'] = d.lif + '#' + d.series.astype(str)
        L += ['== Safety, Leica: series that pass as is'] + safety(d, 'key') + ['']
        L += ['== Rescue, Leica (whole field)'] + leica_rescue(d) + ['']
    ap = os.path.join(HERE, 'allen.csv')
    if os.path.exists(ap):
        L += ['== Allen, 238 cells, each option alone'] + allen_table(both(ap)) + ['']
    open(os.path.join(HERE, 'results_summary.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def selftest():
    a = np.zeros((64, 64), np.float32) + 420; a[20:44, 30:34] = 900
    assert np.allclose(dict(variants(a, 0.1, 'allen'))['offset'].min(), 0)
    st = np.stack([a * 0.2, a, a * 0.3])
    st[1] += np.random.default_rng(0).normal(0, 1, a.shape)
    assert np.allclose(slab(st, 0.6), st[:3].max(0))                # all within 1 um
    assert np.allclose(slab(st, 2.0), st[1])                        # only the sharp plane
    dc = deconv(ndi.gaussian_filter(a, 1.5), 0.1, 'allen')
    assert dc.shape == a.shape and np.isfinite(dc).all()
    x = np.r_[np.full(9000, 100.0), np.linspace(0, 99, 60), np.linspace(101, 400, 1000)]
    lo, hi = auto_bounds(x)
    assert 100 < lo < 110 and hi > 390, (lo, hi)         # skips the 90% peak, like ImageJ
    print('selftest ok')


if __name__ == '__main__':
    c = sys.argv[1]
    assert c in ('summary', 'selftest', 'olympus', 'leica', 'allen'), c
    summary() if c == 'summary' else selftest() if c == 'selftest' else run(c)
