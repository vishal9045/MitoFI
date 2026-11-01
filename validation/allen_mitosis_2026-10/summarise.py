"""Tests and tables PLAN.md specifies, plus the random outline check.

    .venv/bin/python validation/allen_mitosis_2026-10/summarise.py [--no-outlines]
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from mitofi.score import refusal_kind   # noqa: E402

SEED = 20261023
MIT = ('M1M2', 'M3', 'M4M5')
ORDER = ('interphase',) + MIT + ('M6M7',)


def flag(s):
    return s.astype(str).eq('True')


def table(d, a):
    L = [f'{"group":<12}{"cells":>6}{"scored":>8}{"median":>8}{"IQR":>14}  e/m/f      rescue (as is/bg/bp)   refused']
    for g in ORDER:
        x = d[d.group == g]
        ok = x[flag(x[f'{a}_pass'])]
        v = ok[f'{a}_index'].astype(float)
        b = ok[f'{a}_band'].value_counts()
        r = ok[f'{a}_rescue'].fillna('').value_counts()
        why = pd.Series([refusal_kind(z)[0] for z in x[~flag(x[f'{a}_pass'])][f'{a}_reason'].fillna('')],
                        dtype=object).value_counts()
        L.append(f'{g:<12}{len(x):>6}{len(ok):>8}' +
                 (f'{v.median():>8.3f}{v.quantile(.25):>7.2f}-{v.quantile(.75):.2f}  '
                  f'{b.get("elongated", 0):>3}/{b.get("mixed", 0)}/{b.get("fragmented", 0):<5}'
                  f'{r.get("", 0):>8}/{r.get("background", 0)}/{r.get("black point", 0):<8}'
                  if len(ok) else ' ' * 48) + '  ' + ', '.join(f'{k} {n}' for k, n in why.items()))
    return L


def tests(d, a, no_bp=False):
    ok = d[flag(d[f'{a}_pass'])]
    if no_bp:
        ok = ok[ok[f'{a}_rescue'].fillna('') != 'black point']
    m = ok[ok.group.isin(MIT)][f'{a}_index'].astype(float)
    i = ok[ok.group == 'interphase'][f'{a}_index'].astype(float)
    tag = ' (no black-point rows)' if no_bp else ''
    if len(m) < 3 or len(i) < 3:
        return [f'  mitotic vs interphase{tag}: not testable (n {len(m)} vs {len(i)})']
    p = mannwhitneyu(m, i, alternative='greater').pvalue
    return [f'  mitotic vs interphase{tag}: median {m.median():.3f} (n {len(m)}) vs '
            f'{i.median():.3f} (n {len(i)}); one-sided Mann-Whitney p = {p:.2e}; '
            f'AUC {mannwhitneyu(m, i).statistic / (len(m) * len(i)):.3f}']


def paired(d, a):
    ok = d[flag(d[f'{a}_pass'])].set_index('CellId')
    mates = ok[ok.group == 'interphase']
    pr = [(ok.loc[c, f'{a}_index'], mates.loc[mid, f'{a}_index'])
          for mid, c in zip(mates.index, mates.matched_to) if c in ok.index]
    if len(pr) < 5:
        return [f'  paired: too few complete pairs ({len(pr)})']
    x = np.array(pr, float); dif = x[:, 0] - x[:, 1]
    return [f'  paired (mitotic minus same-field interphase): {len(dif)} pairs, median '
            f'difference {np.median(dif):+.3f}, mitotic higher in {(dif > 0).mean():.0%}; '
            f'Wilcoxon signed-rank p = {wilcoxon(dif, alternative="greater").pvalue:.2e}']


def outline_check(d):
    import tempfile
    import urllib.request
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import tifffile
    from mitofi.io import subtract_background, subtract_floor, to_field
    from mitofi.segment import segment
    inp = pd.read_csv(os.path.join(HERE, 'inputs.csv')).set_index('CellId')
    ok = d[flag(d.cell_pass)]
    pick = ok.sample(min(20, len(ok)), random_state=SEED)
    base = 'https://allencell.s3.amazonaws.com/aics/hipsc_single_cell_image_dataset/'
    for k in range(0, len(pick), 5):
        rows = pick.iloc[k:k + 5]
        fig, ax = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
        ax = np.atleast_2d(ax)
        for i, (_, r) in enumerate(rows.iterrows()):
            m = inp.loc[r.CellId]; names = eval(m.name_dict)
            t = tempfile.mkdtemp(); raw, seg = os.path.join(t, 'r.ome.tif'), os.path.join(t, 's.ome.tif')
            urllib.request.urlretrieve(base + m.crop_raw, raw)
            urllib.request.urlretrieve(base + m.crop_seg, seg)
            img = np.asarray(to_field(raw, channel=names['crop_raw'].index('structure'))[0], np.float32)
            cell = tifffile.imread(seg)[:, names['crop_seg'].index('membrane_segmentation')].max(0) > 0
            os.remove(raw); os.remove(seg); os.rmdir(t)
            px = float(r.px_um); res = str(r.cell_rescue) if isinstance(r.cell_rescue, str) else ''
            v = (subtract_background(img, px) if res == 'background' else
                 subtract_floor(img, px, peak=True)[0] if res == 'black point' else img)
            mk = segment(v)
            ax[i, 0].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.8))
            ax[i, 0].contour(cell, [0.5], colors='y', linewidths=0.6)
            ax[i, 0].set_title(f'{r.group}  cell {float(r.cell_index):.2f} {r.cell_band}', fontsize=8)
            ax[i, 1].imshow(v, cmap='gray', vmax=np.percentile(v, 99.8)); ax[i, 1].set_title(res or 'as is', fontsize=8)
            ax[i, 2].imshow(mk & cell, cmap='gray'); ax[i, 2].set_title('mask inside cell', fontsize=8)
        for x in ax.flat:
            x.axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(HERE, f'outline_check_{k // 5 + 1}.png'), dpi=55); plt.close()


def main():
    d = pd.read_csv(os.path.join(HERE, 'results.csv'))
    err = d[d.error.notna()]
    L = [f'{len(d)} cells; read or download errors: {len(err)}', '']
    for a, name in (('cell', 'INSIDE THE CELL (primary)'), ('whole', 'WHOLE CROP')):
        L += [f'== {name} =='] + table(d, a) + tests(d, a) + tests(d, a, no_bp=True) + paired(d, a)
        ok = d[flag(d[f'{a}_pass'])]
        L.append('  medians by stage: ' + ', '.join(
            f'{g} {ok[ok.group == g][f"{a}_index"].astype(float).median():.3f}' for g in ORDER))
        L.append('')
    text = '\n'.join(L)
    open(os.path.join(HERE, 'results_summary.txt'), 'w').write(text + '\n')
    print(text)
    if '--no-outlines' not in sys.argv:
        outline_check(d)
        print('outline check written')


if __name__ == '__main__':
    main()
