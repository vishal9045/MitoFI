"""Summary tables and the random outline check for the all-.lif run.

    .venv/bin/python validation/all_lif_2026-10/summarise.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from mitofi.score import refusal_kind   # noqa: E402

SEED = 20261023


def table(d):
    L = []
    hdr = (f'{"condition":<38}{"n":>4}{"as is":>7}{"+resc":>7}{"median":>8}{"IQR":>14}'
           f'{"e/m/f":>9}  refused')
    tot = d.groupby('lif', sort=False)
    for lif, g in tot:
        L += ['', f'## {lif.split("/")[-1]}  ({len(g)} series, {int(g.final_pass.sum())} scored)', hdr]
        for c, gg in g.groupby('condition', sort=False):
            ok = gg[gg.final_pass]
            v = ok['index'].astype(float)
            why = pd.Series([refusal_kind(r)[0] for r in gg[~gg.final_pass].final_reason],
                            dtype=object).value_counts()
            b = ok.band.value_counts()
            s = f'{c[:37]:<38}{len(gg):>4}{int(gg.first_pass.sum()):>7}{len(ok):>7}'
            s += (f'{v.median():>8.3f}{v.quantile(.25):>7.2f}-{v.quantile(.75):.2f}'
                  f'{b.get("elongated", 0):>4}/{b.get("mixed", 0)}/{b.get("fragmented", 0)}'
                  if len(ok) else ' ' * 31)
            L.append(s + '  ' + ', '.join(f'{k} {n}' for k, n in why.items()))
    return L


def overall(d):
    ok = d[d.final_pass]
    why = pd.Series([refusal_kind(r)[0] for r in d[~d.final_pass].final_reason]).value_counts()
    b = ok.band.value_counts()
    return [f'{len(d)} series; scored as is {int(d.first_pass.sum())}, after the rescue '
            f'{len(ok)} ({len(ok) / len(d):.0%}); refused {len(d) - len(ok)}',
            '  refused by reason: ' + ', '.join(f'{k} {n}' for k, n in why.items()),
            f'  bands of scored: elongated {b.get("elongated", 0)}, mixed {b.get("mixed", 0)}, '
            f'fragmented {b.get("fragmented", 0)}; median index {ok["index"].median():.3f}',
            f'  channel source of scored: '
            + ', '.join(f'{k} {n}' for k, n in ok.channel_source.value_counts().items())]


def outline_check(d):
    """Two scored series per file at random (fixed seed): input and mask side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from readlif.reader import LifFile
    from mitofi.io import subtract_background, to_trained_scale
    from mitofi.segment import segment
    inv = pd.read_csv(os.path.join(HERE, 'inventory.csv'), dtype=str)
    rng = np.random.default_rng(SEED)
    pick = []
    for lif, g in d[d.final_pass].groupby('lif', sort=False):
        pick += list(g.sample(min(2, len(g)), random_state=rng.integers(1 << 31)).index)
    LIFS = os.path.join(os.path.dirname(ROOT), 'testing_data', 'test lifs')
    for k in range(0, len(pick), 8):
        rows = pick[k:k + 8]
        fig, ax = plt.subplots(len(rows), 2, figsize=(8, 4 * len(rows)))
        ax = np.atleast_2d(ax)
        for i, ix in enumerate(rows):
            r = d.loc[ix]
            iv = inv[(inv.lif == r.lif) & (inv.series == str(r.series))].iloc[0]
            img = LifFile(os.path.join(LIFS, r.lif)).get_image(int(r.series))
            a = np.max([np.asarray(img.get_frame(z=z, t=0, c=int(float(iv.mito_channel))))
                        for z in range(int(iv.z))], 0)
            a = np.asarray(to_trained_scale(a, int(iv.bits))[0], np.float32)
            if r.rescued:
                a = subtract_background(a, float(r.px_um))
            m = segment(a)
            t = (f'{r.lif.split("/")[-1][:22]} s{r.series} {r.condition[:22]}\n'
                 f'index {float(r["index"]):.2f} {r.band}{" (rescued)" if r.rescued else ""}')
            ax[i, 0].imshow(a, cmap='gray', vmax=np.percentile(a, 99.7)); ax[i, 0].set_title(t, fontsize=8)
            ax[i, 1].imshow(m, cmap='gray'); ax[i, 1].set_title(f'mask, foreground {m.mean():.2f}', fontsize=8)
        for x in ax.flat:
            x.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, f'outline_check_{k // 8 + 1}.png'), dpi=45)
        plt.close()
    return pick


def main():
    d = pd.read_csv(os.path.join(HERE, 'results.csv'))
    d = d.sort_values(['lif', 'series']).reset_index(drop=True)
    for c in ('first_pass', 'final_pass', 'rescued'):
        d[c] = d[c].astype(str).eq('True')
    d['final_reason'] = d.final_reason.fillna('')
    L = ['Every series of every .lif, scored once as PLAN.md fixes.', ''] + overall(d) + table(d)
    text = '\n'.join(L)
    open(os.path.join(HERE, 'results_summary.txt'), 'w').write(text + '\n')
    print(text)
    if '--no-outlines' not in sys.argv:
        pick = outline_check(d)
        print(f'\noutline check: {len(pick)} series in outline_check_*.png')


if __name__ == '__main__':
    main()
