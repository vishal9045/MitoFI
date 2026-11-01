"""PLAN.md tables and tests: the October ones, plus the rescue-mix checks for MitoFI 1.0.0.

    .venv/bin/python validation/allen_mitosis_1050_2026-10/summarise.py [--no-outlines]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, wilcoxon

from run import HERE, october

S = october('summarise')
MIT = S.MIT


def rescue_mix(d, a):
    ok = d[S.flag(d[f'{a}_pass']) & d.group.isin(MIT + ('interphase',))].copy()
    ok['side'] = np.where(ok.group == 'interphase', 'interphase', 'mitotic')
    ok['res'] = ok[f'{a}_rescue'].fillna('').replace('', 'as is')
    t = pd.crosstab(ok.side, ok.res)
    p = chi2_contingency(t)[1] if t.shape[1] > 1 else float('nan')
    L = ['  rescue mix (scored cells): ' + '; '.join(
        f'{s} ' + ', '.join(f'{c} {t.loc[s, c]} ({t.loc[s, c] / t.loc[s].sum():.0%})' for c in t.columns)
        for s in t.index) + f'; chi-square p = {p:.2g}']
    for res in ('black point', 'background', 'as is'):
        x = ok[ok.res == res]
        m, i = x[x.side == 'mitotic'][f'{a}_index'], x[x.side == 'interphase'][f'{a}_index']
        if len(m) >= 3 and len(i) >= 3:
            pv = mannwhitneyu(m, i, alternative='greater').pvalue
            L.append(f'  within {res} rows only: mitotic {m.median():.3f} (n {len(m)}) vs interphase '
                     f'{i.median():.3f} (n {len(i)}); one-sided p = {pv:.2e}')
        else:
            L.append(f'  within {res} rows only: not testable (n {len(m)} vs {len(i)})')
    by = ok.set_index('CellId')
    pr = [(by.loc[c, f'{a}_index'], r[f'{a}_index']) for cid, r in by[by.side == 'interphase'].iterrows()
          for c in [r.matched_to] if c in by.index and by.loc[c, 'res'] == r.res]
    if len(pr) >= 5:
        dif = np.array(pr, float) @ np.array([1, -1])
        L.append(f'  pairs with the same rescue: {len(dif)}, median difference {np.median(dif):+.3f}, '
                 f'Wilcoxon p = {wilcoxon(dif, alternative="greater").pvalue:.2e}')
    return L


def main():
    d = pd.read_csv(os.path.join(HERE, 'results.csv'))
    err = d[d.error.notna()] if 'error' in d else d.iloc[:0]
    L = ['MitoFI 1.0.0 (pixel v6, index v8) on the October Allen sample. Plan: PLAN.md.',
         f'{len(d)} cells; read or download errors: {len(err)}', '']
    for a, name in (('cell', 'INSIDE THE CELL (primary)'), ('whole', 'WHOLE CROP')):
        L += [f'== {name} =='] + S.table(d, a) + S.tests(d, a) + S.tests(d, a, no_bp=True)
        L += S.paired(d, a) + rescue_mix(d, a)
        ok = d[S.flag(d[f'{a}_pass'])]
        L.append('  medians by stage: ' + ', '.join(
            f'{g} {ok[ok.group == g][f"{a}_index"].astype(float).median():.3f}' for g in S.ORDER))
        L.append('')
    text = '\n'.join(L)
    open(os.path.join(HERE, 'results_summary.txt'), 'w').write(text + '\n')
    print(text)
    if '--no-outlines' not in sys.argv:
        S.outline_check(d)
        print('outline check written')


if __name__ == '__main__':
    main()
