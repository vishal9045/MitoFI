"""Tables and tests PLAN.md fixes, from results/<run>/<group>.csv.

    .venv/bin/python validation/public_sets_2026-10/summarise.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
UCT_CTRL = {'repA_control', 'repC_control', 'repD_control'}
UCT_FCCP = {'repA_fccp', 'repC_fccp', 'repD_fccp'}


def load(run):
    d = os.path.join(HERE, 'results', run)
    if not os.path.isdir(d):
        return None
    parts = []
    for f in sorted(os.listdir(d)):
        if f.endswith('.csv') and not f.endswith('_summary.csv'):
            x = pd.read_csv(os.path.join(d, f), dtype={'rescue': str})
            parts.append(x.assign(group=f[:-4]))
    x = pd.concat(parts, ignore_index=True)
    x['ok'] = x.qc_pass.astype(str).eq('True')
    x['res'] = x.rescue.fillna('').replace('', 'as is')
    return x


def table(x, order):
    from mitofi.score import refusal_kind
    L = [f'   {"group":<20}{"images":>7}{"scored":>7}{"median":>8}{"IQR":>13}  e/m/f      as is/bg/bp   refused']
    for g in order:
        y = x[x.group == g]
        ok = y[y.ok]
        v = ok['index']
        b, r = ok.band.value_counts(), ok.res.value_counts()
        why = pd.Series([refusal_kind(z)[0] for z in y[~y.ok].qc_reason.fillna('')], dtype=object).value_counts()
        L.append(f'   {g:<20}{len(y):>7}{len(ok):>7}' +
                 (f'{v.median():>8.3f}{v.quantile(.25):>7.2f}-{v.quantile(.75):.2f}  '
                  f'{b.get("elongated", 0):>2}/{b.get("mixed", 0)}/{b.get("fragmented", 0):<6}'
                  f'{r.get("as is", 0):>4}/{r.get("background", 0)}/{r.get("black point", 0):<6}'
                  if len(ok) else ' ' * 43) + '  ' + ', '.join(f'{k} {n}' for k, n in why.items()))
    return L


def test(x, lo, hi, label):
    """One-sided: groups `hi` above groups `lo`; plus the rescue-mix check."""
    ok = x[x.ok]
    a, b = ok[ok.group.isin(lo)], ok[ok.group.isin(hi)]
    if len(a) < 3 or len(b) < 3:
        return [f'   {label}: not testable (n {len(a)} vs {len(b)})']
    p = mannwhitneyu(b['index'], a['index'], alternative='greater').pvalue
    L = [f'   {label}: predicted higher {b["index"].median():.3f} (n {len(b)}) vs predicted lower '
         f'{a["index"].median():.3f} (n {len(a)}), '
         f'one-sided p = {p:.2g}  -> {"HOLDS" if p < 0.05 else "not supported"}']
    t = pd.crosstab(np.where(ok.group.isin(hi), 'hi', 'lo')[ok.group.isin(lo | hi)],
                    ok[ok.group.isin(lo | hi)].res)
    if t.shape[1] > 1:
        L.append(f'      rescue mix: ' + '; '.join(f'{"higher" if s == "hi" else "lower"} '
                                                  + ', '.join(f'{c} {t.loc[s, c]}' for c in t.columns)
                                                  for s in t.index) + f'; chi-square p = {chi2_contingency(t)[1]:.2g}')
        for res in t.columns:
            aa, bb = a[a.res == res]['index'], b[b.res == res]['index']
            if len(aa) >= 3 and len(bb) >= 3:
                L.append(f'      within {res}: {bb.median():.3f} (n {len(bb)}) vs {aa.median():.3f} (n {len(aa)}), '
                         f'p = {mannwhitneyu(bb, aa, alternative="greater").pvalue:.2g}')
    return L


def main():
    L = ['Public sets scored with MitoFI 1.0.0 (pixel v6, index v8). Plan: PLAN.md.', '']
    for run, n in (('bbbc053_asis', '1'), ('bbbc053_down4', '2')):
        x = load(run)
        if x is None:
            continue
        L += [f'== {run}'] + table(x, ['DMSO', 'FCCP']) + test(x, {'DMSO'}, {'FCCP'}, f'P{n} FCCP > DMSO') + ['']
    for px in (0.132, 0.088, 0.198):
        x = load(f'mitohacker_cells_{px}')
        if x is None:
            continue
        tag = ' (primary)' if px == 0.132 else ' (sensitivity)'
        L += [f'== mitohacker crops, {px} um{tag}'] + table(x, ['145', '143', '253'])
        L += test(x, {'145'}, {'253'}, 'P3 Drp1-/- (145) below Drp1+/+ (253)')
        ok = x[x.ok]
        copies = ok.group.map({'145': 0, '143': 1, '253': 2})
        r, p = spearmanr(copies, ok['index'])
        L += [f'   P4 index rises with Drp1 copies: Spearman {r:+.3f}, one-sided p = '
              f'{p / 2 if r > 0 else 1 - p / 2:.2g}', '']
    for px in (0.066, 0.044, 0.099):
        x = load(f'mitohacker_fields_{px}')
        if x is None:
            continue
        L += [f'== mitohacker fields, {px} um'] + table(x, ['145', '143', '253'])
        L += test(x, {'145'}, {'253'}, 'P5 Drp1-/- (145) below Drp1+/+ (253)') + ['']
    x = load('uct')
    if x is not None:
        order = sorted(set(x.group), key=lambda g: (g[:4], ['control', 'CT_unlabelled', 'fccp', 'pa24', 'pa72']
                                                    .index(g[5:])))
        L += ['== Cape Town'] + table(x, order)
        L += test(x, UCT_CTRL, UCT_FCCP, 'P6 FCCP > control, pooled (primary)')
        ok = x[x.ok]
        for r in 'ACD':
            c, f = ok[ok.group == f'rep{r}_control']['index'], ok[ok.group == f'rep{r}_fccp']['index']
            L.append(f'   P7 repeat {r}: FCCP {f.median():.3f} (n {len(f)}) vs control {c.median():.3f} '
                     f'(n {len(c)}) -> {"FCCP higher" if len(c) and len(f) and f.median() > c.median() else "not higher"}')
        L += test(x, UCT_CTRL | {'repD_CT_unlabelled'}, UCT_FCCP, 'sensitivity: CT counted as control')
        for r in 'ACD':
            for g in ('pa24', 'pa72'):
                v = ok[ok.group == f'rep{r}_{g}']['index']
                if len(v):
                    L.append(f'   propionic acid {g[2:]} h, repeat {r}: median {v.median():.3f} (n {len(v)}), no prediction')
        L.append('')
    text = '\n'.join(L)
    open(os.path.join(HERE, 'results_summary.txt'), 'w').write(text + '\n')
    print(text)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    main()
