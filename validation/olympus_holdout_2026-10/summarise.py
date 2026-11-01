"""Tables and tests PLAN.md fixes.

    .venv/bin/python validation/olympus_holdout_2026-10/summarise.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))


def load():
    from run import out_for
    d = pd.read_csv(os.path.join(HERE, 'inputs.csv'))
    parts = []
    for f in d.folder.unique():
        p = out_for(f)
        if os.path.exists(p):
            x = pd.read_csv(p, dtype={'rescue': str})
            parts.append(x.assign(folder=f))
    r = pd.concat(parts, ignore_index=True)
    r['name'] = r.image_id.str.replace(r'\.ome$', '', regex=True)
    r['ok'] = r.qc_pass.astype(str).eq('True')
    r['res'] = r.rescue.fillna('').replace('', 'as is')
    return d, r


def describe(y):
    from mitofi.score import refusal_kind
    ok = y[y.ok]
    v, b, rs = ok['index'], ok.band.value_counts(), ok.res.value_counts()
    why = pd.Series([refusal_kind(z)[0] for z in y[~y.ok].qc_reason.fillna('')], dtype=object).value_counts()
    return (f'{len(y):>3} fields, {len(ok):>3} scored' +
            (f', median {v.median():.3f} (IQR {v.quantile(.25):.2f}-{v.quantile(.75):.2f}), '
             f'bands e/m/f {b.get("elongated", 0)}/{b.get("mixed", 0)}/{b.get("fragmented", 0)}, '
             f'as is/bg/bp {rs.get("as is", 0)}/{rs.get("background", 0)}/{rs.get("black point", 0)}'
             if len(ok) else '') + ('; refused ' + ', '.join(f'{k} {n}' for k, n in why.items()) if len(why) else ''))


def main():
    d, r = load()
    m = d.merge(r[['folder', 'name', 'ok', 'res', 'index', 'band', 'qc_reason']], on=['folder', 'name'], how='left')
    from run import EXCLUDED
    m['key'] = m.folder + '/' + m.name
    ex = m.key.isin(EXCLUDED)
    m.loc[ex, 'qc_reason'] = 'excluded at scoring: ' + m.loc[ex, 'key'].map(EXCLUDED)
    missing = (m.ok.isna() & ~ex).sum()
    m['ok'] = m.ok.fillna(False).astype(bool)
    pool = m.drop_duplicates('rel')
    L = ['Olympus files never in development, scored with MitoFI 1.0.0 (pixel v6, index v8). Plan: PLAN.md.',
         f'{pool.rel.nunique()} files; excluded at scoring (no 570 nm channel): {int(ex.sum())}; '
         f'otherwise without a result row: {missing}', '',
         '== Pool (all files): ' + describe(pool), '']
    rows = []
    for cid, g in m[m.comparison.notna()].groupby('comparison', sort=False):
        d_, conf = g.direction.iloc[0], g.confidence.iloc[0]
        a, b = g[g.side == 'first'], g[g.side == 'second']
        L.append(f'== {cid} ({conf}, predicted: first {"more fragmented" if d_ == "frag" else "more elongated" if d_ == "elong" else "-"})')
        L += [f'   first : {describe(a)}', f'   second: {describe(b)}']
        va, vb = a[a.ok]['index'], b[b.ok]['index']
        if len(va) >= 3 and len(vb) >= 3:
            diff = va.median() - vb.median()
            alt = 'greater' if d_ == 'frag' else 'less' if d_ == 'elong' else 'two-sided'
            p = mannwhitneyu(va, vb, alternative=alt).pvalue
            right = (diff > 0) if d_ == 'frag' else (diff < 0) if d_ == 'elong' else None
            L.append(f'   difference {diff:+.3f}, p = {p:.2g} ({alt})' +
                     ('' if right is None else f'  -> {"in predicted direction" if right else "OPPOSITE"}'
                                                f'{", p < 0.05" if p < 0.05 and right else ""}'))
            t = pd.crosstab(g[g.ok].side, g[g.ok].res)
            if t.shape[0] == 2 and t.shape[1] > 1:
                pc = chi2_contingency(t)[1]
                L.append('   rescue mix: ' + '; '.join(f'{s} ' + ', '.join(f'{c} {t.loc[s, c]}' for c in t.columns)
                                               for s in t.index) + f'; chi-square p = {pc:.2g}')
                if pc < 0.05:
                    for res in t.columns:
                        x, y = a[a.ok & (a.res == res)]['index'], b[b.ok & (b.res == res)]['index']
                        if len(x) >= 3 and len(y) >= 3:
                            L.append(f'      within {res}: {x.median():.3f} (n {len(x)}) vs {y.median():.3f} '
                                     f'(n {len(y)}), p = {mannwhitneyu(x, y, alternative=alt).pvalue:.2g}')
            rows.append(dict(id=cid, conf=conf, right=right, sig=bool(right) and p < 0.05, diff=diff))
        else:
            L.append(f'   not testable (scored {len(va)} vs {len(vb)})')
            rows.append(dict(id=cid, conf=conf, right=None, sig=False, diff=np.nan))
        L.append('')
    s = pd.DataFrame(rows)
    for conf in ('firm', 'tentative'):
        x = s[s.conf == conf]
        t = x[x.right.notna()]
        k = int(t.right.sum())
        L.append(f'== {conf}: {len(x)} predictions, {len(t)} testable; in predicted direction {k}/{len(t)} '
                 f'(binomial p = {binomtest(k, len(t), 0.5, "greater").pvalue:.2g}); p < 0.05 {int(x.sig.sum())}/{len(x)}'
                 if len(t) else f'== {conf}: none testable')
        if conf == 'firm' and len(t):
            ok = binomtest(k, len(t), 0.5, 'greater').pvalue < 0.05 and x.sig.sum() >= len(x) / 2
            L.append(f'   -> {"PASS" if ok else "fail"} (binomial p < 0.05 and at least {len(x) / 2:.1f} reach p < 0.05)')
    c = {t: s.set_index('id')['diff'].get(t) for t in ('C12', 'C24', 'C48')}
    L.append('   C infected-minus-mock by time: ' + ', '.join(f'{k} {v:+.3f}' for k, v in c.items() if v == v))
    text = '\n'.join(L)
    open(os.path.join(HERE, 'results_summary.txt'), 'w').write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
