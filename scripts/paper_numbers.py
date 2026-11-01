"""Every development number the manuscript quotes, recomputed for MitoFI 1.0.0.

MitoFI 1.0.0 = pixel classifier v6 masks (data/features_v6.csv, all 500
pilot fields) + index model v8, fitted exactly as validation/scribbles_v2_retrain
fit it (stage7 procedure, fillna(0)). Checks first that the out-of-fold indices
reproduce data/index_v8_oof.csv. Prints aggregates only.

    .venv/bin/python scripts/paper_numbers.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, roc_auc_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'scripts')]
from rater_agreement import MAIN, SEED, fleiss, majority, oof  # noqa: E402
from stage7_index import MODEL_FEATURES as F, PURITY, band_edges  # noqa: E402

D = os.path.join(ROOT, 'data')
C = ['elongated', 'fragmented', 'normal']              # model class order (np.unique)


def load():
    f = pd.read_csv(os.path.join(D, 'features_v6.csv'))
    cov = pd.read_csv(os.path.join(D, 'covariates.csv'))[['image_id', 'session']]
    lab = lambda n, c: pd.read_csv(os.path.join(D, n))[['image_id', 'blinded_call']].rename(columns={'blinded_call': c})
    d = (f.merge(cov, on='image_id').merge(lab('labels_consensus.csv', 'maj'), on='image_id')
          .merge(lab('labels_blind.csv', 'r1'), on='image_id').merge(lab('labels_rater2.csv', 'r2'), on='image_id')
          .merge(lab('labels_rater3.csv', 'r3'), on='image_id'))
    d[F] = d[F].fillna(0)
    return d


def main():
    d = load()
    print(f'{len(d)} pilot fields with v6 features')
    m = d[d.maj.isin(MAIN)].reset_index(drop=True)
    p = oof(m, m.maj.values, F, C)
    idx = p[:, C.index('normal')] * 0.5 + p[:, C.index('fragmented')]
    pred = np.array(C)[p.argmax(1)]
    ref = pd.read_csv(os.path.join(D, 'index_v8_oof.csv')).set_index('image_id')
    dev = np.abs(idx - ref.loc[m.image_id, 'index'].values).max()
    print(f'check: max |index - stored OOF| = {dev:.2e}  ({"reproduced" if dev < 1e-9 else "DIFFERENT"})')
    y = m.maj.values

    print('\n== session-held-out, 444 majority fields, before QC')
    print(f'n {len(m)}: ' + ', '.join(f'{c} {(y == c).sum()}' for c in MAIN))
    acc = accuracy_score(y, pred)
    base = pd.Series(y).value_counts(normalize=True).max()
    print(f'accuracy {acc:.3f}  balanced {balanced_accuracy_score(y, pred):.3f}  '
          f'macro F1 {f1_score(y, pred, average="macro"):.3f}  baseline {base:.3f}  '
          f'difference {100 * (acc - base):.1f} pp  kappa {cohen_kappa_score(y, pred):.3f}')
    for c in MAIN:
        k = y == c
        print(f'  {c:<11} {(pred[k] == c).sum()} of {k.sum()} matched')
    ends = np.isin(y, ['fragmented', 'elongated'])
    print(f'AUC fragmented vs elongated {roc_auc_score(y[ends] == "fragmented", idx[ends]):.3f} '
          f'({(y == "fragmented").sum()} vs {(y == "elongated").sum()})')
    for c in MAIN:
        v = idx[y == c]
        print(f'  median index {c:<11} {np.median(v):.3f} ({np.percentile(v, 25):.3f}-{np.percentile(v, 75):.3f})')
    r = {c: i for i, c in enumerate(MAIN)}
    gap = np.abs([r[a] - r[b] for a, b in zip(y, pred)])
    print(f'disagreements {(gap > 0).sum()}: adjacent {(gap == 1).sum()}, extremes exchanged {(gap == 2).sum()}')
    top = p.max(1)
    hc = top >= 0.85
    print(f'max probability >= 0.85: {hc.sum()} fields, {(pred[hc] == y[hc]).sum()} match '
          f'({(pred[hc] == y[hc]).mean():.1%}), mean max probability {top[hc].mean():.1%}')
    lo, hi = band_edges(idx, y, PURITY)
    spec = json.load(open(os.path.join(ROOT, 'mitofi', 'models', 'index_model_v8.json')))
    print(f'band cuts recomputed {lo} / {hi}; shipped {spec["lo"]} / {spec["hi"]}')
    lo, hi = spec['lo'], spec['hi']
    b, a = idx < lo, idx > hi
    print(f'below {lo}: {(y[b] == "elongated").sum()} of {b.sum()} elongated; '
          f'above {hi}: {(y[a] == "fragmented").sum()} of {a.sum()} fragmented; '
          f'coverage {b.sum() + a.sum()} of {len(y)} ({(b.sum() + a.sum()) / len(y):.1%})')
    print(f'index vs pixel size rho {stats.spearmanr(idx, m.px_um)[0]:+.3f}; '
          f'vs mean image intensity rho {stats.spearmanr(idx, m.img_mean_int)[0]:+.3f}')
    from mitofi.score import qc
    q = np.array([qc(row)[0] for row in m.to_dict('records')])
    print(f'gate (v6 masks, current rules): refuses {(~q).sum()} of {len(m)} majority fields')
    ok = q
    print(f'  after the gate: accuracy {accuracy_score(y[ok], pred[ok]):.3f} on {ok.sum()}')

    print('\n== quality gate on the 500-field pool, first observer\'s labels')
    qa = np.array([qc(row)[0] for row in d.to_dict('records')])
    unc = (d.r1 == 'uncertain').values
    rd = d.r1.isin(MAIN).values
    print(f'rejects {(~qa & unc).sum()} of {unc.sum()} uncertain, {(~qa & rd).sum()} of {rd.sum()} readable')
    from mitofi.score import refusal_kind
    why = pd.Series([refusal_kind(qc(row)[1])[0] for row, ok_ in zip(d.to_dict('records'), qa) if not ok_]).value_counts()
    print('  reasons: ' + ', '.join(f'{k} {v}' for k, v in why.items()))

    print('\n== fit target: rater 1 vs majority on the same 444, same folds')
    t = d[d.r1.isin(MAIN)].reset_index(drop=True)
    t['maj3'] = [majority(rr) for rr in t[['r1', 'r2', 'r3']].values]
    k = t.maj3.isin(MAIN) & t[['r1', 'r2', 'r3']].isin(MAIN).all(axis=1)
    mm = t[k].reset_index(drop=True)
    fr1 = np.array(C)[oof(mm, mm.r1.values, F, C).argmax(1)]
    fmj = np.array(C)[oof(mm, mm.maj3.values, F, C).argmax(1)]
    h1, hm = fr1 == mm.maj3.values, fmj == mm.maj3.values
    rng = np.random.default_rng(SEED)
    ses = mm.session.unique()
    at = {s: np.where(mm.session.values == s)[0] for s in ses}
    boot = [np.concatenate([at[s] for s in rng.choice(ses, len(ses))]) for _ in range(5000)]
    g = [hm[i].mean() - h1[i].mean() for i in boot]
    print(f'n {len(mm)}: fit to rater 1 {h1.mean():.3f}, fit to majority {hm.mean():.3f}, '
          f'difference {100 * (hm.mean() - h1.mean()):+.1f} pp, 95% session bootstrap '
          f'{100 * np.percentile(g, 2.5):+.1f} to {100 * np.percentile(g, 97.5):+.1f}')

    print('\n== each observer held out vs the other two where they agree (rater-1 model, v6 masks)')
    t['tool'] = np.array(C)[oof(t, t.r1.values, F, C).argmax(1)]
    a3 = t[t[['r2', 'r3']].isin(MAIN).all(axis=1)].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    for h in ('r1', 'r2', 'r3'):
        o = [x for x in ('r1', 'r2', 'r3') if x != h]
        q3 = a3[a3[o[0]] == a3[o[1]]]
        gp = (q3[h] == q3[o[0]]).values.astype(int) - (q3.tool == q3[o[0]]).values
        ci = np.percentile([rng.choice(gp, len(gp)).mean() for _ in range(5000)], [2.5, 97.5])
        print(f'  {h}: n {len(q3)}, observer {(q3[h] == q3[o[0]]).mean():.3f}, model {(q3.tool == q3[o[0]]).mean():.3f}, '
              f'difference {100 * gp.mean():+.1f} pp [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]')

    print('\n== index model v8 coefficients (standardised), class order', spec['classes'])
    cf = pd.DataFrame(np.array(spec['coef']).T, index=spec['features'], columns=spec['classes'])
    cf.loc['intercept'] = spec['intercept']
    print(cf[['elongated', 'normal', 'fragmented']].round(4).to_string())


if __name__ == '__main__':
    main()


def held_out_observer_majority_style():
    """Each observer held out; model fitted only to the other two's agreed labels.

    Reference = fields where the other two agree (and all three readable). The model
    never sees the held-out observer's labels, so observer and model are compared on
    equal terms. Out of fold by session; 95% interval from 5,000 session bootstraps.
    """
    d = load()
    t = d[d[['r1', 'r2', 'r3']].isin(MAIN).all(axis=1)].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    for h in ('r1', 'r2', 'r3'):
        o = [x for x in ('r1', 'r2', 'r3') if x != h]
        m = t[t[o[0]] == t[o[1]]].reset_index(drop=True)
        ref = m[o[0]].values
        pred = np.array(C)[oof(m, ref, F, C).argmax(1)]
        hum, mod = (m[h].values == ref), (pred == ref)
        ses = m.session.unique()
        at = {s: np.where(m.session.values == s)[0] for s in ses}
        boot = [np.concatenate([at[s] for s in rng.choice(ses, len(ses))]) for _ in range(5000)]
        g = [hum[i].mean() - mod[i].mean() for i in boot]
        print(f'  {h} held out: n {len(m)}, observer {hum.mean():.3f}, model {mod.mean():.3f}, '
              f'difference {100 * (hum.mean() - mod.mean()):+.1f} pp '
              f'[{100 * np.percentile(g, 2.5):+.1f}, {100 * np.percentile(g, 97.5):+.1f}]')


def feature_screen_release():
    """The stage-5 feature screen rerun on the release inputs (majority labels, v6 masks).

    Descriptive only: the nine features were fixed earlier on rater-1 labels and v5
    masks. Effect = (H - 2) / (n - 3) from Kruskal-Wallis over the three classes;
    |Spearman| with pixel size and mean image intensity; and the effect of the
    declared exception (mean skeleton length per object) on the index's pixel-size
    association, out of fold.
    """
    d = load()
    m = d[d.maj.isin(MAIN)].reset_index(drop=True)
    n = len(m)
    for f in F:
        groups = [m[m.maj == c][f].values for c in MAIN]
        H = stats.kruskal(*groups).statistic
        print(f'  {f:<26} effect {(H - 2) / (n - 3):.3f}  |rho| px {abs(stats.spearmanr(m[f], m.px_um)[0]):.3f}  '
              f'int {abs(stats.spearmanr(m[f], m.img_mean_int)[0]):.3f}')
    for feats, lab in ((F, 'nine features'), ([x for x in F if x != 'mean_skel_len_per_obj_um'], 'without skeleton length')):
        p = oof(m, m.maj.values, feats, C)
        idx = p[:, C.index('normal')] * 0.5 + p[:, C.index('fragmented')]
        print(f'  {lab:<24} index vs pixel size rho {stats.spearmanr(idx, m.px_um)[0]:+.3f}, '
              f'accuracy {(np.array(C)[p.argmax(1)] == m.maj.values).mean():.3f}')
