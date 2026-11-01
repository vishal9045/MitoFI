"""Stage 12: what more raters say about the ceiling, and about refitting.

The tool's 66% agreement with one observer means nothing without knowing what
two observers agree on. This measures that, then asks whether a consensus
target is a better thing to fit than either rater alone. A third rater removes
the tie-break: a majority exists wherever two of three agree, and each rater
can be held out and scored against the other two, exactly like the tool.

    .venv/bin/python scripts/rater_agreement.py
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MAIN = ['elongated', 'normal', 'fragmented']          # ordered, low to high index
RANK = {c: i for i, c in enumerate(MAIN)}
SEED, FOLDS = 20260905, 5


def load():
    M = json.load(open(os.path.join(ROOT, 'mitofi/models/index_model_v7.json')))
    f = pd.read_csv(os.path.join(ROOT, 'data/features_v5.csv'))
    def lab(name, col):
        d = pd.read_csv(os.path.join(ROOT, 'data', name))
        return d[['image_id', 'blinded_call']].rename(columns={'blinded_call': col})
    d = (f.merge(lab('labels_blind.csv', 'r1'), on='image_id')
          .merge(lab('labels_rater2.csv', 'r2'), on='image_id')
          .merge(lab('labels_rater3.csv', 'r3'), on='image_id')
          .dropna(subset=M['features']).reset_index(drop=True))
    return M, d


def oof(d, y, features, classes):
    """Out-of-fold probabilities, held out by imaging session."""
    X, g = d[features].values, d.session.values
    p = np.zeros((len(d), 3))
    for tr, te in GroupKFold(FOLDS).split(X, y, g):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=SEED)).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, [list(m.classes_).index(c) for c in classes]]
    return p


def adjudicate(r1, r2, tie):
    """One label per image from two. `tie` breaks a one-step disagreement.

    There is no principled tie-break, which is the point: the rule changes the
    answer, so all of them are reported rather than one being chosen.
    """
    out = []
    for a, b in zip(r1, r2):
        if a not in RANK or b not in RANK:
            out.append(None); continue
        i, j = RANK[a], RANK[b]
        if i == j:
            out.append(a)
        elif (i + j) % 2 == 0:
            out.append(MAIN[(i + j) // 2])                # a true midpoint exists
        elif tie == 'up':
            out.append(MAIN[max(i, j)])
        elif tie == 'down':
            out.append(MAIN[min(i, j)])
        elif tie == 'middle':
            out.append('normal')
        else:
            out.append(None)                              # drop the disagreement
    return pd.Series(out)


def fleiss(calls):
    """Fleiss' kappa over rows of calls, one row per image."""
    n = np.array([[list(r).count(c) for c in MAIN] for r in calls], float)
    k = n[0].sum()
    agree = ((n * (n - 1)).sum(1) / (k * (k - 1))).mean()
    chance = ((n.sum(0) / n.sum()) ** 2).sum()
    return (agree - chance) / (1 - chance)


def majority(calls):
    """The call at least two of three made, or None on a three-way split."""
    v = pd.Series(list(calls)).value_counts()
    return v.index[0] if v.iloc[0] >= 2 else None


def three_raters(d, F, C):
    R = ['r1', 'r2', 'r3']
    t = d[d.r1.isin(MAIN)].reset_index(drop=True)
    t['tool'] = np.array(C)[oof(t, t.r1.values, F, C).argmax(1)]   # v6, fit to rater 1, out of fold
    a = t[t[['r2', 'r3']].isin(MAIN).all(axis=1)].reset_index(drop=True)

    print(f'\n== three raters: {len(a)} images where none said uncertain ==')
    print(f'  {"pair":<16}{"agree":>8}{"kappa":>8}{"weighted":>10}')
    for x, y in (('r1', 'r2'), ('r1', 'r3'), ('r2', 'r3'),
                 ('tool', 'r1'), ('tool', 'r2'), ('tool', 'r3')):
        print(f'  {x + " vs " + y:<16}{(a[x] == a[y]).mean():>8.3f}'
              f'{cohen_kappa_score(a[x], a[y], labels=MAIN):>8.3f}'
              f'{cohen_kappa_score(a[x], a[y], labels=MAIN, weights="quadratic"):>10.3f}')
    print(f'  Fleiss kappa, three raters  {fleiss(a[R].values):.3f}')
    print('  the tool learned rater 1, so tool vs r1 flatters it; r2 and r3 are unseen')
    k = a[R].nunique(axis=1)
    print(f'  unanimous {(k == 1).mean():.1%}   two against one {(k == 2).mean():.1%}'
          f'   three-way split {(k == 3).mean():.1%}')

    print('\n== each rater held out, scored against the other two where they agree ==')
    print(f'  {"held out":<10}{"n":>5}{"human":>8}{"kappa":>8}{"tool":>8}{"kappa":>8}'
          f'   human - tool, 95% bootstrap')
    rng = np.random.default_rng(SEED)
    for h in R:
        o = [r for r in R if r != h]
        m = a[a[o[0]] == a[o[1]]]
        gap = (m[h] == m[o[0]]).values.astype(int) - (m.tool == m[o[0]]).values
        ci = np.percentile([rng.choice(gap, len(gap)).mean() for _ in range(5000)], [2.5, 97.5])
        print(f'  {h:<10}{len(m):>5}{(m[h] == m[o[0]]).mean():>8.3f}'
              f'{cohen_kappa_score(m[h], m[o[0]], labels=MAIN):>8.3f}'
              f'{(m.tool == m[o[0]]).mean():>8.3f}'
              f'{cohen_kappa_score(m.tool, m[o[0]], labels=MAIN):>8.3f}'
              f'   {gap.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]')
    print('  only the r1 row is clean for the tool: there the reference is r2 and r3,')
    print('  neither of which it saw. In the other two rows rater 1 is half the reference.')

    print('\n== refit to the majority: both models scored on the same images ==')
    t['maj'] = [majority(r) for r in t[R].values]         # an uncertain call is a vote too
    rng = np.random.default_rng(SEED)
    for name, keep in (('one uncertain call allowed', t.maj.isin(MAIN)),
                       ('no rater said uncertain', t.maj.isin(MAIN) & t[R].isin(MAIN).all(axis=1))):
        m = t[keep].reset_index(drop=True)
        fit = {'rater 1': np.array(C)[oof(m, m.r1.values, F, C).argmax(1)],
               'majority': np.array(C)[oof(m, m.maj.values, F, C).argmax(1)]}
        print(f'\n  {name}: {len(m)} images with a two-of-three majority, out of fold by session')
        print(f'  {"":<16}{"fit to rater 1":>29}{"fit to majority":>27}')
        print(f'  {"scored against":<16}{"n":>5}' + f'{"acc":>8}{"kappa":>8}{"balacc":>8}' * 2)
        for ref in ('maj', 'r1', 'r2', 'r3'):
            k = m[ref].isin(MAIN).values
            y = m[ref].values[k]
            print(f'  {ref:<16}{k.sum():>5}' + ''.join(
                f'{accuracy_score(y, p[k]):>8.3f}{cohen_kappa_score(y, p[k], labels=MAIN):>8.3f}'
                f'{balanced_accuracy_score(y, p[k]):>8.3f}' for p in fit.values()))
        for c in MAIN:
            k = (m.maj == c).values
            print(f'    majority said {c:<11} n={k.sum():>3}   '
                  + '   '.join(f'{(p[k] == c).mean():.3f}' for p in fit.values()))
        hit = {n: (p == m.maj.values) for n, p in fit.items()}
        ses = m.session.unique()
        at = {s_: np.where(m.session.values == s_)[0] for s_ in ses}
        boot = [np.concatenate([at[s_] for s_ in rng.choice(ses, len(ses))]) for _ in range(5000)]
        gain = [hit['majority'][i].mean() - hit['rater 1'][i].mean() for i in boot]
        print(f'  accuracy gain against the majority {hit["majority"].mean() - hit["rater 1"].mean():+.3f},'
              f' 95% bootstrap over sessions [{np.percentile(gain, 2.5):+.3f}, {np.percentile(gain, 97.5):+.3f}]')
    print('\n  both intervals span zero. The gain is mostly more fragmented calls, which')
    print('  the majority holds more of; balanced accuracy barely moves. v7 ships the')
    print('  majority fit from the second set anyway (consensus_labels.py).')


def main():
    M, d = load()
    F, C = M['features'], M['classes']
    both = d[d.r1.isin(MAIN) & d.r2.isin(MAIN)].reset_index(drop=True)

    print(f'{len(d)} images with features and both ratings; '
          f'{len(both)} where neither rater said uncertain\n')

    print('== the ceiling: how well do two trained observers agree ==')
    print(f'  raw agreement        {(both.r1 == both.r2).mean():.3f}')
    print(f'  Cohen kappa          {cohen_kappa_score(both.r1, both.r2, labels=MAIN):.3f}')
    print(f'  weighted kappa       '
          f'{cohen_kappa_score(both.r1, both.r2, labels=MAIN, weights="quadratic"):.3f}')
    cm = confusion_matrix(both.r1, both.r2, labels=MAIN)
    print('\n  rows rater 1, cols rater 2')
    print('  ' + pd.DataFrame(cm, index=MAIN, columns=MAIN).to_string().replace('\n', '\n  '))
    err = np.abs([RANK[a] - RANK[b] for a, b in zip(both.r1, both.r2)])
    print(f'\n  exact {(err == 0).mean():.1%}   one category apart {(err == 1).mean():.1%}'
          f'   opposite ends {(err == 2).mean():.1%}')

    print('\n== fitting each target, held out by session ==')
    print(f'  {"target":<36}{"n":>5}{"chance":>8}{"acc":>8}{"balacc":>8}{"kappa":>8}')
    def row(y, name):
        keep = y.notna().values
        m, yy = both[keep].reset_index(drop=True), y[keep].values
        p = np.array(C)[oof(m, yy, F, C).argmax(1)]
        print(f'  {name:<36}{len(m):>5}'
              f'{pd.Series(yy).value_counts(normalize=True).max():>8.3f}'
              f'{accuracy_score(yy, p):>8.3f}{balanced_accuracy_score(yy, p):>8.3f}'
              f'{cohen_kappa_score(yy, p, labels=MAIN):>8.3f}')
    row(both.r1, 'rater 1 alone (v6)')
    row(both.r2, 'rater 2 alone')
    for tie, lab in (('up', 'tie -> more fragmented'), ('down', 'tie -> more elongated'),
                     ('middle', 'tie -> normal'), ('drop', 'disagreements dropped')):
        row(adjudicate(both.r1, both.r2, tie), f'consensus, {lab}')
    print('\n  kappa spans a range set by an arbitrary tie-break, and one defensible')
    print('  choice lands on rater 1 alone. Consensus is not free improvement.')

    print('\n== what the index does where the raters disagree ==')
    p = oof(both, both.r1.values, F, C)
    pred = np.array(C)[p.argmax(1)]
    idx = p[:, C.index('normal')] * 0.5 + p[:, C.index('fragmented')]
    dis = (both.r1 != both.r2).values
    a = (pred[dis] == both.r1[dis].values).sum()
    b = (pred[dis] == both.r2[dis].values).sum()
    print(f'  {dis.sum()} disagreements: sided with rater 1 {a / dis.sum():.1%}, '
          f'rater 2 {b / dis.sum():.1%}, neither {(dis.sum() - a - b) / dis.sum():.1%}')
    print('  median index, agreed labels and the disagreements between them:')
    ag = ~dis
    for lo, hi in zip(MAIN, MAIN[1:]):
        s = ag & (both.r1 == lo).values
        print(f'    both said {lo:<11}      n={s.sum():>3}  {np.median(idx[s]):.3f}')
        c = (((both.r1 == lo) & (both.r2 == hi)) | ((both.r1 == hi) & (both.r2 == lo))).values
        print(f'    {lo} vs {hi:<11} n={c.sum():>3}  {np.median(idx[c]):.3f}')
    s = ag & (both.r1 == MAIN[-1]).values
    print(f'    both said {MAIN[-1]:<11}     n={s.sum():>3}  {np.median(idx[s]):.3f}')

    three_raters(d, F, C)


if __name__ == '__main__':
    main()
