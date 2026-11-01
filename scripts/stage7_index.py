"""Stage 7: turn the Stage 6 probabilities into a calibrated index with an
explicit mixed band.

Band names here are the names the tool reports: elongated, mixed, fragmented.
They used to be elongated_end, ambiguous and fragmented_end, so the research
record and the shipped CSV disagreed about what to call the same three
regions.

The band is not a cosmetic hedge. The Stage 6 confusion matrix shows the two
ends of the scale are separated well and the middle is not, so the honest
product is a number plus a region in which the number declines to commit.

The index runs 1 = fragmented to 0 = elongated, matching the name. It is fit
to the three-rater consensus from consensus_labels.py, so "the call" below is
the call two of three blind raters made.

Band rule, fixed before looking at the result: take the smallest high cut such
that at least PURITY of the images above it were called fragmented, and the
largest low cut such that at least PURITY of the images below it were called
elongated. Everything between is the mixed band. The choice of PURITY is
arbitrary, so the sensitivity of coverage to it is reported.

    .venv/bin/python scripts/stage7_index.py
"""
import os
import sklearn
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = ['fragmented', 'normal', 'elongated']
CLEAN = ['p90_aspect', 'p90_ecc', 'areawt_formfactor', 'med_aspect', 'med_ecc',
         'hole_area_frac', 'areawt_aspect', 'frac_fg_largest']
# mean_skel_len_per_obj_um is added deliberately, and it does NOT pass the screen
# above: its |rho| with image mean intensity is 0.406 against a cutoff of 0.400.
# It is included anyway for a reason the screen cannot see. The screen tests the
# marginal correlation of a FEATURE with a nuisance variable; what matters is the
# correlation of the INDEX. Adding this one takes the index's dependence on pixel
# size from |rho| 0.092 to 0.003, because the model can use it to correct rather
# than being misled by it. Declared here as an exception rather than hidden by
# moving the threshold, so the departure from the rule is visible.
#
# Both figures were measured on v6, fitted to rater 1 alone. On the consensus
# labels the index's pixel-size rho is +0.04, p 0.44.
#
# The cost is honest: on the development pool it gains +0.017 three-class
# accuracy, and a session-level bootstrap puts the 95% interval on that gain at
# -0.002 to +0.040 -- it does not exclude zero. The accuracy claim is not made.
# The invariance is the reason.
EXTRA = ['mean_skel_len_per_obj_um']
MODEL_FEATURES = CLEAN + EXTRA

SEED, FOLDS, PURITY = 20260905, 5, 0.90


def band_edges(idx, y, purity):
    """Cuts meeting the purity requirement: below lo is elongated, above hi is
    fragmented. Returns (lo, hi) with lo < hi."""
    lo = hi = None
    for t in np.unique(np.round(idx, 3)):
        below = y[idx < t]
        if len(below) >= 20 and (below == 'elongated').mean() >= purity:
            lo = t
    for t in np.unique(np.round(idx, 3))[::-1]:
        above = y[idx > t]
        if len(above) >= 20 and (above == 'fragmented').mean() >= purity:
            hi = t
    return lo, hi


def main():
    f = pd.read_csv(os.path.join(ROOT, 'data', 'features_v5.csv'))
    l = pd.read_csv(os.path.join(ROOT, 'data', 'labels_consensus.csv'))
    d = f.merge(l[['image_id', 'blinded_call']], on='image_id')
    m = d[d.blinded_call.isin(MAIN)].reset_index(drop=True)
    y, g = m.blinded_call.values, m.session.values

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    cv = list(GroupKFold(n_splits=FOLDS).split(m, y, g))
    prob = cross_val_predict(pipe, m[MODEL_FEATURES], y, cv=cv, method='predict_proba')
    cls = list(np.unique(y))
    idx = prob[:, cls.index('normal')] * 0.5 + prob[:, cls.index('fragmented')] * 1.0

    print("== calibration of the out-of-fold probabilities ==")
    print("A well-calibrated model that says 0.7 is right about 70% of the time.")
    for c in MAIN:
        p = prob[:, cls.index(c)]
        print(f"  {c:<12} Brier {brier_score_loss((y == c).astype(int), p):.3f}")
    print(f"  {'reliability':<12}  predicted -> observed frequency of the top class")
    top, hit = prob.max(axis=1), (np.array(cls)[prob.argmax(axis=1)] == y)
    for lo_, hi_ in [(.33, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]:
        s = (top >= lo_) & (top < hi_)
        if s.sum():
            print(f"    {lo_:.2f}-{hi_:.2f}   n {s.sum():>4}   predicted {top[s].mean():.2f}"
                  f"   observed {hit[s].mean():.2f}")
    s = top >= .85
    print(f"    0.85 and up  n {s.sum():>4}   predicted {top[s].mean():.3f}   observed {hit[s].mean():.3f}")

    print("\n== the index as a two-class separator, ends of the scale only ==")
    ends = np.isin(y, ['fragmented', 'elongated'])
    auc = roc_auc_score((y[ends] == 'fragmented').astype(int), idx[ends])
    print(f"  fragmented vs elongated, out of fold: AUC {auc:.3f}  (n {ends.sum()})")

    print(f"\n== mixed band at {PURITY:.0%} purity ==")
    lo, hi = band_edges(idx, y, PURITY)
    inb = (idx >= lo) & (idx <= hi)
    print(f"  index < {lo:.3f}  -> elongated")
    print(f"  index > {hi:.3f}  -> fragmented")
    print(f"  between          -> mixed, no call")
    print(f"  coverage: {(~inb).mean():.1%} of images get a call, {inb.mean():.1%} abstain")
    for name, sel in [('below (elongated)', idx < lo), ('in the band', inb),
                      ('above (fragmented)', idx > hi)]:
        c = pd.Series(y[sel]).value_counts()
        print(f"  {name:<16} n {sel.sum():>4}   " +
              "  ".join(f"{k} {c.get(k, 0)}" for k in MAIN))

    print("\n  sensitivity of the band to the purity requirement")
    for p_ in (0.80, 0.85, 0.90, 0.95):
        a, b = band_edges(idx, y, p_)
        if a is None or b is None:
            print(f"    {p_:.0%}  no band satisfies this")
            continue
        cov = ((idx < a) | (idx > b)).mean()
        print(f"    {p_:.0%}  band {a:.3f}-{b:.3f}   {cov:.1%} of images get a call")

    print("\n== images any rater could not read ==")
    fit = pipe.fit(m[MODEL_FEATURES], y)
    unc = d[d.blinded_call == 'uncertain']
    pu = fit.predict_proba(unc[MODEL_FEATURES])
    iu = pu[:, cls.index('normal')] * 0.5 + pu[:, cls.index('fragmented')] * 1.0
    outside = ((iu < lo) | (iu > hi)).mean()
    print(f"  n {len(unc)}   median index {np.median(iu):.3f}")
    print(f"  {outside:.0%} fall OUTSIDE the band, i.e. would receive a confident call.")
    print("  The band is a statement about morphological ambiguity, not image quality.")
    print("  Unreadable images must be caught by the Stage 4 QC metrics, not by this index.")

    print("\n== stability across sessions ==")
    per = m.assign(index=idx).groupby('session')['index'].agg(['size', 'median'])
    per = per[per['size'] >= 5]
    print(f"  {len(per)} sessions with n >= 5: median index spans "
          f"{per['median'].min():.3f} to {per['median'].max():.3f}")
    kw = stats.kruskal(*[m.assign(index=idx).loc[m.session == s, 'index']
                         for s in per.index])
    print(f"  Kruskal-Wallis across sessions: H {kw.statistic:.1f}, p {kw.pvalue:.2g}")
    print("  Sessions differ in cell type and in acquisition. The index is a")
    print("  morphology measurement, so between-session spread is expected; what")
    print("  matters is that it is not driven by acquisition, checked below.")

    print("\n== confounds ==")
    for col, lab in [('img_mean_int', 'image mean intensity'), ('px_um', 'pixel size')]:
        r = stats.spearmanr(idx, m[col])
        print(f"  index vs {lab:<22} rho {r.statistic:+.2f}  p {r.pvalue:.2g}")

    out = m[['image_id', 'session', 'blinded_call']].assign(
        index=idx.round(4),
        band=np.where(idx < lo, 'elongated',
                      np.where(idx > hi, 'fragmented', 'mixed')))
    out.to_csv(os.path.join(ROOT, 'data', 'index_v7_calibrated.csv'), index=False)
    print(f"\nindex and band -> data/index_v7_calibrated.csv")

    # Written as numbers, not as a pickled estimator: a pickle carries the
    # scikit-learn version that made it, and this model fitted under 1.9 raised
    # AttributeError inside predict_proba when unpickled under 1.7. Into the
    # package, not beside it, so a refit cannot leave the shipped model behind.
    import json
    sc, lr = fit[0], fit[-1]
    assert list(lr.classes_) == list(cls), (lr.classes_, cls)
    spec = dict(kind='standardise + multinomial logistic regression',
                features=list(MODEL_FEATURES), classes=list(cls),
                mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                coef=lr.coef_.tolist(), intercept=lr.intercept_.tolist(),
                lo=float(lo), hi=float(hi), purity=float(PURITY), seed=int(SEED),
                fitted_with=f'scikit-learn {sklearn.__version__}')
    with open(os.path.join(ROOT, 'mitofi', 'models', 'index_model_v7.json'), 'w') as fh:
        json.dump(spec, fh, indent=1)
        fh.write('\n')
    print("fitted model and band -> mitofi/models/index_model_v7.json")


if __name__ == '__main__':
    main()
