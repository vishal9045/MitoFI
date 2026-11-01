"""Stage 6: predict the blinded call from the Stage 5 features, held out by session.

Validation is GroupKFold on session. Random cross-validation would place images
from the same dish in both train and test, and the model would score well by
recognising a session's illumination rather than its morphology. The script
reports both so the size of that inflation is visible.

The primary model uses only features that separate the calls AND are clean of
the pixel-size and intensity confounds, as selected by rank_features.py.

Labels are the three-rater consensus from consensus_labels.py: the call two of
three blind raters made. `uncertain` (any rater could not read the image) is
excluded from training -- it denotes unreadable signal, not a morphology class --
and then scored, to see where the model places it. `split` (all three disagreed)
is excluded and not scored.

    .venv/bin/python scripts/fit_model.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = ['fragmented', 'normal', 'elongated']
SEED = 20260905
FOLDS = 5

# clean = separates the calls (eps2 > 0.1) and |rho| <= 0.4 on both pixel size
# and intensity. Selected by rank_features.py, restated here so the model is
# reproducible without rerunning the ranking.
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



def load():
    f = pd.read_csv(os.path.join(ROOT, 'data', 'features_v5.csv'))
    l = pd.read_csv(os.path.join(ROOT, 'data', 'labels_consensus.csv'))
    return f.merge(l[['image_id', 'blinded_call']], on='image_id')


def models():
    return {
        'logistic regression': make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        'random forest': RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    }


def score(name, y, pred):
    print(f"  {name:<34} acc {accuracy_score(y, pred):.3f}   "
          f"balanced {balanced_accuracy_score(y, pred):.3f}   "
          f"macro F1 {f1_score(y, pred, average='macro'):.3f}")


def main():
    d = load()
    m = d[d.blinded_call.isin(MAIN)].reset_index(drop=True)
    y = m.blinded_call.values
    g = m.session.values
    allfeat = [c for c in d.columns
               if c not in {'image_id', 'session', 'blinded_call'} and d[c].dtype != object]

    print(f"training images {len(m)}  (uncertain held out: {(d.blinded_call == 'uncertain').sum()}, "
          f"three-way split: {(d.blinded_call == 'split').sum()})")
    print(f"sessions {len(set(g))}   folds {FOLDS}   clean features {len(MODEL_FEATURES)}   "
          f"all features {len(allfeat)}\n")

    gkf = list(GroupKFold(n_splits=FOLDS).split(m, y, g))
    skf = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(m, y))

    print("held out by session (GroupKFold) -- the honest number")
    dummy = cross_val_predict(DummyClassifier(strategy='most_frequent'), m[MODEL_FEATURES], y, cv=gkf)
    score('majority-class baseline', y, dummy)
    best = None
    for name, mdl in models().items():
        p = cross_val_predict(mdl, m[MODEL_FEATURES], y, cv=gkf)
        score(name + ', clean features', y, p)
        if best is None:
            best = p
        p2 = cross_val_predict(mdl, m[allfeat], y, cv=gkf)
        score(name + ', all features', y, p2)

    print("\nrandom split (StratifiedKFold) -- inflated, shown for comparison")
    for name, mdl in models().items():
        p = cross_val_predict(mdl, m[MODEL_FEATURES], y, cv=skf)
        score(name + ', clean features', y, p)

    print("\nconfusion matrix, logistic regression on clean features, held out by session")
    cm = confusion_matrix(y, best, labels=MAIN)
    print(f"  {'true \\ pred':<14}" + "".join(f"{c[:6]:>9}" for c in MAIN) + f"{'n':>7}")
    for i, c in enumerate(MAIN):
        print(f"  {c:<14}" + "".join(f"{v:>9}" for v in cm[i]) + f"{cm[i].sum():>7}")

    print("\nstandardised coefficients (positive = pushes towards that call)")
    fit = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=SEED)).fit(m[MODEL_FEATURES], y)
    lr = fit[-1]
    print(f"  {'feature':<22}" + "".join(f"{c[:10]:>12}" for c in lr.classes_))
    for j, f in enumerate(MODEL_FEATURES):
        print(f"  {f:<22}" + "".join(f"{lr.coef_[i][j]:>12.2f}" for i in range(len(lr.classes_))))

    # fragmentation index: 1 = fragmented, 0 = elongated
    prob = cross_val_predict(make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=SEED)),
                             m[MODEL_FEATURES], y, cv=gkf, method='predict_proba')
    order = {c: i for i, c in enumerate(sorted(set(y)))}
    idx = (prob[:, order['normal']] * 0.5 + prob[:, order['fragmented']] * 1.0)
    m = m.assign(index=idx)
    print("\nfragmentation index by call (1 = fragmented, 0 = elongated)")
    for c in MAIN:
        v = m.loc[m.blinded_call == c, 'index']
        print(f"  {c:<12} median {v.median():.3f}   IQR {v.quantile(.25):.3f}-{v.quantile(.75):.3f}")

    print("\nconfound check on the index")
    from scipy import stats
    for col, lab in [('img_mean_int', 'image mean intensity'), ('px_um', 'pixel size')]:
        r = stats.spearmanr(m['index'], m[col])
        print(f"  vs {lab:<22} rho = {r.statistic:+.2f}   p = {r.pvalue:.2g}")

    # where does the model place the images the rater could not read?
    unc = d[d.blinded_call == 'uncertain']
    if len(unc):
        pu = fit.predict_proba(unc[MODEL_FEATURES])
        iu = pu[:, order['normal']] * 0.5 + pu[:, order['fragmented']] * 1.0
        conf = pu.max(axis=1)
        print("\nuncertain images scored by the fitted model (not in training)")
        print(f"  n {len(unc)}   index median {np.median(iu):.3f}   "
              f"IQR {np.percentile(iu,25):.3f}-{np.percentile(iu,75):.3f}")
        print(f"  top-class probability: median {np.median(conf):.3f} vs "
              f"{np.median(prob.max(axis=1)):.3f} for rated images")

    out = os.path.join(ROOT, 'data', 'index_v7.csv')
    m[['image_id', 'session', 'blinded_call', 'index']].to_csv(out, index=False)
    print(f"\nout-of-fold index -> data/index_v7.csv")


if __name__ == '__main__':
    main()
