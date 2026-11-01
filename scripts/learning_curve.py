"""Does rating more images help, or is the model already at its ceiling?

Two curves, because they answer different questions:

  by image   -- training sets are subsampled at random from the sessions already
                held. Answers "should I rate more fields from experiments I have
                already done".
  by session -- training sets are subsampled whole sessions at a time. Answers
                "do I need more experiments, not just more fields".

Test folds are always the full held-out sessions, so every point is comparable
and none of them is inflated by session leakage. Each point is repeated over
several seeds and reported as mean and standard deviation.

    .venv/bin/python scripts/learning_curve.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = ['fragmented', 'normal', 'elongated']
CLEAN = ['p90_aspect', 'p90_ecc', 'areawt_formfactor', 'med_aspect', 'med_ecc',
         'hole_area_frac', 'areawt_aspect', 'frac_fg_largest']
SEED, FOLDS, REPEATS = 20260905, 5, 8


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  random_state=SEED, n_jobs=-1)


def curve(X, y, g, sizes, mode, make):
    """Mean and sd of balanced accuracy at each training size."""
    folds = list(GroupKFold(n_splits=FOLDS).split(X, y, g))
    out = []
    for n in sizes:
        scores = []
        for rep in range(REPEATS):
            rng = np.random.default_rng(SEED + rep)
            for tr, te in folds:
                if mode == 'image':
                    if n > len(tr):
                        continue
                    pick = rng.choice(tr, n, replace=False)
                else:
                    sess = np.unique(g[tr])
                    if n > len(sess):
                        continue
                    keep = set(rng.choice(sess, n, replace=False))
                    pick = np.array([i for i in tr if g[i] in keep])
                if len(np.unique(y[pick])) < 3:
                    continue
                m = make().fit(X.iloc[pick], y[pick])
                scores.append(balanced_accuracy_score(y[te], m.predict(X.iloc[te])))
        if scores:
            out.append((n, len(scores), float(np.mean(scores)), float(np.std(scores))))
    return out


def show(title, rows, unit):
    print(f"\n{title}")
    print(f"  {unit:>10}{'balanced acc':>15}{'sd':>8}{'fits':>7}")
    for n, k, mu, sd in rows:
        bar = '#' * int(round((mu - 0.30) * 100))
        print(f"  {n:>10}{mu:>15.3f}{sd:>8.3f}{k:>7}  {bar}")


def main():
    f = pd.read_csv(os.path.join(ROOT, 'data', 'features_v5.csv'))
    l = pd.read_csv(os.path.join(ROOT, 'data', 'labels_blind.csv'))
    d = f.merge(l[['image_id', 'blinded_call']], on='image_id')
    m = d[d.blinded_call.isin(MAIN)].reset_index(drop=True)
    X, y, g = m[CLEAN], m.blinded_call.values, m.session.values
    ntrain = min(len(tr) for tr, _ in GroupKFold(n_splits=FOLDS).split(X, y, g))
    nsess = min(len(np.unique(g[tr])) for tr, _ in GroupKFold(n_splits=FOLDS).split(X, y, g))
    print(f"{len(m)} labelled images, {len(set(g))} sessions")
    print(f"smallest training fold: {ntrain} images from {nsess} sessions")
    print(f"{REPEATS} repeats per point, {FOLDS} folds, test folds always whole sessions")

    img_sizes = [40, 80, 120, 180, 240, 300, ntrain]
    sess_sizes = [4, 8, 12, 16, 20, 24, nsess]

    show("more images from the sessions you already have (logistic regression)",
         curve(X, y, g, img_sizes, 'image', lr), 'images')
    show("more images from the sessions you already have (random forest)",
         curve(X, y, g, img_sizes, 'image', rf), 'images')
    show("more sessions (logistic regression)",
         curve(X, y, g, sess_sizes, 'session', lr), 'sessions')
    show("more sessions (random forest)",
         curve(X, y, g, sess_sizes, 'session', rf), 'sessions')


if __name__ == '__main__':
    main()
