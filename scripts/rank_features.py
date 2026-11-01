"""Rank the Stage 5 features by how well they separate the blinded calls,
and screen each one for dependence on pixel size and image intensity.

Effect size is epsilon-squared from the Kruskal-Wallis H, computed over the
three substantive calls only; `uncertain` is excluded because it denotes
unreadable signal rather than a morphology class.

The confound columns are Spearman |rho| against px_um and mean image intensity.
A feature that separates the calls well AND tracks pixel size is not measuring
morphology alone -- the pool spans a 4.3-fold range of pixel sizes.

    python3 scripts/rank_features.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = ['fragmented', 'normal', 'elongated']
SKIP = {'image_id', 'session', 'blinded_call', 'frame_px', 'sizeZ'}
CONFOUND = 0.4          # |rho| above which a feature is flagged
MIN_EFFECT = 0.02       # only flag features that actually carry signal


def load():
    f = pd.read_csv(os.path.join(ROOT, 'data', 'features_v5.csv'))
    l = pd.read_csv(os.path.join(ROOT, 'data', 'labels_blind.csv'))
    return f.merge(l[['image_id', 'blinded_call']], on='image_id')


def rank(d):
    m = d[d.blinded_call.isin(MAIN)]
    out = []
    for c in [c for c in d.columns if c not in SKIP and d[c].dtype != object]:
        groups = [m.loc[m.blinded_call == k, c].values for k in MAIN]
        H, p = stats.kruskal(*groups)
        eps2 = (H - 2) / (len(m) - 3)                    # 3 groups: k-1 = 2
        med = [float(np.median(g)) for g in groups]
        out.append(dict(
            feature=c, eps2=eps2, p=p,
            r_px=abs(stats.spearmanr(m[c], m.px_um).statistic),
            r_int=abs(stats.spearmanr(m[c], m.img_mean_int).statistic),
            monotone=(med[0] < med[1] < med[2]) or (med[0] > med[1] > med[2]),
            med_F=med[0], med_N=med[1], med_E=med[2]))
    return pd.DataFrame(out).sort_values('eps2', ascending=False).reset_index(drop=True)


def main():
    d = load()
    r = rank(d)
    print(f"{len(d)} images, {r.shape[0]} features, calls used: {', '.join(MAIN)}\n")
    print(f"{'feature':<26}{'eps2':>7}{'p':>10}{'|r|px':>7}{'|r|int':>7}{'mono':>6}"
          f"   medians F/N/E")
    for _, x in r.head(20).iterrows():
        print(f"{x.feature:<26}{x.eps2:7.3f}{x.p:10.2e}{x.r_px:7.2f}{x.r_int:7.2f}"
              f"{'yes' if x.monotone else '-':>6}   {x.med_F:.3g} {x.med_N:.3g} {x.med_E:.3g}")

    flag = r[(r.eps2 > MIN_EFFECT) & ((r.r_px > CONFOUND) | (r.r_int > CONFOUND))]
    print(f"\nconfounded (eps2 > {MIN_EFFECT}, |rho| > {CONFOUND} on pixel size or intensity): "
          f"{len(flag)} of {len(r)}")
    for _, x in flag.iterrows():
        print(f"  {x.feature:<26} eps2={x.eps2:.3f}  |r|px={x.r_px:.2f}  |r|int={x.r_int:.2f}")

    clean = r[(r.eps2 > 0.1) & (r.r_px <= CONFOUND) & (r.r_int <= CONFOUND)]
    print(f"\nusable for the index -- eps2 > 0.1 and clean on both confounds:")
    for _, x in clean.iterrows():
        print(f"  {x.feature:<26} eps2={x.eps2:.3f}  |r|px={x.r_px:.2f}  |r|int={x.r_int:.2f}")
    r.to_csv(os.path.join(ROOT, 'data', 'feature_ranking.csv'), index=False)
    print(f"\nfull table -> data/feature_ranking.csv")


if __name__ == '__main__':
    main()
