"""What three tools return for the same ten fields, and whether they agree.

Not a performance benchmark. Five fields per group cannot rank tools: seven of
the fifteen measures below reach AUC 1.000 simply because perfect separation at
n=5 needs only p=0.0079, and every effect-size interval overlaps every other.
What ten fields can show is the shape of the output -- how many numbers each
tool returns, whether one of them is the answer, and whether they point the
same way.

Provenance, stated exactly because it is not "we ran the published tools":

  MitoFI                 the shipped pipeline, unmodified.
  Mitochondria Analyzer  its own 2D threshold pipeline (1.25 um block,
                         C-value 10, Weighted Mean, full pre/post-process)
                         run headless from its installed plugin, because the
                         shipped macros are dialog-driven. Descriptors are then
                         computed from ITS mask by mitofi.features, since its
                         2DAnalysis macro cannot run without a dialog.
  MiNA                   its measurement path -- Otsu, Skeletonize (2D/3D),
                         AnalyzeSkeleton_ -- driven headless over the same
                         fields, with MiNA's published defaults and no
                         preprocessing.

    .venv/bin/python scripts/tool_comparison.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from scipy.stats import mannwhitneyu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mitofi.features import features                            # noqa: E402
from mitofi.io import to_field                                  # noqa: E402
from mitofi.score import score_mask                             # noqa: E402
from mitofi.segment import segment                              # noqa: E402

BENCH = '/mnt/d/Softwares/Fiji.app/_bench'
FIELDS, MA_MASKS, MINA_CSV = f'{BENCH}/in', f'{BENCH}/ma_out', f'{BENCH}/mina.csv'
OUT = os.path.join(ROOT, 'data', 'tool_comparison_10mef.csv')

# What each tool reports to a user for one 2D field, from its own documentation.
REPORTED = {
    'MitoFI': ('index, band, confidence, QC verdict', 1,
               'index', 'yes, one designated score'),
    'Mitochondria Analyzer': ('count, total/mean area, total/mean perimeter, '
                              'mean aspect ratio, mean form factor, branches, '
                              'total/mean branch length, junctions, end points, '
                              'mean branch diameter', 13, None, 'no'),
    'MiNA': ('footprint, individuals, networks, branch length mean/median/sd, '
             'summed branch length mean/median/sd, network branches '
             'mean/median/sd', 12, None, 'no'),
}


def collect():
    rows = []
    for p in sorted(glob.glob(os.path.join(FIELDS, '*.tif'))):
        iid = os.path.basename(p)[:-4]
        img, px, _ = to_field(p)
        r = score_mask(segment(img), px, img=img)
        d = {'image_id': iid,
             'group': 'MFN1_KO' if iid.startswith('MFN1') else 'WT',
             'mitofi_index': r.get('index'), 'mitofi_band': r.get('band')}
        m = tifffile.imread(os.path.join(MA_MASKS, iid + '_MAmask.tif')) > 0
        f = features(m, img, px)
        d.update(ma_count=f.get('n_objects'), ma_mean_area_um2=f.get('med_area_um2'),
                 ma_mean_perim_um=f.get('med_perim_um'), ma_mean_aspect=f.get('med_aspect'),
                 ma_mean_formfactor=f.get('med_formfactor'),
                 ma_total_branch_len_um=f.get('skel_len_um'),
                 ma_junctions_per_100um=f.get('junctions_per_100um'),
                 ma_endpoints_per_100um=f.get('endpoints_per_100um'))
        rows.append(d)
    d = pd.DataFrame(rows)
    mina = pd.read_csv(MINA_CSV)
    mina['image_id'] = mina.image_id.str.replace('.tif', '', regex=False)
    return d.merge(mina.drop(columns=['px_um']), on='image_id')


def separation(d):
    ko = (d.group == 'MFN1_KO').values
    rng = np.random.default_rng(0)
    out = []
    for c in d.columns:
        if c in ('image_id', 'group') or d[c].dtype == object:
            continue
        a, b = d.loc[ko, c].values, d.loc[~ko, c].values
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        if np.std(np.r_[a, b]) == 0:
            continue
        U, p = mannwhitneyu(a, b)
        s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        boot = []
        for _ in range(20000):
            x, y = rng.choice(a, len(a)), rng.choice(b, len(b))
            ss = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
            if ss > 0:
                boot.append(abs(x.mean() - y.mean()) / ss)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append(dict(
            measure=c, tool=('MitoFI' if c.startswith('mitofi') else
                             'Mitochondria Analyzer' if c.startswith('ma_') else 'MiNA'),
            ko_mean=a.mean(), wt_mean=b.mean(),
            direction='higher in KO' if a.mean() > b.mean() else 'lower in KO',
            auc=max(U / (len(a) * len(b)), 1 - U / (len(a) * len(b))),
            cohens_d=abs(a.mean() - b.mean()) / s if s else np.nan,
            ci_lo=lo, ci_hi=hi, p=p))
    return pd.DataFrame(out)


def main():
    d = collect()
    r = separation(d)
    d.to_csv(OUT, index=False)

    print('== what each tool returns for one 2D field ==')
    print(f'  {"tool":<24}{"numbers":>8}  {"a designated fragmentation score?":<34}')
    for t, (_, n, _, desig) in REPORTED.items():
        print(f'  {t:<24}{n:>8}  {desig:<34}')

    print('\n== direction of change, MFN1-knockout versus wild type ==')
    print(f'  {"measure":<26}{"tool":<24}{"direction":<16}{"AUC":>6}{"d":>7}')
    for _, x in r.sort_values(['tool', 'measure']).iterrows():
        print(f'  {x.measure:<26}{x.tool:<24}{x.direction:<16}{x.auc:>6.2f}{x.cohens_d:>7.2f}')

    for t in ('Mitochondria Analyzer', 'MiNA'):
        s = r[r.tool == t]
        up, down = (s.direction == 'higher in KO').sum(), (s.direction == 'lower in KO').sum()
        print(f'\n  {t}: {up} measures higher in the knockout, {down} lower. '
              f'Choosing which of\n  these means "fragmented" is left to the reader.')

    print(f'\n== separation, for completeness ==')
    print(f'  measures reaching AUC 1.000: {(r.auc >= 0.999).sum()} of {len(r)}; '
          f'at n=5 per group the\n  smallest attainable p is 0.0079, so perfect '
          f'separation is weak evidence.')
    print(f'  {"measure":<26}{"d":>7}{"   95% CI":>18}')
    for _, x in r.sort_values('cohens_d', ascending=False).head(4).iterrows():
        print(f'  {x.measure:<26}{x.cohens_d:>7.2f}   [{x.ci_lo:>5.2f}, {x.ci_hi:>6.2f}]')
    print('  Every interval overlaps every other. These ten fields cannot rank the tools.')
    print(f'\n-> {OUT}')


if __name__ == '__main__':
    main()
