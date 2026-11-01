"""Derive the QC gate's haze limit and noise-texture floor from the development pool.

Both limits come from development data only, each by a rule that refuses about
one percent of good development fields by construction:

  MAX_HAZE       99th percentile of haze among the rater-1-readable fields,
                 rounded up to two decimals.
  MIN_LARGE_SNR  1st percentile of local signal-to-noise among the readable
                 fields whose objects are large (p90 major axis >= 2.25 um),
                 rounded down to two decimals. Large objects are what the
                 speckle rule never examines, and where a field segmented out
                 of noise texture -- one large hole-riddled sponge -- passed and
                 scored as confidently elongated.

Haze is measured the way production measures it: on the mask the shipped
Python classifier makes (not the Labkit masks), with the fast background
estimate. Masks are cached in data/masks_python_v6/ (gitignored) because
segmenting all 500 takes about 20 minutes.

    .venv/bin/python scripts/gate_thresholds.py
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import tifffile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mitofi.features import features, haze_fraction, local_contrast  # noqa: E402
from mitofi.score import MIN_P90_MAJOR_UM                       # noqa: E402
from mitofi.io import to_field                                  # noqa: E402
from mitofi.segment import segment                              # noqa: E402

CACHE = os.path.join(ROOT, 'data', 'masks_python_v6')
OUT = os.path.join(ROOT, 'data', 'gate_dev.csv')
PERCENTILE = 99


def mask_for(iid, img):
    p = os.path.join(CACHE, f'{iid}.ome_segmentation.tif')
    if os.path.exists(p):
        return tifffile.imread(p) > 0
    m = segment(img)
    os.makedirs(CACHE, exist_ok=True)
    tifffile.imwrite(p, m.astype(np.uint8), compression='zlib')
    return m


def main():
    cov = pd.read_csv(os.path.join(ROOT, 'data', 'covariates.csv'))
    r1 = pd.read_csv(os.path.join(ROOT, 'data', 'labels_blind.csv')).set_index('image_id')
    rows = []
    for k, (iid, px) in enumerate(zip(cov.image_id, cov.px_um), 1):
        img, _, _ = to_field(os.path.join(ROOT, 'data', 'images', f'{iid}.ome.tif'))
        m = mask_for(iid, img)
        f = features(m, img, px) if m.any() else {}
        rows.append(dict(image_id=iid, rater1=r1.loc[iid, 'blinded_call'],
                         haze=haze_fraction(img, m, px) if m.any() else float('nan'),
                         p90_major_um=f.get('p90_major_um', float('nan')),
                         local_snr=(local_contrast(img, m, px, local=True)
                                    if f.get('n_objects') else float('nan'))))
        if k % 100 == 0:
            print(f'  {k}/{len(cov)}', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    read = d[d.rater1 != 'uncertain'].haze.dropna()
    unc = d[d.rater1 == 'uncertain'].haze.dropna()
    limit = math.ceil(np.percentile(read, PERCENTILE) * 100) / 100
    print(f'\n{len(read)} readable development fields: haze median {read.median():.3f}, '
          f'p95 {read.quantile(.95):.3f}, p{PERCENTILE} {read.quantile(PERCENTILE / 100):.3f}, '
          f'max {read.max():.3f}')
    print(f'{len(unc)} uncertain: median {unc.median():.3f}, max {unc.max():.3f}')
    print(f'\nMAX_HAZE = {limit:.2f}')
    print(f'  refuses {(read > limit).sum()} of {len(read)} readable development fields '
          f'and {(unc > limit).sum()} of {len(unc)} uncertain')

    big = d[d.p90_major_um >= MIN_P90_MAJOR_UM]
    rb = big[big.rater1 != 'uncertain'].local_snr.dropna()
    ub = big[big.rater1 == 'uncertain'].local_snr.dropna()
    floor = math.floor(np.percentile(rb, 100 - PERCENTILE) * 100) / 100
    print(f'\n{len(rb)} readable large-object fields: local SNR median {rb.median():.2f}, '
          f'p1 {rb.quantile(.01):.2f}')
    print(f'MIN_LARGE_SNR = {floor:.2f}')
    print(f'  refuses {(rb < floor).sum()} of {len(rb)} readable large-object fields '
          f'and {(ub < floor).sum()} of {len(ub)} uncertain')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
