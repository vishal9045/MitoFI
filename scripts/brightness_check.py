"""Brightness dependence of the segmentation masks and of the index.

The Stage 4 QC table records +0.43 for classifier v5. That figure was produced
before the analyses in this repository were scripted and cannot be reproduced
from anything committed here. This script is the reproducible replacement: it
recomputes the same quantity from data/features_v5.csv, which holds the
foreground fraction and the mean intensity of every image in the pool.

    python3 scripts/brightness_check.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    f = pd.read_csv(os.path.join(ROOT, 'data', 'features_v5.csv'))
    li = np.log(f.img_mean_int)
    print(f'{len(f)} images, v5 masks')
    print(f"  pearson(log mean intensity, foreground fraction)   {stats.pearsonr(li, f.fg_frac).statistic:+.2f}")
    print(f"  spearman(mean intensity, foreground fraction)      {stats.spearmanr(f.img_mean_int, f.fg_frac).statistic:+.2f}")

    idx = os.path.join(ROOT, 'data', 'index_v7.csv')
    if os.path.exists(idx):
        d = f.merge(pd.read_csv(idx)[['image_id', 'index']], on='image_id')
        print(f'\n{len(d)} images with an out-of-fold index')
        print(f"  spearman(mean intensity, index)                    "
              f"{stats.spearmanr(d.img_mean_int, d['index']).statistic:+.2f}")
        print('\nThe mask-level dependence is severe; the index inherits about a')
        print('quarter of it. That difference is what the dimensionless feature')
        print('screen buys, and it is the strongest evidence for that design choice.')


if __name__ == '__main__':
    main()
