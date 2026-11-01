"""Print the feature values and index for named images.

Used to show, for a handful of images, exactly what the model received and what
it returned -- the table quoted when explaining the pipeline.

    .venv/bin/python scripts/worked_example.py [image_id ...]
"""
import os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = ['img_26e13e', 'img_b5f009', 'img_e9d85d']
SHOW = ['image_id', 'blinded_call', 'n_objects', 'p90_aspect', 'p90_ecc',
        'areawt_formfactor', 'med_aspect', 'med_ecc', 'hole_area_frac',
        'areawt_aspect', 'frac_fg_largest', 'index', 'band']


def main(ids):
    f = pd.read_csv(os.path.join(ROOT, 'data', 'features_v5.csv'))
    l = pd.read_csv(os.path.join(ROOT, 'data', 'labels_consensus.csv'))
    # index_v7_calibrated.csv, not index_v7.csv: Stage 6 writes the latter
    # without a `band` column, so reading the band from it worked only until
    # Stage 6 was next re-run. The band is written by Stage 7, here.
    i = pd.read_csv(os.path.join(ROOT, 'data', 'index_v7_calibrated.csv'))
    d = (f.merge(l[['image_id', 'blinded_call']], on='image_id')
          .merge(i[['image_id', 'index', 'band']], on='image_id', how='left'))
    missing = set(ids) - set(d.image_id)
    if missing:
        print("not found:", sorted(missing))
    sel = d[d.image_id.isin(ids)][SHOW]
    print(sel.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == '__main__':
    main(sys.argv[1:] or DEFAULT)
