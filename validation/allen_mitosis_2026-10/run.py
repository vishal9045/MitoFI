"""Score the frozen Allen TOMM20 sample once, as PLAN.md fixes.

Per cell: download the crop and its masks, put the TOMM20 channel through the
tool's own reader (maximum projection, bit depth), score it twice -- the whole
crop as a user would, and only the mitochondria inside Allen's cell outline --
each with the tool's rescue chain for refused fields. The 3D files are deleted
after use. Resumable: cells already in results.csv are skipped.

    .venv/bin/python validation/allen_mitosis_2026-10/run.py [--jobs N]
"""
import csv
import os
import sys
import tempfile
import urllib.request
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
BASE = 'https://allencell.s3.amazonaws.com/aics/hipsc_single_cell_image_dataset/'
OUT = os.path.join(HERE, 'results.csv')
FIELDS = ['CellId', 'group', 'cell_stage', 'FOVId', 'matched_to', 'px_um', 'cell_frac']
for a in ('whole', 'cell'):
    FIELDS += [f'{a}_pass', f'{a}_reason', f'{a}_index', f'{a}_band', f'{a}_rescue',
               f'{a}_haze', f'{a}_local_snr']
FIELDS += ['error']


def get(url, path, tries=3):
    for k in range(tries):
        try:
            urllib.request.urlretrieve(url, path)
            return
        except Exception:
            if k == tries - 1:
                raise


def chain(variants, px, cellmask=None):
    """The tool's order: as is, background, black point -- stop at the first pass."""
    from mitofi.cli import _rescuable
    from mitofi.score import score_mask
    r = None
    for name, (img, seg) in variants():
        m = seg if cellmask is None else seg & cellmask
        r = score_mask(m, px, img=img)
        r['rescue'] = name
        if r['qc_pass'] or not _rescuable(r):
            return r
    return r



def one(r):
    from mitofi.io import subtract_background, subtract_floor, to_field
    from mitofi.segment import segment
    import tifffile
    out = {k: r[k] for k in ('CellId', 'group', 'cell_stage', 'FOVId', 'matched_to')}
    tmp = tempfile.mkdtemp()
    raw, seg = os.path.join(tmp, 'raw.ome.tif'), os.path.join(tmp, 'seg.ome.tif')
    try:
        names = eval(r['name_dict'])
        get(BASE + r['crop_raw'], raw)
        get(BASE + r['crop_seg'], seg)
        img, _, _ = to_field(raw, channel=names['crop_raw'].index('structure'))
        img = np.asarray(img, np.float32)
        s = tifffile.imread(seg)
        cell = s[:, names['crop_seg'].index('membrane_segmentation')].max(0) > 0
        px = float(eval(r['scale_micron'])[0])          # the files record none
        cache = {}

        def variants():
            for name in ('', 'background', 'black point'):
                if name not in cache:
                    v = (img if name == '' else
                         subtract_background(img, px) if name == 'background' else
                         subtract_floor(img, px, peak=True)[0])
                    cache[name] = (v, segment(v))
                yield name, cache[name]

        out.update(px_um=px, cell_frac=round(float(cell.mean()), 3))
        for a, cm in (('whole', None), ('cell', cell)):
            q = chain(variants, px, cm)
            out.update({f'{a}_pass': q['qc_pass'], f'{a}_reason': q['qc_reason'],
                        f'{a}_index': q.get('index'), f'{a}_band': q.get('band'),
                        f'{a}_rescue': q['rescue'], f'{a}_haze': q.get('haze'),
                        f'{a}_local_snr': q.get('local_snr')})
    except Exception as e:                              # reported, never dropped
        out['error'] = repr(e)
    finally:
        for p in (raw, seg):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(tmp)
    return out


def main(jobs):
    rows = list(csv.DictReader(open(os.path.join(HERE, 'inputs.csv'))))
    done = set()
    if os.path.exists(OUT):
        done = {x['CellId'] for x in csv.DictReader(open(OUT))}
    todo = [x for x in rows if x['CellId'] not in done]
    new = not os.path.exists(OUT)
    with open(OUT, 'a', newline='') as fh, Pool(jobs) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        if new:
            w.writeheader()
        for i, o in enumerate(pool.imap_unordered(one, todo), 1):
            w.writerow(o); fh.flush()
            print(f"{i}/{len(todo)} {o['group']:<10} whole "
                  f"{o.get('whole_index') if o.get('whole_pass') else 'refused'} | cell "
                  f"{o.get('cell_index') if o.get('cell_pass') else 'refused'} {o.get('error', '')}",
                  flush=True)


if __name__ == '__main__':
    main(int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 8)
