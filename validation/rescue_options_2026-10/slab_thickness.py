"""Does a thin slab cut tubules into pieces? Index against slab thickness, Allen.

Each of the 238 cells: slab of 0.5, 1, 2 and 4 um around the sharpest plane,
and the full stack, each followed by the shipped black point; mask kept inside
Allen's cell outline. See PLAN.md, "Slab thickness".

    .venv/bin/python validation/rescue_options_2026-10/slab_thickness.py [summary]
"""
import os
import sys
import tempfile
import urllib.request
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, 'validation', 'scribbles_v2_retrain')]
OUT = os.path.join(HERE, 'allen_slab_thickness.csv')
THICK = [0.5, 1.0, 2.0, 4.0, 'full']


def one(r):
    import tifffile
    from run import slab
    from retrain import ALLEN_BASE
    from mitofi.io import _significant_bits, subtract_floor, to_field, to_trained_scale
    from mitofi.score import score_mask
    from mitofi.segment import segment
    base = dict(CellId=r['CellId'], group=r['group'], matched_to=r['matched_to'])
    tmp = tempfile.mkdtemp()
    raw, seg = os.path.join(tmp, 'raw.ome.tif'), os.path.join(tmp, 'seg.ome.tif')
    try:
        nd = eval(r['name_dict'])
        for url, p in ((r['crop_raw'], raw), (r['crop_seg'], seg)):
            for t in range(3):
                try:
                    urllib.request.urlretrieve(ALLEN_BASE + url, p); break
                except Exception:
                    if t == 2:
                        raise
        ch = nd['crop_raw'].index('structure')
        mip = np.asarray(to_field(raw, channel=ch)[0], np.float32)
        st = np.asarray(to_trained_scale(tifffile.imread(raw)[:, ch], _significant_bits(raw))[0], np.float32)
        cell = tifffile.imread(seg)[:, nd['crop_seg'].index('membrane_segmentation')].max(0) > 0
        px, out = 0.108333, []
        for th in THICK:
            a = mip if th == 'full' else np.asarray(slab(st, px, th / 2), np.float32)
            b = subtract_floor(a, px, peak=True)[0]
            q = score_mask(segment(b, jobs=1) & cell, px, img=b)
            out.append(dict(base, thickness=str(th), qc_pass=q['qc_pass'], index=q.get('index'),
                            raw_index=q.get('index_unvalidated'), reason=q['qc_reason']))
        return out
    except Exception as e:
        return [dict(base, thickness='error', reason=repr(e))]
    finally:
        for p in (raw, seg):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(tmp)


def summary():
    d = pd.read_csv(OUT)
    d = d[d.thickness != 'error']
    L = ['Slab thickness on Allen (inside the cell, black point after the slab). DEVELOPMENT.', '',
         f'{"slab":>6}  {"scored":>7}  {"mitotic":>8}  {"interph":>8}  {"gap":>6}   by stage M0 / M1M2 / M3 / M4M5',
         '   (medians of cells that pass; "raw" = same, all cells, gate ignored)']
    med = {}
    for th in [str(t) for t in THICK]:
        g = d[d.thickness == th]
        ok = g[g.qc_pass == True]                                           # noqa: E712
        mi, it = ok[ok.group != 'M0']['index'].median(), ok[ok.group == 'M0']['index'].median()
        rmi, rit = g[g.group != 'M0'].raw_index.median(), g[g.group == 'M0'].raw_index.median()
        med[th] = (mi, it)
        st = ' / '.join(f'{ok[ok.group == s]["index"].median():.2f}' for s in ('M0', 'M1M2', 'M3', 'M4M5'))
        L.append(f'{th:>6}  {len(ok):>3}/{len(g):<3}  {mi:>8.3f}  {it:>8.3f}  {mi - it:>+6.3f}   {st}'
                 f'   raw {rmi:.3f} / {rit:.3f}')
    dm = abs(med['2.0'][0] - med['4.0'][0]); di = abs(med['2.0'][1] - med['4.0'][1])
    ok = dm <= 0.05 and di <= 0.05
    L += ['', f'2 vs 4 um: mitotic moves {dm:.3f}, interphase {di:.3f} -> '
          f'{"PLATEAU (rule met)" if ok else "no plateau (rule not met)"}']
    open(os.path.join(HERE, 'slab_thickness_summary.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    rows_in = pd.read_csv(os.path.join(ROOT, 'validation', 'scribbles_v2_retrain', 'allen_inputs.csv')
                          ).fillna('').to_dict('records')
    rows = []
    with Pool(11) as pool:
        for k, o in enumerate(pool.imap_unordered(one, rows_in), 1):
            rows += o
            if k % 20 == 0:
                print(f'{k}/{len(rows_in)} cells', flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    summary()


if __name__ == '__main__':
    summary() if 'summary' in sys.argv else main()
