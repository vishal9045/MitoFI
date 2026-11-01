"""Score every series of every .lif in testing_data/test lifs, as PLAN.md fixes.

Per series: the mitochondrial channel from inventory.csv, maximum-projected over
Z (timepoint 0), put on the trained scale from its true bit depth, segmented
and scored; a field the gate refuses is scored again after the background
rescue (rolling ball, 2.5 um), the one correction the tool offers.

Resumable: rows already in results.csv are skipped.

    .venv/bin/python validation/all_lif_2026-10/run.py [--jobs N]
"""
import csv
import os
import re
import sys
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
LIFS = os.path.join(os.path.dirname(ROOT), 'testing_data', 'test lifs')
OUT = os.path.join(HERE, 'results.csv')
FIELDS = ['lif', 'series', 'name', 'condition', 'px_um', 'mito_channel', 'channel_source',
          'first_pass', 'first_reason', 'first_index', 'rescued', 'final_pass',
          'final_reason', 'index', 'band', 'confidence', 'haze', 'local_snr', 'error']


def condition(name):
    """Series name without numbering, zoom, replicate, batch and position tags."""
    s = name.split('/')[0].lower()
    s = re.sub(r'\d+(\.\d+)?\s*zoom|zoom\s*\d+(\.\d+)?', ' ', s)
    s = re.sub(r'\b(rep(licate)?|image|guess|oe)\b', ' ', s)
    s = re.sub(r'\bb\d\b|\bp\d+(\s*-?\s*p\d+)?\b|\bi\d\b', ' ', s)
    s = re.sub(r'\b\d+\b(?!\s*(mg|h|hpi|ko)\b)', ' ', s)
    # spelling only: variants of the same word, never different conditions
    for pat, rep in ((r'cdk i\b', 'cdki'), (r'(\d+)\s+mg', r'\1mg'), (r'pyru\w*', 'pyruvate'),
                     (r'\b(?<!mfn)1\s?ko\b', 'mfn1 ko'), (r'\b(?<!mfn)2\s?ko\b', 'mfn2 ko'),
                     (r'\bvehicle\b', 'veh'), (r'\bmock2\b', 'mock'), (r'\binf1\b', 'inf'),
                     (r'(\d+)h hpi', r'\1hpi'), (r'mdivi-', 'mdivi'), (r'poool', 'pool'),
                     (r'\bb\b|\s\.$', ' '), (r'huh-', 'huh-7')):
        s = re.sub(pat, rep, s)
    return re.sub(r'\s+', ' ', s).strip()


def score_one(r):
    from readlif.reader import LifFile
    from mitofi.io import subtract_background, to_trained_scale
    from mitofi.score import score_mask
    from mitofi.segment import segment
    out = dict(lif=r['lif'], series=r['series'], name=r['name'],
               condition=condition(r['name']), px_um=r['px_um'],
               mito_channel=r['mito_channel'], channel_source=r['channel_source'])
    try:
        img = LifFile(os.path.join(LIFS, r['lif'])).get_image(int(r['series']))
        c = int(float(r['mito_channel']))
        a = np.max([np.asarray(img.get_frame(z=z, t=0, c=c)) for z in range(int(r['z']))], 0)
        a, _ = to_trained_scale(a, int(r['bits']))
        a = np.asarray(a, np.float32)
        px = float(r['px_um'])
        first = score_mask(segment(a), px, img=a)
        final, rescued = first, False
        if not first['qc_pass']:
            b = subtract_background(a, px)
            final = score_mask(segment(b), px, img=b)
            rescued = bool(final['qc_pass'])
        out.update(first_pass=first['qc_pass'], first_reason=first['qc_reason'],
                   first_index=first.get('index'), rescued=rescued,
                   final_pass=final['qc_pass'], final_reason=final['qc_reason'],
                   index=final.get('index'), band=final.get('band'),
                   confidence=final.get('confidence'), haze=final.get('haze'),
                   local_snr=final.get('local_snr'))
    except Exception as e:                       # reported, never dropped
        out.update(first_pass=False, final_pass=False, final_reason=f'could not be read: {e}',
                   error=repr(e))
    return out


def main(jobs):
    inv = list(csv.DictReader(open(os.path.join(HERE, 'inventory.csv'))))
    done = set()
    if os.path.exists(OUT):
        done = {(r['lif'], r['series']) for r in csv.DictReader(open(OUT))}
    todo = [r for r in inv if (r['lif'], r['series']) not in done]
    new = not os.path.exists(OUT)
    with open(OUT, 'a', newline='') as fh, Pool(jobs) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(score_one, todo), 1):
            w.writerow(row); fh.flush()
            print(f"{i}/{len(todo)} {row['lif'].split('/')[-1][:30]} s{row['series']} "
                  f"{row.get('index') if row.get('final_pass') else row.get('final_reason', '')[:40]}",
                  flush=True)


if __name__ == '__main__':
    main(int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 6)
