"""BBBC053 as published, one tool run per image, 11 at a time.

Added after the plan (2026-10-28) for speed only: on these 2048^2 images at
0.02 um/px the 2.5 um rolling ball is a 125-pixel radius, so each refused field
took ~4 min per rescue and the folder runs one image at a time (2-3 h left at
the time). A field's score does not depend on the other files in its folder, so
the rows are the same; this is checked against the rows the folder runs had
already written (kept outside the repo before they were stopped).

Each image is linked alone into a temporary folder and run through the same
command, `python -m mitofi.cli score <folder> --rescue -o <csv>`; the rows are
then joined into results/bbbc053_asis/<group>.csv.

    .venv/bin/python validation/public_sets_2026-10/run_split.py
"""
import glob
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from run import PUB, PY, ROOT, HERE, check_tool

OUT = os.path.join(HERE, 'results', 'bbbc053_asis')


def one(job):
    group, path = job
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(OUT, 'per_image', group, f'{stem}.csv')
    if os.path.exists(out[:-4] + '_summary.csv'):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        os.symlink(path, os.path.join(d, os.path.basename(path)))
        r = subprocess.run([PY, '-m', 'mitofi.cli', 'score', d, '--rescue', '-o', out, '--jobs', '1'],
                           cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    open(out[:-4] + '.log', 'w').write(r.stdout + r.stderr)
    return out


def main(workers=11):
    check_tool()
    jobs = [(g, p) for g in ('DMSO', 'FCCP')
            for p in sorted(glob.glob(os.path.join(PUB, 'bbbc053', 'FCCP', g, '*.tif')))]
    with ThreadPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(one, jobs), 1):
            print(f'{k}/{len(jobs)}', flush=True)
    for g in ('DMSO', 'FCCP'):
        parts = [pd.read_csv(f, dtype={'rescue': str})
                 for f in sorted(glob.glob(os.path.join(OUT, 'per_image', g, '*.csv')))
                 if not f.endswith('_summary.csv')]
        pd.concat(parts, ignore_index=True).to_csv(os.path.join(OUT, f'{g}.csv'), index=False)


if __name__ == '__main__':
    sys.exit(main())
