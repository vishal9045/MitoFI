"""Score every converted folder once with the released tool, as PLAN.md fixes.

One command-line run per source folder: `python -m mitofi.cli score <folder>
--channel-ex <nm> --rescue -o results/<folder>.csv`. Folders run 4 at a time.
Refuses to run unless the tool is 1.0.0 with the v6 / v8 models. Resumable
(a folder whose summary CSV exists is skipped).

    .venv/bin/python validation/olympus_holdout_2026-10/run.py <holdout dir>
"""
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(ROOT, '.venv', 'bin', 'python')
# Found at scoring (2026-10-28), before any result was looked at: these files have
# no 570 nm channel at all, so the plan's channel rule cannot be applied, and the
# tool stops the whole folder on them. They are left out and reported as such.
EXCLUDED = {
    '17-03-2023/A549 KO MAVS TF HA Calnexin F2 NO MITO': 'no 570 nm channel (file named NO MITO)',
    '17-03-2023/A549 KO MAVS TF HA Calnexin F3 NO MITO': 'no 570 nm channel (file named NO MITO)',
    '27-11-2023/Huh-7 CoV-2/huh-7 CoV-2 LC3 F17': 'no 570 nm channel (430, 500, 650 nm only)',
}
MODELS = {'pixel_classifier_v6.joblib': '820cd4855402c109', 'index_model_v8.json': '4dfdd1c65e9dfda2'}


def check_tool():
    sys.path.insert(0, ROOT)
    import mitofi
    assert mitofi.__version__ == '1.0.0', mitofi.__version__
    for f, want in MODELS.items():
        got = hashlib.sha256(open(os.path.join(ROOT, 'mitofi', 'models', f), 'rb').read()).hexdigest()
        assert got.startswith(want), (f, got)


def out_for(folder):
    return os.path.join(HERE, 'results', folder.replace('/', '__') + '.csv')


def one(job):
    H, folder, ex, n = job
    out = out_for(folder)
    if os.path.exists(out[:-4] + '_summary.csv'):
        return folder, 'done'
    src = os.path.join(H, 'converted', folder)
    have = len([f for f in os.listdir(src) if f.endswith('.ome.tif')]) if os.path.isdir(src) else 0
    if have != n:
        return folder, f'NOT CONVERTED ({have}/{n})'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    skip = {k.split('/')[-1] + '.ome.tif' for k in EXCLUDED if os.path.dirname(k) == folder}
    if skip:                                   # score the rest of the folder via links
        import tempfile
        tmp = tempfile.mkdtemp(prefix='mitofi_')
        for f in os.listdir(src):
            if f.endswith('.ome.tif') and f not in skip:
                os.symlink(os.path.join(src, f), os.path.join(tmp, f))
        src = tmp
    r = subprocess.run([PY, '-m', 'mitofi.cli', 'score', src, '--suffix', '.ome.tif', '--channel-ex', str(ex),
                        '--rescue', '--jobs', '3', '-o', out], cwd=ROOT, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True)
    open(out[:-4] + '.log', 'w').write(r.stdout + r.stderr)
    return folder, f'exit {r.returncode}'


def main(H, workers=4):
    check_tool()
    d = pd.read_csv(os.path.join(HERE, 'inputs.csv'))
    jobs = [(H, f, int(g.channel_ex.iloc[0]), g.rel.nunique()) for f, g in d.groupby('folder')]
    print(f'MitoFI 1.0.0, models checked; {len(jobs)} folders', flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for k, (f, st) in enumerate(ex.map(one, jobs), 1):
            print(f'[{k}/{len(jobs)}] {st:<24} {f}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
