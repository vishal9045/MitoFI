"""Score the three public sets once with the released tool, as PLAN.md fixes.

Each group folder goes through the command line as a user would run it:
`python -m mitofi.cli score <folder> --rescue -o results/<run>/<group>.csv`.
Refuses to run unless the tool is 1.0.0 with the v6 / v8 models. Skips runs
whose CSV already exists (resumable).

    .venv/bin/python validation/public_sets_2026-10/run.py
"""
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PUB = os.path.join(os.path.dirname(ROOT), 'testing_data', 'public')
PY = os.path.join(ROOT, '.venv', 'bin', 'python')
MODELS = {'pixel_classifier_v6.joblib': '820cd4855402c109', 'index_model_v8.json': '4dfdd1c65e9dfda2'}

UCT = [f'rep{r}_{g}' for r in 'ACD' for g in ('control', 'fccp', 'pa24', 'pa72')] + ['repD_CT_unlabelled']
RUNS = (  # run name, folder per group, pixel size given at scoring (None: from the file)
    [('bbbc053_asis', g, os.path.join(PUB, 'bbbc053', 'FCCP', g), None) for g in ('DMSO', 'FCCP')]
    + [('bbbc053_down4', g, os.path.join(PUB, 'bbbc053', 'prepared', f'down4_{g}'), None) for g in ('DMSO', 'FCCP')]
    + [(f'mitohacker_cells_{px}', ln, os.path.join(PUB, 'mitohacker', 'prepared', f'cells_{ln}'), px)
       for px in (0.132, 0.088, 0.198) for ln in ('145', '143', '253')]
    + [(f'mitohacker_fields_{px}', ln, os.path.join(PUB, 'mitohacker', 'prepared', f'fields_{ln}'), px)
       for px in (0.066, 0.044, 0.099) for ln in ('145', '143', '253')]
    + [('uct', g, os.path.join(PUB, 'uct', 'prepared', g), None) for g in UCT])


def check_tool():
    sys.path.insert(0, ROOT)
    import mitofi
    assert mitofi.__version__ == '1.0.0', mitofi.__version__
    for f, want in MODELS.items():
        got = hashlib.sha256(open(os.path.join(ROOT, 'mitofi', 'models', f), 'rb').read()).hexdigest()
        assert got.startswith(want), (f, got)


def main():
    check_tool()
    print('MitoFI 1.0.0, models checked', flush=True)
    for k, (run, group, folder, px) in enumerate(RUNS, 1):
        out = os.path.join(HERE, 'results', run, f'{group}.csv')
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cmd = [PY, '-m', 'mitofi.cli', 'score', folder, '--rescue', '-o', out]
        if px:
            cmd += ['--px-um', str(px)]
        print(f'[{k}/{len(RUNS)}] {run} {group}', flush=True)
        r = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        open(out[:-4] + '.log', 'w').write(r.stdout + r.stderr)
        if r.returncode:
            print(f'   exit {r.returncode}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}', flush=True)


if __name__ == '__main__':
    main()
