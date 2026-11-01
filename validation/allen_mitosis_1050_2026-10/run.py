"""Score the October Allen sample once with MitoFI 1.0.0, as PLAN.md fixes.

Reuses ../allen_mitosis_2026-10/run.py unchanged (same reader, rescue chain and
two analyses); only the folder it reads inputs from and writes results to is this
one. Refuses to run unless the shipped models are the 1.0.0 ones. Resumable.

    .venv/bin/python validation/allen_mitosis_1050_2026-10/run.py [--jobs N]
"""
import hashlib
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODELS = {'pixel_classifier_v6.joblib': '820cd4855402c109', 'index_model_v8.json': '4dfdd1c65e9dfda2'}


def october(name):
    """Load ../allen_mitosis_2026-10/<name>.py, pointed at this folder."""
    path = os.path.join(ROOT, 'validation', 'allen_mitosis_2026-10', f'{name}.py')
    spec = importlib.util.spec_from_file_location(f'october_{name}', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m              # so worker processes can find its functions
    spec.loader.exec_module(m)
    m.HERE, m.OUT = HERE, os.path.join(HERE, 'results.csv')
    return m


def check_tool():
    sys.path.insert(0, ROOT)
    import mitofi
    import importlib
    score, segment = (importlib.import_module(f'mitofi.{m}') for m in ('score', 'segment'))
    assert mitofi.__version__ == '1.0.0', mitofi.__version__
    for f, want in MODELS.items():
        got = hashlib.sha256(open(os.path.join(ROOT, 'mitofi', 'models', f), 'rb').read()).hexdigest()
        assert got.startswith(want), (f, got)
    assert os.path.basename(segment.PIXEL_MODEL) == 'pixel_classifier_v6.joblib'
    assert os.path.basename(score.INDEX_MODEL) == 'index_model_v8.json'
    assert (score.MAX_HAZE, score.MIN_LARGE_SNR) == (0.06, 1.16)
    return mitofi.__version__


if __name__ == '__main__':
    print('MitoFI', check_tool(), 'models checked', flush=True)
    october('run').main(int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 8)
