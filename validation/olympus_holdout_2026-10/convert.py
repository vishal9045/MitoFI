"""Convert .oir to OME-TIFF quickly: copy each file from the USB disk to local
disk first (one sequential read), convert there with OirToOme.java, delete the
copy. Reading .oir straight from the USB disk through WSL took ~60 s per file
(many small reads); copy + convert takes ~2-8 s.

    .venv/bin/python validation/olympus_holdout_2026-10/convert.py <jobs.tsv> [workers]

jobs.tsv: "<input .oir><TAB><output .ome.tif>" per line. Existing outputs are skipped.
"""
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
CP = '/mnt/d/Fiji/jars/*:/mnt/d/Fiji/jars/bio-formats/*'


def worker(q, done, lock):
    tmp = tempfile.mkdtemp(prefix='oir_')
    jvm = subprocess.Popen(['java', '-cp', CP, os.path.join(HERE, 'OirToOme.java')], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    while True:
        job = q.get()
        if job is None:
            break
        src, out = job
        local = os.path.join(tmp, 'in.oir')
        try:
            shutil.copyfile(src, local)
            jvm.stdin.write(f'{local}\t{out}\n')
            jvm.stdin.flush()
            msg = jvm.stdout.readline().strip()
            msg = f'ok {out}' if msg.startswith('ok') else f'ERROR {src} {msg}'
        except Exception as e:
            msg = f'ERROR {src} {e!r}'
        finally:
            if os.path.exists(local):
                os.remove(local)
        with lock:
            done.append(msg)
            print(f'{len(done)} {msg[:160]}', flush=True)
    jvm.stdin.close()
    jvm.wait()
    shutil.rmtree(tmp, ignore_errors=True)


def main(jobs, n=4):
    todo = [l.rstrip('\n').split('\t') for l in open(jobs) if l.strip()]
    todo = [t for t in todo if not os.path.exists(t[1])]
    print(len(todo), 'to convert', flush=True)
    q, done, lock = queue.Queue(), [], threading.Lock()
    th = [threading.Thread(target=worker, args=(q, done, lock)) for _ in range(n)]
    for t in th:
        t.start()
    for t in todo:
        q.put(t)
    for _ in th:
        q.put(None)
    for t in th:
        t.join()
    print('errors:', sum(m.startswith('ERROR') for m in done))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
