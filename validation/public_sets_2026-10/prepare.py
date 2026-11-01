"""Put the three public sets into files MitoFI reads, without scoring anything.

Only format changes a user would make before running the tool:

    bbbc053     as published (single-channel 2048^2 TIFF, 0.0201 um/px in the file);
                plus a 4x block-mean copy at 0.0804 um/px (secondary, PLAN.md)
    mitohacker  RGB composites -> the green (mito-YFP) channel on its own, 8-bit.
                No pixel size in the files; it is given at scoring (PLAN.md).
    uct         Zeiss .czi -> the channel excited at 561 nm (MitoTracker), Z kept,
                pixel size and z-step written into the TIFF

Writes prepared images under testing_data/public/<set>/prepared/<group>/ and
the list of every file, its group and pixel size to inputs.csv here.

    .venv/bin/python validation/public_sets_2026-10/prepare.py
"""
import csv
import glob
import os
import re

import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), 'testing_data', 'public')
rows = []


def put(dataset, group, src, a, px, z_um=None, note=''):
    out = os.path.join(PUB, dataset, 'prepared', group, os.path.splitext(os.path.basename(src))[0] + '.tif')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    kw = {}
    if px:
        kw = dict(resolution=(1 / px, 1 / px), metadata={'unit': 'um', **({'spacing': z_um} if z_um else {}),
                                                         'axes': 'ZYX' if a.ndim == 3 else 'YX'})
    tifffile.imwrite(out, a, imagej=True, **kw)
    rows.append(dict(dataset=dataset, group=group, source=os.path.relpath(src, PUB),
                     prepared=os.path.relpath(out, PUB), px_um=px or '', shape='x'.join(map(str, a.shape)),
                     note=note))


def bbbc053():
    for f in sorted(glob.glob(os.path.join(PUB, 'bbbc053', 'FCCP', '*', '*.tif'))):
        g = os.path.basename(os.path.dirname(f))
        t = tifffile.TiffFile(f)
        a = t.pages[0].asarray()
        px = t.pages[0].tags['XResolution'].value[1] / t.pages[0].tags['XResolution'].value[0]
        rows.append(dict(dataset='bbbc053', group=g, source=os.path.relpath(f, PUB), prepared=os.path.relpath(f, PUB),
                         px_um=round(px, 5), shape='x'.join(map(str, a.shape)), note='as published'))
        h, w = (s // 4 * 4 for s in a.shape)
        d = a[:h, :w].reshape(h // 4, 4, w // 4, 4).astype(np.float64).mean((1, 3))
        put('bbbc053', f'down4_{g}', f, np.round(d).astype(np.uint16), round(px * 4, 5), note='4x block mean')


def mitohacker():
    base = os.path.join(PUB, 'mitohacker', 'Mito-Hacker-master', 'Sample_images')
    for line in ('145', '143', '253'):
        for f in sorted(glob.glob(os.path.join(base, 'Multi-Cell_images', line, '*.tif'))):
            g = tifffile.imread(f)[..., 1]
            g = (g // 256).astype(np.uint8) if g.dtype == np.uint16 else g        # 8-bit stored as x256
            put('mitohacker', f'fields_{line}', f, g, None, note='green channel; field')
        for f in sorted(glob.glob(os.path.join(base, 'Single-Cell_images', line, 'Individual_cells', '*.tif'))):
            put('mitohacker', f'cells_{line}', f, tifffile.imread(f)[..., 1], None,
                note='green channel; Mito-Hacker crop, field shrunk 2x')


UCT_GROUPS = [(r'^AC_', 'repA_control'), (r'^AF25_', 'repA_fccp'), (r'^AP24_', 'repA_pa24'),
              (r'^AP72_', 'repA_pa72'), (r'^C1_CC_', 'repC_control'), (r'^C1_CF_', 'repC_fccp'),
              (r'^C2_CP24_', 'repC_pa24'), (r'^C2_CP72_', 'repC_pa72'),
              (r' C [a-z]\.czi$', 'repD_control'), (r' CT [a-z]\.czi$', 'repD_CT_unlabelled'),
              (r' F [a-z]\.czi$', 'repD_fccp'), (r' P24 [a-z]\.czi$', 'repD_pa24'),
              (r' P72 [a-z]\.czi$', 'repD_pa72')]


def uct():
    import czifile
    for f in sorted(glob.glob(os.path.join(PUB, 'uct_figshare', '*', '*', '*.czi'))):
        name = os.path.basename(f)
        if ' slow' in name:                      # the same field scanned again, slowly
            rows.append(dict(dataset='uct', group='excluded', source=os.path.relpath(f, PUB), prepared='',
                             px_um='', shape='', note='second scan of a field already in the set'))
            continue
        group = [g for pat, g in UCT_GROUPS if re.search(pat, name)]
        assert len(group) == 1, (name, group)
        if os.path.getsize(f) == 0:              # empty in the published zip itself
            rows.append(dict(dataset='uct', group=group[0], source=os.path.relpath(f, PUB), prepared='',
                             px_um='', shape='', note='unreadable: 0 bytes in the deposited zip'))
            continue
        c = czifile.CziFile(f)
        im = c.scenes[0]
        chans = list(im.channels.items())
        k = [i for i, (_, v) in enumerate(chans) if round(float(v.get('ExcitationWavelength') or 0)) == 561]
        assert len(k) == 1 and im.dims[0] == 'C', (name, im.dims, chans)
        a = im.asarray()[k[0]]
        assert im.dims[1:] == ('Z', 'Y', 'X'), im.dims
        sx = float(c.metadata().split('<Distance Id="X">')[1].split('<Value>')[1].split('<')[0]) * 1e6
        sz = float(c.metadata().split('<Distance Id="Z">')[1].split('<Value>')[1].split('<')[0]) * 1e6
        put('uct', group[0], f, a, round(sx, 5), round(sz, 4), note=f'channel {k[0]} (561 nm), {chans[k[0]][1].get("Fluor")}')


if __name__ == '__main__':
    bbbc053()
    mitohacker()
    uct()
    with open(os.path.join(HERE, 'inputs.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import collections
    print(collections.Counter((r['dataset'], r['group']) for r in rows))
