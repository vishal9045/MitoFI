"""Choose scribble fields by image condition, from every in-house source (D25).

    measure   every candidate field: Olympus development pool, every Leica .lif
              series (px <= 0.2 um), and Allen cells outside the mitosis sample
    select    one field at random per combination of four conditions, plus fields
              for morphologies the grid does not guarantee
    montage   512 px tiles, each contrast-stretched for display, for scribbling in
              Labkit; the original values are kept per tile for training

Conditions, each split at its median over all candidates (dimensionless except
brightness, which is on the classifier's own 12-bit scale -- the thing it reacts to):

    brightness  99.5th percentile
    haze        background under the cell (rolling ball) / brightness
    noise       robust SD of fine-scale residual / brightness
    density     fraction of pixels well above the local background

    .venv/bin/python scripts/condition_montage.py measure
    .venv/bin/python scripts/condition_montage.py select
    .venv/bin/python scripts/condition_montage.py dim [N]

dim (2026-10-27, for the next version): N more fields (default 12) for the condition the
black point exists for -- refused by pixel classifier v6 as is AND after
background removal. Drawn at random from every in-house candidate in that
condition, pooled, never by microscope. Fields already scribbled and the 238
fresh Allen test cells are left out. Written as field_33... next to the others,
to be added to the same ilastik project.
"""
import csv
import glob
import os
import sys
import urllib.request
import warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, 'data', 'scribbles_v2')
LIFS = os.path.join(os.path.dirname(ROOT), 'testing_data', 'test lifs')
ALLEN = 'https://allencell.s3.amazonaws.com/aics/hipsc_single_cell_image_dataset/'
SEED = 20261025
TILE = 512


def metrics(a, px):
    from scipy.ndimage import gaussian_filter
    from mitofi.io import estimate_background
    from skimage.filters import threshold_otsu
    a = np.asarray(a, np.float32)
    bright = float(np.percentile(a, 99.5)) or 1.0
    bg = estimate_background(a, px, fast=True)
    sm = gaussian_filter(a, 0.2 / px)
    cell = sm > threshold_otsu(sm)
    haze = float(np.median(bg[cell])) / bright if cell.any() else 0.0
    # inside the cell: processed images clip their background to zero, which
    # makes a whole-field noise estimate read 0 whatever the cells look like
    res = (a - gaussian_filter(a, 1.0))[cell] if cell.any() else a.ravel()
    noise = float(1.4826 * np.median(np.abs(res - np.median(res)))) / bright
    density = float(((a - bg) > 0.25 * (bright - np.median(bg))).mean())
    return dict(brightness=round(bright, 1), haze=round(haze, 4), noise=round(noise, 4),
                density=round(density, 4))


# ---------------------------------------------------------------- readers
def read(row):
    """2D field on the classifier's scale, and its pixel size."""
    from mitofi.io import to_field, to_trained_scale
    if row['source'] == 'olympus':
        img, px, _ = to_field(os.path.join(ROOT, 'data', 'images', row['id'] + '.ome.tif'))
        return np.asarray(img, np.float32), float(px)
    if row['source'] == 'leica':
        from readlif.reader import LifFile
        im = LifFile(os.path.join(LIFS, row['lif'])).get_image(int(row['series']))
        a = np.max([np.asarray(im.get_frame(z=z, t=0, c=int(float(row['channel']))))
                    for z in range(int(row['z']))], 0)
        return np.asarray(to_trained_scale(a, int(row['bits']))[0], np.float32), float(row['px_um'])
    img, _, _ = to_field(os.path.join(OUT, 'allen_src', row['id'] + '.ome.tif'), channel=2)
    return np.asarray(img, np.float32), 0.108333


def measure_one(row):
    try:
        a, px = read(row)
        row.update(metrics(a, px), px_um=px, h=a.shape[0], w=a.shape[1])
    except Exception as e:
        row['error'] = repr(e)
    return row


def candidates():
    import pandas as pd
    rows = []
    f = pd.read_csv(os.path.join(ROOT, 'data', 'features_v5.csv'))
    lab = pd.read_csv(os.path.join(ROOT, 'data', 'labels_consensus.csv'))[['image_id', 'blinded_call']]
    f = f.merge(lab, on='image_id', how='left')
    for iid, call in zip(f.image_id, f.blinded_call):
        rows.append(dict(source='olympus', id=iid, morphology=str(call)))
    inv = pd.read_csv(os.path.join(ROOT, 'validation', 'all_lif_2026-10', 'inventory.csv'), dtype=str)
    res = pd.read_csv(os.path.join(ROOT, 'validation', 'all_lif_2026-10', 'results.csv'), dtype=str)
    inv = inv.merge(res[['lif', 'series', 'condition']], on=['lif', 'series'], how='left')
    for _, r in inv[inv.px_um.astype(float) <= 0.2].iterrows():
        rows.append(dict(source='leica', id=f"{os.path.basename(r.lif)}#s{r.series}", lif=r.lif,
                         series=r.series, channel=r.mito_channel, z=r.z, bits=r.bits,
                         px_um=r.px_um, morphology=str(r.condition)))
    for p in sorted(glob.glob(os.path.join(OUT, 'allen_src', '*.ome.tif'))):
        cid, stage = os.path.basename(p).split('.')[0].split('_')
        rows.append(dict(source='allen', id=f'{cid}_{stage}', morphology=stage))
    return rows


def allen_meta():
    """TOMM20 rows of the Allen metadata.csv (1.7 GB), streamed once and kept."""
    import pandas as pd
    out = os.path.join(OUT, 'allen_tomm20_metadata.csv')
    if not os.path.exists(out):
        cols = {'CellId', 'FOVId', 'structure_name', 'crop_raw', 'crop_seg', 'name_dict',
                'scale_micron', 'cell_stage', 'PlateId'}
        parts = [ch[ch.structure_name == 'TOMM20'] for ch in
                 pd.read_csv(ALLEN + 'metadata.csv', usecols=lambda c: c in cols,
                             chunksize=20000, low_memory=False)]
        pd.concat(parts).to_csv(out, index=False)
    return out


def fetch_allen(n_each=30):
    """Allen cells never in the mitosis sample: n interphase, n mitotic (M3, M4M5)."""
    import pandas as pd
    os.makedirs(os.path.join(OUT, 'allen_src'), exist_ok=True)
    meta = pd.read_csv(allen_meta())
    used = set(pd.read_csv(os.path.join(ROOT, 'validation', 'allen_mitosis_2026-10',
                                        'inputs.csv')).CellId)
    meta = meta[~meta.CellId.isin(used)]
    rng = np.random.default_rng(SEED)
    pick = [meta[meta.cell_stage == 'M0'], meta[meta.cell_stage.isin(['M3', 'M4M5'])]]
    pick = pd.concat([p.loc[rng.choice(p.index, n_each, replace=False)] for p in pick])
    todo = [(r.CellId, r.cell_stage, r.crop_raw) for r in pick.itertuples()]
    with Pool(8) as pool:
        pool.map(_fetch, todo)


def _fetch(t):
    cid, stage, path = t
    out = os.path.join(OUT, 'allen_src', f'{cid}_{stage}.ome.tif')
    if not os.path.exists(out):
        urllib.request.urlretrieve(ALLEN + path, out)


# ---------------------------------------------------------------- selection
def select():
    import pandas as pd
    d = pd.read_csv(os.path.join(OUT, 'pool.csv'))
    d = d[d.error.isna()] if 'error' in d else d
    d = d[(d.brightness >= 100) & (d.density > 0.005)]     # blank or near-empty series
    for k in ('haze', 'noise', 'density'):
        d[k + '_hi'] = d[k] > d[k].median()
    d['brightness_hi'] = np.log(d.brightness) > np.log(d.brightness).median()
    axes = ['brightness_hi', 'haze_hi', 'noise_hi', 'density_hi']
    d['cell'] = d[axes].astype(int).astype(str).agg(''.join, axis=1)
    rng = np.random.default_rng(SEED)
    chosen = []
    for cell, g in sorted(d.groupby('cell')):
        src = rng.choice(sorted(g.source.unique()))
        r = g[g.source == src].sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
        chosen.append(dict(r, why=f'grid {cell} (bright,haze,noise,dense)'))
    taken = {c['id'] for c in chosen}
    # condition names live under validation/ (scripts/check_blind.py keeps them
    # out of everything else); the Leica morphology groups are read from there
    groups = pd.read_csv(os.path.join(ROOT, 'validation', 'all_lif_2026-10',
                                      'morphology_groups.csv'))
    extras = [('very dim', d.brightness < 1000, 3),
              ('haziest', d.haze >= d.haze.quantile(0.95), 2)]
    extras += [(g.morphology, d.morphology.str.contains(g.pattern, na=False), int(g.n))
               for g in groups.itertuples()]
    extras += [('mitotic', d.morphology.isin(['M3', 'M4M5']), 2),
               ('elongated', d.morphology.eq('elongated'), 1)]
    for why, mask, n in extras:
        g = d[mask & ~d.id.isin(taken)]
        for _, r in g.sample(min(n, len(g)), random_state=int(rng.integers(1 << 31))).iterrows():
            chosen.append(dict(r, why=why)); taken.add(r.id)
    sel = pd.DataFrame(chosen)
    sel.to_csv(os.path.join(OUT, 'selection.csv'), index=False)
    print(f'{len(sel)} fields:', sel.source.value_counts().to_dict())
    print(sel[['source', 'id', 'why', 'brightness', 'haze', 'noise', 'density']].to_string())
    montage(sel)


def window(a, bg_frac=0.25):
    """Top-left of the TILE window holding the most bright structure."""
    h, w = a.shape
    if h <= TILE and w <= TILE:
        return 0, 0
    m = a > np.percentile(a, 90)
    best, at = -1, (0, 0)
    for y in range(0, max(h - TILE, 0) + 1, 128):
        for x in range(0, max(w - TILE, 0) + 1, 128):
            s = m[y:y + TILE, x:x + TILE].sum()
            if s > best:
                best, at = s, (y, x)
    return at


def montage(sel, cols=6):
    import tifffile
    os.makedirs(os.path.join(OUT, 'tiles'), exist_ok=True)
    n = len(sel); rows = -(-n // cols)
    disp = np.zeros((rows * TILE, cols * TILE), np.uint8)
    tiles = []
    for k, (_, r) in enumerate(sel.iterrows()):
        a, px = read(r.to_dict())
        y0, x0 = window(a)
        t = a[y0:y0 + TILE, x0:x0 + TILE]
        raw = np.zeros((TILE, TILE), np.float32); raw[:t.shape[0], :t.shape[1]] = t
        lo, hi = np.percentile(t, [0.5, 99.8])
        v = (np.clip((t - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        R, C = divmod(k, cols)
        disp[R * TILE:R * TILE + t.shape[0], C * TILE:C * TILE + t.shape[1]] = v
        tifffile.imwrite(os.path.join(OUT, 'tiles', f'tile_{k:02d}.tif'), raw)
        tiles.append(dict(tile_index=k, col=C, row=R, x=C * TILE, y=R * TILE,
                          width=t.shape[1], height=t.shape[0], source=r.source, id=r.id,
                          crop_y=y0, crop_x=x0, px_um=px, why=r.why))
    tifffile.imwrite(os.path.join(OUT, 'montage_v2.tif'), disp)
    with open(os.path.join(OUT, 'montage_v2_tiles.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(tiles[0])); w.writeheader(); w.writerows(tiles)
    _key(disp, tiles)
    print(f'montage {disp.shape} -> {OUT}')


def _key(disp, tiles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(disp.shape[1] / 200, disp.shape[0] / 200), dpi=100)
    plt.imshow(disp, cmap='gray'); plt.axis('off')
    for t in tiles:
        plt.text(t['x'] + 8, t['y'] + 30, f"{t['tile_index']}: {t['source']}\n{t['why']}",
                 color='yellow', fontsize=7, va='top')
    plt.savefig(os.path.join(OUT, 'montage_v2_key.png'), bbox_inches='tight'); plt.close()


# ---------------------------------------------------------------- dim cytoplasm
def needs_black_point(row):
    """(id, True if v6 refuses it as is and after background removal)."""
    from mitofi.io import subtract_background
    from mitofi.score import score_mask
    from mitofi.segment import segment
    a, px = read(row)
    if px > 0.2:
        return row['id'], False
    for img in (a, subtract_background(a, px)):
        if score_mask(segment(img, jobs=1), px, img=img)['qc_pass']:
            return row['id'], False
    return row['id'], True


def dim(n=12):
    import pandas as pd
    import tifffile
    d = pd.read_csv(os.path.join(OUT, 'pool.csv'))
    d = d[d.error.isna()] if 'error' in d else d
    done = set(pd.read_csv(os.path.join(OUT, 'fields.csv')).id)
    d = d[~d.id.isin(done) & (d.brightness >= 100) & (d.density > 0.005)]
    d = d[(d.source != 'olympus') | d.morphology.isin(['elongated', 'normal', 'fragmented'])]
    # Leica: the v6 run on every series already says which ones background leaves refused
    L = pd.read_csv(os.path.join(ROOT, 'validation', 'scribbles_v2_retrain', 'leica_v5_a_b.csv'))
    L = L[(L.model == 'a') & (L.rescue == 'refused')]
    leica_ids = {f'{os.path.basename(a)}#s{b}' for a, b in zip(L.lif, L.series)}
    need = set(d[(d.source == 'leica') & d.id.isin(leica_ids)].id)
    other = [r for r in d[d.source != 'leica'].to_dict('records')]
    with Pool(8) as pool:
        need |= {i for i, ok in pool.map(needs_black_point, other) if ok}
    g = d[d.id.isin(need)].sort_values('id')
    print(f'{len(g)} candidates refused after background:', g.source.value_counts().to_dict())
    pick = g.sample(min(n, len(g)), random_state=20261027)
    start = 1 + max(int(f.split('_')[1]) for f in os.listdir(os.path.join(OUT, 'fields'))
                    if f.startswith('field_'))
    rows = []
    for k, (_, r) in enumerate(pick.iterrows(), start):
        a, px = read(r.to_dict())
        name = f'field_{k:02d}_{r.source}.tif'
        tifffile.imwrite(os.path.join(OUT, 'fields', name), np.clip(a, 0, 65535).astype(np.uint16),
                         imagej=True, resolution=(1 / px, 1 / px), metadata={'unit': 'um'})
        rows.append(dict(field=name, source=r.source, id=r.id, px_um=px, shape=f'{a.shape[0]}x{a.shape[1]}',
                         why='dim cytoplasm: refused as is and after background (v6)'))
    with open(os.path.join(OUT, 'fields.csv'), 'a', newline='') as fh:
        csv.DictWriter(fh, fieldnames=list(rows[0])).writerows(rows)
    print(pd.DataFrame(rows)[['field', 'source', 'shape']].to_string(index=False))


def measure():
    rows = candidates()
    with Pool(8) as pool:
        out = pool.map(measure_one, rows)
    keys = sorted({k for r in out for k in r})
    with open(os.path.join(OUT, 'pool.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(out)
    print(len(out), 'candidates;', sum('error' in r for r in out), 'errors')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == 'dim':
        dim(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    else:
        {'fetch': fetch_allen, 'measure': measure, 'select': select}[sys.argv[1]]()
