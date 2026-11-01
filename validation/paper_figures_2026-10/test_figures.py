"""Figures 4 and 5: one row per tested comparison -- a representative field and its
mask for each group, and every scored field's index.

Representative field = the scored field whose index is closest to its group's
median (a fixed rule, no hand picking). The mask shown is the one the tool scored:
the recorded rescue is reapplied before segmenting. Indices, n and one-sided P
come from the committed results of each frozen test, unchanged.

    Figure 4  public sets: Allen mitosis, BBBC053, Mito-Hacker, Cape Town
    Figure 5  in-house Olympus archive (A, L_Apr, I, C48, O) and Leica CDKi
              (development data)

    .venv/bin/python validation/paper_figures_2026-10/test_figures.py <out dir>
"""
import os
import sys
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
V = os.path.join(ROOT, 'validation')
PUB = os.path.join(os.path.dirname(ROOT), 'testing_data', 'public')
ARCH = os.path.join(os.path.dirname(ROOT), 'olympus_holdout', 'converted')
LIFS = os.path.join(os.path.dirname(ROOT), 'testing_data', 'test lifs')
ALLEN = 'https://allencell.s3.amazonaws.com/aics/hipsc_single_cell_image_dataset/'
sys.path[:0] = [ROOT]
BLUE, RED, INK, GREY = '#2f5f8a', '#b5533c', '#1b2631', '#8a959e'


# ------------------------------------------------------------------ loaders
def rescued(img, px, res):
    from mitofi.io import subtract_background, subtract_floor
    if res == 'background':
        return subtract_background(img, px)
    if res == 'black point':
        return subtract_floor(img, px, peak=True)[0]
    return img


def tif(path, px=None, ex=None, channel=None):
    from mitofi.io import channel_by_excitation, to_field
    if ex is not None:
        channel = channel_by_excitation(path, ex)
    a, p, _ = to_field(path, channel=channel)
    return np.asarray(a, np.float32), float(px or p), None


def allen_cell(row):
    import tifffile
    from mitofi.io import to_field
    nd = eval(row['name_dict'])
    t = tempfile.mkdtemp()
    raw, seg = os.path.join(t, 'r.ome.tif'), os.path.join(t, 's.ome.tif')
    urllib.request.urlretrieve(ALLEN + row['crop_raw'], raw)
    urllib.request.urlretrieve(ALLEN + row['crop_seg'], seg)
    a = np.asarray(to_field(raw, channel=nd['crop_raw'].index('structure'))[0], np.float32)
    cell = tifffile.imread(seg)[:, nd['crop_seg'].index('membrane_segmentation')].max(0) > 0
    for p in (raw, seg):
        os.remove(p)
    os.rmdir(t)
    return a, 0.108333, cell


def leica_series(row):
    from readlif.reader import LifFile
    from mitofi.io import to_trained_scale
    im = LifFile(os.path.join(LIFS, row['lif'])).get_image(int(row['series']))
    c = int(float(row['mito_channel']))
    a = np.max([np.asarray(im.get_frame(z=z, t=0, c=c)) for z in range(int(row['z']))], 0)
    return np.asarray(to_trained_scale(a, int(row['bits']))[0], np.float32), float(row['px_um']), None


# ------------------------------------------------------------------ comparisons
def g(df, label):
    """Normalise a result table to: key, index, ok, rescue, loader; tag group."""
    return df.assign(group=label)


def allen():
    r = pd.read_csv(os.path.join(V, 'allen_mitosis_1050_2026-10', 'results.csv'))
    inp = pd.read_csv(os.path.join(V, 'allen_mitosis_1050_2026-10', 'inputs.csv')).set_index('CellId')
    r = r[r.group.isin(['interphase', 'M1M2', 'M3', 'M4M5'])]
    out = pd.DataFrame(dict(key=r.CellId, index=r.cell_index, ok=r.cell_pass.astype(str).eq('True'),
                            rescue=r.cell_rescue.fillna(''),
                            group=np.where(r.group == 'interphase', 'Interphase', 'Mitotic')))
    out['load'] = [lambda c=c: allen_cell(inp.loc[c].to_dict() | {}) for c in r.CellId]
    return out


def public(run, groups, path_of, px=None):
    rows = []
    for grp, label in groups:
        d = pd.read_csv(os.path.join(V, 'public_sets_2026-10', 'results', run, f'{grp}.csv'), dtype={'rescue': str})
        for _, x in d.iterrows():
            p = path_of(grp, x.image_id)
            rows.append(dict(key=x.image_id, index=x['index'], ok=str(x.qc_pass) == 'True',
                             rescue=x.rescue if isinstance(x.rescue, str) else '', group=label,
                             load=lambda p=p: tif(p, px=px)))
    return pd.DataFrame(rows)


def archive(cid, labels):
    sys.path.insert(0, os.path.join(V, 'olympus_holdout_2026-10'))
    from run import out_for
    inp = pd.read_csv(os.path.join(V, 'olympus_holdout_2026-10', 'inputs.csv'))
    rows = []
    for side, label in zip(('second', 'first'), labels):
        s = inp[(inp.comparison == cid) & (inp.side == side)]
        for folder, f in s.groupby('folder'):
            res = pd.read_csv(out_for(folder), dtype={'rescue': str})
            res['name'] = res.image_id.str.replace(r'\.ome$', '', regex=True)
            res = res[res.name.isin(f.name)]
            ex = int(f.channel_ex.iloc[0])
            for _, x in res.iterrows():
                p = os.path.join(ARCH, folder, x['name'] + '.ome.tif')
                rows.append(dict(key=x['name'], index=x['index'], ok=str(x.qc_pass) == 'True',
                                 rescue=x.rescue if isinstance(x.rescue, str) else '', group=label,
                                 load=lambda p=p, ex=ex: tif(p, ex=ex)))
    return pd.DataFrame(rows)


def leica_cdki():
    sys.path.insert(0, os.path.join(V, 'all_lif_2026-10'))
    d = pd.read_csv(os.path.join(V, 'scribbles_v2_retrain', 'leica_v5_a_b.csv'))
    d = d[(d.model == 'a') & d.lif.str.contains('CDKi treatment', regex=False)]
    inv = pd.read_csv(os.path.join(V, 'all_lif_2026-10', 'inventory.csv'), dtype=str)
    d = d.assign(series=d.series.astype(str)).merge(inv[['lif', 'series', 'z', 'bits', 'mito_channel']], on=['lif', 'series'])
    lab = {'veh cdki inf': 'Vehicle', 'veh inf for cdki': 'Vehicle', 'cdki inf': 'CDKi'}
    d = d[d.condition.isin(lab)]
    return pd.DataFrame(dict(key=d.lif + '#' + d.series, index=d['index'], ok=d.qc_pass.astype(bool),
                             rescue=d.rescue.fillna('').replace('refused', ''), group=d.condition.map(lab),
                             load=[lambda r=r: leica_series(r) for r in d.to_dict('records')]))


# ------------------------------------------------------------------ drawing
def crop(a, mask, px, cell=None, side_um=32.0):
    """Square window: the cell's box (Allen) or side_um around the densest mask."""
    from scipy.ndimage import uniform_filter
    h, w = mask.shape
    if cell is not None and cell.any():
        ys, xs = np.nonzero(cell)
        cy, cx, s = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2, max(np.ptp(ys), np.ptp(xs)) + 10
    else:
        s = int(min(side_um / px, h, w))
        dens = uniform_filter(mask.astype(np.float32), s)
        cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
    y0 = int(np.clip(cy - s // 2, 0, max(h - s, 0))); x0 = int(np.clip(cx - s // 2, 0, max(w - s, 0)))
    return slice(y0, y0 + s), slice(x0, x0 + s)


def representative(d, group):
    x = d[(d.group == group) & d.ok]
    med = x['index'].median()
    return x.iloc[(x['index'] - med).abs().argmin()]


def pfmt(p):
    """0.0007 -> '0.0007'; 2.1e-16 -> '2 × 10⁻¹⁶'."""
    if p >= 0.001:
        return f'{p:.3f}' if p >= 0.01 else f'{p:.4f}'.rstrip('0')
    m, e = f'{p:.0e}'.split('e')
    sup = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
    return f'{m} × 10{str(int(e)).translate(sup)}'


def draw_row(fig, gs, r, title, d, groups, note=''):
    from mitofi.segment import segment
    import matplotlib.pyplot as plt
    axes = [fig.add_subplot(gs[r, k]) for k in range(4)]
    for k, grp in enumerate(groups):
        rep = representative(d, grp)
        a, px, cell = rep['load']()
        v = rescued(a, px, rep['rescue'])
        m = segment(v)
        if cell is not None:
            m = m & cell
        sl = crop(a, m, px, cell)
        ia, ma = axes[2 * k], axes[2 * k + 1]
        im = a[sl]
        ia.imshow(np.clip(im / max(np.percentile(im, 99.7), 1e-6), 0, 1), cmap='gray')
        ma.imshow(m[sl], cmap='gray')
        if cell is not None:
            ia.contour(cell[sl], [0.5], colors='#e3c84a', linewidths=0.5)
        ia.set_title(grp, fontsize=7.5, color=BLUE if k == 0 else RED, pad=2)
        ma.set_title('mask', fontsize=6.8, pad=2)
        ma.text(0.04, 0.05, f'{rep["index"]:.2f}', transform=ma.transAxes, color='white', fontsize=7,
                weight='bold', va='bottom', bbox=dict(fc='black', ec='none', alpha=0.55, pad=0.8))
        if rep['rescue']:
            ma.text(0.04, 0.95, rep['rescue'], transform=ma.transAxes, color='#f2d16b', fontsize=5.6,
                    va='top', bbox=dict(fc='black', ec='none', alpha=0.55, pad=0.6))
        L = 5.0 / px                                     # 5 um scale bar
        hh, ww = im.shape
        ia.plot([ww * 0.95 - L, ww * 0.95], [hh * 0.92] * 2, color='white', lw=1.6)
        for x in (ia, ma):
            x.set_xticks([]); x.set_yticks([])
    axes[0].set_ylabel(title, fontsize=7.5, rotation=0, ha='right', va='center', labelpad=6)
    pa = fig.add_subplot(gs[r, 5])
    pa.set_ylabel('MitoFI index', fontsize=6.2, labelpad=1)
    rng = np.random.default_rng(0)
    vals = []
    for k, grp in enumerate(groups):
        v = d[(d.group == grp) & d.ok]['index'].values
        vals.append(v)
        pa.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=BLUE if k == 0 else RED, alpha=0.6, lw=0)
        pa.plot([k - 0.25, k + 0.25], [np.median(v)] * 2, color=INK, lw=1.6)
        tot = (d.group == grp).sum()
        pa.text(k, -0.2, f'{len(v)}/{tot}', ha='center', fontsize=5.8, color='#5d6d7e')
    alt = 'greater' if np.median(vals[1]) > np.median(vals[0]) else 'less'
    p = mannwhitneyu(vals[1], vals[0], alternative=alt).pvalue
    pa.set_title('P = ' + pfmt(p), fontsize=6.8, pad=2)
    pa.set_xticks(range(len(groups))); pa.set_xticklabels(groups, fontsize=6)
    pa.set_ylim(-0.27, 1.02); pa.set_yticks([0, 0.5, 1]); pa.tick_params(labelsize=6, length=2)
    pa.set_xlim(-0.6, len(groups) - 0.4)
    for s in ('top', 'right'):
        pa.spines[s].set_visible(False)
    if note:
        pa.text(1.02, 0.5, note, transform=pa.transAxes, fontsize=5.5, rotation=90, va='center', color='#5d6d7e')
    return p


def figure(rows, path, caption_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    fig = plt.figure(figsize=(7.4, 1.45 * len(rows) + 0.3), dpi=300)
    gs = fig.add_gridspec(len(rows), 6, width_ratios=[1, 1, 1, 1, 0.32, 1.25], left=0.15, right=0.965,
                          top=1 - 0.25 / (1.45 * len(rows) + 0.3), bottom=0.04, wspace=0.12, hspace=0.45)
    ps = []
    for r, (letter, title, d, groups, note) in enumerate(rows):
        ps.append(draw_row(fig, gs, r, title, d, groups, note))
        fig.text(0.01, gs[r, 0].get_position(fig).y1 + 0.004, letter, fontsize=11, weight='bold', va='bottom')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    for (letter, title, d, groups, _), p in zip(rows, ps):
        caption_rows.append(f'{letter} {title.replace(chr(10), " ")}: ' + ' vs '.join(
            f'{g_} {d[(d.group == g_) & d.ok]["index"].median():.3f} (n {(d.ok & (d.group == g_)).sum()}/{(d.group == g_).sum()})'
            for g_ in groups) + f'; P = {p:.2g}')


def main(out):
    cap = []
    pub = [
        ('A', 'Allen hiPSC\nTOMM20', allen(), ['Interphase', 'Mitotic'], 'inside cell outline'),
        ('B', 'BBBC053\nTOM20, CAD', public('bbbc053_asis', [('DMSO', 'Vehicle'), ('FCCP', 'FCCP')],
                                           lambda g_, i: os.path.join(PUB, 'bbbc053', 'FCCP', g_, i + '.tif')),
         ['Vehicle', 'FCCP'], ''),
        ('C', 'Mito-Hacker\nmito-YFP', public('mitohacker_cells_0.132', [('253', 'Drp1 +/+'), ('145', 'Drp1 −/−')],
                                             lambda g_, i: os.path.join(PUB, 'mitohacker', 'prepared', f'cells_{g_}', i + '.tif'),
                                             px=0.132), ['Drp1 +/+', 'Drp1 −/−'], 'pixel size assumed'),
        ('D', 'MitoTracker\nLSM780', public('uct', [(f'rep{r}_control', 'Control') for r in 'ACD'] +
                                           [(f'rep{r}_fccp', 'FCCP') for r in 'ACD'],
                                           lambda g_, i: os.path.join(PUB, 'uct', 'prepared', g_, i + '.tif')),
         ['Control', 'FCCP'], '3 repeats pooled'),
    ]
    figure(pub, os.path.join(out, 'Figure_4_public_tests_v1.0.0.png'), cap)
    ins = [
        ('A', 'Huh7, CCCP\n6 Nov 2023', archive('A', ['Mock', 'CCCP']), ['Mock', 'CCCP'], ''),
        ('B', 'Huh7, IFN\n29 Apr 2024', archive('L_Apr', ['Vehicle', 'IFN']), ['Vehicle', 'IFN'], ''),
        ('C', 'Huh7, MAVS KO\nmito-GFP', archive('I', ['WT', 'MAVS KO']), ['WT', 'MAVS KO'], ''),
        ('D', 'Huh7, infection\n48 h', archive('C48', ['Mock', 'Infected']), ['Mock', 'Infected'], ''),
        ('E', 'Huh7, JEV\nmito-GFP', archive('O', ['EV', 'FL-MAVS']), ['EV', 'FL-MAVS'], ''),
        ('F', 'Leica, CDKi\n(development)', leica_cdki(), ['Vehicle', 'CDKi'], 'development data'),
    ]
    figure(ins, os.path.join(out, 'Figure_5_inhouse_tests_v1.0.0.png'), cap)
    open(os.path.join(out, 'Figures_4_5_values.txt'), 'w').write('\n'.join(cap) + '\n')
    print('\n'.join(cap))


if __name__ == '__main__':
    main(sys.argv[1])
