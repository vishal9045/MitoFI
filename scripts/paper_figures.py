"""Manuscript Figures 1-3 for MitoFI 1.0.0 (pixel classifier v6, index model v8).

Rebuilds the three figures of the draft from repository data, same layout and
pixel size as the originals so they can replace them in the document:

    Figure 1  pipeline with a worked example (pilot field img_017653, the field
              the original figure used; found by image matching, r = 0.94)
    Figure 2  session-held-out confusion matrix and index by majority class
    Figure 3  masks of one field from Labkit, the earlier and the released
              classifier; quality-gate rejection by gate and classifier version

    .venv/bin/python scripts/paper_figures.py <output folder>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import tifffile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'scripts')]
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

D = os.path.join(ROOT, 'data')
BLUE, RED, GREY, TEAL, INK = '#2f5f8a', '#b5533c', '#8a959e', '#1f6f68', '#1b2631'
LBLUE, LRED, LTEAL, LGREY = '#eaf1f8', '#fbeee9', '#eef5f3', '#f3f5f7'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})


def disp(a, hi=99.7):
    a = np.asarray(a, np.float32)
    return np.clip(a / max(np.percentile(a, hi), 1e-6), 0, 1)


def scalebar(ax, shape, px, um=10, color='white'):
    h, w = shape
    L = um / px
    ax.add_patch(Rectangle((w * 0.93 - L, h * 0.93), L, h * 0.012, color=color))
    ax.text(w * 0.93 - L / 2, h * 0.9, f'{um} µm', color=color, ha='center', va='bottom', fontsize=9)


# --------------------------------------------------------------------- Figure 1
def figure1(out):
    from mitofi import score
    from mitofi.features import features, haze_fraction, local_contrast
    from mitofi.io import to_field
    from mitofi.segment import feature_stack, segment
    iid = 'img_017653'
    img, px, _ = to_field(os.path.join(D, 'images', f'{iid}.ome.tif'))
    img = np.asarray(img, np.float32)
    px = float(px)
    mask = segment(img)
    q = score.score_mask(mask, px, img=img)
    M = score.load()
    f = features(mask, img, px)
    p = score.predict_proba(M, [f[k] for k in M['features']])
    P = dict(zip(M['classes'], p))
    stack = [pl for _, pl in feature_stack(img, sigmas=[1, 2, 4, 8])]

    fig = plt.figure(figsize=(4252 / 300, 4371 / 300), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 103); ax.axis('off')

    def box(x, y, w, h, fc, ec, lw=1.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.8',
                                    fc=fc, ec=ec, lw=lw))

    def arrow(x0, y0, x1, y1, c=BLUE):
        ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=c, lw=1.6))

    ax.text(4, 99, 'A', fontsize=22, weight='bold', va='center')
    ax.text(8, 99, 'Signal identification', fontsize=20, weight='bold', color=BLUE, va='center')
    ax.text(4, 93.5, 'TIFF / OME-TIFF', fontsize=13, weight='bold')
    ax.text(96, 93.5, 'Binary foreground mask', fontsize=13, weight='bold', ha='right')
    ia = fig.add_axes([0.04, 0.70, 0.18, 0.175]); ia.imshow(disp(img), cmap='gray'); ia.set_xticks([]); ia.set_yticks([])
    scalebar(ia, img.shape, px)
    for k, ch in enumerate([0, 7, 14, 21]):
        sa = fig.add_axes([0.27 + 0.015 * k, 0.725 - 0.016 * k, 0.07, 0.07])
        sa.imshow(disp(stack[ch]), cmap='gray')
        sa.set_xticks([]); sa.set_yticks([])
        for s in sa.spines.values():
            s.set_edgecolor(BLUE)
    ax.text(32.5, 66.5, '25 feature planes', color=BLUE, fontsize=13, weight='bold', ha='center')
    arrow(23, 81.5, 26, 81.5); arrow(39, 81.5, 43, 81.5)
    box(44, 74, 22, 15, LGREY, BLUE)
    ax.text(55, 85, 'Random forest', color=BLUE, fontsize=15, weight='bold', ha='center')
    ax.text(55, 81, '200 trees', fontsize=14, ha='center')
    ax.text(55, 77, 'Pixel classifier v6', color='#5d6d7e', fontsize=11.5, ha='center')
    ax.text(55, 69.8, 'Scribbles: Labkit montage + ilastik\nfields from three microscopes',
            color='#5d6d7e', fontsize=10.5, ha='center', va='top')
    arrow(67, 81.5, 76, 81.5)
    ma = fig.add_axes([0.78, 0.70, 0.18, 0.175]); ma.imshow(mask, cmap='gray'); ma.set_xticks([]); ma.set_yticks([])
    scalebar(ma, mask.shape, px)
    ax.text(87, 66.5, 'Inspect the mask', color=BLUE, fontsize=12.5, ha='center')
    ax.text(4, 66.5, 'Select channel (position or excitation\nwavelength); project Z by maximum intensity',
            fontsize=11.5, va='top')
    ax.text(4, 61.3, 'Read or supply spatial calibration', color='#5d6d7e', fontsize=11.5)
    ax.plot([97, 98.5, 98.5, 1.6, 1.6], [81.5, 81.5, 58.5, 58.5, 13.5], color='#5d6d7e', lw=1.2)
    arrow(1.6, 39, 3.9, 39, '#5d6d7e'); arrow(1.6, 13.5, 3.9, 13.5, '#5d6d7e')

    ax.text(4, 54, 'B', fontsize=22, weight='bold', va='center')
    ax.text(8, 54, 'Morphological scoring', fontsize=20, weight='bold', color=TEAL, va='center')
    box(4, 28.4, 29, 21.1, LTEAL, TEAL)
    ax.text(5.5, 47, 'Nine mask descriptors', color=TEAL, fontsize=13.5, weight='bold')
    names = ['90th-percentile aspect ratio', '90th-percentile eccentricity', 'Area-weighted form factor',
             'Median aspect ratio', 'Median eccentricity', 'Hole-area fraction', 'Area-weighted aspect ratio',
             'Largest-object area fraction', 'Mean skeleton length per object']
    for k, n in enumerate(names):
        ax.text(5.5, 43.8 - k * 1.58, n, fontsize=11.5)
    ax.text(5.5, 29.3, 'Inspect the descriptor vector', color=TEAL, fontsize=11)
    ax.text(4, 26.6, '8-connected objects; minimum area 0.05 µm²', color='#5d6d7e', fontsize=10.5)
    arrow(34, 44, 37.5, 44, TEAL)
    box(38.5, 40, 26, 9.5, LTEAL, TEAL)
    ax.text(51.5, 47, 'Standardize descriptors', color=TEAL, fontsize=13, weight='bold', ha='center')
    ax.text(51.5, 44.3, 'Multinomial logistic regression', fontsize=12, ha='center')
    ax.text(51.5, 41.6, 'Fixed v8 coefficients', color='#5d6d7e', fontsize=11.5, ha='center')
    arrow(51.5, 39.5, 51.5, 37.3, TEAL)
    ax.text(38.5, 35.4, 'Example class probabilities', color=TEAL, fontsize=13, weight='bold')
    for k, (c, lab, col) in enumerate([('elongated', 'Elongated', BLUE), ('normal', 'Normal', GREY),
                                       ('fragmented', 'Fragmented', RED)]):
        y = 32.6 - k * 3.2
        ax.text(38.5, y, lab, fontsize=12); ax.text(64.5, y, f'{P[c]:.3f}', fontsize=12, ha='right')
        ax.add_patch(Rectangle((38.5, y - 1.5), 26, 0.7, color='#e6e9ec'))
        ax.add_patch(Rectangle((38.5, y - 1.5), 26 * P[c], 0.7, color=col))
    arrow(65.5, 29, 70.5, 29, TEAL)
    box(71.5, 29.5, 25, 20, 'white', TEAL)
    ax.text(84, 47, 'Candidate index', color=TEAL, fontsize=14, weight='bold', ha='center')
    ax.text(84, 42.8, r'$I = 0.5\,P_N + P_F$', fontsize=17, ha='center')
    ax.text(84, 39.6, 'N: normal; F: fragmented', color='#5d6d7e', fontsize=11, ha='center')
    grad = np.linspace(0, 1, 256)[None]
    ga = fig.add_axes([0.725, 0.345, 0.205, 0.016])
    ga.imshow(grad, aspect='auto', cmap=matplotlib.colors.LinearSegmentedColormap.from_list('x', [BLUE, RED]))
    ga.axvline(q['index_unvalidated'] * 255, color=INK, lw=2); ga.set_xticks([]); ga.set_yticks([])
    ax.text(73, 33.3, '0', fontsize=12); ax.text(95, 33.3, '1', fontsize=12, ha='right')
    ax.text(73, 31.4, 'Elongated', color=BLUE, fontsize=11); ax.text(95, 31.4, 'Fragmented', color=RED, fontsize=11, ha='right')
    arrow(84, 29, 84, 23.7, TEAL)
    box(4, 4.5, 60, 18, LGREY, '#5d6d7e')
    ax.text(5.5, 20.0, 'QC gate: reject if any rule applies', fontsize=13.5, weight='bold')
    rules = ['Pixels coarser than 0.2 µm', 'No retained objects',
             'More than 6% of signal under the mask is smooth haze', 'Foreground fraction < 1%',
             'Large objects (90th-percentile major axis ≥ 2.25 µm) < 1.16 × local noise',
             '90th-percentile major axis < 2.25 µm AND local contrast < 4 background SD']
    for k, r in enumerate(rules):
        ax.text(5.5, 17.0 - k * 2.1, '• ' + r, fontsize=11.2)
    ax.text(66.5, 21, 'Pass', color=TEAL, fontsize=11.5, ha='center'); arrow(64.5, 19.5, 70.5, 19.5, TEAL)
    box(71.5, 16, 25, 7.5, LTEAL, TEAL, 2)
    ax.text(84, 21.2, 'QC pass: reported index', color=TEAL, fontsize=12.5, weight='bold', ha='center')
    ax.text(84, 17.4, f'{q["index"]:.4f}', color=TEAL, fontsize=18, weight='bold', ha='center')
    ax.text(66.5, 12.3, 'Fail', color='#5d6d7e', fontsize=11.5, ha='center'); arrow(64.5, 10.8, 70.5, 10.8, '#5d6d7e')
    ax.text(84, 12.6, 'Optional marked rescues', fontsize=12, weight='bold', ha='center')
    ax.text(84, 10.2, 'background removal, then black point', color='#5d6d7e', fontsize=10.5, ha='center')
    ax.text(84, 7.4, 'Still failing: no validated index;', fontsize=11, ha='center')
    ax.text(84, 5.3, 'QC failure reason recorded', color='#5d6d7e', fontsize=10.5, ha='center')
    ax.plot([4, 96], [3, 3], color='#d5dbe0', lw=1)
    ax.text(4, 1.5, 'Batch CLI • GUI folder selection, progress and mask inspection • rescued fields marked • '
                    'settings and model SHA-256 recorded', fontsize=10.5, va='center')
    fig.savefig(os.path.join(out, 'Figure 1_v1.0.0.png'), dpi=300)
    plt.close(fig)
    return iid, q, P


# --------------------------------------------------------------------- Figure 2
def figure2(out):
    from paper_numbers import C, F, load
    from rater_agreement import MAIN, oof
    d = load()
    m = d[d.maj.isin(MAIN)].reset_index(drop=True)
    p = oof(m, m.maj.values, F, C)
    idx = p[:, C.index('normal')] * 0.5 + p[:, C.index('fragmented')]
    pred = np.array(C)[p.argmax(1)]
    y = m.maj.values
    spec = json.load(open(os.path.join(ROOT, 'mitofi', 'models', 'index_model_v8.json')))
    lo, hi = spec['lo'], spec['hi']
    lab = {'elongated': 'Elongated', 'normal': 'Intermediate', 'fragmented': 'Fragmented'}
    fig = plt.figure(figsize=(4560 / 300, 2550 / 300), dpi=300)
    a = fig.add_axes([0.13, 0.25, 0.29, 0.58])
    cm = np.array([[((y == r) & (pred == c)).sum() for c in MAIN] for r in MAIN])
    frac = cm / cm.sum(1, keepdims=True)
    a.imshow(frac, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', ['#f2f6f9', '#2f6690']), vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            w = 'bold' if i == j else 'normal'
            a.text(j, i, f'{cm[i, j]}\n({frac[i, j]:.0%})', ha='center', va='center', fontsize=15, weight=w,
                   color='white' if frac[i, j] > 0.55 else INK)
    a.set_xticks(range(3)); a.set_xticklabels([lab[c] for c in MAIN], fontsize=12)
    a.set_yticks(range(3)); a.set_yticklabels([lab[c] for c in MAIN], fontsize=13)
    a.set_xlabel('MitoFI predicted class', fontsize=15, labelpad=14)
    a.set_ylabel('Observer-majority reference', fontsize=13)
    a.tick_params(length=0)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_xticks(np.arange(-.5, 3), minor=True); a.set_yticks(np.arange(-.5, 3), minor=True)
    a.grid(which='minor', color='white', lw=4); a.tick_params(which='minor', length=0)
    fig.text(0.115, 0.9, 'A', fontsize=22, weight='bold'); fig.text(0.145, 0.9, 'Three-class classification', fontsize=17, weight='bold')

    b = fig.add_axes([0.535, 0.25, 0.43, 0.6])
    b.axhspan(hi, 1.0, color=LRED, zorder=0); b.axhspan(0, lo, color=LBLUE, zorder=0)
    for t in (lo, hi):
        b.axhline(t, color='#4a4f55', ls='--', lw=1.8, zorder=1)
    rng = np.random.default_rng(1)
    cols = {'elongated': BLUE, 'normal': GREY, 'fragmented': RED}
    for k, c in enumerate(MAIN):
        v = idx[y == c]
        b.scatter(k + rng.uniform(-0.2, 0.2, len(v)), v, s=22, color=cols[c], alpha=0.6, lw=0, zorder=2)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        b.plot([k, k], [q1, q3], color='white', lw=9, zorder=3); b.plot([k, k], [q1, q3], color=INK, lw=6, zorder=4)
        b.plot([k - 0.15, k + 0.15], [med, med], color='white', lw=8, zorder=3)
        b.plot([k - 0.15, k + 0.15], [med, med], color=INK, lw=5, zorder=4)
    nb, na = (idx < lo), (idx > hi)
    b.text(2.38, (hi + 1) / 2 - 0.035, f'Index > {hi}\n{((y == "fragmented") & na).sum()}/{na.sum()} fragmented',
           color='#9c3d28', fontsize=12, va='center', bbox=dict(fc='white', ec='none', alpha=0.85))
    b.text(2.38, lo / 2 + 0.012, f'Index < {lo}\n{((y == "elongated") & nb).sum()}/{nb.sum()} elongated',
           color='#1d4f7a', fontsize=12, va='center', bbox=dict(fc='white', ec='none', alpha=0.85))
    b.set_xlim(-0.45, 3.1); b.set_ylim(0, 1.0)
    b.set_xticks(range(3)); b.set_xticklabels([f'{lab[c]}\n(n={(y == c).sum()})' for c in MAIN], fontsize=13)
    b.set_ylabel('MitoFI index', fontsize=15); b.set_xlabel('Observer-majority reference', fontsize=15, labelpad=12)
    b.grid(axis='y', color='#dfe3e6'); b.set_axisbelow(True)
    for s in ('top', 'right'):
        b.spines[s].set_visible(False)
    fig.text(0.53, 0.9, 'B', fontsize=22, weight='bold'); fig.text(0.56, 0.9, 'Index and endpoint bands', fontsize=17, weight='bold')
    fig.savefig(os.path.join(out, 'Figure_2_v1.0.0.png'), dpi=300)
    plt.close(fig)
    return cm, (nb.sum(), na.sum())


# --------------------------------------------------------------------- Figure 3
def figure3(out):
    from mitofi.io import to_field
    from mitofi.score import qc, score_mask
    from mitofi.segment import segment
    from paper_numbers import load
    iid = 'img_dd0462'
    img, px, _ = to_field(os.path.join(D, 'images', f'{iid}.ome.tif'))
    lk = tifffile.imread(os.path.join(D, 'masks_batch_v5', f'{iid}.ome_segmentation.tif')) > 0
    v5 = tifffile.imread(os.path.join(D, 'masks_python_v5', f'{iid}.ome_segmentation.tif')) > 0
    v6 = tifffile.imread(os.path.join(D, 'scribbles_v2', 'masks_a', f'{iid}.ome_segmentation.tif')) > 0
    iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)

    # gate on the 500-field pool: released classifier + current gate
    d = load()
    ok = np.array([qc(r)[0] for r in d.to_dict('records')])
    unc, rd = (d.r1 == 'uncertain').values, d.r1.isin(['elongated', 'normal', 'fragmented']).values
    cur = ((~ok & unc).sum(), unc.sum(), (~ok & rd).sum(), rd.sum())
    # the ten WT / MFN1-knockout examples, current tool
    ex = sorted(glob.glob(os.path.join(D, 'test', '*', '*.tif')))
    rej10 = 0
    for f in ex:
        a, p_, _ = to_field(f)
        a = np.asarray(a, np.float32)
        rej10 += not score_mask(segment(a), float(p_), img=a)['qc_pass']

    fig = plt.figure(figsize=(3600 / 300, 2265 / 300), dpi=300)
    fig.text(0.025, 0.955, 'A', fontsize=22, weight='bold'); fig.text(0.075, 0.955, 'Segmentation by classifier version', fontsize=17, weight='bold')
    panels = [('Source image', disp(img), None), ('Labkit mask', lk, None),
              ('Earlier Python classifier', v5, f'IoU with Labkit {iou(v5, lk):.2f}'),
              ('Released classifier (v6)', v6, f'IoU with Labkit {iou(v6, lk):.2f}')]
    for k, (t, a, sub) in enumerate(panels):
        ax = fig.add_axes([0.05 + k * 0.24, 0.585, 0.19, 0.3])
        ax.imshow(a, cmap='gray'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(t, fontsize=13, weight='bold')
        if sub:
            ax.text(0.5, -0.07, sub, transform=ax.transAxes, ha='center', fontsize=11)
        if k == 0:
            scalebar(ax, a.shape, float(px))
    fig.text(0.04, 0.47, 'B', fontsize=22, weight='bold'); fig.text(0.09, 0.47, 'Quality-gate rejection in development data', fontsize=17, weight='bold')
    ax = fig.add_axes([0.2, 0.1, 0.75, 0.31])
    groups = [('Uncertain\n(n=35)', [(30, 35), (22, 35), (cur[0], cur[1])]),
              ('Readable\n(n=465)', [(39, 465), (23, 465), (cur[2], cur[3])]),
              ('WT + MFN1-knockout\nexamples (n=10)', [(3, 10), (0, 10), (rej10, len(ex))])]
    names = ['Size-only gate, development masks', 'Speckle gate, development masks', 'Current gate, released classifier (1.0.0)']
    colors = ['#7b8691', '#9fc5c1', TEAL]
    for gi, (g, vals) in enumerate(groups):
        for k, (n, t) in enumerate(vals):
            yv = gi * 4 + k
            pct = 100 * n / t
            ax.barh(yv, pct, color=colors[k], height=0.8, label=names[k] if gi == 0 else None)
            if n == 0:
                ax.plot(0, yv, 'o', color=colors[k], ms=7, clip_on=False)
            ax.text(pct + 0.8, yv, f'{n}/{t} ({pct:.1f}%)', va='center', fontsize=10.5, weight='bold', color=INK)
    ax.set_yticks([1, 5, 9]); ax.set_yticklabels([g for g, _ in groups], fontsize=12)
    ax.invert_yaxis(); ax.set_xlim(0, 103); ax.set_xlabel('Fields rejected (%)', fontsize=13)
    ax.grid(axis='x', color='#dfe3e6'); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(loc='lower right', frameon=False, fontsize=10.5)
    fig.savefig(os.path.join(out, 'Figure_3_v1.0.0.png'), dpi=300)
    plt.close(fig)
    return iou(v5, lk), iou(v6, lk), cur, (rej10, len(ex))


if __name__ == '__main__':
    out = sys.argv[1]
    print('Figure 1:', figure1(out))
    print('Figure 2:', figure2(out))
    print('Figure 3:', figure3(out))
