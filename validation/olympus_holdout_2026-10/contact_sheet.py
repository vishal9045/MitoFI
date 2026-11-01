"""Channel check before any scoring: every channel of two sample files per folder.

The Olympus acquisition preset names a channel "Mito Tracker Red" whether or not
the stain was used, so the channel holding mitochondria is confirmed by eye, not
by name (investigator's rule). One PDF page per source folder; each row a sample
file, each panel the maximum projection of one channel, labelled with the
recorded dye name and laser line. No index is computed.

Inputs and the PDF live outside the repository (they carry condition names):

    .venv/bin/python validation/olympus_holdout_2026-10/contact_sheet.py <holdout dir>
"""
import os
import sys

import numpy as np
import pandas as pd
import tifffile


def mip_channels(path):
    t = tifffile.TiffFile(path)
    s = t.series[0]
    a = s.asarray()
    ax = s.axes
    if 'Z' in ax:
        a = a.max(ax.index('Z'))
        ax = ax.replace('Z', '')
    if 'C' not in ax:
        return [a]
    return [np.take(a, c, ax.index('C')) for c in range(a.shape[ax.index('C')])]


def main(H):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    s = pd.read_pickle(os.path.join(H, 'samples.pkl'))
    out = os.path.join(H, 'channel_check.pdf')
    with PdfPages(out) as pdf:
        for k, (folder, g) in enumerate(sorted(s.groupby('folder'), key=lambda x: x[0])):
            rows = []
            for _, r in g.iterrows():
                p = os.path.join(H, 'samples', r.rel[:-4] + '.ome.tif')
                if os.path.exists(p):
                    rows.append((r, mip_channels(p)))
            if not rows:
                continue
            nc = max(len(c) for _, c in rows)
            fig, axs = plt.subplots(len(rows), nc, figsize=(3.2 * nc, 3.4 * len(rows) + 0.6), squeeze=False)
            fig.suptitle(f'[{k + 1}] {folder}', fontsize=9)
            for i, (r, chans) in enumerate(rows):
                for c in range(nc):
                    a_ = axs[i, c]
                    a_.axis('off')
                    if c >= len(chans):
                        continue
                    im = chans[c].astype(np.float32)
                    lo, hi = np.percentile(im, [0.5, 99.8])
                    a_.imshow(np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1), cmap='gray')
                    lab = r.chans[c] if c < len(r.chans) else '?'
                    star = '  <- named mito' if c == r.mito_ch else ''
                    a_.set_title(f'ch{c + 1} {lab}{star}', fontsize=6.5)
                axs[i, 0].text(0, -0.13, os.path.basename(r.rel)[:60], transform=axs[i, 0].transAxes, fontsize=6)
            plt.tight_layout()
            pdf.savefig(fig, dpi=70)
            plt.close(fig)
    print(out)


if __name__ == '__main__':
    main(sys.argv[1])
