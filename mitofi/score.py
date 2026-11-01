"""Quality gate and fragmentation index.

The index runs 0 for a fully elongated network to 1 for a fully fragmented one.

The gate exists because the index cannot distinguish a genuinely fragmented cell
from an image too poor to read: both produce many small disconnected objects,
and the model scores the second one confidently. Of the 40 images at least one
of three blind raters marked unreadable in the pilot, 40% fall outside the
shipped model's mixed band and would have been reported as confident calls.

Gate thresholds were derived from the 35 of those that rater 1 marked
unreadable, against the 465 rater 1 could read.

On that pool, measured from the Labkit masks and their images:

    size only (superseded)  rejects 30/35 unreadable (86%), 39/465 readable (8%)
    current (size AND       rejects 22/35 unreadable (63%), 23/465 readable (5%)
    local contrast)

The current rule was adopted because the size-only rule was rejecting genuinely
fragmented cells -- it discarded three of five MFN1-null fields that scored 0.93
to 0.98 when allowed through. It catches fewer unreadable images in exchange.
The 86%/8% figures described the superseded rule and were quoted here for
longer than they were true.

These are properties of that pool, that classifier and that microscope, and the
contrast cutoff was itself revised after looking at those images. Revalidate
before trusting them on data from elsewhere.

The gate applies these three rules and no others. It does not detect saturation,
blur, defocus, the wrong channel, or a field containing no mitochondria: a
constant all-foreground field passes it and scores 1.0. `qc_pass` means "passed
these rules".
"""
import json
import os
from functools import lru_cache

import numpy as np

from .features import features, haze_fraction, local_contrast

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
INDEX_MODEL = os.path.join(MODELS, 'index_model_v8.json')

MIN_FG_FRAC = 0.01           # below this the segmentation has collapsed
MIN_P90_MAJOR_UM = 2.25      # objects this small are either fragmented or noise
MIN_LOCAL_CONTRAST = 4.0     # ...and this tells those two apart
MAX_HAZE = 0.06              # p99 of readable development fields, on pixel classifier v6 masks
MIN_LARGE_SNR = 1.16         # p1 of readable large-object development fields, same masks
# Coarsest training pixel was 0.196 um (data/features_v5.csv). At 0.24 um/px a
# mitochondrion is one or two pixels wide; the masks traced whole-cell outlines
# and MFN1/MFN2 knockouts scored 0.000-0.003, passing every other check.
MAX_PX_UM = 0.2
HAZE_REASON = 'too much diffuse background'
NOISE_REASON = 'mask looks like noise texture'
COARSE_REASON = 'pixels too coarse'

# Why a field can be refused: its reason starts with the key. Each kind has a
# short name for counts and a sentence saying what it means and what to try.
# "37 rejected" alone cannot tell a user a wrong channel from a dim image from
# haze they could remove, and those need different things done about them.
REFUSALS = {
    COARSE_REASON: (
        'pixels too coarse',
        'the pixels are larger than in any image the tool learned from, so '
        'mitochondria are only a pixel or two wide. Image at higher zoom.'),
    'no objects in mask': (
        'nothing segmented',
        'no mitochondria were found. Check this is the mitochondrial channel '
        'and that the field is not empty.'),
    'segmentation collapsed': (
        'almost nothing segmented',
        'under 1% of the field was called mitochondria. Usually the wrong '
        'channel, an empty field, or a very dim one.'),
    HAZE_REASON: (
        'diffuse background',
        'most of the signal sits on smooth cytoplasmic haze. Removing the '
        'background, offered after the run, often lets these be scored.'),
    NOISE_REASON: (
        'noise texture',
        'what was segmented is pixel noise across the cell body, not '
        'mitochondria: too little signal over a bright, noisy cell.'),
    'mask is speckle': (
        'speckle',
        'the objects are small and barely brighter than their surroundings: '
        'too dim or noisy to read, or genuinely tiny mitochondria at low '
        'contrast, which this cannot tell apart.'),
}
OTHER_REFUSAL = ('could not be read', 'the file could not be processed; the '
                 'reason column gives the error.')


def refusal_kind(reason):
    """(short name, explanation) for a refusal reason."""
    reason = str(reason or '')
    for key, kind in REFUSALS.items():
        if reason.startswith(key):
            return kind
    return OTHER_REFUSAL


@lru_cache(maxsize=None)
def load():
    """The fitted index model: nine standardised features, three classes.

    Plain numbers in a JSON file, not a pickle. A pickled estimator carries the
    version of the library that made it: this model was fitted under
    scikit-learn 1.9 and, unpickled under 1.7, raised AttributeError inside
    predict_proba because the attribute it reaches for had been removed. A
    standardiser and a multinomial logistic regression are a mean, a scale, a
    coefficient matrix and three intercepts; stored as numbers they will still
    be readable when every library here has moved on.
    """
    with open(INDEX_MODEL) as fh:
        M = json.load(fh)
    for k in ('mean', 'scale', 'coef', 'intercept'):
        M[k] = np.array(M[k], dtype=float)
    return M


def predict_proba(M, x):
    """Class probabilities for one feature vector: standardise, then softmax."""
    z = (np.asarray(x, dtype=float) - M['mean']) / M['scale']
    logit = M['coef'] @ z + M['intercept']
    e = np.exp(logit - logit.max())          # shift for numerical stability
    return e / e.sum()


def qc(f):
    """(passed, reason). Reason is empty when passed.

    Small objects alone are not grounds for rejection. An earlier version cut on
    object size, and on MFN1 knockout MEFs -- where losing a fusion protein makes
    mitochondria genuinely small -- it discarded three of five images that scored
    0.93 to 0.98 when allowed through. It was rejecting the phenotype.

    A mask is called speckle only when its objects are both small AND barely
    above the background around them. On the images measured, that keeps all
    five MFN1-null fields and most readable pilot images while still catching
    most of the unreadable ones.
    """
    px = f.get('px_um')
    if px is not None and px > MAX_PX_UM:
        return False, (f"{COARSE_REASON}: {px:.3f} um per pixel, coarser than any "
                       f"training image (limit {MAX_PX_UM} um); image at higher zoom")
    if f.get('n_objects', 0) == 0:
        return False, 'no objects in mask'
    h = f.get('haze')
    if MAX_HAZE is not None and h is not None and np.isfinite(h) and h > MAX_HAZE:
        # Checked before size: a hazy field segments as one large textured
        # object, which the size rule waves through, and then scores as
        # confidently elongated. Background subtraction is the remedy, and the
        # reason says so.
        return False, (f"{HAZE_REASON}: {h:.1%} of the signal under the mask is "
                       f"smooth haze (limit {MAX_HAZE:.0%}); subtract the "
                       f"background and score again")
    if f['fg_frac'] < MIN_FG_FRAC:
        return False, f"segmentation collapsed: foreground {f['fg_frac']:.2%}"
    snr = f.get('local_snr')
    if (f['p90_major_um'] >= MIN_P90_MAJOR_UM and snr is not None
            and np.isfinite(snr) and snr < MIN_LARGE_SNR):
        # Large objects were never examined: the speckle rule below only looks at
        # small ones. A field whose noise texture was segmented as one large
        # sponge passed and scored as confidently elongated. Its objects barely
        # rise above their own surroundings, which real mitochondria do not do.
        return False, (f"{NOISE_REASON}: large objects stand only {snr:.1f} x "
                       f"their surrounding noise (limit {MIN_LARGE_SNR})")
    lc = f.get('local_contrast')
    if f['p90_major_um'] < MIN_P90_MAJOR_UM:
        if lc is None or not np.isfinite(lc):
            # no image supplied, so contrast cannot be measured: fall back to the
            # size rule alone and say so, rather than pass something unchecked
            return False, (f"mask is speckle: 90th percentile major axis "
                           f"{f['p90_major_um']:.3f} um is below {MIN_P90_MAJOR_UM} um "
                           f"(no image given, so local contrast could not be checked)")
        if lc < MIN_LOCAL_CONTRAST:
            return False, (f"mask is speckle: objects average {f['p90_major_um']:.3f} um "
                           f"and stand only {lc:.1f} background SD above their "
                           f"surroundings")
    return True, ''


def score_mask(mask, px_um, img=None):
    """One result dict for one mask. A rejected image gets no index."""
    M = load()
    f = features(mask, img, px_um)
    f['px_um'] = float(px_um)
    if img is not None and f.get('n_objects', 0):
        f['local_contrast'] = local_contrast(img, mask, px_um)
        f['haze'] = haze_fraction(img, mask, px_um)
        f['local_snr'] = local_contrast(img, mask, px_um, local=True)
    passed, reason = qc(f)
    lc, h, snr = f.get('local_contrast'), f.get('haze'), f.get('local_snr')
    out = dict(qc_pass=passed, qc_reason=reason,
               fg_frac=round(f['fg_frac'], 4),
               n_objects=f.get('n_objects', 0),
               p90_major_um=round(f.get('p90_major_um', 0.0), 3),
               local_contrast=(round(lc, 2) if lc is not None and np.isfinite(lc) else ''),
               haze=(round(h, 4) if h is not None and np.isfinite(h) else ''),
               local_snr=(round(snr, 2) if snr is not None and np.isfinite(snr) else ''))
    if all(k in f for k in M['features']):
        p = predict_proba(M, [f[k] for k in M['features']])
        cls = M['classes']
        idx = float(p[cls.index('normal')] * 0.5 + p[cls.index('fragmented')] * 1.0)
        # Band names describe where the number falls, not a proven morphology.
        # The ends are high-purity zones: in the pilot, over 90% of images beyond
        # each cut carried the matching call. "mixed" is the remainder, and it is
        # not homogeneous -- it held fragmented, normal and elongated images alike.
        band = ('fragmented' if idx > M['hi'] else
                'elongated' if idx < M['lo'] else 'mixed')
        out.update(index=round(idx, 4) if passed else None,
                   band=band if passed else 'qc_fail',
                   confidence=round(float(p.max()), 3),
                   # reported separately so overriding the gate is a deliberate act
                   index_unvalidated=round(idx, 4))
    return out
