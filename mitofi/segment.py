"""Segmentation with the shipped pixel classifier.

Pixel classifier v6: a random forest on the same 25-plane feature stack as the
Labkit-equivalent v5, trained on v5's 43,873 scribbled pixels plus 83,029 more
from 30 fields chosen by image condition across three microscopes (Olympus,
Leica, Allen): dim, bright, hazy, grainy, punctate, swollen and mitotic.
Pixels are weighted so no field, class or microscope dominates. Training and
its tests: validation/scribbles_v2_retrain/ (decision D26).
"""
import os
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
PIXEL_MODEL = os.path.join(MODELS, 'pixel_classifier_v6.joblib')


def feature_stack(img, sigmas):
    """Yield (name, plane) one at a time.

    Generated rather than returned as an array: 25 float32 planes of a large
    field cost hundreds of megabytes held simultaneously, for no benefit.
    """
    im = img.astype(np.float32)
    yield 'identity', im
    for s in sigmas:
        g = ndi.gaussian_filter(im, s)
        yield f'gauss{s:g}', g
        yield f'dog{s:g}', g - ndi.gaussian_filter(im, s / 2)
        yield f'gradmag{s:g}', ndi.gaussian_gradient_magnitude(im, s)
        yield f'log{s:g}', ndi.gaussian_laplace(im, s)
        e = hessian_matrix_eigvals(hessian_matrix(im, sigma=s, order='rc',
                                                  use_gaussian_derivatives=True))
        yield f'hess{s:g}_1', e[0]
        yield f'hess{s:g}_2', e[1]


@lru_cache(maxsize=None)
def load():
    import joblib
    return joblib.load(PIXEL_MODEL)


def segment(img, jobs=None):
    """Binary mask: True where the classifier says mitochondrion.

    `jobs` sets how many cores the forest votes on. The pickled model already
    says every core, and that matters: over a million pixels, one core takes
    5.1 s against 0.7 s on twelve. It is settable because inheriting a number
    from a pickle is a poor way to decide how much of a shared machine to take.
    """
    M = load()
    if jobs:
        M['rf'].n_jobs = jobs
    planes = [p.ravel() for _, p in feature_stack(img, M['sigmas'])]
    pred = M['rf'].predict(np.stack(planes, axis=1))
    return pred.reshape(img.shape).astype(bool)
