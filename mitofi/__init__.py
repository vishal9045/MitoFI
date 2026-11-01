"""A continuous, calibrated mitochondrial fragmentation index for 2D confocal images.

    from mitofi import segment, score_mask
    mask = segment(image)
    result = score_mask(mask, px_um=0.0829)

The index runs 0 (fully elongated network) to 1 (fully fragmented). It is an
instrument for measuring morphology, calibrated against the majority call of
three blind observers. It is not an assay for any treatment, and was never trained on or validated against an
experimental condition.
"""
from .features import features
from .score import qc, score_mask
from .segment import segment

__version__ = '1.0.0'
__all__ = ['segment', 'features', 'qc', 'score_mask', '__version__']
