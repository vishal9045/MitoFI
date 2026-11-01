"""Per-image morphology features from a binary segmentation mask.

Physical units throughout, from each image's own pixel size. No watershed and no
instance-splitting parameter: objects are the connected components of the mask
as segmented, and nothing is split or merged afterwards, so no feature can be
tuned by choosing a separation threshold.
"""
import math

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

MIN_OBJ_UM2 = 0.05   # physical, so the cut is the same object size at every pixel size
NET_UM2 = 2.0        # "large object" threshold for the network-fraction feature


def check_px(px_um):
    """The pixel size as a float, or ValueError.

    Every feature here is in physical units, so the pixel size multiplies into
    all of them. An unchecked one does not fail, it answers: infinity passed the
    quality gate and returned an index of NaN with a confidence of NaN, and zero
    produced "no objects in mask" for a mask full of objects. A number that
    cannot be a length must be refused where it enters, not carried through
    twenty multiplications into a confident wrong result.
    """
    try:
        px = float(px_um)
    except (TypeError, ValueError):
        raise ValueError(f'pixel size must be a number, got {px_um!r}') from None
    if not math.isfinite(px) or px <= 0:
        raise ValueError(f'pixel size must be a finite positive number of '
                         f'micrometres, got {px_um!r}')
    return px


def as_mask(m):
    """A binary mask as bool, or ValueError.

    Anything but bool is checked before it is converted. A uint8 mask is the
    common case and it fails in two different ways: `~mask` on uint8 is a
    bitwise not, so every background pixel becomes 254 or 255 and stays truthy,
    and `img[mask]` with 0/255 is fancy indexing that raises IndexError or, on a
    large enough array, quietly returns rows 0 and 255. A label image with
    values 0..7 is refused outright rather than flattened into one foreground.
    """
    a = np.asarray(m)
    if a.dtype == bool:
        return a
    if a.dtype.kind not in 'buif':
        raise ValueError(f'a mask must be numeric or boolean, got {a.dtype}')
    u = np.unique(a)
    if not np.isin(u, (0, 1, 255)).all():
        raise ValueError(
            f'a mask must be boolean, 0/1 or 0/255; this one holds '
            f'{len(u)} distinct values ({", ".join(str(x) for x in u[:6])}'
            f'{" ..." if len(u) > 6 else ""}). A multi-class label image is not '
            f'a binary mask -- threshold it first, saying which label is '
            f'mitochondria.')
    return a > 0


def local_contrast(img, mask, px_um, ring_um=0.5, max_obj=400, local=False):
    """Median over objects of how far each stands above the background touching it.

    Object size alone cannot tell a severely fragmented cell from an image too
    poor to read: both give small objects, and their size distributions overlap
    almost completely. They differ locally. A mask built from noise has objects
    barely above their own surroundings; small mitochondria are far above them.

    Background here means the unmasked pixels within `ring_um` of the object's
    bounding box -- a local neighbourhood, not a distance-defined annulus.
    Neighbouring objects are excluded from it, so a crowded field is compared
    against real background rather than against its neighbours.

    Objects are the same objects the index measures: 8-connected, and above the
    same physical area cut. They used to be 4-connected and unfiltered, so the
    gate was reasoning about a different set of things from the features it was
    gating.

    Returned in units of the image's background standard deviation. Sampling is
    capped because a noisy mask can hold tens of thousands of specks.

    With `local=True` each object is measured in units of its own ring's noise
    instead: a signal-to-noise ratio. The whole-field SD mixes black background
    with bright cytoplasm, so its scale depends on the microscope -- a cutoff
    of 4.0 set on Olympus images refused clearly fragmented Leica fields. The
    local ratio reads the same range on both.
    """
    px_um = check_px(px_um)
    mask = as_mask(mask)
    lab = label(mask, connectivity=2)
    im = np.asarray(img, dtype=np.float64)
    bg_sd = im[~mask].std() if (~mask).any() else 1.0
    r = max(1, int(round(ring_um / px_um)))
    objs = ndi.find_objects(lab)
    min_px = MIN_OBJ_UM2 / (px_um * px_um)
    keep = [k for k, sl in enumerate(objs)
            if sl is not None and (lab[sl] == k + 1).sum() >= min_px]
    if not keep:
        return float('nan')
    # ceil, not floor: 799 objects over a cap of 400 floor-divided to a step of
    # 1, which examined all 799 -- the cap was not a cap
    step = max(1, math.ceil(len(keep) / max_obj))
    vals = []
    for k in keep[::step]:
        sl = objs[k]
        pad = tuple(slice(max(0, a.start - r), min(d, a.stop + r))
                    for a, d in zip(sl, lab.shape))
        sub = im[pad]
        this = lab[pad] == (k + 1)
        near = (lab[pad] == 0)            # background only, not neighbouring objects
        if this.sum() < 3 or near.sum() < 5:
            continue
        sd = sub[near].std() if local else bg_sd
        vals.append((sub[this].mean() - sub[near].mean()) / max(sd, 1e-6))
    return float(np.median(vals)) if vals else float('nan')


def haze_fraction(img, mask, px_um):
    """Share of the masked signal that is smooth background, 0 to 1.

    Crisp mitochondria on a dark field sit near zero. A field whose
    mitochondria ride on diffuse cytoplasmic signal is mostly haze, and the
    classifier then segments the textured cytoplasm as one large object --
    which used to pass the gate, because contrast was only consulted for small
    objects, and score as confidently elongated.

    Uses the fast background estimate: this is a ratio over the whole mask, so
    pixel-level error in the background averages out.
    """
    from .io import estimate_background
    px_um = check_px(px_um)
    mask = as_mask(mask)
    if not mask.any():
        return float('nan')
    im = np.asarray(img, dtype=np.float64)
    bg = np.minimum(estimate_background(im, px_um, fast=True), im)
    total = im[mask].sum()
    return float(bg[mask].sum() / total) if total > 0 else float('nan')


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float('nan')


_ORTHO = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.uint8)
_DIAG = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], np.uint8)


def skeleton_stats(mask, px):
    """Skeleton path length and endpoint/junction counts, in physical units.

    Length is the sum of the steps between neighbouring skeleton pixels, not the
    count of skeleton pixels: a diagonal step spans sqrt(2) pixels, not one. The
    pixel count made a 40-pixel horizontal line and a 40-pixel diagonal line the
    same 4.0 um, when they span 3.9 and 5.52. Each step is seen from both of its
    ends, hence the halving.

    Junctions are connected groups of high-degree pixels, not the pixels
    themselves: a Y meeting in two adjacent pixels is one junction.
    """
    sk = skeletonize(mask)
    u8 = sk.astype(np.uint8)
    n = ndi.convolve(u8, np.ones((3, 3), np.uint8), mode='constant') - u8
    deg = n[sk]
    steps = (float(ndi.convolve(u8, _ORTHO, mode='constant')[sk].sum())
             + float(ndi.convolve(u8, _DIAG, mode='constant')[sk].sum()) * math.sqrt(2))
    hubs = sk & (n >= 3)
    return dict(
        skel_len_um=steps / 2.0 * px,
        n_endpoints=int((deg == 1).sum()),
        n_junctions=int(label(hubs, connectivity=2).max()),
    )


def features(mask, img, px):
    px = check_px(px)
    mask = as_mask(mask)
    if mask.ndim != 2:
        raise ValueError(f'a mask must be 2D, got shape {mask.shape}')
    if img is not None:
        img = np.asarray(img)
        if img.shape != mask.shape:
            raise ValueError(f'image is {img.shape} but its mask is {mask.shape}; '
                             f'they must be the same field')
        if img.dtype.kind == 'f' and not np.isfinite(img).all():
            raise ValueError('image holds NaN or infinite pixel values')
    area_px = px * px
    lab = label(mask, connectivity=2)
    props = [p for p in regionprops(lab) if p.area * area_px >= MIN_OBJ_UM2]

    f = {}
    fg_px = int(mask.sum())
    f['frame_px'] = int(mask.size)
    f['fg_frac'] = fg_px / mask.size
    f['fg_area_um2'] = fg_px * area_px

    if not props:
        return f

    A = np.array([p.area * area_px for p in props])
    P = np.array([p.perimeter * px for p in props])
    maj = np.array([p.axis_major_length * px for p in props])
    mnr = np.array([p.axis_minor_length * px for p in props])
    sol = np.array([p.solidity for p in props])
    ecc = np.array([p.eccentricity for p in props])
    ext = np.array([p.extent for p in props])
    # aspect ratio guards against degenerate minor axes (single-pixel-wide objects)
    ar = maj / np.maximum(mnr, px)
    ff = 4 * math.pi * A / np.maximum(P, px) ** 2      # 1.0 = perfect circle
    holes = np.array([1 - p.euler_number for p in props])

    f['n_objects'] = len(props)
    f['obj_per_um2_fg'] = len(props) / f['fg_area_um2']

    for name, v in [('area_um2', A), ('perim_um', P), ('major_um', maj), ('aspect', ar),
                    ('solidity', sol), ('ecc', ecc), ('extent', ext), ('formfactor', ff)]:
        f['med_' + name] = q(v, 50)
        f['p90_' + name] = q(v, 90)
    # weighting by area stops a swarm of small objects from drowning the tubules
    f['areawt_aspect'] = float(np.average(ar, weights=A))
    f['areawt_formfactor'] = float(np.average(ff, weights=A))
    f['frac_fg_largest'] = float(A.max() / A.sum())
    f['frac_fg_over_2um2'] = float(A[A >= NET_UM2].sum() / A.sum())
    f['frac_obj_over_2um2'] = float((A >= NET_UM2).mean())

    f['n_holes'] = int(holes.sum())
    f['frac_obj_with_hole'] = float((holes > 0).mean())
    filled = ndi.binary_fill_holes(mask)
    hole_mask = filled & ~mask
    f['hole_area_frac'] = float(hole_mask.sum() / max(fg_px, 1))
    hp = [p for p in regionprops(label(hole_mask, connectivity=2)) if p.area * area_px >= MIN_OBJ_UM2]
    if hp:
        ha = np.array([p.area * area_px for p in hp])
        hf = np.array([4 * math.pi * (p.area * area_px) / max(p.perimeter * px, px) ** 2 for p in hp])
        f['med_hole_um2'] = q(ha, 50)
        f['med_hole_roundness'] = q(hf, 50)
        f['n_holes_round'] = int(((hf > 0.7) & (ha > 0.1)).sum())   # donut candidates
    else:
        f['med_hole_um2'] = f['med_hole_roundness'] = 0.0
        f['n_holes_round'] = 0

    s = skeleton_stats(mask, px)
    f['skel_len_um'] = s['skel_len_um']
    f['skel_len_per_um2_fg'] = s['skel_len_um'] / f['fg_area_um2']
    f['endpoints_per_100um'] = 100 * s['n_endpoints'] / max(s['skel_len_um'], 1e-9)
    f['junctions_per_100um'] = 100 * s['n_junctions'] / max(s['skel_len_um'], 1e-9)
    f['mean_skel_len_per_obj_um'] = s['skel_len_um'] / len(props)

    # intensity is recorded as a covariate for the confound check, never as a shape feature
    if img is not None:
        v = img[mask]
        f['fg_mean_int'] = float(v.mean()) if v.size else 0.0
        f['img_mean_int'] = float(img.mean())
    return f


