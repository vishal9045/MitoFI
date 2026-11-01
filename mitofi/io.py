"""Reading OME-TIFF, and reducing it to the 2D field the index expects.

A raw acquisition is usually a Z-stack with several channels, sometimes several
timepoints. The index is defined on a single 2D image: a maximum-intensity
projection of the mitochondrial channel, which is what the pilot was built on.

This module performs that reduction explicitly and says what it did, rather than
silently guessing. Where the file records its own pixel size, that is used, so
the caller need not supply one.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np
import tifffile

# OME expresses length in these; the index works in micrometres.
_TO_UM = {'m': 1e6, 'cm': 1e4, 'mm': 1e3, 'µm': 1.0, 'um': 1.0,
          'nm': 1e-3, 'Å': 1e-4, 'angstrom': 1e-4}

# OME units that mean "no spatial calibration" rather than a length
_UNCALIBRATED = {'pixel', 'pixels', 'reference frame'}

# X and Y within this fraction of each other are treated as square
_ANISOTROPY_TOL = 0.01


class CalibrationError(ValueError):
    """The file states a pixel size this tool cannot honestly use.

    Distinct from "no pixel size", which is recoverable by passing --px-um.
    This one means the file said something specific and the tool would have to
    ignore or misread it to continue, so it is raised through pixel_size_um
    rather than swallowed into None.
    """


def _from_ome(tf):
    xml = tf.ome_metadata
    if not xml:
        return None
    root = ET.fromstring(xml)
    for px in root.iter():
        if px.tag.rsplit('}', 1)[-1] != 'Pixels':
            continue
        vx, vy = px.get('PhysicalSizeX'), px.get('PhysicalSizeY')
        if not vx:
            return None
        ux = px.get('PhysicalSizeXUnit', 'µm')
        # "pixel" is OME's way of saying uncalibrated -- Bio-Formats writes it
        # for any image with no spatial scale. That is the recoverable case
        # (pass --px-um), not an unknown unit, which is a refusal.
        if ux in _UNCALIBRATED:
            return None
        if ux not in _TO_UM:
            raise CalibrationError(
                f'pixel size is recorded in {ux!r}, a unit this tool does not '
                f'know. It used to be multiplied by 1.0 and treated as '
                f'micrometres. Re-export the file with its pixel size in '
                f'micrometres.')
        sx = float(vx) * _TO_UM[ux]
        if vy:
            uy = px.get('PhysicalSizeYUnit', ux)
            if uy in _UNCALIBRATED:
                return None
            if uy not in _TO_UM:
                raise CalibrationError(
                    f'the Y pixel size is recorded in {uy!r}, a unit this tool '
                    f'does not know. Re-export the file with its pixel size in '
                    f'micrometres.')
            sy = float(vy) * _TO_UM[uy]
            if sx > 0 and abs(sy - sx) / sx > _ANISOTROPY_TOL:
                raise CalibrationError(
                    f'pixels are not square: X is {sx:g} um and Y is {sy:g} um. '
                    f'Every feature here assumes square pixels -- areas would be '
                    f'wrong by {sy / sx:.2f}x and axis lengths would depend on '
                    f'orientation. Resample to square pixels first.')
        return sx
    return None


def _from_imagej(tf):
    """ImageJ writes calibration as TIFF resolution tags plus a unit string.

    A TIFF saved by Fiji is not OME, so the OME reader finds nothing; the pixel
    size is nonetheless right there in the resolution tag. XResolution is pixels
    per unit, so the pixel size is its reciprocal.
    """
    page = tf.pages[0]
    tag = page.tags.get('XResolution')
    if tag is None:
        return None
    val = tag.value
    num, den = (val if isinstance(val, tuple) else (val, 1))
    if not num:
        return None
    per_unit = num / den
    if per_unit <= 0:
        return None
    unit = (tf.imagej_metadata or {}).get('unit', '') if tf.is_imagej else ''
    unit = {'micron': 'µm', 'microns': 'µm', 'um': 'µm', '\xb5m': 'µm'}.get(unit, unit)
    scale = _TO_UM.get(unit)
    if scale is None:
        return None            # unitless or in pixels: no physical size to report
    return (1.0 / per_unit) * scale


def pixel_size_um(path):
    """Pixel size in micrometres from the file, or None.

    Tries the OME metadata first, then the ImageJ/TIFF resolution tags, so both
    a Bio-Formats export and a TIFF saved by Fiji carry their own scale.
    """
    try:
        with tifffile.TiffFile(path) as tf:
            for reader in (_from_ome, _from_imagej):
                try:
                    v = reader(tf)
                except CalibrationError:
                    raise           # the file said something we must not ignore
                # a malformed or unusual file, not a bug here: AttributeError and
                # KeyError are deliberately not caught, so our own mistakes surface
                except (ET.ParseError, ValueError, TypeError, ZeroDivisionError):
                    v = None
                if v is not None and math.isfinite(v) and v > 0:
                    return v
    except CalibrationError:
        raise
    except (OSError, ValueError, TypeError):
        return None
    return None


def _axes(path, arr):
    with tifffile.TiffFile(path) as tf:
        if tf.series and len(tf.series) > 1:
            raise ValueError(
                f'this file holds {len(tf.series)} image series. MitoFI reads one '
                f'field per file and will not pick a series for you; split the '
                f'file first, or export the series you mean.')
        ax = tf.series[0].axes if tf.series else ''
    if ax and len(ax) == arr.ndim:
        return ax
    # No usable metadata: assume the trailing two axes are the image, and that
    # anything before them is Z. Stated in the returned note, never silent.
    return ('?' * (arr.ndim - 2)) + 'YX'


# The pixel classifier was trained only on 12-bit images (0 to 4095) and reads
# raw intensity, so every field is put on that scale before it is segmented.
TRAINED_MAX = 4095.0


def _significant_bits(path):
    """Bit depth the file says it was acquired at, or None if it does not say."""
    try:
        with tifffile.TiffFile(path) as tf:
            xml = tf.ome_metadata
        if not xml:
            return None
        for e in ET.fromstring(xml).iter():
            if e.tag.rsplit('}', 1)[-1] == 'Pixels':
                v = e.get('SignificantBits')
                return int(v) if v else None
    except (OSError, ET.ParseError, ValueError):
        return None
    return None


def to_trained_scale(arr, significant_bits=None):
    """(field on the classifier's 12-bit scale, note -- empty when unchanged).

    Bit depth is units, not image processing: an 8-bit and a 12-bit image of the
    same cells must give the same answer. Native 8-bit Leica fields used to
    reach the classifier 16 times dimmer than anything it was trained on, and
    almost every one collapsed to an empty mask.

    The depth comes from the file when it records one (OME SignificantBits).
    Otherwise it is inferred: 8-bit containers are 8-bit; a wider integer
    container is the smallest of 12, 16 and 32 bits that holds the image's
    maximum. That inference is what keeps the development images unchanged --
    they are 12-bit data in 16-bit containers that record no depth, every one
    with a maximum of 4095 or less, so they scale by exactly 1. A float image
    in 0..1 is read as a fraction of full scale.
    """
    mx = float(arr.max()) if arr.size else 0.0
    if arr.dtype.kind == 'f':
        if mx <= 1.0:
            full, what = 1.0, 'floating-point 0-1 data'
        else:
            full = next((f for f in (TRAINED_MAX, 65535.0) if mx <= f), mx)
            what = 'floating-point data'
    else:
        if significant_bits:
            bits = significant_bits
        elif arr.dtype.itemsize == 1:
            bits = 8
        else:
            bits = 12
        while mx > 2 ** bits - 1:          # recorded depth is wrong, or wider data
            bits = {8: 12, 12: 16}.get(bits, 32)
        full, what = 2.0 ** bits - 1, f'{bits}-bit data'
    if full == TRAINED_MAX:
        return arr, ''
    factor = TRAINED_MAX / full
    return (arr.astype(np.float32) * factor,
            f'{what} scaled x{factor:.4g} to the 12-bit range the classifier '
            f'was trained on')


# Rolling-ball radius for background: wider than a mitochondrion (0.5-1 um
# across), narrower than the diffuse cytoplasmic haze it is there to remove.
BACKGROUND_UM = 2.5


def estimate_background(img, px_um, radius_um=BACKGROUND_UM, fast=False):
    """The smooth background under the mitochondria, by rolling ball.

    The radius is in micrometres and converted with the field's own pixel
    size, so it is the same physical neighbourhood on every microscope.

    `fast` shrinks the field first, about 100 times quicker. It shrinks by the
    MINIMUM of each block, as Fiji does, never by averaging: averaging blurs
    dense puncta into a smooth field that the ball then calls background, so a
    densely fragmented field read as three times hazier than it is -- the
    phenotype mistaken for the artefact. Within a few percent of the
    full-resolution answer, which is what measuring haze needs; subtraction
    still always runs at full size.
    """
    from skimage.restoration import rolling_ball
    from skimage.transform import resize
    a = np.asarray(img, dtype=np.float32)
    r = radius_um / px_um
    f = max(1, int(r // 8)) if fast else 1
    if f == 1:
        return rolling_ball(a, radius=r)
    h, w = a.shape
    padded = np.pad(a, ((0, -h % f), (0, -w % f)), mode='edge')
    H, W = padded.shape
    small = padded.reshape(H // f, f, W // f, f).min(axis=(1, 3))
    bg = rolling_ball(small, radius=r / f)
    return resize(bg, (H, W), order=1, preserve_range=True)[:h, :w].astype(np.float32)


def subtract_background(img, px_um, radius_um=BACKGROUND_UM):
    """The field with its smooth background removed, at full resolution."""
    a = np.asarray(img, dtype=np.float32)
    return np.clip(a - estimate_background(a, px_um, radius_um), 0, None)


def cytoplasm_floor(img, px_um, q=50, peak=False):
    """Brightness of the cytoplasm under the mitochondria, found per image: the
    q-th percentile of the cell area, or with peak=True its most common level.

    The cell area is split from empty slide by Otsu on a lightly smoothed copy.
    Most of a cell is cytoplasm, so its median lands on the haze level -- as long
    as mitochondria cover under half the cell area. The peak does not need that:
    where mitochondria are dense the commonest level is still the haze, or the
    dark gaps between them.
    """
    from scipy.ndimage import gaussian_filter1d
    from skimage.filters import gaussian, threshold_otsu
    a = np.asarray(img, dtype=np.float32)
    sm = gaussian(a, sigma=0.2 / px_um, preserve_range=True)
    cell = sm > threshold_otsu(sm)
    v = a[cell]
    if not peak:
        return float(np.percentile(v, q))
    h, edges = np.histogram(v, bins=256, range=(0, float(a.max()) or 1.0))
    k = int(np.argmax(gaussian_filter1d(h.astype(float), 2)))
    return float((edges[k] + edges[k + 1]) / 2)


def subtract_floor(img, px_um, q=50, peak=False):
    """(field with everything below the cytoplasm floor cut and the rest stretched
    to the trained range, floor).

    For grainy cytoplasmic signal, which a rolling ball leaves in place because
    it is not smooth: the same move as setting the minimum in Fiji's
    Brightness/Contrast, with the minimum read from the image itself.
    """
    a = np.asarray(img, dtype=np.float32)
    fl = cytoplasm_floor(a, px_um, q, peak)
    # ponytail: stretch uses the image max; a lone hot pixel compresses the rest
    return np.clip(a - fl, 0, None) * (TRAINED_MAX / max(float(a.max()) - fl, 1.0)), fl


def to_field(path, channel=None, timepoint=0):
    """(image2d, px_um_or_None, note).

    Z is maximum-projected, matching how the pilot's images were made. A channel
    must be named when the file holds more than one, because picking one for the
    user is exactly the kind of silent guess that produces a confident wrong
    answer.
    """
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr, scaled = to_trained_scale(arr, _significant_bits(path))
        return arr, pixel_size_um(path), '; '.join(
            n for n in ('2D image, used as is', scaled) if n)

    ax = _axes(path, arr)
    note = [f'axes {ax}']

    # RGB is not a channel stack. tifffile labels the sample axis S, and the
    # reduction below used to take it for an unlabelled Z: a 32x40x3 RGB TIFF
    # came back as a 40x3 "image", having max-projected away Y.
    if 'S' in ax:
        raise ValueError(
            f'this is an RGB or multi-sample TIFF (axes {ax}). MitoFI scores a '
            f'single-channel greyscale field; export the mitochondrial channel '
            f'on its own rather than as one colour of an RGB composite.')

    if 'T' in ax:
        i = ax.index('T')
        n = arr.shape[i]
        if not 0 <= timepoint < n:
            raise ValueError(f'--timepoint {timepoint} but the file has {n} '
                             f'timepoint(s), numbered 0..{n - 1}')
        if n > 1:
            note.append(f'timepoint {timepoint} of {n}')
        arr = np.take(arr, timepoint, axis=i)
        ax = ax[:i] + ax[i + 1:]

    if 'C' in ax:
        i = ax.index('C')
        n = arr.shape[i]
        if n > 1:
            if channel is None:
                raise ValueError(
                    f'{n} channels in this file. Name the mitochondrial one with '
                    f'--channel 0..{n - 1}; guessing it is not safe.')
            if not 0 <= channel < n:
                raise ValueError(f'--channel {channel} but the file has {n} channels')
            note.append(f'channel {channel} of {n}')
        arr = np.take(arr, channel or 0, axis=i)
        ax = ax[:i] + ax[i + 1:]

    # Reduce by axis name. Y and X are the image and are never projected away;
    # anything else left over is Z, or an unlabelled axis assumed to be Z.
    while arr.ndim > 2:
        spare = [j for j, a in enumerate(ax) if a not in 'YX']
        if not spare:
            raise ValueError(f'cannot reduce axes {ax} to one 2D field')
        i = ax.index('Z') if 'Z' in ax else spare[0]
        note.append(f'max projection over {arr.shape[i]} planes'
                    + ('' if ax[i] == 'Z' else
                       f" (axis {ax[i]!r} not labelled; assumed Z)"))
        arr = arr.max(axis=i)
        ax = ax[:i] + ax[i + 1:]
    if ax != 'YX' and set(ax) != {'?'}:
        raise ValueError(f'after reduction the axes are {ax}, not YX')

    arr, scaled = to_trained_scale(arr, _significant_bits(path))
    if scaled:
        note.append(scaled)
    return arr, pixel_size_um(path), '; '.join(note)


def channels(path):
    """Per-channel labels, or [] when the file holds a single channel.

    Position is not identity. The same experiment exported CH1/CH3/CH4 in one
    file and CH1/CH2/CH3 in another, so the mitochondrial dye does not sit at a
    fixed index -- taking a fixed one is how an earlier version of this pipeline
    scored the wrong stain. Names travel with the channel; indices do not.

    Labels come from the OME Channel elements (Name, else Fluor, else the
    emission wavelength). A file that records no names still reports its channel
    count, as 'channel 0', 'channel 1', ... -- placeholders the caller must not
    treat as identifying, because they are only positions again.
    """
    try:
        with tifffile.TiffFile(path) as tf:
            ax = tf.series[0].axes if tf.series else ''
            n = tf.series[0].shape[ax.index('C')] if 'C' in ax else 1
            if n < 2:
                return []
            names = _ome_channel_names(tf)
    except (OSError, ValueError, TypeError, IndexError):
        return []
    if not names or len(names) != n:
        return [f'channel {i}' for i in range(n)]
    return names


def excitations(path):
    """Excitation wavelength in nm per channel, None where unrecorded; [] if one channel.

    Names cannot be trusted to identify a stain. On Leica exports the channel
    named "Mito Tracker Red" sat at index 1 with a 514 nm laser while the
    mitochondria were in channel 0 under 594 nm, named "EGFP", "DAPI" or
    nothing. The laser line is set by the acquisition; the name is typed.
    """
    try:
        with tifffile.TiffFile(path) as tf:
            ax = tf.series[0].axes if tf.series else ''
            n = tf.series[0].shape[ax.index('C')] if 'C' in ax else 1
            xml = tf.ome_metadata
    except (OSError, ValueError, TypeError, IndexError):
        return []
    if n < 2:
        return []
    out = [None] * n
    if xml:
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return out
        img = next((e for e in root.iter() if e.tag.rsplit('}', 1)[-1] == 'Image'), root)
        chs = [c for c in img.iter() if c.tag.rsplit('}', 1)[-1] == 'Channel']
        for i, c in enumerate(chs[:n]):
            try:
                out[i] = float(c.get('ExcitationWavelength'))
            except (TypeError, ValueError):
                pass
    return out


def channel_by_excitation(path, nm, tol=1.0):
    """Index of the one channel excited at `nm`, or None for a one-channel file.

    Refuses rather than guesses when no channel, or more than one, matches.
    """
    ex = excitations(path)
    if not ex:
        return None
    hits = [i for i, w in enumerate(ex) if w is not None and abs(w - nm) <= tol]
    if len(hits) == 1:
        return hits[0]
    have = ', '.join('?' if w is None else f'{w:g}' for w in ex)
    if not any(w is not None for w in ex):
        raise ValueError(f'this file records no excitation wavelengths, so a '
                         f'channel cannot be chosen by laser line; use --channel')
    raise ValueError(f'{len(hits) or "no"} channel(s) excited at {nm:g} nm; this '
                     f'file has {have} nm')


def _ome_channel_names(tf):
    xml = tf.ome_metadata
    if not xml:
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    # the first Image only: a file with several Images would otherwise report
    # their channels concatenated, and the count would not match the array
    img = next((e for e in root.iter()
                if e.tag.rsplit('}', 1)[-1] == 'Image'), root)
    out = []
    for ch in img.iter():
        if ch.tag.rsplit('}', 1)[-1] != 'Channel':
            continue
        label = ch.get('Name') or ch.get('Fluor') or ''
        em = ch.get('EmissionWavelength')
        if em:
            try:
                label = f'{label} {float(em):.0f}'.strip()
            except ValueError:
                pass
        out.append(label or f'channel {len(out)}')
    return out or None
