"""The window icon: a mitochondrion, drawn rather than shipped.

A few lines of numpy and stdlib zlib produce it at whatever size Tk asks for,
which keeps a binary out of the repository and out of the package data, and
means the icon is sharp on a high-density display instead of a scaled 32 px
bitmap.
"""
import struct
import zlib

import numpy as np

OUTER = (46, 160, 110)       # outer membrane
MATRIX = (18, 74, 56)        # matrix, darker so the cristae read against it
CRISTAE = (120, 226, 170)
ANGLE = -25.0                # the organelle lies on a diagonal
FOLDS = 6                    # cristae
SS = 4                       # supersampling, for the anti-aliased edge


def _png(rgba):
    """Encode an HxWx4 uint8 array as PNG. Tk 8.6 takes these bytes directly."""
    h, w = rgba.shape[:2]
    raw = b''.join(b'\x00' + rgba[y].tobytes() for y in range(h))   # filter 0 per row

    def chunk(tag, data):
        body = tag + data
        return struct.pack('>I', len(data)) + body + struct.pack('>I', zlib.crc32(body))

    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 9))
            + chunk(b'IEND', b''))


def _draw(n):
    """Hard-edged HxWx4 picture, drawn oversized so it can be smoothed down."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    cx = cy = (n - 1) / 2
    t = np.radians(ANGLE)
    # rotate into the organelle's own frame, then work in units of its half-axes
    u = ((x - cx) * np.cos(t) + (y - cy) * np.sin(t)) / (0.46 * n)
    v = (-(x - cx) * np.sin(t) + (y - cy) * np.cos(t)) / (0.26 * n)

    r = np.hypot(u, v)
    body = r <= 1.0
    membrane = body & (r > 0.78)
    inner = body & (r < 0.74)

    # Cristae are folds of the inner membrane, each running in from one wall and
    # stopping partway across, alternating sides. Drawn as a set of curved bars
    # rather than one continuous wave: a single wave reads as a letter, not as
    # an organelle.
    cristae = np.zeros_like(body)
    half = 2.0 / (FOLDS + 1)
    for i in range(FOLDS):
        at = -1.0 + half * (i + 1)
        side = 1 if i % 2 else -1
        curve = at + 0.05 * side * np.cos(v * np.pi / 2)
        span = (v * side < 0.95) & (v * side > -0.25)
        cristae |= inner & span & (np.abs(u - curve) < 0.05)

    rgba = np.zeros((n, n, 4), np.uint8)
    rgba[body] = (*MATRIX, 255)
    rgba[cristae] = (*CRISTAE, 255)
    rgba[membrane] = (*OUTER, 255)
    return rgba


def art(n=64):
    """An HxWx4 picture of a mitochondrion: membrane, matrix and cristae.

    Drawn at SS times the size and averaged down, which is the whole of the
    anti-aliasing: hard-edged ellipses at 32 px look like they were cut with
    scissors. Averaged with the alpha premultiplied, so the transparent
    surround does not bleed a dark halo into the membrane.
    """
    big = _draw(n * SS).astype(np.float64)
    a = big[..., 3:] / 255.0
    pre = np.concatenate([big[..., :3] * a, big[..., 3:]], axis=2)
    box = pre.reshape(n, SS, n, SS, 4).mean(axis=(1, 3))
    out = np.zeros((n, n, 4))
    alpha = box[..., 3:] / 255.0
    out[..., :3] = np.divide(box[..., :3], np.where(alpha > 0, alpha, 1))
    out[..., 3:] = box[..., 3:]
    return np.clip(out, 0, 255).astype(np.uint8)


def png(n=64):
    return _png(art(n))


if __name__ == '__main__':      # write one to look at
    import sys
    open(sys.argv[1] if len(sys.argv) > 1 else 'icon.png', 'wb').write(png(256))
