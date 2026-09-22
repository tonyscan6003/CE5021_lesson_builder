"""Shared utilities for morphological operation question generators.

Binary images are generated from a random polar curve (high-order closed
contour) so the foreground region is always simply connected.
"""
import numpy as np

IMG_SIZE = 9   # default image size (square)

SE_SHAPE_NAMES = {
    'cross':   'cross-shaped',
    'diamond': 'diamond-shaped',
    'oval':    'oval-shaped',
}


# ---------------------------------------------------------------------------
# Structuring element
# ---------------------------------------------------------------------------

def make_se(rng, size=None, shape=None):
    """Return (se_array, shape_str, size_int) for a random SE.

    size  : int, one of [3, 5, 7]
    shape : str, one of ['cross', 'diamond', 'oval']
    """
    if size is None:
        size = int(rng.choice([3, 5]))
    if shape is None:
        shape = str(rng.choice(['cross', 'diamond', 'oval']))
    c = size // 2
    se = np.zeros((size, size), dtype=int)
    if shape == 'cross':
        se[c, :] = 1
        se[:, c] = 1
    elif shape == 'diamond':
        for i in range(size):
            for j in range(size):
                if abs(i - c) + abs(j - c) <= c:
                    se[i, j] = 1
    elif shape == 'oval':
        ry = c + 0.6
        rx = max(c * 0.75 + 0.3, 0.5)
        for i in range(size):
            for j in range(size):
                if (i - c) ** 2 / ry ** 2 + (j - c) ** 2 / rx ** 2 <= 1.0:
                    se[i, j] = 1
    return se, shape, size


# ---------------------------------------------------------------------------
# Binary blob generation
# ---------------------------------------------------------------------------

def make_blob(rng, size=IMG_SIZE, min_frac=0.12, max_frac=0.62, max_tries=40):
    """Generate a random connected binary blob using a polar curve.

    Returns a (size × size) int array with 1s inside a closed polar contour.
    If no valid blob is found in max_tries, a simple circle is returned.
    """
    cy = cx = (size - 1) / 2.0
    min_ones = int(min_frac * size * size)
    max_ones = int(max_frac * size * size)

    for _ in range(max_tries):
        n_terms = int(rng.integers(2, 6))
        amps   = rng.uniform(0.05, 0.28, n_terms)
        freqs  = rng.integers(2, 8, n_terms).astype(float)
        phases = rng.uniform(0, 2 * np.pi, n_terms)
        base_r = rng.uniform(0.23, 0.43) * size

        thetas = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        rs = base_r * (1 + np.sum(
            amps[:, None] * np.cos(freqs[:, None] * thetas[None, :] + phases[:, None]),
            axis=0))
        rs = np.maximum(rs, 0.05)

        poly_x = cx + rs * np.cos(thetas)
        poly_y = cy + rs * np.sin(thetas)
        img = _fill_polygon(poly_x, poly_y, size)
        n = int(img.sum())
        if min_ones <= n <= max_ones:
            return img

    # Fallback: filled circle
    img = np.zeros((size, size), dtype=int)
    c = size // 2
    r_sq = (size // 3) ** 2
    for i in range(size):
        for j in range(size):
            if (i - c) ** 2 + (j - c) ** 2 <= r_sq:
                img[i, j] = 1
    return img


def _fill_polygon(poly_x, poly_y, size):
    """Ray-casting point-in-polygon fill.  Returns (size×size) int array."""
    rr, cc = np.mgrid[0:size, 0:size]
    px = cc.ravel().astype(float) + 0.5   # pixel centre
    py = rr.ravel().astype(float) + 0.5
    n = len(poly_x)
    inside = np.zeros(len(px), dtype=bool)
    xj, yj = poly_x[-1], poly_y[-1]
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        straddles = (yi > py) != (yj > py)
        denom = yj - yi
        with np.errstate(divide='ignore', invalid='ignore'):
            x_cross = np.where(
                straddles & (denom != 0),
                xi + (xj - xi) * (py - yi) / denom,
                np.inf)
        inside ^= straddles & (px < x_cross)
        xj, yj = xi, yi
    return inside.reshape(size, size).astype(int)


# ---------------------------------------------------------------------------
# Morphological operations
# ---------------------------------------------------------------------------

def _se_offsets(se):
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    return [(si - ph, sj - pw) for si in range(se.shape[0])
            for sj in range(se.shape[1]) if se[si, sj]]


def erode(img, se):
    """Binary erosion: output(r,c)=1 iff SE centred at (r,c) fits in foreground."""
    h, w = img.shape
    offsets = _se_offsets(se)
    result = np.zeros_like(img)
    for r in range(h):
        for c in range(w):
            if all(0 <= r + dr < h and 0 <= c + dc < w and img[r + dr, c + dc]
                   for dr, dc in offsets):
                result[r, c] = 1
    return result


def dilate(img, se):
    """Binary dilation: output(r,c)=1 iff SE centred at (r,c) hits any foreground."""
    h, w = img.shape
    offsets = _se_offsets(se)
    result = np.zeros_like(img)
    for r in range(h):
        for c in range(w):
            if img[r, c]:
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        result[nr, nc] = 1
    return result


# ---------------------------------------------------------------------------
# LaTeX formatting
# ---------------------------------------------------------------------------

def fmt_matrix(arr, name=''):
    """Return a $$ ... $$ LaTeX bmatrix for the given 2-D array."""
    rows = [' & '.join(str(int(v)) for v in row) for row in arr]
    body = ' \\\\\n'.join(rows)
    eq = f'{name} = ' if name else ''
    return f'$${eq}\\begin{{bmatrix}}\n{body}\n\\end{{bmatrix}}$$'
