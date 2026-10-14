"""Feature-map volume drawing, shared by every lesson that draws CNN volumes.

Volumes follow the convention of the CNN literature and of the course videos:
the horizontal x direction is the channel (depth) axis, and the y and z axes
are the spatial dimensions. A volume with many channels is therefore a LONG
box, and the square end cap you see at its right is the spatial extent. This
is the orientation used in the AlexNet/VGG-style architecture diagrams, and it
is what makes the kernel read correctly: a kernel spans every channel of its
input, so it is drawn as a dashed cube running the full length of the input
volume but covering only a small patch of the spatial cross-section.

Extracted from engines/s1_convnets/advanced_convolution.py (lesson 1.3) when
Section 2 needed the same drawings.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ._common import C

OBLIQUE = (0.36, 0.28)      # (dx, dy) per unit of end-cap side, for the z axis


def chan_len(channels: int) -> float:
    """Horizontal extent of a volume — its channel count, log-compressed.

    Compressed because a 512-channel volume drawn to scale beside a 3-channel
    one would be off the canvas.
    """
    c = max(int(channels), 1)
    return 0.34 + 1.75 * np.log10(c) / np.log10(512)


def spatial_side(size: int) -> float:
    """Side of the square end cap — the spatial size.

    Square-root rather than log compression: halving the spatial size is a
    thing the lessons point at repeatedly, and under log compression a 32 -> 16
    halving is barely visible.
    """
    s = max(int(size), 1)
    return 0.35 + 1.45 * np.sqrt(s / 64.0)


def volume(ax, x0, ymid, length, side, face, label=None, *, dashed=False,
            lw=1.3, alpha=0.95, zorder=2, fontsize=9.5, label_dy=0.17):
    """Draw a feature-map volume as a cuboid lying along the channel axis.

    `x0` is the left end and `ymid` the vertical centre; `length` is the extent
    along the channel axis and `side` the spatial end-cap side. Returns the
    right-hand extent (oblique offset included) so callers can lay volumes out
    left to right and set `xlim` from the result rather than guessing.
    """
    ox, oy = OBLIQUE[0] * side, OBLIQUE[1] * side
    y0 = ymid - side / 2
    kw = dict(ec=C["red"] if dashed else C["ink"], lw=lw, zorder=zorder,
              ls=(0, (4, 2.5)) if dashed else "solid")
    fc = "none" if dashed else face
    al = 1.0 if dashed else alpha
    # top face
    ax.add_patch(plt.Polygon([(x0, y0 + side), (x0 + ox, y0 + side + oy),
                              (x0 + length + ox, y0 + side + oy),
                              (x0 + length, y0 + side)],
                             fc=fc, alpha=al * (1 if dashed else 0.70), **kw))
    # right end cap: the spatial cross-section
    ax.add_patch(plt.Polygon([(x0 + length, y0), (x0 + length + ox, y0 + oy),
                              (x0 + length + ox, y0 + side + oy),
                              (x0 + length, y0 + side)],
                             fc=fc, alpha=al * (1 if dashed else 0.52), **kw))
    # long front face: height is spatial, length is the channel count
    ax.add_patch(plt.Rectangle((x0, y0), length, side, fc=fc, alpha=al, **kw))
    if label:
        ax.text(x0 + (length + ox) / 2, y0 - label_dy, label,
                ha="center", va="top", fontsize=fontsize)
    return x0 + length + ox


def kernel_cube(ax, x0, ymid, channels, host_side, frac=0.34, label=None):
    """Dashed kernel cube inside a host volume.

    The kernel must have as many channels as the input, so it runs the full
    length of the host volume; spatially it covers only a small patch. Drawn
    with no fill so the host volume shows through.
    """
    side = host_side * frac
    right = volume(ax, x0, ymid, chan_len(channels), side, None,
                    dashed=True, lw=1.5, zorder=6)
    if label:
        ax.text(x0 + chan_len(channels) / 2, ymid - side / 2 - 0.10, label,
                ha="center", va="top", fontsize=8.5, color=C["red"])
    return right


def network_diagram(ax, maps, vecs=(), *, ymid=0.0, side_fn=spatial_side,
                    chan_scale=1.0, gap=1.4, vec_gap=1.1, vec_height=None,
                    vertical=False, fontsize=8.5, op_y=None, shape_y=None,
                    faces=None):
    """Draw a CNN as a row of volumes, each operation in the gap it performs.

    maps  list of dicts {h, c, kind, op, k}: one feature-map volume each, in
          order. `kind` is input / conv / pool (sets the colour), `op` the label
          of the operation that PRODUCED this map (None for the input) and `k`
          its kernel or window size, which sets the dashed cube's size.
    vecs  list of dicts {n, op}: flattened / dense layers, drawn as bars.

    For every operation a dashed cube is drawn inside its input volume — full
    length, because a kernel spans every input channel — with dotted
    projection lines to the single output cell it produces, and the label sits
    in the gap between the two volumes. With `vertical=True` the operation and
    shape labels are rotated to read upwards, which lets a deep network fit;
    otherwise they sit on two rows at `op_y` and `shape_y`.

    Returns (right edge, lowest y used) so callers can set limits.
    """
    faces = faces or {"input": C["light"], "conv": C["sky"], "pool": C["orange"]}
    proj = dict(color=C["red"], lw=0.9, ls=":", zorder=7)
    rot = dict(rotation=90, rotation_mode="anchor") if vertical else {}
    low = ymid

    def op_text(xc, top, s):
        if vertical:
            ax.text(xc, top, s, ha="right", va="center", fontsize=fontsize, **rot)
        else:
            ax.text(xc, op_y, s, ha="center", va="top", fontsize=fontsize)

    def shape_text(xc, below, s):
        nonlocal low
        if vertical:
            ax.text(xc, below, s, ha="right", va="center", fontsize=fontsize,
                    color=C["red"], family="monospace", **rot)
            low = min(low, below - 0.13 * fontsize * len(s) / 8.5 * 1.1)
        else:
            ax.text(xc, shape_y, s, ha="center", va="top", fontsize=fontsize,
                    color=C["red"], family="monospace")

    x, prev = 0.0, None
    for m in maps:
        side, length = side_fn(m["h"]), chan_scale * chan_len(m["c"])
        if prev is not None:
            ps = prev["side"]
            frac = min(max(m["k"] / prev["h"], 0.22), 0.6)
            ks = ps * frac
            ky = ymid + ps / 2 - ks / 2 - 0.10 * ps
            # dashed kernel: full length of the input volume, k × k spatially
            volume(ax, prev["x"], ky, prev["len"], ks, None, dashed=True, lw=1.4,
                   zorder=6)
            cell = max(side / m["h"], 0.05)
            cy = ymid + (ky - ymid) / ps * side
            ax.add_patch(plt.Rectangle((x, cy - cell / 2), 0.05, cell,
                                       fc=C["red"], ec=C["red"], zorder=8))
            xr = prev["x"] + prev["len"]
            for dx, dy in [(0, -ks / 2), (0, ks / 2),
                           (OBLIQUE[0] * ks, ks / 2 + OBLIQUE[1] * ks)]:
                ax.plot([xr + dx, x], [ky + dy, cy], **proj)
            op_text((prev["right"] + x) / 2, ymid - 0.12, m["op"])
        right = volume(ax, x, ymid, length, side, faces[m["kind"]], lw=1.1, alpha=0.9)
        shape_text(x + length / 2, ymid - side / 2 - 0.10,
                   f"{m['h']}×{m['h']}×{m['c']}")
        low = min(low, ymid - side / 2)
        prev = dict(m, x=x, side=side, len=length, right=right)
        x = right + gap

    if not vecs:
        return prev["right"], low
    vec_height = vec_height or (lambda n: 0.3 + np.sqrt(n / vecs[0]["n"]))
    x -= gap - vec_gap
    bars = []
    for v in vecs:
        h = vec_height(v["n"])
        ax.add_patch(plt.Rectangle((x, ymid - h / 2), 0.22, h, fc=C["purple"],
                                   ec=C["ink"], lw=1.1, alpha=0.6))
        shape_text(x + 0.11, ymid - h / 2 - 0.10, str(v["n"]))
        bars.append((x, h, v["op"]))
        x += 0.22 + vec_gap
    # flatten: the whole last volume feeds the first vector
    xl, s = prev["x"] + prev["len"], prev["side"]
    x0, h0, op0 = bars[0]
    for sgn in (1, -1):
        ax.plot([xl, x0], [ymid + sgn * s / 2, ymid + sgn * h0 / 2], **proj)
    op_text((prev["right"] + x0) / 2,
            ymid - max(s, h0) / 2 - 0.12 if vertical else None, op0)
    # dense: every unit to every unit
    for (xa, ha, _), (xb, hb, op) in zip(bars, bars[1:]):
        for s0 in (1, -1):
            for s1 in (1, -1):
                ax.plot([xa + 0.22, xb], [ymid + s0 * ha / 2, ymid + s1 * hb / 2], **proj)
        if vertical:
            op_text((xa + 0.22 + xb) / 2, ymid - max(ha, hb) / 2 - 0.12, op)
        else:
            op_text((xa + 0.22 + xb) / 2, None, op)
    return x - vec_gap, low
