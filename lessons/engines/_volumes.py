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
