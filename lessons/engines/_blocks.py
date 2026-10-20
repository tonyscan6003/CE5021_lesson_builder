"""Pipeline-diagram primitives in the style of the lecture slides.

All functions draw on an axes with an equal aspect ratio and y pointing up
(call `canvas()` to get one). Feature maps follow the course's volume
convention (engines/_volumes.py): the channel axis is horizontal, and the
spatial extent is the square end cap, on which a region of interest is drawn.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch, Polygon

from ._common import C
from ._volumes import OBLIQUE, volume

ARROW_FC, ARROW_EC = "#cfe2f3", "#7ea6cf"


def canvas(w, h, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def photo(ax, img, x, y, h, label=None, fs=8.5, boxes=(), box_col=C["red"], lw=1.2,
          box_labels=None, crop=None):
    """An image `h` high with its lower-left corner at (x, y); boxes are in the
    image's pixel coordinates. Returns (x0, y0, x1, y1)."""
    if crop is not None:
        x1c, y1c, x2c, y2c = crop
        img = img[y1c:y2c, x1c:x2c]
        boxes = [np.asarray(b, float) - [x1c, y1c, x1c, y1c] for b in boxes]
    ih, iw = img.shape[:2]
    w = h * iw / ih
    ax.imshow(img, extent=(x, x + w, y, y + h), zorder=2)
    s = h / ih
    for k, b in enumerate(boxes):
        col = box_col[k] if isinstance(box_col, (list, tuple)) else box_col
        bx0, by1 = x + b[0] * s, y + h - b[1] * s
        bw, bh = (b[2] - b[0]) * s, (b[3] - b[1]) * s
        ax.add_patch(plt.Rectangle((bx0, by1 - bh), bw, bh, fill=False, ec=col, lw=lw, zorder=3))
        if box_labels and box_labels[k]:
            ax.text(bx0, by1 + 0.03, box_labels[k], fontsize=fs - 1, color="k", va="bottom",
                    zorder=4, bbox=dict(fc=col, ec="none", pad=0.8, alpha=0.9))
    if label:
        ax.text(x + w / 2, y - 0.08, label, ha="center", va="top", fontsize=fs)
    return x, y, x + w, y + h


def block(ax, x, y, w, h, text, fc=None, ec=None, fs=9, color="k"):
    fc = fc or "#d9ead3"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec or C["grey"], lw=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            zorder=3)
    return x, y, x + w, y + h


def arrow(ax, x0, y0, x1, y1, width=0.16):
    """The slides' light-blue block arrow from (x0, y0) to (x1, y1)."""
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    hl = min(0.18, L * 0.45)
    ax.add_patch(FancyArrow(x0, y0, dx, dy, width=width, head_width=width * 2.1,
                            head_length=hl, length_includes_head=True, fc=ARROW_FC,
                            ec=ARROW_EC, lw=0.8, zorder=1))


def line_arrow(ax, x0, y0, x1, y1, col=None, lw=1.6, ls="-", style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=col or C["red"], lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0), zorder=5)


def fmap(ax, x, ymid, length, side, fc="#e6e6e6", label=None, fs=8.5, roi=None,
         roi_col=None, lw=1.0):
    """A feature-map volume (channels along x, spatial end cap). `roi` =
    (fx0, fy0, fx1, fy1), fractions of the end cap measured from its top-left,
    is drawn dashed on the end cap. Returns the volume's right edge."""
    right = volume(ax, x, ymid, length, side, fc, lw=lw, alpha=0.95)
    if roi is not None:
        cap_region(ax, x + length, ymid, side, roi, roi_col or C["red"], fill="#f8d0cb")
    if label:
        ax.text(x + (length + OBLIQUE[0] * side) / 2, ymid - side / 2 - 0.08, label,
                ha="center", va="top", fontsize=fs)
    return right


def cap_point(xr, ymid, side, fx, fy):
    """A point on the end cap at fractions (fx across the depth, fy down from the top)."""
    ox, oy = OBLIQUE[0] * side, OBLIQUE[1] * side
    y0 = ymid - side / 2
    return xr + fx * ox, y0 + fx * oy + (1 - fy) * side


def cap_region(ax, xr, ymid, side, roi, col, ls=(0, (3, 2)), fill=None):
    fx0, fy0, fx1, fy1 = roi
    pts = [cap_point(xr, ymid, side, fx0, fy0), cap_point(xr, ymid, side, fx1, fy0),
           cap_point(xr, ymid, side, fx1, fy1), cap_point(xr, ymid, side, fx0, fy1)]
    ax.add_patch(Polygon(pts, closed=True, fill=fill is not None, fc=fill or "none", ec=col,
                         lw=1.4, ls=ls, zorder=6))
    return pts


def cap_grid(ax, xr, ymid, side, n, col="k", lw=0.8):
    """An n x n grid on a volume's end cap (the spatial bins of a pooled map)."""
    for k in range(1, n):
        f = k / n
        ax.plot(*zip(cap_point(xr, ymid, side, 0, f), cap_point(xr, ymid, side, 1, f)),
                color=col, lw=lw, zorder=4)
        ax.plot(*zip(cap_point(xr, ymid, side, f, 0), cap_point(xr, ymid, side, f, 1)),
                color=col, lw=lw, zorder=4)


def bar(ax, x, ymid, h, w=0.16, fc="#eeeeee", ec=None, label=None, fs=8.5, label_above=False):
    """A fully connected layer or a feature vector, drawn as a tall thin bar."""
    ax.add_patch(plt.Rectangle((x, ymid - h / 2), w, h, fc=fc, ec=ec or C["grey"], lw=1.0,
                               zorder=2))
    if label:
        ax.text(x + w / 2, ymid + h / 2 + 0.06 if label_above else ymid - h / 2 - 0.06, label,
                ha="center", va="bottom" if label_above else "top", fontsize=fs)
    return x + w


def bracket(ax, x0, x1, y, text, above=True, fs=9):
    d = 0.1 if above else -0.1
    ax.plot([x0, x0, x1, x1], [y - d, y, y, y - d], color="k", lw=1.0)
    ax.text((x0 + x1) / 2, y + (0.06 if above else -0.06), text, ha="center",
            va="bottom" if above else "top", fontsize=fs)


def dashed_split(ax, x0, x1, y):
    ax.plot([x0, x1], [y, y], color="k", lw=1.2, ls=(0, (4, 3)))


RED_ROI = C["red"]
VEC_FC = "#f4c7c3"
FC_FC = "#eeeeee"
CONV_FC = "#d9ead3"
