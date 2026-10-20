"""Shared stage drawings for the R-CNN family diagrams of lesson 3.3."""
from __future__ import annotations

import numpy as np

from .._common import C
from .. import _blocks as B
from .._volumes import OBLIQUE, volume

FMAP_LEN, FMAP_SIDE = 0.45, 1.9          # output feature maps: channels x spatial
POOL_LEN, POOL_SIDE = 0.4, 0.75          # RoI-pooled 7 x 7 x C volume


def backbone(ax, img, x, ymid, box=None, img_h=1.5, label="input image", fmap_roi=None,
             conv_label="conv layers", fs=8.5):
    """image -> conv layers -> output feature maps. Returns a dict of positions."""
    x0, y0, x1, y1 = B.photo(ax, img, x, ymid - img_h / 2, img_h, label=label, fs=fs,
                             boxes=[box] if box is not None else (), lw=1.6)
    B.arrow(ax, x1 + 0.08, ymid, x1 + 0.45, ymid)
    cx = x1 + 0.5
    B.block(ax, cx, ymid - 0.45, 1.0, 0.9, conv_label, fc=B.CONV_FC, fs=fs)
    B.arrow(ax, cx + 1.08, ymid, cx + 1.45, ymid)
    fx = cx + 1.5
    right = B.fmap(ax, fx, ymid, FMAP_LEN, FMAP_SIDE, label="output feature\nmaps", fs=fs,
                   roi=fmap_roi)
    return dict(img=(x0, y0, x1, y1), conv=cx, fmap_x=fx, fmap_right=right,
                cap_x=fx + FMAP_LEN)


def roi_head(ax, x, ymid, fs=8.5, pool_label="RoI pooling"):
    """RoI pooling -> FC, FC -> RoI feature vector -> class and box branches.
    Returns the x positions of the parts, for brackets and annotations."""
    B.arrow(ax, x, ymid, x + 0.35, ymid)
    px = x + 0.4
    pr = volume(ax, px, ymid, POOL_LEN, POOL_SIDE, "#f4c7c3", lw=1.2, alpha=0.95)
    ax.text(px + POOL_LEN / 2, ymid + POOL_SIDE / 2 + OBLIQUE[1] * POOL_SIDE + 0.08,
            pool_label, ha="center", va="bottom", fontsize=fs)
    ax.text(px + POOL_LEN / 2, ymid - POOL_SIDE / 2 - 0.08, "7 × 7 × C", ha="center",
            va="top", fontsize=fs - 0.5)
    B.arrow(ax, pr + 0.08, ymid, pr + 0.42, ymid)
    f1 = pr + 0.48
    B.bar(ax, f1, ymid, 0.75, fc=B.FC_FC)
    B.bar(ax, f1 + 0.3, ymid, 0.75, fc=B.FC_FC)
    ax.text(f1 + 0.23, ymid - 0.45, "FC\nlayers", ha="center", va="top", fontsize=fs)
    B.arrow(ax, f1 + 0.5, ymid, f1 + 0.82, ymid)
    vx = f1 + 0.88
    B.bar(ax, vx, ymid, 1.0, w=0.13, fc=B.VEC_FC)
    ax.text(vx + 0.065, ymid - 0.58, "RoI feature\nvector", ha="center", va="top", fontsize=fs)
    return dict(pool=px, fc=f1, vec=vx, end=heads(ax, vx + 0.13, ymid, fs))


def heads(ax, x, ymid, fs=8.5, fc_layer=True):
    """Two branches: FC -> softmax (class) above, FC -> box regressor below."""
    for dy, name in [(0.55, "softmax\n(class)"), (-0.55, "bounding-box\nregressor")]:
        B.arrow(ax, x + 0.05, ymid + dy * 0.3, x + 0.4, ymid + dy)
        hx = x + 0.45
        if fc_layer:
            B.bar(ax, hx, ymid + dy, 0.5, fc=B.FC_FC)
            ax.text(hx + 0.08, ymid + dy + (0.3 if dy > 0 else -0.3), "FC", ha="center",
                    va="bottom" if dy > 0 else "top", fontsize=fs - 0.5)
            B.arrow(ax, hx + 0.2, ymid + dy, hx + 0.5, ymid + dy)
            hx += 0.55
        B.bar(ax, hx, ymid + dy, 0.5, w=0.12, fc=B.VEC_FC)
        ax.text(hx + 0.2, ymid + dy, name, ha="left", va="center", fontsize=fs)
    B.dashed_split(ax, x + 0.45, hx + 0.5, ymid)
    return hx + 1.3


def projection_arrow(ax, img_box, box, img_shape, cap_x, ymid, roi, top, text="RoI projection",
                     fs=8.5):
    """The slides' red arrow from a box on the input image over to the RoI on
    the end cap of the feature maps."""
    x0, y0, x1, y1 = img_box
    s = (y1 - y0) / img_shape[0]
    bx = x0 + (box[0] + box[2]) / 2 * s
    by = y1 - box[1] * s
    tx, ty = B.cap_point(cap_x, ymid, FMAP_SIDE, (roi[0] + roi[2]) / 2, roi[1])
    ax.plot([bx, bx, tx], [by, top, top], color=C["red"], lw=1.6, zorder=5)
    B.line_arrow(ax, tx, top, tx, ty + 0.02)
    ax.text((bx + tx) / 2, top + 0.06, text, ha="center", va="bottom", fontsize=fs,
            color=C["red"])


def roi_fraction(box, img_shape):
    """A box in image pixels as end-cap fractions: the end cap is the spatial
    plane, image width along its oblique edge and image height vertical."""
    h, w = img_shape[:2]
    return (box[0] / w, box[1] / h, box[2] / w, box[3] / h)
