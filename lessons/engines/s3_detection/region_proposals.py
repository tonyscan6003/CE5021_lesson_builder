"""Lesson 3.2 — Region proposals.

Figures
    proposal_quality      real selective-search proposals scored as TP / FP / FN
    objectness_saliency   multi-scale spectral-residual saliency and the windows it favours
    objectness_cues       colour contrast, edge density and superpixel straddling,
                          each computed for thousands of windows; successes and failures
    objectness_combined   windows ranked by the combined cues
    selective_search      over-segmentation, then greedy merging stepped through
    region_similarity     colour, texture, size and fill similarity of two real regions
    proposal_recall       recall against number of proposals on 13 annotated objects
    svm_training          positives, hard negatives (IoU 0.2-0.5) and ignored proposals

Everything is computed here, in NumPy/scikit-image: objectness after Alexe,
Deselaers and Ferrari (2012), selective search after Uijlings et al. (2013).
Ground truth for the evaluation photos was drawn by hand at 400 px.

Source: reference/legacy_md/lesson_3_2_region_proposals + lesson_3_2_script.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import color, feature, segmentation

from .._common import BuildContext, Figure, C, SERIES, save_mpl, load_image
from .. import _detmetrics as M

FIGURES = ["proposal_quality", "objectness_saliency", "objectness_cues",
           "objectness_combined", "selective_search", "region_similarity",
           "proposal_recall", "svm_training"]

SIDE = 400

# hand-drawn ground truth at SIDE px (independent of every algorithm here)
GT = {
    "sheep": np.array([[69, 214, 183, 316], [200, 218, 312, 298], [398, 189, 446, 250],
                       [507, 139, 572, 195], [527, 180, 590, 253], [584, 155, 661, 207],
                       [670, 144, 752, 204]], float) / 2,
    "horses": np.array([[12, 26, 275, 213], [255, 14, 382, 200], [355, 118, 400, 185]], float),
    "cow_field": np.array([[2, 16, 252, 142]], float),
    "tanker": np.array([[42, 78, 322, 194]], float),
    "zebra": np.array([[48, 40, 357, 252]], float),
}


@lru_cache(maxsize=None)
def photo(key):
    return load_image(key, max_side=SIDE)


def _box(ax, b, col, lw=2.0, ls="-", label=None, fs=8):
    ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col,
                               lw=lw, ls=ls))
    if label:
        ax.text(b[0] + 1, b[1] - 2, label, fontsize=fs, va="bottom",
                bbox=dict(fc=col, ec="none", pad=1.0, alpha=0.9))


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# Objectness (Alexe, Deselaers and Ferrari)
# ---------------------------------------------------------------------------

def sample_windows(h, w, n, seed=0):
    """Random windows, positions uniform and sides log-uniform from 1/20 of the
    image (small enough for the smallest sheep) up to the whole image, as the
    candidate set objectness scores."""
    rng = np.random.default_rng(seed)
    bw = np.exp(rng.uniform(np.log(w / 20), np.log(w), n))
    bh = np.exp(rng.uniform(np.log(h / 20), np.log(h), n))
    x1 = rng.uniform(0, w - bw); y1 = rng.uniform(0, h - bh)
    return np.stack([x1, y1, x1 + bw, y1 + bh], 1).round()


def _integral(a):
    return np.pad(a, ((1, 0), (1, 0)) + ((0, 0),) * (a.ndim - 2)).cumsum(0).cumsum(1)


def _boxsum(I, b):
    """Sums of the array whose integral image is I over boxes b (N, 4)."""
    b = b.astype(int)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return I[y2, x2] - I[y1, x2] - I[y2, x1] + I[y1, x1]


def _clip(b, h, w):
    b = b.copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w); b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b


def _scaled(b, f):
    """Boxes scaled about their centres by linear factor f."""
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    hw, hh = (b[:, 2] - b[:, 0]) * f / 2, (b[:, 3] - b[:, 1]) * f / 2
    return np.stack([cx - hw, cy - hh, cx + hw, cy + hh], 1)


def spectral_saliency(img, scale):
    """Spectral-residual saliency (Hou and Zhang, 2007) at one scale: the log
    amplitude spectrum minus its local average, back-transformed with the
    original phase. Returned at full image size, normalised to [0, 1]."""
    g = cv2.resize(color.rgb2gray(img), (scale, scale), interpolation=cv2.INTER_AREA)
    F = np.fft.fft2(g)
    logA, ph = np.log(np.abs(F) + 1e-9), np.angle(F)
    resid = logA - ndimage.uniform_filter(logA, 3, mode="wrap")
    s = np.abs(np.fft.ifft2(np.exp(resid + 1j * ph))) ** 2
    s = ndimage.gaussian_filter(s, scale / 32 + 0.5)
    s = cv2.resize(s, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (s - s.min()) / (s.max() - s.min() + 1e-12)


def cue_saliency(img, boxes, scales=(16, 32, 64)):
    """MS(w) = sum over scales of Σ_{p in w, I(p) >= θ} I(p) · (fraction of w above θ)."""
    total = np.zeros(len(boxes))
    for s in scales:
        I = spectral_saliency(img, s)
        on = I >= np.quantile(I, 0.8)
        si, ci = _integral(I * on), _integral(on.astype(float))
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        total += _boxsum(si, boxes) * _boxsum(ci, boxes) / np.maximum(area, 1)
    return total


def cue_colour_contrast(img, boxes, ring=1.5, bins=8):
    """χ² distance between the LAB histograms of a window and of the ring
    around it (the window enlarged by `ring`, clipped to the image)."""
    h, w = img.shape[:2]
    lab = color.rgb2lab(img)
    q = [np.clip(((lab[..., 0] / 100) * bins).astype(int), 0, bins - 1),
         np.clip(((lab[..., 1] + 100) / 200 * bins).astype(int), 0, bins - 1),
         np.clip(((lab[..., 2] + 100) / 200 * bins).astype(int), 0, bins - 1)]
    onehot = np.concatenate([np.eye(bins)[c] for c in q], axis=2)      # (h, w, 3·bins)
    I = _integral(onehot)
    inner = _boxsum(I, boxes)
    outer = _boxsum(I, _clip(_scaled(boxes, ring), h, w).round()) - inner
    hi = inner / np.maximum(inner.sum(1, keepdims=True), 1)
    ho = outer / np.maximum(outer.sum(1, keepdims=True), 1)
    return 0.5 * np.sum((hi - ho) ** 2 / (hi + ho + 1e-12), 1)


def canny_edges(img):
    return feature.canny(color.rgb2gray(img), sigma=1.0)


def cue_edge_density(img, boxes, inner=0.7):
    """Density of Canny edges in the ring between a window and the window
    shrunk to `inner` of its size: object boundaries lie in that ring."""
    E = _integral(canny_edges(img).astype(float))
    small = _scaled(boxes, inner).round()
    e = _boxsum(E, boxes) - _boxsum(E, small)
    a = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
         - (small[:, 2] - small[:, 0]) * (small[:, 3] - small[:, 1]))
    return e / np.maximum(a, 1)


def superpixels(img, scale=120):
    return segmentation.felzenszwalb(img, scale=scale, sigma=0.8, min_size=40)


def cue_superpixel_straddling(img, boxes, sp=None):
    """SS(w) = 1 - Σ_s min(|s outside w|, |s inside w|) / |w|: a window that
    cuts through superpixels, rather than containing or excluding them, scores
    low, as superpixels do not straddle object boundaries."""
    sp = superpixels(img) if sp is None else sp
    labels = np.unique(sp)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    straddle = np.zeros(len(boxes))
    for s in labels:
        m = (sp == s)
        inside = _boxsum(_integral(m.astype(np.float32)), boxes)
        straddle += np.minimum(m.sum() - inside, inside)
    return 1 - straddle / np.maximum(area, 1)


def _rank01(v):
    return np.argsort(np.argsort(v)) / (len(v) - 1)


def objectness(img, n=4000, seed=0):
    """Windows and their four cue scores plus a combined score (mean rank)."""
    h, w = img.shape[:2]
    boxes = sample_windows(h, w, n, seed)
    cues = dict(MS=cue_saliency(img, boxes), CC=cue_colour_contrast(img, boxes),
                ED=cue_edge_density(img, boxes), SS=cue_superpixel_straddling(img, boxes))
    combined = np.mean([_rank01(v) for v in cues.values()], 0)
    return boxes, cues, combined


def top_windows(boxes, score, k, t_iou=0.5):
    keep, _ = M.nms(boxes, score, t_iou)
    return boxes[keep[:k]], np.asarray(keep[:k])


# ---------------------------------------------------------------------------
# Selective search (Uijlings et al.)
# ---------------------------------------------------------------------------

def _region_features(img, labels):
    """Per-region colour (25 bins/channel) and texture (8 orientations × 10
    bins per channel) histograms, size and bounding box."""
    n = labels.max() + 1
    f = img.astype(float) / 255
    hc = np.zeros((n, 75))
    for c in range(3):
        b = np.clip((f[..., c] * 25).astype(int), 0, 24)
        hc[:, c * 25:(c + 1) * 25] = np.bincount(labels.ravel() * 25 + b.ravel(),
                                                  minlength=n * 25).reshape(n, 25)
    ht = np.zeros((n, 240))
    for c in range(3):
        gx = ndimage.gaussian_filter(f[..., c], 1.0, order=(0, 1))
        gy = ndimage.gaussian_filter(f[..., c], 1.0, order=(1, 0))
        ang = np.arctan2(gy, gx)
        mag = np.hypot(gx, gy)
        o = np.clip(((ang + np.pi) / (2 * np.pi) * 8).astype(int), 0, 7)
        mb = np.clip((mag / (mag.max() + 1e-12) * 10).astype(int), 0, 9)
        idx = (o * 10 + mb).ravel()
        ht[:, c * 80:(c + 1) * 80] = np.bincount(labels.ravel() * 80 + idx,
                                                  minlength=n * 80).reshape(n, 80)
    size = np.bincount(labels.ravel(), minlength=n).astype(float)
    ys, xs = np.indices(labels.shape)
    bb = np.stack([ndimage.minimum(xs, labels, np.arange(n)), ndimage.minimum(ys, labels, np.arange(n)),
                   ndimage.maximum(xs, labels, np.arange(n)) + 1,
                   ndimage.maximum(ys, labels, np.arange(n)) + 1], 1).astype(float)
    hc /= hc.sum(1, keepdims=True); ht /= ht.sum(1, keepdims=True)
    return hc, ht, size, bb


def similarity(a, b, im_size):
    """s = s_colour + s_texture + s_size + s_fill, each in [0, 1]."""
    s_col = np.minimum(a["hc"], b["hc"]).sum()
    s_tex = np.minimum(a["ht"], b["ht"]).sum()
    s_size = 1 - (a["size"] + b["size"]) / im_size
    bb = [min(a["bb"][0], b["bb"][0]), min(a["bb"][1], b["bb"][1]),
          max(a["bb"][2], b["bb"][2]), max(a["bb"][3], b["bb"][3])]
    s_fill = 1 - ((bb[2] - bb[0]) * (bb[3] - bb[1]) - a["size"] - b["size"]) / im_size
    return dict(colour=s_col, texture=s_tex, size=s_size, fill=s_fill)


def selective_search(img, k=100, seed=0):
    """Hierarchical grouping. Returns the initial label image, the merge list
    (i, j, new) and every region's bounding box, ranked as in the paper: a
    region's value is its level in the hierarchy (1 = the last merge) times a
    random number, and proposals are sorted by increasing value."""
    labels = segmentation.felzenszwalb(img, scale=k, sigma=0.8, min_size=k)
    hc, ht, size, bb = _region_features(img, labels)
    R = {i: dict(hc=hc[i], ht=ht[i], size=size[i], bb=bb[i]) for i in range(len(size))}
    im_size = float(labels.size)
    # neighbours: label pairs that touch horizontally or vertically
    pairs = set()
    for a, b in [(labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])]:
        d = a != b
        for p, q in zip(a[d].ravel(), b[d].ravel()):
            pairs.add((min(p, q), max(p, q)))
    S = {pq: sum(similarity(R[pq[0]], R[pq[1]], im_size).values()) for pq in pairs}
    merges, created = [], list(R)
    nxt = max(R) + 1
    while S:
        (i, j) = max(S, key=S.get)
        a, b = R[i], R[j]
        n = a["size"] + b["size"]
        R[nxt] = dict(hc=(a["hc"] * a["size"] + b["hc"] * b["size"]) / n,
                      ht=(a["ht"] * a["size"] + b["ht"] * b["size"]) / n, size=n,
                      bb=np.array([min(a["bb"][0], b["bb"][0]), min(a["bb"][1], b["bb"][1]),
                                   max(a["bb"][2], b["bb"][2]), max(a["bb"][3], b["bb"][3])]))
        nbrs = {p for pq in S if i in pq or j in pq for p in pq} - {i, j}
        S = {pq: s for pq, s in S.items() if i not in pq and j not in pq}
        for t in nbrs:
            S[(min(t, nxt), max(t, nxt))] = sum(similarity(R[t], R[nxt], im_size).values())
        merges.append((i, j, nxt))
        created.append(nxt)
        nxt += 1
    rng = np.random.default_rng(seed)
    level = {r: len(created) - k for k, r in enumerate(created)}   # 1 = last merge
    value = {r: level[r] * rng.random() for r in created}
    order = sorted(created, key=value.get)
    boxes = np.array([R[r]["bb"] for r in order])
    # duplicates (the same box from different levels) are removed, keeping rank
    _, first = np.unique(boxes, axis=0, return_index=True)
    boxes = boxes[np.sort(first)]
    return labels, merges, R, boxes


@lru_cache(maxsize=None)
def ss(key):
    return selective_search(photo(key))


def labels_after(labels, merges, n_merges):
    """The segmentation after the first n_merges merges."""
    parent = {}
    for i, j, new in merges[:n_merges]:
        parent[i] = new; parent[j] = new

    def root(r):
        while r in parent:
            r = parent[r]
        return r
    lut = np.array([root(r) for r in range(labels.max() + 1)])
    return lut[labels]


def recall_at(boxes, gt, ks, t=0.5):
    out = []
    for k in ks:
        b = boxes[:k]
        out.append((M.iou(gt, b).max(1) > t).mean() if len(b) else 0.0)
    return np.array(out)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _boundaries(img, labels):
    return segmentation.mark_boundaries(img, labels, color=(1, 1, 0), mode="thin")


def fig_proposal_quality(ctx: BuildContext) -> Figure:
    img = photo("sheep"); gt = GT["sheep"]
    _, _, _, boxes = ss("sheep")
    k = 60
    b = boxes[:k]
    Mx = M.iou(b, gt)
    tp = Mx.max(1) > 0.5
    found = Mx.max(0) > 0.5
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.imshow(img); _clean(ax)
    for bb, t in zip(b, tp):
        if not t:
            _box(ax, bb, C["red"], lw=0.8)
    for bb, t in zip(b, tp):
        if t:
            _box(ax, bb, C["green"], lw=2)
    for g, f in zip(gt, found):
        if not f:
            _box(ax, g, C["orange"], lw=2.2, ls="--")
    ax.set_ylim(img.shape[0] * 0.7, img.shape[0] * 0.08)
    ax.set_title(f"the first {k} selective-search proposals: {tp.sum()} true positives "
                 f"(green), {(~tp).sum()} false positives (red), {(~found).sum()} objects "
                 "missed (dashed)", fontsize=10)
    png = save_mpl(fig, ctx, "proposal_quality")
    return Figure("proposal_quality",
                  "Region proposals are regions of the image that are likely to contain "
                  "an object, and only these regions are passed to the classifier. A good "
                  "proposal algorithm produces far fewer regions than a sliding window, "
                  "finds objects of any shape and size, and produces few **false "
                  "positive** proposals, which waste calls to the classifier, and few "
                  "**false negative** proposals, as an object without a proposal cannot be "
                  "recovered by the classifier and the recall of the detector is reduced. "
                  f"Here the first {k} proposals of selective search, computed on the "
                  "photograph, are scored against the seven sheep: a proposal is a true "
                  "positive if its IoU with a sheep exceeds 0.5. "
                  f"{found.sum()} of the 7 sheep are covered.",
                  png=png)


def fig_objectness_saliency(ctx: BuildContext) -> Figure:
    img = photo("dairy_cows")
    scales = (16, 64)
    boxes = sample_windows(*img.shape[:2], 4000, seed=1)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 2.9))
    axes[0].imshow(img); axes[0].set_title("image", fontsize=10)
    for ax, s in zip(axes[1:3], scales):
        ax.imshow(spectral_saliency(img, s), cmap="magma")
        ax.set_title(f"saliency at scale {s} × {s}" + (" (coarse)" if s == 16 else " (fine)"),
                     fontsize=10)
    score = cue_saliency(img, boxes, scales)
    best, _ = top_windows(boxes, score, 6)
    axes[3].imshow(img)
    for b in best:
        _box(axes[3], b, C["yellow"] if "yellow" in C else C["orange"], lw=1.6)
    axes[3].set_title("the 6 highest-scoring windows\n(of 4,000 random windows)", fontsize=10)
    for ax in axes:
        _clean(ax)
    png = save_mpl(fig, ctx, "objectness_saliency")
    return Figure("objectness_saliency",
                  "**Objectness** is a measure of how likely it is that an image window "
                  "covers an object of any class. It is computed from several image cues, "
                  "as no single cue works for every image. The first cue is **multi-scale "
                  "saliency**: salient pixels stand out from their surroundings. Here the "
                  "saliency map is computed from the Fourier transform of the image "
                  "(the spectral residual method) at a coarse and a fine scale. At the "
                  "coarse scale the group of cows on the left stands out as a broad blob; "
                  "at the fine scale the line of cows along the horizon appears in more "
                  "detail, while the uniform sky and grass are not salient at either "
                  "scale. A window scores highly "
                  "when it contains many salient pixels. 4,000 random windows are scored, "
                  "and the best are shown after overlapping windows are removed.",
                  png=png)


def fig_objectness_cues(ctx: BuildContext) -> Figure:
    rows = [("CC", "sheep", "colour contrast: χ² distance between LAB histograms of\n"
             "the window and of the ring around it", True),
            ("ED", "tanker", "edge density: Canny edges in the inner ring of the window", False),
            ("SS", "zebra", "superpixel straddling: windows that cut through\n"
             "superpixels score low", True)]
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11))
    info = []
    for r, (cue, key, desc, _) in enumerate(rows):
        img = photo(key)
        boxes = sample_windows(*img.shape[:2], 4000, seed=2 + r)
        fn = {"CC": cue_colour_contrast, "ED": cue_edge_density,
              "SS": cue_superpixel_straddling}[cue]
        score = fn(img, boxes)
        best, _ = top_windows(boxes, score, 5)
        hit = (M.iou(best, GT[key]).max(1) > 0.5).sum()
        # a window is 'off the object' when most of its area lies outside every
        # ground-truth box (IoU alone cannot tell: a small window on the hull
        # also has a small IoU with the whole ship)
        g = GT[key]
        inter = np.array([[max(0, min(b[2], q[2]) - max(b[0], q[0])) *
                           max(0, min(b[3], q[3]) - max(b[1], q[1])) for q in g] for b in best])
        off = int((inter.max(1) / ((best[:, 2] - best[:, 0]) * (best[:, 3] - best[:, 1])) < 0.5).sum())
        info.append((cue, key, hit, off))
        ax = axes[r, 0]
        if cue == "CC":
            b = best[0]
            ax.imshow(img)
            _box(ax, b, C["sky"], lw=2.2, label="window")
            _box(ax, _clip(_scaled(b[None], 1.5), *img.shape[:2])[0], C["orange"], lw=2.2,
                 label="surrounding ring")
        elif cue == "ED":
            ax.imshow(canny_edges(img), cmap="gray_r")
        else:
            ax.imshow(_boundaries(img, superpixels(img)))
        ax.set_title(desc, fontsize=9.5)
        ax = axes[r, 1]
        ax.imshow(img)
        for g in GT[key]:
            _box(ax, g, "white", lw=1.4, ls="--")
        for k, b in enumerate(best):
            _box(ax, b, SERIES[k % len(SERIES)], lw=2)
        ax.set_title(f"top 5 windows by {cue}: {hit} of 5 cover an object (IoU > 0.5)",
                     fontsize=9.5)
    for ax in axes.ravel():
        _clean(ax)
    png = save_mpl(fig, ctx, "objectness_cues")
    d = {c: (h, o) for c, k, h, o in info}
    return Figure("objectness_cues",
                  "Three further objectness cues, each computed for 4,000 random windows "
                  "(ground truth dashed white). Top: **colour contrast**. The colour "
                  "histogram, in LAB colour space, of the pixels inside a window is "
                  "compared with that of a ring of pixels just outside it; the white sheep "
                  "against the green field give a high contrast. An object with a colour "
                  "similar to its surroundings would not be found. Middle: **edge "
                  "density**. Objects have many edges near their boundary, so the density "
                  "of Canny edges in an inner ring of the window is measured. The ripples "
                  "on the water also produce many edges, so small windows on the water, "
                  "the wake and parts of the hull all score highly: none of the top 5 "
                  f"windows covers the ship, and {d['ED'][1]} lie mainly on the water. "
                  "Bottom: **superpixel straddling**. Superpixels group pixels of similar "
                  "colour and texture and do not cross object boundaries, so a window that "
                  "cuts through superpixels, rather than containing them, is unlikely to "
                  "fit an object. Here the highest-scoring windows contain the whole zebra "
                  f"with some of the surrounding grass, and {d['SS'][0]} of the top 5 fits "
                  "it with an IoU above 0.5. "
                  "Superpixel straddling was found by Alexe et al. to be the best single "
                  "cue, although small objects whose boundaries are not preserved by the "
                  "superpixels can be missed. The final "
                  "cue, **location and size**, favours windows that are likely from the "
                  "statistics of real images, such as windows in the centre of the image.",
                  png=png)


def fig_objectness_combined(ctx: BuildContext) -> Figure:
    keys = ["cow_field", "horses", "sheep"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.7))
    for ax, key in zip(axes, keys):
        img = photo(key)
        boxes, cues, comb = objectness(img, n=4000, seed=5)
        best, _ = top_windows(boxes, comb, 10)
        ax.imshow(img); _clean(ax)
        for g in GT[key]:
            _box(ax, g, C["sky"], lw=2.4)
        for b in best:
            _box(ax, b, C["orange"], lw=1.2)
        hit = (M.iou(GT[key], best).max(1) > 0.5).sum()
        ax.set_title(f"top 10 windows: {hit} of {len(GT[key])} objects covered", fontsize=10)
    png = save_mpl(fig, ctx, "objectness_combined")
    return Figure("objectness_combined",
                  "Region proposals from objectness. Candidate windows are generated at "
                  "random (4,000 here), each is scored by the combined cues (here the mean "
                  "of the rank of each window under the four cues), and the highest-scoring "
                  "windows, after overlapping windows are removed, are proposed (orange; "
                  "ground truth in blue). The large animals are covered by the top ten "
                  "windows, but the small sheep are not: they occupy only a few of the "
                  "random windows, and their cue scores are lower than those of larger "
                  "regions of the field. More candidate windows give better proposals, "
                  "at a higher computational cost; an exhaustive search would be slower "
                  "still, with more than 100,000 windows for a large image. In the "
                  "original method the cues are combined by a classifier trained on "
                  "annotated images, which is complex to train.",
                  png=png)


def fig_selective_search(ctx: BuildContext) -> Figure:
    img = photo("sheep")
    labels, merges, R, boxes = ss("sheep")
    n0 = labels.max() + 1
    steps = [n0 - n for n in (n0, 45, 20, 8)]          # merges to reach those region counts
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.0))
    for ax, s in zip(axes, steps):
        lab = labels_after(labels, merges, s)
        ax.imshow(_boundaries(img, lab)); _clean(ax)
        regs = np.unique(lab)
        for r in regs:
            b = R[r]["bb"]
            if (b[2] - b[0]) * (b[3] - b[1]) > 150:
                _box(ax, b, C["sky"], lw=0.8)
        ax.set_title(f"{'initial over-segmentation' if s == 0 else f'after {s} merges'}: "
                     f"{len(regs)} regions", fontsize=10)
    png = save_mpl(fig, ctx, "selective_search")
    return Figure("selective_search",
                  "**Selective search** is based on image segmentation. The image is first "
                  "over-segmented (Felzenszwalb's graph-based method, here into "
                  f"{n0} regions), so that only the smallest objects consist of a single "
                  "region. The two most similar neighbouring regions are then merged, "
                  "repeatedly, until a single region remains, and the bounding box of "
                  "every region formed along the way is a region proposal (blue). As the "
                  "regions grow, proposals are produced at every scale, from parts of "
                  "objects to whole sheep and groups of sheep, without random windows "
                  f"having to be generated and scored. In total {len(boxes)} distinct "
                  "proposals are produced for this image.",
                  png=png)


def fig_region_similarity(ctx: BuildContext) -> Figure:
    img = photo("sheep")
    labels, merges, R, _ = ss("sheep")
    im_size = float(labels.size)
    # the first merge involving a region inside the large sheep: a real pair
    g = GT["sheep"][0]
    def inside(r):
        b = R[r]["bb"]
        return b[0] >= g[0] - 3 and b[1] >= g[1] - 3 and b[2] <= g[2] + 3 and b[3] <= g[3] + 3
    i, j, new = next(m for m in merges if inside(m[0]) and inside(m[1])
                     and R[m[0]]["size"] > 60 and R[m[1]]["size"] > 60)
    s = similarity(R[i], R[j], im_size)
    lab = labels_after(labels, merges, merges.index((i, j, new)))
    fig = plt.figure(figsize=(14, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.3, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    over = img.astype(float).copy()
    for r, col in [(i, C["blue"]), (j, C["orange"])]:
        m = lab == r
        over[m] = 0.4 * over[m] + 0.6 * np.array(plt.matplotlib.colors.to_rgb(col)) * 255
    pad = 12
    ax.imshow(over.astype(np.uint8)); _clean(ax)
    ax.set_xlim(g[0] - pad, g[2] + pad); ax.set_ylim(g[3] + pad, g[1] - pad)
    ax.set_title("two neighbouring regions", fontsize=10)
    ax = fig.add_subplot(gs[1])
    x = np.arange(75)
    ax.bar(x, R[i]["hc"], color=C["blue"], alpha=0.6, label="region A")
    ax.bar(x, R[j]["hc"], color=C["orange"], alpha=0.6, label="region B")
    for c in (25, 50):
        ax.axvline(c - 0.5, color=C["grey"], lw=0.6)
    ax.set_xticks([12, 37, 62]); ax.set_xticklabels(["R", "G", "B"])
    ax.set_ylabel("fraction of pixels"); ax.legend(fontsize=8.5)
    ax.set_title(f"colour histograms: overlap = s_colour = {s['colour']:.2f}", fontsize=10)
    ax = fig.add_subplot(gs[2])
    names = ["colour", "texture", "size", "fill"]
    ax.barh(names, [s[n] for n in names], color=[SERIES[k] for k in range(4)])
    for k, n in enumerate(names):
        ax.text(s[n] + 0.02, k, f"{s[n]:.3f}", va="center", fontsize=9.5)
    ax.set_xlim(0, 1.25); ax.invert_yaxis()
    ax.set_title(f"similarity s = {sum(s.values()):.2f}", fontsize=10)
    png = save_mpl(fig, ctx, "region_similarity")
    return Figure("region_similarity",
                  "The similarity of two neighbouring regions, computed for a real pair "
                  "from the segmentation of Figure 5. Four measures are combined. **Colour "
                  "similarity**: the overlap (histogram intersection) of the colour "
                  "histograms of the two regions, 25 bins per channel. **Texture "
                  "similarity**: the overlap of histograms of gradient orientations, "
                  "measured with Gaussian derivatives in 8 directions, similar to a HOG "
                  "descriptor. **Size similarity**: small regions are merged first, so "
                  "that objects are built up at all scales. **Fill** (shape "
                  "compatibility): regions that fit together, so that their combined "
                  "bounding box is well filled, are merged first, as they are likely to "
                  "form part of a larger object. The pair with the largest sum is "
                  "merged.",
                  png=png)


def fig_proposal_recall(ctx: BuildContext) -> Figure:
    keys = list(GT)
    ks = np.unique(np.round(np.logspace(0, np.log10(2000), 30)).astype(int))
    curves = {"selective search": [], "objectness": [], "random windows": []}
    n_gt = 0
    for key in keys:
        img, gt = photo(key), GT[key]
        n_gt += len(gt)
        ss_boxes = ss(key)[3]
        boxes, _, comb = objectness(img, n=4000, seed=5)
        ob_boxes, _ = top_windows(boxes, comb, 2000, t_iou=0.8)
        rnd = sample_windows(*img.shape[:2], 2000, seed=9)
        for name, b in [("selective search", ss_boxes), ("objectness", ob_boxes),
                        ("random windows", rnd)]:
            curves[name].append(recall_at(b, gt, ks) * len(gt))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    final = {}
    for (name, rows), col in zip(curves.items(), [C["blue"], C["orange"], C["grey"]]):
        r = np.sum(rows, 0) / n_gt
        final[name] = r
        ax.semilogx(ks, r, "-", color=col, lw=2.2, label=name)
    ax.set_xlabel("number of proposals per image"); ax.set_ylabel("recall (IoU > 0.5)")
    ax.set_ylim(0, 1.03); ax.legend(fontsize=9.5, loc="lower right"); ax.grid(alpha=0.3)
    ax.set_title(f"proposal recall on {len(keys)} photographs, {n_gt} annotated objects",
                 fontsize=10.5)
    png = save_mpl(fig, ctx, "proposal_recall")
    k100 = int(np.argmin(np.abs(ks - 100)))
    return Figure("proposal_recall",
                  "The quality of a proposal method is measured by its **recall**: the "
                  "fraction of the objects covered by at least one proposal (IoU > 0.5), "
                  "as a function of the number of proposals. Both methods are run on "
                  f"{len(keys)} photographs with {n_gt} hand-annotated objects. With "
                  f"{ks[k100]} proposals per image, selective search covers "
                  f"{final['selective search'][k100]:.0%} of the objects and objectness "
                  f"{final['objectness'][k100]:.0%}, against "
                  f"{final['random windows'][k100]:.0%} for the same number of random "
                  "windows. Far fewer windows than the thousands of a sliding window are "
                  "needed for a high recall, which is the purpose of region proposals. "
                  "(The implementations here are simplified versions of the published "
                  "methods, and the evaluation set is small.)",
                  png=png)


def fig_svm_training(ctx: BuildContext) -> Figure:
    img = photo("sheep"); gt = GT["sheep"]
    boxes = ss("sheep")[3][:400]
    v = M.iou(boxes, gt).max(1)
    hard = boxes[(v >= 0.2) & (v <= 0.5)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.7))
    for ax in axes:
        ax.imshow(img); _clean(ax)
        ax.set_ylim(img.shape[0] * 0.66, img.shape[0] * 0.15)
    for g in gt:
        _box(axes[0], g, C["green"], lw=2.2)
    axes[0].set_title(f"positive examples: the {len(gt)} ground-truth boxes", fontsize=10)
    for b in hard[:40]:
        _box(axes[1], b, C["red"], lw=1.0)
    axes[1].set_title(f"hard negative examples: {len(hard)} proposals overlapping "
                      "a sheep with IoU 0.2–0.5", fontsize=10)
    png = save_mpl(fig, ctx, "svm_training")
    return Figure("svm_training",
                  "Training an object recognition system with selective search, as in the "
                  "original work, where a support vector machine (SVM) is the classifier. "
                  "The ground-truth boxes provide the positive examples (left). The "
                  "selective-search proposals that overlap a ground-truth box by only "
                  "20–50% are used as **hard negative** examples (right; the first 40 are "
                  "drawn): they are similar in appearance to the objects but are not "
                  "correctly placed, so the classifier must learn to reject them. After "
                  "training, the false positives found on the training images are added "
                  "to the negative examples and the SVM is retrained.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_proposal_quality(ctx), fig_objectness_saliency(ctx),
            fig_objectness_cues(ctx), fig_objectness_combined(ctx),
            fig_selective_search(ctx), fig_region_similarity(ctx),
            fig_proposal_recall(ctx), fig_svm_training(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("3.2 Selective search")
    key = st.selectbox("image", list(GT))
    img = photo(key)
    labels, merges, R, boxes = ss(key)
    n0 = labels.max() + 1
    s = st.slider("merges", 0, len(merges), 0)
    lab = labels_after(labels, merges, s)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(_boundaries(img, lab)); _clean(ax)
    for r in np.unique(lab):
        _box(ax, R[r]["bb"], C["sky"], lw=0.8)
    st.pyplot(fig)
    st.caption(f"{len(np.unique(lab))} regions (started from {n0})")
