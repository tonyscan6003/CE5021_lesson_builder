"""Lesson 4.6 — Histogram of oriented gradients.

Figures
    hog_window          the 64x128 detector window, its 8x8 cells and 2x2 blocks; a positive and a negative patch
    hog_cell            gradients in one 8x8 cell and its 9-bin unsigned histogram with bilinear voting
    hog_descriptor      cells -> normalised blocks -> the 3780-element vector
    hog_visualisation   the HOG representation of a pedestrian patch and of a scene
    hog_pipeline        training positives and negatives -> HOG -> SVM; detection with a sliding window
    pedestrian_pyramid  detections with a coarse and a fine image pyramid, before and after NMS
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from skimage.feature import hog as sk_hog

from .._common import BuildContext, Figure, C, save_mpl, load_image, norm01, to_uint8
from .._vendor import hog_bilin

FIGURES = ["hog_window", "hog_cell", "hog_descriptor", "hog_visualisation", "hog_pipeline", "pedestrian_pyramid"]

WIN_W, WIN_H = 64, 128


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def people_hog():
    h = cv2.HOGDescriptor(); h.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()); return h


def detect(rgb, scale=1.05, stride=8, group=True):
    """Pretrained HOG pedestrian detector; group=False returns the raw, overlapping hits."""
    rects, w = people_hog().detectMultiScale(
        rgb, 0.0, (stride, stride), (0, 0), scale, 2.0 if group else 0.0)
    return [tuple(int(v) for v in r) for r in rects], np.asarray(w).ravel()


def pedestrian_patches(key="arlit_street", max_side=900):
    """A positive 64x128 patch (a detected person) and a negative patch (background)."""
    rgb = load_image(key, max_side=max_side); g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rects, w = detect(rgb)
    x, y, ww, hh = max(rects, key=lambda r: r[2] * r[3])
    pos = cv2.resize(g[y:y + hh, x:x + ww], (WIN_W, WIN_H), interpolation=cv2.INTER_AREA)
    neg = cv2.resize(g[20:20 + 2 * hh, 20:20 + 2 * ww], (WIN_W, WIN_H), interpolation=cv2.INTER_AREA)
    return rgb, g, pos, neg, (x, y, ww, hh)


def cell_gradients(patch):
    p = patch.astype(float)
    gx = np.zeros_like(p); gy = np.zeros_like(p)
    gx[:, 1:-1] = p[:, 2:] - p[:, :-2]; gy[1:-1, :] = p[2:, :] - p[:-2, :]
    return np.hypot(gx, gy), np.degrees(np.arctan2(gy, gx)) % 180


def unsigned_hist_bilinear(mag, ang, n_bins=9):
    """9-bin histogram over 0..180 with each magnitude split between the two nearest bin centres."""
    width = 180 / n_bins; centres = np.arange(n_bins) * width
    h = np.zeros(n_bins)
    for m, a in zip(mag.ravel(), ang.ravel()):
        i = int(a // width) % n_bins; j = (i + 1) % n_bins
        ci = centres[i]; frac = (a - ci) / width
        h[i] += m * (1 - frac); h[j] += m * frac
    return h, centres


def nms(rects, weights, thr=0.3):
    if len(rects) == 0:
        return []
    boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in rects], float); order = np.argsort(-np.asarray(weights).ravel())
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]); area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_o - inter)
        order = order[1:][iou < thr]
    return [rects[i] for i in keep]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_hog_window(ctx: BuildContext) -> Figure:
    rgb, g, pos, neg, (x, y, w, h) = pedestrian_patches()
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[2.2, 0.7, 0.7, 0.7]))
    out = rgb.copy(); cv2.rectangle(out, (x, y), (x + w, y + h), (213, 94, 0), 3)
    axes[0].imshow(out); axes[0].set_title("a 64 × 128 detector window on a pedestrian")
    axes[1].imshow(pos, cmap="gray"); axes[1].set_title("positive patch")
    axes[2].imshow(neg, cmap="gray"); axes[2].set_title("negative patch")
    axes[3].imshow(pos, cmap="gray", alpha=0.6)
    for cx in range(0, WIN_W + 1, 8):
        axes[3].axvline(cx - 0.5, color=C["sky"], lw=0.6)
    for cy in range(0, WIN_H + 1, 8):
        axes[3].axhline(cy - 0.5, color=C["sky"], lw=0.6)
    axes[3].add_patch(plt.Rectangle((15.5, 31.5), 16, 16, fill=False, ec=C["red"], lw=2)); axes[3].set_title("8 × 8 cells; a 2 × 2 block in red", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "hog_window")
    return Figure("hog_window",
                  "The HOG detector window is 128 × 64 pixels, a shape that fits an upright "
                  "pedestrian; every training image is this size. The window is tiled "
                  "with 8 × 8 pixel cells, grouped into overlapping 2 × 2 blocks. Because "
                  "the training images are upright, the trained detector is not rotation "
                  "invariant.", png=png)


def strongest_cell(pos):
    """Top-left corner of the 8x8 cell carrying the largest total gradient magnitude."""
    mag, _ = cell_gradients(pos)
    best, best_v = (40, 24), -1.0
    for cy in range(1, pos.shape[0] - 9, 8):
        for cx in range(1, pos.shape[1] - 9, 8):
            v = float(mag[cy:cy + 8, cx:cx + 8].sum())
            if v > best_v:
                best, best_v = (cy, cx), v
    return best


def fig_hog_cell(ctx: BuildContext) -> Figure:
    _, _, pos, _, _ = pedestrian_patches()
    cy, cx = strongest_cell(pos)
    cell = pos[cy - 1:cy + 9, cx - 1:cx + 9]
    mag, ang = cell_gradients(cell)
    mag, ang, cell_in = mag[1:-1, 1:-1], ang[1:-1, 1:-1], cell[1:-1, 1:-1]
    hist, centres = unsigned_hist_bilinear(mag, ang)
    p = hog_bilin.generate_params(seed=3); c = hog_bilin.compute(p)
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.4, 0.9, 1.3, 1.2])
    ax = fig.add_subplot(gs[0]); ax.imshow(pos, cmap="gray"); ax.add_patch(plt.Rectangle((cx - 0.5, cy - 0.5), 8, 8, fill=False, ec=C["red"], lw=2)); ax.set_title("one 8 × 8 cell"); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1]); ax.imshow(cell_in, cmap="gray")
    yy, xx = np.mgrid[0:8, 0:8]
    ax.quiver(xx, yy, np.cos(np.deg2rad(ang)) * mag, -np.sin(np.deg2rad(ang)) * mag,
              color=C["orange"], angles="xy", scale_units="xy", scale=mag.max() * 1.6, width=0.02)
    ax.set_title("gradient at each pixel\n(centre difference, unsigned)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[2]); ax.bar(centres, hist, width=16, color=C["blue"]); ax.set_xticks(centres); ax.set_xticklabels([f"{int(v)}°" for v in centres], fontsize=7)
    ax.set_title("9-bin histogram, 0° to 180°", fontsize=9); ax.set_xlabel("bin centre")
    ax = fig.add_subplot(gs[3]); ax.axis("off")
    ax.text(0, 0.95, "voting by bilinear interpolation", fontsize=10, weight="bold")
    ax.text(0, 0.8, f"pixel: magnitude μ = {p['mu']}, angle θ = {p['theta']}°", fontsize=9)
    ax.text(0, 0.66, f"nearest bin centres: {c['phi1']}° and {c['phi2']}°", fontsize=9)
    ax.text(0, 0.50, f"μ₁ = μ·|{c['phi2']} − {p['theta']}| / 20 = {c['mu1']}", fontsize=9)
    ax.text(0, 0.38, f"μ₂ = μ·|{p['theta']} − {c['phi1']}| / 20 = {c['mu2']}", fontsize=9)
    ax.text(0, 0.16, "the closer bin centre receives the\nlarger share; angles above 180°\nhave 180° subtracted first", fontsize=8, color=C["grey"], va="top")
    png = save_mpl(fig, ctx, "hog_cell")
    return Figure("hog_cell",
                  "The HOG cell histogram. Gradients are computed with simple centre "
                  "differences and no pre-smoothing, since fine gradients matter. Each "
                  "cell's 64 gradients are binned by unsigned angle (0° to 180°) into 9 "
                  "bins; each magnitude is split between the two nearest bin centres in "
                  "proportion to the angular distance, so a small change in angle moves "
                  "the histogram only slightly.", png=png)


def fig_hog_descriptor(ctx: BuildContext) -> Figure:
    _, _, pos, _, _ = pedestrian_patches()
    fd = sk_hog(pos, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    fig = plt.figure(figsize=(13, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 2])
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 8); ax.set_ylim(16, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for i in range(9):
        ax.axvline(i, color=C["light"]);
    for j in range(17):
        ax.axhline(j, color=C["light"])
    for k, (bx, by) in enumerate([(1, 2), (2, 2), (1, 3)]):
        ax.add_patch(plt.Rectangle((bx, by), 2, 2, fill=False, ec=[C["red"], C["blue"], C["green"]][k], lw=2))
    ax.set_title("8 × 16 cells; three of the 7 × 15\noverlapping 2 × 2 blocks", fontsize=9)
    ax = fig.add_subplot(gs[1]); ax.axis("off")
    ax.text(0, 0.9, "one block:", fontsize=10, weight="bold")
    ax.text(0, 0.78, "4 cells × 9 bins = 36 values,\nnormalised together", fontsize=9, va="top")
    ax.text(0, 0.5, "whole window:", fontsize=10, weight="bold")
    ax.text(0, 0.38, "7 × 15 = 105 blocks\n105 × 36 = 3780 values", fontsize=9, va="top")
    ax.text(0, 0.1, f"skimage.feature.hog gives {fd.size}", fontsize=9, color=C["green"])
    ax = fig.add_subplot(gs[2]); ax.plot(fd, color=C["blue"], lw=0.5); ax.set_title(f"the descriptor vector of the positive patch: {fd.size} elements"); ax.set_xlabel("element")
    png = save_mpl(fig, ctx, "hog_descriptor")
    return Figure("hog_descriptor",
                  "From cells to the final descriptor. Cells are grouped into 2 × 2 "
                  "blocks whose four 9-bin histograms are concatenated into 36 values and "
                  "normalised together; the blocks overlap by one cell, so each cell "
                  "appears in up to four blocks with different normalisations, which "
                  "improves invariance to illumination. The 7 × 15 blocks of the window "
                  "give a 3780-element feature vector.", png=png)


def fig_hog_visualisation(ctx: BuildContext) -> Figure:
    rgb, g, pos, neg, _ = pedestrian_patches()
    _, vis_pos = sk_hog(pos, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    _, vis_neg = sk_hog(neg, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    small = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    _, vis_scene = sk_hog(small, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    fig, axes = plt.subplots(1, 5, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[0.6, 0.6, 0.6, 0.6, 2.4]))
    for ax, im, t in zip(axes, [pos, norm01(vis_pos) ** 0.5, neg, norm01(vis_neg) ** 0.5, norm01(vis_scene) ** 0.5],
                         ["positive", "its HOG", "negative", "its HOG", "HOG of the whole scene"]):
        ax.imshow(im, cmap="gray"); ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "hog_visualisation")
    return Figure("hog_visualisation",
                  "Visualising HOG. Each cell is drawn as a star of lines whose "
                  "brightness gives the histogram value in each direction. The "
                  "pedestrian's silhouette appears as a pattern of edges that the "
                  "support vector machine learns to separate from background patches. "
                  "The representation is sensitive to fine contours and is fast to "
                  "compute for a single window.", png=png)


def fig_hog_pipeline(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 3.4)); ax.set_xlim(0, 12); ax.set_ylim(0, 3.4); ax.axis("off")
    def box(xy, w, h, text, fc="#e8f1f8", fs=8):
        ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fc=fc, ec=C["grey"])); ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)
    def arr(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=12, color=C["ink"]))
    ax.text(0.2, 3.1, "training", fontsize=10, weight="bold", color=C["grey"])
    box((0.2, 1.9), 2.4, 0.9, "128 × 64 patches:\npedestrians (label 1)\nbackground (label 0)"); arr((2.65, 2.35), (3.2, 2.35))
    box((3.2, 1.9), 2.0, 0.9, "HOG\n(3780 values each)", fc="#fdf1e3"); arr((5.25, 2.35), (5.8, 2.35))
    box((5.8, 1.9), 2.2, 0.9, "linear SVM\nlearns a weight vector", fc="#e6f4ee")
    ax.text(0.2, 1.3, "detection", fontsize=10, weight="bold", color=C["grey"])
    box((0.2, 0.2), 2.4, 0.9, "test image\n+ image pyramid"); arr((2.65, 0.65), (3.2, 0.65))
    box((3.2, 0.2), 2.0, 0.9, "sliding window:\nHOG of each window", fc="#fdf1e3"); arr((5.25, 0.65), (5.8, 0.65))
    box((5.8, 0.2), 2.2, 0.9, "SVM score > threshold?", fc="#e6f4ee"); arr((8.05, 0.65), (8.6, 0.65))
    box((8.6, 0.2), 3.2, 0.9, "non-maximum suppression\n→ bounding boxes")
    png = save_mpl(fig, ctx, "hog_pipeline")
    return Figure("hog_pipeline",
                  "Pedestrian detection with HOG is a binary classification problem. "
                  "Labelled 128 × 64 patches are converted to HOG vectors and used to "
                  "train a support vector machine; the trained SVM is then applied to "
                  "every window of an image pyramid, and overlapping detections are "
                  "merged by non-maximum suppression. Collecting and preparing the "
                  "training patches is the laborious part.", png=png)


def fig_pedestrian_pyramid(ctx: BuildContext) -> Figure:
    rgb = load_image("arlit_street", max_side=1200)
    panels = [("coarse pyramid, scale 1.5", 1.5, False, False),
              ("fine pyramid, scale 1.05: raw window hits", 1.05, False, False),
              ("fine pyramid after non-maximum suppression", 1.05, False, True)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (name, sc, _grp, do_nms) in zip(axes, panels):
        rects, w = detect(rgb, scale=sc, group=False)
        keep = nms(rects, w) if do_nms else rects
        out = rgb.copy()
        for x, y, ww, hh in keep:
            cv2.rectangle(out, (x, y), (x + ww, y + hh), (213, 94, 0), 2)
        ax.imshow(out); ax.set_title(f"{name}: {len(keep)} boxes", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "pedestrian_pyramid")
    return Figure("pedestrian_pyramid",
                  "Localisation and scale with OpenCV's pretrained HOG pedestrian "
                  "detector, with the detector's internal grouping switched off so that "
                  "every window hit is shown. A coarse pyramid whose levels are 1.5 times "
                  "apart has few levels and finds little (left). A fine pyramid, factor "
                  "1.05, fires on many overlapping windows around each person (middle), "
                  "and non-maximum suppression collapses each cluster to a single box "
                  "(right). Some boxes on background survive, and the finer pyramid "
                  "costs proportionately more classifications.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_hog_window(ctx), fig_hog_cell(ctx), fig_hog_descriptor(ctx), fig_hog_visualisation(ctx),
            fig_hog_pipeline(ctx), fig_pedestrian_pyramid(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("4.6 Histogram of oriented gradients")
    tab1, tab2 = st.tabs(["HOG of a patch", "Pedestrian detection"])
    with tab1:
        rgb, g, pos, neg, _ = pedestrian_patches()
        ppc = st.select_slider("pixels per cell", [4, 8, 16], value=8); ori = st.select_slider("orientation bins", [4, 6, 9, 12, 18], value=9)
        fd, vis = sk_hog(pos, orientations=ori, pixels_per_cell=(ppc, ppc), cells_per_block=(2, 2), visualize=True)
        c1, c2 = st.columns(2); c1.image(pos, caption="patch", width=200); c2.image(to_uint8(norm01(vis) ** 0.5), caption=f"HOG: {fd.size} values", width=200)
    with tab2:
        key = st.selectbox("image", ["crowd", "arlit_street", "friends"])
        rgb = load_image(key, max_side=900)
        sc = st.slider("pyramid scale factor", 1.01, 1.6, 1.05, 0.01); use_nms = st.checkbox("non-maximum suppression", True)
        rects, w = detect(rgb, scale=sc, group=False)
        keep = nms(rects, w) if use_nms else rects
        out = rgb.copy()
        for x, y, ww, hh in keep:
            cv2.rectangle(out, (x, y), (x + ww, y + hh), (213, 94, 0), 2)
        st.image(out, caption=f"{len(keep)} boxes", width="stretch")
