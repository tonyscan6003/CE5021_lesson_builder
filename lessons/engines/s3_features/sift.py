"""Lesson 3.4 — The SIFT detector and descriptor.

Figures
    sift_keypoints_photo   OpenCV SIFT keypoints drawn with their scale and orientation
    sift_scale_space       octaves and scales: the Gaussian and difference-of-Gaussian stacks
    dog_extrema            extrema of the DoG stack across 26 neighbours on a real image
    quadratic_localisation fitting a parabola to three samples refines the position of a maximum
    orientation_assignment 36-bin weighted histogram, smoothed, with peaks above 80% of the maximum
    descriptor_layout      the 4 x 4 grid of 8-bin histograms as a 128-vector, and bilinear splitting
    sift_two_views         SIFT matches between two photographs of the Trevi fountain
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from .._common import BuildContext, Figure, C, SERIES, save_mpl, load_image, norm01
from ._sift_common import sift_keypoints, ratio_matches, draw_matches, gradients, orientation_histogram

FIGURES = ["sift_keypoints_photo", "sift_scale_space", "dog_extrema", "quadratic_localisation",
           "orientation_assignment", "descriptor_layout", "sift_two_views"]

N_SCALES = 3          # scales per octave used for extrema
SIGMA0 = 1.6


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def gaussian_octave(img, sigma0=SIGMA0, n=N_SCALES):
    """Gaussian stack for one octave: n + 3 images with sigma0 * 2^(i/n)."""
    k = 2 ** (1 / n)
    sigmas = [sigma0 * k ** i for i in range(n + 3)]
    stack = [cv2.GaussianBlur(img.astype(float), (0, 0), s) for s in sigmas]
    return np.array(stack), sigmas


def dog_stack(gstack):
    return gstack[1:] - gstack[:-1]


def dog_extrema(dog, thresh=0.03):
    """Positions (scale, row, col) that are maxima or minima over 26 neighbours."""
    mx = ndimage.maximum_filter(dog, size=3)
    mn = ndimage.minimum_filter(dog, size=3)
    cand = ((dog == mx) | (dog == mn)) & (np.abs(dog) > thresh * np.abs(dog).max())
    cand[0] = cand[-1] = False
    return np.argwhere(cand)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_sift_keypoints_photo(ctx: BuildContext) -> Figure:
    g = load_image("trevi_detail", gray=True, max_side=640)
    kp, _ = sift_keypoints(g, n=400)
    vis = cv2.drawKeypoints(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), kp, None, color=(213, 94, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(vis); ax.set_title(f"{len(kp)} strongest SIFT keypoints: circle radius = scale, line = characteristic orientation"); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "sift_keypoints_photo")
    return Figure("sift_keypoints_photo",
                  "SIFT keypoints detected by OpenCV. Each keypoint carries a position, "
                  "a scale (circle radius) and a characteristic orientation (line). SIFT "
                  "is both a detector and a descriptor; the literature sometimes refers "
                  "to only one of the two parts.", png=png)


def fig_sift_scale_space(ctx: BuildContext) -> Figure:
    g = load_image("trevi_detail", gray=True, max_side=400)
    gst, sig = gaussian_octave(g)
    dg = dog_stack(gst)
    fig = plt.figure(figsize=(14, 4.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1])
    for j in range(6):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(gst[j], vmin=0, vmax=255); ax.set_title(f"Gaussian σ = {sig[j]:.2f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for j in range(5):
        ax = fig.add_subplot(gs[1, j]); v = np.percentile(np.abs(dg[j]), 99); ax.imshow(dg[j], cmap="gray", vmin=-v, vmax=v)
        ax.set_title(f"DoG {j + 1} = G{j + 2} − G{j + 1}" + ("  (extrema here)" if 1 <= j <= 3 else "  (auxiliary)"), fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 5]); ax.axis("off")
    ax.text(0, 0.95, "One octave with 3 scales:\n\n6 Gaussian images give\n5 DoG images; extrema are\nsearched in the middle 3,\nthe outer 2 supply the\nneighbours above and below.\n\nThe next octave starts from\nthe image downsampled by 2.", fontsize=9, va="top")
    png = save_mpl(fig, ctx, "sift_scale_space")
    return Figure("sift_scale_space",
                  "The SIFT scale space for one octave. Top: Gaussian images at σ₀ 2^(i/3). "
                  "Bottom: adjacent Gaussians are subtracted to give difference-of-Gaussian "
                  "images that approximate the Laplacian. Three extra scales are computed "
                  "so that the three principal DoG scales have neighbours above and below "
                  "for the extrema search. Further octaves repeat this on the image "
                  "downsampled by two.", png=png)


def fig_dog_extrema(ctx: BuildContext) -> Figure:
    g = load_image("trevi_detail", gray=True, max_side=400)
    gst, sig = gaussian_octave(g)
    dg = dog_stack(gst) / 255.0
    ex = dog_extrema(dg, thresh=0.08)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].imshow(g, vmin=0, vmax=255)
    for s, r, c in ex:
        axes[0].add_patch(plt.Circle((c, r), 2.5 * sig[s], fill=False, ec=C["orange"], lw=0.8))
    axes[0].set_title(f"{len(ex)} DoG extrema in this octave; radius ∝ scale")
    for j, s in enumerate([1, 3]):
        v = np.percentile(np.abs(dg[s]), 99); axes[j + 1].imshow(dg[s], cmap="gray", vmin=-v, vmax=v)
        sel = ex[ex[:, 0] == s]; axes[j + 1].plot(sel[:, 2], sel[:, 1], "o", mfc="none", mec=C["orange"], ms=5)
        axes[j + 1].set_title(f"extrema found in DoG scale {s} (σ ≈ {sig[s]:.1f})")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "dog_extrema")
    return Figure("dog_extrema",
                  "Extrema detection. A sample of the DoG stack is a candidate keypoint "
                  "if it is larger (or smaller) than all 26 neighbours in the three "
                  "adjacent scales. The candidates are quantised to the sampling grid and "
                  "still include weak and edge responses, which the next stage removes.",
                  png=png)


def fig_quadratic_localisation(ctx: BuildContext) -> Figure:
    xs = np.linspace(-2, 2, 400)
    true_peak = 0.35
    f = lambda x: 1 - 0.6 * (x - true_peak) ** 2  # noqa: E731
    samp_x = np.array([-1, 0, 1]); samp_y = f(samp_x)
    a, b, c = np.polyfit(samp_x, samp_y, 2)
    x_hat = -b / (2 * a)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(xs, f(xs), color=C["light"], lw=4, label="true response")
    axes[0].plot(samp_x, samp_y, "o", color=C["blue"], ms=9, label="samples on the grid")
    axes[0].plot(xs, np.polyval([a, b, c], xs), "--", color=C["orange"], label="quadratic fit")
    axes[0].axvline(0, color=C["blue"], lw=0.8); axes[0].axvline(x_hat, color=C["orange"], lw=0.8)
    axes[0].set_title(f"grid maximum at 0; interpolated maximum at {x_hat:.2f}"); axes[0].legend(fontsize=8); axes[0].set_xlabel("offset from the grid sample")
    X, Y = np.meshgrid(xs, xs)
    Z = 1 - 0.6 * (X - 0.35) ** 2 - 0.4 * (Y + 0.2) ** 2
    ax = axes[1]; ax.contourf(X, Y, Z, levels=20, cmap="viridis")
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            ax.plot(i, j, "o", color="white", ms=6, mec="k")
    ax.plot(0.35, -0.2, "*", color=C["orange"], ms=16, mec="k"); ax.set_title("in 2D (and scale): the fitted peak lies between grid samples"); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "quadratic_localisation")
    return Figure("quadratic_localisation",
                  "Localising the extremum. The DoG is only sampled on a grid, so the true "
                  "peak generally lies between samples. Fitting a quadratic (a Taylor "
                  "expansion) to the samples around the candidate and taking its maximum "
                  "refines the keypoint's position and scale to sub-pixel accuracy.",
                  png=png)


def fig_orientation_assignment(ctx: BuildContext) -> Figure:
    g = load_image("trevi_detail", gray=True, max_side=640).astype(float)
    kp, _ = sift_keypoints(g.astype(np.uint8), n=60)
    kp = sorted(kp, key=lambda k: -k.response)[3]
    x, y, s = int(kp.pt[0]), int(kp.pt[1]), kp.size / 2
    R = int(max(8, 3 * s))
    patch = cv2.GaussianBlur(g, (0, 0), max(s / 2, 0.8))[y - R:y + R + 1, x - R:x + R + 1]
    mag, ang = gradients(patch)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    w = np.exp(-(xx ** 2 + yy ** 2) / (2 * (1.5 * s) ** 2))
    h = orientation_histogram(mag[1:-1, 1:-1], ang[1:-1, 1:-1], 36, w[1:-1, 1:-1])
    hs = np.convolve(np.r_[h[-1], h, h[0]], [1 / 3] * 3, mode="valid")
    peaks = [i for i in range(36) if hs[i] >= 0.8 * hs.max() and hs[i] >= hs[i - 1] and hs[i] >= hs[(i + 1) % 36]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].imshow(patch, cmap="gray"); axes[0].set_title(f"patch of radius 3σ = {R} px around the keypoint"); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(w, cmap="viridis"); axes[1].set_title("Gaussian weight (σ_w = 1.5σ) applied to magnitudes"); axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[2].bar(np.arange(36) * 10, h, width=9, color=C["light"], label="raw"); axes[2].plot(np.arange(36) * 10, hs, color=C["blue"], label="smoothed")
    axes[2].axhline(0.8 * hs.max(), color=C["grey"], ls="--", label="80% of the maximum")
    for p in peaks:
        axes[2].plot(p * 10, hs[p], "v", color=C["red"], ms=9)
    axes[2].set_title(f"36-bin orientation histogram: {len(peaks)} orientation(s) assigned"); axes[2].set_xlabel("gradient angle (°)"); axes[2].legend(fontsize=7)
    png = save_mpl(fig, ctx, "orientation_assignment")
    return Figure("orientation_assignment",
                  "Characteristic orientation in SIFT. Gradient magnitudes in a patch of "
                  "size proportional to the keypoint's σ are weighted by a Gaussian so "
                  "that distant pixels count less, accumulated into a 36-bin histogram "
                  "over 0° to 360°, and smoothed. Every local peak within 80% of the "
                  "global maximum becomes an orientation; a keypoint with two such peaks "
                  "gets two descriptors.", png=png)


def fig_descriptor_layout(ctx: BuildContext) -> Figure:
    g = load_image("trevi_detail", gray=True, max_side=640)
    kp, des = sift_keypoints(g, n=60)
    idx = int(np.argsort([-k.response for k in kp])[3])
    d = des[idx].reshape(4, 4, 8)
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1.1])
    ax = fig.add_subplot(gs[0])
    for i in range(4):
        for j in range(4):
            cx, cy = j + 0.5, 3.5 - i
            for b in range(8):
                a = np.deg2rad(b * 45); L = 0.45 * d[i, j, b] / (d.max() + 1e-9)
                ax.plot([cx, cx + L * np.cos(a)], [cy, cy + L * np.sin(a)], color=C["blue"], lw=1.3)
    for k in range(5):
        ax.axhline(k, color=C["light"]); ax.axvline(k, color=C["light"])
    ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("(a) 4 × 4 cells, 8 orientations each")
    ax = fig.add_subplot(gs[1]); ax.bar(np.arange(128), des[idx], color=[SERIES[(i // 8) % 6] for i in range(128)]); ax.set_title("(b) the 128-element descriptor (16 histograms × 8 bins)"); ax.set_xlabel("element")
    ax = fig.add_subplot(gs[2]); ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_aspect("equal")
    for k in range(5):
        ax.axhline(k, color=C["light"]); ax.axvline(k, color=C["light"])
    px, py = 1.7, 2.3
    ax.plot(px, py, "o", color=C["red"], ms=9)
    for cx, cy in [(1.5, 1.5), (2.5, 1.5), (1.5, 2.5), (2.5, 2.5)]:
        wgt = (1 - abs(px - cx)) * (1 - abs(py - cy))
        ax.plot(cx, cy, "s", color=C["blue"], ms=8); ax.annotate("", xy=(cx, cy), xytext=(px, py), arrowprops=dict(arrowstyle="->", color=C["grey"]))
        ax.text(cx + 0.08, cy + 0.12, f"{wgt:.2f}", fontsize=8, color=C["blue"])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("(c) a pixel's magnitude is shared bilinearly\nbetween the 4 nearest histogram centres")
    png = save_mpl(fig, ctx, "descriptor_layout")
    return Figure("descriptor_layout",
                  "The SIFT descriptor. (a) The patch around the keypoint, rotated to its "
                  "characteristic orientation, is divided into a 4 × 4 grid of cells with "
                  "an 8-bin histogram each, drawn here as arrows. (b) Flattened, the 16 "
                  "histograms form a 128-element vector, which is capped at 0.2 and "
                  "normalised for illumination invariance. (c) Each pixel's magnitude is "
                  "split bilinearly between the four nearest histograms (weights shown), "
                  "and linearly between the two nearest bins.", png=png)


def fig_sift_two_views(ctx: BuildContext) -> Figure:
    a = load_image("trevi", gray=True, max_side=640)
    b = load_image("trevi_detail", gray=True, max_side=640)
    k1, d1 = sift_keypoints(a); k2, d2 = sift_keypoints(b)
    good, _ = ratio_matches(d1, d2, 0.7)
    good = sorted(good, key=lambda m: m.distance)
    vis = draw_matches(a, k1, b, k2, good, max_draw=80)
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.imshow(vis); ax.set_title(f"two photographs of the Trevi fountain by different photographers: {len(good)} SIFT matches (80 drawn)"); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "sift_two_views")
    return Figure("sift_two_views",
                  "SIFT in practice. Two independent photographs of the same fountain, at "
                  "different scales, viewpoints and lighting, are matched by their SIFT "
                  "descriptors. Scale invariance comes from the scale space, rotation "
                  "invariance from the characteristic orientation and illumination "
                  "invariance from descriptor normalisation.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_sift_keypoints_photo(ctx), fig_sift_scale_space(ctx), fig_dog_extrema(ctx),
            fig_quadratic_localisation(ctx), fig_orientation_assignment(ctx), fig_descriptor_layout(ctx),
            fig_sift_two_views(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("3.4 SIFT")
    tab1, tab2 = st.tabs(["Keypoints", "DoG extrema"])
    with tab1:
        key = st.selectbox("image", ["trevi_detail", "notre_dame_b", "galvani", "plate_dutch"])
        g = load_image(key, gray=True, max_side=640)
        n = st.slider("number of keypoints", 50, 2000, 400, 50)
        thr = st.slider("contrast threshold", 0.01, 0.2, 0.04, 0.01)
        sift = cv2.SIFT_create(nfeatures=n, contrastThreshold=thr)
        kp = sift.detect(g, None)
        st.image(cv2.drawKeypoints(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), kp, None, color=(213, 94, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), caption=f"{len(kp)} keypoints", width="stretch")
    with tab2:
        g = load_image("trevi_detail", gray=True, max_side=400)
        gst, sig = gaussian_octave(g); dg = dog_stack(gst) / 255.0
        thr = st.slider("extrema magnitude threshold (fraction of max)", 0.01, 0.3, 0.08, 0.01)
        ex = dog_extrema(dg, thresh=thr)
        fig, ax = plt.subplots(figsize=(7, 5)); ax.imshow(g, cmap="gray")
        for s, r, c in ex:
            ax.add_patch(plt.Circle((c, r), 2.5 * sig[s], fill=False, ec=C["orange"], lw=0.8))
        ax.set_title(f"{len(ex)} extrema"); ax.axis("off"); st.pyplot(fig)
