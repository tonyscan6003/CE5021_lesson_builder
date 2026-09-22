"""Lesson 3.1 — Keypoints and the Laplacian-of-Gaussian blob detector.

Figures
    keypoint_uses        keypoints matched between two photographs of the same building; a panorama
    detector_criteria    a toy found among similar toys at a different scale and rotation (SIFT matches)
    harris_scale         Harris corners on a scene at two scales: different corners are found
    log_kernels          LoG kernels for several sigma, in 2D and as profiles
    log_blob_response    synthetic dots convolved with one LoG kernel; the normalised square response
    log_multi_scale      blobs detected at their own scale with a family of LoG kernels (interactive)
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import ndimage
from skimage.feature import blob_log, corner_harris, corner_peaks

from .._common import (BuildContext, Figure, C, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, load_image, skimage_image, norm01, to_uint8)
from ._sift_common import sift_keypoints, ratio_matches, draw_matches

FIGURES = ["keypoint_uses", "detector_criteria", "harris_scale", "log_kernels",
           "log_blob_response", "log_multi_scale"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def log_kernel(sigma: float, size: int | None = None) -> np.ndarray:
    """Normalised Laplacian of Gaussian, sigma^2 * (Gxx + Gyy), sampled on a grid."""
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return ((x ** 2 + y ** 2 - 2 * sigma ** 2) / sigma ** 4) * g * sigma ** 2


def dots_image(n: int = 256, seed: int = 1) -> tuple[np.ndarray, list]:
    """Dark background with bright discs of three radii; returns (image, [(x, y, r), ...])."""
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n), float)
    specs = []
    for r in [5, 10, 20]:
        for _ in range(4):
            for _try in range(50):
                x, y = rng.integers(r + 4, n - r - 4, 2)
                if all((x - a) ** 2 + (y - b) ** 2 > (r + c + 6) ** 2 for a, b, c in specs):
                    break
            cv2.circle(img, (int(x), int(y)), r, 1.0, -1)
            specs.append((int(x), int(y), r))
    return img, specs


def normalised_log_response(img: np.ndarray, sigma: float) -> np.ndarray:
    return sigma ** 2 * ndimage.gaussian_laplace(img.astype(float), sigma)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_keypoint_uses(ctx: BuildContext) -> Figure:
    a = load_image("notre_dame_a", gray=True, max_side=560)
    b = load_image("notre_dame_b", gray=True, max_side=560)
    pano = load_image("liege_panorama", max_side=1400)
    k1, d1 = sift_keypoints(a); k2, d2 = sift_keypoints(b)
    good, _ = ratio_matches(d1, d2, 0.7)
    good = sorted(good, key=lambda m: m.distance)
    vis = draw_matches(a, k1, b, k2, good, max_draw=60)
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1])
    ax = fig.add_subplot(gs[0]); ax.imshow(vis); ax.set_title(f"(a) {len(good)} keypoints matched between two photographs of Notre-Dame ({min(60, len(good))} drawn)")
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1]); ax.imshow(pano); ax.set_title("(b) a panorama: five photographs aligned by matching their keypoints")
    ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "keypoint_uses")
    return Figure("keypoint_uses",
                  "Uses of keypoints. (a) The same interest points are detected in two "
                  "different photographs of the same façade and matched by their "
                  "descriptors, verifying that the images show the same object. (b) "
                  "Corresponding keypoints in overlapping photographs allow them to be "
                  "aligned into a panorama.", png=png)


def fig_detector_criteria(ctx: BuildContext) -> Figure:
    rgb = load_image("baby_toys", max_side=720)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    # template: the 2x2 block of a 5x5 grid with the most SIFT keypoints (a whole toy), cut from the scene
    kp_all, _ = sift_keypoints(g)
    n = 6
    counts = np.zeros((n, n), int)
    for k in kp_all:
        counts[min(n - 1, int(k.pt[1] * n / h)), min(n - 1, int(k.pt[0] * n / w))] += 1
    bi, bj = np.unravel_index(np.argmax(counts[1:-1, 1:-1]), (n - 2, n - 2)); bi += 1; bj += 1
    ch, cw = h / n, w / n
    tpl = g[int((bi - 0.3) * ch):int((bi + 1.3) * ch), int((bj - 0.3) * cw):int((bj + 1.3) * cw)]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 25, 0.7)
    scene = cv2.warpAffine(g, M, (w, h), borderValue=90)
    k1, d1 = sift_keypoints(tpl); k2, d2 = sift_keypoints(scene)
    good, _ = ratio_matches(d1, d2, 0.7)
    vis = draw_matches(tpl, k1, scene, k2, sorted(good, key=lambda m: m.distance), max_draw=50)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(vis); ax.set_title(f"a toy cut from the scene (left) located after the scene is rotated 25° and scaled 0.7: {len(good)} matches")
    ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "detector_criteria")
    return Figure("detector_criteria",
                  "What a good feature detector achieves. A small region containing one "
                  "toy is matched into a scene in which it appears smaller, rotated and "
                  "surrounded by similar objects. The detector is local (many keypoints "
                  "survive the clutter), invariant to scale and rotation, and accurate "
                  "enough that the matches land on the same physical points.", png=png)


def fig_harris_scale(ctx: BuildContext) -> Figure:
    rgb = load_image("balloons", max_side=640)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(g, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    R = corner_harris(g / 255.0, method="k", k=0.05, sigma=2)
    pk1 = corner_peaks(R, min_distance=5, threshold_rel=0.02)
    pk2 = corner_peaks(corner_harris(small / 255.0, method="k", k=0.05, sigma=2), min_distance=5, threshold_rel=0.02)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw=dict(width_ratios=[1.3, 1.3, 0.6]))
    axes[0].imshow(g, vmin=0, vmax=255); axes[0].plot(pk1[:, 1], pk1[:, 0], "o", mfc="none", mec=C["red"], ms=4)
    axes[0].set_title(f"full size: {len(pk1)} Harris corners on lettering and stripes")
    Rd = np.sign(R) * np.sqrt(np.abs(R))          # compress the range so weak negative (edge) responses are visible
    v = np.percentile(np.abs(Rd), 99.5); axes[1].imshow(Rd, cmap="RdBu_r", vmin=-v, vmax=v); axes[1].set_title("corner response (signed √|R|): balloon outlines are edges (blue)")
    axes[2].imshow(small, vmin=0, vmax=255); axes[2].plot(pk2[:, 1], pk2[:, 0], "o", mfc="none", mec=C["red"], ms=4); axes[2].set_title(f"0.4 scale: {len(pk2)} corners")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "harris_scale")
    return Figure("harris_scale",
                  "The Harris detector is not scale invariant. With its fixed window it "
                  "finds small corners in the lettering and stripes of the balloons, "
                  "while the large curved outline of each balloon registers only as an "
                  "edge in the response map. Shrinking the image changes which corners "
                  "are found, so the same keypoints cannot be relied on at two scales.",
                  png=png)


def fig_log_kernels(ctx: BuildContext) -> Figure:
    sigmas = [2, 4, 8]
    fig = plt.figure(figsize=(12, 6.2))
    for j, s in enumerate(sigmas):
        k = log_kernel(s)
        ax = fig.add_subplot(2, 3, j + 1); ax.imshow(k, cmap="RdBu_r", vmin=-np.abs(k).max(), vmax=np.abs(k).max())
        ax.set_title(f"LoG kernel, σ = {s}: {k.shape[0]} × {k.shape[0]} pixels"); ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(2, 3, j + 4, projection="3d")
        r = k.shape[0] // 2; X, Y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        ax.plot_surface(X, Y, -k, cmap="viridis", linewidth=0); ax.set_title(f"−LoG, σ = {s} (surface)"); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    png = save_mpl(fig, ctx, "log_kernels")
    return Figure("log_kernels",
                  "Laplacian of Gaussian (LoG) kernels: the Laplacian operator applied to a "
                  "Gaussian. Each is a dark centre surrounded by a bright ring (shown "
                  "inverted as a surface), and the ring radius grows with σ. Convolving an "
                  "image with a LoG kernel responds most strongly to circular blobs of the "
                  "matching size.", png=png)


def fig_log_blob_response(ctx: BuildContext) -> Figure:
    img, specs = dots_image()
    sigma = 6.0
    resp = -normalised_log_response(img, sigma)          # bright blobs give negative LoG; flip
    sq = norm01(np.maximum(resp, 0) ** 2)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("(a) discs of radius 5, 10 and 20")
    axes[1].imshow(resp, cmap="gray"); axes[1].set_title(f"(b) image ∗ LoG, σ = {sigma:.0f}")
    axes[2].imshow(sq, cmap="inferno"); axes[2].set_title("(c) normalised square response")
    for x, y, r in specs:
        if r == 10:
            axes[2].add_patch(plt.Circle((x, y), r + 4, fill=False, ec=C["sky"], lw=1.2))
    k = log_kernel(sigma); axes[3].imshow(-k, cmap="gray"); axes[3].set_title("(d) the kernel: dark centre, bright ring")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "log_blob_response")
    return Figure("log_blob_response",
                  "Blob detection with one LoG kernel. Convolving the disc image (a) with a "
                  "σ = 6 kernel gives (b); squaring and normalising (c) shows that the "
                  "strongest responses sit on the discs of radius 10 (circled), whose size "
                  "matches the kernel (radius ≈ √2 σ). The kernel (d) is itself a bright "
                  "ring around a dark centre, so the response is largest where the image "
                  "looks like the kernel.", png=png)


def fig_log_multi_scale(ctx: BuildContext) -> Figure:
    img, specs = dots_image()
    blobs = blob_log(img, min_sigma=2, max_sigma=20, num_sigma=15, threshold=0.05)
    hub = skimage_image("hubble_deep_field", gray=True)[100:600, 150:750]
    hb = blob_log(hub / 255.0, min_sigma=1.5, max_sigma=15, num_sigma=12, threshold=0.06)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].imshow(img, cmap="gray")
    for y, x, s in blobs:
        axes[0].add_patch(plt.Circle((x, y), s * np.sqrt(2), fill=False, ec=C["orange"], lw=1.4))
    axes[0].set_title(f"(a) {len(blobs)} blobs found over σ = 2 … 20; circle radius = √2 σ")
    axes[1].imshow(hub, cmap="gray")
    for y, x, s in hb:
        axes[1].add_patch(plt.Circle((x, y), s * np.sqrt(2), fill=False, ec=C["orange"], lw=1.0))
    axes[1].set_title(f"(b) {len(hb)} galaxies of different sizes in the Hubble deep field")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "log_multi_scale")

    sigmas = [3, 4.5, 7, 10, 14, 20]
    frames = []
    for s in sigmas:
        r = norm01(np.maximum(-normalised_log_response(img, s), 0) ** 2)
        frames.append(go.Frame(name=str(s), data=[go.Image(source=to_data_uri(to_uint8(r)))],
                               layout=go.Layout(title=f"normalised square LoG response at σ = {s} (discs of radius ≈ {s * 1.41:.0f} respond)")))
    fig2 = go.Figure(data=frames[1].data, frames=frames)
    fig2.update_layout(**slider_frames_layout(sigmas, "σ = "), title=frames[1].layout.title.text)
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    return Figure("log_multi_scale",
                  "The scale-invariant blob detector: convolve with LoG kernels of many "
                  "σ, take the normalised square response of each, and find the maxima. "
                  "Each blob is found at the σ matching its size, so the circles drawn "
                  "with radius √2 σ fit the discs (a) and the galaxies (b). Use the "
                  "slider to see which discs respond at each σ.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_keypoint_uses(ctx), fig_detector_criteria(ctx), fig_harris_scale(ctx),
            fig_log_kernels(ctx), fig_log_blob_response(ctx), fig_log_multi_scale(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("3.1 Keypoints and LoG blob detection")
    tab1, tab2 = st.tabs(["LoG response at one scale", "Multi-scale blob detection"])
    img, specs = dots_image()
    with tab1:
        s = st.slider("σ", 1.0, 20.0, 6.0, 0.5)
        r = norm01(np.maximum(-normalised_log_response(img, s), 0) ** 2)
        c1, c2 = st.columns(2)
        c1.image(to_uint8(img), caption="discs of radius 5, 10, 20", width="stretch")
        c2.image(to_uint8(r), caption=f"normalised square response, σ = {s} (radius ≈ {1.41 * s:.1f})", width="stretch")
    with tab2:
        key = st.selectbox("image", ["dots", "hubble deep field", "monarch", "fruit"])
        if key == "dots":
            g = img
        elif key == "hubble deep field":
            g = skimage_image("hubble_deep_field", gray=True)[100:600, 150:750] / 255.0
        else:
            g = load_image(key, gray=True, max_side=520) / 255.0
        smin, smax = st.slider("σ range", 1.0, 30.0, (2.0, 15.0))
        thr = st.slider("threshold", 0.01, 0.3, 0.06, 0.01)
        blobs = blob_log(g, min_sigma=smin, max_sigma=smax, num_sigma=12, threshold=thr)
        fig, ax = plt.subplots(figsize=(7, 5)); ax.imshow(g, cmap="gray")
        for y, x, s in blobs:
            ax.add_patch(plt.Circle((x, y), s * np.sqrt(2), fill=False, ec=C["orange"], lw=1.2))
        ax.set_title(f"{len(blobs)} blobs"); ax.axis("off"); st.pyplot(fig)
