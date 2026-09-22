"""Lesson 3.2 — Scale spaces and scale-invariant detection.

Figures
    gaussian_scale_space   an image at increasing sigma: fine structure disappears (interactive)
    characteristic_scale   normalised Laplacian at one image point as a function of sigma
    laplacian_kernels      second-derivative Gaussian kernels and their sum, the LoG
    scale_invariance       the same feature at two zooms peaks at sigma and 2 sigma
    maxima_26              the 26-neighbour search in (x, y, scale)
    harris_vs_laplacian    Laplacian responds to edges too; the Harris response rejects them
    dog_vs_log             difference of Gaussians approximates the normalised LoG
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import ndimage
from skimage.feature import corner_harris

from .._common import (BuildContext, Figure, C, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, load_image, norm01, to_uint8)

FIGURES = ["gaussian_scale_space", "characteristic_scale", "laplacian_kernels", "scale_invariance",
           "maxima_26", "harris_vs_laplacian", "dog_vs_log"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def owl_gray(max_side=520):
    return load_image("owl", gray=True, max_side=max_side).astype(float)


def norm_laplacian(img, sigma):
    return sigma ** 2 * ndimage.gaussian_laplace(img, sigma)


def pick_scale_point(stack, sigmas, s_lo=4.0, s_hi=16.0):
    """Choose a point in the upper half of the image whose normalised-Laplacian
    response over scale is a clean single peak at a mid-range sigma: the
    strongest response among scales in [s_lo, s_hi], measured relative to the
    point's response at the smallest scale (so tiny texture does not win)."""
    h = stack.shape[1] // 2
    sel = (sigmas >= s_lo) & (sigmas <= s_hi)
    score = stack[sel][:, :h, :].max(axis=0) - 1.5 * stack[0, :h, :]
    r, c = np.unravel_index(np.argmax(score), score.shape)
    curve = stack[:, r, c]
    return r, c, sigmas[np.argmax(curve)], curve


def log_axis(ax):
    ticks = [1.5, 2, 3, 5, 8, 12, 20, 30]
    ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()


def gauss_2nd_kernels(sigma, r=None):
    r = r or int(3 * sigma)
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    gxx = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * g
    gyy = (y ** 2 / sigma ** 4 - 1 / sigma ** 2) * g
    return gxx, gyy


def log_1d(x, s):
    return s ** 2 * (x ** 2 / s ** 4 - 1 / s ** 2) * np.exp(-x ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)


def dog_1d(x, s, rho):
    g = lambda ss: np.exp(-x ** 2 / (2 * ss ** 2)) / (np.sqrt(2 * np.pi) * ss)  # noqa: E731
    return (g(rho * s) - g(s)) / (rho - 1)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_gaussian_scale_space(ctx: BuildContext) -> Figure:
    g = owl_gray()
    sigmas = [0, 2, 4, 8, 16]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, s in zip(axes, sigmas):
        im = cv2.GaussianBlur(g, (0, 0), s) if s else g
        ax.imshow(im, vmin=0, vmax=255); ax.set_title(f"L(x; σ = {s})"); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "gaussian_scale_space")
    ss = [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24]
    frames = [go.Frame(name=str(s), data=[go.Image(source=to_data_uri(to_uint8(cv2.GaussianBlur(g, (0, 0), s)), "jpg"))],
                       layout=go.Layout(title=f"Gaussian scale space, σ = {s}")) for s in ss]
    fig2 = go.Figure(data=frames[2].data, frames=frames)
    fig2.update_layout(**slider_frames_layout(ss, "σ = "), title=frames[2].layout.title.text)
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    return Figure("gaussian_scale_space",
                  "A Gaussian scale space: the image convolved with Gaussians of increasing "
                  "σ. Each level L(x; σ) keeps only structure larger than about σ, so the "
                  "feather texture vanishes first and the head and eyes survive longest. "
                  "The Gaussian is used because it introduces no artefacts.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def fig_characteristic_scale(ctx: BuildContext) -> Figure:
    g = owl_gray()
    sigmas = np.geomspace(1.5, 30, 28)
    stack = np.array([np.abs(norm_laplacian(g, s)) for s in sigmas])
    r, c, s_star, curve = pick_scale_point(stack, sigmas)
    fig = plt.figure(figsize=(13, 6.4))
    gs = fig.add_gridspec(2, 4)
    for j, s in enumerate([2, 5, 10, 20]):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(norm01(norm_laplacian(g, s) ** 2), cmap="inferno"); ax.set_title(f"(σ² ∇²L)², σ = {s}")
        ax.add_patch(plt.Rectangle((c - 30, r - 30), 60, 60, fill=False, ec=C["yellow"], lw=1.5)); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :2]); ax.imshow(g, vmin=0, vmax=255); ax.plot(c, r, "+", color=C["yellow"], ms=14, mew=2)
    ax.add_patch(plt.Circle((c, r), s_star * np.sqrt(2), fill=False, ec=C["yellow"], lw=1.5)); ax.set_title(f"the point studied, with a circle of radius √2 σ* = {s_star * 1.41:.0f} px"); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 2:]); ax.plot(sigmas, curve, "o-", color=C["blue"], ms=4); ax.axvline(s_star, color=C["red"], ls="--")
    log_axis(ax); ax.set_xlabel("σ"); ax.set_ylabel("|σ² ∇²L| at the point"); ax.set_title(f"one maximum over scale, at σ* = {s_star:.1f}")
    png = save_mpl(fig, ctx, "characteristic_scale")
    return Figure("characteristic_scale",
                  "Characteristic scale. Top: the squared normalised Laplacian σ²∇²L of the "
                  "owl at four scales; inside the yellow box the response grows, peaks and "
                  "fades. Bottom right: at the marked point the normalised Laplacian has a "
                  "single maximum over σ. That σ* is the characteristic scale of the "
                  "structure there, and is related to its size in pixels (circle).", png=png)


def fig_laplacian_kernels(ctx: BuildContext) -> Figure:
    gxx, gyy = gauss_2nd_kernels(3.0)
    fig = plt.figure(figsize=(12, 6))
    for j, (k, name) in enumerate([(gxx, "∂²G/∂x²"), (gyy, "∂²G/∂y²"), (gxx + gyy, "∂²G/∂x² + ∂²G/∂y² = LoG")]):
        v = np.abs(k).max()
        ax = fig.add_subplot(2, 3, j + 1); ax.imshow(k, cmap="RdBu_r", vmin=-v, vmax=v); ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(2, 3, j + 4, projection="3d")
        r = k.shape[0] // 2; X, Y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        ax.plot_surface(X, Y, k, cmap="viridis", linewidth=0); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    png = save_mpl(fig, ctx, "laplacian_kernels")
    return Figure("laplacian_kernels",
                  "The Laplacian scale space is built from second-order Gaussian "
                  "derivative kernels. Each second derivative has three extrema; summing "
                  "the x and y kernels gives the Laplacian of Gaussian seen in the previous "
                  "lesson, now derived from the principle of a function with a single "
                  "maximum over scale.", png=png)


def fig_scale_invariance(ctx: BuildContext) -> Figure:
    g = owl_gray(400)
    h, w = g.shape
    zoom = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    sigmas = np.geomspace(1.5, 40, 30)
    stack = np.array([np.abs(norm_laplacian(g, s)) for s in sigmas])
    r, c, _, _ = pick_scale_point(stack, sigmas, s_lo=3.0, s_hi=12.0)
    c1 = stack[:, r, c]
    c2 = np.array([abs(norm_laplacian(zoom, s)[2 * r, 2 * c]) for s in sigmas])
    s1, s2 = sigmas[np.argmax(c1)], sigmas[np.argmax(c2)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    axes[0].imshow(g, vmin=0, vmax=255); axes[0].add_patch(plt.Circle((c, r), s1 * 1.41, fill=False, ec=C["yellow"], lw=1.5)); axes[0].set_title(f"original: σ* = {s1:.1f}")
    zh, zw = zoom.shape
    y0, x0 = max(0, min(2 * r - 200, zh - 400)), max(0, min(2 * c - 200, zw - 400))
    axes[1].imshow(zoom[y0:y0 + 400, x0:x0 + 400], vmin=0, vmax=255)
    axes[1].add_patch(plt.Circle((2 * c - x0, 2 * r - y0), s2 * 1.41, fill=False, ec=C["yellow"], lw=1.5)); axes[1].set_title(f"zoomed ×2 (crop): σ* = {s2:.1f}")
    axes[2].plot(sigmas, c1 / c1.max(), "o-", color=C["blue"], ms=3, label="original"); axes[2].plot(sigmas, c2 / c2.max(), "s-", color=C["orange"], ms=3, label="zoomed ×2")
    log_axis(axes[2]); axes[2].set_xlabel("σ"); axes[2].set_ylabel("normalised Laplacian (scaled)"); axes[2].legend(fontsize=8)
    axes[2].set_title(f"the peak moves from σ = {s1:.1f} to {s2:.1f}: ratio {s2 / s1:.2f}")
    for ax in axes[:2]:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "scale_invariance")
    return Figure("scale_invariance",
                  "Scale invariance arises naturally in scale space. The same image "
                  "feature reaches its normalised-Laplacian maximum at σ* in the original "
                  "and at about 2σ* after the image is enlarged twofold. Because the "
                  "characteristic scale tracks the feature's size, the same keypoint is "
                  "found in both images, each at its own scale.", png=png)


def fig_maxima_26(ctx: BuildContext) -> Figure:
    fig = plt.figure(figsize=(11, 4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for z, col in zip([0, 1, 2], [C["light"], C["sky"], C["light"]]):
        for i in range(3):
            for j in range(3):
                centre = (i == 1 and j == 1 and z == 1)
                ax.bar3d(i, j, z, 0.9, 0.9, 0.9, color=C["red"] if centre else col, alpha=1.0 if centre else 0.2, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("scale"); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([0.5, 1.5, 2.5]); ax.set_zticklabels(["σ below", "σ", "σ above"])
    ax.set_title("the candidate (red) and its 26 neighbours")
    ax = fig.add_subplot(1, 2, 2); ax.axis("off")
    ax.text(0, 0.9, "A point is a keypoint if its normalised Laplacian is larger\nthan all 26 neighbours:", fontsize=10, va="top")
    ax.text(0.05, 0.65, "•  8 neighbours in the same scale\n•  9 in the scale below\n•  9 in the scale above", fontsize=10, va="top")
    ax.text(0, 0.3, "This finds the spatial position and the scale\nof each feature in one search.", fontsize=10, va="top")
    png = save_mpl(fig, ctx, "maxima_26")
    return Figure("maxima_26",
                  "Local maxima in scale and space are found with a nearest-neighbour "
                  "search over 26 neighbours: the 8 surrounding pixels at the same scale "
                  "and the 9 pixels at each of the neighbouring scales.", png=png)


def fig_harris_vs_laplacian(ctx: BuildContext) -> Figure:
    g = load_image("monarch", gray=True, max_side=520).astype(float)
    s = 4.0
    lap = norm_laplacian(g, s) ** 2
    R = corner_harris(g / 255.0, method="k", k=0.05, sigma=s)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].imshow(g, vmin=0, vmax=255); axes[0].set_title("image")
    axes[1].imshow(norm01(lap), cmap="inferno"); axes[1].set_title(f"squared normalised Laplacian, σ = {s:.0f}: blobs and edges")
    v = np.percentile(np.abs(R), 99.5); axes[2].imshow(R, cmap="RdBu_r", vmin=-v, vmax=v); axes[2].set_title("Harris response: edges negative (blue)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "harris_vs_laplacian")
    return Figure("harris_vs_laplacian",
                  "Why combine Harris with the Laplacian. The Laplacian responds to edges "
                  "of a similar scale as well as to blobs, so the wing veins and outline "
                  "light up along with the spots. The Harris response marks those edges "
                  "negative. The Harris-Laplacian detector requires a spatial maximum of "
                  "the Harris response and a scale maximum of the Laplacian, giving fewer "
                  "but more robust keypoints.", png=png)


def fig_dog_vs_log(ctx: BuildContext) -> Figure:
    x = np.linspace(-15, 15, 600)
    s = 3.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(x, log_1d(x, s), color=C["ink"], lw=2, label="normalised LoG, σ² ∂²G/∂x²")
    for rho, col in [(2.0, C["orange"]), (1.4, C["green"]), (1.1, C["blue"])]:
        axes[0].plot(x, dog_1d(x, s, rho) / (rho - 1) * (rho - 1), "--", color=col, label=f"DoG, ρ = {rho}")
    axes[0].axhline(0, color="k", lw=0.5); axes[0].legend(fontsize=8); axes[0].set_title("difference of Gaussians approaches σ ∂G/∂σ ∝ normalised LoG")
    rhos = np.linspace(1.02, 2.5, 60)
    err = [np.abs(dog_1d(x, s, r) - log_1d(x, s)).max() / np.abs(log_1d(x, s)).max() for r in rhos]
    axes[1].plot(rhos, err, color=C["red"]); axes[1].set_xlabel("ρ"); axes[1].set_ylabel("relative peak error"); axes[1].set_title("approximation error falls to zero as ρ → 1")
    axes[1].set_ylim(0, None)
    png = save_mpl(fig, ctx, "dog_vs_log")
    return Figure("dog_vs_log",
                  "The difference of Gaussians. Subtracting a Gaussian of width σ from one "
                  "of width ρσ and scaling by 1/(ρ − 1) approximates σ ∂G/∂σ, which equals "
                  "the normalised Laplacian of Gaussian. The approximation improves as ρ "
                  "approaches 1. Because a Gaussian scale space can be built cheaply, the "
                  "DoG is a fast substitute for the Laplacian scale space.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_gaussian_scale_space(ctx), fig_characteristic_scale(ctx), fig_laplacian_kernels(ctx),
            fig_scale_invariance(ctx), fig_maxima_26(ctx), fig_harris_vs_laplacian(ctx), fig_dog_vs_log(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("3.2 Scale space")
    tab1, tab2 = st.tabs(["Normalised Laplacian at a point", "Difference of Gaussians"])
    with tab1:
        g = owl_gray()
        h, w = g.shape
        c1, c2 = st.columns(2)
        r = c1.slider("row", 0, h - 1, h // 3); c = c2.slider("column", 0, w - 1, w // 2)
        sigmas = np.geomspace(1.5, 30, 24)
        curve = [abs(norm_laplacian(g, s)[r, c]) for s in sigmas]
        s_star = sigmas[int(np.argmax(curve))]
        a, b = st.columns(2)
        fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(g, cmap="gray"); ax.plot(c, r, "+", color="yellow", ms=14, mew=2)
        ax.add_patch(plt.Circle((c, r), s_star * 1.41, fill=False, ec="yellow")); ax.axis("off"); a.pyplot(fig)
        fig, ax = plt.subplots(figsize=(5, 4)); ax.plot(sigmas, curve, "o-"); ax.set_xscale("log"); ax.set_xlabel("σ"); ax.set_title(f"σ* = {s_star:.1f}"); b.pyplot(fig)
    with tab2:
        rho = st.slider("ρ", 1.05, 3.0, 1.6, 0.05)
        x = np.linspace(-15, 15, 600)
        fig, ax = plt.subplots(figsize=(7, 3)); ax.plot(x, log_1d(x, 3.0) / 3.0, color="k", label="normalised LoG / σ"); ax.plot(x, dog_1d(x, 3.0, rho), "--", color=C["orange"], label=f"DoG, ρ = {rho}")
        ax.legend(); st.pyplot(fig)
