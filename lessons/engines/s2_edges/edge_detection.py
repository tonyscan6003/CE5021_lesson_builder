"""Lesson 2.1 — Edge detection with derivative kernels.

Figures
    edge_profile         a dark band, its intensity profile and first/second derivatives
    noise_edges          noise creates false edges; smoothing first helps, too much smoothing hurts
    prewitt_kernels      the two Prewitt kernels and their response to vertical / horizontal edges
    gradient_photo       Gx, Gy, magnitude and direction of a photograph; offset vs magnitude display
    sobel_decomposition  Sobel = central difference x averaging
    gaussian_derivative  Gaussian derivative kernels and edges at several scales (interactive)
    laplacian            second derivative, the Laplacian kernel and its double response
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import convolve2d

from .._common import (BuildContext, Figure, C, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, load_image, sync_matrix, to_uint8, norm01)

FIGURES = ["edge_profile", "noise_edges", "prewitt_kernels", "gradient_photo",
           "sobel_decomposition", "gaussian_derivative", "laplacian"]

PREWITT_X = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], float)
PREWITT_Y = PREWITT_X.T.copy()
SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float)
SOBEL_Y = SOBEL_X.T.copy()
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def conv(img, k):
    return convolve2d(img.astype(float), k, mode="same", boundary="symm")


def gradient(img, kx=SOBEL_X, ky=SOBEL_Y):
    gx, gy = conv(img, kx), conv(img, ky)
    return gx, gy, np.hypot(gx, gy), np.arctan2(gy, gx)


def gauss_deriv_kernels(sigma: float):
    r = int(3 * sigma)
    x = np.arange(-r, r + 1, dtype=float)
    g = np.exp(-x ** 2 / (2 * sigma ** 2)); g /= g.sum()
    dg = -x / sigma ** 2 * g
    return np.outer(g, dg), np.outer(dg, g)   # d/dx, d/dy


def photo(key="notre_dame_b", max_side=480):
    return load_image(key, gray=True, max_side=max_side)


def direction_rgb(mag, ang):
    """Colour-code gradient direction by hue, magnitude by value."""
    h = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    v = to_uint8(np.clip(mag / np.percentile(mag, 99), 0, 1))
    hsv = np.dstack([h, np.full_like(h, 255), v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_edge_profile(ctx: BuildContext) -> Figure:
    n = 120
    x = np.arange(n)
    prof = 220 - 180 / (1 + np.exp(-(x - 40) / 2.0)) + 180 / (1 + np.exp(-(x - 80) / 2.0))
    img = np.tile(prof, (60, 1))
    d1 = np.gradient(prof)
    d2 = np.gradient(d1)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
    axes[0].imshow(img, vmin=0, vmax=255); axes[0].set_title("(a) dark vertical band"); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].plot(x, prof, color=C["ink"]); axes[1].set_title("(b) intensity along a row"); axes[1].set_ylabel("intensity")
    axes[2].plot(x, d1, color=C["blue"]); axes[2].axhline(0, color="k", lw=0.5); axes[2].set_title("(c) first derivative")
    axes[2].annotate("falling edge: negative", xy=(40, d1.min()), xytext=(2, d1.min() * 0.9), fontsize=7, color=C["red"])
    axes[2].annotate("rising edge: positive", xy=(80, d1.max()), xytext=(60, d1.max() * 0.9), fontsize=7, color=C["green"])
    axes[3].plot(x, d2, color=C["orange"]); axes[3].axhline(0, color="k", lw=0.5); axes[3].set_title("(d) second derivative: zero crossings at the edges")
    for ax in axes[1:]:
        ax.set_xlabel("column")
    png = save_mpl(fig, ctx, "edge_profile")
    return Figure("edge_profile",
                  "An edge is a rapid change of intensity over a few pixels. (b) Across a "
                  "dark band the profile steps down and then up. (c) The first derivative "
                  "responds with a negative peak at the falling edge and a positive peak "
                  "at the rising edge. (d) The second derivative crosses zero exactly at "
                  "each edge.", png=png)


def fig_noise_edges(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(0)
    g = photo("notre_dame_b", 420).astype(float)
    noisy = np.clip(g + rng.normal(0, 20, g.shape), 0, 255)
    panels = [("noisy image", noisy)]
    for s in [0, 1.5, 5]:
        im = cv2.GaussianBlur(noisy, (0, 0), s) if s else noisy
        panels.append((f"Prewitt magnitude, smoothing σ = {s}", gradient(im, PREWITT_X, PREWITT_Y)[2]))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (t, im) in zip(axes, panels):
        vmax = 255 if t.startswith("noisy") else np.percentile(im, 99.5)
        ax.imshow(im, vmin=0, vmax=vmax); ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "noise_edges")
    return Figure("noise_edges",
                  "Derivative filters respond to any rapid change, so noise produces a "
                  "carpet of false edges (second panel). Smoothing with a Gaussian first "
                  "removes them (third), but too much smoothing also blurs the real edges "
                  "and makes them harder to detect (fourth).", png=png)


def fig_prewitt_kernels(ctx: BuildContext) -> Figure:
    vert = np.array([[10, 10, 80, 80], [10, 10, 80, 80], [10, 10, 80, 80], [10, 10, 80, 80]])
    horz = vert.T.copy()
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6))
    sync_matrix(axes[0, 0], PREWITT_X, cmap="Oranges", title="X kernel (central difference along rows)")
    sync_matrix(axes[0, 1], vert, title="patch with a vertical edge")
    sync_matrix(axes[0, 2], convolve2d(vert, PREWITT_X, mode="valid"), cmap="Purples", title="X kernel ∗ patch: responds")
    sync_matrix(axes[0, 3], convolve2d(vert, PREWITT_Y, mode="valid"), cmap="Purples", title="Y kernel ∗ patch: zero")
    sync_matrix(axes[1, 0], PREWITT_Y, cmap="Oranges", title="Y kernel (central difference down columns)")
    sync_matrix(axes[1, 1], horz, title="patch with a horizontal edge")
    sync_matrix(axes[1, 2], convolve2d(horz, PREWITT_Y, mode="valid"), cmap="Purples", title="Y kernel ∗ patch: responds")
    sync_matrix(axes[1, 3], convolve2d(horz, PREWITT_X, mode="valid"), cmap="Purples", title="X kernel ∗ patch: zero")
    png = save_mpl(fig, ctx, "prewitt_kernels")
    return Figure("prewitt_kernels",
                  "The Prewitt kernels repeat the 1D central difference f[n+1] − f[n−1] "
                  "three times. The X-direction kernel detects vertical edges and gives "
                  "nothing on a horizontal edge; the Y-direction kernel does the opposite.",
                  png=png)


def fig_gradient_photo(ctx: BuildContext) -> Figure:
    g = photo("notre_dame_b", 480)
    gx, gy, mag, ang = gradient(g, PREWITT_X, PREWITT_Y)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    panels = [("image", g, "gray", 0, 255), ("|Gx| (X kernel): vertical edges", np.abs(gx), "gray", 0, None),
              ("|Gy| (Y kernel): horizontal edges", np.abs(gy), "gray", 0, None),
              ("magnitude √(Gx² + Gy²)", mag, "gray", 0, None),
              ("Gx with offset: sign kept", gx, "gray", -np.percentile(np.abs(gx), 99), np.percentile(np.abs(gx), 99)),
              ("direction atan2(Gy, Gx), colour = angle", direction_rgb(mag, ang), None, None, None)]
    for ax, (t, im, cm, vmin, vmax) in zip(axes.ravel(), panels):
        if cm is None:
            ax.imshow(im)
        else:
            ax.imshow(im, cmap=cm, vmin=vmin, vmax=vmax if vmax is not None else np.percentile(im, 99.5))
        ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "gradient_photo")
    return Figure("gradient_photo",
                  "Prewitt filtering of a photograph. Gx responds to vertical structure and "
                  "Gy to horizontal structure; the two combine into a gradient magnitude "
                  "and a direction. Displaying Gx with an offset (bottom middle) keeps the "
                  "sign, so rising edges are white and falling edges black; the magnitude "
                  "image makes every edge bright but loses the sign.", png=png)


def fig_sobel_decomposition(ctx: BuildContext) -> Figure:
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    sync_matrix(axes[0], SOBEL_X, cmap="Oranges", title="Sobel X")
    axes[1].axis("off"); axes[1].text(0.5, 0.5, "=", fontsize=28, ha="center", va="center")
    sync_matrix(axes[2], np.array([[1], [2], [1]]), cmap="Greens", title="averaging [1 2 1]ᵀ")
    sync_matrix(axes[3], np.array([[1, 0, -1]]), cmap="Purples", title="⊗ gradient [1 0 −1]")
    png = save_mpl(fig, ctx, "sobel_decomposition")
    return Figure("sobel_decomposition",
                  "The Sobel X kernel is the outer product of a vertical averaging kernel "
                  "and the horizontal central difference. The averaging smooths along the "
                  "edge before differentiating across it, which makes Sobel less sensitive "
                  "to noise than Prewitt. The Y kernel is the transpose.", png=png)


def fig_gaussian_derivative(ctx: BuildContext) -> Figure:
    g = photo("galvani", 520).astype(float)
    x = np.linspace(-6, 6, 400)
    s = 1.5
    gauss = np.exp(-x ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0]); ax.plot(x, gauss, color=C["blue"], label="G(x)"); ax.plot(x, -x / s ** 2 * gauss, color=C["red"], label="dG/dx")
    ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=8); ax.set_title(f"1D Gaussian and derivative, σ = {s}")
    kx, ky = gauss_deriv_kernels(2.0)
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(kx, cmap="RdBu_r"); ax.set_title("2D kernel ∂G/∂x (σ = 2)"); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(ky, cmap="RdBu_r"); ax.set_title("2D kernel ∂G/∂y (σ = 2)"); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 3]); ax.imshow(g, vmin=0, vmax=255); ax.set_title("image"); ax.set_xticks([]); ax.set_yticks([])
    for j, sig in enumerate([1, 2, 4, 8]):
        kx, ky = gauss_deriv_kernels(sig)
        mag = np.hypot(conv(g, kx), conv(g, ky))
        ax = fig.add_subplot(gs[1, j]); ax.imshow(mag, vmin=0, vmax=np.percentile(mag, 99.5)); ax.set_title(f"Gaussian-derivative magnitude, σ = {sig}")
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "gaussian_derivative")

    sigmas = [1, 1.5, 2, 3, 4, 6, 8]
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("|∂G/∂x ∗ I|", "gradient magnitude"))
    frames = []
    for sig in sigmas:
        kx, ky = gauss_deriv_kernels(sig)
        gx = conv(g, kx); mag = np.hypot(gx, conv(g, ky))
        a = to_uint8(np.clip(np.abs(gx) / np.percentile(np.abs(gx), 99.5), 0, 1))
        b = to_uint8(np.clip(mag / np.percentile(mag, 99.5), 0, 1))
        frames.append(go.Frame(name=str(sig), data=[go.Image(source=to_data_uri(a, "jpg")), go.Image(source=to_data_uri(b, "jpg"))], traces=[0, 1]))
    fig2.add_trace(frames[2].data[0], row=1, col=1); fig2.add_trace(frames[2].data[1], row=1, col=2)
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(sigmas, "σ = "), title="Edges at different scales")
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    return Figure("gaussian_derivative",
                  "Gaussian derivative filters combine smoothing and differentiation in "
                  "one kernel: the 2D kernels are outer products of the 1D Gaussian with "
                  "its derivative, sampled over a window of about 6σ + 1. Small σ responds "
                  "to fine edges; large σ removes fine structure and keeps only the "
                  "coarse, large-scale edges.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def fig_laplacian(ctx: BuildContext) -> Figure:
    g = photo("notre_dame_b", 420).astype(float)
    x = np.linspace(-6, 6, 400)
    edge = 1 / (1 + np.exp(-x * 2))
    d1 = np.gradient(edge, x); d2 = np.gradient(d1, x)
    lap = conv(cv2.GaussianBlur(g, (0, 0), 1.2), LAPLACIAN)
    fig = plt.figure(figsize=(13, 3.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.9, 1.4])
    ax = fig.add_subplot(gs[0]); ax.plot(x, edge, color=C["ink"]); ax.set_title("(a) edge")
    ax = fig.add_subplot(gs[1]); ax.plot(x, d1, color=C["blue"]); ax.set_title("(b) 1st derivative: peak")
    ax = fig.add_subplot(gs[2]); ax.plot(x, d2, color=C["orange"]); ax.axhline(0, color="k", lw=0.5); ax.set_title("(c) 2nd derivative: zero crossing")
    ax = fig.add_subplot(gs[3]); sync_matrix(ax, LAPLACIAN, cmap="Oranges", title="(d) Laplacian kernel")
    ax = fig.add_subplot(gs[4]); v = np.percentile(np.abs(lap), 99); ax.imshow(lap[60:260, 100:340], cmap="gray", vmin=-v, vmax=v)
    ax.set_title("(e) Laplacian of the image (crop)"); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "laplacian")
    return Figure("laplacian",
                  "The Laplacian is a second-order operator: an edge becomes a peak after "
                  "the first derivative and a positive-negative pair with a zero crossing "
                  "after the second. A single 3 × 3 kernel (d) approximates the sum of "
                  "second derivatives in both directions, so it responds to edges of any "
                  "orientation, producing a pair of dark and bright lines either side of "
                  "each true edge (e).", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_edge_profile(ctx), fig_noise_edges(ctx), fig_prewitt_kernels(ctx), fig_gradient_photo(ctx),
            fig_sobel_decomposition(ctx), fig_gaussian_derivative(ctx), fig_laplacian(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("2.1 Edge detection")
    tab1, tab2 = st.tabs(["Kernels on a photo", "Gaussian derivative scale"])
    with tab1:
        key = st.selectbox("image", ["notre_dame_b", "galvani", "trevi", "office_facade"])
        g = load_image(key, gray=True, max_side=480)
        kern = st.radio("kernel", ["Prewitt", "Sobel", "Laplacian"], horizontal=True)
        pre = st.slider("Gaussian smoothing σ before filtering", 0.0, 5.0, 0.0, 0.5)
        im = cv2.GaussianBlur(g, (0, 0), pre) if pre else g
        if kern == "Laplacian":
            out = conv(im, LAPLACIAN); v = np.percentile(np.abs(out), 99)
            st.image(to_uint8(norm01(np.clip(out, -v, v))), caption="Laplacian (offset display)", width="stretch")
        else:
            kx, ky = (PREWITT_X, PREWITT_Y) if kern == "Prewitt" else (SOBEL_X, SOBEL_Y)
            gx, gy, mag, ang = gradient(im, kx, ky)
            c1, c2, c3 = st.columns(3)
            c1.image(to_uint8(norm01(np.abs(gx))), caption="|Gx|", width="stretch")
            c2.image(to_uint8(norm01(np.abs(gy))), caption="|Gy|", width="stretch")
            c3.image(to_uint8(np.clip(mag / np.percentile(mag, 99.5), 0, 1)), caption="magnitude", width="stretch")
    with tab2:
        g = photo("galvani", 520).astype(float)
        sig = st.slider("σ", 0.5, 10.0, 2.0, 0.5)
        kx, ky = gauss_deriv_kernels(sig)
        mag = np.hypot(conv(g, kx), conv(g, ky))
        st.image(to_uint8(np.clip(mag / np.percentile(mag, 99.5), 0, 1)), caption=f"σ = {sig}", width="stretch")
