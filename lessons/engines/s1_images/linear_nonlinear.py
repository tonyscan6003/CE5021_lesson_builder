"""Lesson 1.4 — Linear filters, image noise and non-linear filters.

Figures
    gaussian_kernel     continuous vs sampled 1D Gaussian, 2D kernel, blur ladder (interactive)
    separable_kernel    outer product, two-pass convolution, multiply counts
    superposition       filtering a sum of images equals the sum of filtered images
    shift_invariance    a shifted input gives a shifted output
    commutativity       blur-then-sharpen equals sharpen-then-blur
    image_noise         Gaussian and salt-and-pepper noise models
    median_filter       3x3 median on a patch; median vs Gaussian on impulse noise; repeated medians
    bilateral_filter    Gaussian vs bilateral smoothing near an edge
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import convolve2d

from .._common import (BuildContext, Figure, C, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, load_image, sync_matrix)

FIGURES = ["gaussian_kernel", "separable_kernel", "superposition", "shift_invariance",
           "commutativity", "image_noise", "median_filter", "bilateral_filter"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def gauss1d(sigma: float, size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sampled 1D Gaussian over the rule-of-thumb window M = 6σ + 1 (odd)."""
    if size is None:
        size = int(6 * sigma + 1) | 1
    r = size // 2
    x = np.arange(-r, r + 1)
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return x, g / g.sum()


def gauss2d(sigma: float) -> np.ndarray:
    _, g = gauss1d(sigma)
    return np.outer(g, g)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    k = int(6 * sigma + 1) | 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def add_gaussian_noise(img: np.ndarray, sigma: float, rng) -> np.ndarray:
    n = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(float) + n, 0, 255).astype(np.uint8)


def add_salt_pepper(img: np.ndarray, p: float, rng) -> np.ndarray:
    out = img.copy()
    u = rng.random(img.shape[:2])
    out[u < p / 2] = 0
    out[u > 1 - p / 2] = 255
    return out


SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float)
SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float)


def cat_gray(max_side=512):
    return load_image("cat", gray=True, max_side=max_side)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_gaussian_kernel(ctx: BuildContext) -> Figure:
    rgb = load_image("macaw", max_side=512)
    sigma = 1.5
    x, g = gauss1d(sigma)
    xc = np.linspace(-6, 6, 600)
    gc = np.exp(-xc ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.1, 1])
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(xc, gc, color=C["blue"], label="continuous Gaussian")
    ax.bar(x, g, width=0.9, color=C["orange"], alpha=0.7, label=f"sampled, M = 6σ+1 = {len(x)}")
    ax.set_title(f"(a) 1D Gaussian, σ = {sigma}"); ax.set_xlabel("x (pixels)"); ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    G = gauss2d(sigma)
    r = G.shape[0] // 2
    X, Y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    ax.plot_surface(X, Y, G, cmap="viridis")
    ax.set_title("(b) 2D kernel"); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax = fig.add_subplot(gs[0, 3:])
    sync_matrix(ax, G * 1000, fmt="{:.0f}", title="(c) 2D kernel weights (×1000)", fontsize=7)
    for i, s in enumerate([0.5, 1, 2, 4, 8]):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(blur(rgb, s)); ax.set_title(f"σ = {s}")
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "gaussian_kernel")

    sigmas = [0.5, 1, 2, 3, 4, 6, 8, 12, 16]
    fig2 = make_subplots(rows=1, cols=2, column_widths=[0.35, 0.65],
                         subplot_titles=("sampled 1D kernel", "blurred image"))
    frames = []
    for s in sigmas:
        x, g = gauss1d(s)
        frames.append(go.Frame(name=str(s), data=[
            go.Bar(x=x, y=g, marker_color=C["orange"]),
            go.Image(source=to_data_uri(blur(rgb, s), "jpg"))], traces=[0, 1],
            layout=go.Layout(title=f"σ = {s}: kernel window M = {len(x)} pixels")))
    fig2.add_trace(frames[1].data[0], row=1, col=1); fig2.add_trace(frames[1].data[1], row=1, col=2)
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(sigmas, "σ = "), title=frames[1].layout.title.text)
    fig2.update_xaxes(visible=False, row=1, col=2); fig2.update_yaxes(visible=False, row=1, col=2)
    fig2.update_xaxes(title_text="x (pixels)", row=1, col=1)
    return Figure("gaussian_kernel",
                  "The Gaussian kernel. (a) The continuous Gaussian is sampled at integer "
                  "pixel positions over a window of M = 6σ + 1 samples, which captures the "
                  "whole bell. (b, c) The 2D kernel is the outer product of the 1D sample "
                  "and is normalised to sum to one. Bottom: larger σ removes progressively "
                  "finer detail, leaving only coarse structure.",
                  png=png, html=plotly_html(fig2, height=450), kind="plotly")


def fig_separable_kernel(ctx: BuildContext) -> Figure:
    h1 = np.array([1, 2, 1])
    H = np.outer(h1, h1)
    img = np.array([[1, 2, 0], [0, 1, 3], [2, 1, 0]])
    step1 = convolve2d(img, h1.reshape(-1, 1), mode="same")
    step2 = convolve2d(step1, h1.reshape(1, -1), mode="same")
    direct = convolve2d(img, H, mode="same")
    assert np.array_equal(step2, direct)
    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.1])
    ax = fig.add_subplot(gs[0, 0]); sync_matrix(ax, h1.reshape(-1, 1), title="H₁ (column)", cmap="Oranges")
    ax = fig.add_subplot(gs[0, 1]); sync_matrix(ax, h1.reshape(1, -1), title="H₁ᵀ (row)", cmap="Oranges")
    ax = fig.add_subplot(gs[0, 2]); sync_matrix(ax, H, title="H = H₁ H₁ᵀ (outer product)", cmap="Oranges")
    ax = fig.add_subplot(gs[0, 3:])
    ks = np.arange(3, 32, 2)
    ax.plot(ks, ks ** 2, color=C["red"], label="2D kernel: k² multiplies per pixel")
    ax.plot(ks, 2 * ks, color=C["green"], label="separable: 2k multiplies per pixel")
    ax.set_xlabel("kernel size k"); ax.set_ylabel("multiplications per output pixel")
    ax.set_title("cost of direct vs separable convolution"); ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 0]); sync_matrix(ax, img, title="image I")
    ax = fig.add_subplot(gs[1, 1]); sync_matrix(ax, step1, title="I ∗ H₁ (columns)", cmap="Greens")
    ax = fig.add_subplot(gs[1, 2]); sync_matrix(ax, step2, title="(I ∗ H₁) ∗ H₁ᵀ", cmap="Greens")
    ax = fig.add_subplot(gs[1, 3]); sync_matrix(ax, direct, title="I ∗ H directly", cmap="Purples")
    ax = fig.add_subplot(gs[1, 4:]); ax.axis("off")
    ax.text(0, 0.6, "Same result:\nL = I ∗ H = (I ∗ H₁) ∗ H₁ᵀ\n\n(one ring of zero padding,\n'same' output size)",
            fontsize=11, va="center")
    png = save_mpl(fig, ctx, "separable_kernel")
    return Figure("separable_kernel",
                  "Separable convolution. Top: a symmetric kernel H is the outer product of "
                  "a 1D kernel with its own transpose, and the cost of applying it grows as "
                  "k² per pixel directly but only 2k when applied as two 1D passes. Bottom: "
                  "convolving a 3 × 3 image with H₁ down the columns, then with H₁ᵀ along the "
                  "rows, gives exactly the result of the single 2D convolution.", png=png)


def fig_superposition(ctx: BuildContext) -> Figure:
    a = load_image("pyramids", gray=True, max_side=480).astype(float)
    b = load_image("balloons", gray=True, max_side=480).astype(float)
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    s = 3.0
    fa, fb = blur(a, s), blur(b, s)
    fsum = blur(a + b, s)
    diff = np.abs(fa + fb - fsum)
    fig, axes = plt.subplots(2, 3, figsize=(11, 4.6))
    panels = [("I₁", a), ("I₂", b), ("I₁ + I₂", a + b),
              ("I₁ ∗ H", fa), ("I₂ ∗ H", fb), ("(I₁ + I₂) ∗ H", fsum)]
    for ax, (t, im) in zip(axes.ravel(), panels):
        ax.imshow(im, vmin=0, vmax=max(255, im.max())); ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"(I₁ ∗ H) + (I₂ ∗ H) − (I₁ + I₂) ∗ H: maximum difference = {diff.max():.1e}", fontsize=10)
    png = save_mpl(fig, ctx, "superposition")
    return Figure("superposition",
                  "Superposition. Blurring two images separately and adding the results "
                  "(bottom left and middle) gives the same image as blurring their sum "
                  "(bottom right); the difference is zero to floating-point precision. "
                  "Linear filters obey this principle; non-linear ones do not.", png=png)


def fig_shift_invariance(ctx: BuildContext) -> Figure:
    n = 160
    yy, xx = np.mgrid[0:n, 0:n]
    def disc(cx, cy):
        return ((xx - cx) ** 2 + (yy - cy) ** 2 < 30 ** 2).astype(float) * 255
    a = disc(55, 70); b = disc(100, 95)
    ea = np.abs(convolve2d(a, SOBEL_X, mode="same")); eb = np.abs(convolve2d(b, SOBEL_X, mode="same"))
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.6))
    axes[0, 0].imshow(a); axes[0, 0].set_title("(a) input")
    axes[0, 1].imshow(ea); axes[0, 1].set_title("(a) edge filter output")
    axes[1, 0].imshow(b); axes[1, 0].set_title("(b) input shifted by (+45, +25)")
    axes[1, 1].imshow(eb); axes[1, 1].set_title("(b) output shifted by (+45, +25)")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "shift_invariance")
    return Figure("shift_invariance",
                  "Shift invariance. Moving the object in the input moves the filter's "
                  "output by exactly the same amount, because convolution applies the same "
                  "kernel at every position.", png=png)


def fig_commutativity(ctx: BuildContext) -> Figure:
    g = load_image("alpaca", gray=True, max_side=480).astype(np.float32)
    G = gauss2d(2.0)
    bs = convolve2d(convolve2d(g, G, mode="same", boundary="symm"), SHARPEN, mode="same", boundary="symm")
    sb = convolve2d(convolve2d(g, SHARPEN, mode="same", boundary="symm"), G, mode="same", boundary="symm")
    b1 = convolve2d(g, G, mode="same", boundary="symm")
    s1 = convolve2d(g, SHARPEN, mode="same", boundary="symm")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    panels = [("input", g), ("blur first", b1), ("then sharpen", bs),
              ("input", g), ("sharpen first", s1), ("then blur", sb)]
    for ax, (t, im) in zip(axes.ravel(), panels):
        ax.imshow(np.clip(im, 0, 255), vmin=0, vmax=255); ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"final images differ by at most {np.abs(bs - sb).max():.1e} (interior)", fontsize=10)
    png = save_mpl(fig, ctx, "commutativity")
    return Figure("commutativity",
                  "Convolution is commutative and associative. Blurring then sharpening "
                  "(top) and sharpening then blurring (bottom) pass through different "
                  "intermediate images but end at the same result. Equivalently the two "
                  "kernels could be convolved with each other first and applied once.",
                  png=png)


def fig_image_noise(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(1)
    g = load_image("rubiks_cube", gray=True, max_side=400)
    gn = add_gaussian_noise(g, 25, rng)
    sp = add_salt_pepper(g, 0.06, rng)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    x = np.linspace(-100, 100, 400)
    axes[0, 0].plot(x, np.exp(-x ** 2 / (2 * 25 ** 2)) / (25 * np.sqrt(2 * np.pi)), color=C["blue"])
    axes[0, 0].set_title("Gaussian noise: P(g), σ = 25"); axes[0, 0].set_xlabel("noise value g")
    axes[0, 1].imshow(g, vmin=0, vmax=255); axes[0, 1].set_title("original")
    axes[0, 2].imshow(gn, vmin=0, vmax=255); axes[0, 2].set_title("with Gaussian noise")
    axes[1, 0].stem([0, 255], [0.03, 0.03], linefmt=C["red"], markerfmt="o", basefmt=" ")
    axes[1, 0].set_xlim(-20, 275); axes[1, 0].set_ylim(0, 0.05)
    axes[1, 0].set_title("Impulse noise: P(g) at a = 0 and b = 255"); axes[1, 0].set_xlabel("pixel value")
    axes[1, 1].imshow(g, vmin=0, vmax=255); axes[1, 1].set_title("original")
    axes[1, 2].imshow(sp, vmin=0, vmax=255); axes[1, 2].set_title("with salt-and-pepper noise (6%)")
    for ax in axes[:, 1:].ravel():
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "image_noise")
    return Figure("image_noise",
                  "Two noise models. Top: Gaussian noise adds to each pixel a random value "
                  "drawn from a bell-shaped distribution; the wider the distribution "
                  "(larger σ) the noisier the image. Bottom: impulse or salt-and-pepper "
                  "noise sets a random subset of pixels to the extreme values 0 or 255.",
                  png=png)


def fig_median_filter(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(2)
    g = cat_gray(400)
    sp = add_salt_pepper(g, 0.08, rng)
    med = cv2.medianBlur(sp, 3)
    gau = blur(sp, 1.5)
    patch = np.array([[12, 200, 15], [14, 13, 255], [0, 16, 11]])
    vals = sorted(patch.ravel())
    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0]); sync_matrix(ax, patch, title="(a) 3 × 3 window", cmap="Blues")
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.text(0, 0.7, "sorted: " + ", ".join(str(v) for v in vals), fontsize=9)
    ax.text(0, 0.5, f"median (5th of 9) = {vals[4]}", fontsize=11, weight="bold")
    ax.text(0, 0.3, f"mean = {patch.mean():.0f} (pulled up by outliers)", fontsize=9, color=C["red"])
    for j, (t, im) in enumerate([("(b) salt-and-pepper", sp), ("(b) Gaussian blur σ=1.5", gau)]):
        ax = fig.add_subplot(gs[0, 2 + j]); ax.imshow(im, vmin=0, vmax=255); ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 0]); ax.imshow(med, vmin=0, vmax=255); ax.set_title("(b) 3 × 3 median"); ax.set_xticks([]); ax.set_yticks([])
    cur = g.copy()
    for j, n in enumerate([1, 5, 20]):
        while True:
            cur = cv2.medianBlur(cur, 5)
            n -= 1
            if n == 0:
                break
        ax = fig.add_subplot(gs[1, 1 + j]); ax.imshow(cur, vmin=0, vmax=255)
        ax.set_title(f"(c) 5 × 5 median applied ×{[1, 5, 20][j]}"); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "median_filter")
    return Figure("median_filter",
                  "The median filter. (a) The nine values under a 3 × 3 structuring element "
                  "are sorted and the middle one is taken, so outliers such as 200 and 255 "
                  "have no influence. (b) On salt-and-pepper noise a Gaussian blur only "
                  "smears the impulses, whereas the median removes them. (c) Applying the "
                  "median repeatedly strips fine texture but keeps the main edges sharp.",
                  png=png)


def fig_bilateral_filter(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(3)
    g = cat_gray(400)
    gn = add_gaussian_noise(g, 18, rng)
    gau = blur(gn, 3.0)
    bil = cv2.bilateralFilter(gn, d=0, sigmaColor=40, sigmaSpace=3.0)
    # 1D illustration of the range kernel near an edge
    xs = np.arange(-15, 16)
    edge = np.where(xs < 0, 60.0, 180.0) + rng.normal(0, 6, xs.size)
    centre = 5
    spatial = np.exp(-(xs - centre) ** 2 / (2 * 4.0 ** 2))
    rangew = np.exp(-(edge - edge[xs == centre][0]) ** 2 / (2 * 30.0 ** 2))
    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0]); ax.plot(xs, edge, "k.-", lw=0.8); ax.axvline(centre, color=C["red"], ls="--")
    ax.set_title("(a) intensity profile across an edge"); ax.set_xlabel("x"); ax.set_ylabel("intensity")
    ax = fig.add_subplot(gs[0, 1]); ax.plot(xs, spatial, color=C["blue"], label="spatial Gaussian")
    ax.plot(xs, rangew, color=C["orange"], label="range Gaussian")
    ax.plot(xs, spatial * rangew, color=C["green"], lw=2, label="product = bilateral weights")
    ax.axvline(centre, color=C["red"], ls="--"); ax.legend(fontsize=8); ax.set_title("(a) weights at the red pixel")
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    ax.text(0, 0.8, "Gaussian filter: weights depend on\nspatial distance only → averages\nacross the edge.", fontsize=9, va="top")
    ax.text(0, 0.4, "Bilateral filter: weights also depend on\nintensity difference → pixels on the\nother side of the edge are ignored.", fontsize=9, va="top")
    for j, (t, im) in enumerate([("(b) noisy input", gn), ("(b) Gaussian σ = 3", gau), ("(b) bilateral σ_s = 3, σ_r = 40", bil)]):
        ax = fig.add_subplot(gs[1, j]); ax.imshow(im, vmin=0, vmax=255); ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "bilateral_filter")
    return Figure("bilateral_filter",
                  "The bilateral filter. (a) At the red pixel the spatial Gaussian would "
                  "average both sides of the edge; multiplying by a range Gaussian in "
                  "intensity keeps only the pixels on the same side, so the effective "
                  "weights are truncated at the edge. Because the weights change from pixel "
                  "to pixel the filter is non-linear. (b) It removes noise while blurring "
                  "edges far less than a Gaussian of the same spatial width.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_gaussian_kernel(ctx), fig_separable_kernel(ctx), fig_superposition(ctx),
            fig_shift_invariance(ctx), fig_commutativity(ctx), fig_image_noise(ctx),
            fig_median_filter(ctx), fig_bilateral_filter(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("1.4 Linear filters, noise and non-linear filters")
    tab1, tab2, tab3 = st.tabs(["Gaussian kernel", "Noise and the median filter", "Bilateral filter"])
    rng = np.random.default_rng(0)
    with tab1:
        s = st.slider("σ", 0.5, 16.0, 2.0, 0.5)
        x, g = gauss1d(s)
        c1, c2 = st.columns([1, 2])
        fig, ax = plt.subplots(figsize=(4, 2.6)); ax.bar(x, g, color=C["orange"]); ax.set_title(f"M = {len(x)}")
        c1.pyplot(fig)
        c2.image(blur(load_image("macaw", max_side=512), s), caption=f"Gaussian blur σ = {s}", width="stretch")
    with tab2:
        g = cat_gray(400)
        kind = st.radio("noise", ["salt-and-pepper", "Gaussian"], horizontal=True)
        if kind == "salt-and-pepper":
            p = st.slider("fraction of corrupted pixels", 0.0, 0.3, 0.08, 0.01)
            noisy = add_salt_pepper(g, p, rng)
        else:
            sd = st.slider("noise σ", 0, 60, 20)
            noisy = add_gaussian_noise(g, sd, rng)
        k = st.select_slider("filter size", [3, 5, 7, 9], value=3)
        c1, c2, c3 = st.columns(3)
        c1.image(noisy, caption="noisy", width="stretch")
        c2.image(blur(noisy, k / 3), caption=f"Gaussian, σ = {k/3:.1f}", width="stretch")
        c3.image(cv2.medianBlur(noisy, k), caption=f"{k} × {k} median", width="stretch")
    with tab3:
        g = cat_gray(400)
        gn = add_gaussian_noise(g, 18, rng)
        ss = st.slider("spatial σ", 1.0, 8.0, 3.0, 0.5)
        sr = st.slider("range σ (intensity)", 5, 150, 40, 5)
        c1, c2, c3 = st.columns(3)
        c1.image(gn, caption="noisy", width="stretch")
        c2.image(blur(gn, ss), caption=f"Gaussian σ = {ss}", width="stretch")
        c3.image(cv2.bilateralFilter(gn, 0, sr, ss), caption=f"bilateral σ_s = {ss}, σ_r = {sr}", width="stretch")
