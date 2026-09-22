"""Lesson 3.5 (supplementary) — Scale space and image pyramids.

Figures
    sigma_composition   two successive Gaussian blurs equal one blur with sigma = sqrt(s1^2 + s2^2)
    gaussian_pyramid    blur and subsample: a pyramid of progressively smaller images
    octave_pyramid      scale space built by successive convolution within octaves and subsampling between them
    pyramid_cost        multiplications per pixel for direct large kernels versus the pyramid
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .._common import BuildContext, Figure, C, save_mpl, load_image

FIGURES = ["sigma_composition", "gaussian_pyramid", "octave_pyramid", "pyramid_cost"]


def fig_sigma_composition(ctx: BuildContext) -> Figure:
    g = load_image("cat", gray=True, max_side=400).astype(float)
    s1, s2 = 2.0, 3.0
    s3 = np.hypot(s1, s2)
    direct = cv2.GaussianBlur(g, (0, 0), s3)
    seq = cv2.GaussianBlur(cv2.GaussianBlur(g, (0, 0), s1), (0, 0), s2)
    x = np.linspace(-12, 12, 500)
    gauss = lambda s: np.exp(-x ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)  # noqa: E731
    conv = np.convolve(gauss(s1), gauss(s2), mode="same") * (x[1] - x[0])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(x, gauss(s1), label=f"G(σ₁ = {s1})"); axes[0].plot(x, gauss(s2), label=f"G(σ₂ = {s2})")
    axes[0].plot(x, conv, color=C["red"], lw=2.5, label="G(σ₁) ∗ G(σ₂)"); axes[0].plot(x, gauss(s3), "k--", label=f"G(√(σ₁²+σ₂²) = {s3:.2f})")
    axes[0].legend(fontsize=7); axes[0].set_title("convolving two Gaussians gives a Gaussian")
    axes[1].imshow(seq, vmin=0, vmax=255); axes[1].set_title(f"blur σ = {s1} then σ = {s2}")
    axes[2].imshow(direct, vmin=0, vmax=255); axes[2].set_title(f"single blur σ = {s3:.2f}: max difference {np.abs(seq - direct).max():.2f}")
    for ax in axes[1:]:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "sigma_composition")
    return Figure("sigma_composition",
                  "Successive Gaussian convolutions combine: G(σ₁) ∗ G(σ₂) = G(√(σ₁² + σ₂²)). "
                  "A scale-space level L(x; σ₃) can therefore be computed from the previous "
                  "level with a small kernel instead of from the original image with a "
                  "large one, and the two results agree to rounding error.", png=png)


def fig_gaussian_pyramid(ctx: BuildContext) -> Figure:
    g = load_image("cat", gray=True, max_side=512)
    levels = [g]
    for _ in range(4):
        levels.append(cv2.pyrDown(levels[-1]))
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.6), gridspec_kw=dict(width_ratios=[16, 8, 4, 2, 1]))
    for ax, lv, i in zip(axes, levels, range(5)):
        ax.imshow(lv, vmin=0, vmax=255); ax.set_title(f"level {i}\n{lv.shape[1]} × {lv.shape[0]}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "gaussian_pyramid")
    return Figure("gaussian_pyramid",
                  "A Gaussian image pyramid. Each level is the previous one blurred with a "
                  "Gaussian and then subsampled by two, shown here at true relative size. "
                  "Blurring before subsampling prevents aliasing; the blurred image contains "
                  "redundant information, so fewer samples suffice to represent it.", png=png)


def fig_octave_pyramid(ctx: BuildContext) -> Figure:
    g = load_image("cat", gray=True, max_side=384).astype(float)
    n_scales, n_oct = 3, 3
    k = 2 ** (1 / n_scales)
    fig, axes = plt.subplots(n_oct, n_scales + 1, figsize=(12, 8.2))
    base = g.copy(); sigma_abs = 1.0
    for o in range(n_oct):
        cur = cv2.GaussianBlur(base, (0, 0), 1.0)
        for s in range(n_scales + 1):
            ax = axes[o, s]
            ax.imshow(cur, vmin=0, vmax=255, extent=[0, g.shape[1], g.shape[0], 0]); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"octave {o}, scale {s}: {cur.shape[1]} px wide, σ ≈ {sigma_abs * k ** s:.1f}", fontsize=8)
            if s < n_scales:
                inc = np.sqrt((sigma_abs * k ** (s + 1)) ** 2 - (sigma_abs * k ** s) ** 2) / (2 ** o)
                cur = cv2.GaussianBlur(cur, (0, 0), inc)
        base = cur[::2, ::2]
        sigma_abs *= 2
    png = save_mpl(fig, ctx, "octave_pyramid")
    return Figure("octave_pyramid",
                  "Scale space with an image pyramid. Within an octave (a row) each scale "
                  "is obtained from the previous one by a small incremental blur. When the "
                  "blur reaches twice the octave's starting σ the image is subsampled by "
                  "two to start the next octave, so σ doubles while the kernel stays small. "
                  "Images are displayed at the same size; their pixel counts halve per "
                  "octave.", png=png)


def fig_pyramid_cost(ctx: BuildContext) -> Figure:
    sig = np.array([1, 2, 4, 8, 16, 32])
    direct = (6 * sig + 1) * 2                       # separable direct kernel from the original image
    pyramid = np.full_like(sig, (6 * 1 + 1) * 2) / 4 ** np.arange(len(sig))   # constant small kernel on shrinking images
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(np.arange(len(sig)) - 0.2, direct, 0.4, color=C["red"], label="direct: kernel 6σ+1 on the full image")
    ax.bar(np.arange(len(sig)) + 0.2, pyramid, 0.4, color=C["green"], label="pyramid: fixed small kernel on a ¼-size image per octave")
    ax.set_xticks(range(len(sig))); ax.set_xticklabels([f"σ = {s}" for s in sig]); ax.set_yscale("log"); ax.set_ylabel("multiplications per original pixel (separable)")
    ax.legend(fontsize=8); ax.set_title("cost of reaching each scale")
    png = save_mpl(fig, ctx, "pyramid_cost")
    return Figure("pyramid_cost",
                  "Why pyramids matter for real-time detectors. Reaching a large σ directly "
                  "needs a kernel of 6σ + 1 samples over every pixel of the full image; in "
                  "a pyramid the kernel stays small and each octave has a quarter as many "
                  "pixels as the last. The price is an extra interpolation step to place "
                  "keypoints found in subsampled octaves back in the original image.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_sigma_composition(ctx), fig_gaussian_pyramid(ctx), fig_octave_pyramid(ctx), fig_pyramid_cost(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("3.5 Image pyramids")
    g = load_image("cat", gray=True, max_side=512)
    c1, c2 = st.columns(2)
    s1 = c1.slider("first blur σ₁", 0.5, 6.0, 2.0, 0.5); s2 = c2.slider("second blur σ₂", 0.5, 6.0, 3.0, 0.5)
    s3 = float(np.hypot(s1, s2))
    seq = cv2.GaussianBlur(cv2.GaussianBlur(g.astype(float), (0, 0), s1), (0, 0), s2)
    direct = cv2.GaussianBlur(g.astype(float), (0, 0), s3)
    a, b = st.columns(2)
    a.image(np.clip(seq, 0, 255).astype(np.uint8), caption=f"σ₁ then σ₂", width="stretch")
    b.image(np.clip(direct, 0, 255).astype(np.uint8), caption=f"single σ = √(σ₁²+σ₂²) = {s3:.2f}; max difference {np.abs(seq - direct).max():.2f}", width="stretch")
    st.subheader("Pyramid")
    n = st.slider("levels", 1, 6, 4)
    levels = [g]
    for _ in range(n):
        levels.append(cv2.pyrDown(levels[-1]))
    st.image([lv for lv in levels], caption=[f"{lv.shape[1]}×{lv.shape[0]}" for lv in levels], width=140)
