"""Lesson 2.3 — Corner detection and the Harris algorithm.

Figures
    window_shift        change of intensity E for small window shifts on flat, edge and corner patches
    second_moment       eigenvalue ellipses of the second-moment matrix at flat, edge and corner points
    eigenvalue_regions  the (lambda1, lambda2) plane and the corner response R
    harris_response     Harris response map, thresholding and local maxima on a photograph (interactive)
    harris_invariance   rotation invariance, and failure to detect large-scale corners
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import plotly.graph_objects as go
from skimage.feature import corner_harris, corner_peaks

from .._common import (BuildContext, Figure, C, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, load_image, to_uint8, norm01)

FIGURES = ["window_shift", "second_moment", "eigenvalue_regions", "harris_response", "harris_invariance"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def synthetic_scene(n=200):
    """Bright polygon and disc on a dark background: gives flat, edge and corner points."""
    img = np.zeros((n, n), float)
    cv2.rectangle(img, (40, 50), (120, 150), 1.0, -1)
    pts = np.array([[130, 30], [190, 60], [160, 120]], np.int32)
    cv2.fillPoly(img, [pts], 1.0)
    cv2.circle(img, (150, 165), 22, 1.0, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    return img


def shift_energy(img, r, c, win=7, shifts=range(-3, 4)):
    """E(x, y) = sum over the window of (I(u+x, v+y) − I(u, v))^2 for a grid of shifts."""
    h = win // 2
    ref = img[r - h:r + h + 1, c - h:c + h + 1]
    E = np.zeros((len(shifts), len(shifts)))
    for i, dy in enumerate(shifts):
        for j, dx in enumerate(shifts):
            sh = img[r - h + dy:r + h + 1 + dy, c - h + dx:c + h + 1 + dx]
            E[i, j] = ((sh - ref) ** 2).sum()
    return E


def harris_map(g, sigma=2.0, k=0.05):
    return corner_harris(g.astype(float) / 255.0, method="k", k=k, sigma=sigma)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_window_shift(ctx: BuildContext) -> Figure:
    img = synthetic_scene()
    pts = {"flat": (100, 80), "edge": (100, 40), "corner": (50, 40)}
    fig = plt.figure(figsize=(13, 3.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1, 1, 1])
    ax = fig.add_subplot(gs[0]); ax.imshow(img, cmap="gray"); ax.set_title("synthetic scene with test windows")
    for (name, (r, c)), col in zip(pts.items(), [C["green"], C["blue"], C["red"]]):
        ax.add_patch(plt.Rectangle((c - 3.5, r - 3.5), 7, 7, fill=False, ec=col, lw=2)); ax.text(c + 6, r, name, color=col, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    Es = {name: shift_energy(img, r, c) for name, (r, c) in pts.items()}
    zmax = max(E.max() for E in Es.values())
    for j, (name, col) in enumerate(zip(pts, [C["green"], C["blue"], C["red"]])):
        E = Es[name]
        ax = fig.add_subplot(gs[j + 1], projection="3d")
        X, Y = np.meshgrid(range(-3, 4), range(-3, 4))
        ax.plot_surface(X, Y, E, cmap="viridis", vmin=0, vmax=zmax); ax.set_title(f"E(x, y) at the {name}", color=col)
        ax.set_xlabel("shift x"); ax.set_ylabel("shift y"); ax.set_zlim(0, zmax)
    png = save_mpl(fig, ctx, "window_shift")
    return Figure("window_shift",
                  "Change of intensity E(x, y) when a small window is shifted by (x, y). On "
                  "a flat region E stays near zero in every direction; on an edge it is "
                  "zero for shifts along the edge and large across it; at a corner it "
                  "rises in every direction. A corner is a point where all shifts give a "
                  "large E.", png=png)


def second_moment_matrix(img, sigma=2.0):
    """A = w*X², B = w*Y², C = w*XY with Sobel gradients and a Gaussian window."""
    X = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    A = cv2.GaussianBlur(X * X, (0, 0), sigma)
    B = cv2.GaussianBlur(Y * Y, (0, 0), sigma)
    Cc = cv2.GaussianBlur(X * Y, (0, 0), sigma)
    return A, B, Cc


def fig_second_moment(ctx: BuildContext) -> Figure:
    img = synthetic_scene()
    A, B, Cc = second_moment_matrix(img, 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("iso-contours of E(x, y) at sample points")
    lam1 = np.zeros_like(A); lam2 = np.zeros_like(A)
    for r in range(15, 200, 15):
        for c in range(15, 200, 15):
            M = np.array([[A[r, c], Cc[r, c]], [Cc[r, c], B[r, c]]])
            w, v = np.linalg.eigh(M)          # ascending: w[1] largest, v[:, 1] its (x, y) eigenvector
            if w[1] < 0.05:
                axes[0].plot(c, r, ".", color=C["green"], ms=4); continue
            ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
            length = lambda lam: min(28.0, 5.0 / np.sqrt(max(lam, 1e-3)))  # noqa: E731
            col = C["red"] if w[0] > 0.2 * w[1] else C["blue"]
            axes[0].add_patch(Ellipse((c, r), 2 * length(w[1]), 2 * length(w[0]), angle=ang, fill=False, ec=col, lw=1.3))
    axes[0].set_xticks([]); axes[0].set_yticks([])
    # eigenvalues everywhere for the scatter
    tr = A + B; det = A * B - Cc ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    lam1, lam2 = tr / 2 + disc, tr / 2 - disc
    axes[1].scatter(lam2.ravel()[::5], lam1.ravel()[::5], s=3, c=C["grey"], alpha=0.4)
    axes[1].set_xlabel("λ₂ (smaller)"); axes[1].set_ylabel("λ₁ (larger)"); axes[1].set_title("eigenvalue pairs of every pixel")
    axes[1].text(0.03, 0.92, "edges: λ₁ ≫ λ₂", color=C["blue"], fontsize=9, transform=axes[1].transAxes)
    axes[1].text(0.60, 0.80, "corners: both large", color=C["red"], fontsize=9, transform=axes[1].transAxes)
    axes[1].text(0.30, 0.06, "flat: both small (near the origin)", color=C["green"], fontsize=9, transform=axes[1].transAxes)
    png = save_mpl(fig, ctx, "second_moment")
    return Figure("second_moment",
                  "The second-moment matrix M summarises the gradients around a point. Its "
                  "eigenvalues are the curvatures of E in the two principal directions. "
                  "Left: contours of constant E, with axes proportional to 1/√λ. On an edge "
                  "E barely changes for shifts along the edge, giving a long thin ellipse "
                  "(blue); at a corner E rises in every direction, giving a small round "
                  "ellipse (red); on flat regions the contour is unbounded (green dots). "
                  "Right: the eigenvalue pair of every pixel.", png=png)


def fig_eigenvalue_regions(ctx: BuildContext) -> Figure:
    l = np.linspace(0, 1, 300)
    L1, L2 = np.meshgrid(l, l)
    k = 0.05
    R = L1 * L2 - k * (L1 + L2) ** 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].contourf(L1, L2, R, levels=[-1, -0.02, 0.02, 1], colors=[C["sky"], C["light"], "#f6c8b0"])
    axes[0].contour(L1, L2, R, levels=[0], colors=[C["ink"]], linewidths=1)
    axes[0].text(0.6, 0.6, "corner\nR > 0", ha="center", fontsize=10, color=C["red"])
    axes[0].text(0.8, 0.12, "edge, R < 0", ha="center", fontsize=9, color=C["blue"]); axes[0].text(0.12, 0.8, "edge\nR < 0", ha="center", fontsize=9, color=C["blue"])
    axes[0].text(0.1, 0.1, "flat\n|R| small", ha="center", fontsize=9, color=C["grey"])
    axes[0].set_xlabel("λ₁"); axes[0].set_ylabel("λ₂"); axes[0].set_title(f"corner response R = λ₁λ₂ − k(λ₁+λ₂)², k = {k}")
    axes[1].axis("off")
    axes[1].text(0, 0.9, "det M = λ₁ λ₂ = AB − C²", fontsize=12)
    axes[1].text(0, 0.75, "trace M = λ₁ + λ₂ = A + B", fontsize=12)
    axes[1].text(0, 0.55, "R = det M − k (trace M)²", fontsize=13, weight="bold")
    axes[1].text(0, 0.35, "No eigenvalue decomposition is needed:\nR is computed directly from A, B, C.\n\nR > 0: corner    R < 0: edge    R ≈ 0: flat", fontsize=10, va="top")
    png = save_mpl(fig, ctx, "eigenvalue_regions")
    return Figure("eigenvalue_regions",
                  "Classifying a point by the eigenvalues of M. Both small: flat. One "
                  "large: edge. Both large: corner. Rather than computing eigenvalues, the "
                  "Harris corner response R uses the determinant and trace of M, which are "
                  "cheap to evaluate from its elements; R is positive at corners and "
                  "negative on edges.", png=png)


def fig_harris_response(ctx: BuildContext) -> Figure:
    g = load_image("office_facade", gray=True, max_side=520)
    R = harris_map(g, sigma=2.0, k=0.05)
    peaks = corner_peaks(R, min_distance=6, threshold_rel=0.05)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    axes[0].imshow(g, vmin=0, vmax=255); axes[0].set_title("image")
    v = np.percentile(np.abs(R), 99.5)
    axes[1].imshow(R, cmap="RdBu_r", vmin=-v, vmax=v); axes[1].set_title("corner response R (red > 0, blue < 0)")
    axes[2].imshow(R > 0.05 * R.max(), cmap="gray"); axes[2].set_title("R above threshold")
    axes[3].imshow(g, vmin=0, vmax=255); axes[3].plot(peaks[:, 1], peaks[:, 0], "o", mfc="none", mec=C["red"], ms=5, mew=1.2)
    axes[3].set_title(f"local maxima of R: {len(peaks)} corners")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "harris_response")

    sigmas = [1, 2, 3, 5, 8]
    frames = []
    for s in sigmas:
        Rs = harris_map(g, sigma=s, k=0.05); pk = corner_peaks(Rs, min_distance=6, threshold_rel=0.05)
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        for r, c in pk:
            cv2.circle(rgb, (int(c), int(r)), 4, (213, 94, 0), 1)
        frames.append(go.Frame(name=str(s), data=[go.Image(source=to_data_uri(rgb, "jpg"))],
                               layout=go.Layout(title=f"Gaussian window σ = {s}: {len(pk)} corners")))
    fig2 = go.Figure(data=frames[1].data, frames=frames)
    fig2.update_layout(**slider_frames_layout(sigmas, "σ = "), title=frames[1].layout.title.text)
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    return Figure("harris_response",
                  "The Harris algorithm on a photograph of a building. The corner response "
                  "R is positive (red) at window corners and negative (blue) along the "
                  "long edges; thresholding and taking local maxima marks the corners. "
                  "The slider changes the size of the Gaussian window, which sets the "
                  "scale of corner the detector responds to.",
                  png=png, html=plotly_html(fig2, height=420), kind="plotly")


def fig_harris_invariance(ctx: BuildContext) -> Figure:
    g = load_image("office_facade", gray=True, max_side=400)
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 0.75)
    rot = cv2.warpAffine(g, M, (w, h), borderValue=128)
    pk1 = corner_peaks(harris_map(g, 2.0), min_distance=6, threshold_rel=0.05)
    pk2 = corner_peaks(harris_map(rot, 2.0), min_distance=6, threshold_rel=0.05)
    n = 240
    syn = np.zeros((n, n), np.uint8)
    cv2.circle(syn, (70, 120), 55, 255, -1); cv2.circle(syn, (185, 70), 12, 255, -1); cv2.circle(syn, (185, 160), 7, 255, -1)
    Rs = harris_map(syn, 2.0)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    axes[0].imshow(g, vmin=0, vmax=255); axes[0].plot(pk1[:, 1], pk1[:, 0], "o", mfc="none", mec=C["red"], ms=4); axes[0].set_title(f"{len(pk1)} corners")
    axes[1].imshow(rot, vmin=0, vmax=255); axes[1].plot(pk2[:, 1], pk2[:, 0], "o", mfc="none", mec=C["red"], ms=4); axes[1].set_title(f"rotated 30°, scaled 0.75: {len(pk2)} corners")
    axes[2].imshow(syn, cmap="gray"); axes[2].set_title("discs of three sizes")
    v = np.percentile(np.abs(Rs), 99.9); axes[3].imshow(Rs, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[3].set_title("R: small discs are corners, the large one is an edge")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "harris_invariance")
    return Figure("harris_invariance",
                  "Invariance. Left pair: the same window corners are found after the image "
                  "is rotated and translated, because the eigenvalues of M do not depend on "
                  "orientation. Right pair: the detector has a fixed window size, so small "
                  "discs register as corners while the large disc's boundary is seen only "
                  "as an edge. Harris is rotation invariant but not scale invariant.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_window_shift(ctx), fig_second_moment(ctx), fig_eigenvalue_regions(ctx),
            fig_harris_response(ctx), fig_harris_invariance(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("2.3 Harris corner detection")
    key = st.selectbox("image", ["office_facade", "notre_dame_b", "rubiks_cube", "baby_toys"])
    g = load_image(key, gray=True, max_side=520)
    c1, c2, c3 = st.columns(3)
    s = c1.slider("Gaussian window σ", 1.0, 8.0, 2.0, 0.5)
    k = c2.slider("k", 0.01, 0.2, 0.05, 0.01)
    thr = c3.slider("threshold (fraction of max R)", 0.01, 0.5, 0.05, 0.01)
    R = harris_map(g, sigma=s, k=k)
    pk = corner_peaks(R, min_distance=6, threshold_rel=thr)
    v = np.percentile(np.abs(R), 99.5)
    a, b = st.columns(2)
    a.image(to_uint8(norm01(np.clip(R, -v, v))), caption="corner response R (offset display)", width="stretch")
    fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(g, cmap="gray"); ax.plot(pk[:, 1], pk[:, 0], "o", mfc="none", mec=C["red"], ms=5); ax.axis("off")
    ax.set_title(f"{len(pk)} corners"); b.pyplot(fig)
