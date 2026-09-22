"""Lesson 5.2 — Principal component analysis.

Figures
    curse_dimensionality  sparsity as dimensions are added, and error against the number of features
    covariance_shapes     four 2D distributions with their sample covariance matrices
    eigenvectors_2d       eigenvectors of the covariance matrix as the principal components
    pca_transform         projecting onto the eigenvectors removes covariance; dropping one gives 1D
    pca_algorithm         the steps, as a diagram
    pca_compression       image patches compressed with k principal components (interactive)
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, load_image, to_uint8, norm01)

FIGURES = ["curse_dimensionality", "covariance_shapes", "eigenvectors_2d", "pca_transform",
           "pca_algorithm", "pca_compression"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def correlated_2d(n=200, seed=0, cov=((5, 3.5), (3.5, 3))):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal([0, 0], np.array(cov, float), n)


def sample_cov(X):
    Xc = X - X.mean(axis=0)
    return (Xc.T @ Xc) / (len(X) - 1)


def patchify(img, p=16):
    h, w = img.shape
    h, w = h - h % p, w - w % p
    img = img[:h, :w]
    patches = img.reshape(h // p, p, w // p, p).transpose(0, 2, 1, 3).reshape(-1, p * p)
    return patches, (h, w)


def unpatchify(patches, shape, p=16):
    h, w = shape
    return patches.reshape(h // p, w // p, p, p).transpose(0, 2, 1, 3).reshape(h, w)


def pca_compress(img, k, p=16):
    patches, shape = patchify(img.astype(float), p)
    pca = PCA(n_components=k).fit(patches)
    rec = pca.inverse_transform(pca.transform(patches))
    kept = k * (patches.shape[0] + p * p)               # coefficients + the basis itself
    return unpatchify(rec, shape, p), pca, kept / patches.size


def sphinx_gray(max_side=512):
    g = load_image("sphinx", gray=True, max_side=max_side)
    return g


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_curse_dimensionality(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(0)
    n = 24
    fig = plt.figure(figsize=(13, 3.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3])
    a = rng.uniform(0, 1, (n, 3)); b = rng.uniform(0, 1, (n, 3)) * 0.9 + 0.1
    ax = fig.add_subplot(gs[0])
    ax.plot(a[:, 0], np.zeros(n), "o", color=SERIES[0], ms=6); ax.plot(b[:, 0], np.zeros(n) + 0.02, "o", color=SERIES[1], ms=6)
    ax.set_ylim(-0.5, 0.5); ax.set_yticks([]); ax.set_title("(a) 1 feature: classes overlap", fontsize=9); ax.set_xlabel("feature 1")
    ax = fig.add_subplot(gs[1])
    ax.plot(a[:, 0], a[:, 1], "o", color=SERIES[0], ms=6); ax.plot(b[:, 0], b[:, 1], "o", color=SERIES[1], ms=6)
    ax.set_title("(b) 2 features: better separated,\nbut the points are further apart", fontsize=9); ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax = fig.add_subplot(gs[2], projection="3d")
    ax.scatter(*a.T, color=SERIES[0], s=18); ax.scatter(*b.T, color=SERIES[1], s=18)
    ax.set_title("(c) 3 features: sparser again", fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # mean nearest-neighbour distance grows with dimension at fixed sample count
    dims = np.arange(1, 21)
    sparsity = []
    for d in dims:
        P = rng.uniform(0, 1, (48, d))
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2); np.fill_diagonal(D, np.inf)
        sparsity.append(D.min(axis=1).mean())
    ax = fig.add_subplot(gs[3])
    h1, = ax.plot(dims, sparsity, "o-", color=C["red"], ms=4,
                  label="mean nearest-neighbour distance (48 points, fixed)")
    ax.set_ylim(0, max(sparsity) * 1.45)
    ax2 = ax.twinx()
    err = 0.55 * np.exp(-dims / 3.0) + 0.06 + 0.015 * dims
    h2, = ax2.plot(dims, err, color=C["blue"], lw=2, label="classification error (typical)")
    best = int(dims[np.argmin(err)])
    h3 = ax2.axvline(best, color=C["grey"], ls=":", lw=1.2,
                     label=f"optimal number of features ({best})")
    ax2.set_ylim(0, max(err) * 1.5)
    ax2.set_ylabel("error", color=C["blue"])
    ax.set_xlabel("number of features"); ax.set_ylabel("sparsity", color=C["red"])
    ax.set_title("(d) sparsity always rises; error has an optimum", fontsize=9)
    ax.legend(handles=[h1, h2, h3], fontsize=6.5, loc="lower right", framealpha=0.9)
    png = save_mpl(fig, ctx, "curse_dimensionality")
    return Figure("curse_dimensionality",
                  "The curse of dimensionality. One feature cannot separate these two "
                  "classes; adding features separates them better but, with the number of "
                  "samples fixed, the points grow further apart. That sparsity is what "
                  "makes a classifier overfit, and the training data needed to hold "
                  "generalisation constant grows exponentially with each dimension. For a "
                  "fixed dataset there is an optimal number of features; a 256 × 256 image "
                  "used as raw pixels has 65,536, far beyond it.", png=png)


def fig_covariance_shapes(ctx: BuildContext) -> Figure:
    cases = [("positive covariance", ((5, 3.5), (3.5, 3))), ("negative covariance", ((5, -3.5), (-3.5, 3))),
             ("no covariance, unequal variance", ((5, 0), (0, 1))), ("no covariance, equal variance", ((3, 0), (0, 3)))]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (name, cov) in zip(axes, cases):
        X = correlated_2d(700, 1, cov)
        S = sample_cov(X)
        ax.plot(X[:, 0], X[:, 1], "o", color=C["blue"], ms=3, alpha=0.6)
        ax.set_aspect("equal"); ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
        ax.set_xlabel("$d_1$"); ax.set_ylabel("$d_2$")
        ax.set_title(f"{name}\n$\\Sigma \\approx$ [[{S[0,0]:.1f}, {S[0,1]:.1f}], [{S[1,0]:.1f}, {S[1,1]:.1f}]]", fontsize=8.5)
    png = save_mpl(fig, ctx, "covariance_shapes")
    return Figure("covariance_shapes",
                  "The sample covariance matrix describes the shape of a distribution. Its "
                  "diagonal entries are the variances of the data along each axis; the "
                  "off-diagonal entries are the covariance between the axes. Positive "
                  "covariance means d₂ tends to rise with d₁, negative that it falls, and "
                  "zero that the two variables are independent.", png=png)


def fig_eigenvectors_2d(ctx: BuildContext) -> Figure:
    X = correlated_2d(300, 2)
    S = sample_cov(X)
    w, V = np.linalg.eigh(S)
    order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
    # the sign of an eigenvector is arbitrary; orient each one into the positive half-plane
    V = V * np.where(V[np.abs(V).argmax(axis=0), np.arange(V.shape[1])] < 0, -1.0, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axes[0]
    ax.plot(X[:, 0], X[:, 1], "o", color=C["grey"], ms=3, alpha=0.6)
    mu = X.mean(axis=0)
    for k, col in zip(range(2), [C["red"], C["blue"]]):
        v = V[:, k] * 2.2 * np.sqrt(w[k])
        ax.annotate("", xy=mu + v, xytext=mu, arrowprops=dict(arrowstyle="->", color=col, lw=3))
        lab = mu + v * (1.12 if k == 0 else 3.4)
        ax.text(*lab, f"$u_{k+1}$, $\\lambda_{k+1}$ = {w[k]:.1f}", color=col, fontsize=9,
                ha="left" if k == 0 else "right", va="bottom")
    ax.set_aspect("equal"); ax.set_title("eigenvectors of the covariance matrix"); ax.set_xlabel("$d_1$"); ax.set_ylabel("$d_2$")
    ax = axes[1]; ax.axis("off")
    ax.text(0, 0.95, "The eigenvectors of the sample covariance matrix\npoint along the directions of maximum variance.", fontsize=9.5, va="top")
    ax.text(0, 0.72, r"$\Sigma\, u = \lambda\, u$", fontsize=15)
    ax.text(0, 0.56, f"$\\lambda_1$ = {w[0]:.2f} along $u_1$ = ({V[0,0]:.2f}, {V[1,0]:.2f})", fontsize=10, color=C["red"])
    ax.text(0, 0.44, f"$\\lambda_2$ = {w[1]:.2f} along $u_2$ = ({V[0,1]:.2f}, {V[1,1]:.2f})", fontsize=10, color=C["blue"])
    ax.text(0, 0.24, "They are orthogonal, and they form a new coordinate\nsystem in which the data has zero covariance.\nEach eigenvalue is the variance along its eigenvector,\nso a small eigenvalue means little information is lost\nby discarding that direction.", fontsize=8.5, va="top", color=C["grey"])
    png = save_mpl(fig, ctx, "eigenvectors_2d")
    return Figure("eigenvectors_2d",
                  "The eigenvectors of the sample covariance matrix give the directions of "
                  "maximum variance and are the principal components of the data. They are "
                  "orthogonal, and each eigenvalue is the variance of the data along its "
                  "own eigenvector.", png=png)


def fig_pca_transform(ctx: BuildContext) -> Figure:
    X = correlated_2d(300, 2)
    mu = X.mean(axis=0)
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    Z1 = Z.copy(); Z1[:, 1] = 0
    back = pca.inverse_transform(Z1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(X[:, 0], X[:, 1], "o", color=C["grey"], ms=3, alpha=0.6)
    for k, col in zip(range(2), [C["red"], C["blue"]]):
        v = pca.components_[k] * 2.2 * np.sqrt(pca.explained_variance_[k])
        axes[0].annotate("", xy=mu + v, xytext=mu, arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
    axes[0].set_title("(a) original, covariant data"); axes[0].set_xlabel("$d_1$"); axes[0].set_ylabel("$d_2$")
    S = sample_cov(Z)
    axes[1].plot(Z[:, 0], Z[:, 1], "o", color=C["green"], ms=3, alpha=0.6)
    axes[1].set_title(f"(b) after multiplying by the eigenvectors\ncovariance {S[0,1]:.0e} ≈ 0"); axes[1].set_xlabel("$u_1$"); axes[1].set_ylabel("$u_2$")
    axes[2].plot(X[:, 0], X[:, 1], "o", color=C["light"], ms=3)
    axes[2].plot(back[:, 0], back[:, 1], "o", color=C["orange"], ms=3)
    axes[2].set_title(f"(c) keeping only $u_1$: 2D → 1D\n{100 * pca.explained_variance_ratio_[0]:.0f}% of the variance retained")
    axes[2].set_xlabel("$d_1$"); axes[2].set_ylabel("$d_2$")
    for ax in axes:
        ax.set_aspect("equal")
    png = save_mpl(fig, ctx, "pca_transform")
    return Figure("pca_transform",
                  "The two uses of the principal components. Multiplying the mean-centred "
                  "data by the eigenvectors transforms it to a coordinate system in which "
                  "the covariance is zero (b). Keeping only the k eigenvectors with the "
                  "largest eigenvalues reduces the dimensionality: here the second "
                  "component carries little variance, so projecting onto the first alone "
                  "loses little information (c).", png=png)


def fig_pca_algorithm(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(13, 3.0)); ax.set_xlim(0, 13); ax.set_ylim(0, 3.0); ax.axis("off")
    steps = ["1. arrange the $n$ samples\nof dimension $d$ into\na $d \\times n$ matrix $X$",
             "2. subtract the mean\nof each dimension\nfrom every sample",
             "3. form the sample\ncovariance matrix\n$\\Sigma = X X^T$",
             "4. eigendecompose $\\Sigma$\n(or use the singular\nvalue decomposition)",
             "5. order the eigenvalues\nand keep the $k$ largest\nand their eigenvectors"]
    x = 0.15
    for i, text in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x, 0.9), 2.1, 1.5, boxstyle="round,pad=0.02,rounding_size=0.02",
                                    fc="#e8f1f8" if i < 4 else "#e6f4ee", ec=C["grey"]))
        ax.text(x + 1.05, 1.65, text, ha="center", va="center", fontsize=8)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 2.18, 1.65), (x + 2.62, 1.65), arrowstyle="->",
                                         mutation_scale=13, color=C["ink"], lw=1.2))
        x += 2.56
    ax.text(0.15, 0.45, "The k retained eigenvectors are the principal components. Multiplying a sample by them gives its "
                        "compressed representation; multiplying back reconstructs an approximation of it.", fontsize=8.5, color=C["grey"])
    png = save_mpl(fig, ctx, "pca_algorithm")
    return Figure("pca_algorithm",
                  "The PCA algorithm for data of any dimension. In Python this is "
                  "`sklearn.decomposition.PCA`: `fit` performs steps 2 to 5, `transform` "
                  "produces the compressed representation and `inverse_transform` "
                  "reconstructs an approximation of the original data.", png=png)


def fig_pca_compression(ctx: BuildContext) -> Figure:
    g = sphinx_gray(512)
    ks = [1, 2, 4, 8, 16, 32, 64]
    rec8, pca, ratio8 = pca_compress(g, 64)
    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.85, 1.0], hspace=0.12)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(g, cmap="gray", vmin=0, vmax=255)
    ax.set_title("(a) image, divided into 16 × 16 patches\n(every fourth boundary drawn)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for cx in range(0, g.shape[1], 64):
        ax.axvline(cx, color=C["sky"], lw=0.5)
    for cy in range(0, g.shape[0], 64):
        ax.axhline(cy, color=C["sky"], lw=0.5)
    ax = fig.add_subplot(gs[0, 1:3])
    comp = pca.components_[:8].reshape(-1, 16, 16)
    strip = np.hstack([np.pad(norm01(c), 1, constant_values=1.0) for c in comp])
    ax.imshow(strip, cmap="gray")
    ax.set_title("(b) the first 8 eigenvectors, reshaped to 16 × 16 patches", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax = fig.add_subplot(gs[0, 3])
    cum = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(np.arange(1, len(cum) + 1), cum, "-", color=C["blue"], lw=2)
    ax.plot([1, 4, 16, 64], cum[[0, 3, 15, 63]], "o", color=C["red"], ms=5, zorder=3)
    for k in (1, 4, 16, 64):
        ax.annotate(f"k = {k}", (k, cum[k - 1]), textcoords="offset points", xytext=(6, -9),
                    fontsize=7.5, color=C["red"])
    ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16, 64]); ax.set_xticklabels([1, 4, 16, 64])
    ax.set_xlabel("components kept, $k$"); ax.set_ylabel("variance retained (%)")
    ax.set_title("(c) variance captured", fontsize=9); ax.set_ylim(0, 103)
    for j, (tag, k) in enumerate(zip("defg", [1, 4, 16, 64])):
        rec, _, ratio = pca_compress(g, k)
        ax = fig.add_subplot(gs[1, j]); ax.imshow(np.clip(rec, 0, 255), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"({tag}) k = {k}: {100 * ratio:.0f}% of the data", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    png = save_mpl(fig, ctx, "pca_compression")

    frames = []
    for k in ks:
        rec, _, ratio = pca_compress(g, k)
        frames.append(go.Frame(name=str(k), data=[go.Image(source=to_data_uri(to_uint8(np.clip(rec, 0, 255)), "jpg"))],
                               layout=go.Layout(title=f"k = {k} of 256 components: {100 * ratio:.0f}% of the original data")))
    fig2 = go.Figure(data=frames[3].data, frames=frames)
    fig2.update_layout(**slider_frames_layout(ks, "components k = "), title=frames[3].layout.title.text)
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    return Figure("pca_compression",
                  "Image compression with PCA. The image is divided into 16 × 16 patches, "
                  "each flattened into a 256-element vector, and PCA is fitted to the set "
                  "of patches. The eigenvectors, reshaped back into patches (b), are the "
                  "basis; the first few already capture most of the variance (c). "
                  "Reconstructing each patch from only k coefficients (d to g) loses detail "
                  "gradually, and the image is recognisable with a small fraction of the "
                  "original data.",
                  png=png, html=plotly_html(fig2, height=470), kind="plotly")


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_curse_dimensionality(ctx), fig_covariance_shapes(ctx), fig_eigenvectors_2d(ctx),
            fig_pca_transform(ctx), fig_pca_algorithm(ctx), fig_pca_compression(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("5.2 Principal component analysis")
    tab1, tab2 = st.tabs(["PCA on 2D data", "Image compression"])
    with tab1:
        c1, c2, c3 = st.columns(3)
        v1 = c1.slider("variance of $d_1$", 0.5, 8.0, 5.0, 0.5)
        v2 = c2.slider("variance of $d_2$", 0.5, 8.0, 3.0, 0.5)
        cv = c3.slider("covariance", -4.0, 4.0, 3.5, 0.25)
        cov = np.array([[v1, cv], [cv, v2]])
        if np.linalg.eigvalsh(cov).min() <= 0:
            st.warning("That covariance matrix is not positive definite; reduce the covariance.")
        else:
            X = correlated_2d(300, 2, cov)
            pca = PCA(2).fit(X); mu = X.mean(axis=0)
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            axes[0].plot(X[:, 0], X[:, 1], "o", color=C["grey"], ms=3)
            for k, col in zip(range(2), [C["red"], C["blue"]]):
                v = pca.components_[k] * 2.2 * np.sqrt(pca.explained_variance_[k])
                axes[0].annotate("", xy=mu + v, xytext=mu, arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
            axes[0].set_aspect("equal"); axes[0].set_title("data and principal components")
            Z = pca.transform(X); axes[1].plot(Z[:, 0], Z[:, 1], "o", color=C["green"], ms=3); axes[1].set_aspect("equal")
            axes[1].set_title("after the PCA transform")
            st.pyplot(fig)
            st.write(f"variance retained by the first component: **{100 * pca.explained_variance_ratio_[0]:.1f}%**")
    with tab2:
        key = st.selectbox("image", ["sphinx", "cat", "macaw", "trevi"])
        g = load_image(key, gray=True, max_side=512)
        k = st.select_slider("components kept (of 256)", [1, 2, 4, 8, 16, 32, 64, 128], value=8)
        rec, pca, ratio = pca_compress(g, k)
        c1, c2 = st.columns(2)
        c1.image(g, caption="original", width="stretch")
        c2.image(to_uint8(np.clip(rec, 0, 255)), caption=f"k = {k}: {100 * ratio:.0f}% of the data, {100 * pca.explained_variance_ratio_.sum():.1f}% of the variance", width="stretch")
