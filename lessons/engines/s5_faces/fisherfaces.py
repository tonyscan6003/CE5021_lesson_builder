"""Lesson 5.5 — Fisherfaces.

Figures
    multiclass_lda      three classes in 2D: the multi-class between-class scatter about the global mean
    singularity         the condition number of S_w against the number of PCA components kept
    fisherfaces_gallery the Fisherfaces basis, compared with the eigenfaces basis
    fisherface_pipeline PCA then LDA, as a diagram, with the C-1 limit on the number of components
    eigen_vs_fisher     recognition accuracy under increasing illumination variation
    fisher_query        recognising a query face: apply both transforms, then nearest neighbour
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout, norm01)
from ._faces_common import olivetti, split_by_identity, as_image, illumination_gradient, mean_face
from .lda import unit, draw_projection

FIGURES = ["multiclass_lda", "singularity", "fisherfaces_gallery", "fisherface_pipeline",
           "eigen_vs_fisher", "fisher_query"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def three_classes(n=80, seed=2):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [4.5, 1.2], [2.2, 4.6]])
    cov = np.array([[2.4, 1.6], [1.6, 1.4]])
    X = np.vstack([rng.multivariate_normal(c, cov, n) for c in centres])
    y = np.repeat(np.arange(3), n)
    return X, y


def multiclass_scatters(X, y):
    """S_w = sum of per-class scatters; S_b weighted by class size about the global mean."""
    mu = X.mean(axis=0)
    Sw = np.zeros((X.shape[1],) * 2); Sb = np.zeros_like(Sw)
    for c in np.unique(y):
        Z = X[y == c]; mc = Z.mean(axis=0)
        Sw += (Z - mc).T @ (Z - mc)
        d = (mc - mu)[:, None]
        Sb += len(Z) * (d @ d.T)
    return Sw, Sb, mu


def sw_condition(Xtr, ytr, k):
    """Condition number of the within-class scatter after reducing to k PCA components."""
    Z = PCA(n_components=k, svd_solver="randomized", random_state=0).fit_transform(Xtr)
    Sw, _, _ = multiclass_scatters(Z, ytr)
    return np.linalg.cond(Sw)


def fisher_pipeline(Xtr, ytr, k_pca=80, n_lda=None):
    """PCA to k_pca dimensions, then multi-class LDA. Returns (pca, lda)."""
    pca = PCA(n_components=k_pca, svd_solver="randomized", random_state=0).fit(Xtr)
    Z = pca.transform(Xtr)
    n_classes = len(np.unique(ytr))
    n_lda = n_lda or (n_classes - 1)
    lda = LinearDiscriminantAnalysis(n_components=min(n_lda, n_classes - 1, k_pca)).fit(Z, ytr)
    return pca, lda


def nn_accuracy(A, ytr, B, yte):
    D = np.linalg.norm(B[:, None, :] - A[None, :, :], axis=2)
    return (ytr[D.argmin(axis=1)] == yte).mean()


def accuracy_under_illumination(strengths, k_pca=80, n_train=7):
    """Eigenface (PCA only) versus Fisherface (PCA+LDA) accuracy as illumination varies."""
    Xtr, ytr, Xte, yte = split_by_identity(n_train)
    eig, fish = [], []
    for s in strengths:
        # the query images are lit from one side, the enrolled ones are not
        Xte_s = illumination_gradient(Xte, s)
        pca = PCA(n_components=k_pca, svd_solver="randomized", random_state=0).fit(Xtr)
        eig.append(nn_accuracy(pca.transform(Xtr), ytr, pca.transform(Xte_s), yte))
        pca2, lda = fisher_pipeline(Xtr, ytr, k_pca)
        fish.append(nn_accuracy(lda.transform(pca2.transform(Xtr)), ytr,
                                lda.transform(pca2.transform(Xte_s)), yte))
    return np.array(eig), np.array(fish)


def show_face(ax, vec, title=None, fs=8):
    ax.imshow(as_image(vec), cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fs)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_multiclass_lda(ctx: BuildContext) -> Figure:
    X, y = three_classes()
    Sw, Sb, mu = multiclass_scatters(X, y)
    v = np.linalg.eigh(np.linalg.solve(Sw, Sb))[1][:, -1]
    v_pca = PCA(2).fit(X).components_[0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    ax = axes[0]
    for c in range(3):
        ax.plot(X[y == c, 0], X[y == c, 1], "o", color=SERIES[c], ms=3.5, alpha=0.6, label=f"class {c}")
        mc = X[y == c].mean(axis=0)
        ax.plot(*mc, "X", color=SERIES[c], ms=13, mec="k")
        ax.annotate("", xy=mc, xytext=mu, arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.2, ls="--"))
    ax.plot(*mu, "P", color=C["ink"], ms=12)
    ax.text(mu[0] - 0.4, mu[1] + 0.4, "global mean $\\mu$", fontsize=8, ha="right", va="bottom")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=7)
    ax.set_title("$S_B$ now sums $n_j(\\mu_j - \\mu)(\\mu_j - \\mu)^T$", fontsize=9)
    draw_projection(axes[1], X, y, v, f"the first LDA direction\n(3 classes → at most 2 useful directions)")
    ax = axes[2]
    for c in range(3):
        ax.hist(X[y == c] @ (v / np.linalg.norm(v)), bins=26, alpha=0.6, color=SERIES[c], label=f"class {c}")
    ax.set_title("the three classes along the first LDA direction", fontsize=9); ax.legend(fontsize=8)
    ax.set_xlabel("position along the line")
    png = save_mpl(fig, ctx, "multiclass_lda")
    return Figure("multiclass_lda",
                  "Multi-class LDA. The objective is unchanged, maximise the between-class "
                  "scatter over the within-class scatter, but with more than two classes "
                  "the between-class scatter is defined from the deviation of each class "
                  "mean from the global mean, weighted by the number of points in that "
                  "class. The within-class scatter is still the sum of the per-class "
                  "scatter matrices. With C classes at most C − 1 directions carry "
                  "non-zero eigenvalues, so here two.", png=png)


def fig_singularity(ctx: BuildContext) -> Figure:
    Xtr, ytr, Xte, yte = split_by_identity(7)
    n_train = len(Xtr)
    ks = [10, 20, 40, 80, 150, 200, 260, 280]
    conds = [sw_condition(Xtr, ytr, k) for k in ks]
    Sw_raw, _, _ = multiclass_scatters(Xtr, ytr)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    ax.semilogy(ks, conds, "o-", color=C["blue"], ms=5)
    ax.axhline(1e12, color=C["red"], ls="--"); ax.text(ks[0], 1.6e12, "effectively singular", color=C["red"], fontsize=8)
    ax.set_xlabel("PCA components kept before LDA"); ax.set_ylabel("condition number of $S_W$")
    ax.set_title("$S_W$ becomes singular as the dimension approaches\nthe number of training images", fontsize=9)
    ax = axes[1]; ax.axis("off")
    ax.text(0, 0.95, "the singularity problem", fontsize=10, weight="bold", va="top")
    ax.text(0, 0.80, f"On raw pixels $S_W$ is 4096 × 4096 but is built from only\n{n_train} training images, so its condition number is\n{np.linalg.cond(Sw_raw):.1e}: it cannot be inverted and $S_W^{{-1}} S_B$\ncannot be formed.", fontsize=8.5, va="top")
    ax.text(0, 0.50, "The fix is to apply PCA first and run LDA in the\nreduced space:", fontsize=8.5, va="top")
    ax.text(0.04, 0.35, "•  too few components: the discriminative\n    information is lost\n"
                        "•  too many: $S_W$ is still singular", fontsize=8.5, va="top")
    ax.text(0, 0.12, "The number of components is found by trial; the condition\nnumber of $S_W$ is a useful check.", fontsize=8.5, va="top", color=C["grey"])
    png = save_mpl(fig, ctx, "singularity")
    return Figure("singularity",
                  "Why Fisherfaces applies PCA before LDA. The within-class scatter "
                  "matrix has one row and column per feature but is estimated from the "
                  "training images alone, so with image-sized feature vectors it is "
                  "singular and cannot be inverted. Reducing the dimensionality with PCA "
                  "first makes it invertible, provided the number of components kept is "
                  "comfortably below the number of training images. The condition number "
                  "of S_W is a practical test.", png=png)


def fig_fisherfaces_gallery(ctx: BuildContext) -> Figure:
    Xtr, ytr, Xte, yte = split_by_identity(7)
    k_pca = 80
    pca, lda = fisher_pipeline(Xtr, ytr, k_pca)
    # map each LDA direction back to pixel space: v_pixels = V_pca^T @ w_lda
    fisher = (lda.scalings_[:, :7].T @ pca.components_)
    fig, axes = plt.subplots(2, 8, figsize=(13, 4.2))
    show_face(axes[0, 0], mean_face(Xtr), "mean face", fs=8)
    for k in range(7):
        show_face(axes[0, k + 1], norm01(pca.components_[k]), f"eigenface {k + 1}", fs=8)
    axes[1, 0].axis("off")
    axes[1, 0].text(0.0, 0.5, "eigenfaces (top)\nmaximise variance\n\nfisherfaces (bottom)\nmaximise class\nseparation", fontsize=8, va="center")
    for k in range(7):
        show_face(axes[1, k + 1], norm01(fisher[k]), f"fisherface {k + 1}", fs=8)
    png = save_mpl(fig, ctx, "fisherfaces_gallery")
    return Figure("fisherfaces_gallery",
                  "The two bases compared. Eigenfaces (top) are the directions of greatest "
                  "variance in the database and look like smooth, face-shaped lighting "
                  "patterns. Fisherfaces (bottom), obtained by running LDA in the "
                  "PCA-reduced space and mapping the result back to pixels, look noisier "
                  "because they emphasise the local differences that separate one identity "
                  "from another rather than the variance that all faces share.", png=png)


def fig_fisherface_pipeline(ctx: BuildContext) -> Figure:
    Xtr, ytr, _, _ = split_by_identity(7)
    C_classes = len(np.unique(ytr))
    fig, ax = plt.subplots(figsize=(13, 3.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")

    def box(xy, w, h, text, fc="#e8f1f8", fs=7.8):
        ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=C["grey"]))
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=13,
                                     color=C["ink"], lw=1.2))

    ax.text(0.15, 3.4, "building the Fisherface database", fontsize=10, weight="bold", color=C["grey"])
    steps = [(f"database $X$:\n{len(Xtr)} faces,\n{C_classes} identities", "#e8f1f8"),
             ("PCA step:\nmean, covariance,\neigenvectors $V$", "#fdf1e3"),
             ("reduce to $X' = V_k^T X$\n(k by trial, so that\n$S_W$ is invertible)", "#fdf1e3"),
             ("LDA step: class means,\nper-class scatters,\n$S_W$ and $S_B$", "#e6f4ee"),
             (f"solve $S_W^{{-1}} S_B$,\nkeep the $C - 1$ = {C_classes - 1}\nnon-zero directions", "#e6f4ee"),
             ("store each face's\nfinal coordinates", "#f3e8f5")]
    x = 0.15
    for i, (text, fc) in enumerate(steps):
        box((x, 1.5), 1.72, 1.5, text, fc=fc)
        if i < len(steps) - 1:
            arrow((x + 1.8, 2.25), (x + 2.2, 2.25))
        x += 2.13
    ax.text(0.15, 1.05, "LDA reduces the dimensionality very aggressively: with C classes at most C − 1 eigenvalues are non-zero, "
                        f"so at most {C_classes - 1} directions here.", fontsize=8.5)
    ax.text(0.15, 0.6, "That is usually a benefit, but it can also discard discriminating information and make the classifier fail.",
            fontsize=8.5, color=C["grey"])
    png = save_mpl(fig, ctx, "fisherface_pipeline")
    return Figure("fisherface_pipeline",
                  "The Fisherface algorithm. As in the original paper, PCA is applied "
                  "first to reduce the dimensionality without using the labels, and "
                  "multi-class LDA is then run in that reduced space. The number of PCA "
                  "components is chosen by trial: too few loses discriminative "
                  "information, too many leaves S_W singular. LDA itself keeps at most "
                  "C − 1 directions for C identities.", png=png)


def fig_eigen_vs_fisher(ctx: BuildContext) -> Figure:
    strengths = np.linspace(0, 1.2, 7)
    eig, fish = accuracy_under_illumination(strengths)
    _, _, Xte, _ = split_by_identity(7)
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.5], hspace=0.45)
    for j, s in enumerate(strengths[::2][:4]):
        ax = fig.add_subplot(gs[0, j]); show_face(ax, illumination_gradient(Xte[:1], s)[0], f"illumination {s:.1f}", fs=8)
    ax = fig.add_subplot(gs[0, 4]); ax.axis("off")
    ax.text(0, 0.5, "query images are lit\nfrom one side; the\nenrolled images are not", fontsize=8, va="center")
    ax = fig.add_subplot(gs[1, :])
    ax.plot(strengths, 100 * eig, "o-", color=C["red"], ms=5, label="Eigenfaces (PCA only)")
    ax.plot(strengths, 100 * fish, "s-", color=C["green"], ms=5, label="Fisherfaces (PCA then LDA)")
    ax.set_xlabel("strength of the illumination ramp added to the query images")
    ax.set_ylabel("nearest-neighbour accuracy (%)"); ax.set_ylim(0, 103)
    ax.legend(fontsize=9, loc="lower left")
    png = save_mpl(fig, ctx, "eigen_vs_fisher")
    return Figure("eigen_vs_fisher",
                  "Eigenfaces versus Fisherfaces under illumination change. A linear "
                  "illumination ramp of increasing strength is added to the query images "
                  "only. Because PCA keeps the directions of greatest variance, and "
                  "illumination produces a great deal of variance, the Eigenface "
                  "representation degrades as the lighting changes. Fisherfaces, which "
                  "choose directions that separate identities rather than directions of "
                  "large variance, hold up much better. Belhumeur's original paper "
                  "reported the same pattern on the Harvard and Yale datasets.", png=png)


def fig_fisher_query(ctx: BuildContext) -> Figure:
    Xtr, ytr, Xte, yte = split_by_identity(7)
    pca, lda = fisher_pipeline(Xtr, ytr, 80)
    A = lda.transform(pca.transform(Xtr)); B = lda.transform(pca.transform(Xte))
    i = 5
    D = np.linalg.norm(B[i] - A, axis=1)
    order = np.argsort(D)
    fig = plt.figure(figsize=(13, 3.4))
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 0.4, 1, 1, 1, 0.2, 2.1])
    ax = fig.add_subplot(gs[0]); show_face(ax, Xte[i], f"query\n(true identity {yte[i]})", fs=9)
    ax = fig.add_subplot(gs[1]); ax.axis("off"); ax.annotate("", xy=(1, 0.5), xytext=(0, 0.5), arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.5))
    ax.text(0.5, 0.62, "PCA\nthen LDA", ha="center", fontsize=7.5)
    for j in range(3):
        ax = fig.add_subplot(gs[2 + j])
        idx = order[j]
        show_face(ax, Xtr[idx], f"match {j + 1}: id {ytr[idx]}\ndistance {D[idx]:.2f}", fs=8)
    ax = fig.add_subplot(gs[6]); ax.axis("off")
    ax.text(0, 0.95, "recognition with Fisherfaces", fontsize=10, weight="bold", va="top")
    ax.text(0, 0.78, "Apply the stored PCA and LDA transforms to\nthe query, then take the nearest stored vector.", fontsize=8.5, va="top")
    acc = nn_accuracy(A, ytr, B, yte)
    ax.text(0, 0.52, f"final representation: {A.shape[1]} numbers per face\n(from 4096 pixels)", fontsize=9, va="top")
    ax.text(0, 0.32, f"accuracy on the {len(Xte)} held-out images: {100 * acc:.1f}%", fontsize=9.5, va="top", color=C["green"])
    ax.text(0, 0.14, "A distance threshold, or the ratio of the nearest to\nthe second-nearest distance, rejects unknown faces.", fontsize=8, va="top", color=C["grey"])
    png = save_mpl(fig, ctx, "fisher_query")
    return Figure("fisher_query",
                  "Recognising a query face with Fisherfaces. Both stored transforms, PCA "
                  "and then LDA, are applied to the query image and a nearest-neighbour "
                  "search over the enrolled vectors gives the label. The three closest "
                  "gallery faces are shown with their distances; a threshold on the "
                  "distance, or on the ratio of the first to the second, lets the system "
                  "answer \"unknown\" in open-set use.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_multiclass_lda(ctx), fig_singularity(ctx), fig_fisherfaces_gallery(ctx),
            fig_fisherface_pipeline(ctx), fig_eigen_vs_fisher(ctx), fig_fisher_query(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("5.5 Fisherfaces")
    tab1, tab2 = st.tabs(["Eigenfaces vs Fisherfaces", "Multi-class LDA in 2D"])
    with tab1:
        c1, c2 = st.columns(2)
        k_pca = c1.select_slider("PCA components before LDA", [20, 40, 80, 120, 200], value=80)
        n_train = c2.slider("images enrolled per identity", 2, 9, 7)
        strengths = np.linspace(0, 1.2, 5)
        eig, fish = accuracy_under_illumination(strengths, k_pca, n_train)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.plot(strengths, 100 * eig, "o-", color=C["red"], label="Eigenfaces")
        ax.plot(strengths, 100 * fish, "s-", color=C["green"], label="Fisherfaces")
        ax.set_xlabel("illumination ramp on the query images"); ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 101); ax.legend()
        st.pyplot(fig)
        Xtr, ytr, _, _ = split_by_identity(n_train)
        st.write(f"condition number of $S_W$ with {k_pca} components: **{sw_condition(Xtr, ytr, k_pca):.2e}**")
        st.caption("A condition number beyond about 1e12 means S_W is effectively singular "
                   "and the LDA step becomes unreliable.")
    with tab2:
        seed = st.number_input("data seed", 0, 99, 2)
        X, y = three_classes(80, int(seed))
        Sw, Sb, mu = multiclass_scatters(X, y)
        v = np.linalg.eigh(np.linalg.solve(Sw, Sb))[1][:, -1]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        draw_projection(axes[0], X, y, v, "first LDA direction")
        for c in range(3):
            axes[1].hist(X[y == c] @ (v / np.linalg.norm(v)), bins=24, alpha=0.6, color=SERIES[c])
        axes[1].set_title("projected classes")
        st.pyplot(fig)
