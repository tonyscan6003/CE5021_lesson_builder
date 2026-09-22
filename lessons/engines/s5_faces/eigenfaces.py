"""Lesson 5.3 — Eigenfaces.

Figures
    naive_fr            raw-pixel nearest neighbour: long vectors, and it confuses identities
    eigenfaces_gallery  the mean face and the leading eigenfaces
    face_as_weights     one face as the mean plus a weighted sum of eigenfaces
    face_reconstruction reconstruction quality against the number of eigenfaces kept (interactive)
    weight_space        faces of three identities plotted on the first two eigenface axes
    eigenface_database  the enrolment and query algorithms, as a diagram
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, to_uint8, norm01)
from ._faces_common import olivetti, split_by_identity, as_image, mean_face

FIGURES = ["naive_fr", "eigenfaces_gallery", "face_as_weights", "face_reconstruction",
           "weight_space", "eigenface_database"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def fit_eigenfaces(X, k=60):
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(X)
    return pca


def show_face(ax, vec, title=None, fs=8):
    ax.imshow(as_image(vec), cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fs)


def nn_accuracy(Xtr, ytr, Xte, yte, k=None):
    """Nearest-neighbour accuracy on raw pixels (k=None) or on k PCA weights."""
    if k:
        pca = fit_eigenfaces(Xtr, k)
        A, B = pca.transform(Xtr), pca.transform(Xte)
    else:
        A, B = Xtr, Xte
    D = np.linalg.norm(B[:, None, :] - A[None, :, :], axis=2)
    return (ytr[D.argmin(axis=1)] == yte).mean(), A.shape[1]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_naive_fr(ctx: BuildContext) -> Figure:
    images, X, y = olivetti()
    Xtr, ytr, Xte, yte = split_by_identity()
    acc_raw, dim_raw = nn_accuracy(Xtr, ytr, Xte, yte)
    acc_pca, dim_pca = nn_accuracy(Xtr, ytr, Xte, yte, 20)
    # a test face whose raw-pixel nearest neighbour is the wrong identity
    D = np.linalg.norm(Xte[:, None, :] - Xtr[None, :, :], axis=2)
    nn = D.argmin(axis=1); wrong = np.flatnonzero(ytr[nn] != yte)
    i = int(wrong[0]) if len(wrong) else 0
    fig = plt.figure(figsize=(13, 4.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.7, 1.6, 0.7, 0.7, 1.5])
    ax = fig.add_subplot(gs[0]); show_face(ax, Xte[i], f"query, identity {yte[i]}")
    ax = fig.add_subplot(gs[1]); ax.plot(Xte[i], color=C["blue"], lw=0.5)
    ax.set_title("the same face as a 4096-element\nraw-pixel vector", fontsize=9); ax.set_xlabel("pixel index")
    ax = fig.add_subplot(gs[2]); show_face(ax, Xtr[nn[i]], f"nearest in the gallery:\nidentity {ytr[nn[i]]}", fs=8)
    ax = fig.add_subplot(gs[3]); show_face(ax, Xtr[ytr == yte[i]][0], f"correct identity {yte[i]}", fs=8)
    ax = fig.add_subplot(gs[4]); ax.axis("off")
    ax.text(0, 0.95, "naive facial recognition", fontsize=10, weight="bold", va="top")
    ax.text(0, 0.78, "Use all 4096 pixels as the vector and match by\nEuclidean distance to every gallery face.", fontsize=8.5, va="top")
    ax.text(0, 0.56, f"nearest neighbour on raw pixels:\n   {dim_raw} dimensions, accuracy {100 * acc_raw:.1f}%", fontsize=9, va="top", color=C["red"])
    ax.text(0, 0.36, f"on PCA weights (next figures):\n   {dim_pca} dimensions, accuracy {100 * acc_pca:.1f}%", fontsize=9, va="top", color=C["green"])
    ax.text(0, 0.16, f"The same accuracy from {dim_pca} numbers instead of\n{dim_raw}: {dim_raw // dim_pca}× less storage and a {dim_raw // dim_pca}× shorter\ndistance computation per comparison.", fontsize=8, va="top", color=C["grey"])
    png = save_mpl(fig, ctx, "naive_fr")
    return Figure("naive_fr",
                  "Why raw pixels are a poor representation for faces. Using all 4096 "
                  "pixels as the vector, this query face's nearest neighbour in the "
                  "gallery is the wrong person. On a dataset as small and as tightly "
                  "constrained as this one raw pixels still classify most faces "
                  "correctly, but every comparison needs 4096 numbers, and with many "
                  "identities and few examples each the curse of dimensionality bites. "
                  "Compressing each face to a few dozen PCA weights, the subject of the "
                  "rest of this lesson, reaches the same accuracy from a vector two "
                  "hundred times shorter.", png=png)


def fig_eigenfaces_gallery(ctx: BuildContext) -> Figure:
    images, X, y = olivetti()
    pca = fit_eigenfaces(X, 60)
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(2, 9)
    ax = fig.add_subplot(gs[:, 0:2]); show_face(ax, mean_face(X), "mean face $\\mu$\n(the average of all 400 images)", fs=9)
    for k in range(14):
        ax = fig.add_subplot(gs[k // 7, 2 + k % 7])
        show_face(ax, norm01(pca.components_[k]), f"$v_{{{k + 1}}}$", fs=8)
    fig.suptitle("the first 14 eigenfaces: the eigenvectors of the covariance matrix of the face database, reshaped to 64 × 64", fontsize=10)
    png = save_mpl(fig, ctx, "eigenfaces_gallery")
    return Figure("eigenfaces_gallery",
                  "Eigenfaces are the principal components of a database of face images. "
                  "Each eigenvector has the same length as a face vector, so it can be "
                  "reshaped and displayed as an image. The first few capture broad "
                  "lighting and shape variation across the database; later ones capture "
                  "progressively finer detail. Every face is represented relative to the "
                  "mean face.", png=png)


def fig_face_as_weights(ctx: BuildContext) -> Figure:
    images, X, y = olivetti()
    pca = fit_eigenfaces(X, 60)
    x = X[y == 9][0]
    w = pca.transform(x[None])[0]
    mu = mean_face(X)
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(2, 7, height_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[0, 0]); show_face(ax, x, "face $x_i$", fs=9)
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off"); ax.text(0.5, 0.5, "≈", fontsize=26, ha="center", va="center")
    ax = fig.add_subplot(gs[0, 2]); show_face(ax, mu, "mean $\\mu$", fs=9)
    for j, k in enumerate(range(4)):
        ax = fig.add_subplot(gs[0, 3 + j])
        ax.axis("off")
        sub = ax.get_subplotspec().subgridspec(1, 1)
        a = fig.add_subplot(sub[0])
        show_face(a, norm01(pca.components_[k]), f"{w[k]:+.2f} × $v_{{{k + 1}}}$", fs=8)
    for j, k in enumerate([1, 2, 4, 10, 30, 60]):
        rec = mu + pca.components_[:k].T @ w[:k]
        ax = fig.add_subplot(gs[1, j + 1]); show_face(ax, rec, f"$k$ = {k}", fs=8)
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    ax.text(0, 0.8, "$x_i \\approx \\mu + \\sum_k w_{ik} v_k$", fontsize=11, va="top")
    ax.text(0, 0.42, "the weight vector $w_i$ is the\ncompressed face; below, the\nsum truncated after $k$ terms", fontsize=8, va="top", color=C["grey"])
    png = save_mpl(fig, ctx, "face_as_weights")
    return Figure("face_as_weights",
                  "A face as a weighted combination of eigenfaces. Multiplying the "
                  "mean-centred face by the eigenface matrix gives a weight vector wᵢ, one "
                  "weight per eigenface; the face is then the mean face plus each "
                  "eigenface scaled by its weight. The weight vector is a set of "
                  "coordinates in the k-dimensional eigenface space, and it is what the "
                  "database actually stores.", png=png)


def fig_face_reconstruction(ctx: BuildContext) -> Figure:
    images, X, y = olivetti()
    pca_full = fit_eigenfaces(X, 120)
    mu = mean_face(X)
    x = X[y == 9][0]
    w = pca_full.transform(x[None])[0]
    ks = [1, 2, 5, 10, 20, 40, 80, 120]
    fig = plt.figure(figsize=(12, 5.0))
    gs = fig.add_gridspec(2, 5, hspace=0.3)
    cells = [gs[0, 0], gs[0, 1], gs[0, 2], gs[0, 3], gs[0, 4], gs[1, 0], gs[1, 1], gs[1, 2]]
    show_face(fig.add_subplot(cells[0]), x, "original", fs=9)
    for cell, k in zip(cells[1:], [1, 5, 10, 20, 40, 80, 120]):
        rec = mu + pca_full.components_[:k].T @ w[:k]
        err = np.linalg.norm(rec - x)
        show_face(fig.add_subplot(cell), rec, f"$k$ = {k}, error {err:.1f}", fs=9)
    ax = fig.add_subplot(gs[1, 3:5])
    errs = [np.linalg.norm(mu + pca_full.components_[:k].T @ w[:k] - x) for k in range(1, 121)]
    ax.plot(range(1, 121), errs, color=C["blue"], lw=2)
    for k in (1, 5, 10, 20, 40, 80, 120):
        ax.plot(k, errs[k - 1], "o", color=C["red"], ms=4)
    ax.set_xlabel("eigenfaces kept, $k$"); ax.set_ylabel("reconstruction error")
    ax.set_title("error falls steeply, then flattens", fontsize=9)
    png = save_mpl(fig, ctx, "face_reconstruction")
    frames = []
    for k in ks:
        rec = mu + pca_full.components_[:k].T @ w[:k]
        frames.append(go.Frame(name=str(k), data=[go.Image(source=to_data_uri(to_uint8(np.clip(as_image(rec), 0, 1) * 255)))],
                               layout=go.Layout(title=f"reconstruction from k = {k} eigenface weights (of 4096 pixels)")))
    fig2 = go.Figure(data=frames[4].data, frames=frames)
    fig2.update_layout(**slider_frames_layout(ks, "k = "), title=frames[4].layout.title.text)
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    return Figure("face_reconstruction",
                  "Reconstructing one face from its weights. With a single eigenface the "
                  "result is barely more than the mean face; by twenty the identity is "
                  "clear, and beyond about forty the improvement is slight. A face that "
                  "took 4096 numbers is represented by a few dozen, which is the storage "
                  "and speed advantage of the eigenface method.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def fig_weight_space(ctx: BuildContext) -> Figure:
    images, X, y = olivetti()
    pca = fit_eigenfaces(X, 60)
    W = pca.transform(X)
    ids = [3, 9, 21]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axes[0]
    for c, ident in enumerate(ids):
        sel = y == ident
        ax.plot(W[sel, 0], W[sel, 1], "o", color=SERIES[c], ms=8, label=f"identity {ident}")
    ax.set_xlabel("weight on eigenface 1"); ax.set_ylabel("weight on eigenface 2")
    ax.set_title("(a) the ten images of three identities,\nplotted on the first two eigenface axes only", fontsize=9.5)
    ax.legend(fontsize=8)
    ax = axes[1]
    # within-identity versus between-identity distances in weight space
    within, between = [], []
    for ident in np.unique(y):
        Wi = W[y == ident]
        d = np.linalg.norm(Wi[:, None, :] - Wi[None, :, :], axis=2)
        within += list(d[np.triu_indices(len(Wi), 1)])
    rng = np.random.default_rng(0)
    for _ in range(600):
        a, b = rng.integers(0, len(W), 2)
        if y[a] != y[b]:
            between.append(np.linalg.norm(W[a] - W[b]))
    ax.hist(within, bins=40, color=C["green"], alpha=0.75, density=True, label="same identity")
    ax.hist(between, bins=40, color=C["red"], alpha=0.55, density=True, label="different identities")
    ax.set_xlabel("distance in eigenface space"); ax.set_ylabel("density"); ax.legend(fontsize=8)
    ax.set_title(f"(b) using all {W.shape[1]} axes: same-identity distances\nare usually the smaller ones", fontsize=9.5)
    png = save_mpl(fig, ctx, "weight_space")
    return Figure("weight_space",
                  "Recognition in eigenface space. Each face becomes a point whose "
                  "coordinates are its eigenface weights; the ten images of one identity "
                  "cluster together (left, first two axes only). Across the whole "
                  "database, distances between images of the same person are mostly "
                  "smaller than distances between different people (right), which is why "
                  "a nearest-neighbour search on the weights recognises faces. The "
                  "overlap between the two histograms is where the errors come from.",
                  png=png)


def fig_eigenface_database(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(13, 4.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")

    def box(xy, w, h, text, fc="#e8f1f8", fs=8):
        ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=C["grey"]))
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=13,
                                     color=C["ink"], lw=1.2))

    ax.text(0.15, 3.95, "(a) building the database", fontsize=10, weight="bold", color=C["grey"])
    x = 0.15
    for text in ["$N$ face images,\neach a 4096-vector", "subtract the mean face $\\mu$",
                 "covariance matrix,\neigen­decomposition", "keep $K$ eigenfaces $V_K$",
                 "store $w_i = V_K^T (x_i - \\mu)$\nfor every face"]:
        box((x, 2.4), 2.1, 1.2, text, fs=7.5)
        if x < 10:
            arrow((x + 2.18, 3.0), (x + 2.62, 3.0))
        x += 2.56
    ax.text(0.15, 1.95, "(b) recognising a query face", fontsize=10, weight="bold", color=C["grey"])
    x = 0.15
    for text, fc in [("query image $x_q$", "#e8f1f8"), ("subtract $\\mu$, multiply by $V_K$\n→ weights $w_q$", "#fdf1e3"),
                     ("distance from $w_q$ to every\nstored $w_i$", "#fdf1e3"), ("smallest distance → label\n(or reject if too large)", "#e6f4ee")]:
        box((x, 0.4), 2.7, 1.2, text, fc=fc, fs=7.5)
        if x < 9.5:
            arrow((x + 2.78, 1.0), (x + 3.26, 1.0))
        x += 3.22
    png = save_mpl(fig, ctx, "eigenface_database")
    return Figure("eigenface_database",
                  "The two eigenface algorithms. Building the database: mean-centre the N "
                  "face vectors, take the eigenvectors of their covariance matrix, keep "
                  "the K with the largest eigenvalues, and store each face as its K "
                  "weights. Recognising a query: project it onto the same K eigenfaces "
                  "and find the nearest stored weight vector. An unknown face can be "
                  "rejected when even the nearest distance is too large, or when the "
                  "nearest and second-nearest distances are too similar.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_naive_fr(ctx), fig_eigenfaces_gallery(ctx), fig_face_as_weights(ctx),
            fig_face_reconstruction(ctx), fig_weight_space(ctx), fig_eigenface_database(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("5.3 Eigenfaces")
    images, X, y = olivetti()
    tab1, tab2 = st.tabs(["Reconstruction", "Recognition accuracy"])
    with tab1:
        c1, c2 = st.columns(2)
        ident = c1.slider("identity", 0, 39, 9); img = c2.slider("image of that identity", 0, 9, 0)
        k = st.select_slider("eigenfaces kept, k", [1, 2, 5, 10, 20, 40, 80, 150], value=20)
        pca = fit_eigenfaces(X, 150); mu = mean_face(X)
        x = X[y == ident][img]
        w = pca.transform(x[None])[0]
        rec = mu + pca.components_[:k].T @ w[:k]
        a, b, c = st.columns(3)
        a.image(as_image(x), caption="original", width=190, clamp=True)
        b.image(norm01(as_image(rec)), caption=f"reconstruction from k = {k} weights", width=190)
        c.image(norm01(as_image(np.abs(rec - x))), caption="absolute difference", width=190)
        st.write(f"reconstruction error: **{np.linalg.norm(rec - x):.2f}**  "
                 f"(storing {k} numbers instead of 4096)")
    with tab2:
        n_train = st.slider("images per identity used for enrolment", 1, 9, 7)
        Xtr, ytr, Xte, yte = split_by_identity(n_train)
        ks = [5, 10, 20, 40, 80]
        rows = [("raw pixels", *nn_accuracy(Xtr, ytr, Xte, yte))]
        for k in ks:
            acc, dim = nn_accuracy(Xtr, ytr, Xte, yte, k)
            rows.append((f"{k} eigenfaces", acc, dim))
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar([r[0] for r in rows], [100 * r[1] for r in rows],
               color=[C["red"]] + [C["green"]] * len(ks))
        ax.set_ylabel("nearest-neighbour accuracy (%)"); ax.set_ylim(0, 100)
        st.pyplot(fig)
        st.caption(f"{len(Xtr)} enrolled images, {len(Xte)} query images, 40 identities.")
