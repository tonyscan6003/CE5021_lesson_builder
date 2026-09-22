"""Lesson 4.2 — Machine learning for image recognition.

Figures
    supervised_learning     training on labelled feature vectors, then predicting labels for new images
    classification_regression   a decision boundary between two classes versus a fitted regression line
    dataset_partition       train / validation / test split, and k-fold cross-validation
    raw_pixel_features      handwritten digits as 64-element pixel vectors, and a compressed 2D PCA representation
    classifier_complexity   underfitting, a good fit and overfitting on the same data; bias and variance
    train_val_curves        training and validation error against training effort for three cases (interactive k)
    unsupervised_kmeans     unlabelled points clustered by k-means, and test points assigned to clusters
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import plotly.graph_objects as go
from sklearn.datasets import load_digits, make_moons
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout, use_source)

FIGURES = ["supervised_learning", "classification_regression", "dataset_partition", "raw_pixel_features",
           "classifier_complexity", "train_val_curves", "unsupervised_kmeans"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def moons(n=300, noise=0.3, seed=0):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X, y


def decision_regions(clf, X, pad=0.5, res=200):
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, res), np.linspace(y0, y1, res))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, Z


def knn_error_curves(seed=0):
    X, y = moons(400, 0.35, seed)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.4, random_state=seed)
    ks = [101, 75, 51, 35, 25, 15, 11, 7, 5, 3, 1]
    tr, va = [], []
    for k in ks:
        clf = KNeighborsClassifier(k).fit(Xtr, ytr)
        tr.append(1 - clf.score(Xtr, ytr)); va.append(1 - clf.score(Xva, yva))
    return ks, tr, va, (Xtr, Xva, ytr, yva)


def box(ax, xy, w, h, text, fc="#e8f1f8", fontsize=9):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=C["grey"], lw=1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=14, color=C["ink"], lw=1.2))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_supervised_learning(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 3.8)); ax.set_xlim(0, 12); ax.set_ylim(0, 3.8); ax.axis("off")
    ax.text(0.1, 3.3, "(a) training: learning from labelled examples", fontsize=10, weight="bold", color=C["grey"])
    box(ax, (0.2, 2.2), 2.2, 0.8, "training images\nwith labels"); arrow(ax, (2.45, 2.6), (3.1, 2.6))
    box(ax, (3.1, 2.2), 2.4, 0.8, "representation", fc="#fdf1e3"); arrow(ax, (5.55, 2.6), (6.2, 2.6))
    box(ax, (6.2, 2.2), 2.0, 0.8, "feature vectors\n+ labels"); arrow(ax, (8.25, 2.6), (8.9, 2.6))
    box(ax, (8.9, 2.2), 2.9, 0.8, "classifier: internal parameters\nare updated to fit the data", fc="#e6f4ee", fontsize=8)
    ax.text(0.1, 1.5, "(b) prediction on an unlabelled image", fontsize=10, weight="bold", color=C["grey"])
    box(ax, (0.2, 0.4), 2.2, 0.8, "test image"); arrow(ax, (2.45, 0.8), (3.1, 0.8))
    box(ax, (3.1, 0.4), 2.4, 0.8, "same representation", fc="#fdf1e3"); arrow(ax, (5.55, 0.8), (6.2, 0.8))
    box(ax, (6.2, 0.4), 2.0, 0.8, "feature vector"); arrow(ax, (8.25, 0.8), (8.9, 0.8))
    box(ax, (8.9, 0.4), 2.9, 0.8, "trained classifier → predicted label", fc="#e6f4ee", fontsize=8)
    png = save_mpl(fig, ctx, "supervised_learning")
    return Figure("supervised_learning",
                  "Supervised learning: the classifier learns from a training set of "
                  "labelled examples, adjusting its internal parameters so that its "
                  "predictions match the labels. The trained classifier then predicts "
                  "the class of unlabelled test images converted to the same "
                  "representation.", png=png)


def fig_classification_regression(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(1)
    X, y = moons(200, 0.25, 2)
    clf = SVC(kernel="rbf", C=1.0, gamma=1.0).fit(X, y)
    xx, yy, Z = decision_regions(clf, X)
    xr = rng.uniform(0, 10, 60); yr = 2.5 * xr + 3 + rng.normal(0, 3, 60)
    a, b = np.polyfit(xr, yr, 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[C["sky"], "#f6c8b0"], alpha=0.5)
    for c in (0, 1):
        axes[0].plot(X[y == c, 0], X[y == c, 1], "o", color=SERIES[c], ms=4, label=f"class {c}")
    axes[0].set_title("classification: feature vectors → labels, via a decision boundary"); axes[0].legend(fontsize=8); axes[0].set_xlabel("feature 1"); axes[0].set_ylabel("feature 2")
    axes[1].plot(xr, yr, "o", color=C["grey"], ms=4, label="data"); axes[1].plot([0, 10], [b, a * 10 + b], color=C["red"], lw=2, label=f"fitted model y = {a:.1f}x + {b:.1f}")
    axes[1].set_title("regression: input → continuous output"); axes[1].legend(fontsize=8); axes[1].set_xlabel("input"); axes[1].set_ylabel("output")
    png = save_mpl(fig, ctx, "classification_regression")
    return Figure("classification_regression",
                  "Classification versus regression. A classifier maps feature vectors to "
                  "class labels and can be pictured as a decision boundary between the "
                  "classes in feature space. A regression model maps its input directly "
                  "to a continuous output value. This module trains classifiers.", png=png)


def fig_dataset_partition(ctx: BuildContext) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")
    for x0, w, col, lab in [(0, 7, C["blue"], "training set (70%)"), (7, 1.5, C["orange"], "validation\n(15%)"), (8.5, 1.5, C["green"], "test\n(15%)")]:
        ax.add_patch(plt.Rectangle((x0, 0.6), w, 0.8, fc=col, alpha=0.7, ec="white")); ax.text(x0 + w / 2, 1.0, lab, ha="center", va="center", fontsize=8, color="white", weight="bold")
    ax.text(0, 1.75, "(a) partitioning a labelled dataset", fontsize=10, weight="bold", color=C["grey"])
    ax.text(0, 0.25, "train on the training set; check for problems on the validation set;\nreport final performance on the test set, which is never used for training", fontsize=7.5, va="top")
    ax = axes[1]; k = 5; ax.set_xlim(0, 10); ax.set_ylim(-0.5, k + 1.2); ax.axis("off")
    ax.text(0, k + 0.8, "(b) k-fold cross-validation (k = 5)", fontsize=10, weight="bold", color=C["grey"])
    accs = [0.91, 0.88, 0.93, 0.90, 0.89]
    for i in range(k):
        for j in range(k):
            col = C["orange"] if j == i else C["blue"]
            ax.add_patch(plt.Rectangle((j * 1.6, k - 1 - i), 1.5, 0.8, fc=col, alpha=0.7, ec="white"))
        ax.text(8.2, k - 1 - i + 0.4, f"trial {i + 1}: {accs[i]:.2f}", va="center", fontsize=8)
    ax.text(0, -0.3, f"blue = train, orange = validate; mean validation accuracy {np.mean(accs):.2f}", fontsize=8)
    png = save_mpl(fig, ctx, "dataset_partition")
    return Figure("dataset_partition",
                  "Datasets for supervised learning. (a) The labelled data is split into "
                  "a training set, a smaller validation set used to detect problems "
                  "during training, and a test set used only to evaluate the final "
                  "classifier. (b) With limited data, k-fold cross-validation trains k "
                  "times on different partitions and averages the validation accuracy, "
                  "avoiding an over- or under-estimate from one lucky split.", png=png)


def fig_raw_pixel_features(ctx: BuildContext) -> Figure:
    use_source("sklearn_digits")
    d = load_digits()
    X, y = d.data, d.target
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.4, 1.4])
    ax = fig.add_subplot(gs[0]); ax.imshow(d.images[7], cmap="gray_r", interpolation="nearest"); ax.set_title(f"an 8 × 8 digit image, label {y[7]}"); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1]); ax.bar(np.arange(64), X[7], color=C["blue"]); ax.set_title("raw pixels as a 64-element feature vector"); ax.set_xlabel("pixel index"); ax.set_ylabel("intensity")
    ax = fig.add_subplot(gs[2]); sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=6); ax.set_title("the same digits in a 2D PCA representation"); ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
    fig.colorbar(sc, ax=ax, ticks=range(10), fraction=0.04, label="digit")
    png = save_mpl(fig, ctx, "raw_pixel_features")
    return Figure("raw_pixel_features",
                  "Features and representations. For a simple task like handwritten "
                  "digit recognition the raw pixels of an 8 × 8 image, flattened into a "
                  "64-element vector, can serve as the feature vector. Raw pixels are "
                  "highly redundant; a compressed representation such as principal "
                  "component analysis (right, two components) already separates many of "
                  "the classes. Harder tasks need more discriminative representations.",
                  png=png)


def fig_classifier_complexity(ctx: BuildContext) -> Figure:
    X, y = moons(300, 0.3, 0)
    models = [("underfitting: linear classifier (high bias)", LogisticRegression()),
              ("good fit: moderate complexity", SVC(kernel="rbf", C=1.0, gamma=1.0)),
              ("overfitting: 1-nearest-neighbour (high variance)", KNeighborsClassifier(1))]
    ks, tr, va, _ = knn_error_curves()
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
    ax = fig.add_subplot(gs[0, :])
    comp = np.linspace(0, 1, 100)
    bias2 = 0.5 * (1 - comp) ** 2 + 0.02; var = 0.45 * comp ** 3 + 0.02
    ax.plot(comp, bias2, color=C["blue"], label="bias²"); ax.plot(comp, var, color=C["orange"], label="variance"); ax.plot(comp, bias2 + var + 0.05, color=C["ink"], lw=2, label="total error")
    opt = comp[np.argmin(bias2 + var)]; ax.axvline(opt, color=C["grey"], ls="--"); ax.text(opt + 0.01, 0.5, "optimal complexity", fontsize=8, color=C["grey"])
    ax.set_xlabel("classifier complexity →"); ax.set_ylabel("error"); ax.set_yticks([]); ax.legend(fontsize=8); ax.set_title("bias and variance against model complexity")
    ax.text(0.02, 0.05, "underfitting:\nhigh bias, low variance", fontsize=8, color=C["blue"]); ax.text(0.78, 0.05, "overfitting:\nlow bias, high variance", fontsize=8, color=C["orange"])
    for j, (name, clf) in enumerate(models):
        clf.fit(X, y); xx, yy, Z = decision_regions(clf, X)
        ax = fig.add_subplot(gs[1, j]); ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[C["sky"], "#f6c8b0"], alpha=0.5)
        for c in (0, 1):
            ax.plot(X[y == c, 0], X[y == c, 1], "o", color=SERIES[c], ms=3)
        ax.set_title(name, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "classifier_complexity")
    return Figure("classifier_complexity",
                  "Choosing the classifier complexity. Bias is the error inherent in the "
                  "classifier's model; variance is how much its result changes with "
                  "different training data. A simple linear classifier underfits (high "
                  "bias): its straight boundary cannot follow the data. A very complex "
                  "classifier overfits (high variance): its boundary wraps around every "
                  "training point, including outliers. The optimum follows the general "
                  "trend of the data.", png=png)


def fig_train_val_curves(ctx: BuildContext) -> Figure:
    t = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    cases = [("underfitting", 0.45 - 0.05 * t, 0.47 - 0.04 * t),
             ("overfitting", 0.45 * np.exp(-4 * t) + 0.02, 0.45 * np.exp(-3 * t) + 0.12 + 0.25 * t ** 2),
             ("good fit", 0.45 * np.exp(-4 * t) + 0.06, 0.45 * np.exp(-3.5 * t) + 0.09)]
    for ax, (name, tr, va) in zip(axes, cases):
        ax.plot(t, tr, color=C["blue"], label="training error"); ax.plot(t, va, color=C["orange"], label="validation error")
        ax.set_title(name); ax.set_xlabel("training effort →"); ax.set_xticks([]); ax.legend(fontsize=8)
    axes[0].set_ylabel("error"); axes[0].set_ylim(0, 0.6)
    png = save_mpl(fig, ctx, "train_val_curves")
    ks, trk, vak, _ = knn_error_curves()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(len(ks))), y=trk, mode="lines+markers", name="training error", line=dict(color=C["blue"])))
    fig2.add_trace(go.Scatter(x=list(range(len(ks))), y=vak, mode="lines+markers", name="validation error", line=dict(color=C["orange"])))
    fig2.update_layout(title="k-nearest neighbours on 2D data: complexity increases as k decreases",
                       xaxis=dict(tickmode="array", tickvals=list(range(len(ks))), ticktext=[f"k={k}" for k in ks], title="classifier complexity →"),
                       yaxis_title="error rate")
    return Figure("train_val_curves",
                  "Reading the training and validation errors. Underfitting: both stay "
                  "high, because the model cannot fit the data however long it trains. "
                  "Overfitting: the training error keeps falling while the validation "
                  "error turns upward, the signature of a classifier fitting noise. Good "
                  "fit: both fall together. The interactive plot shows the same pattern "
                  "for k-nearest neighbours as k is reduced.",
                  png=png, html=plotly_html(fig2, height=400), kind="plotly")


def fig_unsupervised_kmeans(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(4)
    centres = np.array([[0, 0], [4, 1], [1.5, 4]])
    X = np.vstack([c + rng.normal(0, 0.7, (60, 2)) for c in centres])
    km = KMeans(3, n_init=5, random_state=0).fit(X)
    tests = np.array([[3, 3], [-1, 1], [2, 0.5]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(X[:, 0], X[:, 1], "o", color=C["grey"], ms=4); axes[0].set_title("unlabelled data"); axes[0].set_xticks([]); axes[0].set_yticks([])
    for c in range(3):
        axes[1].plot(X[km.labels_ == c, 0], X[km.labels_ == c, 1], "o", color=SERIES[c], ms=4)
        axes[1].plot(*km.cluster_centers_[c], "X", color=SERIES[c], ms=14, mec="k")
    for tpt in tests:
        lab = km.predict(tpt[None])[0]; axes[1].plot(*tpt, "*", color=SERIES[lab], ms=16, mec="k")
    axes[1].set_title("k-means finds 3 clusters; test points (stars) join the nearest centre"); axes[1].set_xticks([]); axes[1].set_yticks([])
    png = save_mpl(fig, ctx, "unsupervised_kmeans")
    return Figure("unsupervised_kmeans",
                  "Unsupervised learning finds structure in data without labels. k-means, "
                  "met in the segmentation lessons, discovers a chosen number of cluster "
                  "centres in unlabelled data and then labels any new point by the "
                  "centre it is nearest to. Supervised learning remains the dominant "
                  "paradigm for vision, but unsupervised methods are useful when labels "
                  "are unavailable.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_supervised_learning(ctx), fig_classification_regression(ctx), fig_dataset_partition(ctx),
            fig_raw_pixel_features(ctx), fig_classifier_complexity(ctx), fig_train_val_curves(ctx),
            fig_unsupervised_kmeans(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("4.2 Machine learning: classifier complexity")
    noise = st.slider("data noise", 0.05, 0.6, 0.3, 0.05)
    k = st.select_slider("k for k-nearest neighbours (smaller = more complex)", [101, 51, 25, 15, 7, 3, 1], value=15)
    X, y = moons(400, noise, 0)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.4, random_state=0)
    clf = KNeighborsClassifier(k).fit(Xtr, ytr)
    xx, yy, Z = decision_regions(clf, X)
    fig, ax = plt.subplots(figsize=(6, 4.5)); ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[C["sky"], "#f6c8b0"], alpha=0.5)
    for c in (0, 1):
        ax.plot(Xtr[ytr == c, 0], Xtr[ytr == c, 1], "o", color=SERIES[c], ms=3)
    ax.set_title(f"k = {k}: training error {1 - clf.score(Xtr, ytr):.2f}, validation error {1 - clf.score(Xva, yva):.2f}"); ax.set_xticks([]); ax.set_yticks([])
    st.pyplot(fig)
