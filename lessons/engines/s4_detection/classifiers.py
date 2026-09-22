"""Lesson 4.3 — Classifiers: k-nearest neighbours, support vector machines, naive Bayes.

Figures
    precision_recall_tradeoff  score distributions of two classes; moving the threshold trades precision for recall (interactive)
    knn_vote                   majority vote among the k nearest neighbours of a test point
    knn_complexity             decision regions for small and large k
    svm_margin                 the maximal-margin hyperplane and its support vectors
    svm_soft_margin            overlapping classes: small C tolerates misclassification, large C does not
    kernel_trick               circular data lifted to 3D by a radial basis function becomes separable
    bayes_example              the pedestrian / road-sign example with numbers
    naive_bayes_digits         Gaussian naive Bayes on the digits data: confusion matrix
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.datasets import load_digits, make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout, use_source, sync_matrix)
from .machine_learning import moons, decision_regions

FIGURES = ["precision_recall_tradeoff", "knn_vote", "knn_complexity", "svm_margin", "svm_soft_margin",
           "kernel_trick", "bayes_example", "naive_bayes_digits"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def score_populations(seed=0, n=400):
    rng = np.random.default_rng(seed)
    neg = rng.normal(0.35, 0.13, n)           # scores of images without a pedestrian
    pos = rng.normal(0.68, 0.13, n // 2)      # scores of images with a pedestrian
    return neg, pos


def pr_at(neg, pos, thr):
    tp = (pos >= thr).sum(); fp = (neg >= thr).sum(); fn = (pos < thr).sum()
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, tp, fp, fn


def separable_blobs(seed=0, n=40, gap=2.2):
    rng = np.random.default_rng(seed)
    a = rng.normal([0, 0], 0.6, (n, 2)); b = rng.normal([gap, gap], 0.6, (n, 2))
    X = np.vstack([a, b]); y = np.r_[np.zeros(n), np.ones(n)]
    return X, y


def plot_svm(ax, clf, X, y, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200), np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z > 0, levels=[-0.5, 0.5, 1.5], colors=[C["sky"], "#f6c8b0"], alpha=0.4)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=[C["grey"], C["ink"], C["grey"]], linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
    for c in (0, 1):
        ax.plot(X[y == c, 0], X[y == c, 1], "o", color=SERIES[c], ms=4)
    sv = clf.support_vectors_
    ax.plot(sv[:, 0], sv[:, 1], "o", mfc="none", mec="k", ms=10, mew=1.2)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_precision_recall_tradeoff(ctx: BuildContext) -> Figure:
    neg, pos = score_populations()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, thr in zip(axes, [0.4, 0.5, 0.65]):
        ax.hist(neg, bins=40, range=(0, 1), color=C["grey"], alpha=0.6, label="no pedestrian")
        ax.hist(pos, bins=40, range=(0, 1), color=C["green"], alpha=0.6, label="pedestrian")
        ax.axvline(thr, color=C["red"], lw=2)
        p, r, tp, fp, fn = pr_at(neg, pos, thr)
        ax.set_title(f"threshold {thr}: precision {p:.2f}, recall {r:.2f}\nTP {tp}, FP {fp}, FN {fn}", fontsize=9); ax.set_xlabel("classifier score")
    axes[0].legend(fontsize=8)
    png = save_mpl(fig, ctx, "precision_recall_tradeoff")
    thrs = [round(t, 2) for t in np.arange(0.2, 0.86, 0.05)]
    frames = []
    for thr in thrs:
        p, r, tp, fp, fn = pr_at(neg, pos, thr)
        frames.append(go.Frame(name=str(thr), data=[go.Scatter(x=[thr, thr], y=[0, 60], mode="lines", line=dict(color=C["red"], width=3), name="threshold")], traces=[2],
                               layout=go.Layout(title=f"threshold {thr}: precision {p:.2f}, recall {r:.2f}  (TP {tp}, FP {fp}, FN {fn})")))
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=neg, xbins=dict(start=0, end=1, size=0.025), marker_color=C["grey"], opacity=0.6, name="no pedestrian"))
    fig2.add_trace(go.Histogram(x=pos, xbins=dict(start=0, end=1, size=0.025), marker_color=C["green"], opacity=0.6, name="pedestrian"))
    fig2.add_trace(frames[6].data[0]); fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(thrs, "threshold = "), title=frames[6].layout.title.text, barmode="overlay", xaxis_title="classifier score")
    return Figure("precision_recall_tradeoff",
                  "The precision-recall trade-off. A classifier produces a score, and a "
                  "threshold turns it into a decision. A low threshold catches almost "
                  "every pedestrian (high recall) but raises many false alarms (low "
                  "precision); a high threshold makes few false alarms but misses "
                  "pedestrians. For a braking system the misses are the dangerous ones.",
                  png=png, html=plotly_html(fig2, height=420), kind="plotly")


def flip_query(X, y, k_small=3, k_large=7):
    """A query point whose majority vote differs between k_small and k_large."""
    best = None
    for qx in np.arange(X[:, 0].min(), X[:, 0].max(), 0.05):
        for qy in np.arange(X[:, 1].min(), X[:, 1].max(), 0.05):
            q = np.array([qx, qy]); d = np.linalg.norm(X - q, axis=1); o = np.argsort(d)
            v_s = np.bincount(y[o[:k_small]].astype(int), minlength=2)
            v_l = np.bincount(y[o[:k_large]].astype(int), minlength=2)
            if v_s.argmax() != v_l.argmax():
                score = v_s.max() + v_l.max()          # prefer decisive votes on both sides
                if best is None or score > best[0]:
                    best = (score, q)
    return best[1] if best else np.array([0.6, 0.35])


def fig_knn_vote(ctx: BuildContext) -> Figure:
    X, y = moons(80, 0.35, 0)
    q = flip_query(X, y)
    d = np.linalg.norm(X - q, axis=1); order = np.argsort(d)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, k in zip(axes, [3, 7]):
        for c in (0, 1):
            ax.plot(X[y == c, 0], X[y == c, 1], "o", color=SERIES[c], ms=5, alpha=0.6, label=f"class {c}" if k == 3 else None)
        nb = order[:k]; votes = np.bincount(y[nb].astype(int), minlength=2)
        ax.add_patch(plt.Circle(q, d[order[k - 1]] * 1.03, fill=False, ec=C["ink"], ls="--"))
        for i in nb:
            ax.plot([q[0], X[i, 0]], [q[1], X[i, 1]], color=C["grey"], lw=0.8)
        ax.plot(*q, "*", color=SERIES[int(votes.argmax())], ms=22, mec="k")
        ax.set_title(f"k = {k}: {votes[0]} votes for class 0, {votes[1]} for class 1\n→ class {votes.argmax()}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    axes[0].legend(fontsize=8, loc="lower right")
    png = save_mpl(fig, ctx, "knn_vote")
    return Figure("knn_vote",
                  "k-nearest neighbours. The distance from the test point (star) to every "
                  "training point is computed, the k closest are found (dashed circle), "
                  "and the test point takes the majority class among them. k is chosen "
                  "odd to avoid ties. This test point sits between the two classes, so "
                  "the three nearest neighbours and the seven nearest neighbours "
                  "disagree: the prediction depends on k.", png=png)


def fig_knn_complexity(ctx: BuildContext) -> Figure:
    X, y = moons(300, 0.3, 0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, k in zip(axes, [1, 15, 101]):
        clf = KNeighborsClassifier(k).fit(X, y); xx, yy, Z = decision_regions(clf, X)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[C["sky"], "#f6c8b0"], alpha=0.5)
        for c in (0, 1):
            ax.plot(X[y == c, 0], X[y == c, 1], "o", color=SERIES[c], ms=3)
        ax.set_title({1: "k = 1: overfitting, every point gets its own island", 15: "k = 15: a sensible boundary", 101: "k = 101: underfitting, nearly a straight line"}[k], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "knn_complexity")
    return Figure("knn_complexity",
                  "k controls the complexity of the k-NN classifier. A small k lets the "
                  "nearest training points, including noise, dictate the label "
                  "(overfitting); a large k averages over so many points that the "
                  "boundary becomes crude (underfitting). Trying several k against the "
                  "validation set finds the best generalisation.", png=png)


def fig_svm_margin(ctx: BuildContext) -> Figure:
    X, y = separable_blobs(gap=3.0)
    clf = SVC(kernel="linear", C=100).fit(X, y)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    for c in (0, 1):
        ax.plot(X[y == c, 0], X[y == c, 1], "o", color=SERIES[c], ms=4)
    for m, b, col in [(-0.6, 2.4, C["red"]), (-2.5, 4.5, C["red"]), (-1.0, 3.0, C["green"])]:
        xs = np.array([-1.5, 4.5]); ax.plot(xs, m * xs + b, color=col, lw=1.5, ls="--" if col == C["red"] else "-")
    ax.set_title("many lines separate the classes; which is best?", fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-1.5, 4.5); ax.set_ylim(-1.5, 4.5)
    plot_svm(axes[1], clf, X, y, "the SVM chooses the maximal-margin line; circled points are the support vectors")
    png = save_mpl(fig, ctx, "svm_margin")
    return Figure("svm_margin",
                  "The support vector machine is a linear binary classifier. Among all "
                  "the lines that separate two classes it chooses the one with the "
                  "largest margin, the distance to the nearest points of either class. "
                  "Those nearest points (circled) are the support vectors; they alone "
                  "determine the boundary.", png=png)


def fig_svm_soft_margin(ctx: BuildContext) -> Figure:
    X, y = separable_blobs(seed=3, gap=1.8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, Cv in zip(axes, [0.05, 100]):
        clf = SVC(kernel="linear", C=Cv).fit(X, y)
        plot_svm(ax, clf, X, y, f"C = {Cv}: " + ("wide margin, some points misclassified" if Cv < 1 else "narrow margin, boundary pulled by outliers"))
    png = save_mpl(fig, ctx, "svm_soft_margin")
    return Figure("svm_soft_margin",
                  "Real classes overlap, so the soft-margin SVM balances two goals: a "
                  "wide margin and few misclassified training points. The parameter C "
                  "sets the weighting. A small C favours the margin and tolerates "
                  "mistakes; a large C penalises every mistake, so outliers drag the "
                  "boundary and the margin shrinks. C is chosen by trial on the "
                  "validation set.", png=png)


def fig_kernel_trick(ctx: BuildContext) -> Figure:
    X, y = make_circles(n_samples=300, factor=0.4, noise=0.08, random_state=0)
    z = np.exp(-(X ** 2).sum(axis=1) * 1.5)
    clf = SVC(kernel="rbf", C=1, gamma=2).fit(X, y)
    fig = plt.figure(figsize=(13, 4))
    ax = fig.add_subplot(1, 3, 1)
    for c in (0, 1):
        ax.plot(X[y == c, 0], X[y == c, 1], "o", color=SERIES[c], ms=4)
    ax.set_title("(a) not linearly separable in 2D"); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    for c in (0, 1):
        ax.scatter(X[y == c, 0], X[y == c, 1], z[y == c], color=SERIES[c], s=8)
    xx, yy = np.meshgrid(np.linspace(-1.3, 1.3, 10), np.linspace(-1.3, 1.3, 10)); ax.plot_surface(xx, yy, np.full_like(xx, 0.45), alpha=0.25, color=C["grey"])
    ax.set_title("(b) lifted by a radial basis function: a plane separates them", fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax = fig.add_subplot(1, 3, 3); xx, yy, Z = decision_regions(clf, X, pad=0.3)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[C["sky"], "#f6c8b0"], alpha=0.5)
    for c in (0, 1):
        ax.plot(X[y == c, 0], X[y == c, 1], "o", color=SERIES[c], ms=4)
    ax.set_title("(c) the RBF-kernel SVM boundary in the original space"); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "kernel_trick")
    return Figure("kernel_trick",
                  "The kernel trick. Data that no straight line can separate (a) becomes "
                  "separable by a plane after a radial basis function adds a third "
                  "dimension (b). Back in the original space that plane is a circular "
                  "boundary (c). Most SVM implementations offer the RBF kernel by "
                  "default, and for high-dimensional image features it is often needed.",
                  png=png)


def fig_bayes_example(ctx: BuildContext) -> Figure:
    p_d_ped, p_ped = 0.80, 0.60
    p_d_sign, p_sign = 0.15, 0.40
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]; ax.axis("off")
    ax.text(0, 0.95, "Bayes' theorem:  p(c | d) = p(d | c) p(c) / p(d)", fontsize=11, weight="bold")
    ax.text(0, 0.78, "feature d = orange dot present in the image", fontsize=9)
    ax.text(0, 0.62, f"pedestrian class:   p(d | ped) = {p_d_ped},   p(ped) = {p_ped}", fontsize=9)
    ax.text(0, 0.50, f"road-sign class:    p(d | sign) = {p_d_sign},   p(sign) = {p_sign}", fontsize=9)
    ax.text(0, 0.30, f"p(ped | d)  ∝ {p_d_ped} × {p_ped} = {p_d_ped * p_ped:.2f}", fontsize=10, color=SERIES[0])
    ax.text(0, 0.18, f"p(sign | d) ∝ {p_d_sign} × {p_sign} = {p_d_sign * p_sign:.2f}", fontsize=10, color=SERIES[1])
    ax.text(0, 0.02, f"p(d) is the same for both classes and cancels; a pedestrian is {p_d_ped * p_ped / (p_d_sign * p_sign):.0f}× more likely.", fontsize=8, color=C["grey"])
    ax = axes[1]
    ax.bar(["pedestrian", "road sign"], [p_d_ped * p_ped, p_d_sign * p_sign], color=SERIES[:2])
    ax.set_ylabel("p(d | c) p(c)"); ax.set_title("argmax picks the pedestrian class")
    png = save_mpl(fig, ctx, "bayes_example")
    return Figure("bayes_example",
                  "Bayes' theorem on a toy problem. The probability of a class given an "
                  "observed feature is proportional to the probability of the feature "
                  "given the class times the prior probability of the class. Since the "
                  "evidence p(d) is the same for every class it need not be computed; "
                  "the classifier simply picks the class with the largest product.",
                  png=png)


def fig_naive_bayes_digits(ctx: BuildContext) -> Figure:
    use_source("sklearn_digits")
    d = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.3, random_state=0)
    clf = GaussianNB().fit(Xtr, ytr)
    pred = clf.predict(Xte)
    M = confusion_matrix(yte, pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
    sync_matrix(axes[0], M, cmap="Blues", fmt="{:.0f}", fontsize=7)
    axes[0].set_xticks(range(10)); axes[0].set_yticks(range(10)); axes[0].set_xticklabels(range(10)); axes[0].set_yticklabels(range(10)); axes[0].tick_params(length=0)
    axes[0].set_xlabel("predicted digit"); axes[0].set_ylabel("actual digit"); axes[0].set_title(f"naive Bayes on {len(yte)} test digits: accuracy {(pred == yte).mean():.2f}")
    axes[1].axis("off")
    axes[1].text(0, 0.9, "Naive Bayes assumes the features are independent:", fontsize=9)
    axes[1].text(0, 0.76, "p(d | c) = p(d₁ | c) p(d₂ | c) … p(dₙ | c)", fontsize=11)
    axes[1].text(0, 0.56, "Decision rule (argmax over classes):", fontsize=9)
    axes[1].text(0, 0.42, "ĉ = argmax_c  p(c) Π p(dᵢ | c)", fontsize=11)
    axes[1].text(0, 0.18, "Each per-feature probability is estimated from the\ntraining set (here a Gaussian per pixel and class).\nThe independence assumption is crude, yet the\nclassifier works well on many problems.", fontsize=8, va="top", color=C["grey"])
    png = save_mpl(fig, ctx, "naive_bayes_digits")
    return Figure("naive_bayes_digits",
                  "The naive Bayes classifier applied to handwritten digits with the 64 "
                  "pixels as features. Treating the features as independent turns the "
                  "class likelihood into a product of one-dimensional terms that are "
                  "cheap to estimate from statistics of the training set. It is a "
                  "multi-class model; the confusion matrix shows which digits it "
                  "confuses.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_precision_recall_tradeoff(ctx), fig_knn_vote(ctx), fig_knn_complexity(ctx), fig_svm_margin(ctx),
            fig_svm_soft_margin(ctx), fig_kernel_trick(ctx), fig_bayes_example(ctx), fig_naive_bayes_digits(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("4.3 Classifiers")
    tab1, tab2, tab3 = st.tabs(["k-NN", "SVM", "Precision / recall threshold"])
    with tab1:
        k = st.slider("k", 1, 101, 15, 2); noise = st.slider("noise", 0.05, 0.6, 0.3, 0.05)
        X, y = moons(300, noise, 0); clf = KNeighborsClassifier(k).fit(X, y); xx, yy, Z = decision_regions(clf, X)
        fig, ax = plt.subplots(figsize=(6, 4.5)); ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[C["sky"], "#f6c8b0"], alpha=0.5)
        for c in (0, 1):
            ax.plot(X[y == c, 0], X[y == c, 1], "o", color=SERIES[c], ms=3)
        ax.set_xticks([]); ax.set_yticks([]); st.pyplot(fig)
    with tab2:
        kern = st.radio("kernel", ["linear", "rbf"], horizontal=True)
        Cv = st.select_slider("C", [0.01, 0.1, 1, 10, 100], value=1)
        gap = st.slider("class separation", 0.5, 3.5, 1.8, 0.1)
        X, y = separable_blobs(seed=3, gap=gap)
        clf = SVC(kernel=kern, C=Cv, gamma="scale").fit(X, y)
        fig, ax = plt.subplots(figsize=(6, 4.5)); plot_svm(ax, clf, X, y, f"{kern} SVM, C = {Cv}: {len(clf.support_vectors_)} support vectors"); st.pyplot(fig)
    with tab3:
        neg, pos = score_populations()
        thr = st.slider("threshold", 0.0, 1.0, 0.5, 0.01)
        p, r, tp, fp, fn = pr_at(neg, pos, thr)
        fig, ax = plt.subplots(figsize=(7, 3)); ax.hist(neg, bins=40, range=(0, 1), color=C["grey"], alpha=0.6); ax.hist(pos, bins=40, range=(0, 1), color=C["green"], alpha=0.6); ax.axvline(thr, color=C["red"], lw=2)
        ax.set_title(f"precision {p:.2f}, recall {r:.2f}"); st.pyplot(fig)
