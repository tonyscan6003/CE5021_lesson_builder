"""Lesson 5.4 — Linear discriminant analysis.

Figures
    pca_vs_lda          PCA maximises variance, which can destroy class separation; LDA preserves it
    projected_means     the scalar projection of a feature vector onto a line, and the transformed means
    which_line          maximum separation of means alone is not enough: scatter matters
    objective_function  J(v) as the line angle is swept, with the transformed histograms (interactive)
    lda_solution        the closed-form solution v* from the within- and between-class scatter
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout)

FIGURES = ["pca_vs_lda", "projected_means", "which_line", "objective_function", "lda_solution"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def two_classes(n=120, seed=1, spread=(3.0, 0.6), angle=35, gap=3.2):
    """Two elongated Gaussian clusters whose long axis is not the discriminative one."""
    rng = np.random.default_rng(seed)
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    off = np.array([0.0, gap])
    c0 = (rng.normal(0, 1, (n, 2)) * np.array(spread) - off / 2) @ R.T
    c1 = (rng.normal(0, 1, (n, 2)) * np.array(spread) + off / 2) @ R.T
    X = np.vstack([c0, c1])
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    return X, y


def two_classes_wide(n=120, seed=5, spread=(3.4, 0.45), offset=(4.0, 2.6)):
    """Two horizontally elongated clusters offset along both axes.

    The horizontal direction separates the means further apart but carries almost
    all the within-class scatter, so it is the *worse* direction to project onto.
    """
    rng = np.random.default_rng(seed)
    off = np.asarray(offset, float)
    c0 = rng.normal(0, 1, (n, 2)) * np.array(spread) - off / 2
    c1 = rng.normal(0, 1, (n, 2)) * np.array(spread) + off / 2
    X = np.vstack([c0, c1])
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    return X, y


def scatter_matrices(X, y):
    """Within-class scatter S_w (sum of per-class scatters) and between-class scatter S_b."""
    mus = np.stack([X[y == c].mean(axis=0) for c in (0, 1)])
    Sw = np.zeros((2, 2))
    for c in (0, 1):
        Z = X[y == c] - mus[c]
        Sw += Z.T @ Z
    d = (mus[0] - mus[1])[:, None]
    Sb = d @ d.T
    return Sw, Sb, mus


def objective(X, y, v):
    """J(v) = (transformed mean difference)^2 / (sum of transformed scatters)."""
    v = v / np.linalg.norm(v)
    Sw, Sb, _ = scatter_matrices(X, y)
    num = float(v @ Sb @ v); den = float(v @ Sw @ v)
    return num / den if den else 0.0


def lda_direction(X, y):
    Sw, _, mus = scatter_matrices(X, y)
    v = np.linalg.solve(Sw, mus[0] - mus[1])
    return v / np.linalg.norm(v)


def unit(angle_deg):
    a = np.deg2rad(angle_deg)
    return np.array([np.cos(a), np.sin(a)])


def draw_projection(ax, X, y, v, title, show_hist=True):
    v = v / np.linalg.norm(v)
    t = X @ v
    classes = np.unique(y)
    for c in classes:
        ax.plot(X[y == c, 0], X[y == c, 1], "o", color=SERIES[c], ms=3, alpha=0.55)
    L = 1.15 * np.abs(t).max()
    ax.plot([-L * v[0], L * v[0]], [-L * v[1], L * v[1]], color=C["ink"], lw=1.5)
    if show_hist:
        n = v[::-1] * np.array([-1, 1])            # unit normal, for offsetting the histograms
        for c in classes:
            proj = np.outer(t[y == c], v)
            ax.plot(proj[:, 0] + 0.6 * n[0], proj[:, 1] + 0.6 * n[1], "|", color=SERIES[c], ms=7, alpha=0.8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_pca_vs_lda(ctx: BuildContext) -> Figure:
    X, y = two_classes()
    v_pca = PCA(2).fit(X).components_[0]
    v_lda = lda_direction(X, y)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    draw_projection(axes[0], X, y, v_pca, f"(a) PCA: the direction of maximum variance\nJ(v) = {objective(X, y, v_pca):.2f}")
    draw_projection(axes[1], X, y, v_lda, f"(b) LDA: the direction that best discriminates\nJ(v) = {objective(X, y, v_lda):.2f}")
    ax = axes[2]
    for c in (0, 1):
        ax.hist(X[y == c] @ (v_pca / np.linalg.norm(v_pca)), bins=30, alpha=0.65, color=SERIES[c], label=f"class {c}")
    ax.set_title("(c) the two classes after the PCA projection:\nno longer separable", fontsize=9)
    ax.set_xlabel("position along the PCA line"); ax.legend(fontsize=8)
    png = save_mpl(fig, ctx, "pca_vs_lda")
    return Figure("pca_vs_lda",
                  "PCA versus LDA. PCA finds the direction of greatest variance without "
                  "using the class labels, and projecting onto it can merge the two "
                  "classes completely (a, c). LDA uses the labels and finds the direction "
                  "that keeps them apart (b), which need not lie anywhere near a "
                  "principal component. Reducing dimensionality usually helps "
                  "classification, but PCA can make it worse.", png=png)


def fig_projected_means(ctx: BuildContext) -> Figure:
    X, y = two_classes(60, 3)
    v = unit(20)
    t = X @ v
    mus = np.stack([X[y == c].mean(axis=0) for c in (0, 1)])
    tm = mus @ v
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1, 1.05]))
    ax = axes[0]
    for c in (0, 1):
        ax.plot(X[y == c, 0], X[y == c, 1], "o", color=SERIES[c], ms=4, alpha=0.5)
    L = 1.2 * np.abs(t).max()
    ax.plot([-L * v[0], L * v[0]], [-L * v[1], L * v[1]], color=C["ink"], lw=1.5)
    ax.text(L * v[0] * 0.75, L * v[1] * 0.75 + 0.6, "line $v$", fontsize=9)
    i = int(np.argmax(X[y == 0] @ np.array([0.0, 1.0])))
    p = X[y == 0][i]; q = (p @ v) * v
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.2, ls="--"))
    ax.plot(*p, "o", color=SERIES[0], ms=9, mec="k"); ax.plot(*q, "s", color=SERIES[0], ms=8, mec="k")
    ax.text(p[0] + 0.2, p[1] + 0.3, "$x_i$", fontsize=11)
    ax.text(q[0] + 0.25, q[1] - 0.7, "$y_i = v^T x_i$", fontsize=10)
    for c in (0, 1):
        ax.plot(*((tm[c]) * v), "X", color=SERIES[c], ms=14, mec="k")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("scalar projection onto a line")
    ax = axes[1]; ax.axis("off")
    ax.text(0, 0.95, "the scalar projection of $x_i$ onto the line $v$ is", fontsize=9.5, va="top")
    ax.text(0.05, 0.80, "$y_i = v \\cdot x_i = v^T x_i$", fontsize=13, va="top")
    ax.text(0, 0.62, "the transformed mean of class 1 follows by\ntaking $v^T$ outside the sum:", fontsize=9.5, va="top")
    ax.text(0.05, 0.42, "$\\hat\\mu_1 = \\frac{1}{n_1}\\sum_i v^T x_i = v^T \\mu_1$", fontsize=13, va="top")
    ax.text(0, 0.22, f"so the transformed means (crosses) are simply the\nuntransformed means projected onto the line:\n"
                     f"$\\hat\\mu_1$ = {tm[0]:.2f},  $\\hat\\mu_2$ = {tm[1]:.2f},  separation {abs(tm[0] - tm[1]):.2f}", fontsize=9, va="top")
    png = save_mpl(fig, ctx, "projected_means")
    return Figure("projected_means",
                  "Projecting the data onto a candidate line. The scalar projection of a "
                  "feature vector onto the line v is the dot product vᵗxᵢ. Because vᵗ can "
                  "be taken outside the sum, the mean of the projected values of a class "
                  "is just the projection of that class's mean, which is what makes the "
                  "LDA objective tractable.", png=png)


def fig_which_line(ctx: BuildContext) -> Figure:
    X, y = two_classes_wide()
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.2),
                             gridspec_kw=dict(height_ratios=[0.34, 1], hspace=0.4))
    for col, (v, name) in enumerate([(unit(0), "horizontal line"), (unit(90), "vertical line")]):
        t = X @ v
        mus = np.stack([X[y == c].mean(axis=0) for c in (0, 1)]) @ v
        scat = sum(((X[y == c] @ v - mus[c]) ** 2).sum() for c in (0, 1))
        draw_projection(axes[0, col], X, y, v,
                        f"{name}: mean separation {abs(mus[0] - mus[1]):.2f}, total scatter {scat:.0f}")
        for c in (0, 1):
            axes[1, col].hist(t[y == c], bins=30, alpha=0.65, color=SERIES[c], label=f"class {c}")
            axes[1, col].axvline(mus[c], color=SERIES[c], lw=2)
        axes[1, col].set_xlabel(f"position along the {name.split()[0]} line"); axes[1, col].legend(fontsize=8)
        axes[1, col].set_title("the classes overlap after projection" if col == 0
                               else "the classes separate after projection", fontsize=9)
    png = save_mpl(fig, ctx, "which_line")
    return Figure("which_line",
                  "Why the separation of the means is not enough. The horizontal line "
                  "(left) gives the wider separation between the two transformed means, "
                  "but both classes have a large variance in that direction, so after "
                  "projection they overlap almost completely. The vertical line (right) "
                  "separates the means less but the projected classes stay apart. The scatter "
                  "of each class, another word for its variance, must appear in the "
                  "criterion.", png=png)


def fig_objective_function(ctx: BuildContext) -> Figure:
    X, y = two_classes_wide()
    angles = np.arange(0, 180, 2)
    J = np.array([objective(X, y, unit(a)) for a in angles])
    best = angles[int(np.argmax(J))]
    v_lda = lda_direction(X, y)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    draw_projection(axes[0], X, y, unit(best), f"the best line: {best}°, J = {J.max():.2f}")
    axes[1].plot(angles, J, color=C["blue"]); axes[1].axvline(best, color=C["red"], ls="--")
    axes[1].set_xlabel("angle of the line $v$ (degrees)"); axes[1].set_ylabel("$J(v)$")
    axes[1].set_title("the objective function over all directions", fontsize=9)
    for c in (0, 1):
        axes[2].hist(X[y == c] @ unit(best), bins=30, alpha=0.65, color=SERIES[c], label=f"class {c}")
    axes[2].set_title("the classes projected onto the best line", fontsize=9); axes[2].legend(fontsize=8)
    axes[2].set_xlabel("position along the line")
    png = save_mpl(fig, ctx, "objective_function")

    show = list(range(0, 180, 10))
    fig2 = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5],
                         subplot_titles=("projection direction", "the projected classes"))
    frames = []
    for a in show:
        v = unit(a); t = X @ v
        L = 1.2 * np.abs(t).max()
        data = [go.Scatter(x=[-L * v[0], L * v[0]], y=[-L * v[1], L * v[1]], mode="lines", line=dict(color=C["ink"], width=2), name="v"),
                go.Histogram(x=t[y == 0], marker_color=SERIES[0], opacity=0.65, nbinsx=30, name="class 0"),
                go.Histogram(x=t[y == 1], marker_color=SERIES[1], opacity=0.65, nbinsx=30, name="class 1")]
        frames.append(go.Frame(name=str(a), data=data, traces=[2, 3, 4],
                               layout=go.Layout(title=f"line at {a}°:  J(v) = {objective(X, y, v):.2f}")))
    for c in (0, 1):
        fig2.add_trace(go.Scatter(x=X[y == c, 0], y=X[y == c, 1], mode="markers",
                                  marker=dict(color=SERIES[c], size=4), name=f"class {c}"), row=1, col=1)
    fig2.add_trace(frames[0].data[0], row=1, col=1)
    fig2.add_trace(frames[0].data[1], row=1, col=2); fig2.add_trace(frames[0].data[2], row=1, col=2)
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(show, "angle = "), title=frames[0].layout.title.text, barmode="overlay")
    fig2.update_yaxes(scaleanchor="x", row=1, col=1)
    return Figure("objective_function",
                  "The LDA objective. J(v) is the squared distance between the "
                  "transformed means divided by the sum of the transformed scatters, so "
                  "maximising it pulls the class means apart while keeping each class "
                  "tight. Sweeping the direction of the line shows a single clear "
                  "maximum, and the projected histograms at that angle are cleanly "
                  "separated.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def fig_lda_solution(ctx: BuildContext) -> Figure:
    X, y = two_classes_wide()
    Sw, Sb, mus = scatter_matrices(X, y)
    v = lda_direction(X, y)
    angles = np.arange(0, 180, 0.5)
    J = np.array([objective(X, y, unit(a)) for a in angles])
    best_num = angles[int(np.argmax(J))]
    ang_closed = np.degrees(np.arctan2(v[1], v[0])) % 180
    sk = LinearDiscriminantAnalysis(solver="eigen").fit(X, y)
    v_sk = sk.scalings_[:, 0] / np.linalg.norm(sk.scalings_[:, 0])
    ang_sk = np.degrees(np.arctan2(v_sk[1], v_sk[0])) % 180
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), gridspec_kw=dict(width_ratios=[1, 1.2]))
    draw_projection(axes[0], X, y, v, f"the closed-form solution $v^*$ at {ang_closed:.0f}°")
    ax = axes[1]; ax.axis("off")
    ax.text(0, 0.97, "writing the objective with the scatter matrices", fontsize=9.5, weight="bold", va="top")
    ax.text(0.03, 0.85, "$J(v) = \\dfrac{v^T S_B\\, v}{v^T S_W\\, v}$", fontsize=14, va="top")
    ax.text(0, 0.64, "setting the derivative to zero gives a generalised\neigenvalue problem, and if $S_W$ can be inverted", fontsize=9, va="top")
    ax.text(0.03, 0.45, "$v^* = S_W^{-1} (\\mu_1 - \\mu_2)$", fontsize=14, va="top")
    ax.text(0, 0.30, f"$S_W$ = [[{Sw[0,0]:.0f}, {Sw[0,1]:.0f}], [{Sw[1,0]:.0f}, {Sw[1,1]:.0f}]]      "
                     f"$\\mu_1 - \\mu_2$ = ({mus[0,0] - mus[1,0]:.2f}, {mus[0,1] - mus[1,1]:.2f})", fontsize=8.5, va="top")
    ax.text(0, 0.17, f"closed form: {ang_closed:.1f}°     numerical sweep: {best_num:.1f}°     scikit-learn: {ang_sk:.1f}°",
            fontsize=9, va="top", color=C["green"])
    ax.text(0, 0.04, "For images $S_W$ is usually singular, which is the problem the next lesson solves.",
            fontsize=8.5, va="top", color=C["grey"])
    png = save_mpl(fig, ctx, "lda_solution")
    return Figure("lda_solution",
                  "Solving for the best line. Written in terms of the between-class "
                  "scatter S_B and the within-class scatter S_W, both computed from the "
                  "untransformed data, the objective becomes a ratio of quadratic forms. "
                  "Setting its derivative to zero gives a generalised eigenvalue problem "
                  "whose solution, when S_W is invertible, is S_W⁻¹(μ₁ − μ₂). The closed "
                  "form, a numerical sweep over all directions and scikit-learn all agree.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_pca_vs_lda(ctx), fig_projected_means(ctx), fig_which_line(ctx),
            fig_objective_function(ctx), fig_lda_solution(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("5.4 Linear discriminant analysis")
    c1, c2, c3 = st.columns(3)
    spread = c1.slider("class elongation", 0.5, 5.0, 3.4, 0.1)
    gap = c2.slider("separation of the means", 0.5, 6.0, 2.4, 0.1)
    rot = c3.slider("orientation of the classes (°)", 0, 180, 90, 5)
    X, y = two_classes(120, 5, spread=(spread, 0.5), angle=rot, gap=gap)
    angle = st.slider("angle of the projection line (°)", 0, 179, 90, 1)
    v = unit(angle)
    v_lda = lda_direction(X, y); best = np.degrees(np.arctan2(v_lda[1], v_lda[0])) % 180
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    draw_projection(axes[0], X, y, v, f"your line: {angle}°, J = {objective(X, y, v):.2f}")
    draw_projection(axes[1], X, y, v_lda, f"LDA solution: {best:.0f}°, J = {objective(X, y, v_lda):.2f}")
    for c in (0, 1):
        axes[2].hist(X[y == c] @ v, bins=30, alpha=0.65, color=SERIES[c])
    axes[2].set_title("your projection"); axes[2].set_xlabel("position along the line")
    st.pyplot(fig)
    angles = np.arange(0, 180, 2)
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.plot(angles, [objective(X, y, unit(a)) for a in angles], color=C["blue"])
    ax.axvline(angle, color=C["grey"]); ax.axvline(best, color=C["red"], ls="--")
    ax.set_xlabel("angle (°)"); ax.set_ylabel("J(v)")
    st.pyplot(fig)
