"""Lesson 1.6 — Overfitting and regularisation.

Figures
    bias_variance          the U-shaped total error and under/optimal/over fits
    train_val_curves       underfitting, overfitting and a good fit, with early stopping
    curse_of_dimensionality  sparsity as dimensions are added, and the data required
    l1_l2_regularisation   constraint regions and what the penalties do to the weights
    dropout_mask           units dropped per mini-batch, with the inverted-dropout scaling
    spatial_dropout        dropping whole feature-map channels instead of pixels
    data_augmentation      affine transforms on one credited photograph

The fits in bias_variance and train_val_curves are real polynomial least-squares
fits to sampled data, not drawn curves.

Source: reference/legacy_md/lesson_1_7_overfitting_regularisation + lesson_1_7_script.
"""
from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout, load_image)

FIGURES = ["bias_variance", "train_val_curves", "curse_of_dimensionality",
           "l1_l2_regularisation", "dropout_mask", "spatial_dropout",
           "data_augmentation"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def _truth(x):
    return np.sin(1.6 * x) + 0.30 * x


def sample(n, noise=0.32, seed=0, lo=-3.0, hi=3.0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(lo, hi, n))
    return x, _truth(x) + rng.normal(0, noise, n)


def poly_fit_error(degree, n=22, noise=0.32, trials=60, seed=0):
    """Mean training and test error for polynomial fits of a given degree.

    Averaged over many resampled training sets, so the variance component is
    measured rather than asserted.
    """
    tr, te = [], []
    xt = np.linspace(-3, 3, 200)
    for t in range(trials):
        x, y = sample(n, noise, seed + t)
        try:
            c = np.polyfit(x, y, degree)
        except Exception:
            continue
        tr.append(np.mean((np.polyval(c, x) - y) ** 2))
        te.append(np.mean((np.polyval(c, xt) - _truth(xt)) ** 2))
    return float(np.mean(tr)), float(np.mean(te))


def training_curves(kind, epochs=90, seed=1):
    """Synthetic but rule-driven training/validation error curves."""
    rng = np.random.default_rng(seed)
    e = np.arange(epochs)
    n = lambda s: rng.normal(0, s, epochs)
    if kind == "underfit":
        tr = 0.42 + 0.12 * np.exp(-e / 18) + n(0.006)
        va = 0.46 + 0.12 * np.exp(-e / 18) + n(0.008)
    elif kind == "overfit":
        tr = 0.05 + 0.55 * np.exp(-e / 16) + n(0.006)
        va = 0.17 + 0.45 * np.exp(-e / 12) + 0.0032 * np.maximum(e - 28, 0) + n(0.008)
    else:
        tr = 0.07 + 0.52 * np.exp(-e / 17) + n(0.006)
        va = 0.10 + 0.52 * np.exp(-e / 17) + n(0.008)
    return e, tr, va


def inverted_dropout(x, rate, seed=0):
    """Inverted dropout: zero with probability `rate`, scale the rest by 1/(1-rate)."""
    rng = np.random.default_rng(seed)
    mask = (rng.random(x.shape) >= rate).astype(float)
    return x * mask / (1.0 - rate), mask


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _truth_bv(x):
    return np.sin(1.6 * x + 0.5) + 0.3 * x


def bias_variance(degrees, n=18, noise=0.4, seed=4):
    """Exact bias-variance decomposition for polynomial least squares.

    The classic setting: the input positions x are fixed and only the noise on
    the targets changes from one training set to the next. The fit is then
    linear in y, so with H the matrix mapping training targets to predictions,
        mean fit  = H f(x)                  -> bias²    = mean (H f(x) − f)²
        spread    = noise² · ||rows of H||² -> variance
    and the expected training error is the training-set bias plus
    noise²·(1 − p/n). Computed exactly — nothing is sampled, so there is no
    Monte Carlo noise in the curves. Legendre basis for numerical stability;
    errors are measured over the span of the data (no extrapolation).
    """
    from numpy.polynomial import legendre as Lg
    x = np.sort(np.random.default_rng(seed).uniform(-3, 3, n))
    xt = np.linspace(x.min(), x.max(), 300)
    rows = []
    for d in degrees:
        A, At = Lg.legvander(x / 3, d), Lg.legvander(xt / 3, d)
        P = np.linalg.pinv(A)
        H, Htr = At @ P, A @ P
        b2 = np.mean((H @ _truth_bv(x) - _truth_bv(xt)) ** 2)
        var = noise ** 2 * np.mean(np.sum(H ** 2, 1))
        tr = np.mean((Htr @ _truth_bv(x) - _truth_bv(x)) ** 2) + noise ** 2 * (1 - (d + 1) / n)
        rows.append((b2, var, tr))
    b2, var, tr = map(np.array, zip(*rows))
    return x, b2, var, tr


def fig_bias_variance(ctx: BuildContext) -> Figure:
    degrees = list(range(1, 10))
    NOISE = 0.4
    x, b2, var, tr = bias_variance(degrees, noise=NOISE)
    tot = b2 + var
    best = degrees[int(np.argmin(tot))]
    # The claims the figure makes: bias falls, variance rises, the total has an
    # interior minimum, and the training error falls throughout.
    assert np.all(np.diff(b2) < 1e-3) and np.all(np.diff(var) > 0), (b2, var)
    assert 1 < best < degrees[-1] and np.all(np.diff(tr) < 0), (best, tr)

    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1], hspace=0.30, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    YMAX = 0.8
    ax.plot(degrees, tot, color=C["red"], lw=2.6, marker="o", ms=4,
            label="total error = bias² + variance")
    ax.plot(degrees, b2, color=C["blue"], lw=2.0, marker="o", ms=4, label="bias²")
    ax.plot(degrees, var, color=C["orange"], lw=2.0, marker="o", ms=4, label="variance")
    ax.plot(degrees, tr, color=C["grey"], lw=1.6, ls="--", label="training error")
    for d, v in zip(degrees, tot):
        if v > YMAX:                       # off the scale: say so rather than clip silently
            ax.annotate(f"{v:.1f} (off scale)", xy=(d, YMAX), xytext=(d, YMAX * 0.80),
                        fontsize=8.5, color=C["red"], ha="center",
                        arrowprops=dict(arrowstyle="->", color=C["red"], lw=1))
    ax.axvline(best, color=C["green"], ls=":", lw=1.5)
    ax.text(best + 0.1, YMAX * 0.62, f"optimum complexity\n(degree {best})",
            fontsize=9, color=C["green"])
    ax.text(1.1, YMAX * 0.93, "high bias\n(underfitting)", fontsize=9, color=C["grey"],
            va="top")
    ax.text(degrees[-1] - 1.9, YMAX * 0.62, "high variance\n(overfitting)", fontsize=9,
            color=C["grey"])
    ax.set_ylim(0, YMAX); ax.set_xticks(degrees)
    ax.set_xlabel("model complexity (polynomial degree)")
    ax.set_ylabel("mean squared error")
    ax.set_title(f"Bias and variance of least-squares polynomial fits to {len(x)} noisy "
                 "points", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper center", ncol=2)

    y = _truth_bv(x) + np.random.default_rng(11).normal(0, NOISE, len(x))
    xt = np.linspace(x.min(), x.max(), 300)
    for col, (deg, name) in enumerate([(1, "underfitting: degree 1"),
                                       (best, f"good fit: degree {best}"),
                                       (degrees[-1],
                                        f"overfitting: degree {degrees[-1]}")]):
        ax = fig.add_subplot(gs[1, col])
        ax.scatter(x, y, s=22, color=C["grey"], zorder=3, label="training data")
        ax.plot(xt, _truth_bv(xt), color=C["light"], lw=3, zorder=1, label="true function")
        ax.plot(xt, np.polyval(np.polyfit(x, y, deg), xt), color=SERIES[col], lw=2,
                zorder=4, label="fit")
        ax.set_ylim(-3.2, 3.2); ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.legend(fontsize=7.5, loc="upper left")
    png = save_mpl(fig, ctx, "bias_variance")
    return Figure("bias_variance",
                  "**Bias** is the error inherent in the complexity of the model, and "
                  "**variance** is the amount by which the result changes when different "
                  "training data is used. Here a polynomial is fitted by least squares to "
                  f"{len(x)} noisy samples of a known function (bottom), and both "
                  "quantities are calculated exactly over all possible draws of the noise: "
                  "**bias²** is the squared difference between the average fit and the true "
                  "function, and **variance** is the spread of the individual fits about the "
                  "average fit. As the complexity increases, the bias decreases and the "
                  "variance increases, so the total error, their sum, follows a U-shaped "
                  f"curve with a minimum at an optimal complexity, degree {best} here. A "
                  "low-complexity model is too simple to fit the data, which is known as "
                  "**underfitting**. A high-complexity model fits the training points "
                  "closely, including the noise, and the fit changes significantly when the "
                  "training data changes, which is known as **overfitting**. It should be "
                  "noted that the training error (dashed) decreases throughout as the "
                  "complexity increases, and for this reason it cannot be used to select a "
                  "model. The same decomposition applies to a neural network, in which the "
                  "complexity is set by the number of layers and parameters.",
                  png=png)


def fig_train_val_curves(ctx: BuildContext) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, kind, title in zip(axes, ["underfit", "overfit", "good"],
                               ["(a) underfitting", "(b) overfitting",
                                "(c) a good fit"]):
        e, tr, va = training_curves(kind)
        ax.plot(e, tr, color=C["blue"], lw=2, label="training error")
        ax.plot(e, va, color=C["red"], lw=2, label="validation error")
        if kind == "overfit":
            stop = int(np.argmin(va))
            ax.axvline(stop, color=C["green"], ls="--", lw=1.6)
            ax.scatter([stop], [va[stop]], color=C["green"], s=60, zorder=5)
            ax.annotate("early stopping:\nstop here", xy=(stop, va[stop]),
                        xytext=(stop + 6, va[stop] + 0.22), fontsize=8.5,
                        color=C["green"],
                        arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.2))
            ax.annotate("divergence", xy=(78, va[78]), xytext=(52, va[78] + 0.16),
                        fontsize=8.5, color=C["grey"],
                        arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1))
        ax.set_xlabel("epoch"); ax.set_ylim(0, 0.75)
        ax.set_title(title, fontsize=10.5)
        if ax is axes[0]:
            ax.set_ylabel("error"); ax.legend(fontsize=8.5)
    png = save_mpl(fig, ctx, "train_val_curves")
    return Figure("train_val_curves",
                  "Underfitting and overfitting can be detected during training from the "
                  "error on the training and validation sets. (a) **Underfitting**: both "
                  "errors remain high. This is the result of high bias; the data cannot be "
                  "fitted by the model, and further training does not help. "
                  "(b) **Overfitting**: the two errors diverge. The training error continues "
                  "to decrease, while the validation error reaches a minimum and then "
                  "increases, as the training set is fitted more and more closely and the "
                  "ability to predict new data is lost. For this reason validation data must "
                  "be used during training, as the training error alone is misleading. This "
                  "is also the basis of **early stopping**, in which training is stopped at "
                  "the minimum of the validation curve. (c) **A good fit**: both errors "
                  "decrease together to a low value. Underfitting is addressed by an "
                  "increase in model complexity, and overfitting by more data, reduced "
                  "complexity or regularisation.",
                  png=png)


def fig_curse_of_dimensionality(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(4)
    n = 30
    fig = plt.figure(figsize=(12.5, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.3)

    pts = rng.uniform(0, 1, (n, 3))
    lbl = (pts[:, 0] + 0.35 * rng.normal(size=n) > 0.5)
    cols = [C["blue"] if v else C["orange"] for v in lbl]

    ax = fig.add_subplot(gs[0])
    ax.scatter(pts[:, 0], np.zeros(n), c=cols, s=34)
    ax.set_yticks([]); ax.set_xlim(-0.05, 1.05)
    ax.set_title("1D: classes overlap", fontsize=10)

    ax = fig.add_subplot(gs[1])
    ax.scatter(pts[:, 0], pts[:, 1], c=cols, s=34)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("2D: separable, but sparser", fontsize=10)

    ax = fig.add_subplot(gs[2], projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=26)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("3D: sparser still", fontsize=10)

    # mean nearest-neighbour distance for the same 30 points in d dimensions
    ax = fig.add_subplot(gs[3])
    dims = np.arange(1, 13)
    dist = []
    for d in dims:
        p = rng.uniform(0, 1, (n, int(d)))
        dd = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dd, np.inf)
        dist.append(dd.min(1).mean())
    ax.plot(dims, dist, color=C["red"], lw=2, marker="o", ms=4,
            label="mean nearest-neighbour distance")
    ax2 = ax.twinx()
    ax2.semilogy(dims, 10.0 ** dims, color=C["blue"], lw=2, ls="--",
                 label="samples for constant density")
    ax.set_xlabel("dimensions")
    ax.set_ylabel("mean NN distance", color=C["red"])
    ax2.set_ylabel("samples needed", color=C["blue"])
    ax.set_title("sparsity grows, data required explodes", fontsize=10)
    png = save_mpl(fig, ctx, "curse_of_dimensionality")
    return Figure("curse_of_dimensionality",
                  "When an image is applied to a classifier, each pixel is treated as a "
                  "dimension, so even a small 256 × 256 image has 65,536 dimensions. The "
                  "addition of dimensions helps to discriminate between classes (here the two "
                  "classes overlap in 1D and are separated in 2D), but with the same number "
                  "of data points, the distances between the points increase. This is known "
                  "as **sparsity**, and it increases further with more dimensions. Sparse "
                  "training data leads to overfitting, and to compensate, the amount of data "
                  "must increase *exponentially* with each dimension in order to maintain "
                  "generalisation. This exponential requirement is known as the curse of "
                  "dimensionality. Raw images are applied to neural networks, so the curse of "
                  "dimensionality applies; however, as the convolution layers operate locally "
                  "and features are extracted before classification in the final layers, "
                  "neural networks are less susceptible to it than classical methods.",
                  png=png)


def fig_l1_l2_regularisation(ctx: BuildContext) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))

    # (a) constraint regions, with the constrained optima actually solved for
    # rather than placed by hand, so the contour really does touch the boundary
    ax = axes[0]
    w_hat = np.array([0.45, 2.00])          # unconstrained least-squares optimum
    A = np.array([[1.0, 0.25], [0.25, 0.62]])

    def J(W):
        d = W - w_hat
        return np.einsum("...i,ij,...j->...", d, A, d)

    th = np.linspace(0, 2 * np.pi, 4001)
    circle = np.stack([np.cos(th), np.sin(th)], -1)             # L2 ball, c = 1
    s = np.linspace(-1, 1, 4001)                                 # L1 ball, c = 1
    diamond = np.concatenate([np.stack([s, 1 - np.abs(s)], -1),
                              np.stack([s, np.abs(s) - 1], -1)])
    w_l2 = circle[np.argmin(J(circle))]
    w_l1 = diamond[np.argmin(J(diamond))]
    on_corner = np.min(np.abs(w_l1)) < 1e-2

    gx, gy = np.meshgrid(np.linspace(-2.0, 2.6, 300), np.linspace(-1.6, 2.6, 300))
    Z = J(np.stack([gx, gy], -1))
    # The L2 optimum sits on a lower contour than the L1 one, so sort the levels
    # (matplotlib requires increasing) and keep each colour with its own level.
    lv = sorted([(J(w_l2) * 0.35, C["light"]), (float(J(w_l2)), C["blue"]),
                 (float(J(w_l1)), C["orange"])], key=lambda p: p[0])
    ax.contour(gx, gy, Z, levels=[v for v, _ in lv],
               colors=[c for _, c in lv], linewidths=1.2)
    ax.plot(np.cos(th), np.sin(th), color=C["blue"], lw=2.2,
            label="L2: $w_1^2+w_2^2 \\leq c$")
    ax.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color=C["orange"], lw=2.2,
            label="L1: $|w_1|+|w_2| \\leq c$")
    ax.scatter(*w_hat, color=C["grey"], s=55, zorder=5, marker="x")
    ax.text(w_hat[0] + 0.12, w_hat[1], "unconstrained\noptimum", fontsize=8,
            color=C["grey"], va="center")
    ax.scatter(*w_l1, color=C["orange"], s=75, zorder=6)
    ax.scatter(*w_l2, color=C["blue"], s=75, zorder=6)
    ax.annotate(f"L1 optimum on a corner:\n$w_1$ = {w_l1[0]:.2g}"
                if on_corner else f"L1 optimum ({w_l1[0]:.2f}, {w_l1[1]:.2f})",
                xy=tuple(w_l1), xytext=(-1.95, 1.35), fontsize=8.5, color=C["orange"],
                arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1))
    ax.annotate(f"L2 optimum: both\nweights non-zero", xy=tuple(w_l2),
                xytext=(0.95, -1.15), fontsize=8.5, color=C["blue"],
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1))
    ax.axhline(0, color=C["grey"], lw=0.7); ax.axvline(0, color=C["grey"], lw=0.7)
    ax.set_xlim(-2.0, 2.6); ax.set_ylim(-1.6, 2.6); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a) the constraint regions", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower left")

    # (b) fitted weights with and without a penalty, measured
    x, y = sample(22, seed=7)
    X = np.vander(x, 12)
    Xs = X / np.abs(X).max(0)
    ax = axes[1]
    for lam, col, name in [(0.0, C["red"], "no penalty"),
                           (1e-3, C["orange"], "L2, λ = 1e−3"),
                           (1e-1, C["blue"], "L2, λ = 1e−1")]:
        w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ y)
        ax.plot(np.abs(w), color=col, lw=2, marker="o", ms=4, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("weight index"); ax.set_ylabel("|weight|")
    ax.set_title("(b) L2 shrinks the weights", fontsize=10.5)
    ax.legend(fontsize=8.5)

    # (c) the resulting fits
    ax = axes[2]
    xt = np.linspace(-3, 3, 300)
    Xt = np.vander(xt, 12) / np.abs(X).max(0)
    ax.scatter(x, y, s=22, color=C["grey"], zorder=3)
    ax.plot(xt, _truth(xt), color=C["light"], lw=3, zorder=1)
    for lam, col, name in [(0.0, C["red"], "no penalty"),
                           (1e-1, C["blue"], "L2, λ = 1e−1")]:
        w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ y)
        ax.plot(xt, Xt @ w, color=col, lw=2, zorder=4, label=name)
    ax.set_ylim(-3.6, 3.6); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(c) and so smooths the fit\n(the unpenalised fit leaves the frame)",
                 fontsize=10.5)
    ax.legend(fontsize=8.5)
    png = save_mpl(fig, ctx, "l1_l2_regularisation")
    return Figure("l1_l2_regularisation",
                  "In L1 and L2 regularisation, large parameter values are penalised in the "
                  "loss function, so that the model is constrained towards parameters of more "
                  "equal size and better generalisation is obtained. In **L1** "
                  "regularisation the absolute values of the weights are penalised, and in "
                  "**L2** regularisation the squares. (a) The difference in behaviour is due "
                  "to the shape of the constraint region: the L1 diamond has corners on the "
                  "axes, so the optimum is often located exactly at a corner and a weight is "
                  "driven to zero, giving a sparse solution. The L2 circle has no corners, so "
                  "the weights are reduced but not eliminated. (b) When the same degree-11 "
                  "polynomial is fitted with increasing λ, the weights are reduced by orders "
                  "of magnitude, and (c) the fit becomes correspondingly smoother. In PyTorch, "
                  "L2 regularisation is applied through the weight_decay argument of the "
                  "optimiser, and L1 regularisation is added to the loss.",
                  png=png)


def fig_dropout_mask(ctx: BuildContext) -> Figure:
    rate = 0.4
    layers = [6, 8, 8, 4]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.0))
    for ax, seed, title in zip(axes[:3], (1, 2, 3),
                               ["mini-batch 1", "mini-batch 2", "mini-batch 3"]):
        rng = np.random.default_rng(seed)
        pos, alive = [], []
        for li, nL in enumerate(layers):
            ys = np.linspace(-(nL - 1) / 2, (nL - 1) / 2, nL)
            keep = np.ones(nL, bool) if li in (0, len(layers) - 1) else \
                rng.random(nL) >= rate
            pos.append([(li * 1.35, y) for y in ys]); alive.append(keep)
        for li in range(len(layers) - 1):
            for (p, ka) in zip(pos[li], alive[li]):
                for (q, kb) in zip(pos[li + 1], alive[li + 1]):
                    if ka and kb:
                        ax.plot([p[0], q[0]], [p[1], q[1]], color=C["light"],
                                lw=0.7, zorder=1)
        for li, (ps, keep) in enumerate(zip(pos, alive)):
            ax.scatter([p[0] for p in ps], [p[1] for p in ps], s=105,
                       facecolors=["white" if k else C["light"] for k in keep],
                       edgecolors=[C["blue"] if k else C["grey"] for k in keep],
                       linewidths=[1.8 if k else 1.0 for k in keep], zorder=3)
            for p, k in zip(ps, keep):
                if not k:
                    ax.plot([p[0] - 0.12, p[0] + 0.12], [p[1] - 0.12, p[1] + 0.12],
                            color=C["red"], lw=1.4, zorder=4)
                    ax.plot([p[0] - 0.12, p[0] + 0.12], [p[1] + 0.12, p[1] - 0.12],
                            color=C["red"], lw=1.4, zorder=4)
        ax.set_xlim(-0.5, 4.55); ax.set_ylim(-4.6, 4.6); ax.axis("off")
        ax.set_title(f"{title}\na different subset is dropped", fontsize=9.5)

    ax = axes[3]
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out, mask = inverted_dropout(a, rate, seed=2)
    idx = np.arange(len(a))
    ax.bar(idx - 0.2, a, width=0.4, color=C["light"], edgecolor=C["grey"],
           label="before dropout")
    ax.bar(idx + 0.2, out, width=0.4, color=C["blue"], edgecolor=C["grey"],
           label=f"after, rate = {rate}")
    for i, (b, o) in enumerate(zip(a, out)):
        ax.text(i + 0.2, o + 0.15, "0" if o == 0 else f"{o:.2f}", ha="center",
                fontsize=8)
    ax.set_title(f"inverted dropout: survivors scaled\nby 1/(1−{rate}) = "
                 f"{1 / (1 - rate):.3g}, so the mean is preserved", fontsize=9.5)
    ax.set_xticks(idx); ax.set_xlabel("unit"); ax.legend(fontsize=8)
    png = save_mpl(fig, ctx, "dropout_mask")
    return Figure("dropout_mask",
                  "In **dropout**, nodes are randomly removed during training: at each "
                  "training step a unit is dropped with a given probability, so that a "
                  "sparser, and different, network is produced at each step. The network is "
                  "prevented from relying on any particular neuron or path, and a "
                  "regularising effect similar to an L2 penalty is obtained, as the weight "
                  "values are kept small. The right-hand panel shows that in **inverted "
                  "dropout** the remaining activations are divided by (1 − rate), so that the "
                  "expected output of the layer is unchanged and no rescaling is required at "
                  "inference time. In PyTorch, dropout is implemented as an independent layer, nn.Dropout, "
                  "which is active only in training mode. "
                  "If too large a rate is used, convergence is slowed or underfitting occurs.",
                  png=png)


def fig_spatial_dropout(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(6)
    c, n = 6, 8
    # correlated feature maps: smooth blobs, as in a real CNN
    maps = []
    for i in range(c):
        m = rng.normal(size=(n, n))
        m = cv2.GaussianBlur(m, (0, 0), 1.6)
        maps.append((m - m.min()) / (m.max() - m.min()))
    maps = np.array(maps)

    std_mask = (rng.random(maps.shape) >= 0.4)
    chan_keep = rng.random(c) >= 0.4
    fig, axes = plt.subplots(3, c, figsize=(11.5, 5.6))
    for j in range(c):
        axes[0, j].imshow(maps[j], cmap="viridis", vmin=0, vmax=1)
        axes[1, j].imshow(maps[j] * std_mask[j], cmap="viridis", vmin=0, vmax=1)
        if chan_keep[j]:
            axes[2, j].imshow(maps[j], cmap="viridis", vmin=0, vmax=1)
        else:
            axes[2, j].imshow(np.zeros((n, n)), cmap="viridis", vmin=0, vmax=1)
            axes[2, j].text(n / 2 - 0.5, n / 2 - 0.5, "dropped", ha="center",
                            va="center", fontsize=8, color="white")
        for r in range(3):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[0, j].set_title(f"map {j + 1}", fontsize=9)
    for r, lab in enumerate(["feature maps", "standard dropout\n(random pixels)",
                             "spatial dropout\n(whole channels)"]):
        axes[r, 0].set_ylabel(lab, fontsize=9)
    png = save_mpl(fig, ctx, "spatial_dropout")
    return Figure("spatial_dropout",
                  "Dropout was designed for fully connected networks, in which the removal of "
                  "the connections of a neuron changes what is computed by the layer. In a "
                  "convolutional network dropout is less effective. The kernel is applied "
                  "across a *region* of the preceding feature map, and neighbouring pixels in "
                  "a feature map are highly correlated, as shown in the top row. When "
                  "scattered pixels are removed (middle row), nearly the same information is "
                  "still carried by the surrounding values, so the output of the convolution "
                  "is changed very little and little regularisation is obtained. In "
                  "**spatial dropout** (bottom row), entire feature maps are dropped instead. "
                  "This has a substantial effect on specific kernel channels, which are "
                  "prevented from relying on particular correlated features during a "
                  "training interval.",
                  png=png)


def fig_data_augmentation(ctx: BuildContext) -> Figure:
    img = load_image("labrador", max_side=320)
    h, w = img.shape[:2]

    def warp(M):
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    ops = [("original", img),
           ("horizontal flip", img[:, ::-1]),
           ("rotate 14°", warp(cv2.getRotationMatrix2D((w / 2, h / 2), 14, 1.0))),
           ("zoom ×1.25", warp(cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.25))),
           ("translate", warp(np.float32([[1, 0, 0.12 * w], [0, 1, -0.08 * h]]))),
           ("shear", warp(np.float32([[1, 0.18, -0.09 * w], [0, 1, 0]]))),
           ("brightness", np.clip(img.astype(np.int16) + 42, 0, 255).astype(np.uint8)),
           ("random crop", cv2.resize(img[int(0.12 * h):int(0.92 * h),
                                          int(0.08 * w):int(0.88 * w)], (w, h)))]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.0))
    for ax, (name, im) in zip(axes.ravel(), ops):
        ax.imshow(im)
        ax.set_title(name, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(C["grey"] if name != "original" else C["red"])
            s.set_linewidth(1.6 if name == "original" else 0.8)
    png = save_mpl(fig, ctx, "data_augmentation")
    return Figure("data_augmentation",
                  "Ideally, overfitting would be prevented by training on new data rather "
                  "than repeating the same examples, but labelled datasets are finite and "
                  "expensive to extend. In **data augmentation** the existing data is "
                  "modified so that variations of the images can be used as new training "
                  "examples. The simplest and most effective approach is the use of **affine "
                  "transforms** (rotation, translation, scaling and shear) together with "
                  "flips and photometric changes such as brightness. More complex "
                  "augmentations are available, in which segments of an image are removed or "
                  "parts of several images are combined, but the basic transforms are "
                  "effective. In PyTorch these are implemented as torchvision transforms, "
                  "applied to each image as it is loaded during training. With too little augmentation the network still "
                  "overfits, and with too much it underfits, so the appropriate level is "
                  "usually found by experiment together with the other regularisation "
                  "methods.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_bias_variance(ctx), fig_train_val_curves(ctx),
            fig_curse_of_dimensionality(ctx), fig_l1_l2_regularisation(ctx),
            fig_dropout_mask(ctx), fig_spatial_dropout(ctx),
            fig_data_augmentation(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("1.6 Overfitting and regularisation")
    tab1, tab2, tab3 = st.tabs(["Model complexity", "L2 penalty", "Dropout"])

    with tab1:
        deg = st.slider("polynomial degree", 1, 15, 3)
        n = st.slider("training points", 8, 60, 22)
        x, y = sample(int(n), seed=7)
        xt = np.linspace(-3, 3, 300)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.scatter(x, y, s=24, color=C["grey"], zorder=3, label="training data")
        ax.plot(xt, _truth(xt), color=C["light"], lw=3, label="true function")
        ax.plot(xt, np.polyval(np.polyfit(x, y, deg), xt), color=C["red"], lw=2,
                label=f"degree {deg} fit")
        ax.set_ylim(-3.5, 3.5); ax.legend(fontsize=8.5)
        st.pyplot(fig)
        tr, te = poly_fit_error(deg, n=int(n), trials=30)
        c1, c2 = st.columns(2)
        c1.metric("training error", f"{tr:.4f}")
        c2.metric("error on unseen data", f"{te:.4f}")
        if te > tr * 4 and deg > 6:
            st.warning("Training error far below test error — this is overfitting.")

    with tab2:
        lam = st.select_slider("L2 penalty λ",
                               options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                               value=1e-3)
        x, y = sample(22, seed=7)
        X = np.vander(x, 12); Xs = X / np.abs(X).max(0)
        w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(12), Xs.T @ y)
        xt = np.linspace(-3, 3, 300)
        Xt = np.vander(xt, 12) / np.abs(X).max(0)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        axes[0].scatter(x, y, s=22, color=C["grey"], zorder=3)
        axes[0].plot(xt, _truth(xt), color=C["light"], lw=3)
        axes[0].plot(xt, Xt @ w, color=C["blue"], lw=2)
        axes[0].set_ylim(-3.5, 3.5); axes[0].set_title("fit")
        axes[1].semilogy(np.abs(w), color=C["blue"], marker="o")
        axes[1].set_title("|weights|"); axes[1].set_xlabel("index")
        st.pyplot(fig)
        st.write(f"largest |weight| = {np.abs(w).max():.4g}")

    with tab3:
        rate = st.slider("dropout rate", 0.0, 0.9, 0.4, 0.05)
        seed = st.number_input("mini-batch (random seed)", 0, 999, 2)
        a = np.arange(1.0, 9.0)
        out, mask = inverted_dropout(a, rate, seed=int(seed)) if rate < 1 else (a * 0, a * 0)
        fig, ax = plt.subplots(figsize=(8, 3.0))
        ax.bar(np.arange(len(a)) - 0.2, a, width=0.4, color=C["light"],
               edgecolor=C["grey"], label="before")
        ax.bar(np.arange(len(a)) + 0.2, out, width=0.4, color=C["blue"],
               edgecolor=C["grey"], label="after")
        ax.legend(fontsize=9); ax.set_xlabel("unit")
        st.pyplot(fig)
        c1, c2 = st.columns(2)
        c1.metric("mean before", f"{a.mean():.3f}")
        c2.metric("mean after", f"{out.mean():.3f}")
        st.caption("Inverted dropout divides the survivors by (1 − rate), so the "
                   "expected value is preserved and inference needs no rescaling. "
                   "Any single draw still varies — the guarantee is on the mean.")
