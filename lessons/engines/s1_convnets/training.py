"""Lesson 1.5 — Training neural networks.

Figures
    supervised_pipeline  classical feature engineering vs end-to-end deep learning
    datasets_table       the standard datasets, by size and task
    cross_entropy        binary and categorical cross-entropy, and why the log works
    gradient_descent     descent on a 2D surface at four learning rates (interactive)
    backprop_chain       forward values and backward gradients on a tiny network
    batch_size_noise     batch, stochastic and mini-batch trajectories
    momentum_path        momentum carries the optimiser through a flat region
    lr_schedules         step, exponential and cosine schedules

Every optimisation trajectory here is really run in NumPy on the surface shown,
not sketched: the paths are what those update rules actually do.

Source: reference/legacy_md/lesson_1_5_training_nn + lesson_1_5_script.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout)

FIGURES = ["supervised_pipeline", "datasets_table", "cross_entropy",
           "gradient_descent", "backprop_chain", "batch_size_noise",
           "momentum_path", "lr_schedules"]

DATASETS = [
    # (name, images, classes, typical size, task)
    ("MNIST",      "70k",   "10",    "28×28 grey", "digit recognition"),
    ("CIFAR-10",   "60k",   "10",    "32×32 RGB",  "object classification"),
    ("CIFAR-100",  "60k",   "100",   "32×32 RGB",  "object classification"),
    ("ImageNet",   "1.28M", "1000",  "~500×400",   "object classification"),
    ("MS COCO",    "330k",  "80",    "~640×480",   "detection, segmentation"),
    ("Cityscapes", "25k",   "30",    "2048×1024",  "driving-scene segmentation"),
    ("CelebA",     "200k",  "40 attr", "178×218",  "facial attributes"),
]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def bce(y, p):
    """Binary cross-entropy for a single example."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bowl(t, a=1.0, b=4.0):
    """An elongated quadratic loss surface, J(t) = a*t0^2 + b*t1^2."""
    return a * t[..., 0] ** 2 + b * t[..., 1] ** 2


def bowl_grad(t, a=1.0, b=4.0):
    return np.stack([2 * a * t[0], 2 * b * t[1]])


def descend(grad, start, lr, steps=40, gamma=0.0, noise=0.0, seed=0):
    """Gradient descent, optionally with momentum and gradient noise.

    v = gamma*v + lr*grad ;  theta = theta - v      (the lesson's update rule)
    """
    rng = np.random.default_rng(seed)
    t = np.array(start, float)
    v = np.zeros_like(t)
    path = [t.copy()]
    for _ in range(steps):
        g = grad(t)
        if noise:
            g = g + rng.normal(0, noise, size=g.shape)
        v = gamma * v + lr * g
        t = t - v
        if not np.all(np.isfinite(t)) or np.abs(t).max() > 1e4:
            path.append(t.copy())
            break
        path.append(t.copy())
    return np.array(path)


# A surface with a shallow local minimum at x ~= -1.32, a barrier at x ~= -0.32
# and the true minimum at x = 2. At lr = 0.08 plain gradient descent stops in
# the local minimum while momentum clears the barrier and reaches the true one,
# which is the whole point of the figure — verified in fig_momentum_path.
def ravine(t):
    x, y = t[..., 0], t[..., 1]
    return (0.05 * (x - 2.0) ** 2
            - 0.30 * np.exp(-((x + 1.5) ** 2) / 0.4)
            + 0.30 * np.exp(-((x + 0.2) ** 2) / 0.4)
            + 1.6 * y ** 2)


def ravine_grad(t):
    x, y = t[0], t[1]
    dx = (0.10 * (x - 2.0)
          + 0.30 * (2 * (x + 1.5) / 0.4) * np.exp(-((x + 1.5) ** 2) / 0.4)
          - 0.30 * (2 * (x + 0.2) / 0.4) * np.exp(-((x + 0.2) ** 2) / 0.4))
    return np.array([dx, 3.2 * y])


def lr_step(e, lr0=0.1, drop=0.5, every=25):
    return lr0 * drop ** (e // every)


def lr_exp(e, lr0=0.1, k=0.015):
    return lr0 * np.exp(-k * e)


def lr_cosine(e, lr0=0.1, total=100, lr_min=0.0):
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * e / total))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, text, fc, fontsize=8.5):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=C["ink"], lw=1.1, alpha=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return x + w


def fig_supervised_pipeline(ctx: BuildContext) -> Figure:
    W, GAP = 1.9, 0.4
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 5.2))
    rows = [
        # (axis, title, boxes, indices of the boxes the optimiser actually updates)
        (axes[0], "(a) classical computer vision: features are engineered",
         [("labelled\nimages", C["light"]),
          ("hand-designed\nfeature extractor", C["orange"]),
          ("feature\nvectors", C["light"]), ("classifier", C["purple"]),
          ("prediction", C["sky"])], [3], "the classifier only"),
        (axes[1], "(b) deep learning: features are learned end to end",
         [("labelled\nimages", C["light"]),
          ("convolution layers\n(learned features)", C["blue"]),
          ("dense layers\n(classifier)", C["purple"]), ("prediction", C["sky"])],
         [1, 2], "every layer, end to end"),
    ]
    for ax, title, boxes, trained, note in rows:
        x = 0.0
        for i, (txt, col) in enumerate(boxes):
            _box(ax, x, 0.3, W, 0.9, txt, col)
            if i < len(boxes) - 1:
                ax.annotate("", xy=(x + W + 0.34, 0.75), xytext=(x + W + 0.06, 0.75),
                            arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.3))
            x += W + GAP
        total = x - GAP
        # loss and optimiser, fed by the prediction
        pred_mid = (len(boxes) - 1) * (W + GAP) + W / 2
        _box(ax, pred_mid - 1.3, -1.52, 2.6, 0.62, "loss J(θ)  →  optimiser", C["red"])
        ax.annotate("", xy=(pred_mid, -0.90), xytext=(pred_mid, 0.26),
                    arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.3))
        # the update arrow points at what is actually optimised, and only that
        lo = trained[0] * (W + GAP)
        hi = trained[-1] * (W + GAP) + W
        ax.plot([lo, lo, hi, hi], [0.30, -0.16, -0.16, 0.30], color=C["red"],
                lw=1.3, ls="--")
        ax.annotate("", xy=((lo + hi) / 2, 0.26), xytext=((lo + hi) / 2, -0.16),
                    arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.4))
        ax.annotate("", xy=(hi + 0.08, -0.16), xytext=(pred_mid - 1.3, -1.21),
                    arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.2,
                                    connectionstyle="arc3,rad=0.3"))
        ax.text((lo + hi) / 2, -0.50, f"update θ — {note}", ha="center", va="top",
                fontsize=8.5, color=C["red"])
        ax.set_xlim(-0.3, max(total, pred_mid + 1.3) + 0.3)
        ax.set_ylim(-1.95, 1.55); ax.axis("off")
        ax.set_title(title, fontsize=10.5, loc="left")
    png = save_mpl(fig, ctx, "supervised_pipeline")
    return Figure("supervised_pipeline",
                  "Supervised learning is the formalisation of learning from examples, "
                  "and the presence of **labelled training data** is its defining "
                  "feature. (a) In classical computer vision the engineer had to "
                  "design a feature extractor that turned images into a more abstract "
                  "representation the classifier could learn from; only the "
                  "classifier's parameters were optimised. (b) In deep learning the "
                  "feature extraction is learned too, **end to end**: the convolution "
                  "layers and the dense classifier are optimised simultaneously "
                  "against the same loss. This is the key difference from earlier "
                  "machine learning, and the cost of it is data and computation — "
                  "deep models have a great many parameters to fit.", png=png)


def fig_datasets_table(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.axis("off")
    tbl = ax.table(cellText=[list(r) for r in DATASETS],
                   colLabels=["dataset", "images", "classes", "typical size", "task"],
                   loc="center", cellLoc="left",
                   colWidths=[0.16, 0.12, 0.13, 0.19, 0.34])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.45)
    for (r_, c_), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["light"])
        if r_ == 0:
            cell.set_text_props(weight="bold")
    png = save_mpl(fig, ctx, "datasets_table")
    return Figure("datasets_table",
                  "Labelled datasets are an integral part of supervised learning, and "
                  "the computer vision field has many public ones. The small early "
                  "sets — MNIST, CIFAR-10 and CIFAR-100 — are still very useful for "
                  "prototyping. The large sets, ImageNet and MS COCO, are what made "
                  "training state-of-the-art deep networks possible in the first "
                  "place. There are also specialised sets for particular tasks, such "
                  "as Cityscapes for driving-scene segmentation and CelebA for facial "
                  "attributes; later sections of the module return to these.", png=png)


def fig_cross_entropy(ctx: BuildContext) -> Figure:
    p = np.linspace(1e-3, 1 - 1e-3, 500)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))

    ax = axes[0]
    ax.plot(p, bce(1, p), color=C["blue"], lw=2.2, label="ground truth y = 1")
    ax.plot(p, bce(0, p), color=C["orange"], lw=2.2, label="ground truth y = 0")
    ax.set_xlabel("predicted probability ŷ"); ax.set_ylabel("loss")
    ax.set_title("(a) binary cross-entropy", fontsize=10.5)
    ax.set_ylim(0, 5); ax.legend(fontsize=9)
    ax.annotate("confident and wrong:\nloss grows without bound",
                xy=(0.03, 3.5), xytext=(0.22, 4.1), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1))

    ax = axes[1]
    classes = ["cat", "dog", "horse", "bird"]
    probs = np.array([0.10, 0.65, 0.15, 0.10])
    onehot = np.array([0, 1, 0, 0])
    ax.bar(classes, probs, color=[C["orange"] if o else C["light"] for o in onehot],
           edgecolor=C["grey"], label="prediction ŷ")
    for i, (pr, o) in enumerate(zip(probs, onehot)):
        ax.text(i, pr + 0.02, f"{pr:.2f}", ha="center", fontsize=9)
        if o:
            ax.text(i, -0.075, "y = 1", ha="center", fontsize=8.5, color=C["red"])
    loss = -np.sum(onehot * np.log(probs))
    ax.set_title(f"(b) categorical: only the true class contributes\n"
                 f"L = −log({probs[1]:.2f}) = {loss:.3f}", fontsize=10.5)
    ax.set_ylim(-0.12, 0.8); ax.set_ylabel("probability")

    ax = axes[2]
    pt = np.linspace(1e-3, 1, 400)
    ax.plot(pt, -np.log(pt), color=C["green"], lw=2.2)
    for q, lbl in [(0.65, "the prediction above"), (0.95, "confident and right"),
                   (0.10, "confident and wrong")]:
        ax.scatter([q], [-np.log(q)], color=C["red"], zorder=5, s=42)
        ax.annotate(f"{lbl}\n−log({q:.2f}) = {-np.log(q):.2f}",
                    xy=(q, -np.log(q)), xytext=(q - 0.02, -np.log(q) + 0.45),
                    fontsize=8, ha="right" if q > 0.5 else "left")
    ax.set_xlabel("probability given to the true class")
    ax.set_ylabel("−log(p)")
    ax.set_title("(c) why the logarithm", fontsize=10.5)
    ax.set_ylim(0, 4)
    png = save_mpl(fig, ctx, "cross_entropy")
    return Figure("cross_entropy",
                  "The loss function measures the model's current error and is what "
                  "the optimiser minimises; using the wrong one for the task can stop "
                  "a network converging at all. (a) **Binary cross-entropy** has two "
                  "terms, of which only one is active per example. When the truth is "
                  "y = 1 the loss is −log ŷ, which falls towards zero as ŷ approaches "
                  "1; when y = 0 the −log(1 − ŷ) term is active instead. (b) "
                  "**Categorical cross-entropy** extends this to many classes. With a "
                  "one-hot ground truth every term but one is multiplied by zero, so "
                  "the loss depends only on the probability given to the correct "
                  "class. (c) The logarithm is what makes it work as a training "
                  "signal: it is near zero for a confident correct answer and grows "
                  "without bound as the probability given to the true class "
                  "approaches zero, so a confidently wrong prediction produces a very "
                  "large gradient.", png=png)


def fig_gradient_descent(ctx: BuildContext) -> Figure:
    lrs = [(0.01, "too low: converges, but slowly"),
           (0.10, "good: reaches the minimum quickly"),
           (0.24, "high: overshoots and oscillates"),
           (0.30, "too high: diverges")]
    gx, gy = np.meshgrid(np.linspace(-2.6, 2.6, 260), np.linspace(-1.3, 1.3, 130))
    Z = bowl(np.stack([gx, gy], -1))
    start = (-2.2, 1.0)

    fig, axes = plt.subplots(2, 4, figsize=(14, 5.6),
                             gridspec_kw=dict(height_ratios=[1.25, 1]))
    for col, (lr, label) in enumerate(lrs):
        path = descend(bowl_grad, start, lr, steps=40)
        ax = axes[0, col]
        ax.contour(gx, gy, Z, levels=14, colors=C["light"], linewidths=0.8)
        ax.contourf(gx, gy, Z, levels=14, cmap="Blues", alpha=0.35)
        vis = path[np.all(np.abs(path) < 3.0, axis=1)]
        ax.plot(vis[:, 0], vis[:, 1], "-o", color=C["red"], ms=3, lw=1.3)
        ax.scatter([0], [0], marker="*", s=160, color=C["green"], zorder=6)
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(-1.3, 1.3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"η = {lr}\n{label}", fontsize=9.5)

        ax = axes[1, col]
        losses = bowl(path)
        ax.plot(losses, color=C["blue"], lw=2)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1e3)
        ax.set_xlabel("iteration", fontsize=9)
        if col == 0:
            ax.set_ylabel("loss J(θ)", fontsize=9)
    fig.suptitle(r"$\theta \leftarrow \theta - \eta\, \partial J/\partial \theta$   "
                 "on  J(θ) = θ₀² + 4θ₁²,  all four runs from the same start",
                 fontsize=10.5)
    png = save_mpl(fig, ctx, "gradient_descent")

    fig2 = go.Figure()
    frames = []
    sweep = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.24, 0.27, 0.30]
    for lr in sweep:
        path = descend(bowl_grad, start, lr, steps=40)
        vis = path[np.all(np.abs(path) < 3.0, axis=1)]
        final = bowl(path[-1])
        frames.append(go.Frame(name=str(lr), data=[
            go.Contour(x=gx[0], y=gy[:, 0], z=Z, showscale=False,
                       colorscale="Blues", opacity=0.55,
                       contours=dict(showlines=False), hoverinfo="skip"),
            go.Scatter(x=vis[:, 0], y=vis[:, 1], mode="lines+markers",
                       line=dict(color=C["red"], width=2),
                       marker=dict(size=5), hoverinfo="skip"),
        ], traces=[0, 1], layout=go.Layout(
            title=f"η = {lr}:  loss after 40 steps = "
                  + (f"{final:.3g}" if np.isfinite(final) else "diverged"))))
    for tr in frames[0].data:
        fig2.add_trace(tr)
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(sweep, "learning rate η = "),
                       title=frames[0].layout.title.text, showlegend=False)
    fig2.update_xaxes(range=[-2.6, 2.6], title_text="θ₀")
    fig2.update_yaxes(range=[-1.3, 1.3], title_text="θ₁")
    return Figure("gradient_descent",
                  "Gradient descent minimises the loss J(θ) by repeatedly stepping "
                  "in the direction of steepest descent, "
                  r"$\theta \leftarrow \theta - \eta\,\partial J/\partial\theta$. The "
                  "**learning rate** η sets the step size, and it matters a great "
                  "deal. Too low and the algorithm needs a great many updates to "
                  "reach the minimum. Too high and it overshoots, oscillating across "
                  "the valley or diverging outright — note the log scale on the loss "
                  "curves underneath. The aim is the largest rate that still "
                  "converges. All four runs here start from the same point on the "
                  "same surface and differ only in η; drag the slider in the "
                  "interactive version to find where the behaviour changes.",
                  png=png, html=plotly_html(fig2, height=470), kind="plotly")


def fig_backprop_chain(ctx: BuildContext) -> Figure:
    # a tiny network: x -> (w1,b1) -> ReLU -> (w2,b2) -> yhat, squared-error loss
    x, w1, b1, w2, b2, y = 1.5, 0.8, -0.2, 1.4, 0.3, 2.0
    z1 = w1 * x + b1
    a1 = max(0.0, z1)
    z2 = w2 * a1 + b2
    L = 0.5 * (z2 - y) ** 2
    dz2 = z2 - y
    dw2, db2, da1 = dz2 * a1, dz2, dz2 * w2
    dz1 = da1 * (1.0 if z1 > 0 else 0.0)
    dw1, db1 = dz1 * x, dz1

    nodes = [("x", 0.0, x, None, None), ("z₁ = w₁x + b₁", 1.0, z1, dz1, "z₁"),
             ("a₁ = ReLU(z₁)", 2.0, a1, da1, "a₁"), ("z₂ = w₂a₁ + b₂", 3.0, z2, dz2, "z₂"),
             ("L = ½(z₂ − y)²", 4.0, L, 1.0, "L")]
    fig, ax = plt.subplots(figsize=(12, 3.9))
    for i, (name, xpos, val, grad, gname) in enumerate(nodes):
        ax.add_patch(plt.Rectangle((xpos * 2.5, 0.55), 2.0, 0.75, fc=C["light"],
                                   ec=C["ink"], lw=1.2, alpha=0.75))
        ax.text(xpos * 2.5 + 1.0, 0.925, name, ha="center", va="center", fontsize=9)
        ax.text(xpos * 2.5 + 1.0, 1.48, f"{val:.4g}", ha="center", fontsize=10,
                color=C["blue"])
        if grad is not None:
            ax.text(xpos * 2.5 + 1.0, 0.24, f"∂L/∂{gname} = {grad:.4g}", ha="center",
                    fontsize=9, color=C["red"])
        if i < len(nodes) - 1:
            ax.annotate("", xy=(xpos * 2.5 + 2.45, 1.08), xytext=(xpos * 2.5 + 2.05, 1.08),
                        arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1.6))
            ax.annotate("", xy=(xpos * 2.5 + 2.05, 0.78), xytext=(xpos * 2.5 + 2.45, 0.78),
                        arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.6))
    ax.text(5.0, 1.78, "forward pass: values", ha="center", fontsize=10,
            color=C["blue"])
    ax.text(5.0, -0.18, "backward pass: gradients, by the chain rule",
            ha="center", fontsize=10, color=C["red"])
    ax.text(5.0, -0.62,
            f"∂L/∂w₂ = ∂L/∂z₂ · a₁ = {dz2:.4g} × {a1:.4g} = {dw2:.4g}          "
            f"∂L/∂w₁ = ∂L/∂z₁ · x = {dz1:.4g} × {x:.4g} = {dw1:.4g}\n"
            f"∂L/∂b₂ = {db2:.4g}                                        "
            f"∂L/∂b₁ = {db1:.4g}",
            ha="center", va="top", fontsize=9.5, family="monospace")
    ax.set_xlim(-0.3, 12.3); ax.set_ylim(-1.45, 2.05); ax.axis("off")
    png = save_mpl(fig, ctx, "backprop_chain")
    return Figure("backprop_chain",
                  "Gradient descent needs the partial derivative of the loss with "
                  "respect to every parameter. **Backpropagation**, published in 1989, "
                  "computes them efficiently using the chain rule. Training data is "
                  "passed forward through the network (blue) to give a prediction, "
                  "which is compared with the true label by the loss function. The "
                  "backward pass (red) then works from the loss back towards the "
                  "input, and at each step the gradient arriving from the right is "
                  "multiplied by the local derivative. The point is that every "
                  "parameter's gradient is obtained in a single backward sweep rather "
                  "than by perturbing parameters one at a time — which is what makes "
                  "training a network with millions of parameters feasible at all. "
                  "Every number here is computed for this tiny network, so the "
                  "arithmetic can be checked by hand.", png=png)


def fig_batch_size_noise(ctx: BuildContext) -> Figure:
    gx, gy = np.meshgrid(np.linspace(-2.6, 2.6, 260), np.linspace(-1.4, 1.4, 140))
    Z = bowl(np.stack([gx, gy], -1))
    cases = [("batch: all examples", 0.0, C["blue"]),
             ("mini-batch: a subset", 0.45, C["orange"]),
             ("stochastic: one example", 1.5, C["red"])]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))
    for ax, (name, noise, col) in zip(axes, cases):
        path = descend(bowl_grad, (-2.2, 1.1), 0.08, steps=60, noise=noise, seed=5)
        ax.contour(gx, gy, Z, levels=14, colors=C["light"], linewidths=0.8)
        ax.contourf(gx, gy, Z, levels=14, cmap="Blues", alpha=0.3)
        ax.plot(path[:, 0], path[:, 1], "-o", color=col, ms=2.6, lw=1.2)
        ax.scatter([0], [0], marker="*", s=150, color=C["green"], zorder=6)
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(-1.4, 1.4)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name, fontsize=10)
    ax = axes[3]
    for (name, noise, col) in cases:
        path = descend(bowl_grad, (-2.2, 1.1), 0.08, steps=60, noise=noise, seed=5)
        ax.plot(bowl(path), color=col, lw=1.8, label=name.split(":")[0])
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("loss J(θ)")
    ax.set_title("convergence", fontsize=10); ax.legend(fontsize=8.5)
    png = save_mpl(fig, ctx, "batch_size_noise")
    return Figure("batch_size_noise",
                  "How much training data to use for each update. **Batch** gradient "
                  "descent uses the entire training set, giving a smooth trajectory, "
                  "but for a large dataset the memory required exceeds even the best "
                  "GPUs. **Stochastic** gradient descent uses a single example, which "
                  "converges noisily and updates the network far too often to be "
                  "computationally efficient — though the noise has a genuine "
                  "benefit, since it can knock the optimiser out of a local minimum. "
                  "**Mini-batch** gradient descent is the compromise used in practice: "
                  "16, 32 or 64 examples at a time keeps some of the helpful noise "
                  "while suiting the GPU's parallelism. The three trajectories here "
                  "are the same update rule on the same surface, differing only in "
                  "how much noise the gradient estimate carries.", png=png)


def fig_momentum_path(ctx: BuildContext) -> Figure:
    LR, START, STEPS = 0.08, (-3.5, 0.0), 220
    xs = np.linspace(-4.0, 4.0, 600)
    prof = ravine(np.stack([xs, np.zeros_like(xs)], -1))
    runs = {g: descend(ravine_grad, START, LR, steps=STEPS, gamma=g)
            for g in (0.0, 0.5, 0.9)}
    # The figure only says what the runs actually do.
    stuck, escaped = runs[0.0][-1, 0], runs[0.9][-1, 0]
    assert stuck < -1.0 and escaped > 1.5, (stuck, escaped)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    ax.plot(xs, prof, color=C["grey"], lw=2)
    ax.axvspan(-1.9, -0.8, color=C["red"], alpha=0.06)
    ax.text(-1.35, prof.max() * 0.80, "local\nminimum", fontsize=8.5,
            color=C["grey"], ha="center")
    ax.text(2.0, prof.max() * 0.80, "true\nminimum", fontsize=8.5,
            color=C["grey"], ha="center")
    for gamma, col, name in [(0.0, C["red"], "plain SGD (γ = 0)"),
                             (0.9, C["green"], "with momentum (γ = 0.9)")]:
        path = runs[gamma]
        ax.plot(path[:, 0], ravine(path) + (0.03 if gamma else -0.03), "o",
                color=col, ms=3.2, alpha=0.8, label=name)
        ax.scatter([path[-1, 0]], [ravine(path[-1:])[0]], color=col, s=110,
                   marker="X", zorder=6, edgecolor="white", linewidth=1.3)
    ax.set_xlabel("θ"); ax.set_ylabel("J(θ)")
    ax.set_title("(a) plain descent stops at the first minimum it meets;\n"
                 "momentum carries through it to the true one", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")

    ax = axes[1]
    for gamma, col in [(0.0, C["red"]), (0.5, C["orange"]), (0.9, C["green"])]:
        ax.plot(ravine(runs[gamma]), color=col, lw=2,
                label=f"γ = {gamma}  →  J = {ravine(runs[gamma][-1:])[0]:.3f}")
    ax.set_xlabel("iteration"); ax.set_ylabel("J(θ)")
    ax.set_title(f"(b) loss against iteration (η = {LR} throughout)", fontsize=10)
    ax.legend(fontsize=9)
    png = save_mpl(fig, ctx, "momentum_path")
    return Figure("momentum_path",
                  "Plain stochastic gradient descent struggles with flat regions and "
                  "local minima. The simplest improvement is **momentum**, which "
                  "reinterprets the update as a velocity: "
                  r"$v \leftarrow \gamma v + \eta\,\partial J/\partial\theta$, then "
                  r"$\theta \leftarrow \theta - v$. Including previous gradients "
                  "damps the effect of noise and preserves the general trend, so the "
                  "optimiser builds speed through regions of large gradient and "
                  "carries that speed across flat regions and through shallow local "
                  "minima. Both runs here use the same learning rate and the same "
                  "starting point and differ only in γ, and the difference is "
                  "decisive: plain descent halts at the first minimum it meets and "
                  "stays there, while momentum carries straight through it to the "
                  "true minimum, at a loss of 0 rather than 0.287. Note the "
                  "trade-off, visible in (b): γ = 0.9 overshoots and oscillates before "
                  "settling. A γ close to 1 gives larger updates but can cause "
                  "convergence problems, which is why it is sometimes increased over "
                  "the course of training. A great many deep networks are trained "
                  "with SGD plus momentum.", png=png)


def fig_lr_schedules(ctx: BuildContext) -> Figure:
    e = np.arange(0, 101)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    ax.step(e, [lr_step(i) for i in e], where="post", color=C["blue"], lw=2,
            label="step decay: ×0.5 every 25 epochs")
    ax.plot(e, [lr_exp(i) for i in e], color=C["orange"], lw=2,
            label="exponential: η₀e^(−0.015t)")
    ax.plot(e, [lr_cosine(i, total=100) for i in e], color=C["green"], lw=2,
            label="cosine annealing")
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate η")
    ax.set_title("(a) three common schedules", fontsize=10.5)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    for fn, col, name in [(lambda i: 0.10, C["red"], "constant η = 0.10"),
                          (lambda i: lr_step(i), C["blue"], "step decay"),
                          (lambda i: lr_cosine(i, total=100), C["green"], "cosine")]:
        t = np.array([-2.2, 1.0])
        losses = []
        for i in range(101):
            t = t - fn(i) * bowl_grad(t)
            losses.append(bowl(t))
        ax.plot(losses, color=col, lw=2, label=name)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss J(θ)")
    ax.set_title("(b) the same surface, the same start", fontsize=10.5)
    ax.legend(fontsize=8.5)
    png = save_mpl(fig, ctx, "lr_schedules")
    return Figure("lr_schedules",
                  "A fixed learning rate forces a compromise: large enough to make "
                  "progress early, small enough to settle at the end. A **schedule** "
                  "avoids the compromise by reducing η as training proceeds. (a) Step "
                  "decay multiplies the rate by a constant factor every so many "
                  "epochs; exponential decay does the same thing smoothly; cosine "
                  "annealing follows a half-cosine from η₀ down to zero. (b) The "
                  "effect on the same optimisation problem: a constant rate plateaus "
                  "once the steps become too large to make further progress, while "
                  "the decaying schedules keep reducing the loss. Both scheduled runs "
                  "start at the same rate as the constant one.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_supervised_pipeline(ctx), fig_datasets_table(ctx),
            fig_cross_entropy(ctx), fig_gradient_descent(ctx),
            fig_backprop_chain(ctx), fig_batch_size_noise(ctx),
            fig_momentum_path(ctx), fig_lr_schedules(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("1.5 Training neural networks")
    tab1, tab2, tab3 = st.tabs(["Gradient descent", "Cross-entropy", "Schedules"])

    with tab1:
        c = st.columns(3)
        lr = c[0].slider("learning rate η", 0.01, 0.35, 0.10, 0.01)
        gamma = c[1].slider("momentum γ", 0.0, 0.95, 0.0, 0.05)
        noise = c[2].slider("gradient noise (smaller batch)", 0.0, 2.0, 0.0, 0.1)
        path = descend(bowl_grad, (-2.2, 1.0), lr, steps=60, gamma=gamma, noise=noise)
        gx, gy = np.meshgrid(np.linspace(-2.8, 2.8, 200), np.linspace(-1.5, 1.5, 110))
        Z = bowl(np.stack([gx, gy], -1))
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        axes[0].contourf(gx, gy, Z, levels=16, cmap="Blues", alpha=0.45)
        vis = path[np.all(np.abs(path) < 3.0, axis=1)]
        axes[0].plot(vis[:, 0], vis[:, 1], "-o", color=C["red"], ms=3)
        axes[0].scatter([0], [0], marker="*", s=180, color=C["green"])
        axes[0].set_xlim(-2.8, 2.8); axes[0].set_ylim(-1.5, 1.5)
        axes[1].plot(bowl(path), color=C["blue"], lw=2)
        axes[1].set_yscale("log"); axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("loss")
        st.pyplot(fig)
        final = bowl(path[-1])
        if not np.isfinite(final) or final > 1e3:
            st.error("Diverged — the learning rate is too high for this surface.")
        else:
            st.success(f"Loss after {len(path) - 1} steps: {final:.3g}")

    with tab2:
        mode = st.radio("loss", ["binary", "categorical"], horizontal=True)
        if mode == "binary":
            y = st.radio("ground truth y", [0, 1], horizontal=True, index=1)
            ph = st.slider("predicted probability ŷ", 0.01, 0.99, 0.7, 0.01)
            st.latex(rf"L = -[y\log\hat y + (1-y)\log(1-\hat y)] = {bce(y, ph):.4f}")
            pp = np.linspace(0.01, 0.99, 200)
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.plot(pp, bce(y, pp), color=C["blue"], lw=2)
            ax.scatter([ph], [bce(y, ph)], color=C["red"], s=60, zorder=5)
            ax.set_xlabel("ŷ"); ax.set_ylabel("loss"); ax.set_ylim(0, 5)
            st.pyplot(fig)
        else:
            txt = st.text_input("predicted probabilities (comma separated)",
                                "0.10, 0.65, 0.15, 0.10")
            try:
                pr = np.array([float(v) for v in txt.split(",")])
            except ValueError:
                st.error("Use comma-separated numbers."); return
            pr = pr / pr.sum()
            true = st.number_input("index of the true class", 0, len(pr) - 1, 1)
            st.latex(rf"L = -\log(p_{{{int(true)}}}) = -\log({pr[int(true)]:.3f})"
                     rf" = {-np.log(pr[int(true)]):.4f}")
            st.caption("With a one-hot ground truth every other term is multiplied "
                       "by zero, so only the true class's probability matters.")

    with tab3:
        c = st.columns(3)
        lr0 = c[0].number_input("initial η", 0.001, 1.0, 0.1)
        total = c[1].number_input("total epochs", 10, 500, 100)
        every = c[2].number_input("step-decay period", 1, 200, 25)
        e = np.arange(0, int(total) + 1)
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.step(e, [lr_step(i, lr0, 0.5, int(every)) for i in e], where="post",
                color=C["blue"], lw=2, label="step")
        ax.plot(e, [lr_exp(i, lr0) for i in e], color=C["orange"], lw=2,
                label="exponential")
        ax.plot(e, [lr_cosine(i, lr0, int(total)) for i in e], color=C["green"],
                lw=2, label="cosine")
        ax.set_xlabel("epoch"); ax.set_ylabel("η"); ax.legend(fontsize=9)
        st.pyplot(fig)
