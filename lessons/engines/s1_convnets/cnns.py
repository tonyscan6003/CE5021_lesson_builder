"""Lesson 1.4 — Convolutional neural networks.

Figures
    fc_on_images          why a dense layer on raw pixels is impractical
    conv_as_nn            a convolution redrawn as a sparse, weight-shared layer
    cnn_structure         conv/act/pool stack to flatten to dense, with tensor shapes
    learned_features      the edge -> texture -> part -> object hierarchy
    receptive_field       the slide's 1D worked example, layer by layer (interactive)
    translation_invariance  a shifted signal survives max pooling
    lenet5                LeNet-5 with every shape and parameter count computed

Source: reference/legacy_md/lesson_1_4_cnns + lesson_1_4_script.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout, sync_matrix)
from .._volumes import network_diagram

FIGURES = ["fc_on_images", "conv_as_nn", "cnn_structure", "learned_features",
           "receptive_field", "translation_invariance", "lenet5"]

# Slide 8's architecture: two 3-kernel stride-1 convolutions with a 2/2 pooling
# layer between them. Listed input-most first; r is found by walking backwards.
#
# Taken here with VALID padding, so the traced span never runs off the edge of
# the input and the picture and the recurrence agree exactly (r = 8). The slide
# works the same architecture with one ring of padding and quotes 7 for the
# deepest neuron, which is what you count if the drawing clips the padded
# position at the boundary; the recurrence itself gives 8 either way. See
# COURSE_PLAN.md section 8.
RF_LAYERS = [("conv k=3, s=1", 3, 1),
             ("pool k=2, s=2", 2, 2),
             ("conv k=3, s=1", 3, 1)]

# Slide 9: LeNet-5, as the script describes it.
LENET = [
    # (name, kind, out_h, out_w, out_c, k, in_c)
    ("input 32×32×1",        "input", 32, 32, 1,  None, None),
    ("conv 5×5, 6 kernels",  "conv",  28, 28, 6,  5,    1),
    ("avg pool 2×2, s=2",    "pool",  14, 14, 6,  2,    6),
    ("conv 5×5, 16 kernels", "conv",  10, 10, 16, 5,    6),
    ("avg pool 2×2, s=2",    "pool",  5,  5,  16, 2,    16),
    ("flatten → 400",        "flat",  1,  1,  400, None, None),
    ("dense 120",            "dense", 1,  1,  120, None, 400),
    ("dense 84",             "dense", 1,  1,  84,  None, 120),
    ("dense 10 + softmax",   "dense", 1,  1,  10,  None, 84),
]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def receptive_field(layers):
    """Receptive field at the input, walking back through the layers.

    r_{l-1} = s_l * r_l + (k_l - s_l), starting from r = 1 at the deepest layer.
    Returns the running value after each backward step, deepest first.
    """
    r, steps = 1, []
    for name, k, s in reversed(layers):
        r = s * r + (k - s)
        steps.append((name, k, s, r))
    return r, steps


def lenet_table():
    """Per-layer shapes and parameter counts for LeNet-5."""
    rows, total = [], 0
    for name, kind, h, w, c, k, cin in LENET:
        if kind == "conv":
            p = k * k * cin * c + c
        elif kind == "dense":
            p = cin * c + c
        else:                       # input, pooling and flatten have none
            p = 0
        total += p
        shape = f"{h}×{w}×{c}" if kind not in ("flat", "dense") else f"{c}"
        rows.append(dict(name=name, kind=kind, shape=shape, params=p,
                         h=h, w=w, c=c, k=k))
    return rows, total


def maxpool1d(x, k, s):
    return np.array([x[i:i + k].max() for i in range(0, len(x) - k + 1, s)])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_fc_on_images(ctx: BuildContext) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2),
                             gridspec_kw=dict(width_ratios=[1, 1.1]))

    ax = axes[0]
    img = np.arange(16).reshape(4, 4) % 5 + 1
    sync_matrix(ax, img, title="a 4 × 4 image …", fontsize=9)
    ax.set_position(ax.get_position())
    ax = axes[0]
    ax.annotate("", xy=(4.3, 1.5), xytext=(3.7, 1.5),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.6))
    ax.text(4.0, 1.15, "flatten", ha="center", fontsize=9)
    for i in range(16):
        ax.add_patch(plt.Rectangle((4.6, i * 0.22 - 0.9), 0.5, 0.2,
                                   fc=C["light"], ec=C["grey"], lw=0.7))
    ax.text(4.85, 2.85, "16 × 1\nvector", ha="center", fontsize=9)
    for j, y in enumerate((0.6, 1.8)):
        ax.add_patch(plt.Circle((6.4, y), 0.22, fc="white", ec=C["blue"], lw=2))
        for i in range(16):
            ax.plot([5.1, 6.2], [i * 0.22 - 0.8, y], color=C["light"], lw=0.5,
                    zorder=0)
    ax.text(6.4, 2.85, "2 neurons", ha="center", fontsize=9)
    ax.text(5.6, -1.5, "16 × 2 weights + 2 biases = 34 parameters",
            ha="center", fontsize=9.5)
    ax.set_xlim(-0.6, 7.4); ax.set_ylim(-1.9, 3.4)
    ax.set_title("(a) a dense layer must see every pixel", fontsize=10.5)
    ax.axis("off")

    ax = axes[1]
    sizes = np.array([4, 8, 16, 32, 64, 128, 256, 512])
    for neurons, col in zip((10, 100, 1000), SERIES):
        ax.loglog(sizes, sizes ** 2 * neurons + neurons, color=col, lw=2,
                  marker="o", ms=4, label=f"{neurons} neurons")
    n256 = 256 ** 2 * 100 + 100
    ax.scatter([256], [n256], color=C["red"], zorder=5, s=55)
    ax.annotate(f"256 × 256 greyscale into 100 neurons\n= {n256:,} parameters",
                xy=(256, n256), xytext=(9, 3e6), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.2))
    ax.set_xlabel("image side (pixels)")
    ax.set_ylabel("parameters in the first layer")
    ax.set_title("(b) the parameter count is quadratic in the image side",
                 fontsize=10.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")
    png = save_mpl(fig, ctx, "fc_on_images")
    return Figure("fc_on_images",
                  "In the naive approach, an image is flattened into a vector and applied to "
                  "a fully connected layer. (a) For a 4 × 4 image and two neurons, 34 "
                  "parameters are required. (b) The number of parameters, however, increases "
                  "with the *square* of the image side. A 256 × 256 greyscale image becomes a "
                  "vector of 65,536 values, and 6,553,700 parameters are required to connect "
                  "it to a layer of 100 neurons, in one layer alone. In addition, the spatial "
                  "arrangement of the pixels is lost when the image is flattened, and pixels "
                  "that are neighbours in the image are not treated as related by a dense "
                  "layer. Both problems are addressed by convolution.",
                  png=png)


def fig_conv_as_nn(ctx: BuildContext) -> Figure:
    img = np.array([[1, 2, 0], [0, 1, 3], [2, 1, 0]])
    # the four valid positions of a 2x2 kernel on a 3x3 input
    pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    idx = [[r * 3 + c, r * 3 + c + 1, (r + 1) * 3 + c, (r + 1) * 3 + c + 1]
           for r, c in pos]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                             gridspec_kw=dict(width_ratios=[0.8, 1.35]))
    ax = axes[0]
    sync_matrix(ax, img, title="3 × 3 input, 2 × 2 kernel, valid padding\n"
                               "→ 4 kernel positions → 2 × 2 output", fontsize=10)
    for (r, c), col in zip(pos, SERIES):
        ax.add_patch(plt.Rectangle((c - 0.45, r - 0.45), 1.9, 1.9, fill=False,
                                   ec=col, lw=2.2, alpha=0.9))

    ax = axes[1]
    # scatter, not plt.Circle: patch radii are in data units, so circles come out
    # as ellipses whenever the axes aspect is not equal
    ys = np.linspace(4.0, 0.0, 9)
    ax.scatter([0.0] * 9, ys, s=190, facecolors=C["light"], edgecolors=C["grey"],
               linewidths=1.2, zorder=3)
    for i, y in enumerate(ys):
        ax.text(-0.42, y, f"x{i}", ha="center", va="center", fontsize=8.5,
                color=C["grey"])
    ny = np.linspace(3.1, 0.9, 4)
    ax.scatter([2.5] * 4, ny, s=420, facecolors="white",
               edgecolors=[SERIES[i] for i in range(4)], linewidths=2.2, zorder=3)
    for n, (y, col) in enumerate(zip(ny, SERIES)):
        ax.text(2.5, y, "σ", ha="center", va="center", fontsize=9, zorder=4)
        for j, w in zip(idx[n], ["w1", "w2", "w3", "w4"]):
            ax.plot([0.13, 2.3], [ys[j], y], color=col, lw=1.2, alpha=0.75)
        ax.text(3.0, y, f"→ out{n}", ha="left", va="center", fontsize=9)
    ax.text(0.0, 4.55, "input pixels\n(flattened)", ha="center", fontsize=9)
    ax.text(2.5, 4.55, "one neuron per\nkernel position", ha="center", fontsize=9)
    ax.text(1.5, -0.85,
            "Not fully connected: each neuron sees only 4 of the 9 pixels.\n"
            "All four neurons share the same w1…w4 and the same bias — "
            "4 weights + 1 bias, not 9 × 4 + 4.",
            ha="center", va="top", fontsize=9.5)
    ax.set_xlim(-1.0, 4.4); ax.set_ylim(-1.9, 5.1); ax.axis("off")
    png = save_mpl(fig, ctx, "conv_as_nn")
    return Figure("conv_as_nn",
                  "The same convolution drawn as a neural network. Each position of the "
                  "kernel corresponds to one neuron: the multiply-and-sum performed by the "
                  "kernel at that position is the same as the weighted sum in a neuron, and "
                  "when a bias and an activation function are added, the operation is "
                  "mathematically the same as a neuron. The result is similar to the dense "
                  "networks of Lesson 1.2, with two important differences. The network is "
                  "**not fully connected**, as each neuron is connected only to the 4 pixels "
                  "under the kernel, which saves computation. In addition, the **same** "
                  "weights and bias are used by every neuron, which saves memory. These two "
                  "properties, local connectivity and weight sharing, are the advantages of a "
                  "convolution layer over a dense layer.",
                  png=png)


def fig_cnn_structure(ctx: BuildContext) -> Figure:
    stages = [("input\n32×32×3", C["light"], "32×32×3"),
              ("conv 3×3, 32\n'same'", C["blue"], "32×32×32"),
              ("ReLU", C["sky"], "32×32×32"),
              ("conv 3×3, 32\n'same'", C["blue"], "32×32×32"),
              ("ReLU", C["sky"], "32×32×32"),
              ("max pool\n2×2, s=2", C["orange"], "16×16×32"),
              ("conv 3×3, 64\n'same'", C["blue"], "16×16×64"),
              ("ReLU", C["sky"], "16×16×64"),
              ("max pool\n2×2, s=2", C["orange"], "8×8×64"),
              ("flatten", C["grey"], "4096"),
              ("dense 10\n+ softmax", C["purple"], "10")]
    fig, ax = plt.subplots(figsize=(13.5, 3.8))
    x = 0.0
    for i, (label, col, shape) in enumerate(stages):
        w = 1.05
        ax.add_patch(plt.Rectangle((x, 0), w, 1.0, fc=col, ec=C["ink"], lw=1.2,
                                   alpha=0.55 if col != C["light"] else 0.8))
        ax.text(x + w / 2, 0.5, label, ha="center", va="center", fontsize=8.2)
        ax.text(x + w / 2, -0.20, shape, ha="center", va="top", fontsize=8.2,
                color=C["red"], family="monospace")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + 0.22, 0.5), xytext=(x + w + 0.02, 0.5),
                        arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.2))
        x += w + 0.24
    ax.text(x / 2, 1.32, "tensor shape after each stage is printed in red; "
                         "pooling halves the spatial size while the number of kernels "
                         "is increased to compensate",
            ha="center", va="bottom", fontsize=9.5)
    ax.set_xlim(-0.2, x); ax.set_ylim(-0.55, 1.62); ax.axis("off")
    png = save_mpl(fig, ctx, "cnn_structure")
    return Figure("cnn_structure",
                  "The basic structure of a CNN, drawn as a sequence of operations rather "
                  "than feature maps. Convolution layers are used first, and are stacked to "
                  "produce a deeper network, which helps the network to learn complex "
                  "features. The activation is shown as a separate layer; it is good practice "
                  "to show the activation explicitly, as in modern CNNs a convolution layer "
                  "may appear without one. Features are summarised by the pooling layers and "
                  "the spatial size is halved, and the number of kernels is typically "
                  "increased as the spatial size is reduced. Finally, the feature map is "
                  "flattened to a vector and applied to a dense layer with one neuron per "
                  "class, and the class probabilities are given by the softmax activation.",
                  png=png)


def fig_learned_features(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(3)
    n = 28
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1) * 2 - 1

    def gabor(theta, f=3.2):
        xr = xx * np.cos(theta) + yy * np.sin(theta)
        return np.exp(-(xx ** 2 + yy ** 2) / 0.55) * np.cos(2 * np.pi * f * xr)

    def blob(cx, cy, r):
        return np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2)))

    layer1 = [gabor(t) for t in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)]
    layer2 = [gabor(0) * gabor(np.pi / 2),
              blob(0, 0, 0.45) - blob(0, 0, 0.22),
              np.maximum(gabor(np.pi / 6), gabor(-np.pi / 3)),
              blob(-0.35, -0.35, 0.3) + blob(0.35, 0.35, 0.3)]
    layer3 = [blob(-0.35, 0.3, 0.16) + blob(0.35, 0.3, 0.16) + blob(0, -0.3, 0.3),
              blob(0, 0.2, 0.42) - blob(0, 0.2, 0.2) + blob(0, -0.45, 0.2),
              np.maximum(blob(-0.3, 0, 0.25), gabor(np.pi / 2) * 0.8),
              blob(0, 0, 0.5) * (1 + 0.6 * np.cos(6 * np.arctan2(yy, xx)))]
    layer4 = [blob(-0.3, 0.25, 0.2) + blob(0.3, 0.25, 0.2)
              + blob(0, -0.15, 0.38) + blob(0, -0.55, 0.18),
              blob(0, 0.35, 0.28) + blob(0, -0.25, 0.45),
              blob(-0.45, 0, 0.22) + blob(0.1, 0, 0.42) + blob(0.55, 0.1, 0.2),
              blob(0, 0, 0.55) * (1 + 0.4 * np.cos(4 * np.arctan2(yy, xx)))
              + blob(0, 0.5, 0.15)]

    groups = [("layer 1\nedges and colours", layer1),
              ("layer 2\ntextures and corners", layer2),
              ("layer 3\nobject parts", layer3),
              ("layer 4\nwhole objects", layer4)]
    fig, axes = plt.subplots(4, 4, figsize=(9.5, 9.8))
    for col, (title, maps) in enumerate(groups):
        for row, m in enumerate(maps):
            ax = axes[row, col]
            ax.imshow(m, cmap="RdBu_r", vmin=-np.abs(m).max(), vmax=np.abs(m).max())
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(C["grey"]); s.set_linewidth(0.8)
        axes[0, col].set_title(title, fontsize=10)
    fig.suptitle("Schematic — synthetic patterns standing in for learned kernels, "
                 "not the output of a trained network", fontsize=9.5, y=0.055)
    png = save_mpl(fig, ctx, "learned_features")
    return Figure("learned_features",
                  "The kernel parameters are learned during training, so after training the "
                  "convolution layers respond to the features that are useful for the task. "
                  "A convolution layer can be considered to perform cross-correlation between "
                  "the image and the learned features, so that the response is large where "
                  "the image resembles the kernel. Simple features such as oriented edges and "
                  "colour boundaries are learned in the early layers. Deeper layers respond "
                  "to textures, then to parts of objects, and then to whole objects. The "
                  "pattern of responses in the final feature map is interpreted by the "
                  "classifier stage and a class probability is output. **These panels are "
                  "synthetic patterns that illustrate the progression, not kernels from a "
                  "trained network**; the visualisation of real learned features is "
                  "described in Lesson 2.5.",
                  png=png)


def rf_layer_stack(layers, n_in=12, unit=None):
    """Lay out every layer of a 1D network for drawing the receptive field.

    Returns one row per layer, input first and deepest last — the same order as
    the worked example's table, where layer 1 acts on the input and the last
    layer is the deepest. Each row gives its cells' centres in input coordinates
    (so the cone lines up), the cell pitch, and the span of cells that feed the
    chosen unit of the deepest layer.
    """
    centres = [np.arange(n_in, dtype=float)]
    pitch = [1.0]
    for _, k, s in layers:
        prev = centres[-1]
        n = (len(prev) - k) // s + 1                  # valid padding
        centres.append(np.array([prev[i * s:i * s + k].mean() for i in range(n)]))
        pitch.append(pitch[-1] * s)
    unit = len(centres[-1]) // 2 if unit is None else unit
    spans = [(unit, unit)]
    for _, k, s in reversed(layers):                  # trace back towards the input
        lo, hi = spans[0]
        spans.insert(0, (lo * s, hi * s + k - 1))
    return centres, pitch, spans


def fig_receptive_field(ctx: BuildContext) -> Figure:
    r_total, steps = receptive_field(RF_LAYERS)
    L = len(RF_LAYERS)
    centres, pitch, spans = rf_layer_stack(RF_LAYERS)
    # r_l for each row, input (l = 0) first, exactly as the worked example labels them
    r_vals = [hi - lo + 1 for lo, hi in spans]
    assert r_vals[0] == r_total and r_vals[-1] == 1
    names = ["input"] + [f"layer {i + 1}: {nm}" for i, (nm, _, _) in enumerate(RF_LAYERS)]
    names[-1] += " (deepest)"
    ys = [L - i for i in range(L + 1)]                 # input at the top

    fig, ax = plt.subplots(figsize=(11, 5.4))
    for row, (cx, pt, (lo, hi), y) in enumerate(zip(centres, pitch, spans, ys)):
        w = 0.82 * min(pt, 2.0)
        for j, x in enumerate(cx):
            inside = lo <= j <= hi
            ax.add_patch(plt.Rectangle((x - w / 2, y - 0.2), w, 0.4,
                                       fc=C["red"] if inside else C["light"],
                                       ec=C["grey"], lw=0.8, alpha=0.85 if inside else 0.6,
                                       zorder=3))
        # connections from each highlighted cell to the cells above that feed it
        if row > 0:
            _, k, s = RF_LAYERS[row - 1]
            for j in range(lo, hi + 1):
                for src in range(j * s, j * s + k):
                    ax.plot([cx[j], centres[row - 1][src]], [y + 0.2, ys[row - 1] - 0.2],
                            color=C["red"], lw=0.7, alpha=0.45, zorder=1)
        ax.text(-1.1, y, names[row], ha="right", va="center", fontsize=9.5)
        ax.text(len(centres[0]) + 0.2, y, f"$r_{row} = {r_vals[row]}$", va="center",
                fontsize=11, color=C["red"])
    ax.text(len(centres[0]) + 0.2, ys[0] + 0.62, "receptive field\n(cells of this row\n"
            "that feed the unit)", fontsize=8.5, color=C["grey"], va="bottom")
    ax.set_xlim(-6.2, len(centres[0]) + 3.2); ax.set_ylim(-0.7, L + 1.2)
    ax.axis("off")
    ax.set_title(f"one unit of the deepest layer sees {r_total} input positions   ·   "
                 r"working upwards: $r_{l-1} = s_l \cdot r_l + (k_l - s_l)$, from $r_3 = 1$",
                 fontsize=10.5)
    png = save_mpl(fig, ctx, "receptive_field")

    # interactive: reveal one layer at a time, starting from the deepest unit
    def shapes(upto):
        out = []
        for row, (cx, pt, (lo, hi), y) in enumerate(zip(centres, pitch, spans, ys)):
            w = 0.82 * min(pt, 2.0)
            lit = row >= L - upto
            for j, x in enumerate(cx):
                on = lit and lo <= j <= hi
                out.append(dict(type="rect", x0=x - w / 2, x1=x + w / 2, y0=y - 0.2,
                                y1=y + 0.2, line=dict(color=C["grey"], width=1),
                                fillcolor=C["red"] if on else C["light"]))
        return out

    frames = []
    for i in range(L + 1):
        if i == 0:
            expl = f"start at the deepest unit (layer {L}): r_{L} = 1"
        else:
            nm, k, s, rv = steps[i - 1]
            l = L - i + 1
            expl = (f"up through layer {l} ({nm}):  r_{l - 1} = {s}·{r_vals[l]} + "
                    f"({k} − {s}) = {rv}")
        frames.append(go.Frame(name=str(i), layout=go.Layout(shapes=shapes(i), title=expl)))
    fig2 = go.Figure(go.Scatter(x=[-1.1] * (L + 1), y=ys, mode="text", text=names,
                                textposition="middle left", hoverinfo="skip"))
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(list(range(L + 1)), "layers traced: "),
                       shapes=frames[0].layout.shapes, title=frames[0].layout.title.text,
                       showlegend=False)
    fig2.update_xaxes(visible=False, range=[-7.5, len(centres[0]) + 0.5])
    fig2.update_yaxes(visible=False, range=[-0.7, L + 0.7])
    return Figure("receptive_field",
                  "The **receptive field** of a neuron is the size of the region of the "
                  "input to which it can respond. The receptive field is important for "
                  "recognition, as it must be large enough to cover the object being "
                  "detected. The layers are drawn in the same order as in the table of the "
                  "worked example, with the input at the top followed by layers 1 to 3, "
                  "where layer 3 is the deepest, and each row is drawn at its true length. "
                  "Starting from one unit of the deepest layer ($r_3 = 1$), the receptive "
                  r"field is calculated **upwards** with $r_{l-1} = s_l \cdot r_l + (k_l - s_l)$, "
                  "following the connections to the cells that feed each unit. With a "
                  "3-kernel convolution, a 2/2 pooling layer and a further 3-kernel "
                  "convolution, all with valid padding, a receptive field of "
                  f"$r_0 = {r_total}$ input positions is obtained, in agreement with the red "
                  "cells in the input row. It can be seen that the effect of the layers "
                  "after the pooling layer is doubled by its stride of 2. In general the "
                  "receptive field increases with depth, so that later layers can view "
                  "enough of the image to recognise whole objects.",
                  png=png, html=plotly_html(fig2, height=450), kind="plotly")


def _shift_right(x, n, fill=1):
    """Shift without wrapping; np.roll would move the tail round to the front
    and produce a spurious change at the boundary."""
    return np.concatenate([np.full(n, fill, dtype=x.dtype), x[:-n]])


def fig_translation_invariance(ctx: BuildContext) -> Figure:
    k = s = 3
    # The peak sits at index 1, so a one-place shift moves it to index 2 — still
    # inside the same pooling window. A shift of 2 would push it across the
    # boundary into the next window, which is why the invariance is only partial.
    base = np.array([1, 9, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1])
    shifted = _shift_right(base, 1)
    p_base, p_shift = maxpool1d(base, k, s), maxpool1d(shifted, k, s)
    assert np.array_equal(p_base, p_shift), (p_base, p_shift)
    far = maxpool1d(_shift_right(base, 2), k, s)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 5.6),
                             gridspec_kw=dict(width_ratios=[1.8, 1]))
    for row, (sig, pooled, name) in enumerate(
            [(base, p_base, "(a) original"),
             (shifted, p_shift, "(b) shifted right by one position")]):
        ax = axes[row, 0]
        peak = int(np.argmax(sig))
        cols = [C["red"] if i == peak else C["light"] for i in range(len(sig))]
        ax.bar(range(len(sig)), sig, color=cols, edgecolor=C["grey"])
        for i, v in enumerate(sig):
            ax.text(i, v + 0.25, str(int(v)), ha="center", fontsize=8.5)
        for w in range(len(pooled)):          # pooling window boundaries
            ax.axvspan(w * s - 0.5, w * s + k - 0.5, color=C["blue"],
                       alpha=0.06 if w % 2 else 0.13)
            ax.axvline(w * s - 0.5, color=C["grey"], lw=0.8, ls=":")
        ax.set_title(f"{name} — input, with the pooling windows shaded", fontsize=10)
        ax.set_xticks(range(len(sig))); ax.set_yticks([]); ax.set_ylim(0, 11.5)

        ax = axes[row, 1]
        ax.bar(range(len(pooled)), pooled,
               color=[C["red"] if v == pooled.max() else C["light"] for v in pooled],
               edgecolor=C["grey"])
        for i, v in enumerate(pooled):
            ax.text(i, v + 0.25, str(int(v)), ha="center", fontsize=9)
        ax.set_title(f"after max pool k={k}, s={s}  →  {[int(v) for v in pooled]}",
                     fontsize=10)
        ax.set_xticks(range(len(pooled))); ax.set_yticks([]); ax.set_ylim(0, 11.5)

    fig.suptitle("The peak moves from position 1 to position 2 — still inside the "
                 "first pooling window — so the pooled output is unchanged.\n"
                 f"Shift it by 2 instead and the peak crosses into the next window: "
                 f"the output becomes {[int(v) for v in far]}. The invariance is real "
                 "but partial.", fontsize=9.5, y=1.13)
    png = save_mpl(fig, ctx, "translation_invariance")
    return Figure("translation_invariance",
                  "**Translation invariance** is the ability of a network to give the same "
                  "response regardless of where an object appears in the image. Translation "
                  "invariance is built into convolution layers, as the same kernel is applied "
                  "everywhere. Pooling also contributes: here a 1D signal is max pooled with "
                  "a window of 3, and in (b) the signal has been shifted one place to the "
                  "right. The peak (red) moves, but it remains inside the same pooling window "
                  "and is still the largest value in that window, so the pooled output is "
                  f"unchanged, {[int(v) for v in p_base]}. The invariance is, however, "
                  "**partial**. When the peak is shifted by two places instead, it moves into "
                  f"the next window and the output changes to {[int(v) for v in far]}. "
                  "Pooling therefore provides tolerance to shifts smaller than the pooling "
                  "window, but not to larger shifts.",
                  png=png)


def fig_lenet5(ctx: BuildContext) -> Figure:
    rows, total = lenet_table()
    fig = plt.figure(figsize=(13, 6.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.62, 1], hspace=0.04)

    # (a) the architecture in the volume notation of Lesson 1.3: channels along
    # x, spatial size on the square end cap, each operation in the gap between
    # its input and output (shared helper, also used for AlexNet and VGG).
    ax = fig.add_subplot(gs[0])
    ops = {"conv": lambda r: f"conv {r['k']}×{r['k']}\n{r['c']} kernels",
           "pool": lambda r: f"avg pool\n{r['k']}×{r['k']}, s=2"}
    maps = [dict(h=r["h"], c=r["c"], kind=r["kind"], k=r["k"],
                 op=ops[r["kind"]](r) if r["kind"] in ops else None)
            for r in rows if r["kind"] in ("input", "conv", "pool")]
    vec_ops = iter(["flatten", "dense", "dense", "dense\n+ softmax"])
    vecs = [dict(n=r["c"], op=next(vec_ops)) for r in rows if r["kind"] in ("flat", "dense")]
    right, _ = network_diagram(ax, maps, vecs, ymid=1.55, gap=1.55, vec_gap=1.03,
                               vec_height=lambda n: 0.30 + np.sqrt(n / 400),
                               op_y=0.72, shape_y=0.32)
    ax.set_xlim(-0.25, right + 0.15); ax.set_ylim(0.05, 2.85); ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title("(a) LeNet-5 (1998): handwritten digit recognition on 32 × 32 "
                 "greyscale images — channels along x, spatial size on the end cap",
                 fontsize=10.5)

    # (b) the layer table, every parameter count computed from the shapes
    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    hdr = ["layer", "output shape", "parameters"]
    cells = [[r["name"], r["shape"], f"{r['params']:,}" if r["params"] else "—"]
             for r in rows]
    cells.append(["total", "", f"{total:,}"])
    tbl = ax.table(cellText=cells, colLabels=hdr, loc="center",
                   colWidths=[0.42, 0.24, 0.22], cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    for (r_, c_), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["light"])
        if r_ == 0:
            cell.set_text_props(weight="bold")
        if r_ == len(cells):
            cell.set_text_props(weight="bold", color=C["red"])
    png = save_mpl(fig, ctx, "lenet5")
    conv_p = sum(r["params"] for r in rows if r["kind"] == "conv")
    dense_p = sum(r["params"] for r in rows if r["kind"] == "dense")
    return Figure("lenet5",
                  "LeNet-5, developed in 1998 for handwritten digit recognition, which is "
                  "small enough to be checked by hand. There are two convolution layers, two "
                  "pooling layers and three fully connected layers. (a) As in Lesson 1.3, "
                  "the feature maps are drawn as volumes, with the channels along the "
                  "horizontal axis and the spatial size on the square end cap. Each "
                  "operation is shown between its input and output volumes: the dashed "
                  "cube inside a volume is one kernel, or one pooling window, and the "
                  "dotted lines connect it to the single output value that it produces. "
                  "A kernel spans every channel of its input, whereas a pooling window is "
                  "applied to each channel separately, so the number of channels is "
                  "unchanged by pooling. The flattened vector and the dense layers are "
                  "drawn as bars, with dotted lines representing the full connections. "
                  "(b) The shapes are "
                  "determined by the rules from the previous lessons: the first convolution "
                  "has 6 kernels and so produces 6 feature maps, and its kernels have 1 "
                  "channel as the input is greyscale; the second convolution has 16 kernels, "
                  "each of which must have 6 channels to match the 6 feature maps before it. "
                  "The spatial size is halved by pooling and the number of channels is "
                  "unchanged. Every parameter count in the table is calculated from the "
                  f"shapes, and the total is {total:,}. Only {conv_p:,} of these are in the "
                  f"convolution layers, while {dense_p:,} are in the three dense layers, as "
                  "predicted in Lesson 1.2; this problem is discussed further in Section 2.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_fc_on_images(ctx), fig_conv_as_nn(ctx), fig_cnn_structure(ctx),
            fig_learned_features(ctx), fig_receptive_field(ctx),
            fig_translation_invariance(ctx), fig_lenet5(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("1.4 Convolutional neural networks")
    tab1, tab2, tab3 = st.tabs(["Dense vs conv", "Receptive field", "LeNet-5"])

    with tab1:
        c = st.columns(3)
        side = c[0].number_input("image side", 4, 1024, 256)
        ch = c[1].number_input("channels", 1, 4, 1)
        neurons = c[2].number_input("neurons", 1, 4096, 100)
        dense = side * side * ch * neurons + neurons
        k = st.slider("compare with one conv layer, kernel k", 1, 11, 3)
        conv = k * k * ch * neurons + neurons
        c1, c2 = st.columns(2)
        c1.metric("dense layer", f"{dense:,}")
        c2.metric(f"conv layer, {neurons} kernels", f"{conv:,}",
                  f"−{100 * (1 - conv / dense):.3f}%")
        st.caption("The conv layer's parameter count does not depend on the image "
                   "size at all — only on the kernel size, the channels and the "
                   "number of kernels.")

    with tab2:
        st.markdown("Add layers from input to output; the receptive field is found "
                    "by walking back from the deepest one.")
        n = st.slider("number of layers", 1, 6, 3)
        layers = []
        for i in range(n):
            c = st.columns(3)
            c[0].write(f"layer {i + 1}")
            k = c[1].number_input("k", 1, 11, 3, key=f"k{i}")
            s = c[2].number_input("s", 1, 4, 1, key=f"s{i}")
            layers.append((f"layer {i + 1}", int(k), int(s)))
        r, steps = receptive_field(layers)
        for name, k, s, rv in steps:
            st.latex(rf"\text{{back through {name}}}:\quad r = {s}\cdot r + ({k}-{s})"
                     rf" = {rv}")
        st.success(f"Receptive field at the input: {r} positions")

    with tab3:
        rows, total = lenet_table()
        st.dataframe([{"layer": r["name"], "output shape": r["shape"],
                       "parameters": r["params"]} for r in rows],
                     width="stretch")
        conv_p = sum(r["params"] for r in rows if r["kind"] == "conv")
        dense_p = sum(r["params"] for r in rows if r["kind"] == "dense")
        c1, c2, c3 = st.columns(3)
        c1.metric("total parameters", f"{total:,}")
        c2.metric("in conv layers", f"{conv_p:,}", f"{100 * conv_p / total:.0f}%")
        c3.metric("in dense layers", f"{dense_p:,}", f"{100 * dense_p / total:.0f}%")
