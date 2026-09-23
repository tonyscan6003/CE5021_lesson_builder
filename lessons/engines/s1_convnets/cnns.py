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
                         h=h, w=w, c=c))
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
                  "The naive way to process an image with a neural network is to "
                  "flatten it into a vector and feed it to a fully connected layer. "
                  "(a) For a 4 × 4 image and two neurons that is 34 parameters, which "
                  "is fine. (b) But the count grows with the *square* of the image "
                  "side. A 256 × 256 greyscale image is a vector of 65,536 values, and "
                  "connecting it to even a modest 100-neuron layer needs 6,553,700 "
                  "parameters — for one layer, before any learning has happened. "
                  "Worse, flattening throws away the spatial arrangement: two pixels "
                  "that are neighbours in the image are nothing special to a dense "
                  "layer. Both problems point at the same fix.", png=png)


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
                  "The same convolution drawn as a neural network. Each position of "
                  "the kernel becomes one neuron: the multiply-and-sum the kernel "
                  "performs there is exactly the weighted sum a neuron computes, and "
                  "adding a bias and an activation function makes it a neuron in the "
                  "full sense. The result looks like the dense networks of Lesson 1.2, "
                  "with two crucial differences. It is **not fully connected** — each "
                  "neuron sees only the 4 pixels under the kernel, which saves "
                  "computation. And every neuron uses the **same** weights and the "
                  "same bias, which saves memory. Those two properties, local "
                  "connectivity and weight sharing, are what a convolution layer buys "
                  "you over a dense one.", png=png)


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
                  "The basic structure of a CNN, drawn as a sequence of operations "
                  "rather than feature maps. Convolution layers come first and are "
                  "stacked to make the network deeper, which helps it learn complex "
                  "features. The activation is drawn as its own layer: it is good "
                  "practice to show it explicitly, because in modern CNNs a "
                  "convolution layer sometimes appears without one. Pooling layers "
                  "summarise features and halve the spatial size, and the number of "
                  "kernels is typically increased as the spatial size falls. Finally "
                  "the feature map is flattened to a vector and fed to a dense layer "
                  "with one neuron per class, whose softmax activation gives the class "
                  "probabilities.", png=png)


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
                  "During training the kernel parameters are learned, so after "
                  "training the convolution layers respond to whatever features turn "
                  "out to be useful. It helps to think of a convolution layer as "
                  "cross-correlating the image with those learned features: the "
                  "response is large where the image looks like the kernel. Early "
                  "layers consistently learn simple things — oriented edges, colour "
                  "boundaries. Deeper layers respond to textures, then to object "
                  "parts, then to whole objects. The classifier stage reads the "
                  "pattern of responses in the final feature map and outputs a class "
                  "probability. **These panels are synthetic patterns illustrating the "
                  "progression, not kernels from a trained network** — Lesson 2.5 "
                  "covers how real learned features are visualised.", png=png)


def fig_receptive_field(ctx: BuildContext) -> Figure:
    r_total, steps = receptive_field(RF_LAYERS)
    n_in = 12
    # the input pixels feeding one deepest-layer neuron, traced back layer by layer
    spans = []
    lo, hi = 0, 0                       # one neuron at the deepest layer
    for name, k, s in reversed(RF_LAYERS):
        lo, hi = lo * s, hi * s + (k - 1)
        spans.append((name, lo, hi))

    fig, axes = plt.subplots(len(spans) + 1, 1, figsize=(10, 5.6), sharex=True)
    titles = ["deepest layer: one neuron"] + [f"back through {n}" for n, _, _ in spans]
    extents = [(0, 0)] + [(lo, hi) for _, lo, hi in spans]
    for ax, title, (lo_, hi_) in zip(axes, titles, extents):
        for i in range(n_in):
            inside = lo_ <= i <= hi_
            ax.add_patch(plt.Rectangle((i, 0), 0.9, 0.8,
                                       fc=C["red"] if inside else C["light"],
                                       ec=C["grey"], lw=0.8,
                                       alpha=0.85 if inside else 0.6))
        ax.set_xlim(-0.3, n_in); ax.set_ylim(-0.1, 0.95)
        ax.set_yticks([])
        ax.set_title(f"{title}  →  {hi_ - lo_ + 1} input positions", fontsize=9.5,
                     loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[-1].set_xticks(range(n_in))
    axes[-1].set_xlabel("position along the 1D input")
    fig.suptitle(f"receptive field at the input = {r_total} positions      "
                 r"$r_{l-1} = s_l \cdot r_l + (k_l - s_l)$", fontsize=10.5)
    png = save_mpl(fig, ctx, "receptive_field")

    # interactive: step back one layer at a time, with the arithmetic shown
    fig2 = go.Figure()
    frames = []
    running = [("deepest layer", 1)] + [(n, r) for n, _, _, r in steps]
    for i, ((lo_, hi_), (name, r)) in enumerate(zip(extents, running)):
        colours = [C["red"] if lo_ <= j <= hi_ else C["light"] for j in range(n_in)]
        if i == 0:
            expl = "start at the deepest layer: r = 1"
        else:
            nm, k, s, rv = steps[i - 1]
            prev = running[i - 1][1]
            expl = f"back through {nm}:  r = {s}·{prev} + ({k} − {s}) = {rv}"
        frames.append(go.Frame(name=str(i), data=[
            go.Bar(x=list(range(n_in)), y=[1] * n_in, marker_color=colours,
                   hoverinfo="skip")],
            traces=[0], layout=go.Layout(title=expl)))
    fig2.add_trace(frames[0].data[0])
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(list(range(len(frames))),
                                              "layers back: "),
                       title=frames[0].layout.title.text, showlegend=False,
                       bargap=0.12)
    fig2.update_yaxes(showticklabels=False, range=[0, 1.25])
    fig2.update_xaxes(title_text="position along the 1D input",
                      tickmode="array", tickvals=list(range(n_in)))
    return Figure("receptive_field",
                  "The **receptive field** of a neuron is the size of the region of "
                  "the input it can respond to. It matters for recognition: the "
                  "receptive field has to be large enough to cover the object being "
                  "detected. It is found by working backwards through the preceding "
                  "convolution and pooling layers with "
                  r"$r_{l-1} = s_l \cdot r_l + (k_l - s_l)$, starting from $r = 1$ at "
                  "the deepest layer. Here a 3-kernel convolution, a 2/2 pooling layer "
                  "and another 3-kernel convolution, all with valid padding, give a "
                  f"receptive field of {r_total} input positions — and the red cells "
                  "counted in the bottom row agree with the recurrence, because with "
                  "valid padding the traced region never runs off the edge of the "
                  "input. Note how the pooling layer's stride of 2 doubles the effect "
                  "of everything before it. Deeper into a network the receptive field "
                  "generally grows, which is the point: later layers need to see "
                  "enough of the image to recognise whole objects.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


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
                  "**Translation invariance** is the ability of a network to give the "
                  "same response wherever in the image an object appears. Convolution "
                  "layers build it in automatically, because the same kernel is applied "
                  "everywhere. Pooling contributes as well: here a 1D signal is max "
                  "pooled with a window of 3, and in (b) the signal has been shifted "
                  "one place to the right. The peak (red) moves, but it stays inside "
                  "the same pooling window and is still the largest value in it, so "
                  f"the pooled output is exactly the same, {[int(v) for v in p_base]}. "
                  "Be precise about what this does and does not give you: the "
                  "invariance is **partial**. Shift the peak by two places instead and "
                  "it crosses into the next window, changing the output to "
                  f"{[int(v) for v in far]}. Pooling buys tolerance to shifts smaller "
                  "than the pooling window, not to arbitrary ones.", png=png)


def fig_lenet5(ctx: BuildContext) -> Figure:
    rows, total = lenet_table()
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.78, 1], hspace=0.10)

    # (a) the architecture, feature maps drawn to scale
    ax = fig.add_subplot(gs[0])
    # Fixed label rows so the names and shapes line up instead of tracking the
    # height of each block.
    Y_MID, Y_NAME, Y_SHAPE = 1.5, 2.78, 0.42
    x = 0.0
    for r in rows:
        if r["kind"] in ("flat", "dense"):
            h = min(r["c"] / 400 * 2.0, 2.0)
            ax.add_patch(plt.Rectangle((x, Y_MID - h / 2), 0.22, h, fc=C["purple"],
                                       ec=C["ink"], lw=1.1, alpha=0.6))
            ax.text(x + 0.11, Y_SHAPE, r["shape"], ha="center", va="top",
                    fontsize=8.5, color=C["red"], family="monospace")
            ax.text(x + 0.11, Y_NAME, r["name"].replace(" + softmax", ""),
                    ha="center", va="bottom", fontsize=8)
            x += 0.78
            continue
        side = r["h"] / 32 * 1.7
        depth = min(0.10 + 0.020 * r["c"], 0.50)
        col = {"input": C["light"], "conv": C["blue"], "pool": C["orange"]}[r["kind"]]
        nshow = min(r["c"], 6)
        for d in range(nshow):
            off = d * depth / max(nshow, 1)
            ax.add_patch(plt.Rectangle((x + off, Y_MID - side / 2 + off * 0.55), side,
                                       side, fc=col, ec=C["ink"], lw=0.7, alpha=0.75))
        ax.text(x + side / 2, Y_SHAPE, r["shape"], ha="center", va="top",
                fontsize=8.5, color=C["red"], family="monospace")
        ax.text(x + side / 2, Y_NAME, r["name"], ha="center", va="bottom", fontsize=8)
        x += side + 0.80
    ax.set_xlim(-0.25, x); ax.set_ylim(0.22, 3.08); ax.axis("off")
    ax.set_title("(a) LeNet-5 (1998): handwritten digit recognition on 32 × 32 "
                 "greyscale images", fontsize=10.5)

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
                  "LeNet-5, developed in 1998 for handwritten digit recognition, and "
                  "small enough to check by hand. Two convolution layers, two pooling "
                  "layers and three fully connected layers. Note how the rules from "
                  "the previous lessons pin everything down: the first convolution has "
                  "6 kernels so it produces 6 feature maps, and its kernels have 1 "
                  "channel because the input is greyscale; the second has 16 kernels "
                  "so its kernels must have 6 channels, matching the 6 maps before it. "
                  "Pooling halves the spatial size and leaves the channel count alone. "
                  "Every parameter count in the table is computed from the shapes "
                  f"rather than quoted, and they total {total:,}. The split is worth "
                  f"noting: the convolution layers hold only {conv_p:,} of them while "
                  f"the three dense layers hold {dense_p:,} — the same lopsidedness "
                  "Lesson 1.2 predicted, and a problem Section 2 returns to.", png=png)


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
