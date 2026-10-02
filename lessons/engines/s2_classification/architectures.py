"""Lesson 2.2 — Architectures for image classification.

Figures
    alexnet_diagram     AlexNet as feature-map volumes, every shape computed
    alexnet_cost        per-layer parameters and multiplications: conv vs dense
    vgg_stacking        VGG-16's blocks, and why stacked 3x3 beats one large kernel
    model_zoo           top-1 accuracy against operations, bubble = parameters
                        (interactive)
    network_in_network  linear convolution vs the mlpconv layer
    global_avg_pool     GAP mechanics, and the parameters it saves over a dense head
    inception_module    the naive module vs the module with 1x1 bottlenecks
    googlenet           the 22-layer stack, per-module cost and the auxiliary heads

Every parameter and operation count is computed by engines/_arch.py from the
layer specification; see _arch.verify() for the checks against published counts.

Source: reference/legacy_md/lesson_2_2_architectures_image_Classification +
lesson_2_2_script.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .._common import BuildContext, Figure, C, SERIES, save_mpl, plotly_html, sync_matrix
from .._volumes import OBLIQUE, chan_len, spatial_side, volume, kernel_cube
from .. import _arch as A

FIGURES = ["alexnet_diagram", "alexnet_cost", "vgg_stacking", "model_zoo",
           "network_in_network", "global_avg_pool", "inception_module", "googlenet"]

# ImageNet single-crop top-1 accuracy (%) for torchvision's reference weights.
# Accuracy is quoted, not computed; parameters and MACs come from _arch, which
# reproduces torchvision's published counts.
ZOO = [
    ("AlexNet",      lambda: A.alexnet(grouped=True), 56.5),
    ("VGG-16",       A.vgg16,                         71.6),
    ("GoogLeNet",    A.googlenet,                     69.8),
    ("ResNet-18",    lambda: A.resnet(18),            69.8),
    ("ResNet-50",    lambda: A.resnet(50),            76.1),
    ("ResNet-152",   lambda: A.resnet(152),           78.3),
    ("DenseNet-121", A.densenet121,                   74.4),
    ("MobileNetV2",  A.mobilenet_v2,                  71.9),
]


def _fmt(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else (f"{n / 1e3:.0f}K" if n >= 1e3 else str(n))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_alexnet_diagram(ctx: BuildContext) -> Figure:
    net = A.alexnet()
    maps = [l for l in net.layers if l.kind in ("input", "conv", "pool")]
    vecs = [l for l in net.layers if l.kind == "fc"]
    fig, ax = plt.subplots(figsize=(14, 5.0))
    x, ymid = 0.0, 2.1
    for l in maps:
        c, h, w = l.out
        face = {"input": C["light"], "conv": C["blue"], "pool": C["orange"]}[l.kind]
        # this network spans 227 px down to 6 px, beyond the range _volumes'
        # spatial_side() is tuned for, so scale against the input size here
        side = 0.6 + 1.9 * np.sqrt(h / 227)
        right = volume(ax, x, ymid, chan_len(c), side, face, lw=1.0, alpha=0.8)
        ax.text(x + (chan_len(c) + OBLIQUE[0] * side) / 2, ymid - side / 2 - 0.12,
                f"{h}×{w}×{c}", ha="center", va="top", fontsize=8.2, color=C["red"],
                family="monospace")
        nm = l.name.replace(" ", "\n", 1) if l.kind != "input" else "input"
        ax.text(x + chan_len(c) / 2, ymid + side / 2 + OBLIQUE[1] * side + 0.1, nm,
                ha="center", va="bottom", fontsize=7.8)
        x = right + 0.32
    for l in vecs:
        n = l.out[0]
        hh = 0.5 + 2.2 * np.log10(n) / np.log10(4096)
        ax.add_patch(plt.Rectangle((x, ymid - hh / 2), 0.2, hh, fc=C["purple"],
                                   ec=C["ink"], lw=1.0, alpha=0.6))
        ax.text(x + 0.1, ymid - hh / 2 - 0.12, str(n), ha="center", va="top",
                fontsize=8.2, color=C["red"], family="monospace")
        ax.text(x + 0.1, ymid + hh / 2 + 0.1, l.name.split()[0], ha="center",
                va="bottom", fontsize=7.8)
        x += 0.62
    ax.set_xlim(-0.2, x); ax.set_ylim(0.35, 4.75); ax.axis("off")
    ax.set_title("AlexNet (single-GPU form): 5 convolution, 3 pooling and 3 dense "
                 "layers. Volumes drawn with the channel axis horizontal.", fontsize=10.5)
    png = save_mpl(fig, ctx, "alexnet_diagram")
    return Figure("alexnet_diagram",
                  "AlexNet, the first network to win the ImageNet challenge, drawn in "
                  "the volume notation of Lesson 1.3: the horizontal length of each box "
                  "is its channel count and the end cap its spatial size. Structurally "
                  "it is LeNet-5 scaled up — 5 convolution, 3 pooling and 3 dense layers "
                  "ending in a 1000-way softmax — and the scale is the difference. The "
                  "first layer uses large 11 × 11 kernels at stride 4 on the 227 × 227 "
                  "input, and 96 of them. Going deeper, the channel count rises to 384 "
                  "while pooling shrinks the spatial size, the usual trade that keeps "
                  "each layer's cost in balance, and the kernels get smaller. Every "
                  "shape here is computed with Lesson 1.1's output-size formula. (The "
                  "paper says the input is 224 × 224, but an 11 × 11 kernel at stride 4 "
                  "only produces the paper's 55 × 55 from 227 — try it.)", png=png)


def fig_alexnet_cost(ctx: BuildContext) -> Figure:
    net = A.alexnet()
    layers = [l for l in net.layers if l.kind in ("conv", "fc")]
    names = [l.name.split()[0] for l in layers]
    params = np.array([l.params for l in layers])
    macs = np.array([l.macs for l in layers])
    is_fc = np.array([l.kind == "fc" for l in layers])
    cols = [C["purple"] if f else C["blue"] for f in is_fc]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    for ax, vals, title in [(axes[0], params, "parameters (memory)"),
                            (axes[1], macs, "multiply-accumulates (computation)")]:
        ax.bar(names, vals, color=cols, edgecolor=C["ink"], lw=0.7)
        ax.set_yscale("log")
        for i, v in enumerate(vals):
            ax.text(i, v * 1.25, _fmt(v), ha="center", fontsize=8)
        conv_share = vals[~is_fc].sum() / vals.sum()
        ax.set_title(f"{title}\nconvolution {conv_share:.0%}  ·  dense {1 - conv_share:.0%}",
                     fontsize=10.5)
        ax.set_ylim(vals.min() / 3, vals.max() * 8)
    axes[0].set_ylabel("count (log scale)")
    tot_p, tot_m = net.params, net.macs
    fig.suptitle(f"total: {tot_p:,} parameters ({tot_p * 2 / 1e6:.0f} MB at 16-bit)   ·   "
                 f"{tot_m / 1e9:.2f} G multiply-accumulates per image", fontsize=10.5)
    png = save_mpl(fig, ctx, "alexnet_cost")
    g = A.alexnet(grouped=True)
    return Figure("alexnet_cost",
                  "Where AlexNet's memory and computation go, layer by layer — note the "
                  "log scales. The memory requirement is the parameter count: for a "
                  "convolution layer the sum of its kernels' weights, for a dense layer "
                  "inputs × neurons. The convolution layers hold tens to hundreds of "
                  "thousands each, the dense layers tens of millions, so the dense "
                  f"layers hold {params[is_fc].sum() / params.sum():.0%} of the total. "
                  "Computation is the other way round: sliding kernels across the image "
                  "multiplies each weight many times, so the convolution layers do "
                  f"{macs[~is_fc].sum() / macs.sum():.0%} of the work. This matters in "
                  "practice: dense layers are limited by memory bandwidth and do not use "
                  "a GPU's full speed, while convolution does. These figures are for the "
                  f"single-GPU form drawn above. The original two-GPU AlexNet split "
                  "conv2, conv4 and conv5 into halves, which reduces it to "
                  f"{g.params:,} parameters and {g.macs / 1e9:.2f} G multiply-"
                  "accumulates — the figures usually quoted.", png=png)


def fig_vgg_stacking(ctx: BuildContext) -> Figure:
    net = A.vgg16()
    fig = plt.figure(figsize=(13.5, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1], width_ratios=[1.35, 1],
                          hspace=0.28, wspace=0.18)

    # (a) VGG-16 as blocks, shapes and parameters computed
    ax = fig.add_subplot(gs[0, :])
    st = net.stage_totals()
    blocks = [f"block{i}" for i in range(1, 6)] + ["fc"]
    shapes = {}
    for l in net.layers:
        if l.kind == "pool":
            shapes[l.stage] = l.out
    x = 0.0
    for b in blocks:
        convs = [l for l in net.layers if l.stage == b and l.kind == "conv"]
        p = st[b][0]
        if b == "fc":
            w = 2.2
            ax.add_patch(plt.Rectangle((x, 0), w, 1, fc=C["purple"], alpha=0.5,
                                       ec=C["ink"], lw=1.1))
            ax.text(x + w / 2, 0.5, "3 dense\n4096·4096·1000", ha="center", va="center",
                    fontsize=8.5)
        else:
            w = 0.62 * len(convs) + 0.55
            for i, l in enumerate(convs):
                ax.add_patch(plt.Rectangle((x + i * 0.62, 0), 0.56, 1, fc=C["blue"],
                                           alpha=0.55, ec=C["ink"], lw=1.0))
                ax.text(x + i * 0.62 + 0.28, 0.5, f"3×3\n{l.out[0]}", ha="center",
                        va="center", fontsize=7.3)
            px = x + len(convs) * 0.62
            ax.add_patch(plt.Rectangle((px, 0.15), 0.42, 0.7, fc=C["orange"],
                                       alpha=0.6, ec=C["ink"], lw=1.0))
            ax.text(px + 0.21, 0.5, "pool", ha="center", va="center", fontsize=7,
                    rotation=90)
            c, h, ww = shapes[b]
            ax.text(x + w / 2, -0.13, f"→ {h}×{ww}×{c}", ha="center", va="top",
                    fontsize=8, color=C["red"], family="monospace")
        ax.text(x + w / 2, 1.12, f"{b}: {_fmt(p)} params", ha="center", fontsize=8.3)
        x += w + 0.25
    ax.set_xlim(-0.1, x); ax.set_ylim(-0.5, 1.45); ax.axis("off")
    fc_p = st["fc"][0]
    ax.set_title(f"(a) VGG-16: 13 convolutions, all 3 × 3, in five blocks — "
                 f"{net.params:,} parameters, of which the dense layers hold "
                 f"{fc_p / net.params:.0%}", fontsize=10.5, loc="left")

    # (b) receptive field of a stack of 3x3 vs one large kernel
    ax = fig.add_subplot(gs[1, 0])
    n = 9
    for row, (k_list, label) in enumerate([([3, 3, 3], "three stacked 3 × 3"),
                                           ([7], "one 7 × 7")]):
        y0 = 1.6 - row * 1.6
        # receptive field grows by k-1 per stride-1 layer
        span = 1
        for li, k in enumerate(k_list):
            span += k - 1
        for i in range(n):
            inside = (n - span) // 2 <= i < (n - span) // 2 + span
            ax.add_patch(plt.Rectangle((i, y0), 0.9, 0.9, fc=C["red"] if inside else
                                       C["light"], alpha=0.75 if inside else 0.5,
                                       ec=C["grey"], lw=0.6))
        ax.text(-0.3, y0 + 0.45, label, ha="right", va="center", fontsize=9.5)
        ax.text(n + 0.2, y0 + 0.45, f"receptive field {span}", va="center", fontsize=9.5)
    ax.set_xlim(-4.3, n + 3.6); ax.set_ylim(-0.3, 2.8); ax.axis("off")
    ax.set_title("(b) same receptive field …", fontsize=10.5, loc="left")

    # (c) parameters per channel pair
    ax = fig.add_subplot(gs[1, 1])
    labels = ["2 × (3×3)", "5×5", "3 × (3×3)", "7×7"]
    vals = [2 * 9, 25, 3 * 9, 49]
    ax.bar(labels, vals, color=[C["blue"], C["grey"], C["blue"], C["grey"]],
           edgecolor=C["ink"], lw=0.7)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, str(v), ha="center", fontsize=9.5)
    ax.text(0.5, 40, "receptive\nfield 5", ha="center", fontsize=8.5, color=C["grey"])
    ax.text(2.5, 54, "receptive\nfield 7", ha="center", fontsize=8.5, color=C["grey"])
    ax.set_ylabel("weights per (C_in, C_out) pair")
    ax.set_ylim(0, 66)
    ax.set_title("(c) … for fewer parameters and operations", fontsize=10.5, loc="left")
    png = save_mpl(fig, ctx, "vgg_stacking")
    return Figure("vgg_stacking",
                  "(a) VGG-16 has the same overall plan as AlexNet — convolution, "
                  "pooling, then dense layers — but every convolution is 3 × 3 and "
                  "they are stacked in blocks, giving 13 convolution layers instead of "
                  "5, with the channel count reaching 512. (b) The reason for the "
                  "stacking: back-to-back 3 × 3 convolutions see as large a region of "
                  "the input as one larger kernel — two give a receptive field of 5, "
                  "three give 7, by Lesson 1.4's formula. (c) But they need fewer "
                  "weights and fewer multiplications to do it: 27 per channel pair for "
                  "three 3 × 3 layers against 49 for one 7 × 7, and there is an "
                  "activation between each, so the stack is also more expressive. "
                  "VGG keeps AlexNet's large dense layers, though, and they still hold "
                  "most of its parameters — the problem the rest of this lesson "
                  "addresses.", png=png)


def fig_model_zoo(ctx: BuildContext) -> Figure:
    rows = []
    for name, fn, acc in ZOO:
        net = fn()
        rows.append((name, acc, net.macs / 1e9, net.params / 1e6))
    # GoogLeNet and ResNet-18 land almost on top of each other, so labels sit
    # beside their bubbles rather than on them
    offset = {"GoogLeNet": (-14, 0, "right"), "ResNet-18": (14, 0, "left"),
              "MobileNetV2": (0, 14, "center"), "DenseNet-121": (-14, 6, "right")}
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for i, (name, acc, g, p) in enumerate(rows):
        ax.scatter(g, acc, s=p * 22 + 30, color=SERIES[i % len(SERIES)], alpha=0.55,
                   edgecolor=C["ink"], lw=0.8)
        dx, dy, ha = offset.get(name, (0, -2, "center"))
        ax.annotate(f"{name}\n{p:.1f}M", (g, acc), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("operations per image (G multiply-accumulates, log scale)")
    ax.set_ylabel("ImageNet top-1 accuracy (%)")
    ax.set_xlim(0.15, 40); ax.set_ylim(52, 82)
    ax.set_title("Accuracy against computation; bubble area ∝ parameters", fontsize=10.5)
    ax.grid(alpha=0.25, which="both")
    png = save_mpl(fig, ctx, "model_zoo")

    fig2 = go.Figure()
    for i, (name, acc, g, p) in enumerate(rows):
        fig2.add_trace(go.Scatter(
            x=[g], y=[acc], mode="markers+text", text=[name], textposition="top center",
            marker=dict(size=8 + np.sqrt(p) * 5, color=SERIES[i % len(SERIES)],
                        opacity=0.6, line=dict(color=C["ink"], width=1)),
            hovertemplate=(f"<b>{name}</b><br>top-1 {acc}%<br>{g:.2f} G MACs<br>"
                           f"{p:.1f} M parameters<extra></extra>"), showlegend=False))
    fig2.update_xaxes(type="log", title_text="G multiply-accumulates per image")
    fig2.update_yaxes(title_text="ImageNet top-1 accuracy (%)", range=[52, 82])
    fig2.update_layout(title="Hover a network for its numbers")
    return Figure("model_zoo",
                  "The model zoo: ImageNet top-1 accuracy against the computation "
                  "needed per image, with bubble area proportional to the parameter "
                  "count. AlexNet sits bottom left — cheap to run and not very accurate, "
                  "but with a very large number of parameters. VGG-16 is more accurate "
                  "but costs far more computation and has the largest parameter count of "
                  "all, which makes it unsuitable for mobile devices and slow to train, "
                  "since its dense layers are limited by memory bandwidth. Those "
                  "concerns drove the search for architectures without large dense "
                  "layers: GoogLeNet, ResNet, DenseNet and MobileNetV2 all reach VGG's "
                  "accuracy or better with a fraction of its parameters. Parameters and "
                  "operations are computed from each architecture (and match torchvision's "
                  "published counts); the accuracies are torchvision's published "
                  "single-crop results.", png=png, html=plotly_html(fig2, height=480),
                  kind="plotly")


def fig_network_in_network(ctx: BuildContext) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for ax, mlp, title in [(axes[0], False, "(a) ordinary convolution: one linear "
                                            "filter, then activation"),
                           (axes[1], True, "(b) mlpconv: a small MLP slid across "
                                           "the input")]:
        ax.add_patch(plt.Rectangle((0, 0), 3, 3, fc=C["light"], ec=C["ink"], lw=1.2))
        for i in range(4):
            ax.plot([0, 3], [i * 0.75, i * 0.75], color=C["grey"], lw=0.3)
            ax.plot([i * 0.75, i * 0.75], [0, 3], color=C["grey"], lw=0.3)
        ax.add_patch(plt.Rectangle((0.75, 0.75), 1.5, 1.5, fill=False, ec=C["red"], lw=2.2))
        ax.text(1.5, -0.35, "input patch", ha="center", fontsize=9)
        if not mlp:
            ax.scatter([5.0], [1.5], s=900, facecolors="white", edgecolors=C["blue"],
                       linewidths=2.2, zorder=3)
            ax.text(5.0, 1.5, "w·x+b\nσ", ha="center", va="center", fontsize=8.5)
            ax.annotate("", xy=(4.6, 1.5), xytext=(2.3, 1.5),
                        arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.4))
            ax.annotate("", xy=(6.6, 1.5), xytext=(5.4, 1.5),
                        arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.4))
            ax.text(7.0, 1.5, "one value per\nposition", va="center", fontsize=9)
        else:
            layers = [3, 4, 2]
            xs = [4.0, 5.2, 6.4]
            pos = [[(xx, y) for y in np.linspace(0.4, 2.6, n)] for xx, n in zip(xs, layers)]
            for a, b in zip(pos, pos[1:]):
                for p in a:
                    for q in b:
                        ax.plot([p[0], q[0]], [p[1], q[1]], color=C["light"], lw=0.8,
                                zorder=1)
            for ps in pos:
                ax.scatter([p[0] for p in ps], [p[1] for p in ps], s=150,
                           facecolors="white", edgecolors=C["blue"], linewidths=1.8,
                           zorder=3)
            ax.annotate("", xy=(3.75, 1.5), xytext=(2.3, 1.5),
                        arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.4))
            ax.text(5.2, -0.35, "= 1 × 1 convolutions in Inception", ha="center",
                    fontsize=9, color=C["red"])
            ax.text(7.0, 1.5, "richer feature\nper position", va="center", fontsize=9)
        ax.set_xlim(-0.3, 8.6); ax.set_ylim(-0.7, 3.4); ax.set_aspect("equal")
        ax.axis("off"); ax.set_title(title, fontsize=10)
    png = save_mpl(fig, ctx, "network_in_network")
    return Figure("network_in_network",
                  "The network-in-network idea (Lin, 2013). (a) An ordinary convolution "
                  "is a linear filter followed by an activation, so the features it can "
                  "represent at each position are limited — which is what drives "
                  "networks like AlexNet and VGG to need large dense layers at the "
                  "output to separate the classes. (b) Network in network replaces the "
                  "filter with a small multilayer perceptron, still slid across the "
                  "input exactly like a convolution. That adds parameters locally and "
                  "makes each feature more discriminative, so the large dense output "
                  "layers can be replaced by global average pooling. Applying the same "
                  "small dense network at every position is exactly what a stack of "
                  "1 × 1 convolutions does, which is how Inception implements the idea "
                  "efficiently.", png=png)


def fig_global_avg_pool(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(4)
    maps = rng.integers(0, 9, (3, 4, 4))
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.55, 2.3], wspace=0.35)
    for i in range(3):
        ax = fig.add_subplot(gs[i])
        sync_matrix(ax, maps[i], title=f"map {i + 1} → mean {maps[i].mean():.2f}",
                    cmap=["Blues", "Greens", "Oranges"][i], fontsize=9)
    ax = fig.add_subplot(gs[3])
    sync_matrix(ax, maps.mean((1, 2))[:, None], fmt="{:.2f}", cmap="Purples",
                title="GAP\noutput", fontsize=9)

    # parameters saved on VGG-16's head: 7x7x512 -> classifier
    ax = fig.add_subplot(gs[4])
    vgg = A.vgg16()
    head_dense = sum(l.params for l in vgg.layers if l.kind == "fc")
    gap_head = 512 * 1000 + 1000
    bars = ax.barh(["VGG-16 head:\nflatten + 3 dense", "global average pool\n+ 1 dense"],
                   [head_dense, gap_head], color=[C["purple"], C["green"]],
                   edgecolor=C["ink"], lw=0.8)
    for b, v in zip(bars, [head_dense, gap_head]):
        ax.text(v * 1.03, b.get_y() + b.get_height() / 2, f"{v:,}", va="center",
                fontsize=9.5)
    ax.set_xscale("log"); ax.set_xlim(1e5, 1e9)
    ax.set_xlabel("parameters (log scale)")
    ax.set_title(f"on VGG-16's 7×7×512 output: {head_dense / gap_head:.0f}× fewer",
                 fontsize=10)
    png = save_mpl(fig, ctx, "global_avg_pool")
    return Figure("global_avg_pool",
                  "Global average pooling, also proposed in the network-in-network "
                  "paper, as an alternative to dense layers acting as the classifier. "
                  "It reduces a feature map of any spatial size to a single number — "
                  "its mean — so C feature maps become a vector of length C, whatever "
                  "the input image size. A dense layer with softmax then turns that "
                  "vector into class probabilities. The saving is dramatic: on VGG-16's "
                  "final 7 × 7 × 512 feature map, flattening into its three dense layers "
                  f"needs {head_dense:,} parameters, while global average pooling into a "
                  f"single 1000-way layer needs {gap_head:,}. Dense layers are also prone "
                  "to overfitting and need regularisation such as dropout. The catch: "
                  "GAP only works if the convolution layers are capable of producing "
                  "sufficiently abstract features, which in practice means a deeper "
                  "network or richer layers, as in network in network.", png=png)


def _inception_diagram(ax, reduce):
    cfg = A.INCEPTION_CFG[0]
    n = A.Net("x")
    x = n.add("in", "input", (192, 28, 28))
    out = A.inception_module(n, x, cfg, reduce=reduce)
    by = {l.name: l for l in n.layers}
    branches = ([("1×1", "3a 1×1")], [("3×3", "3a 3×3")], [("5×5", "3a 5×5")],
                [("max pool 3×3", "3a max pool")])
    if reduce:
        branches = ([("1×1", "3a 1×1")],
                    [("1×1 reduce", "3a 3×3 reduce"), ("3×3", "3a 3×3")],
                    [("1×1 reduce", "3a 5×5 reduce"), ("5×5", "3a 5×5")],
                    [("max pool 3×3", "3a max pool"), ("1×1 proj", "3a pool proj")])
    ax.add_patch(plt.Rectangle((1.2, 0), 5.6, 0.55, fc=C["light"], ec=C["ink"], lw=1.1))
    ax.text(4.0, 0.27, "previous layer   28 × 28 × 192", ha="center", va="center",
            fontsize=8.8)
    for bi, br in enumerate(branches):
        bx = 0.2 + bi * 2.0
        y = 1.0
        ax.annotate("", xy=(bx + 0.85, 0.98), xytext=(min(max(bx + 0.85, 1.4), 6.6), 0.57),
                    arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1))
        for label, key in br:
            l = by[key]
            col = (C["orange"] if "pool" in label else C["sky"] if "1×1" in label
                   else C["blue"])
            ax.add_patch(plt.Rectangle((bx, y), 1.7, 0.7, fc=col, alpha=0.6,
                                       ec=C["ink"], lw=1.0))
            ax.text(bx + 0.85, y + 0.42, label, ha="center", fontsize=8.2)
            ax.text(bx + 0.85, y + 0.14, f"→ {l.out[0]} ch", ha="center", fontsize=7.6,
                    color=C["red"])
            y += 0.95
        ax.annotate("", xy=(bx + 0.85, 3.05), xytext=(bx + 0.85, y - 0.25),
                    arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1))
    ax.add_patch(plt.Rectangle((1.2, 3.1), 5.6, 0.55, fc=C["green"], alpha=0.35,
                               ec=C["ink"], lw=1.1))
    ax.text(4.0, 3.37, f"concatenate → 28 × 28 × {out[0]}", ha="center", va="center",
            fontsize=8.8)
    ax.set_xlim(0, 8.2); ax.set_ylim(-0.2, 4.4); ax.axis("off")
    return n, out


def fig_inception_module(ctx: BuildContext) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    res = {}
    for ax, red, title in [(axes[0], False, "(a) naive module"),
                           (axes[1], True, "(b) with 1 × 1 bottlenecks (GoogLeNet 3a)")]:
        n, out = _inception_diagram(ax, red)
        res[red] = (n, out)
        ax.set_title(f"{title}\n{n.macs / 1e6:.1f} M multiply-accumulates · "
                     f"{n.params:,} parameters · {out[0]} output channels", fontsize=10)
    png = save_mpl(fig, ctx, "inception_module")
    (nn, on), (nr, orr) = res[False], res[True]
    return Figure("inception_module",
                  "The Inception module, computed for GoogLeNet's first module on a "
                  "28 × 28 × 192 input. It departs from the simple sequential stack of "
                  "AlexNet and VGG: 1 × 1, 3 × 3 and 5 × 5 convolutions and a pooling "
                  "path run in parallel, finding correlations across channels and over "
                  "progressively wider regions, and their outputs are concatenated. "
                  "Each module is a 'network within a network'. (a) Built naively, the "
                  f"module costs {nn.macs / 1e6:.0f} M multiply-accumulates, and because "
                  "the pooling path passes all 192 input channels straight through, the "
                  f"output grows to {on[0]} channels — so stacking such modules would "
                  "make every one more expensive than the last. (b) GoogLeNet inserts "
                  "1 × 1 bottleneck convolutions before the 3 × 3 and 5 × 5 paths and "
                  f"after the pool. That cuts the cost to {nr.macs / 1e6:.0f} M, "
                  f"{nn.macs / nr.macs:.1f} times cheaper, and holds the output to "
                  f"{orr[0]} channels — exactly the bottleneck idea of Lesson 1.3.",
                  png=png)


def fig_googlenet(ctx: BuildContext) -> Figure:
    net = A.googlenet()
    st = net.stage_totals()
    mods = [s for s in st if s.startswith("inception")]
    outs = {s: [l.out for l in net.layers if l.stage == s and l.kind == "concat"][0]
            for s in mods}
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.72, 1], hspace=0.32)

    ax = fig.add_subplot(gs[0])
    seq = ["stem"] + mods[:2] + ["pool"] + mods[2:7] + ["pool"] + mods[7:] + ["head"]
    x = 0.0
    aux_at = {"inception 4a", "inception 4d"}
    for s in seq:
        if s == "pool":
            ax.add_patch(plt.Rectangle((x, 0.25), 0.35, 0.5, fc=C["orange"], alpha=0.6,
                                       ec=C["ink"], lw=1))
            x += 0.5
            continue
        w = 1.0 if s in ("stem", "head") else 0.9
        col = C["light"] if s == "stem" else C["purple"] if s == "head" else C["blue"]
        ax.add_patch(plt.Rectangle((x, 0), w, 1, fc=col, alpha=0.55, ec=C["ink"], lw=1))
        lab = (s.replace("inception ", "") if s.startswith("inception") else
               "stem\nconv+pool" if s == "stem" else "GAP\n+ dense")
        ax.text(x + w / 2, 0.55, lab, ha="center", va="center", fontsize=8.3)
        if s in outs:
            c, h, _ = outs[s]
            ax.text(x + w / 2, 0.16, f"{h}²×{c}", ha="center", fontsize=6.8,
                    color=C["red"])
        if s in aux_at:
            ax.annotate("", xy=(x + w / 2 + 0.25, 1.75), xytext=(x + w / 2, 1.05),
                        arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.3))
            ax.add_patch(plt.Rectangle((x + w / 2 - 0.15, 1.75), 1.1, 0.48,
                                       fc=C["red"], alpha=0.18, ec=C["red"], lw=1))
            ax.text(x + w / 2 + 0.4, 1.99, "auxiliary\nclassifier", ha="center",
                    va="center", fontsize=7.5, color=C["red"])
        x += w + 0.12
    ax.set_xlim(-0.2, x); ax.set_ylim(-0.25, 2.4); ax.axis("off")
    ax.set_title("(a) GoogLeNet: a stem, nine Inception modules, global average "
                 "pooling and one dense layer — 22 layers with weights; the auxiliary "
                 "classifiers exist only during training", fontsize=10, loc="left")

    ax = fig.add_subplot(gs[1])
    labels = ["stem"] + [m.replace("inception ", "") for m in mods] + ["head"]
    vals = [st["stem"][0]] + [st[m][0] for m in mods] + [st["head"][0]]
    macs = [st["stem"][1]] + [st[m][1] for m in mods] + [st["head"][1]]
    ax.bar(labels, np.array(vals) / 1e3, color=C["blue"], alpha=0.7, edgecolor=C["ink"],
           lw=0.7, label="parameters (thousands)")
    ax2 = ax.twinx()
    ax2.plot(labels, np.array(macs) / 1e6, color=C["red"], marker="o", lw=2,
             label="multiply-accumulates (millions)")
    ax.set_ylabel("parameters (thousands)", color=C["blue"])
    ax2.set_ylabel("M multiply-accumulates", color=C["red"])
    alex = A.alexnet(grouped=True)
    ax.set_title(f"(b) per stage: {net.params / 1e6:.1f} M parameters in total, about "
                 f"{alex.params / net.params:.0f}× fewer than AlexNet's "
                 f"{alex.params / 1e6:.0f} M", fontsize=10, loc="left")
    png = save_mpl(fig, ctx, "googlenet")
    return Figure("googlenet",
                  "GoogLeNet, the Inception network that won the 2014 challenge. "
                  "(a) At 22 layers it was the deepest CNN published at the time, and it "
                  "is wide as well as deep, the width coming from the parallel paths in "
                  "each module. A global average pooling layer feeds the final dense "
                  "layer, with dropout during training. The two **auxiliary classifiers** "
                  "are extra outputs attached part-way up; during training their losses "
                  "are added to the main loss with a weight of 0.3, so the network is "
                  "trained as a superposition of networks of different depths and the "
                  "early layers receive a strong gradient. They are removed for "
                  "inference. (b) Where the cost sits, computed per stage. In total "
                  f"GoogLeNet has about {net.params / 1e6:.0f} million parameters, against "
                  f"AlexNet's {alex.params / 1e6:.0f} million, and still performed "
                  "better. (The paper itself quotes 12 times fewer; computing it from the "
                  "published layer table gives about nine.)", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    A.verify()
    return [fig_alexnet_diagram(ctx), fig_alexnet_cost(ctx), fig_vgg_stacking(ctx),
            fig_model_zoo(ctx), fig_network_in_network(ctx), fig_global_avg_pool(ctx),
            fig_inception_module(ctx), fig_googlenet(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("2.2 Architectures for image classification")
    tab1, tab2 = st.tabs(["Layer-by-layer cost", "Inception module"])

    with tab1:
        choice = st.selectbox("network", ["AlexNet", "AlexNet (2-GPU)", "VGG-16",
                                          "GoogLeNet"])
        net = {"AlexNet": A.alexnet, "AlexNet (2-GPU)": lambda: A.alexnet(True),
               "VGG-16": A.vgg16, "GoogLeNet": A.googlenet}[choice]()
        rows = [{"layer": l.name, "output": "×".join(map(str, l.out)),
                 "parameters": l.params, "MACs": l.macs}
                for l in net.layers if l.kind in ("conv", "fc", "pool", "concat")]
        st.dataframe(rows, width="stretch", height=380)
        c = st.columns(3)
        c[0].metric("parameters", f"{net.params:,}")
        c[1].metric("G MACs", f"{net.macs / 1e9:.3f}")
        c[2].metric("16-bit size", f"{net.params * 2 / 1e6:.0f} MB")

    with tab2:
        st.markdown("Change the branch widths of GoogLeNet's first module.")
        c = st.columns(6)
        vals = [c[i].number_input(lbl, 1, 512, v) for i, (lbl, v) in
                enumerate([("1×1", 64), ("3×3 reduce", 96), ("3×3", 128),
                           ("5×5 reduce", 16), ("5×5", 32), ("pool proj", 32)])]
        cfg = ("3a", 192, *[int(v) for v in vals])
        for red in (False, True):
            n = A.Net("x")
            out = A.inception_module(n, n.add("in", "input", (192, 28, 28)), cfg,
                                     reduce=red)
            st.write(f"**{'with bottlenecks' if red else 'naive'}**: "
                     f"{n.macs / 1e6:.1f} M MACs, {n.params:,} params, "
                     f"{out[0]} output channels")
