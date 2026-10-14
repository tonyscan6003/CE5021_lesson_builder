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

import re

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .._common import BuildContext, Figure, C, SERIES, save_mpl, plotly_html, sync_matrix
from .._volumes import OBLIQUE, chan_len, spatial_side, volume, kernel_cube, network_diagram
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
    ksize = lambda l: int(re.search(r"(\d+)×", l.name).group(1))

    def op_label(l):
        if l.kind == "pool":
            return "max pool\n3×3, s=2"
        stride = " s=4" if "/4" in l.name else ""
        return f"conv {ksize(l)}×{ksize(l)}{stride}\n{l.out[0]} kernels"

    maps = [dict(h=l.out[1], c=l.out[0], kind=l.kind,
                 k=ksize(l) if l.kind != "input" else None,
                 op=op_label(l) if l.kind != "input" else None)
            for l in net.layers if l.kind in ("input", "conv", "pool")]
    vec_ops = {"flatten": "flatten", "fc6": "dense", "fc7": "dense",
               "fc8 + softmax": "dense\n+ softmax"}
    vecs = [dict(n=l.out[0], op=vec_ops[l.name]) for l in net.layers
            if l.kind in ("flatten", "fc")]
    fig, ax = plt.subplots(figsize=(14, 4.6))
    # this network spans 227 px down to 6 px, beyond the range spatial_side()
    # is tuned for, so the end caps are scaled against the input size here
    right, low = network_diagram(
        ax, maps, vecs, ymid=0.0, side_fn=lambda h: 0.42 + 1.25 * np.sqrt(h / 227),
        chan_scale=0.72, gap=0.78, vec_gap=0.62, vertical=True, fontsize=10,
        vec_height=lambda n: 0.25 + 1.3 * np.sqrt(n / 9216))
    ax.set_xlim(-0.15, right + 0.2); ax.set_ylim(low - 0.1, 1.4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("AlexNet (single-GPU form): 5 convolution, 3 pooling and 3 dense layers, "
                 "with the channel axis horizontal", fontsize=11)
    png = save_mpl(fig, ctx, "alexnet_diagram")
    return Figure("alexnet_diagram",
                  "AlexNet, the first network to win the ImageNet challenge, drawn in the "
                  "volume notation of Lesson 1.3, in which the horizontal length of each box "
                  "represents the number of channels and the end cap the spatial size. Each "
                  "operation is shown in the gap between its input and output volumes: the "
                  "dashed cube inside a volume is one kernel or pooling window, and the "
                  "dotted lines connect it to the single output value that it produces. The "
                  "structure is that of LeNet-5 on a larger scale, with 5 convolution, 3 "
                  "pooling and 3 dense layers ending in a 1000-way softmax. Large 11 × 11 "
                  "kernels are used in the first layer, at stride 4 on the 227 × 227 input, "
                  "and there are 96 of them. Deeper in the network, the number of channels "
                  "increases to 384 while the spatial size is reduced by pooling, which "
                  "balances the cost of each layer, and the kernels become smaller. Every "
                  "shape is calculated with the output-size formula of Lesson 1.1. (An input "
                  "size of 224 × 224 is given in the paper, but the 55 × 55 output of the "
                  "first layer is only obtained from an 11 × 11 kernel at stride 4 with an "
                  "input of 227 × 227.)",
                  png=png)


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
                  "The memory and computation of AlexNet, layer by layer (log scales are "
                  "used). The memory requirement is given by the number of parameters: for a "
                  "convolution layer, the sum of the weights of its kernels, and for a dense "
                  "layer, inputs × neurons. The convolution layers contain tens to hundreds "
                  "of thousands of parameters each and the dense layers tens of millions, so "
                  f"{params[is_fc].sum() / params.sum():.0%} of the parameters are in the "
                  "dense layers. For computation the reverse is the case: as the kernels are "
                  "moved across the image, each weight is used in many multiplications, and "
                  f"{macs[~is_fc].sum() / macs.sum():.0%} of the computation is performed "
                  "in the convolution layers. This is significant in practice, as dense "
                  "layers are limited by memory bandwidth and do not use the full speed of a "
                  "GPU, whereas convolution layers do. These values are for the single-GPU "
                  "form shown above. In the original two-GPU AlexNet, conv2, conv4 and conv5 "
                  f"were split into halves, which reduces the network to {g.params:,} "
                  f"parameters and {g.macs / 1e9:.2f} G multiply-accumulates, the values "
                  "usually quoted.",
                  png=png)


def fig_vgg_stacking(ctx: BuildContext) -> Figure:
    net = A.vgg16()
    fig = plt.figure(figsize=(13.5, 6.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1], width_ratios=[1.35, 1],
                          hspace=0.08, wspace=0.18)

    # (a) VGG-16 in the volume notation, one volume per block output: the
    # block's stacked 3 × 3 convolutions are one labelled operation, the
    # pooling the next. Shapes and parameters computed by _arch.
    ax = fig.add_subplot(gs[0, :])
    st = net.stage_totals()
    maps = [dict(h=224, c=3, kind="input", k=None, op=None)]
    for b in range(1, 6):
        convs = [l for l in net.layers if l.stage == f"block{b}" and l.kind == "conv"]
        pool_ = [l for l in net.layers if l.stage == f"block{b}" and l.kind == "pool"][0]
        c, h = convs[-1].out[0], convs[-1].out[1]
        maps.append(dict(h=h, c=c, kind="conv", k=3,
                         op=f"{len(convs)} × conv 3×3\n{c} kernels\n"
                            f"{_fmt(st[f'block{b}'][0])} params"))
        maps.append(dict(h=pool_.out[1], c=pool_.out[0], kind="pool", k=2,
                         op="max pool\n2×2, s=2"))
    fcs = [l for l in net.layers if l.kind in ("flatten", "fc")]
    vec_ops = ["flatten"] + ["dense"] * (len(fcs) - 2) + ["dense\n+ softmax"]
    vecs = [dict(n=l.out[0], op=o) for l, o in zip(fcs, vec_ops)]
    right, low = network_diagram(
        ax, maps, vecs, ymid=0.0, side_fn=lambda h: 0.42 + 1.25 * np.sqrt(h / 224),
        chan_scale=0.55, gap=0.98, vec_gap=0.98, vertical=True, fontsize=9.5,
        vec_height=lambda n: 0.25 + 1.3 * np.sqrt(n / 25088))
    ax.set_xlim(-0.15, right + 0.2); ax.set_ylim(low - 0.1, 1.4)
    ax.set_aspect("equal"); ax.axis("off")
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
                  "(a) VGG-16 has the same overall structure as AlexNet, with convolution, "
                  "pooling and then dense layers, but every convolution is 3 × 3 and the "
                  "convolutions are stacked in blocks, giving 13 convolution layers instead "
                  "of 5, with up to 512 channels. It is drawn in the same volume notation, "
                  "with one volume for the output of each block of stacked convolutions "
                  "and one for each pooling layer. (b) The reason for stacking: consecutive "
                  "3 × 3 convolutions view as large a region of the input as one larger "
                  "kernel, with a receptive field of 5 for two layers and 7 for three, by "
                  "the formula of Lesson 1.4. (c) Fewer weights and fewer multiplications are "
                  "required: 27 per channel pair for three 3 × 3 layers compared with 49 for "
                  "one 7 × 7 layer, and as there is an activation between each layer, the "
                  "stack is also more expressive. The large dense layers of AlexNet are "
                  "retained in VGG, however, and they still contain most of its parameters; "
                  "this problem is addressed in the remainder of this lesson.",
                  png=png)


def fig_model_zoo(ctx: BuildContext) -> Figure:
    rows = []
    for name, fn, acc in ZOO:
        net = fn()
        rows.append((name, acc, net.macs / 1e9, net.params / 1e6))
    # Every label sits just outside its own bubble, in a direction chosen so
    # neighbours (GoogLeNet / ResNet-18, DenseNet / ResNet-50) do not collide.
    # The offset is the bubble's radius in points plus a margin.
    side = {"AlexNet": "right", "VGG-16": "below", "ResNet-152": "above",
            "ResNet-50": "above", "DenseNet-121": "left", "GoogLeNet": "left",
            "ResNet-18": "right", "MobileNetV2": "above"}
    place = {"right": (1, 0, "left", "center"), "left": (-1, 0, "right", "center"),
             "above": (0, 1, "center", "bottom"), "below": (0, -1, "center", "top")}
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for i, (name, acc, g, p) in enumerate(rows):
        s = p * 22 + 30
        ax.scatter(g, acc, s=s, color=SERIES[i % len(SERIES)], alpha=0.55,
                   edgecolor=C["ink"], lw=0.8)
        ux, uy, ha, va = place[side[name]]
        r = np.sqrt(s / np.pi) + 4
        ax.annotate(f"{name}\n{p:.1f}M", (g, acc), xytext=(ux * r, uy * r),
                    textcoords="offset points", ha=ha, va=va, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("operations per image (G multiply-accumulates, log scale)")
    ax.set_ylabel("ImageNet top-1 accuracy (%)")
    ax.set_xlim(0.15, 40); ax.set_ylim(50, 83)
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
                  "The model zoo: ImageNet top-1 accuracy against the computation required "
                  "per image, with bubble area proportional to the number of parameters. "
                  "AlexNet is located at the bottom left, as it requires little computation "
                  "and is not very accurate, but it has a very large number of parameters. "
                  "VGG-16 is more accurate but requires far more computation and has the "
                  "largest number of parameters, which makes it unsuitable for mobile devices "
                  "and slow to train, as its dense layers are limited by memory bandwidth. "
                  "These concerns led to the development of architectures without large "
                  "dense layers: GoogLeNet, ResNet, DenseNet and MobileNetV2 all reach or "
                  "exceed the accuracy of VGG with a fraction of its parameters. The numbers "
                  "of parameters and operations are calculated from each architecture (and "
                  "match the counts published for torchvision); the accuracies are the "
                  "single-crop results published for torchvision.",
                  png=png, html=plotly_html(fig2, height=480),
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
                  "The network-in-network concept (Lin, 2013). (a) An ordinary convolution "
                  "is a linear filter followed by an activation, so the features that can be "
                  "represented at each position are limited, and for this reason large dense "
                  "layers are required at the output of networks such as AlexNet and VGG to "
                  "separate the classes. (b) In network in network, the filter is replaced by "
                  "a small multilayer perceptron, which is still applied across the input in "
                  "the same way as a convolution. Parameters are added locally and each "
                  "feature becomes more discriminative, so the large dense output layers can "
                  "be replaced by global average pooling. The application of the same small "
                  "dense network at every position is equivalent to a stack of 1 × 1 "
                  "convolutions, and this is how the concept is implemented efficiently in "
                  "Inception.",
                  png=png)


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
                  "Global average pooling, also proposed in the network-in-network paper, as "
                  "an alternative to dense layers in the classifier. A feature map of any "
                  "spatial size is reduced to a single value, its mean, so C feature maps "
                  "become a vector of length C regardless of the input image size. This "
                  "vector is converted into class probabilities by a dense layer with "
                  "softmax. The saving is large: for the final 7 × 7 × 512 feature map of "
                  "VGG-16, flattening into its three dense layers requires "
                  f"{head_dense:,} parameters, whereas global average pooling into a single "
                  f"1000-way layer requires {gap_head:,}. Dense layers are also prone to "
                  "overfitting, and regularisation such as dropout is required. However, "
                  "global average pooling is only effective if sufficiently abstract "
                  "features can be produced by the convolution layers, which in practice "
                  "requires a deeper network or more powerful layers, as in network in "
                  "network.",
                  png=png)


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
        for j, (label, key) in enumerate(br):
            l = by[key]
            if j:       # the bottleneck (or pool) feeds the next stage of the branch
                ax.annotate("", xy=(bx + 0.85, y - 0.01), xytext=(bx + 0.85, y - 0.25),
                            arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1))
            col = (C["orange"] if "pool" in label else C["sky"] if "1×1" in label
                   else C["blue"])
            ax.add_patch(plt.Rectangle((bx, y), 1.7, 0.7, fc=col, alpha=0.6,
                                       ec=C["ink"], lw=1.0))
            ax.text(bx + 0.85, y + 0.42, label, ha="center", fontsize=9)
            ax.text(bx + 0.85, y + 0.13, f"→ {l.out[0]} ch", ha="center", fontsize=8.5,
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
                  "The Inception module, calculated for the first module of GoogLeNet on a "
                  "28 × 28 × 192 input. It departs from the sequential structure of AlexNet "
                  "and VGG: 1 × 1, 3 × 3 and 5 × 5 convolutions and a pooling path are "
                  "applied in parallel, so that correlations across channels and over "
                  "progressively wider regions are found, and their outputs are "
                  "concatenated. Each module is a 'network within a network'. (a) In the "
                  f"naive form the module requires {nn.macs / 1e6:.0f} M multiply-"
                  "accumulates, and as all 192 input channels are passed through by the "
                  f"pooling path, the output increases to {on[0]} channels, so each module "
                  "in a stack would be more expensive than the previous one. (b) In "
                  "GoogLeNet, 1 × 1 bottleneck convolutions are inserted before the 3 × 3 "
                  "and 5 × 5 paths and after the pooling. The cost is reduced to "
                  f"{nr.macs / 1e6:.0f} M, {nn.macs / nr.macs:.1f} times less, and the "
                  f"output is limited to {orr[0]} channels, using the bottleneck concept of "
                  "Lesson 1.3.",
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
                  "GoogLeNet, the Inception network that won the 2014 challenge. (a) With 22 "
                  "layers it was the deepest CNN published at the time, and it is wide as "
                  "well as deep, with the width provided by the parallel paths in each "
                  "module. A global average pooling layer is connected to the final dense "
                  "layer, with dropout applied during training. The two **auxiliary "
                  "classifiers** are additional outputs attached part of the way through the "
                  "network; during training their losses are added to the main loss with a "
                  "weight of 0.3, so the network is trained as a superposition of networks "
                  "of different depths and a strong gradient is received by the early "
                  "layers. The auxiliary classifiers are removed for inference. (b) The "
                  "distribution of cost, calculated per stage. In total GoogLeNet has "
                  f"approximately {net.params / 1e6:.0f} million parameters, compared with "
                  f"{alex.params / 1e6:.0f} million for AlexNet, and performed better. (A "
                  "factor of 12 is quoted in the paper; when calculated from the published "
                  "layer table, the factor is approximately nine.)",
                  png=png)


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
