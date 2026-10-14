"""Lesson 2.3 — Residual networks.

Figures
    degradation        a deeper plain network trains worse; a residual one trains
                       best (measured on digits)
    residual_block     F(x) + x, and the basic vs bottleneck blocks with their costs
    unravelled_paths   2^n paths through n blocks; measured branch gain shows the
                       gradient travels along the short ones
    resnet_family      ResNet-18 ... 152: stage configuration, parameters, operations
    batch_norm         the normalise / scale / shift steps, and activations exploding
                       through 50 layers without it
    batch_norm_running running statistics during training, used at inference
    squeeze_excite     the SE block on a feature map, and its parameter overhead
    densenet           dense connectivity and channel growth inside a dense block

Training experiments use engines/_tinynet.py on the sklearn digits; architecture
counts come from engines/_arch.py.

Source: reference/legacy_md/lesson_2_3_Residual_Networks + lesson_2_3_script.
"""
from __future__ import annotations

from math import comb

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout, use_source, sync_matrix)
from .._volumes import OBLIQUE, volume
from .. import _arch as A
from .. import _tinynet as T

FIGURES = ["degradation", "residual_block", "unravelled_paths", "resnet_family",
           "batch_norm", "batch_norm_running", "squeeze_excite", "densenet"]

DEG_RUNS = [(3, False, "3 layers, plain"), (40, False, "40 layers, plain"),
            (40, True, "40 layers, residual")]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

_DEG_CACHE: dict = {}


def degradation_runs(epochs=30):
    if "runs" not in _DEG_CACHE:
        Xtr, ytr, Xte, yte = T.digits()
        runs = {}
        for depth, res, name in DEG_RUNS:
            net = T.MLP(64, 64, depth, 10, act="relu", residual=res, seed=0)
            h = T.train(net, Xtr, ytr, Xte, yte, epochs=epochs, lr=0.02)
            runs[name] = (h, net)
        _DEG_CACHE["runs"] = (runs, (Xtr, ytr))
    return _DEG_CACHE["runs"]


def branch_gains(net, X, y):
    """For each residual block, ||gradient through the branch|| / ||gradient arriving||.

    The identity shortcut passes the arriving gradient through unchanged (gain 1),
    so this ratio is what each block multiplies a path's gradient by when the
    path goes *through* the block rather than around it.
    """
    logits, cache = net.forward(X)
    L = len(net.W)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    g = p
    g[np.arange(len(y)), y] -= 1
    g /= len(y)
    gains = []
    for i in range(L - 1, -1, -1):
        h, z, a = cache[i]
        dz = g if i == L - 1 else g * net.df(z, a)
        branch = dz @ net.W[i].T
        if net.residual and 0 < i < L - 1:
            gains.append(np.linalg.norm(branch) / np.linalg.norm(g))
            g = branch + g
        else:
            g = branch
    return np.array(gains[::-1])


def bn_forward(x, gamma, beta, eps=1e-5):
    mu, var = x.mean(0), x.var(0)
    xh = (x - mu) / np.sqrt(var + eps)
    return xh, gamma * xh + beta, mu, var


def deep_activation_std(depth=50, width=64, scale=1.3, bn=False, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(256, width))
    out = []
    for _ in range(depth):
        W = rng.normal(0, scale * np.sqrt(2 / width), (width, width))
        z = h @ W
        if bn:
            z = (z - z.mean(0)) / np.sqrt(z.var(0) + 1e-5)
        h = np.maximum(z, 0)
        out.append(h.std())
    return np.array(out)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_degradation(ctx: BuildContext) -> Figure:
    runs, _ = degradation_runs()
    cols = [C["blue"], C["red"], C["green"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for (name, (h, net)), col in zip(runs.items(), cols):
        axes[0].plot(h["epoch"], h["train_loss"], color=col, lw=2.2, label=name)
        axes[1].plot(h["epoch"], h["test_acc"], color=col, lw=2.2,
                     label=f"{name}: {h['test_acc'][-1]:.1%}")
    axes[0].axhline(np.log(10), color=C["grey"], ls=":", lw=1.2)
    axes[0].text(30, np.log(10) * 1.18, "chance: ln 10", ha="right", fontsize=8.5,
                 color=C["grey"])
    axes[0].set_yscale("log"); axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss")
    axes[0].set_title("(a) training loss: the deep plain network does not train",
                      fontsize=10.5)
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("test accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("(b) test accuracy", fontsize=10.5)
    axes[1].legend(fontsize=9, loc="lower right")
    png = save_mpl(fig, ctx, "degradation")
    plain, res = runs["40 layers, plain"][0], runs["40 layers, residual"][0]
    return Figure("degradation",
                  "The degradation problem, measured. Three fully connected networks of the "
                  "same width are trained in the same way on handwritten digits, and differ "
                  "only in depth and connectivity. The 40-layer plain network performs worse "
                  "than the 3-layer network, not only on the test set but on the "
                  "**training** set, where its loss remains at the chance level of ln 10 and "
                  f"its test accuracy at {plain['test_acc'][-1]:.0%}. Overfitting is "
                  "therefore not the explanation; the deeper network cannot be optimised. "
                  "This was found by the ResNet authors when VGG-19 was extended with "
                  "additional layers. When identity shortcuts are added around each layer, "
                  "the same 40-layer network is trained easily and reaches "
                  f"{res['test_acc'][-1]:.1%}, better than the shallow network. The "
                  "networks used here are small and fully connected rather than "
                  "convolutional, but the same effect is observed.",
                  png=png)


def fig_residual_block(ctx: BuildContext) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.1]))

    def box(ax, y, txt, col, w=1.7, x0=0.15):
        ax.add_patch(plt.Rectangle((x0, y), w, 0.55, fc=col, alpha=0.55, ec=C["ink"], lw=1))
        ax.text(x0 + w / 2, y + 0.275, txt, ha="center", va="center", fontsize=8.6)

    def arrow(ax, x, y0, y1):
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.2))

    TOP = 5.9

    def block(ax, steps, post, cx, w, xs):
        """Draw a residual block top to bottom: x, main path, +, post, out.

        Returns the y of the addition and of the bottom of the output arrow.
        """
        y = TOP
        ax.text(cx, y, "x", ha="center", va="center", fontsize=12)
        y_branch = y - 0.25
        arrow(ax, cx, y - 0.15, y - 0.5)
        y -= 0.52
        for txt, col in steps:
            box(ax, y - 0.55, txt, col, w=w, x0=cx - w / 2)
            y -= 0.55
            arrow(ax, cx, y, y - 0.3)
            y -= 0.32
        yc = y - 0.2                                   # the addition
        ax.scatter([cx], [yc], s=480, facecolors="white", edgecolors=C["ink"], lw=1.4,
                   zorder=3)
        ax.text(cx, yc, "+", ha="center", va="center", fontsize=15, zorder=4)
        # identity shortcut: from the input, round the main path, into the +
        ax.plot([cx, xs, xs, cx + 0.45], [y_branch, y_branch, yc, yc], color=C["red"], lw=2)
        ax.annotate("", xy=(cx + 0.19, yc), xytext=(cx + 0.5, yc),
                    arrowprops=dict(arrowstyle="->", color=C["red"], lw=2))
        y = yc - 0.2
        for txt, col in post:
            arrow(ax, cx, y, y - 0.3)
            y -= 0.32
            box(ax, y - 0.55, txt, col, w=w, x0=cx - w / 2)
            y -= 0.55
        # the block's output, on to the next block
        arrow(ax, cx, y, y - 0.42)
        ax.text(cx, y - 0.55, "output", ha="center", va="top", fontsize=9, color=C["grey"])
        return yc, y - 0.55

    # (a) the generic residual block
    ax = axes[0]
    yc, _ = block(ax, [("weight layer", C["blue"]), ("ReLU", C["sky"]),
                       ("weight layer", C["blue"])], [("ReLU", C["sky"])],
                  cx=1.0, w=1.7, xs=2.25)
    ax.text(0.72, yc + 0.08, "F(x) + x", ha="right", va="bottom", fontsize=10,
            weight="bold")
    ax.text(2.35, 3.6, "identity\nshortcut\n(x)", fontsize=9, color=C["red"])
    ax.text(-0.15, 3.95, "F(x)", fontsize=11, color=C["blue"], ha="center")
    ax.set_xlim(-0.6, 3.1); ax.set_ylim(-1.15, 6.2); ax.axis("off")
    ax.set_title("(a) residual learning\nH(x) = F(x) + x", fontsize=10.5)

    # (b) basic and (c) bottleneck, with computed costs on a 56x56 map
    for ax, kind in [(axes[1], "basic"), (axes[2], "bottleneck")]:
        n = A.Net("x")
        if kind == "basic":
            x = n.add("in", "input", (64, 56, 56))
            x = A.conv(n, x, 64, 3, 1, 1, bn=True, name="3×3, 64")
            A.conv(n, x, 64, 3, 1, 1, bn=True, name="3×3, 64")
            steps = [("3×3 conv, 64", C["blue"]), ("BN, ReLU", C["sky"]),
                     ("3×3 conv, 64", C["blue"]), ("BN", C["sky"])]
            title = "(b) basic block (ResNet-18/34)\n64 channels in and out"
        else:
            x = n.add("in", "input", (256, 56, 56))
            x = A.conv(n, x, 64, 1, bn=True, name="1×1, 64")
            x = A.conv(n, x, 64, 3, 1, 1, bn=True, name="3×3, 64")
            A.conv(n, x, 256, 1, bn=True, name="1×1, 256")
            steps = [("1×1 conv, 64  (reduce)", C["sky"]), ("3×3 conv, 64", C["blue"]),
                     ("1×1 conv, 256 (expand)", C["sky"])]
            title = "(c) bottleneck block (ResNet-50/101/152)\n256 channels in and out"
        _, bottom = block(ax, steps, [("ReLU", C["sky"])], cx=1.3, w=2.3, xs=2.75)
        ax.text(1.3, bottom - 0.35,
                f"{n.params:,} parameters\n{n.macs / 1e6:.0f} M multiply-"
                f"accumulates on 56 × 56", ha="center", va="top", fontsize=9,
                color=C["red"])
        ax.set_xlim(-0.1, 3.2); ax.set_ylim(-1.15, 6.2); ax.axis("off")
        ax.set_title(title, fontsize=10.5)
    png = save_mpl(fig, ctx, "residual_block")
    return Figure("residual_block",
                  "(a) The residual block, drawn with the input x at the top: a main path of "
                  "weight layers and activations, and a shortcut by which the input x is "
                  "added to the output of the main path before the final activation, whose "
                  "output is passed to the next block. Without the shortcut, a desired mapping H(x) "
                  "would have to be learned directly by the block. With the shortcut, only "
                  "the **residual** F(x) = H(x) − x must be learned by the main path, and "
                  "H(x) is recovered as F(x) + x. Residual mappings have been found to be "
                  "much easier to train; in particular, if a block should have no effect, "
                  "F(x) need only be driven to zero. (b) Two 3 × 3 convolutions per block "
                  "are used in ResNet-18 and -34. (c) A bottleneck is used in the deeper "
                  "ResNets: the number of channels is reduced from 256 to 64 by a 1 × 1 "
                  "convolution, the 3 × 3 convolution operates on 64 channels, and 256 "
                  "channels are restored by a further 1 × 1 convolution. Four times as many "
                  "channels are handled by the bottleneck block as by the basic block for a "
                  "similar cost, which is why it is used in the deeper networks.",
                  png=png)


def fig_unravelled_paths(ctx: BuildContext) -> Figure:
    runs, (Xtr, ytr) = degradation_runs()
    net = runs["40 layers, residual"][1]
    gains = branch_gains(net, Xtr[:512], ytr[:512])
    n, rho = len(gains), float(gains.mean())
    k = np.arange(n + 1)
    count = np.array([comb(n, int(i)) for i in k], float)
    contrib = count * rho ** k
    share10 = contrib[k <= 10].sum() / contrib.sum()

    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1, 1], wspace=0.3)

    # (a) three blocks unravelled into 8 paths
    ax = fig.add_subplot(gs[0])
    ys = np.linspace(3.5, 0.0, 8)
    for pi, y in enumerate(ys):
        bits = [(pi >> b) & 1 for b in range(3)]
        x = 0.0
        ax.scatter([x], [y], s=26, color=C["ink"], zorder=3)
        for b, through in enumerate(bits):
            nx = x + 1.0
            if through:
                ax.add_patch(plt.Rectangle((x + 0.3, y - 0.14), 0.4, 0.28, fc=C["blue"],
                                           alpha=0.7, ec=C["ink"], lw=0.8))
                ax.text(x + 0.5, y, f"f{b + 1}", ha="center", va="center", fontsize=7,
                        color="white")
            ax.plot([x, nx], [y, y], color=C["blue"] if through else C["red"], lw=1.4,
                    zorder=1)
            x = nx
        ax.scatter([x], [y], s=26, color=C["ink"], zorder=3)
        ax.text(x + 0.12, y, f"length {sum(bits)}", va="center", fontsize=8)
    ax.set_xlim(-0.2, 4.3); ax.set_ylim(-0.4, 4.0); ax.axis("off")
    ax.set_title("(a) 3 residual blocks = 2³ = 8 paths\nblue: through a block · "
                 "red: around it", fontsize=10)

    # (b) how many paths of each length, and (c) how much gradient they carry
    ax = fig.add_subplot(gs[1])
    ax.bar(k, count / count.sum(), color=C["grey"], width=0.85)
    ax.set_xlabel("path length (blocks passed through)")
    ax.set_ylabel("fraction of all paths")
    sup = str(n).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    ax.set_title(f"(b) {n} blocks: 2{sup} paths, most of\nmedium length (peak at {k[count.argmax()]})",
                 fontsize=10)
    ax = fig.add_subplot(gs[2])
    ax.bar(k, contrib / contrib.sum(), color=C["green"], width=0.85)
    ax.set_xlim(-1, n + 1)
    ax.set_xlabel("path length (blocks passed through)")
    ax.set_ylabel("share of total gradient")
    ax.set_title(f"(c) but {share10:.2%} of the gradient\ntravels along paths of ≤ 10",
                 fontsize=10)
    png = save_mpl(fig, ctx, "unravelled_paths")
    return Figure("unravelled_paths",
                  "The explanation of Veit and colleagues for the training of residual "
                  "networks. (a) Each block offers two routes, through its layers or around "
                  "them by the shortcut, so n blocks can be unravelled into 2ⁿ paths of "
                  "different lengths, and a ResNet behaves as an **ensemble** of networks. "
                  "For this reason a block can be removed from a trained ResNet with little "
                  f"effect. (b) For the 40-layer residual network trained above, with {n} "
                  "residual blocks, most paths are of medium length. (c) However, each time "
                  "a path passes *through* a block, its gradient is multiplied by the gain of "
                  f"that block, which is {rho:.3f} on average in this network, while the "
                  "gradient is passed unchanged by the shortcut. When each path length is "
                  "weighted by this measured gain, it can be seen that the gradient is "
                  "carried almost entirely by **short** paths, which allows deep residual "
                  "networks to be trained. (In panel (c) the gains of the blocks are treated "
                  "as independent, which is the approximation made in the analysis of Veit.) "
                  "The exact reason for the effectiveness of shortcuts is still debated, and "
                  "vanishing gradients were considered an unlikely cause by the original "
                  "ResNet authors, but their effectiveness is well established.",
                  png=png)


def fig_resnet_family(ctx: BuildContext) -> Figure:
    depths = [18, 34, 50, 101, 152]
    nets = {d: A.resnet(d) for d in depths}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3),
                             gridspec_kw=dict(width_ratios=[1.45, 1]))
    ax = axes[0]
    ax.axis("off")
    stages = ["conv2_x\n56×56", "conv3_x\n28×28", "conv4_x\n14×14", "conv5_x\n7×7"]
    cells = []
    for d in depths:
        kind, blocks = A.RESNET_CFG[d]
        net = nets[d]
        cells.append([f"ResNet-{d}", kind] + [f"×{b}" for b in blocks] +
                     [f"{net.params / 1e6:.1f}M", f"{net.macs / 1e9:.1f}"])
    tbl = ax.table(cellText=cells,
                   colLabels=["", "block"] + stages + ["params", "G MACs"],
                   loc="center", cellLoc="center",
                   colWidths=[0.15, 0.15, 0.11, 0.11, 0.11, 0.11, 0.12, 0.11])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.2); tbl.scale(1, 2.1)
    for (r_, c_), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["light"])
        if r_ == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("every ResNet: 7 × 7 conv + max pool, then four stages of repeated "
                 "blocks, then\nglobal average pooling and one dense layer — "
                 "parameters computed, matching torchvision exactly", fontsize=9.8)

    ax = axes[1]
    p = [nets[d].params / 1e6 for d in depths]
    m = [nets[d].macs / 1e9 for d in depths]
    vgg = A.vgg16()
    ax.plot(depths, p, color=C["blue"], marker="o", lw=2, label="ResNet parameters (M)")
    ax.plot(depths, m, color=C["red"], marker="s", lw=2, label="ResNet G MACs")
    ax.axhline(vgg.params / 1e6, color=C["blue"], ls=":", lw=1.3)
    ax.axhline(vgg.macs / 1e9, color=C["red"], ls=":", lw=1.3)
    ax.text(152, vgg.params / 1e6 * 0.92, f"VGG-16: {vgg.params / 1e6:.0f} M params",
            ha="right", va="top", fontsize=8.5, color=C["blue"])
    ax.text(152, vgg.macs / 1e9 * 1.15, f"VGG-16: {vgg.macs / 1e9:.1f} G MACs",
            ha="right", fontsize=8.5, color=C["red"])
    ax.set_yscale("log"); ax.set_xlabel("depth (layers)")
    ax.set_xticks(depths)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("even ResNet-152 is smaller and cheaper than VGG-16", fontsize=10)
    png = save_mpl(fig, ctx, "resnet_family")
    return Figure("resnet_family",
                  "The ResNet family. Every member has the same structure: an initial 7 × 7 "
                  "convolution and max pooling, four stages of identical residual blocks on "
                  "56, 28, 14 and 7 pixel feature maps, then global average pooling and a "
                  "single dense layer. The members differ only in the number of blocks in "
                  "each stage and the type of block. With this regular, repeating structure, "
                  "a 152-layer network can be specified in a practical way. The parameter "
                  "counts are calculated from this table and match the figures published for "
                  "torchvision exactly. As ResNet ends with global average pooling instead "
                  "of large dense layers, even ResNet-152, which is more than ten times as "
                  "deep as VGG-16, has fewer than half as many parameters and requires less "
                  "computation.",
                  png=png)


def fig_batch_norm(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(3)
    x = rng.normal(4.0, 2.5, 6).round(1)
    gamma, beta = 0.8, 0.5
    xh, y, mu, var = bn_forward(x[:, None], np.array([gamma]), np.array([beta]))
    fig = plt.figure(figsize=(13, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.25)

    ax = fig.add_subplot(gs[0])
    idx = np.arange(len(x))
    w = 0.27
    ax.bar(idx - w, x, w, color=C["grey"], label="input x")
    ax.bar(idx, xh[:, 0], w, color=C["blue"], label="normalised x̂ (mean 0, var 1)")
    ax.bar(idx + w, y[:, 0], w, color=C["green"], label=f"output γx̂ + β (γ={gamma}, β={beta})")
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.set_xticks(idx); ax.set_xticklabels([f"x{i + 1}={v:g}" for i, v in enumerate(x)],
                                           fontsize=8.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title(f"(a) one mini-batch of one channel: μ_B = {mu[0]:.2f}, "
                 f"σ²_B = {var[0]:.2f}", fontsize=10)

    ax = fig.add_subplot(gs[1])
    s0 = deep_activation_std(bn=False)
    s1 = deep_activation_std(bn=True)
    ax.semilogy(np.arange(1, 51), s0, color=C["red"], lw=2.2, label="without batch norm")
    ax.semilogy(np.arange(1, 51), s1, color=C["green"], lw=2.2, label="with batch norm")
    ax.set_xlabel("layer"); ax.set_ylabel("standard deviation of activations")
    ax.legend(fontsize=9)
    ax.set_title(f"(b) 50 layers, weights 30% too large: {s0[-1]:.1e} vs "
                 f"{s1[-1]:.2f} by the last layer", fontsize=10)
    png = save_mpl(fig, ctx, "batch_norm")
    return Figure("batch_norm",
                  "Batch normalisation, which is inserted after each convolution in ResNet. "
                  "(a) For each channel, the mean and variance are calculated over the "
                  "mini-batch and used to normalise the values to zero mean and unit "
                  "variance; the result is then scaled and shifted by two learned "
                  "parameters, γ and β, so that the normalisation can be reversed if this is "
                  "required during training. Only γ and β are learned, so very few "
                  "parameters are added. (b) The benefit of batch normalisation. Without it, "
                  "a small error in the scale of the weights accumulates through the layers: "
                  "here, with weights 30% larger than required, the activations increase "
                  "geometrically, by a factor of more than 100,000 over 50 layers. With "
                  "batch normalisation the scale is reset at every layer and remains "
                  "constant. This is the effect identified in later work, in which batch "
                  "normalisation prevents the internal states of deep networks from "
                  "exploding, rather than the 'internal covariate shift' of the original "
                  "paper, and it allows larger learning rates and faster convergence.",
                  png=png)


def fig_batch_norm_running(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(11)
    true_mu, true_var, d = 3.0, 1.5, 0.95
    mu_g, var_g = 0.0, 1.0
    rec_mu, rec_var, bmu, bvar = [], [], [], []
    for _ in range(120):
        b = rng.normal(true_mu, np.sqrt(true_var), 16)
        bmu.append(b.mean()); bvar.append(b.var())
        mu_g = d * mu_g + (1 - d) * b.mean()
        var_g = d * var_g + (1 - d) * b.var()
        rec_mu.append(mu_g); rec_var.append(var_g)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    for ax, raw, run, true, name in [(axes[0], bmu, rec_mu, true_mu, "mean"),
                                     (axes[1], bvar, rec_var, true_var, "variance")]:
        ax.plot(raw, ".", color=C["grey"], ms=4, label=f"each mini-batch's {name}")
        ax.plot(run, color=C["blue"], lw=2.2, label=f"running {name} (decay {d})")
        ax.axhline(true, color=C["green"], ls="--", lw=1.4, label=f"population {name}")
        ax.set_xlabel("training step"); ax.legend(fontsize=8.5)
        ax.set_title(f"running {name}: what inference will use", fontsize=10.5)
    png = save_mpl(fig, ctx, "batch_norm_running")
    return Figure("batch_norm_running",
                  "Batch normalisation is applied differently in training and at inference. "
                  "During training, each mini-batch is normalised with its own statistics, "
                  "which are noisy (grey). At the same time a running mean and variance are "
                  "maintained, updated after every step as μ ← d·μ + (1 − d)·μ_B, here with "
                  f"a decay d = {d}; these converge to the statistics of the whole training "
                  "population (blue). At inference there may be only one image, so there is "
                  "no batch from which statistics can be calculated; the stored running mean "
                  "and variance are used instead, and the layer becomes a fixed affine "
                  "transform.",
                  png=png)


def _se_diagram(ax, z, s, out_means, vmax):
    """Panel (a) of the SE figure: the block in the volume notation.

    Channels run along x, so each of the C slices of the input volume lines up
    with one cell of the squeezed vector and one cell of the weight vector, and
    the multiplication is visibly channel by channel.
    """
    from matplotlib import cm
    c = len(z)
    dl, side, ymid = 0.24, 1.7, 0.0
    vir = lambda v: cm.viridis(min(v / vmax, 1.0))

    def sliced(x0, values, ymid, side, label):
        for i, v in enumerate(values):
            volume(ax, x0 + i * dl, ymid, dl, side, vir(v), lw=0.8, alpha=0.95)
        ax.text(x0 + c * dl / 2 + OBLIQUE[0] * side / 2, ymid - side / 2 - 0.15, label,
                ha="center", va="top", fontsize=9.5)
        return x0 + c * dl + OBLIQUE[0] * side

    def cells(x0, values, colour, label, fmt=None):
        for i, v in enumerate(values):
            volume(ax, x0 + i * dl, ymid, dl, 0.32, colour(v), lw=0.8, alpha=0.95)
            if fmt:
                ax.text(x0 + (i + 0.5) * dl, ymid - 0.24, fmt(v), ha="center",
                        va="top", fontsize=8, rotation=90, color=C["red"])
        ax.text(x0 + c * dl / 2, ymid - (0.85 if fmt else 0.3), label, ha="center",
                va="top", fontsize=9.5)
        return x0 + c * dl + OBLIQUE[0] * 0.32

    def step(x0, x1, text):
        ax.annotate("", xy=(x1, ymid), xytext=(x0, ymid),
                    arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.4))
        ax.text((x0 + x1) / 2, ymid + 0.12, text, ha="center", va="bottom", fontsize=9.5)

    xu = 0.0
    ru = sliced(xu, z, ymid, side, f"input U: H × W × {c}")
    xz = ru + 2.5
    step(ru + 0.1, xz - 0.1, "squeeze:\nglobal avg pool")
    rz = cells(xz, z, vir, f"1 × 1 × {c}")
    xs = rz + 2.9
    step(rz + 0.1, xs - 0.1, "excite: FC → ReLU\n→ FC → sigmoid")
    rs = cells(xs, s, lambda v: cm.Reds(0.15 + 0.8 * v), f"weights s, 1 × 1 × {c}",
               fmt=lambda v: f"{v:.2f}")
    xm = rs + 0.75                                    # the channel-wise multiply
    ax.scatter([xm], [ymid], s=520, facecolors="white", edgecolors=C["ink"], lw=1.4,
               zorder=5)
    ax.text(xm, ymid, "×", ha="center", va="center", fontsize=15, zorder=6)
    ax.annotate("", xy=(xm - 0.2, ymid), xytext=(rs + 0.05, ymid),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.4))
    # U also feeds the multiply, round underneath
    yb = ymid - side / 2 - 0.75
    ax.plot([xu + c * dl / 2, xu + c * dl / 2, xm, xm], [ymid - side / 2 - 0.55, yb, yb,
            ymid - 0.55], color=C["red"], lw=1.8)
    ax.annotate("", xy=(xm, ymid - 0.2), xytext=(xm, ymid - 0.6),
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.8))
    ax.text((xu + xm) / 2, yb - 0.08, "the input U, channel by channel", ha="center",
            va="top", fontsize=9, color=C["red"])
    xo = xm + 1.1
    ax.annotate("", xy=(xo - 0.08, ymid), xytext=(xm + 0.2, ymid),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.4))
    ro = sliced(xo, out_means, ymid, side, "output: each channel × its weight")
    ax.set_xlim(-0.2, ro + 0.9); ax.set_ylim(yb - 0.55, ymid + side / 2 + 0.75)
    ax.set_aspect("equal"); ax.axis("off")


def fig_squeeze_excite(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(2)
    c, r, n = 8, 4, 6
    maps = np.abs(rng.normal(size=(c, n, n))) * rng.uniform(0.3, 1.6, c)[:, None, None]
    z = maps.mean((1, 2))
    W1 = rng.normal(0, 0.9, (c, c // r)); W2 = rng.normal(0, 0.9, (c // r, c))
    s = 1 / (1 + np.exp(-(np.maximum(z @ W1, 0) @ W2)))
    out = maps * s[:, None, None]
    se = A.resnet(50, se_ratio=16)
    base = A.resnet(50)

    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(3, c + 3, height_ratios=[1.5, 1, 1], wspace=0.15, hspace=0.5)
    vmax = max(maps.max(), out.max())
    ax = fig.add_subplot(gs[0, :])
    _se_diagram(ax, z, s, out.mean((1, 2)), z.max())
    ax.set_title("(a) the squeeze-and-excite block: one weight per channel, multiplied "
                 "back onto that channel", fontsize=10.5, loc="left")
    for i in range(c):
        ax = fig.add_subplot(gs[1, i])
        if i == 0:
            ax.set_title("ch 1", fontsize=8)
            ax.text(-0.12, 1.42, "(b) the block applied to an 8-channel feature map",
                    transform=ax.transAxes, fontsize=10.5, ha="left", va="bottom")
        ax.imshow(maps[i], cmap="viridis", vmin=0, vmax=vmax); ax.axis("off")
        if i:
            ax.set_title(f"ch {i + 1}", fontsize=8)
        if i == 0:
            ax.text(-0.12, 0.5, "input", transform=ax.transAxes, ha="right",
                    va="center", fontsize=9)
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(out[i], cmap="viridis", vmin=0, vmax=vmax); ax.axis("off")
        ax.set_title(f"× {s[i]:.2f}", fontsize=8, color=C["red"])
        if i == 0:
            ax.text(-0.12, 0.5, "rescaled", transform=ax.transAxes, ha="right",
                    va="center", fontsize=9)
    ax = fig.add_subplot(gs[1:, c + 1:])
    ax.barh(np.arange(c), s, color=C["red"], alpha=0.7)
    ax.set_yticks(np.arange(c)); ax.set_yticklabels([f"ch {i + 1}" for i in range(c)],
                                                    fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("channel weight (sigmoid)")
    ax.set_title(f"squeeze → FC {c}→{c // r} → ReLU\n→ FC {c // r}→{c} → sigmoid",
                 fontsize=9.5)
    png = save_mpl(fig, ctx, "squeeze_excite")
    extra = se.params - base.params
    return Figure("squeeze_excite",
                  "(a) The squeeze-and-excite block, drawn in the volume notation, and (b) the "
                  "block applied to an 8-channel feature map. In the "
                  "**squeeze** step, each channel is reduced to one value by global average "
                  "pooling. In the **excite** step, these values are converted by two small "
                  "dense layers, the second with a sigmoid, into a weight between 0 and 1 "
                  "for each channel, and every channel of the input is multiplied by its "
                  "weight; as the channels are drawn along the horizontal axis in (a), each "
                  "slice of the input lines up with its weight. The **dependencies between channels** are therefore modelled, "
                  "with some channels emphasised and others suppressed according to the "
                  "content of the whole feature map. (Random weights are used here to show "
                  "the mechanism; in a trained network the weights are learned.) The block "
                  "is lightweight and can be inserted into a ResNet block, between the last "
                  "convolution and the shortcut addition, or into Inception. When it is "
                  "added to every block of ResNet-50 with a reduction ratio of 16, "
                  f"{extra / 1e6:.2f} M additional parameters ({extra / base.params:.0%}) "
                  "and almost no additional computation are required, and a reduction of "
                  "approximately one tenth in ImageNet error is reported in the paper.",
                  png=png)


def fig_densenet(ctx: BuildContext) -> Figure:
    net = A.densenet121()
    rn = A.resnet(50)
    k0, k, L = 64, 32, 6
    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.22)

    ax = fig.add_subplot(gs[0])
    xs = np.arange(L + 1) * 1.25
    for i, x in enumerate(xs):
        ch = k0 + i * k
        hgt = 0.6 + 1.9 * ch / (k0 + L * k)
        ax.add_patch(plt.Rectangle((x, -hgt / 2), 0.5, hgt, fc=C["light"] if i == 0
                                   else SERIES[(i - 1) % len(SERIES)], alpha=0.75,
                                   ec=C["ink"], lw=1))
        ax.text(x + 0.25, -hgt / 2 - 0.12, f"{ch}", ha="center", va="top", fontsize=8.5,
                color=C["red"])
        if i > 0:
            ax.text(x + 0.25, hgt / 2 + 0.08, f"H{i}", ha="center", fontsize=9)
    for i in range(L + 1):
        for j in range(i + 1, L + 1):
            xa, xb = xs[i] + 0.5, xs[j]
            h = 0.25 + 0.17 * (j - i)
            ax.annotate("", xy=(xb, 1.35 + 0.02 * j), xytext=(xa, 1.35 + 0.02 * i),
                        arrowprops=dict(arrowstyle="->", color=SERIES[i % len(SERIES)],
                                        lw=0.9, alpha=0.7,
                                        connectionstyle=f"arc3,rad=-{0.15 + 0.05 * (j - i)}"))
    ax.text(xs[-1] / 2, -1.85, f"channels after layer l:  k₀ + l·k   (k₀ = {k0}, "
                               f"growth rate k = {k})", ha="center", fontsize=9.5)
    ax.set_xlim(-0.3, xs[-1] + 0.8); ax.set_ylim(-2.1, 3.2); ax.axis("off")
    ax.set_title("(a) a dense block: every layer receives every earlier layer's "
                 "output,\nconcatenated — so the channel count grows by k per layer",
                 fontsize=10)

    ax = fig.add_subplot(gs[1])
    names = ["ResNet-50", "DenseNet-121"]
    vals = [rn.params / 1e6, net.params / 1e6]
    acc = [76.1, 74.4]
    bars = ax.bar(names, vals, color=[C["blue"], C["green"]], edgecolor=C["ink"], lw=0.8)
    for b, v, a in zip(bars, vals, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f} M params\n"
                f"top-1 {a}%", ha="center", fontsize=9.5)
    ax.set_ylabel("parameters (millions)"); ax.set_ylim(0, 33)
    ax.set_title(f"(b) {rn.params / net.params:.1f}× fewer parameters for similar "
                 "accuracy", fontsize=10)
    png = save_mpl(fig, ctx, "densenet")
    return Figure("densenet",
                  "In DenseNet the ResNet concept is extended with much greater "
                  "connectivity. (a) Inside a dense block, the outputs of **all** earlier "
                  "layers are received by every composite function (batch normalisation, "
                  "ReLU and convolution), and its own output is **concatenated** with them "
                  "rather than added. Only k new feature maps are contributed by each layer "
                  f"(the growth rate, typically {k}), so with k₀ input channels there are "
                  "k₀ + l·k channels after l layers. As concatenation would increase the "
                  "number of channels without limit, the network is divided into dense "
                  "blocks separated by transition layers of 1 × 1 convolution and average "
                  "pooling. As in ResNet there are many short paths, and the gradient flows "
                  "easily. (b) The resulting network is narrow but deep, and uses parameters "
                  f"efficiently: DenseNet-121 has {net.params / 1e6:.1f} M parameters "
                  f"compared with {rn.params / 1e6:.1f} M for ResNet-50, for similar "
                  "ImageNet accuracy. Both parameter counts are calculated (and match "
                  "torchvision); the accuracies are the results published for torchvision.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    A.verify()
    T.gradcheck()
    return [fig_degradation(ctx), fig_residual_block(ctx), fig_unravelled_paths(ctx),
            fig_resnet_family(ctx), fig_batch_norm(ctx), fig_batch_norm_running(ctx),
            fig_squeeze_excite(ctx), fig_densenet(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("2.3 Residual networks")
    tab1, tab2 = st.tabs(["Plain vs residual", "Batch norm"])

    with tab1:
        c = st.columns(3)
        depth = c[0].slider("depth (hidden layers)", 2, 60, 30)
        epochs = c[1].slider("epochs", 5, 40, 15)
        lr = c[2].select_slider("learning rate", [0.005, 0.01, 0.02, 0.05], 0.02)
        if st.button("train both"):
            Xtr, ytr, Xte, yte = T.digits()
            fig, ax = plt.subplots(figsize=(8, 3.4))
            for res, col in [(False, C["red"]), (True, C["green"])]:
                net = T.MLP(64, 64, depth, 10, residual=res, seed=0)
                h = T.train(net, Xtr, ytr, Xte, yte, epochs=epochs, lr=lr)
                ax.plot(h["epoch"], h["train_loss"], color=col, lw=2,
                        label=f"{'residual' if res else 'plain'}: test "
                              f"{h['test_acc'][-1]:.1%}")
            ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("train loss")
            ax.legend()
            st.pyplot(fig)

    with tab2:
        scale = st.slider("weight scale (1 = correct)", 0.5, 1.6, 1.3, 0.05)
        s0 = deep_activation_std(scale=scale, bn=False)
        s1 = deep_activation_std(scale=scale, bn=True)
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.semilogy(s0, color=C["red"], lw=2, label="without BN")
        ax.semilogy(s1, color=C["green"], lw=2, label="with BN")
        ax.set_xlabel("layer"); ax.set_ylabel("activation std"); ax.legend()
        st.pyplot(fig)
        st.caption("Below 1 the activations shrink towards zero instead — batch norm "
                   "fixes both directions.")
