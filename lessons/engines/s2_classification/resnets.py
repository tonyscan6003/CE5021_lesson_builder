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
    use_source("sklearn_digits")
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
                  "The degradation problem, measured. Three fully connected networks "
                  "of the same width are trained identically on handwritten digits; "
                  "only their depth and connectivity differ. The 40-layer plain "
                  "network does worse than the 3-layer one — not on the test set, but "
                  "on the **training** set, where its loss stays at the chance level "
                  f"of ln 10 and its test accuracy at {plain['test_acc'][-1]:.0%}. "
                  "That rules out overfitting as the explanation: the deeper network "
                  "simply cannot be optimised. This is what the ResNet authors found "
                  "when they extended VGG-19 with more layers. Add identity shortcuts "
                  "around each layer and the same 40-layer network trains easily, "
                  f"reaching {res['test_acc'][-1]:.1%} — better than the shallow "
                  "network. The networks here are small and fully connected rather "
                  "than convolutional, but the effect is the same one.", png=png)


def fig_residual_block(ctx: BuildContext) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.1]))

    def box(ax, y, txt, col, w=1.7, x0=0.15):
        ax.add_patch(plt.Rectangle((x0, y), w, 0.55, fc=col, alpha=0.55, ec=C["ink"], lw=1))
        ax.text(x0 + w / 2, y + 0.275, txt, ha="center", va="center", fontsize=8.6)

    def arrow(ax, x, y0, y1):
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.2))

    # (a) the generic residual block
    ax = axes[0]
    ax.text(1.0, 0.0, "x", ha="center", fontsize=12)
    arrow(ax, 1.0, 0.18, 0.62)
    box(ax, 0.65, "weight layer", C["blue"])
    arrow(ax, 1.0, 1.22, 1.45)
    box(ax, 1.48, "ReLU", C["sky"])
    arrow(ax, 1.0, 2.05, 2.28)
    box(ax, 2.31, "weight layer", C["blue"])
    arrow(ax, 1.0, 2.88, 3.2)
    ax.scatter([1.0], [3.42], s=520, facecolors="white", edgecolors=C["ink"], lw=1.4,
               zorder=3)
    ax.text(1.0, 3.42, "+", ha="center", va="center", fontsize=15, zorder=4)
    arrow(ax, 1.0, 3.62, 3.95)
    box(ax, 3.98, "ReLU", C["sky"])
    ax.text(1.0, 4.85, "H(x) = F(x) + x", ha="center", fontsize=11, weight="bold")
    ax.plot([1.0, 2.25, 2.25, 1.18], [0.10, 0.10, 3.42, 3.42], color=C["red"], lw=2)
    ax.annotate("", xy=(1.17, 3.42), xytext=(1.6, 3.42),
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=2))
    ax.text(2.35, 1.8, "identity\nshortcut\n(x)", fontsize=9, color=C["red"])
    ax.text(-0.15, 1.8, "F(x)", fontsize=11, color=C["blue"], ha="center")
    ax.set_xlim(-0.6, 3.1); ax.set_ylim(-0.35, 5.15); ax.axis("off")
    ax.set_title("(a) residual learning", fontsize=10.5)

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
        y = 0.4
        for txt, col in steps:
            box(ax, y, txt, col, w=2.3)
            arrow(ax, 1.3, y + 0.55, y + 0.85)
            y += 0.88
        ax.scatter([1.3], [y + 0.05], s=420, facecolors="white", edgecolors=C["ink"],
                   lw=1.4, zorder=3)
        ax.text(1.3, y + 0.05, "+", ha="center", va="center", fontsize=14, zorder=4)
        # the input, feeding both the main path and the shortcut
        ax.text(1.3, -0.12, "x", ha="center", va="center", fontsize=12)
        arrow(ax, 1.3, 0.02, 0.38)
        ax.plot([1.3, 2.75, 2.75, 1.75], [0.15, 0.15, y + 0.05, y + 0.05],
                color=C["red"], lw=2)
        ax.annotate("", xy=(1.47, y + 0.05), xytext=(1.8, y + 0.05),
                    arrowprops=dict(arrowstyle="->", color=C["red"], lw=2))
        ax.text(1.3, 4.55, f"{n.params:,} parameters\n{n.macs / 1e6:.0f} M multiply-"
                           f"accumulates on 56 × 56", ha="center", fontsize=9,
                color=C["red"])
        ax.set_xlim(-0.1, 3.2); ax.set_ylim(-0.35, 5.15); ax.axis("off")
        ax.set_title(title, fontsize=10.5)
    png = save_mpl(fig, ctx, "residual_block")
    return Figure("residual_block",
                  "(a) The residual block: a main path of weight layers and activations, "
                  "plus a shortcut that adds the input x to the main path's output "
                  "before the final activation. Without the shortcut, a block would "
                  "have to learn some desired mapping H(x) directly. With it, the main "
                  "path only has to learn the **residual** F(x) = H(x) − x, and H(x) is "
                  "recovered as F(x) + x. Residual mappings turn out to be much easier "
                  "to train — in particular, if a block should do nothing, it only has "
                  "to drive F(x) to zero. (b) ResNet-18 and -34 use two 3 × 3 "
                  "convolutions per block. (c) The deeper ResNets use a bottleneck: a "
                  "1 × 1 convolution reduces 256 channels to 64, the 3 × 3 works on "
                  "64, and another 1 × 1 restores 256. Note that the bottleneck block "
                  "handles four times as many channels as the basic block for a "
                  "similar cost — the reason it is used in the deeper networks.",
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
                  "Veit and colleagues' explanation of why residual networks train. "
                  "(a) Each block offers two routes — through its layers or round them "
                  "by the shortcut — so n blocks unravel into 2ⁿ paths of different "
                  "lengths, and a ResNet behaves like an **ensemble** of networks. That "
                  "is why a block can be removed from a trained ResNet with little "
                  "effect. (b) For the 40-layer residual network trained above, with "
                  f"{n} residual blocks, most paths are of medium length. (c) But each "
                  "time a path goes *through* a block, its gradient is multiplied by "
                  f"that block's gain, which measures {rho:.3f} on average in this "
                  "network; the shortcut passes the gradient unchanged. Weighting each "
                  "path length by that measured gain shows the gradient is carried "
                  "almost entirely by **short** paths, which is what lets deep residual "
                  "networks train. (Panel (c) treats the blocks' gains as independent, "
                  "which is the approximation Veit's analysis makes.) The exact reason "
                  "shortcuts help is still debated — the original ResNet authors "
                  "thought vanishing gradients an unlikely cause — but their "
                  "effectiveness is not in doubt.", png=png)


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
                  "The ResNet family. Every member has the same plan — an initial "
                  "7 × 7 convolution and max pooling, then four stages of identical "
                  "residual blocks on 56, 28, 14 and 7 pixel feature maps, then global "
                  "average pooling and a single dense layer — and they differ only in "
                  "how many blocks each stage repeats and which kind of block. The "
                  "regular, repeating structure is what makes a 152-layer network "
                  "practical to specify. The parameter counts are computed from this "
                  "table and match torchvision's published figures exactly. Because "
                  "ResNet ends with global average pooling instead of large dense "
                  "layers, even ResNet-152, more than ten times as deep as VGG-16, has "
                  "fewer than half its parameters and needs less computation.", png=png)


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
                  "Batch normalisation, which ResNet inserts after each convolution. "
                  "(a) For each channel, the mean and variance are computed over the "
                  "mini-batch and used to normalise the values to zero mean and unit "
                  "variance; two learned parameters, γ and β, then scale and shift the "
                  "result, so the layer can undo the normalisation if that is what "
                  "training needs. Only γ and β are learned, so batch norm adds very few "
                  "parameters. (b) Why it helps. Without it, a small mis-scaling of the "
                  "weights compounds through the layers: here weights 30% larger than "
                  "they should be make the activations grow geometrically, by a factor "
                  "of more than 100,000 over 50 layers. With batch norm the scale is "
                  "reset at every layer and stays put. This is the effect that later "
                  "work points to — batch norm stops the internal states of deep "
                  "networks exploding — rather than the 'internal covariate shift' of "
                  "the original paper. It is what allows larger learning rates and "
                  "faster convergence.", png=png)


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
                  "Training and inference use batch norm differently. During training, "
                  "each mini-batch is normalised with its own statistics, which are "
                  "noisy (grey). At the same time a running mean and variance are kept, "
                  "updated after every step as μ ← d·μ + (1 − d)·μ_B, here with a decay "
                  f"d = {d}; they settle on the statistics of the whole training "
                  "population (blue). At inference there may be only one image, so "
                  "there is no batch to take statistics from — the stored running mean "
                  "and variance are used instead, and the layer becomes a fixed affine "
                  "transform.", png=png)


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

    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(2, c + 3, height_ratios=[1, 1], wspace=0.15, hspace=0.45)
    vmax = max(maps.max(), out.max())
    for i in range(c):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(maps[i], cmap="viridis", vmin=0, vmax=vmax); ax.axis("off")
        ax.set_title(f"ch {i + 1}", fontsize=8)
        if i == 0:
            ax.text(-0.12, 0.5, "input", transform=ax.transAxes, ha="right",
                    va="center", fontsize=9)
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(out[i], cmap="viridis", vmin=0, vmax=vmax); ax.axis("off")
        ax.set_title(f"× {s[i]:.2f}", fontsize=8, color=C["red"])
        if i == 0:
            ax.text(-0.12, 0.5, "rescaled", transform=ax.transAxes, ha="right",
                    va="center", fontsize=9)
    ax = fig.add_subplot(gs[:, c + 1:])
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
                  "The squeeze-and-excite block, run on an 8-channel feature map. "
                  "**Squeeze**: global average pooling reduces each channel to one "
                  "number. **Excite**: two small dense layers, the second with a "
                  "sigmoid, turn those numbers into a weight between 0 and 1 for each "
                  "channel, and every channel of the input is multiplied by its weight. "
                  "The block therefore models the **dependencies between channels**, "
                  "emphasising some and suppressing others according to the content of "
                  "the whole feature map. (The weights here are random, to show the "
                  "mechanism; in a trained network they are learned.) It is "
                  "lightweight and can be inserted into a ResNet block, between the "
                  "last convolution and the shortcut addition, or into Inception. "
                  f"Added to every block of ResNet-50 with a reduction ratio of 16, it "
                  f"costs {extra / 1e6:.2f} M extra parameters ({extra / base.params:.0%}) "
                  "and almost no extra computation, and the paper reports roughly a "
                  "tenth less ImageNet error in return.", png=png)


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
                  "DenseNet extends the ResNet idea with far more connectivity. "
                  "(a) Inside a dense block every composite function — batch norm, "
                  "ReLU and convolution — receives the outputs of **all** earlier "
                  "layers, and its own output is **concatenated** onto them rather than "
                  "added. Each layer contributes only k new feature maps (the growth "
                  f"rate, typically {k}), so with k₀ input channels there are k₀ + l·k "
                  "after l layers. Concatenation would grow the channel count without "
                  "limit, so the network is split into dense blocks separated by "
                  "transition layers of 1 × 1 convolution and average pooling. As in "
                  "ResNet there are many short paths, so gradients flow easily. "
                  "(b) The result is narrow but deep, and parameter-efficient: "
                  f"DenseNet-121 has {net.params / 1e6:.1f} M parameters against "
                  f"ResNet-50's {rn.params / 1e6:.1f} M, for similar ImageNet accuracy. "
                  "Both parameter counts are computed (and match torchvision); the "
                  "accuracies are torchvision's published results.", png=png)


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
