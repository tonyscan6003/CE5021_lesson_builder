"""Lesson 3.5 — the slides' diagrams, redrawn with real data.

Figures (built by semantic_seg.build, in lesson order)
    output_tensor        slide 4: a 1 x N vector vs an H x W x N tensor, argmax -> label map,
                         with DeepLabV3's real probabilities
    fcn_architecture     slide 6: classifier vs fully convolutional network with the
                         FCN-32s / 16s / 8s up-sampling paths, shapes computed
    transpose_1d         slide 8: 1D convolution and its transpose by padding / zero
                         insertion, values checked against PyTorch
    segnet_architecture  slide 9a: the VGG-16 encoder-decoder with pooling indices
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .._common import BuildContext, Figure, C, SERIES, save_mpl, load_image
from .. import _blocks as B
from .. import _torchdet as D
from .._volumes import OBLIQUE, volume, chan_len

FS = 9


def _cells(ax, M, x0, y0, cell, fmt="{:.2f}", fc=None, fs=7.5, ec=C["grey"]):
    """A matrix drawn as cells with values, top-left at (x0, y0) (y up)."""
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            col = fc(M[i, j]) if callable(fc) else (fc or "white")
            ax.add_patch(plt.Rectangle((x0 + j * cell, y0 - (i + 1) * cell), cell, cell, fc=col,
                                       ec=ec, lw=0.6, zorder=3))
            if fmt:
                ax.text(x0 + (j + 0.5) * cell, y0 - (i + 0.5) * cell, fmt.format(M[i, j]),
                        ha="center", va="center", fontsize=fs, zorder=4)


def fig_output_tensor(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=400)
    prob = D.semantic_scores("sheep_400", img)["prob"].astype(np.float32)   # every 2nd pixel
    cats = D.categories("deeplab")
    sh, bg = cats.index("sheep"), 0
    lab = prob.argmax(0)
    # the 6 x 6 window with the most even mix of sheep and background
    n = 6
    best, pos = 1.0, (0, 0)
    for i in range(0, lab.shape[0] - n, 2):
        for j in range(0, lab.shape[1] - n, 2):
            f = (lab[i:i + n, j:j + n] == sh).mean()
            if abs(f - 0.5) < best:
                best, pos = abs(f - 0.5), (i, j)
    i0, j0 = pos
    win = prob[:, i0:i0 + n, j0:j0 + n]
    icats = D.categories("resnet50")
    cl = D.classify("sheep", load_image("sheep", max_side=800))
    fig, ax = B.canvas(15.5, 7.4, (15.5, 7.2))
    # (a) image classification: a vector
    ax.text(0.1, 5.95, "(a) image classification: a 1 × N vector", fontsize=10, weight="bold")
    vec = cl["scores"][:5]
    for k, (s, l) in enumerate(zip(vec, cl["labels"])):
        y = 5.3 - k * 0.45
        ax.add_patch(plt.Rectangle((0.3, y - 0.4), 0.8, 0.4, fc=C["sky"] if k == 0 else "white",
                                   ec=C["grey"]))
        ax.text(0.7, y - 0.2, f"{s:.2f}", ha="center", va="center", fontsize=8.5)
        ax.text(1.2, y - 0.2, f"'{icats[l]}'", va="center", fontsize=8.5)
    ax.text(0.3, 2.85, "⋮  (1,000 ImageNet classes;\n5 largest shown)", fontsize=8.5, va="top")
    ax.text(0.3, 2.05, f"argmax → '{icats[cl['labels'][0]]}'", fontsize=9.5, color=C["red"])
    # (b) segmentation: an H x W x N tensor
    ax.text(4.2, 5.95, "(b) semantic segmentation: an H × W × N tensor (DeepLabV3, a 6 × 6 "
            "patch of the real output)", fontsize=10, weight="bold")
    p0 = B.photo(ax, img, 4.3, 2.6, 2.0, boxes=[np.array([j0, i0, j0 + n, i0 + n]) * 2], lw=2,
                 box_col=C["red"], label="input (patch in red)", fs=FS)
    zoom = img[i0 * 2:(i0 + n) * 2, j0 * 2:(j0 + n) * 2]
    B.photo(ax, zoom, 4.75, 0.25, 1.9, label="the patch, enlarged", fs=FS)
    cell = 0.36
    shown = [(bg, "'background'"), (sh, "'sheep'"), (cats.index("cow"), "'cow'")]
    x0, y0 = p0[2] + 0.5, 4.7
    for k, (c, name) in enumerate(shown):
        xx = x0 + k * (n * cell + 0.35)
        _cells(ax, win[c], xx, y0, cell, fc=lambda v: plt.cm.Blues(0.1 + 0.6 * v), fs=7)
        ax.text(xx + n * cell / 2, y0 + 0.12, f"channel {c}: p({name})", ha="center",
                fontsize=8.5)
    xe = x0 + 3 * (n * cell + 0.35)
    ax.text(xe - 0.1, y0 - n * cell / 2, f"…\n+ {len(cats) - 3}\nmore\nchannels", ha="left",
            va="center", fontsize=8.5, color=C["grey"])
    ax.text(x0 + 1.5 * (n * cell + 0.35), y0 - n * cell - 0.2,
            f"H × W × {len(cats)}: a probability for every class at every pixel",
            ha="center", va="top", fontsize=9)
    B.arrow(ax, x0 + 1.5 * (n * cell + 0.35), y0 - n * cell - 0.75,
            x0 + 1.5 * (n * cell + 0.35), y0 - n * cell - 1.35)
    ax.text(x0 + 1.5 * (n * cell + 0.35) + 0.25, y0 - n * cell - 1.05, "argmax over the classes "
            "at each pixel", va="center", fontsize=8.5)
    lx = x0 + 1.5 * (n * cell + 0.35) - n * cell / 2
    ly = y0 - n * cell - 1.45
    _cells(ax, lab[i0:i0 + n, j0:j0 + n], lx, ly, cell, fmt=None,
           fc=lambda v: np.array(plt.matplotlib.colors.to_rgb("#202020" if v == bg else C["orange"])))
    ax.text(lx + n * cell + 0.15, ly - n * cell / 2, "false-colour label map\n(orange: sheep,\n"
            "dark: background)", va="center", fontsize=8.5)
    lx = xe + 0.5
    ax.set_xlim(0, lx + 0.6); ax.set_ylim(-1.2, 6.2)
    png = save_mpl(fig, ctx, "output_tensor")
    return Figure("output_tensor",
                  "The output of a classification network and of a segmentation network. "
                  "(a) An image classifier (here ResNet-50, for the whole photograph) outputs a "
                  "vector of 1 × N class probabilities, and the class is found with an argmax "
                  "over the vector. (b) A segmentation network outputs a probability for every "
                  "class at every pixel, an H × W × N tensor. The values shown are DeepLabV3's "
                  "real probabilities for a 6 × 6 patch at the edge of a sheep (three of the "
                  f"{len(cats)} channels). The argmax is taken across the classes at each "
                  "pixel, and the resulting class at each pixel is shown in false colour.",
                  png=png)


def _vol(ax, x, ym, h, c, fc, side_of, label=None, below=None, fs=FS - 1):
    sd = side_of(h)
    ln = 0.12 + 0.5 * chan_len(c) / chan_len(4096)
    r = volume(ax, x, ym, ln, sd, fc, lw=0.9)
    if label:
        ax.text(x + ln / 2, ym + sd / 2 + OBLIQUE[1] * sd + 0.06, label, ha="center", va="bottom",
                fontsize=fs)
    if below:
        ax.text(x + ln / 2, ym - sd / 2 - 0.06, below, ha="center", va="top", fontsize=fs)
    return r, sd


def fig_fcn_architecture(ctx: BuildContext) -> Figure:
    N = 512
    side_of = lambda h: 0.25 + 1.6 * np.sqrt(h / N)
    stages = [(N // 2, 64), (N // 4, 128), (N // 8, 256), (N // 16, 512), (N // 32, 512)]
    fig, ax = B.canvas(15.0, 9.2, (14.5, 9.0))
    # (a) the classifier
    ya = 7.6
    ax.text(0.1, 9.0, "(a) a network for image classification (VGG-16)", fontsize=10,
            weight="bold")
    x = 0.3
    r, _ = _vol(ax, x, ya, 224, 3, "#eeeeee", lambda h: 0.25 + 1.6 * np.sqrt(h / 224 * 0.5),
                below="fixed input\n224 × 224")
    x = r + 0.25
    for k, (h, c) in enumerate([(112, 64), (56, 128), (28, 256), (14, 512), (7, 512)]):
        r, _ = _vol(ax, x, ya, h, c, "#dae8fc", lambda hh: 0.25 + 1.6 * np.sqrt(hh / 224 * 0.5))
        x = r + 0.12
    ax.text(x - 0.6, ya - 0.95, "conv + pool", ha="center", fontsize=FS - 1)
    for k, n_ in enumerate([4096, 4096, 1000]):
        B.bar(ax, x + 0.2 + k * 0.3, ya, 1.1 if k < 2 else 0.6, w=0.13, fc="#d5e8d4")
    ax.text(x + 0.5, ya - 0.75, "FC layers", ha="center", fontsize=FS - 1)
    B.arrow(ax, x + 1.1, ya, x + 1.6, ya)
    ax.text(x + 1.7, ya, "vector of class\nprobabilities (1 × 1000)", va="center", fontsize=FS)

    # (b) fully convolutional with up-sampling paths
    yb = 4.4
    ax.text(0.1, 6.15, f"(b) the fully convolutional network (FCN), here for a {N} × {N} input",
            fontsize=10, weight="bold")
    x = 0.3
    r, _ = _vol(ax, x, yb, N, 3, "#eeeeee", side_of, below=f"any-size input\n{N} × {N}")
    x = r + 0.3
    pos = {}
    for k, (h, c) in enumerate(stages):
        r, sd = _vol(ax, x, yb, h, c, "#dae8fc", side_of, label=f"pool{k + 1}\n{h}²×{c}")
        pos[k + 1] = (x, r, sd)
        x = r + 0.25
    for k, (n_, lab) in enumerate([(4096, "conv6\n7×7"), (4096, "conv7\n1×1"), (21, "score\n1×1")]):
        r, sd = _vol(ax, x, yb, N // 32, n_, "#d5e8d4", side_of, label=lab)
        x = r + 0.2
    ax.text(x - 1.1, yb + 1.15, "FC layers as convolutions", ha="center", fontsize=FS - 0.5,
            color=C["green"])
    ax.text(x - 1.1, yb - 0.5, f"{N // 32} × {N // 32} × 21", ha="center", va="top", fontsize=FS - 1)
    end = x
    # up-sampling paths
    out_x = end + 2.6
    ax.annotate("", xy=(out_x, yb), xytext=(end + 0.05, yb),
                arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.text(end + 0.8, yb + 0.08, "×32 up", fontsize=FS - 0.5)
    ax.text(out_x + 0.1, yb, "FCN-32s", va="center", fontsize=FS, weight="bold")
    p16 = (pos[4][1] + 1.6, yb - 1.55)
    p8 = (pos[3][1] + 1.0, yb - 2.65)
    for (px, py) in [p16, p8]:
        ax.add_patch(plt.Circle((px, py), 0.17, fc="white", ec="k", lw=1.2, zorder=5))
        ax.text(px, py, "+", ha="center", va="center", fontsize=12, zorder=6)
    # into the + nodes
    ax.annotate("", xy=(p16[0] + 0.17, p16[1] + 0.05), xytext=(end - 0.4, yb - 0.55),
                arrowprops=dict(arrowstyle="->", lw=1.1, connectionstyle="arc3,rad=-0.25"))
    ax.text((end - 0.4 + p16[0]) / 2 + 0.6, yb - 1.0, "×2 up", fontsize=FS - 1)
    ax.annotate("", xy=(p16[0] - 0.17, p16[1] + 0.05), xytext=(pos[4][0] + 0.2, yb - 0.6),
                arrowprops=dict(arrowstyle="->", lw=1.1, connectionstyle="arc3,rad=0.25"))
    ax.text(pos[4][0] - 0.4, yb - 1.25, "score\n(pool4)", fontsize=FS - 1)
    ax.annotate("", xy=(p8[0] + 0.17, p8[1] + 0.05), xytext=(p16[0], p16[1] - 0.17),
                arrowprops=dict(arrowstyle="->", lw=1.1, connectionstyle="arc3,rad=-0.2"))
    ax.text((p8[0] + p16[0]) / 2 + 0.25, p16[1] - 0.75, "×2 up", fontsize=FS - 1)
    ax.annotate("", xy=(p8[0] - 0.17, p8[1] + 0.05), xytext=(pos[3][0] + 0.2, yb - 0.85),
                arrowprops=dict(arrowstyle="->", lw=1.1, connectionstyle="arc3,rad=0.25"))
    ax.text(pos[3][0] - 0.55, yb - 1.6, "score\n(pool3)", fontsize=FS - 1)
    for (px, py), name, f in [(p16, "FCN-16s", 16), (p8, "FCN-8s", 8)]:
        ax.annotate("", xy=(out_x, py), xytext=(px + 0.17, py),
                    arrowprops=dict(arrowstyle="->", lw=1.3))
        ax.text(out_x - 0.9, py + 0.08, f"×{f} up", fontsize=FS - 0.5)
        ax.text(out_x + 0.1, py, name, va="center", fontsize=FS, weight="bold")
    ax.text(out_x + 1.3, yb - 1.4, f"each path:\n{N} × {N} × 21\nclass scores", va="center",
            fontsize=FS - 0.5, color=C["grey"])
    ax.set_xlim(0, out_x + 2.8); ax.set_ylim(1.2, 9.25)
    png = save_mpl(fig, ctx, "fcn_architecture")
    return Figure("fcn_architecture",
                  "The **fully convolutional network** (FCN), with the shapes calculated for a "
                  f"{N} × {N} input. (a) In an image classifier, information from across the "
                  "image is aggregated by the convolution and pooling stages, and fully "
                  "connected layers, which require a fixed input size, output a vector of "
                  "class probabilities. (b) In FCN the fully connected layers are converted "
                  "into convolutions, so an input of any size is accepted, and a score map "
                  f"with 21 channels (the Pascal VOC classes) is produced at a stride of 32, "
                  f"{N // 32} × {N // 32} here. Upsampling it directly by 32 gives FCN-32s. In "
                  "FCN-16s the score map is upsampled by 2 and added to a score map computed "
                  "from pool4, and the sum is upsampled by 16; FCN-8s adds pool3 in the same "
                  "way and upsamples by 8. These **skip connections** bring the finer spatial "
                  "information of the earlier stages into the prediction. The upsampling is "
                  "performed by transpose convolutions.",
                  png=png)


def _row(ax, vals, x, y, cell, fmt="{:g}", zeros=(), fs=8.5, ec=C["grey"]):
    for k, v in enumerate(vals):
        z = k in zeros
        ax.add_patch(plt.Rectangle((x + k * cell, y), cell, cell, fc="#f2f2f2" if z else "white",
                                   ec=ec, lw=0.9, zorder=2))
        ax.text(x + (k + 0.5) * cell, y + cell / 2, fmt.format(v), ha="center", va="center",
                fontsize=fs, color=C["grey"] if z else "k", zorder=3)


def fig_transpose_1d(ctx: BuildContext) -> Figure:
    import torch
    import torch.nn.functional as F
    kern = np.array([1.0, 2.0, 1.0])
    cell = 0.5
    fig, ax = B.canvas(14.0, 7.2, (14, 7.0))

    def conv_panel(x0, y0, xin, s, title):
        out = F.conv1d(torch.tensor(xin)[None, None], torch.tensor(kern)[None, None],
                       stride=s)[0, 0].numpy()
        _row(ax, xin, x0, y0, cell)
        ax.add_patch(plt.Rectangle((x0 - 0.04, y0 - 0.04), 3 * cell + 0.08, cell + 0.08, fill=False,
                                   ec=C["green"], lw=2.2, zorder=4))
        ax.add_patch(plt.Rectangle((x0 + s * cell - 0.07, y0 - 0.07), 3 * cell + 0.14,
                                   cell + 0.14, fill=False, ec=C["green"], lw=2.2, zorder=4))
        ox = x0 + (len(xin) - len(out)) * cell / 2
        _row(ax, out, ox, y0 - 1.5, cell)
        for k in range(len(out)):
            B.line_arrow(ax, x0 + (k * s + 1.5) * cell, y0 - 0.06, ox + (k + 0.5) * cell,
                         y0 - 1.5 + cell + 0.04, col=C["green"], lw=1.2)
        ax.text(x0, y0 + cell + 0.2, title, fontsize=9.5)
        return out

    def trans_panel(x0, y0, xin, s, title):
        ref = F.conv_transpose1d(torch.tensor(xin)[None, None], torch.tensor(kern)[None, None],
                                 stride=s)[0, 0].numpy()
        z = np.zeros((len(xin) - 1) * s + 1); z[::s] = xin
        p = np.concatenate([[0, 0], z, [0, 0]])
        out = np.array([(p[k:k + 3] * kern[::-1]).sum() for k in range(len(p) - 2)])
        assert np.allclose(out, ref)
        zeros = [k for k in range(len(p)) if k < 2 or k >= len(p) - 2 or
                 (k - 2) % s != 0]
        _row(ax, p, x0, y0, cell, zeros=zeros)
        ax.add_patch(plt.Rectangle((x0 - 0.05, y0 - 0.05), 3 * cell + 0.1, cell + 0.1, fill=False,
                                   ec=C["purple"], lw=2.2, zorder=4))
        ox = x0 + cell
        _row(ax, out, ox, y0 - 1.5, cell)
        B.line_arrow(ax, x0 + 1.5 * cell, y0 - 0.06, ox + 0.5 * cell, y0 - 1.5 + cell + 0.04,
                     col=C["purple"], lw=1.2)
        ax.text(x0, y0 + cell + 0.2, title, fontsize=9.5)
        return out

    conv_panel(0.6, 5.6, np.array([1.0, 2, 0, 3]), 1, "(a) convolution of 4 inputs, kernel "
               "[1 2 1], stride 1: 2 outputs")
    trans_panel(0.3, 2.2, np.array([2.0, 1]), 1, "(b) its transpose: pad with 2 zeros on each "
                "side, convolve at stride 1: 4 outputs")
    conv_panel(7.6, 5.6, np.array([1.0, 2, 0, 3, 1]), 2, "(c) convolution of 5 inputs, kernel "
               "[1 2 1], stride 2: 2 outputs")
    trans_panel(7.3, 2.2, np.array([2.0, 1]), 2, "(d) its transpose: insert a zero between "
                "inputs, pad, stride 1: 5 outputs")
    ax.text(0.3, 0.15, "zeros inserted or padded are shaded; every result is identical to "
            "PyTorch's conv_transpose1d", fontsize=8.5, color=C["grey"])
    ax.set_ylim(0, 6.6)
    png = save_mpl(fig, ctx, "transpose_1d")
    return Figure("transpose_1d",
                  "Transpose convolution emulated by direct convolution, shown in one "
                  "dimension. (a) A convolution of 4 inputs with a kernel of size 3 gives 2 "
                  "outputs. (b) The equivalent transpose convolution must give 4 outputs from 2 "
                  "inputs: the input is padded with two zeros on each side (k − 1 = 2) and "
                  "convolved with the kernel at stride 1. (c) A convolution of 5 inputs at "
                  "stride 2 gives 2 outputs. (d) Its transpose inserts one zero (s − 1) between "
                  "the inputs before padding, and the direct convolution at stride 1 then gives "
                  "5 outputs; this is why it is called fractionally strided convolution. The "
                  "first kernel position is outlined in each case. The kernel is flipped for "
                  "the direct convolution; the results agree with PyTorch's conv_transpose1d.",
                  png=png)


def fig_segnet_architecture(ctx: BuildContext) -> Figure:
    H, W = 352, 480
    side_of = lambda h: 0.3 + 1.7 * np.sqrt(h / H)
    blocks = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]
    fig, ax = B.canvas(16.0, 6.1, (16, 5.6))
    ym = 2.4
    x = 0.3
    r, _ = _vol(ax, x, ym, H, 3, "#eeeeee", side_of, below=f"input\n{H} × {W} × 3")
    x = r + 0.2
    pools = []
    h = H
    for k, (c, n) in enumerate(blocks):
        for j in range(n):
            r, sd = _vol(ax, x, ym, h, c, "#dae8fc", side_of)
            x = r - 0.25 + 0.04
        x = r + 0.05
        h2 = h // 2
        r, sd = _vol(ax, x, ym, h2, c, "#d5e8d4", side_of)
        pools.append((x, r, sd, h2, c))
        x = r + 0.15
        h = h2
    enc_end = x
    ax.text((0.3 + enc_end) / 2, ym - 1.45, "encoder: VGG-16 convolution layers (blue) and max "
            "pooling (green)", ha="center", fontsize=FS)
    x += 0.2
    unpools = []
    for k, (c, n) in enumerate(reversed(blocks)):
        h = h * 2
        r, sd = _vol(ax, x, ym, h, c, "#f4cccc", side_of)
        unpools.append((x, r, sd))
        x = r + 0.05
        for j in range(n):
            c_out = c if j < n - 1 else (blocks[len(blocks) - 2 - k][0] if k < 4 else 64)
            r, sd = _vol(ax, x, ym, h, c_out, "#dae8fc", side_of)
            x = r - 0.25 + 0.04
        x = r + 0.15
    r, _ = _vol(ax, x, ym, H, 12, "#fce5cd", side_of, below=f"softmax\n{H} × {W} × K")
    ax.text((enc_end + x) / 2, ym - 1.45, "decoder: unpooling (red) and convolution layers",
            ha="center", fontsize=FS)
    for (px, pr, psd, ph, pc), (ux, ur, usd) in zip(pools, reversed(unpools)):
        x0, y0 = (px + pr) / 2, ym + psd / 2 + OBLIQUE[1] * psd
        x1, y1 = (ux + ur) / 2, ym + usd / 2 + OBLIQUE[1] * usd
        dist = abs(x1 - x0)
        top = 3.6 + 0.32 * (len(pools) - pools.index((px, pr, psd, ph, pc)))
        rad = max(-2 * (top - max(y0, y1)) / max(dist, 1e-6), -0.9)
        ax.annotate("", xy=(x1, y1 + 0.02), xytext=(x0, y0 + 0.02),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color=C["ink"],
                                    connectionstyle=f"arc3,rad={rad:.3f}"))
    ax.text((pools[0][0] + unpools[-1][0]) / 2, 5.75, "pooling indices", ha="center",
            fontsize=FS)
    ax.set_xlim(0, r + 0.5)
    png = save_mpl(fig, ctx, "segnet_architecture")
    return Figure("segnet_architecture",
                  "An **encoder-decoder** network for semantic segmentation (SegNet; DeconvNet "
                  "has the same structure), with the shapes for a "
                  f"{H} × {W} input. The encoder is the convolution part of VGG-16, in which "
                  f"five max-pooling stages reduce the size to {H // 32} × {W // 32}. A complete "
                  "decoder, the mirror image of the encoder, is learned with end-to-end "
                  "training. During each max pooling the positions of the maxima, the **pooling "
                  "indices**, are stored and passed to the corresponding **unpooling** layer of "
                  "the decoder (curved arrows), which places each value back at its stored "
                  "position; convolution layers then fill in the sparse result, as shown in "
                  "the next figure. A softmax over the K classes gives the prediction at every "
                  "pixel.",
                  png=png)
