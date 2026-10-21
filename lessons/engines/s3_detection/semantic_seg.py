"""Lesson 3.5 — Semantic segmentation (part A).

Figures
    segmentation_types     semantic (DeepLabV3) vs instance (Mask R-CNN), real outputs
    segmentation_datasets  Pascal VOC, MS COCO, ADE20K, Cityscapes, KITTI
    segmentation_output    DeepLabV3's real H x W x 21 scores -> argmax -> colour map
    segmentation_metrics   pixel accuracy, mean accuracy, per-class IoU, mIoU, FWIoU on a
                           small prediction / ground-truth pair
    fully_convolutional    VGG-16's dense layers as convolutions: a real score map
    conv_as_matrix         a 2 x 2 convolution as a 4 x 9 matrix, and its transpose
    transpose_conv         stride-1 and stride-2 transpose convolution as direct
                           convolution, checked against PyTorch; the checkerboard
    unpooling              max pooling with indices, unpooling, transpose convolution
    unet                   the U-Net with every shape computed
    (slide diagrams output_tensor, fcn_architecture, transpose_1d and
     segnet_architecture are in seg_diagrams.py)

Source: reference/legacy_md/lesson_3_5_sem_seg_partA + lesson_3_5_script.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .._common import BuildContext, Figure, C, SERIES, save_mpl, load_image, sync_matrix
from .. import _torchdet as D
from .. import _volumes as V

FIGURES = ["segmentation_types", "segmentation_datasets", "output_tensor",
           "segmentation_output", "segmentation_metrics", "fcn_architecture",
           "fully_convolutional", "conv_as_matrix", "transpose_1d",
           "transpose_conv", "segnet_architecture", "unpooling", "unet"]


def voc_palette(n=256):
    """The standard Pascal VOC colour map (bit-interleaved label colours)."""
    pal = np.zeros((n, 3), np.uint8)
    for i in range(n):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j); g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j); c >>= 3
        pal[i] = (r, g, b)
    return pal


PAL = voc_palette()


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _overlay(img, labels, alpha=0.55):
    col = PAL[labels].astype(float)
    m = labels > 0
    out = img.astype(float).copy()
    out[m] = (1 - alpha) * out[m] + alpha * col[m]
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------

def fig_segmentation_types(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=800)
    sem = D.semantic("sheep", img)["labels"]
    ins = D.instance_masks("sheep", img)
    cats = D.categories("deeplab")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.4))
    axes[0].imshow(img); axes[0].set_title("image", fontsize=10.5)
    axes[1].imshow(_overlay(img, sem))
    present = [cats[c] for c in np.unique(sem) if c]
    axes[1].set_title(f"(a) semantic segmentation (DeepLabV3): every pixel\nlabelled "
                      f"'{', '.join(present)}' or background", fontsize=10)
    over = img.astype(float).copy()
    for i, m in enumerate(ins["masks"]):
        col = np.array(plt.matplotlib.colors.to_rgb(SERIES[i % len(SERIES)])) * 255
        over[m] = 0.45 * over[m] + 0.55 * col
    axes[2].imshow(over.astype(np.uint8))
    axes[2].set_title(f"(b) instance segmentation (Mask R-CNN):\n{len(ins['masks'])} separate "
                      "sheep", fontsize=10)
    for ax in axes:
        _clean(ax); ax.set_ylim(img.shape[0] * 0.66, img.shape[0] * 0.12)
    png = save_mpl(fig, ctx, "segmentation_types")
    return Figure("segmentation_types",
                  "Semantic and instance segmentation, produced by two real pre-trained "
                  "networks. Basic segmentation divides an image into regions of similar "
                  "pixels. (a) In **semantic segmentation** each pixel is assigned a class, "
                  "based on the object to which it belongs, which can be considered as image "
                  "classification with a separate prediction at every pixel: all of the "
                  "sheep are labelled with the same class. (b) In **instance "
                  "segmentation** the information from object detection is combined with "
                  "segmentation, so that each instance of a class is segmented separately. "
                  "Instance segmentation is more challenging than either semantic "
                  "segmentation or object detection, and is a key task in benchmarks such as "
                  "MS COCO.",
                  png=png)


SEG_DATASETS = [
    ("Pascal VOC 2012", "20 + background", "2,913 (train + val)", "objects", "general"),
    ("MS COCO", "80 things (+ 91 stuff)", "≈ 123,000 (train + val)", "instances", "general"),
    ("ADE20K", "150 (benchmark)", "≈ 22,000 (train + val)", "objects and parts", "scenes"),
    ("Cityscapes", "30 (19 evaluated)", "5,000 fine, 20,000 coarse", "pixels, instances",
     "driving"),
    ("KITTI (semantic)", "Cityscapes classes", "200 training", "pixels", "driving"),
]


def fig_segmentation_datasets(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(12.5, 2.8))
    ax.axis("off")
    tbl = ax.table(cellText=[list(r) for r in SEG_DATASETS],
                   colLabels=["dataset", "classes", "annotated images", "labels", "domain"],
                   loc="center", cellLoc="left", colWidths=[0.18, 0.22, 0.26, 0.18, 0.12])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.65)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["light"])
        if r == 0:
            cell.set_text_props(weight="bold")
    png = save_mpl(fig, ctx, "segmentation_datasets")
    return Figure("segmentation_datasets",
                  "Datasets for semantic and instance segmentation, with the figures "
                  "published by their authors. The development of a segmentation dataset is "
                  "difficult, as every pixel must be annotated by hand. Pascal VOC includes "
                  "segmentation annotations; MS COCO contains instance masks; ADE20K was "
                  "developed for scene parsing and also annotates the parts of objects; and "
                  "Cityscapes and KITTI contain driving scenes for autonomous driving, "
                  "where detailed representations of the environment are required.",
                  png=png)


def fig_segmentation_output(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=400)
    r = D.semantic_scores("sheep_400", img)
    prob = r["prob"].astype(np.float32)
    cats = D.categories("deeplab")
    shown = [0, cats.index("sheep"), cats.index("cow")]
    lab = prob.argmax(0)
    small = img[::2, ::2]
    fig, axes = plt.subplots(1, 5, figsize=(16, 2.8))
    axes[0].imshow(small); axes[0].set_title("input, H × W × 3", fontsize=10)
    for ax, c in zip(axes[1:4], shown):
        ax.imshow(prob[c], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"score for '{cats[c]}'\n(channel {c} of {prob.shape[0]})", fontsize=10)
    axes[4].imshow(PAL[lab])
    axes[4].set_title("argmax over the 21 channels,\nin false colour", fontsize=10)
    for ax in axes:
        _clean(ax)
    png = save_mpl(fig, ctx, "segmentation_output")
    return Figure("segmentation_output",
                  "The output of a semantic segmentation network, from DeepLabV3 trained on "
                  "the 20 Pascal VOC classes plus background. An image classifier outputs a "
                  "vector of 1 × N class probabilities, and the class is found by an argmax "
                  "over the vector. A segmentation network outputs an H × W × N tensor, a "
                  "class probability for every pixel: three of the 21 channels are shown. "
                  "The argmax is taken across the classes at each pixel, and the class at "
                  "each pixel is shown as a false colour. The 'cow' channel is close to zero "
                  "everywhere, but where the network is less certain, as at the edges of the "
                  "sheep, the probability is shared between classes.",
                  png=png)


def seg_metrics(gt, pred, n):
    """Confusion matrix and the standard semantic segmentation metrics."""
    cm = np.bincount(n * gt.ravel() + pred.ravel(), minlength=n * n).reshape(n, n)
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    iou = tp / (tp + fp + fn)
    freq = cm.sum(1) / cm.sum()
    return dict(cm=cm, tp=tp, fp=fp, fn=fn, iou=iou, pixel_acc=tp.sum() / cm.sum(),
                mean_acc=np.mean(tp / cm.sum(1)), miou=iou.mean(), fwiou=(freq * iou).sum())


def metrics_example():
    """A 12 x 16 ground truth with two objects, and a prediction with typical
    errors: boundary pixels lost, a bulge, and part of one object confused."""
    gt = np.zeros((12, 16), int)
    gt[3:10, 2:8] = 1                         # sheep
    gt[5:11, 10:14] = 2                       # dog
    pred = gt.copy()
    pred[3, 2:8] = 0                          # top edge missed
    pred[4:7, 8] = 1                          # bulge into background
    pred[5:7, 10:12] = 1                      # part of the dog predicted as sheep
    pred[0:2, 13:15] = 2                      # a background false positive
    return gt, pred


def fig_segmentation_metrics(ctx: BuildContext) -> Figure:
    gt, pred = metrics_example()
    names = ["background", "sheep", "dog"]
    m = seg_metrics(gt, pred, 3)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.3]))
    lut = np.array([[0.92, 0.92, 0.92], plt.matplotlib.colors.to_rgb(C["orange"]),
                    plt.matplotlib.colors.to_rgb(C["blue"])])
    axes[0].imshow(lut[gt]); axes[0].set_title("(a) ground truth: background, sheep, dog",
                                                fontsize=10)
    axes[1].imshow(lut[pred]); axes[1].set_title("(b) prediction", fontsize=10)
    k = 1
    tpfn = np.zeros(gt.shape + (3,))
    tpfn[(gt == k) & (pred == k)] = plt.matplotlib.colors.to_rgb(C["green"])
    tpfn[(gt != k) & (pred == k)] = plt.matplotlib.colors.to_rgb(C["red"])
    tpfn[(gt == k) & (pred != k)] = plt.matplotlib.colors.to_rgb(C["orange"])
    tpfn[(gt != k) & (pred != k)] = (0.95, 0.95, 0.95)
    axes[2].imshow(tpfn)
    axes[2].set_title(f"(c) class 'sheep': TP {int(m['tp'][k])} (green), FP {int(m['fp'][k])} "
                      f"(red),\nFN {int(m['fn'][k])} (orange): IoU = {m['iou'][k]:.3f}", fontsize=9.5)
    for ax in axes[:3]:
        ax.set_xticks(np.arange(-0.5, 16, 1), minor=True); ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.5); _clean(ax)
    ax = axes[3]; ax.axis("off")
    rows = [[n, f"{int(m['tp'][i])}", f"{int(m['fp'][i])}", f"{int(m['fn'][i])}", f"{m['iou'][i]:.3f}"]
            for i, n in enumerate(names)]
    tbl = ax.table(cellText=rows, colLabels=["class", "TP", "FP", "FN", "IoU"], loc="upper center",
                   cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["light"])
        if r == 0:
            cell.set_text_props(weight="bold")
    ax.text(0.02, 0.36, f"pixel accuracy = {m['pixel_acc']:.3f}\nmean accuracy = {m['mean_acc']:.3f}\n"
            f"mean IoU = {m['miou']:.3f}\nfrequency-weighted IoU = {m['fwiou']:.3f}",
            fontsize=10, transform=ax.transAxes, va="top", color=C["red"])
    png = save_mpl(fig, ctx, "segmentation_metrics")
    return Figure("segmentation_metrics",
                  "The **intersection over union** (IoU) is the primary measure of the "
                  "performance of a semantic segmentation system. It is calculated for each "
                  "class at the pixel level, as $IoU = TP / (TP + FP + FN)$: a pixel where "
                  "the ground truth and the prediction agree is a true positive, a pixel of "
                  "the class that is not predicted is a false negative, and a pixel "
                  "incorrectly predicted as the class is a false positive. The IoU of each "
                  "class is calculated and the mean is taken (mIoU). It is the same measure "
                  "as for bounding boxes, the ratio of the intersection to the union of the "
                  "ground truth and the prediction. The pixel accuracy is dominated by the "
                  f"large background class ({m['pixel_acc']:.2f} here), which is why the "
                  f"mean IoU ({m['miou']:.2f}) is preferred. Small example; all values "
                  "calculated.",
                  png=png)


def fig_fully_convolutional(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=672)
    icats = D.categories("resnet50")
    ram = icats.index("ram")
    r = D.vgg_fully_convolutional("sheep_672", img, ram)
    s = r["score"]
    c, h, w = (int(v) for v in r["p5_shape"])
    sh, sw = s.shape
    # same weights: on a 224 x 224 input the converted network equals the classifier
    assert float(r["check"]) < 1e-3, float(r["check"])
    assert (sh, sw) == (h - 6, w - 6), (sh, sw, h, w)   # 7 x 7 fc6 kernel, no padding
    n_ram = int((r["top"] == ram).sum())
    assert s[-1].mean() < 0.2 and s[:-2].mean() > 0.6, (s[-1].mean(), s[:-2].mean())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.0), gridspec_kw=dict(width_ratios=[1, 1.15]))
    axes[0].imshow(img); _clean(axes[0])
    axes[0].set_title(f"(a) input {img.shape[1]} × {img.shape[0]}", fontsize=10)
    ax = axes[1]
    im = ax.imshow(s, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    for i in range(sh):
        for j in range(sw):
            ax.text(j, i, f"{s[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="k" if s[i, j] > 0.55 else "w")
    ax.set_xticks(range(sw)); ax.set_yticks(range(sh))
    ax.tick_params(labelsize=7, length=0)
    ax.set_title(f"(b) final feature map, {sw} × {sh} × 1000: the '{icats[ram]}' channel "
                 "after a softmax", fontsize=10)
    ax.set_xlabel(f"pool5 {w} × {h} × {c} → fc6 as 7 × 7 conv {sw} × {sh} × 4096 → "
                  f"fc7 as 1 × 1 conv {sw} × {sh} × 4096 → fc8 as 1 × 1 conv "
                  f"{sw} × {sh} × 1000", fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    png = save_mpl(fig, ctx, "fully_convolutional")
    return Figure("fully_convolutional",
                  "The first step of the **fully convolutional network** (FCN): the fully "
                  "connected layers of an image classifier are converted into convolution "
                  "layers, with exactly the same weights. In VGG-16 the first dense layer, "
                  "which connects the 7 × 7 × 512 output of the last pooling layer to 4,096 "
                  "neurons, becomes 4,096 kernels of 7 × 7 × 512, and the other two dense "
                  "layers become 1 × 1 convolutions, so the network accepts an image of any "
                  f"size. For this {img.shape[1]} × {img.shape[0]} image the last pooling "
                  f"layer gives a {w} × {h} × {c} feature map, and the converted layers give "
                  f"maps of {sw} × {sh} × 4,096, {sw} × {sh} × 4,096 and {sw} × {sh} × 1,000. "
                  "The final feature map therefore holds a vector of 1,000 class scores at "
                  f"each of its {sw} × {sh} positions, in place of the single vector of the "
                  "classifier, and neighbouring positions are 32 pixels apart in the image. "
                  "In (b) a softmax is applied at each position and the channel of the "
                  f"ImageNet class 'ram' is shown: it is the most probable class at {n_ram} "
                  f"of the {sh * sw} positions, and its probability is lowest along the "
                  "bottom row. The semantic content is high, but the spatial resolution is "
                  "very coarse, so the output must be upsampled to give a prediction at "
                  "every pixel.",
                  png=png)


def conv_matrix(kernel, n):
    """The (out² × n²) matrix C with C · vec(x) = vec(valid convolution of x)
    (cross-correlation, as in deep learning frameworks)."""
    k = kernel.shape[0]
    o = n - k + 1
    Cm = np.zeros((o * o, n * n))
    for i in range(o):
        for j in range(o):
            for a in range(k):
                for b in range(k):
                    Cm[i * o + j, (i + a) * n + (j + b)] = kernel[a, b]
    return Cm


def fig_conv_as_matrix(ctx: BuildContext) -> Figure:
    x = np.array([[1, 2, 0], [3, 1, 2], [0, 1, 3]])
    k = np.array([[1, 2], [0, 1]])
    Cm = conv_matrix(k, 3)
    y = (Cm @ x.ravel()).reshape(2, 2)
    direct = np.array([[(x[i:i + 2, j:j + 2] * k).sum() for j in range(2)] for i in range(2)])
    assert np.array_equal(y, direct)
    yt = (Cm.T @ y.ravel()).reshape(3, 3)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), gridspec_kw=dict(width_ratios=[1, 0.8, 3, 0.8, 1]))
    sync_matrix(axes[0], x, title="(a) input 3 × 3")
    sync_matrix(axes[1], k, title="(b) kernel 2 × 2", cmap="Oranges")
    sync_matrix(axes[2], Cm.astype(int), title="(c) the convolution as a 4 × 9 matrix C",
                cmap="Oranges")
    sync_matrix(axes[3], y.astype(int), title="C · x → 2 × 2")
    sync_matrix(axes[4], yt.astype(int), title="(d) Cᵀ · y → 3 × 3")
    png = save_mpl(fig, ctx, "conv_as_matrix")
    return Figure("conv_as_matrix",
                  "**Convolution as a matrix multiplication.** (a, b) A 3 × 3 input is "
                  "convolved with a 2 × 2 kernel, giving a 2 × 2 output. (c) The same "
                  "operation is written as a matrix multiplication: the input is reshaped "
                  "into a 9 × 1 vector and multiplied by a 4 × 9 matrix in which each row "
                  "holds the kernel values at the positions they multiply, giving a 4 × 1 "
                  "vector that is reshaped into the 2 × 2 output. (d) The transpose of the "
                  "matrix, 9 × 4, maps a 2 × 2 input back to a 3 × 3 output: this is the "
                  "**transpose convolution**. A convolution reduces the spatial dimensions, "
                  "and the transpose convolution with the same kernel increases them. (It "
                  "restores the size, not the values: Cᵀ is not the inverse of C.) The "
                  "transpose convolution is also the operation used in backpropagation "
                  "through a convolution layer.",
                  png=png)


def transpose_as_direct(x, k, stride):
    """Transpose convolution emulated by direct convolution: insert stride-1
    zeros between the input samples, pad by k-1 on every side, and convolve
    with the kernel rotated by 180° at stride 1."""
    n = x.shape[0]
    z = np.zeros(((n - 1) * stride + 1,) * 2)
    z[::stride, ::stride] = x
    p = np.pad(z, k.shape[0] - 1)
    kf = k[::-1, ::-1]
    m = p.shape[0] - k.shape[0] + 1
    out = np.array([[(p[i:i + k.shape[0], j:j + k.shape[0]] * kf).sum() for j in range(m)]
                    for i in range(m)])
    return z, p, out


def fig_transpose_conv(ctx: BuildContext) -> Figure:
    import torch
    import torch.nn.functional as F
    k = np.array([[1, 0, 2], [1, 1, 0], [0, 2, 1]], float)
    cases = [(np.array([[2, 1], [0, 3]], float), 1), (np.array([[1, 2], [3, 1]], float), 2)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.0), gridspec_kw=dict(width_ratios=[0.7, 1.3, 1, 1.1]))
    for r, (x, s) in enumerate(cases):
        z, p, out = transpose_as_direct(x, k, s)
        ref = F.conv_transpose2d(torch.tensor(x)[None, None], torch.tensor(k)[None, None],
                                 stride=s)[0, 0].numpy()
        assert np.allclose(out, ref)
        sync_matrix(axes[r, 0], x.astype(int), title=f"input {x.shape[0]} × {x.shape[0]}")
        sync_matrix(axes[r, 1], p.astype(int),
                    title=("pad with 2 zeros" if s == 1 else "insert zeros between samples,\n"
                           "then pad with 2 zeros") + f": {p.shape[0]} × {p.shape[0]}")
        sync_matrix(axes[r, 2], k[::-1, ::-1].astype(int), title="3 × 3 kernel (rotated), stride 1",
                    cmap="Oranges")
        sync_matrix(axes[r, 3], out.astype(int),
                    title=f"output {out.shape[0]} × {out.shape[0]}\n(= PyTorch conv_transpose2d, "
                          f"stride {s})")
    png = save_mpl(fig, ctx, "transpose_conv")
    # checkerboard: how many kernel placements cover each output pixel at stride 2
    cover = transpose_as_direct(np.ones((6, 6)), np.ones((3, 3)), 2)[2]
    return Figure("transpose_conv",
                  "In deep learning frameworks a transpose convolution is not computed with "
                  "the transposed matrix, but is emulated by a **direct convolution**, with "
                  "weights that are learned to perform the upsampling. Top: the transpose of "
                  "a convolution of a 4 × 4 input with a 3 × 3 kernel must give a 4 × 4 "
                  "output from a 2 × 2 input; this is achieved by padding the input with "
                  "two zeros on each side and convolving with a 3 × 3 kernel. Bottom: a "
                  "convolution of a 5 × 5 input with a 3 × 3 kernel at stride 2 gives a "
                  "2 × 2 output, so its transpose must give a 5 × 5 output from a 2 × 2 "
                  "input: a zero is inserted between the input samples, the result is "
                  "padded, and the kernel is applied at stride 1. This is also called **fractionally "
                  "strided convolution**, and the stride-2 case is the one most often used "
                  "for upsampling. Both results are identical to PyTorch's "
                  "conv_transpose2d. When the kernel size is not divisible by the stride, "
                  "some output pixels receive contributions from more kernel placements "
                  f"than others (between {int(cover.min())} and {int(cover.max())} here), "
                  "which can produce a **checkerboard** pattern in the output.",
                  png=png)


def fig_unpooling(ctx: BuildContext) -> Figure:
    import torch
    import torch.nn.functional as F
    x = torch.tensor([[1, 3, 2, 1], [4, 2, 1, 5], [3, 1, 6, 2], [2, 7, 1, 0]], dtype=torch.float32)
    pooled, idx = F.max_pool2d(x[None, None], 2, 2, return_indices=True)
    feat = pooled * torch.tensor([[1.5, 0.5], [1.0, 2.0]])          # decoder features
    un = F.max_unpool2d(feat, idx, 2, 2)[0, 0]
    k = torch.tensor([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
    filled = F.conv2d(un[None, None], k[None, None], padding=1)[0, 0]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    sync_matrix(axes[0], x.numpy().astype(int), title="(a) encoder: max pooling 2 × 2,\n"
                "the positions of the maxima are stored")
    pos = np.zeros((4, 4), int)
    pos.ravel()[idx.numpy().ravel()] = 1
    for i, j in zip(*np.where(pos)):
        axes[0].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=C["red"], lw=2.5))
    sync_matrix(axes[1], feat[0, 0].numpy(), fmt="{:.1f}", title="(b) decoder: a 2 × 2 feature map")
    sync_matrix(axes[2], un.numpy(), fmt="{:.1f}", title="(c) unpooling: each value returns to\n"
                "the stored position; the rest are 0")
    sync_matrix(axes[3], filled.numpy(), fmt="{:.2f}", title="(d) a (transpose) convolution\n"
                "fills in the sparse map")
    png = save_mpl(fig, ctx, "unpooling")
    return Figure("unpooling",
                  "**Unpooling** in the encoder-decoder networks DeconvNet and SegNet, "
                  "computed with PyTorch's max_pool2d and max_unpool2d. (a) During max "
                  "pooling in the encoder, the largest value in each 2 × 2 region is kept and "
                  "the spatial information is lost, but here the position of each maximum is "
                  "also stored. (b, c) In the corresponding stage of the decoder, each value "
                  "is placed back at the stored position, and the other elements remain "
                  "zero. (d) A convolution, with a kernel that would be learned (a fixed "
                  "smoothing kernel is used here), interpolates the sparse map. The stored "
                  "positions help to recover the fine spatial detail lost by pooling, which "
                  "FCN, with upsampling only, does not recover. Unpooling is not a standard "
                  "layer in all frameworks.",
                  png=png)


def unet_shapes(n=572, base=64, depth=4):
    """Shapes through the original U-Net (valid 3 x 3 convolutions, 2 x 2 max
    pooling, 2 x 2 up-convolutions), using o = i - k + 1 (Lesson 1.1)."""
    enc, s, c = [], n, base
    for d in range(depth + 1):
        a = s - 2; b = a - 2                          # two valid 3 x 3 convolutions
        enc.append((s, a, b, c))
        if d < depth:
            s, c = b // 2, c * 2
    dec, s = [], enc[-1][2]
    for d in range(depth - 1, -1, -1):
        c = enc[d][3]
        up = s * 2                                    # 2 x 2 up-convolution
        crop = enc[d][2]                              # the skip map is cropped to `up`
        a = up - 2; b = a - 2
        dec.append((up, crop, a, b, c))
        s = b
    return enc, dec


def fig_unet(ctx: BuildContext) -> Figure:
    enc, dec = unet_shapes()
    fig, ax = plt.subplots(figsize=(14, 6.4))
    y0, dy = 6.0, 1.3
    for d, (s, a, b, c) in enumerate(enc):
        y = y0 - d * dy
        x = 0.4 + d * 0.9
        ax.add_patch(plt.Rectangle((x, y), 2.3, 0.75, fc=C["blue"], alpha=0.5, ec=C["ink"]))
        ax.text(x + 1.15, y + 0.37, f"{s}² → {b}² × {c}", ha="center", va="center", fontsize=9)
        if d < len(enc) - 1:
            ax.annotate("", xy=(x + 1.15 + 0.9, y - dy + 0.77), xytext=(x + 1.15, y),
                        arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.3))
            ax.text(x + 1.75, y - 0.3, "max pool 2 × 2", fontsize=8, color=C["red"])
    for d, (up, crop, a, b, c) in enumerate(dec):
        lvl = len(enc) - 2 - d
        y = y0 - lvl * dy
        x = 11.0 - lvl * 0.9
        ax.add_patch(plt.Rectangle((x, y), 2.5, 0.75, fc=C["green"], alpha=0.5, ec=C["ink"]))
        ax.text(x + 1.25, y + 0.37, f"up {up}² + crop → {b}² × {c}", ha="center", va="center",
                fontsize=9)
        xe = 0.4 + lvl * 0.9 + 2.3
        ax.annotate("", xy=(x - 0.02, y + 0.37), xytext=(xe + 0.02, y + 0.37),
                    arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.2, ls="--"))
        ax.text((x + xe) / 2, y + 0.47, f"copy, crop {crop}² → {up}², concatenate", ha="center",
                fontsize=8, color=C["grey"])
    # up-convolutions: from the bottleneck to the deepest decoder stage, then upwards
    xb = 0.4 + (len(enc) - 1) * 0.9 + 2.3
    yb_ = y0 - (len(enc) - 1) * dy + 0.37
    lvl0 = len(enc) - 2
    ax.annotate("", xy=(11.0 - lvl0 * 0.9 + 1.25, y0 - lvl0 * dy - 0.02), xytext=(xb, yb_),
                arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.4))
    for d in range(len(dec) - 1):
        lv = len(enc) - 2 - d
        ax.annotate("", xy=(11.0 - (lv - 1) * 0.9 + 1.25, y0 - (lv - 1) * dy - 0.02),
                    xytext=(11.0 - lv * 0.9 + 1.25, y0 - lv * dy + 0.77),
                    arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.4))
    ax.text(10.6, y0 - (len(enc) - 1) * dy + 0.2, "2 × 2 up-convolutions (green)", fontsize=8.5,
            color=C["green"])
    final = dec[-1][3]
    ax.text(12.3, y0 + 1.0, f"output: {final} × {final} × 2 (1 × 1 conv)", ha="center",
            fontsize=10, color=C["red"])
    ax.text(1.5, y0 + 1.0, f"input: {enc[0][0]} × {enc[0][0]} × 1", ha="center", fontsize=10)
    ax.set_xlim(0, 14.2); ax.set_ylim(0.3, 7.4); ax.axis("off")
    assert final == 388
    png = save_mpl(fig, ctx, "unet")
    return Figure("unet",
                  "The **U-Net**, with every shape calculated from the output-size formula "
                  "of Lesson 1.1 for the original configuration (unpadded 3 × 3 "
                  "convolutions, 2 × 2 max pooling and 2 × 2 up-convolutions). The encoder "
                  "(left) halves the size and doubles the channels at each stage. U-Net "
                  "does not use unpooling; instead, like FCN, it uses **skip connections**, "
                  "which copy the feature maps of the encoder to the decoder (right), where "
                  "they are **concatenated** with the upsampled maps, so that the spatial "
                  "information of the encoding path is used in the decoding path. As the "
                  "convolutions are unpadded, each encoder map must be cropped to the size of "
                  f"the upsampled map, and a {enc[0][0]} × {enc[0][0]} input gives a "
                  f"{final} × {final} output. The U-Net can be used as a framework with any "
                  "image classification network as the encoder.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    from . import seg_diagrams as SG
    return [fig_segmentation_types(ctx), fig_segmentation_datasets(ctx),
            SG.fig_output_tensor(ctx), fig_segmentation_output(ctx),
            fig_segmentation_metrics(ctx), SG.fig_fcn_architecture(ctx),
            fig_fully_convolutional(ctx), fig_conv_as_matrix(ctx),
            SG.fig_transpose_1d(ctx), fig_transpose_conv(ctx), SG.fig_segnet_architecture(ctx),
            fig_unpooling(ctx), fig_unet(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("3.5 Transpose convolution")
    s = st.select_slider("stride", [1, 2, 3], value=2)
    n = st.slider("input size", 2, 5, 2)
    x = np.arange(1, n * n + 1).reshape(n, n).astype(float)
    k = np.array([[1, 0, 2], [1, 1, 0], [0, 2, 1]], float)
    z, p, out = transpose_as_direct(x, k, s)
    st.write(f"input {n} × {n} → output {out.shape[0]} × {out.shape[0]} "
             f"(= (n − 1)·s + k = {(n - 1) * s + 3})")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    sync_matrix(axes[0], p.astype(int), title="zero-inserted and padded")
    sync_matrix(axes[1], out.astype(int), title="output")
    st.pyplot(fig)
