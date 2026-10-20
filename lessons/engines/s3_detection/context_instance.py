"""Lesson 3.6 — Context and instance segmentation.

Figures
    effective_receptive_field  how far DeepLabV3's decision at one pixel reaches into the
                               image (input gradient, real)
    dilated_conv               receptive field of stacked dilated convolutions, from the
                               impulse response of real convolutions
    dilated_stack              exponential growth for linear parameters; the gridding
                               artefact
    pyramid_scene_parsing      DeepLab's real backbone features pooled at 1, 2, 3 and 6 bins
    dense_crf                  mean-field inference in a fully connected CRF (NumPy),
                               refining DeepLab's real output
    deeplab_framework          atrous backbone, ASPP (rates read from the model), bilinear
                               upsampling, CRF
    instance_vs_semantic       DeepLab vs Mask R-CNN on people, real outputs
    roi_align                  RoIPool vs RoIAlign with torchvision's own operators
    mask_rcnn_output           the real 28 x 28 masks of Mask R-CNN and the final result

Source: reference/legacy_md/lesson_3_6_sem_seg_partB + lesson_3_6_script.
"""
from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt

from .._common import BuildContext, Figure, C, SERIES, save_mpl, load_image, sync_matrix
from .. import _torchdet as D
from .. import _torchvis as TV
from .intro_detection import SHEEP_GT
from .semantic_seg import PAL, _overlay

FIGURES = ["effective_receptive_field", "dilated_conv", "dilated_stack",
           "pyramid_scene_parsing", "dense_crf", "deeplab_framework",
           "instance_vs_semantic", "roi_align", "mask_rcnn_output"]


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _box(ax, b, col, lw=2.0, ls="-", label=None, fs=8):
    ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col,
                               lw=lw, ls=ls))
    if label:
        ax.text(b[0] + 1, b[1] - 2, label, fontsize=fs, va="bottom",
                bbox=dict(fc=col, ec="none", pad=1.0, alpha=0.9))


POINT = (63, 133)          # the centre of the largest sheep, at 400 px


def _sheep400():
    return load_image("sheep", max_side=400)


def _internals():
    return D.deeplab_internals("sheep_400", _sheep400(), POINT, "sheep")


# ---------------------------------------------------------------------------

def fig_effective_receptive_field(ctx: BuildContext) -> Figure:
    img = _sheep400()
    erf = _internals()["erf"]
    e = erf / erf.sum()
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
    axes[0].imshow(img); axes[0].plot(*POINT, "o", color=C["red"], ms=7, mec="white")
    axes[0].set_title("one output pixel, at the centre of a sheep", fontsize=10)
    sm = cv2.GaussianBlur(e, (0, 0), 6)               # smooth before thresholding,
    flat = np.sort(sm.ravel())[::-1]                   # so the 50% region is one area
    thr = flat[np.searchsorted(np.cumsum(flat) / flat.sum(), 0.5)]
    area50 = (sm >= thr).mean()
    g = SHEEP_GT[0] / 2                                  # the sheep's box at 400 px
    inside = np.zeros_like(e, bool)
    inside[int(g[1]):int(g[3]), int(g[0]):int(g[2])] = True
    outside_share = e[~inside].sum()                     # e sums to 1
    density = (e[inside].sum() / inside.mean()) / (outside_share / (~inside).mean())
    axes[1].imshow(img)
    axes[1].imshow(sm, cmap="magma", alpha=0.75)
    axes[1].plot(*POINT, "o", color=C["red"], ms=6, mec="white")
    axes[1].add_patch(plt.Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False,
                                    ec="white", lw=1.3, ls="--"))
    axes[1].set_title(f"|∂ score('sheep') / ∂ input| (smoothed): {outside_share:.0%} of the "
                      "influence\ncomes from outside the sheep's own box (dashed)", fontsize=10)
    for ax in axes:
        _clean(ax)
    png = save_mpl(fig, ctx, "effective_receptive_field")
    return Figure("effective_receptive_field",
                  "**Context**: how much of the image a segmentation network uses to label "
                  "a single pixel. The gradient of DeepLabV3's 'sheep' score at one pixel "
                  "(red) with respect to every input pixel shows which parts of the image "
                  "influence that decision. The influence is strongest on the sheep itself, "
                  f"where it is {density:.0f} times as dense as elsewhere, but as the sheep "
                  "covers only a small part of the image, most of the influence, "
                  f"{outside_share:.0%}, comes from outside the sheep's own bounding box, "
                  "from the surrounding grass and the neighbouring sheep. The input gradient "
                  "is noisy, so its total includes some noise spread over the whole image. "
                  "Pooling and "
                  "downsampling reduce the spatial information in a CNN while the semantic "
                  "information increases, and a large receptive field allows the deep "
                  "neurons of the network to capture the context of the scene. Without "
                  "enough context, segmentation networks make mistakes such as labelling "
                  "a boat on a lake as a car, as a car is unlikely to be found on water.",
                  png=png)


def impulse_rf(dilations, n=33):
    """Receptive field of a stack of 3 x 3 convolutions with the given dilation
    rates, measured from the gradient of the centre output pixel with respect to
    the input: the number of paths from each input pixel to the output."""
    import torch
    import torch.nn.functional as F
    x = torch.zeros(1, 1, n, n, requires_grad=True)
    h = x
    for d in dilations:
        h = F.conv2d(h, torch.ones(1, 1, 3, 3), padding=d, dilation=d)
    h[0, 0, n // 2, n // 2].backward()
    g = x.grad[0, 0].numpy()
    ys, xs = np.nonzero(g)
    return g, (xs.max() - xs.min() + 1) if len(xs) else 0


def fig_dilated_conv(ctx: BuildContext) -> Figure:
    stacks = [([1], "3 × 3 convolution"), ([1, 2], "then 3 × 3, dilation 2"),
              ([1, 2, 4], "then 3 × 3, dilation 4")]
    n = 17
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (ds, title) in zip(axes, stacks):
        g, rf = impulse_rf(ds, n)
        ax.imshow(g > 0, cmap="Greens", vmin=0, vmax=1.6)
        # the input positions sampled by the last layer's kernel
        d = ds[-1]
        c = n // 2
        for dy in (-d, 0, d):
            for dx in (-d, 0, d):
                ax.add_patch(plt.Rectangle((c + dx - 0.5, c + dy - 0.5), 1, 1, fill=False,
                                           ec=C["red"], lw=1.8))
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True); ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.4); _clean(ax)
        ax.set_title(f"{title}\nreceptive field {rf} × {rf}, {9 * len(ds)} weights", fontsize=10)
    rfs = [impulse_rf(ds, n)[1] for ds, _ in stacks]
    assert rfs == [3, 7, 15], rfs
    png = save_mpl(fig, ctx, "dilated_conv")
    return Figure("dilated_conv",
                  "**Dilated** (or atrous) convolution, measured by passing gradients "
                  "through real convolution layers (green: the input pixels that affect "
                  "the centre output; red: the samples taken by the last kernel). In a "
                  "dilated convolution the kernel's inputs are spaced out, so a 3 × 3 kernel "
                  "with a dilation of 2 covers a 5 × 5 region with the same 9 weights. A "
                  "3 × 3 convolution has a receptive field of 3 × 3; following it with a "
                  "dilation-2 convolution gives 7 × 7, and a further dilation-4 "
                  "convolution gives 15 × 15. Dilated convolution is not to be confused "
                  "with the transpose convolution of Lesson 3.5: no zeros are inserted into "
                  "the feature map, and its size is unchanged.",
                  png=png)


def fig_dilated_stack(ctx: BuildContext) -> Figure:
    L = np.arange(1, 8)
    reg = 2 * L + 1
    dil = 2 ** (L + 1) - 1
    for l in (1, 2, 3, 4):
        assert impulse_rf([2 ** k for k in range(l)], 65)[1] == dil[l - 1]
        assert impulse_rf([1] * l, 65)[1] == reg[l - 1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    ax = axes[0]
    ax.semilogy(L, reg, "-o", color=C["blue"], lw=2, label="3 × 3, dilation 1 at every layer")
    ax.semilogy(L, dil, "-o", color=C["green"], lw=2, label="3 × 3, dilation 1, 2, 4, 8, …")
    for l, r in zip(L, dil):
        ax.text(l, r * 1.25, f"{r}", ha="center", fontsize=8.5, color=C["green"])
    ax.set_xlabel(f"number of layers (9 weights each: the same parameters)")
    ax.set_ylabel("receptive field (pixels, log scale)"); ax.legend(fontsize=8.5)
    ax.set_title("(a) exponential growth for a linear number of parameters", fontsize=10)
    for ax, ds, title in [(axes[1], [2, 2, 2], "(b) dilation 2, 2, 2: gridding"),
                          (axes[2], [1, 2, 4], "(c) dilation 1, 2, 4: no gaps")]:
        g, rf = impulse_rf(ds, 21)
        ax.imshow(g, cmap="Greens"); _clean(ax)
        ax.set_title(f"{title}\n(input pixels used by one output, {rf} × {rf})", fontsize=10)
    png = save_mpl(fig, ctx, "dilated_stack")
    return Figure("dilated_stack",
                  "(a) When the dilation is doubled at each layer, the receptive field grows "
                  "exponentially, while the number of parameters grows only linearly, as each "
                  "layer still has 9 weights; with a dilation of 1 at every layer the "
                  "receptive field grows only linearly. All values are measured from the "
                  "gradients of real convolutions. The ordinary convolutions of a network "
                  "such as VGG can therefore be replaced with dilated convolutions to widen "
                  "the receptive field at no additional cost. (b, c) The rates must be "
                  "chosen with care: three layers with the same dilation of 2 use only every "
                  "second input pixel, a **gridding** artefact, whereas the rates 1, 2 and 4 "
                  "cover every pixel (the shade shows the number of paths from each input "
                  "pixel to the output).",
                  png=png)


def fig_pyramid_scene_parsing(ctx: BuildContext) -> Figure:
    import torch
    import torch.nn.functional as F
    img = _sheep400()
    f = torch.tensor(_internals()["feats"].astype(np.float32))
    bins = (1, 2, 3, 6)
    fig, axes = plt.subplots(1, 6, figsize=(16.5, 2.6), gridspec_kw=dict(width_ratios=[1.3] + [1] * 5))
    axes[0].imshow(img); axes[0].set_title("input", fontsize=10)
    ch = int(np.argmax(f[:, 10:22, 3:15].mean((1, 2)) - f.mean((1, 2))))  # a 'sheep' channel
    axes[1].imshow(f[ch].numpy(), cmap="magma")
    axes[1].set_title(f"backbone output, channel {ch}\n{f.shape[2]} × {f.shape[1]} × "
                      f"{f.shape[0]}", fontsize=9.5)
    vmax = float(f[ch].max())
    for ax, b in zip(axes[2:], bins):
        p = F.adaptive_avg_pool2d(f[None, ch:ch + 1], b)[0, 0]
        ax.imshow(p.numpy(), cmap="magma", vmin=0, vmax=float(p.max()) or 1,
                  interpolation="nearest")
        ax.set_title(f"pooled to {b} × {b}", fontsize=9.5)
    for ax in axes:
        _clean(ax)
    png = save_mpl(fig, ctx, "pyramid_scene_parsing")
    c = f.shape[0]
    return Figure("pyramid_scene_parsing",
                  "**Pyramid scene parsing** (PSPNet), applied to the real output feature map "
                  "of DeepLabV3's ResNet-50 backbone, in which some convolutions are dilated "
                  f"so that the map is only 8 times smaller than the image ({f.shape[2]} × "
                  f"{f.shape[1]} × {c}). As in spatial pyramid pooling, the feature map is "
                  "average-pooled to 1 × 1, 2 × 2, 3 × 3 and 6 × 6 bins (one channel that "
                  "responds to the sheep is shown), so that information "
                  "about the features at different scales and positions is retained: the "
                  "1 × 1 level summarises the whole scene and the 6 × 6 level its regions. In "
                  "PSPNet each pooled map is reduced in depth by a 1 × 1 convolution (to "
                  f"512 channels), upsampled, and concatenated with the original features "
                  f"({c} + 4 × 512 = {c + 4 * 512} channels), and a final convolution "
                  "produces the pixel-level prediction. No decoder or skip connections are "
                  "required.",
                  png=png)


def dense_crf(img, P, iters=5, w_app=2.0, th_a=4.0, th_b=8.0, w_sm=1.0, th_g=1.0):
    """Mean-field inference in a fully connected CRF (Krähenbühl and Koltun,
    2011), computed exactly (every pair of pixels) for a small image.

    Unary: -log of the network's class probabilities. Pairwise: a Potts
    penalty for different labels, weighted by an appearance kernel (nearby
    pixels of similar colour) and a smoothness kernel (nearby pixels)."""
    H, W, _ = img.shape
    ys, xs = np.mgrid[0:H, 0:W]
    pos = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    col = img.reshape(-1, 3).astype(np.float32)
    d_pos = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    d_col = ((col[:, None, :] - col[None, :, :]) ** 2).sum(-1)
    Kp = (w_app * np.exp(-d_pos / (2 * th_a ** 2) - d_col / (2 * th_b ** 2))
          + w_sm * np.exp(-d_pos / (2 * th_g ** 2)))
    np.fill_diagonal(Kp, 0)
    U = -np.log(np.clip(P.reshape(P.shape[0], -1).T, 1e-6, 1))
    Q = np.exp(-U); Q /= Q.sum(1, keepdims=True)
    out = [Q.T.reshape(P.shape)]
    for _ in range(iters):
        msg = Kp @ Q
        E = U + (msg.sum(1, keepdims=True) - msg)
        Q = np.exp(-(E - E.min(1, keepdims=True))); Q /= Q.sum(1, keepdims=True)
        out.append(Q.T.reshape(P.shape))
    return out


CRF_SIZE = (100, 67)
CRF_W = dict(w_app=2.0, th_a=4.0, th_b=8.0, w_sm=1.0, th_g=1.0)


def crf_inputs():
    """The reduced image and DeepLabV3's class probabilities on it (the CRF's unary)."""
    img = _sheep400()
    L8 = D.semantic_scores("sheep_400", img)["logits8"]
    W_, H_ = CRF_SIZE
    small = cv2.resize(img, (W_, H_), interpolation=cv2.INTER_AREA)
    lg = np.stack([cv2.resize(L8[k], (W_, H_), interpolation=cv2.INTER_LINEAR)
                   for k in range(L8.shape[0])])
    P = np.exp(lg - lg.max(0)); P /= P.sum(0)
    return small, P


def crf_result():
    """(small image, mean-field iterates Q[0..5], Mask R-CNN reference mask)."""
    small, P = crf_inputs()
    W_, H_ = CRF_SIZE
    res = TV.cached("dense_crf_sheep", dict(size=[W_, H_], iters=5, w=[2, 4, 8, 1, 1]),
                    lambda: dict(q=np.stack(dense_crf(small, P)).astype(np.float16)))
    img800 = load_image("sheep", max_side=800)
    ref = cv2.resize(D.instance_masks("sheep", img800)["masks"].any(0).astype(np.float32),
                     (W_, H_), interpolation=cv2.INTER_AREA) > 0.5
    return small, res["q"].astype(np.float32), ref


def fig_dense_crf(ctx: BuildContext) -> Figure:
    cats = D.categories("deeplab"); sheep = cats.index("sheep")
    W_, H_ = CRF_SIZE
    small, Q, ref = crf_result()
    iou = lambda m: (m & ref).sum() / (m | ref).sum()
    before, after = iou(Q[0].argmax(0) == sheep), iou(Q[-1].argmax(0) == sheep)
    assert after > before, (before, after)
    fig, axes = plt.subplots(1, 4, figsize=(16, 2.9))
    axes[0].imshow(small); axes[0].set_title(f"input, reduced to {W_} × {H_}", fontsize=10)
    axes[1].imshow(Q[0][sheep], cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title(f"DeepLabV3: p(sheep)\nIoU {before:.2f}", fontsize=10)
    axes[2].imshow(Q[-1][sheep], cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title(f"after 5 mean-field iterations of the CRF\nIoU {after:.2f}", fontsize=10)
    axes[3].imshow(ref, cmap="gray"); axes[3].set_title("reference: Mask R-CNN's sheep masks",
                                                        fontsize=10)
    for ax in axes:
        _clean(ax); ax.set_ylim(H_ * 0.7, H_ * 0.15)
    png = save_mpl(fig, ctx, "dense_crf")
    return Figure("dense_crf",
                  "A **conditional random field** (CRF) explicitly models the relationships "
                  "between pixels: the label of each pixel is predicted from the network's "
                  "output (the unary term) and from every other pixel, with pairs of pixels "
                  "that are close together and similar in colour encouraged to take the "
                  "same label (the pairwise term). The model is a fully connected graph, and "
                  "learning it at the pixel level is computationally intractable, but "
                  "approximate inference is efficient with mean-field iterations, as in the "
                  "dense CRF of Krähenbühl and Koltun. Here the inference is computed "
                  "exactly, for every pair of pixels of a reduced image, starting from "
                  "DeepLabV3's real class probabilities: the blurred boundaries of the "
                  "network's output are sharpened to follow the colour edges of the sheep, "
                  f"and the IoU with Mask R-CNN's sheep masks, used as a reference, rises from "
                  f"{before:.2f} to {after:.2f}.",
                  png=png)


def fig_deeplab_framework(ctx: BuildContext) -> Figure:
    r = _internals()
    rates, ch = [int(v) for v in r["rates"]], int(r["aspp_out"])
    c, h, w = r["feats"].shape
    fig, ax = plt.subplots(figsize=(15, 4.6))

    def box(x, y, wd, ht, txt, col, fs=9):
        ax.add_patch(plt.Rectangle((x, y), wd, ht, fc=col, alpha=0.55, ec=C["ink"]))
        ax.text(x + wd / 2, y + ht / 2, txt, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.2))
    box(0, 1.6, 1.6, 1.2, f"image\n{w * 8} × {h * 8}", C["light"])
    box(2.2, 1.4, 2.6, 1.6, f"ResNet-50, last two stages\nwith dilated convolutions\n"
        f"→ {w} × {h} × {c}\n(stride 8, not 32)", C["blue"], 8.5)
    arrow(1.6, 2.2, 2.2, 2.2)
    branches = ["1 × 1 conv"] + [f"3 × 3 conv,\ndilation {d}" for d in rates] + ["image pooling"]
    for k, t in enumerate(branches):
        y = 3.9 - k * 0.9
        box(5.6, y, 1.9, 0.7, t, C["green"], 8.5)
        arrow(4.8, 2.2, 5.6, y + 0.35)
        arrow(7.5, y + 0.35, 8.2, 2.2)
    ax.text(6.55, 4.75, "atrous spatial pyramid pooling (ASPP)", ha="center", fontsize=9.5,
            color=C["green"])
    box(8.2, 1.7, 1.7, 1.0, f"concatenate,\n1 × 1 conv → {ch}", C["green"], 8.5)
    box(10.4, 1.7, 1.6, 1.0, "class scores\n(21 classes)", C["purple"], 8.5)
    arrow(9.9, 2.2, 10.4, 2.2)
    box(12.5, 1.7, 1.7, 1.0, "bilinear\nupsampling × 8", C["sky"], 8.5)
    arrow(12.0, 2.2, 12.5, 2.2)
    box(14.7, 1.7, 1.5, 1.0, "fully connected\nCRF (DeepLab\nv1 and v2)", C["orange"], 8)
    arrow(14.2, 2.2, 14.7, 2.2)
    ax.set_xlim(-0.2, 16.4); ax.set_ylim(-0.3, 5.0); ax.axis("off")
    png = save_mpl(fig, ctx, "deeplab_framework")
    return Figure("deeplab_framework",
                  "The **DeepLab** framework, with the values of torchvision's DeepLabV3 "
                  "model read from the model itself. The backbone, here ResNet-50, has "
                  "dilated convolutions in place of the strided convolutions of its last "
                  "two stages, which reduces the effective downsampling to a factor of 8 "
                  "without increasing the number of parameters. A spatial pyramid pooling "
                  "stage built from dilated convolutions, **atrous spatial pyramid "
                  f"pooling** with dilation rates {rates[0]}, {rates[1]} and {rates[2]} plus "
                  "a 1 × 1 convolution and image-level pooling, handles objects of "
                  "different scales. The output is upsampled by bilinear interpolation, with "
                  "no learned deconvolution stage and no explicit skip connections. In the "
                  "earlier versions of DeepLab, a fully connected CRF was applied as "
                  "post-processing; the CRF can also be trained together with the network.",
                  png=png)


def fig_instance_vs_semantic(ctx: BuildContext) -> Figure:
    img = load_image("friends", max_side=800)
    cats_s, cats_d = D.categories("deeplab"), D.categories("faster_rcnn")
    sem = D.semantic("friends", img)["labels"]
    m = D.mask_head_outputs("friends", img)
    person = m["labels"] == cats_d.index("person")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].imshow(img); axes[0].set_title("image", fontsize=10.5)
    axes[1].imshow(_overlay(img, sem))
    axes[1].set_title("semantic segmentation (DeepLabV3):\nall people share one label, "
                      "'person'", fontsize=10)
    over = img.astype(float).copy()
    for i, mk in enumerate(m["full"][person]):
        b = mk > 127
        col = np.array(plt.matplotlib.colors.to_rgb(SERIES[i % len(SERIES)])) * 255
        over[b] = 0.4 * over[b] + 0.6 * col
    axes[2].imshow(over.astype(np.uint8))
    axes[2].set_title(f"instance segmentation (Mask R-CNN):\n{person.sum()} separate people",
                      fontsize=10)
    for ax in axes:
        _clean(ax)
    png = save_mpl(fig, ctx, "instance_vs_semantic")
    return Figure("instance_vs_semantic",
                  "**Instance segmentation** combines object detection and semantic "
                  "segmentation, to identify the different instances of the same class in a "
                  "scene. Semantic segmentation (centre) labels every person pixel 'person', "
                  "so the people in contact with each other form one region. Earlier "
                  "approaches performed segmentation before recognition: segmentation "
                  "proposals were classified, or a semantic segmentation was cut into "
                  "instances with object proposals, both of which are cumbersome and cannot "
                  "always be trained end to end. Mask R-CNN (right) predicts the class "
                  "labels and a binary mask for each detected object together, as a small "
                  "addition to the Faster R-CNN detector.",
                  png=png)


def fig_roi_align(ctx: BuildContext) -> Figure:
    import torch
    from torchvision.ops import roi_pool, roi_align
    rng = np.random.default_rng(4)
    fm = torch.tensor(rng.integers(0, 10, (8, 8)).astype(np.float32))
    box = torch.tensor([[0, 1.4, 1.7, 6.2, 5.6]])          # (batch, x1, y1, x2, y2)
    p = roi_pool(fm[None, None], box, output_size=2, spatial_scale=1.0)[0, 0].numpy()
    a = roi_align(fm[None, None], box, output_size=2, spatial_scale=1.0, sampling_ratio=2,
                  aligned=True)[0, 0].numpy()
    x1, y1, x2, y2 = box[0, 1:].tolist()
    qx1, qy1, qx2, qy2 = round(x1), round(y1), round(x2), round(y2)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), gridspec_kw=dict(width_ratios=[1.3, 0.7, 1.3, 0.7]))
    for ax in (axes[0], axes[2]):
        sync_matrix(ax, fm.numpy().astype(int), cmap="Blues", fontsize=8.5)
    ax = axes[0]
    ax.add_patch(plt.Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1, y2 - y1, fill=False, ec=C["grey"],
                               lw=1.4, ls="--"))
    ax.add_patch(plt.Rectangle((qx1 - 0.5, qy1 - 0.5), qx2 - qx1 + 1, qy2 - qy1 + 1, fill=False,
                               ec=C["red"], lw=2.4))
    mx, my = (qx1 + qx2 + 1) / 2, (qy1 + qy2 + 1) / 2
    ax.axvline(np.floor(mx) - 0.5, ymin=1 - (qy2 + 1) / 8, ymax=1 - qy1 / 8, color=C["red"], lw=1.4)
    ax.axhline(np.floor(my) - 0.5, xmin=qx1 / 8, xmax=(qx2 + 1) / 8, color=C["red"], lw=1.4)
    ax.set_title("(a) RoIPool: the box (dashed) is rounded to\nwhole cells, then split into "
                 "2 × 2 bins", fontsize=9.5)
    sync_matrix(axes[1], p, fmt="{:.0f}", title="max in each bin", cmap="Reds")
    ax = axes[2]
    ax.add_patch(plt.Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1, y2 - y1, fill=False, ec=C["green"],
                               lw=2.4))
    bw, bh = (x2 - x1) / 2, (y2 - y1) / 2
    for i in range(2):
        for j in range(2):
            for si in range(2):
                for sj in range(2):
                    sx = x1 + j * bw + (sj + 0.5) * bw / 2
                    sy = y1 + i * bh + (si + 0.5) * bh / 2
                    ax.plot(sx - 0.5, sy - 0.5, "o", color=C["green"], ms=6, mec="white")
    ax.axvline(x1 + bw - 0.5, ymin=1 - (y2) / 8, ymax=1 - y1 / 8, color=C["green"], lw=1.2)
    ax.axhline(y1 + bh - 0.5, xmin=x1 / 8, xmax=x2 / 8, color=C["green"], lw=1.2)
    ax.set_title("(b) RoIAlign: no rounding; 4 points per bin,\neach bilinearly interpolated",
                 fontsize=9.5)
    sync_matrix(axes[3], a, fmt="{:.2f}", title="mean of the points", cmap="Greens")
    png = save_mpl(fig, ctx, "roi_align")
    return Figure("roi_align",
                  "**RoIPool and RoIAlign**, computed with torchvision's own operators on a "
                  "small feature map and a proposal with fractional coordinates (2 × 2 bins "
                  "are shown; 7 × 7 are used in practice). (a) In RoI pooling, as in Faster "
                  "R-CNN, the proposal is quantised to the cells of the feature map and the "
                  "maximum is taken in each bin, so the pooled features no longer correspond "
                  "exactly to the proposed region. (b) In Mask R-CNN, RoI pooling is replaced "
                  "by **RoIAlign**: the proposal is not quantised, and the feature map is "
                  "interpolated bilinearly at regularly spaced points in each bin, which "
                  "preserves the spatial correspondence between the features and the region. "
                  "This removal of quantisation was found to be the key to accurate mask "
                  "prediction.",
                  png=png)


def fig_mask_rcnn_output(ctx: BuildContext) -> Figure:
    img = load_image("friends", max_side=800)
    cats = D.categories("faster_rcnn")
    m = D.mask_head_outputs("friends", img)
    i = int(np.argmax((m["boxes"][:, 2] - m["boxes"][:, 0]) * (m["labels"] == 1)))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), gridspec_kw=dict(width_ratios=[1.6, 0.8, 0.9]))
    over = img.astype(float).copy()
    for k, mk in enumerate(m["full"]):
        b = mk > 127
        col = np.array(plt.matplotlib.colors.to_rgb(SERIES[k % len(SERIES)])) * 255
        over[b] = 0.45 * over[b] + 0.55 * col
    axes[0].imshow(over.astype(np.uint8))
    for k, (bx, l, s) in enumerate(zip(m["boxes"], m["labels"], m["scores"])):
        _box(axes[0], bx, SERIES[k % len(SERIES)], lw=1.4, label=f"{cats[l]} {s:.2f}", fs=7)
    axes[0].set_title("(a) Mask R-CNN: class, box and mask for each object", fontsize=10)
    small = m["small"][i].astype(np.float32)
    axes[1].imshow(small, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title(f"(b) the mask head's output for\none person: {small.shape[1]} × "
                      f"{small.shape[0]}", fontsize=10)
    x1, y1, x2, y2 = m["boxes"][i].astype(int)
    axes[2].imshow(img[y1:y2, x1:x2])
    axes[2].imshow(m["full"][i][y1:y2, x1:x2] / 255, cmap="viridis", alpha=0.55, vmin=0, vmax=1)
    axes[2].set_title(f"(c) resized into the {x2 - x1} × {y2 - y1} box", fontsize=10)
    for ax in axes:
        _clean(ax)
    png = save_mpl(fig, ctx, "mask_rcnn_output")
    return Figure("mask_rcnn_output",
                  "The output of a real Mask R-CNN. (a) For each detected object a class, a "
                  "bounding box and a mask are output. The mask is predicted for each region "
                  "of interest by a small fully convolutional network, which retains the "
                  "spatial layout of the RoIAligned features. (b) The mask head outputs one "
                  f"{small.shape[1]} × {small.shape[0]} mask for each class, of which only "
                  "the mask of the class predicted by the classification branch is used. "
                  "(c) The low-resolution mask is resized into the predicted box and "
                  "thresholded, which is why the outlines of the masks are smooth and some "
                  "fine detail is lost. The mask branch adds little computation to the "
                  "Faster R-CNN detector.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_effective_receptive_field(ctx), fig_dilated_conv(ctx), fig_dilated_stack(ctx),
            fig_pyramid_scene_parsing(ctx), fig_dense_crf(ctx), fig_deeplab_framework(ctx),
            fig_instance_vs_semantic(ctx), fig_roi_align(ctx), fig_mask_rcnn_output(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("3.6 Dilated convolution")
    rates = st.multiselect("dilation rates of a stack of 3 × 3 convolutions", [1, 2, 3, 4, 8],
                           default=[1, 2, 4])
    if rates:
        g, rf = impulse_rf(rates, 41)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(g, cmap="Greens"); _clean(ax)
        st.pyplot(fig)
        st.write(f"receptive field {rf} × {rf} with {9 * len(rates)} weights")
