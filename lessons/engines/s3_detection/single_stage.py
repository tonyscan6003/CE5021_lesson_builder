"""Lesson 3.4 — Single-stage object detection.

Figures
    feature_location      a real VGG-16 conv5_3 channel locates the car: position is in
                          the feature map, extent in its depth
    prediction_head       a 3 x 3 head sliding over the grid; one cell's receptive field
                          and its output vector
    anchor_assignment     RetinaNet's real anchors matched to the sheep: positives,
                          negatives, ignored -- the imbalance counted
    box_parameterisation  t_x, t_y, t_w, t_h for a real anchor and a sheep, and back
    detection_loss        smooth L1 against L1 and L2
    focal_loss            CE vs FL against p_t, and the loss of the real anchors split
                          into positives and easy negatives for each gamma
    fpn_levels            RetinaNet's real P3-P7 feature maps; the level of each match
    ssd_multiscale        SSD300's 6 feature maps and 8,732 default boxes (torchvision)
    (the slide diagrams -- detection_system, head_sliding, gt_assignment,
     box_dimensions, scale_grids, ssd_architecture and
     fpn_detector -- are in single_stage_diagrams.py)
    depthwise_separable   standard vs depthwise + pointwise: parameters and MACs
    nms_steps             non-maximum suppression stepped through real raw detections
                          (interactive)

Real networks via engines/_torchvis.py and engines/_torchdet.py (cached in
data/_torch/); costs from engines/_arch.py; NMS from engines/_detmetrics.py.

Source: reference/legacy_md/lesson_3_4_obj_det_partB_single_stage + lesson_3_4_script.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html, load_image,
                       to_data_uri, slider_frames_layout)
from .. import _arch as A
from .. import _detmetrics as M
from .. import _torchdet as D
from .. import _torchvis as TV
from .intro_detection import SHEEP_GT

FIGURES = ["feature_location", "detection_system", "head_sliding", "prediction_head",
           "gt_assignment", "anchor_assignment", "box_dimensions",
           "box_parameterisation", "detection_loss", "focal_loss", "scale_grids", "fpn_levels",
           "ssd_architecture", "ssd_multiscale", "fpn_detector", "depthwise_separable",
           "nms_steps"]


def _box(ax, b, col, lw=2.0, ls="-", label=None, fs=8, alpha=1.0):
    ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col,
                               lw=lw, ls=ls, alpha=alpha))
    if label:
        ax.text(b[0] + 1, b[1] - 2, label, fontsize=fs, va="bottom",
                bbox=dict(fc=col, ec="none", pad=1.0, alpha=0.9))


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _sheep():
    return load_image("sheep", max_side=800)


def _retina():
    return D.retinanet_internals("sheep", _sheep(), SHEEP_GT, "sheep")


# ---------------------------------------------------------------------------

def fig_feature_location(ctx: BuildContext) -> Figure:
    img = load_image("minivan", max_side=800)
    det = D.detect("minivan", img)
    car = det["boxes"][int(np.argmax((det["boxes"][:, 2] - det["boxes"][:, 0]) *
                                     (det["boxes"][:, 3] - det["boxes"][:, 1]) * (det["scores"] > 0.9)))]
    def run():
        import torch
        with torch.no_grad():
            f = TV.features(TV.to_tensor(img), ["conv5_3"])["conv5_3"][0].cpu().numpy()
        return dict(f=f.astype(np.float16))
    f = TV.cached("conv5_minivan", dict(img="minivan", shape=img.shape), run)["f"].astype(np.float32)
    s = 16
    inside = np.zeros(f.shape[1:], bool)
    inside[int(car[1] // s):int(car[3] // s) + 1, int(car[0] // s):int(car[2] // s) + 1] = True
    score = f[:, inside].mean(1) - f[:, ~inside].mean(1)
    ch = int(np.argmax(score))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    axes[0].imshow(img); _clean(axes[0])
    _box(axes[0], car, C["green"], lw=2.2, label="car (Faster R-CNN)")
    axes[0].set_title("input image", fontsize=10)
    axes[1].imshow(f[ch], cmap="viridis"); _clean(axes[1])
    _box(axes[1], np.array(car) / s - 0.5, "white", lw=1.4, ls="--")
    axes[1].set_title(f"VGG-16 conv5_3, channel {ch} of 512 ({f.shape[2]} × {f.shape[1]})",
                      fontsize=10)
    zero = (f == 0).mean()
    ccy, ccx = int((car[1] + car[3]) / 2 // s), int((car[0] + car[2]) / 2 // s)
    active = int((f[:, ccy, ccx] > 0).sum())
    png = save_mpl(fig, ctx, "feature_location")
    return Figure("feature_location",
                  "The feature maps of an image classifier contain both spatial and semantic "
                  "information. The photograph is passed once through the convolution "
                  "layers of VGG-16, trained on ImageNet classification, and one channel of "
                  "the conv5_3 feature map is shown: the channel whose activation is most "
                  "concentrated on the car (dashed). The position of the activation indicates "
                  f"approximately where the object is, but only at the resolution of the map: "
                  f"each cell covers {s} × {s} pixels of the image. The map is also sparse: "
                  f"after the ReLU, {zero:.0%} of its values are zero, and at the cell at the "
                  f"centre of the car only {active} of the 512 channels are active. A detector "
                  "therefore does not read the box directly from the feature map; a prediction "
                  "head is trained to regress the box from these features, and accurate "
                  "localisation, particularly of small objects, needs higher-resolution feature "
                  "maps, which is the purpose of the multi-scale detectors described later in "
                  "this lesson. Unlike a sliding window, the image is processed only once.",
                  png=png)


def fig_prediction_head(ctx: BuildContext) -> Figure:
    img = _sheep()
    s, rf = 16, TV.RF["conv5_3"] + 2 * 16           # conv5_3, plus the 3 x 3 head
    gh, gw = img.shape[0] // s, img.shape[1] // s
    B, Cn = 2, 81                                    # as on the slide: 2 boxes, COCO
    cy, cx = 16, 9                                   # a cell on the large sheep
    fig = plt.figure(figsize=(14.5, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax.imshow(img); _clean(ax)
    for x in range(0, img.shape[1] + 1, s):
        ax.axvline(x, color="white", lw=0.3, alpha=0.6)
    for y in range(0, img.shape[0] + 1, s):
        ax.axhline(y, color="white", lw=0.3, alpha=0.6)
    ax.add_patch(plt.Rectangle(((cx - 1) * s, (cy - 1) * s), 3 * s, 3 * s, fill=False,
                               ec=C["orange"], lw=2.4))
    ax.add_patch(plt.Rectangle((cx * s, cy * s), s, s, fc=C["red"], ec="none", alpha=0.8))
    c = (cx + 0.5) * s, (cy + 0.5) * s
    ax.add_patch(plt.Rectangle((c[0] - rf / 2, c[1] - rf / 2), rf, rf, fill=False, ec=C["red"],
                               lw=1.8, ls="--"))
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0] * 0.8, img.shape[0] * 0.15)
    ax.set_title(f"(a) {gw} × {gh} grid of conv5_3 cells (stride {s}); a 3 × 3 head "
                 f"(orange) at one cell (red),\nwhose receptive field (dashed) is {rf} × {rf} "
                 "pixels", fontsize=10)
    ax = fig.add_subplot(gs[1]); ax.axis("off")
    # the slide's output vector: [4 coordinates][1][4 coordinates][1][81 classes]
    segs = []
    for b in range(B):
        segs += [("4 × coordinates\n(t_x, t_y, t_w, t_h)", 4, "#e1d5e7"), ("1 ×", 1, "#d5e8d4")]
    segs.append((f"{Cn} × class\n(COCO: 80 + background)", 9, "#dae8fc"))
    tot = sum(n for _, n, _ in segs)
    x = 0.02
    for k, (lab, n, col) in enumerate(segs):
        w = 0.96 * n / tot
        ax.add_patch(plt.Rectangle((x, 0.5), w, 0.13, fc=col, ec=C["grey"], lw=0.9,
                                   transform=ax.transAxes))
        ax.text(x + w / 2, 0.565, lab.split("\n")[0], ha="center", va="center", fontsize=8.5,
                transform=ax.transAxes)
        if "\n" in lab:
            ax.text(x + w / 2, 0.47, lab.split("\n")[1], ha="center", va="top", fontsize=7.5,
                    transform=ax.transAxes, color=C["grey"])
        if n == 1:
            ax.annotate("confidence", xy=(x + w / 2, 0.64), xytext=(x + w / 2 - 0.05, 0.78),
                        xycoords="axes fraction", fontsize=8.5, ha="center",
                        arrowprops=dict(arrowstyle="->", lw=0.9))
        x += w
    ax.text(0.02, 0.28, f"one cell: {B} × (4 + 1) + {Cn} = {B * 5 + Cn} values\n"
            f"output tensor: {gw} × {gh} × {B * 5 + Cn}", fontsize=10.5, color=C["red"],
            transform=ax.transAxes, va="top")
    ax.set_title(f"(b) the output vector at each feature-map position ({B} boxes predicted)",
                 fontsize=10)
    png = save_mpl(fig, ctx, "prediction_head")
    return Figure("prediction_head",
                  "The **prediction head** of a basic single-stage detector, a small "
                  "convolutional network, usually with 3 × 3 kernels, applied at every "
                  "position of the output feature map of the backbone. (a) For VGG-16 "
                  f"conv5_3 the grid has a stride of {s} pixels, and each cell has a "
                  f"receptive field of {rf} pixels, calculated with the formula of Lesson "
                  "1.4, which is large enough to contain the whole sheep, so the features at "
                  "that cell are computed from the whole object and a small kernel is "
                  "sufficient for the head; the box itself is learned as a regression from "
                  "the anchor. (b) At each cell the head predicts, for each "
                  "anchor box, four box values relative to the anchor and the cell position "
                  "and a confidence, and, usually once per cell, the class probabilities: "
                  f"here {B} boxes and the {Cn} classes of COCO (80 object classes and "
                  "background). Because the class is predicted once per cell, both boxes "
                  "receive the same class, which can be wrong when two different objects "
                  "share a cell. The predicted box is relative to "
                  "the cell and must be corrected by the coarse position of the cell to "
                  "obtain coordinates in the image. The confidence is sometimes replaced by "
                  "a background class.",
                  png=png)


def fig_anchor_assignment(ctx: BuildContext) -> Figure:
    img = _sheep()
    r = _retina()
    lab, an = r["label"], r["anchors"]
    npos, nneg, nign = int((lab == 1).sum()), int((lab == 0).sum()), int((lab == -1).sum())
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    ax.imshow(img); _clean(ax)
    for g in SHEEP_GT:
        _box(ax, g, "white", lw=1.3, ls="--")
    for b in an[lab == 1]:
        _box(ax, b, C["green"], lw=0.8)
    ign = np.where(lab == -1)[0]
    for b in an[ign[::max(1, len(ign) // 60)]]:
        _box(ax, b, C["orange"], lw=0.6, alpha=0.8)
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0] * 0.72, img.shape[0] * 0.12)
    ax.set_title(f"(a) positive anchors (green, IoU ≥ 0.5) and a sample of ignored "
                 "anchors (orange, 0.4–0.5)", fontsize=10)
    ax = axes[1]
    ax.bar(["positive\nIoU ≥ 0.5", "ignored\n0.4–0.5", "negative\nIoU < 0.4"],
           [npos, nign, nneg], color=[C["green"], C["orange"], C["grey"]])
    ax.set_yscale("log")
    for k, v in enumerate([npos, nign, nneg]):
        ax.text(k, v * 1.15, f"{v:,}", ha="center", fontsize=10)
    ax.set_ylabel("anchors (log scale)")
    ax.set_title(f"(b) {len(lab):,} anchors: {nneg / npos:,.0f} negatives per positive",
                 fontsize=10)
    png = save_mpl(fig, ctx, "anchor_assignment")
    return Figure("anchor_assignment",
                  "**Anchor boxes** and ground-truth assignment in a real single-stage "
                  "detector: RetinaNet from torchvision, applied to the sheep photograph. "
                  f"RetinaNet places 9 anchors (3 sizes × 3 aspect ratios) at every cell of "
                  f"five feature maps, {len(lab):,} anchors for this image. Each anchor is "
                  "compared with every ground-truth box: an anchor with an IoU of at least "
                  "0.5 with an object is a **positive** example and is trained to predict "
                  "that object's box; an anchor with an IoU below 0.4 is a **negative** "
                  "example for the confidence; anchors in between are ignored. Several "
                  "anchors, in the same or neighbouring cells, can be positive for one "
                  f"object, which increases the positive examples to {npos}, but there are "
                  f"still {nneg / npos:,.0f} negatives for every positive. This **imbalance** "
                  "is the main difficulty in training a single-stage detector. Anchors act "
                  "as priors, which speeds up training, but their sizes are set in advance, "
                  "which can bias the detector and make small objects difficult to learn.",
                  png=png)


def encode(anchor, gt):
    """t_x = (x - x_a)/w_a, t_y = (y - y_a)/h_a, t_w = log(w/w_a), t_h = log(h/h_a)."""
    ax_, ay = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
    aw, ah = anchor[2] - anchor[0], anchor[3] - anchor[1]
    x, y = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    w, h = gt[2] - gt[0], gt[3] - gt[1]
    return np.array([(x - ax_) / aw, (y - ay) / ah, np.log(w / aw), np.log(h / ah)]), \
        (ax_, ay, aw, ah), (x, y, w, h)


def decode(anchor, t):
    ax_, ay = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
    aw, ah = anchor[2] - anchor[0], anchor[3] - anchor[1]
    x, y, w, h = ax_ + t[0] * aw, ay + t[1] * ah, aw * np.exp(t[2]), ah * np.exp(t[3])
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])


def fig_box_parameterisation(ctx: BuildContext) -> Figure:
    img = _sheep()
    r = _retina()
    g = SHEEP_GT[0]
    v = D.iou(r["anchors"], g[None])[:, 0]
    cand = np.where(v >= 0.5)[0]                     # anchors positive for THIS sheep
    i = cand[int(np.argmin(v[cand]))]                # the least well placed of them
    a = r["anchors"][i]
    t, (xa, ya, wa, ha), (x, y, w, h) = encode(a, g)
    back = decode(a, t)
    assert np.allclose(back, g, atol=1e-3)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axes[0]
    ax.imshow(img); _clean(ax)
    _box(ax, g, C["green"], lw=2.2, label="ground truth")
    _box(ax, a, C["blue"], lw=2, ls="--", label="anchor")
    ax.plot([xa, x], [ya, y], "-", color=C["red"], lw=1.6)
    ax.plot([xa], [ya], "o", color=C["blue"], ms=5); ax.plot([x], [y], "o", color=C["green"], ms=5)
    pad = 40
    ax.set_xlim(min(a[0], g[0]) - pad, max(a[2], g[2]) + pad)
    ax.set_ylim(max(a[3], g[3]) + pad, min(a[1], g[1]) - pad)
    ax.set_title(f"a positive anchor (IoU {D.iou(a[None], g[None])[0, 0]:.2f}) and the sheep it "
                 "is matched to", fontsize=10)
    ax = axes[1]; ax.axis("off")
    lines = [r"anchor:  $x_a = %.1f,\ y_a = %.1f,\ w_a = %.1f,\ h_a = %.1f$" % (xa, ya, wa, ha),
             r"ground truth:  $x = %.1f,\ y = %.1f,\ w = %.1f,\ h = %.1f$" % (x, y, w, h),
             "",
             r"$t_x = (x - x_a)/w_a = %.3f$" % t[0],
             r"$t_y = (y - y_a)/h_a = %.3f$" % t[1],
             r"$t_w = \log(w/w_a) = %.3f$" % t[2],
             r"$t_h = \log(h/h_a) = %.3f$" % t[3],
             "",
             r"decoded:  $x = x_a + t_x w_a,\ w = w_a e^{t_w}$  →  the ground truth again"]
    for k, s_ in enumerate(lines):
        ax.text(0.0, 0.95 - k * 0.105, s_, fontsize=11, transform=ax.transAxes, va="top")
    png = save_mpl(fig, ctx, "box_parameterisation")
    return Figure("box_parameterisation",
                  "**Bounding-box parameterisation**, worked for a real anchor of RetinaNet "
                  "and one of the sheep. The prediction head does not predict box "
                  "coordinates directly. It predicts the **displacement** of the box centre "
                  "from the anchor centre, normalised by the anchor's width and height, "
                  "and the **scaling** of the box's width and height relative to the "
                  "anchor, as a logarithm. The values that the head should output for this "
                  "anchor are calculated from the ground truth with the same equations, "
                  "and the box is recovered from them by rearranging the equations. These "
                  "are the equations of the Faster R-CNN region proposal network; many "
                  "variations are used in the literature.",
                  png=png)


def smooth_l1(x, beta=1.0):
    a = np.abs(x)
    return np.where(a < beta, 0.5 * a ** 2 / beta, a - 0.5 * beta)


def fig_detection_loss(ctx: BuildContext) -> Figure:
    x = np.linspace(-3, 3, 601)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax = axes[0]
    ax.plot(x, np.abs(x), color=C["blue"], lw=2, label=r"L1  $|t_k - t_k^{gt}|$")
    ax.plot(x, 0.5 * x ** 2, color=C["orange"], lw=2, label=r"L2  $(t_k - t_k^{gt})^2/2$")
    ax.plot(x, smooth_l1(x), color=C["red"], lw=2.6, label="smooth L1 (β = 1)")
    ax.set_ylim(0, 3.2); ax.set_xlabel(r"$t_k - t_k^{gt}$"); ax.set_ylabel("loss")
    ax.legend(fontsize=9); ax.set_title("(a) regression losses", fontsize=10.5)
    ax = axes[1]
    ax.plot(x, np.sign(x), color=C["blue"], lw=2, label="L1")
    ax.plot(x, x, color=C["orange"], lw=2, label="L2")
    ax.plot(x, np.clip(x, -1, 1), color=C["red"], lw=2.6, label="smooth L1")
    ax.set_ylim(-3, 3); ax.set_xlabel(r"$t_k - t_k^{gt}$"); ax.set_ylabel("gradient")
    ax.legend(fontsize=9)
    ax.set_title("(b) their gradients", fontsize=10.5)
    png = save_mpl(fig, ctx, "detection_loss")
    return Figure("detection_loss",
                  "The regression loss for the box values. The **smooth L1** loss combines "
                  "the L1 and L2 losses: for errors smaller than β it behaves as an L2 loss, "
                  "whose gradient decreases near zero and helps convergence, and for larger "
                  "errors as an L1 loss, whose gradient is bounded, which reduces the effect "
                  "of outliers and helps to prevent exploding gradients. It is a special "
                  r"case of the Huber loss. It is applied to the difference $t_k - t_k^{gt}$ "
                  r"between the predicted and ground-truth values of each of $t_x, t_y, t_w$ "
                  r"and $t_h$, and only for anchors that are positive.",
                  png=png)


def fig_focal_loss(ctx: BuildContext) -> Figure:
    r = _retina()
    lab, gammas = r["label"], r["gammas"]
    pt = np.linspace(0.001, 1, 500)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.9))
    ax = axes[0]
    for k, g in enumerate(gammas):
        ax.plot(pt, -(1 - pt) ** g * np.log(pt), lw=2.2 if g in (0, 2) else 1.4,
                color=SERIES[k], label=f"γ = {g:g}" + (" (cross-entropy)" if g == 0 else ""))
    ax.axvspan(0.6, 1, color=C["light"], alpha=0.5)
    ax.text(0.8, 3.6, "well-classified\nexamples", ha="center", fontsize=8.5, color=C["grey"])
    ax.set_ylim(0, 5); ax.set_xlabel("p_t, probability of the ground-truth class")
    ax.set_ylabel("loss"); ax.legend(fontsize=8.5)
    ax.set_title("(a) focal loss −(1 − p_t)^γ log p_t", fontsize=10.5)
    ax = axes[1]
    neg = r["pt"][lab == 0]
    ax.hist(neg, bins=np.linspace(0, 1, 41), color=C["grey"], log=True)
    ax.set_xlabel("p_t of the negative anchors"); ax.set_ylabel("anchors (log scale)")
    ax.set_title(f"(b) {np.mean(neg > 0.99):.1%} of the {len(neg):,} negatives\nhave "
                 "p_t > 0.99: easy, well-classified background", fontsize=10.5)
    ax = axes[2]
    pos_share = []
    for g in gammas:
        L = r[f"loss_g{float(g)}"]
        lp, ln = L[lab == 1].sum(), L[lab == 0].sum()
        pos_share.append((lp, ln))
    x = np.arange(len(gammas))
    tot = np.array([a + b for a, b in pos_share])
    ax.bar(x, [b / t for (a, b), t in zip(pos_share, tot)], color=C["grey"], label="negatives")
    ax.bar(x, [a / t for (a, b), t in zip(pos_share, tot)],
           bottom=[b / t for (a, b), t in zip(pos_share, tot)], color=C["green"], label="positives")
    ax.set_xticks(x); ax.set_xticklabels([f"γ = {g:g}" for g in gammas])
    ax.set_ylabel("share of the total loss"); ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("(c) share of the loss on this image,\nRetinaNet's real outputs", fontsize=10.5)
    shares = [a / t for (a, b), t in zip(pos_share, tot)]
    assert shares[list(gammas).index(2.0)] > 2 * shares[0], shares   # the caption's claim
    png = save_mpl(fig, ctx, "focal_loss")
    i2 = list(gammas).index(2.0)
    return Figure("focal_loss",
                  "**Focal loss** for the imbalance between positive and negative examples. "
                  "(a) Cross-entropy (γ = 0) does not fall to zero for well-classified "
                  "examples, so a very large number of easy negatives, each with a small "
                  "loss, can together dominate the loss and prevent the hard positives "
                  "from being learned. The focal loss multiplies the cross-entropy by "
                  "(1 − p_t)^γ, which reduces the loss of well-classified examples towards "
                  "zero, and γ = 2 is typically chosen. (b, c) The effect, measured with "
                  "RetinaNet's real outputs for the anchors of Figure 6: almost all of the "
                  "negatives are already classified as background with high confidence, "
                  f"but with cross-entropy the positives account for only {shares[0]:.1%} of "
                  "the total loss on this image. With γ = 2 the easy negatives contribute "
                  f"little, and the positives account for {shares[i2]:.0%}. (RetinaNet was "
                  "itself trained with γ = 2.)",
                  png=png)


def fig_fpn_levels(ctx: BuildContext) -> Figure:
    r = _retina()
    img = _sheep()
    shapes, lab, lev = r["shapes"], r["label"], r["level"]
    names = ["P3", "P4", "P5", "P6", "P7"]
    strides = [8, 16, 32, 64, 128]
    sizes = [32, 64, 128, 256, 512]
    fig, axes = plt.subplots(1, 6, figsize=(16, 2.9), gridspec_kw=dict(width_ratios=[1.3] + [1] * 5))
    axes[0].imshow(img); _clean(axes[0]); axes[0].set_title("input", fontsize=10)
    for k, ax in enumerate(axes[1:]):
        m = r[f"map{k}"].astype(np.float32)
        ax.imshow(m, cmap="magma"); _clean(ax)
        n = int(((lab == 1) & (lev == k)).sum())
        ax.set_title(f"{names[k]}: {shapes[k][1]} × {shapes[k][0]}, stride {strides[k]}\n"
                     f"anchors ≈ {sizes[k]} px, {n} positive", fontsize=9.5)
    png = save_mpl(fig, ctx, "fpn_levels")
    per = [int(((lab == 1) & (lev == k)).sum()) for k in range(5)]
    used = [names[k] for k in range(5) if per[k]]
    return Figure("fpn_levels",
                  "The **scale problem** and how multi-scale feature maps solve it, shown "
                  "with RetinaNet's real feature pyramid (the strength of the 256-channel "
                  "feature at each position) for the sheep photograph. A coarse feature map "
                  "with a large receptive field can only localise large objects, while "
                  "small objects require a finer map. In a feature pyramid each level has a "
                  "nominal scale: its anchors are sized for objects around that scale, from "
                  "32 pixels on P3 to 512 on P7. The sheep, between about 45 and 115 pixels "
                  "across, are matched only by anchors on " + " and ".join(used) + " ("
                  + ", ".join(f"{names[k]}: {per[k]}" for k in range(5)) + " positive "
                  "anchors): the anchors of P3 are too small and those of P6 and P7 too "
                  "large to overlap them with an IoU of 0.5, so each object is learned at "
                  "the level that matches its size.",
                  png=png)


def fig_ssd_multiscale(ctx: BuildContext) -> Figure:
    r = D.ssd_default_boxes()
    shapes, chans, an = r["shapes"], r["channels"], r["anchors"]
    counts = [int(h * w) for h, w in shapes]
    k_per = [int(k) for k in r["per_cell"]]          # 4 or 6 default boxes per cell
    total = len(an)
    assert sum(n * k for n, k in zip(counts, k_per)) == total
    img = load_image("sheep", max_side=800)
    sq = np.array(__import__("cv2").resize(img, (300, 300)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, lv in zip(axes[:2], (1, 3)):
        ax.imshow(sq); _clean(ax)
        h, w = shapes[lv]
        for x in np.linspace(0, 300, w + 1):
            ax.axvline(x, color="white", lw=0.5)
        for y in np.linspace(0, 300, h + 1):
            ax.axhline(y, color="white", lw=0.5)
        s0 = sum(n * k for n, k in zip(counts[:lv], k_per[:lv]))
        cell = (h // 2) * w + w // 2
        boxes = an[s0 + cell * k_per[lv]: s0 + (cell + 1) * k_per[lv]]
        for j, b in enumerate(boxes):
            _box(ax, b, SERIES[j % len(SERIES)], lw=1.8)
        ax.set_xlim(0, 300); ax.set_ylim(300, 0)
        ax.set_title(f"map {lv + 1}: {w} × {h}, {k_per[lv]} default boxes per cell",
                     fontsize=10)
    ax = axes[2]; ax.axis("off")
    rows = [[f"{i + 1}", f"{h} × {w} × {c}", f"{k}", f"{h * w * k:,}"]
            for i, ((h, w), c, k) in enumerate(zip(shapes, chans, k_per))]
    rows.append(["", "", "total", f"{total:,}"])
    tbl = ax.table(cellText=rows, colLabels=["map", "size", "boxes/cell", "boxes"], loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    for (rr, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["light"])
        if rr == 0 or rr == len(rows):
            cell.set_text_props(weight="bold")
    ax.set_title("SSD300 default boxes, read from torchvision's model", fontsize=10)
    png = save_mpl(fig, ctx, "ssd_multiscale")
    return Figure("ssd_multiscale",
                  "The **Single Shot MultiBox Detector** (SSD) uses feature maps at several "
                  "scales. The finest map is VGG-16's conv4_3, with L2 normalisation; the "
                  "others are fc7, converted to a convolution after conv5_3, and further "
                  "convolution layers that progressively downsample it, giving six feature "
                  "maps from "
                  f"{shapes[0][1]} × {shapes[0][0]} to 1 × 1 for a 300 × 300 input. At every "
                  "cell of every map, boxes and classes are predicted with 3 × 3 "
                  "convolutions, relative to a set of default (anchor) boxes, which are "
                  "small on the fine maps and large on the coarse maps (two levels are "
                  f"shown on the left, for the centre cell). In total {total:,} boxes are "
                  "predicted for each image.",
                  png=png)


def fig_depthwise_separable(ctx: BuildContext) -> Figure:
    k, C_in, N, Hs = 3, 256, 256, 56
    std_p, std_m = k * k * C_in * N, k * k * C_in * N * Hs * Hs
    dw_p, pw_p = k * k * C_in, C_in * N
    dw_m, pw_m = dw_p * Hs * Hs, pw_p * Hs * Hs
    nets = [("ResNet-50", A.resnet(50)), ("MobileNetV2", A.mobilenet_v2())]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    ax = axes[0]; ax.axis("off")
    txt = [("standard 3 × 3 convolution", f"{N} kernels of {k} × {k} × {C_in}", C["blue"]),
           ("depthwise", f"{C_in} kernels of {k} × {k} × 1, one per channel", C["orange"]),
           ("pointwise", f"{N} kernels of 1 × 1 × {C_in}", C["green"])]
    for i, (a, b, col) in enumerate(txt):
        y = 0.82 - i * 0.3
        ax.add_patch(plt.Rectangle((0.02, y - 0.08), 0.05, 0.16, fc=col, alpha=0.6,
                                   transform=ax.transAxes))
        ax.text(0.1, y + 0.03, a, fontsize=10.5, weight="bold", transform=ax.transAxes)
        ax.text(0.1, y - 0.07, b, fontsize=9.5, transform=ax.transAxes)
    ax.set_title(f"{C_in} → {N} channels on a {Hs} × {Hs} map", fontsize=10.5)
    ax = axes[1]
    ax.bar(["standard", "depthwise\n+ pointwise"], [std_m / 1e6, 0], color=C["blue"])
    ax.bar(["standard", "depthwise\n+ pointwise"], [0, dw_m / 1e6], color=C["orange"])
    ax.bar(["standard", "depthwise\n+ pointwise"], [0, pw_m / 1e6], bottom=[0, dw_m / 1e6],
           color=C["green"])
    ax.text(0, std_m / 1e6, f"{std_m / 1e6:.0f} M\n{std_p:,} params", ha="center", va="bottom",
            fontsize=9)
    ax.text(1, (dw_m + pw_m) / 1e6, f"{(dw_m + pw_m) / 1e6:.0f} M\n{dw_p + pw_p:,} params",
            ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("multiply-accumulates (millions)"); ax.set_ylim(0, std_m / 1e6 * 1.3)
    ax.set_title(f"one layer: {std_m / (dw_m + pw_m):.1f} × less computation", fontsize=10.5)
    ax = axes[2]
    ax.bar([n for n, _ in nets], [net.macs / 1e9 for _, net in nets], color=[C["grey"], C["green"]])
    for i, (n, net) in enumerate(nets):
        ax.text(i, net.macs / 1e9, f"{net.macs / 1e9:.2f} G\n{net.params / 1e6:.1f} M params",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("G multiply-accumulates per image"); ax.set_ylim(0, 5.3)
    ax.set_title("whole backbones (224 × 224)", fontsize=10.5)
    assert abs(std_m / (dw_m + pw_m) - 1 / (1 / N + 1 / k ** 2)) < 1e-6
    png = save_mpl(fig, ctx, "depthwise_separable")
    return Figure("depthwise_separable",
                  "**Depthwise separable convolution**, used in lightweight backbones such "
                  "as MobileNet for mobile and embedded detectors. A standard convolution "
                  "combines spatial filtering and the mixing of channels in one step. A "
                  "depthwise separable convolution splits it in two: a **depthwise** "
                  "convolution filters each channel separately with one k × k kernel, and a "
                  "**pointwise** 1 × 1 convolution then mixes the channels. The cost is "
                  f"reduced by a factor of 1 / (1/N + 1/k²), {std_m / (dw_m + pw_m):.1f} for "
                  f"this layer. Across a whole network, MobileNetV2 requires "
                  f"{nets[1][1].macs / 1e9:.2f} G multiply-accumulates per image against "
                  f"{nets[0][1].macs / 1e9:.2f} G for ResNet-50 (computed from the "
                  "architectures of Section 2).",
                  png=png)


def fig_nms_steps(ctx: BuildContext) -> Figure:
    img = _sheep()
    cats = D.categories("faster_rcnn")
    raw = D.raw_detections("sheep", img)
    sheep = np.array([cats[l] == "sheep" for l in raw["labels"]])
    b, s = raw["boxes"][sheep], raw["scores"][sheep]
    keep, steps = M.nms(b, s, 0.5)
    per_obj = (M.iou(b, SHEEP_GT) > 0.5).sum(0)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 3.6))
    for ax in axes:
        ax.imshow(img); _clean(ax)
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0] * 0.66, img.shape[0] * 0.2)
    for bb in b:
        _box(axes[0], bb, C["orange"], lw=0.9)
    axes[0].set_title(f"(a) before NMS: {len(b)} 'sheep' boxes with confidence > 0.3",
                      fontsize=10)
    for i in keep:
        _box(axes[1], b[i], C["green"], lw=1.8, label=f"{s[i]:.2f}", fs=7)
    axes[1].set_title(f"(b) after NMS at IoU 0.5: {len(keep)} boxes", fontsize=10)
    png = save_mpl(fig, ctx, "nms_steps")

    h, w = img.shape[:2]
    bg = dict(source=to_data_uri(img, "jpg", 80), xref="x", yref="y", x=0, y=0, sizex=w,
              sizey=h, sizing="stretch", layer="below")
    frames, state = [], {"kept": [], "gone": []}

    def shapes_for(current=None):
        out = []
        for i in range(len(b)):
            if i in state["kept"]:
                col, wd = C["green"], 3
            elif i == current:
                col, wd = C["red"], 3
            elif i in state["gone"]:
                continue
            else:
                col, wd = C["orange"], 1
            out.append(dict(type="rect", x0=b[i][0], y0=b[i][1], x1=b[i][2], y1=b[i][3],
                            line=dict(color=col, width=wd)))
        return out
    frames.append(go.Frame(name="0", layout=go.Layout(shapes=shapes_for(),
                           title=f"{len(b)} raw boxes; step through NMS with the slider")))
    for k, (i, drop) in enumerate(steps, 1):
        state["kept"].append(i)
        state["gone"] += drop
        frames.append(go.Frame(name=str(k), layout=go.Layout(
            shapes=shapes_for(), title=f"step {k}: keep the box with confidence {s[i]:.2f}, "
                                       f"remove {len(drop)} overlapping it by IoU > 0.5")))
    fig2 = go.Figure(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(opacity=0)))
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(list(range(len(frames))), "NMS step: "),
                       images=[bg], shapes=frames[0].layout.shapes,
                       title=frames[0].layout.title.text, showlegend=False)
    fig2.update_xaxes(range=[0, w], visible=False)
    fig2.update_yaxes(range=[h * 0.66, h * 0.2], visible=False, scaleanchor="x")
    return Figure("nms_steps",
                  "**Non-maximum suppression** (NMS) applied to the real output of Faster "
                  "R-CNN with its own NMS switched off. A dense detector predicts several "
                  "overlapping boxes around each object, from neighbouring cells and "
                  "anchors. NMS keeps the box with the highest confidence, removes every "
                  "box that overlaps it with an IoU above a threshold (0.5 here), and "
                  "repeats with the highest remaining box until none is left. "
                  f"The {len(b)} boxes, between {per_obj.min()} and {per_obj.max()} for "
                  f"each sheep, are reduced to {len(keep)}, one for each sheep. (Many of "
                  "the raw boxes are almost identical, so they appear as a single thick "
                  "box in (a).) NMS is applied separately to "
                  "each class. In the interactive version each step of NMS is shown with "
                  "the slider.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def build(ctx: BuildContext) -> list[Figure]:
    from . import single_stage_diagrams as SD
    return [fig_feature_location(ctx), SD.fig_detection_system(ctx), SD.fig_head_sliding(ctx),
            fig_prediction_head(ctx), SD.fig_gt_assignment(ctx), fig_anchor_assignment(ctx), SD.fig_box_dimensions(ctx),
            fig_box_parameterisation(ctx), fig_detection_loss(ctx), fig_focal_loss(ctx),
            SD.fig_scale_grids(ctx), fig_fpn_levels(ctx), SD.fig_ssd_architecture(ctx),
            fig_ssd_multiscale(ctx), SD.fig_fpn_detector(ctx), fig_depthwise_separable(ctx),
            fig_nms_steps(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("3.4 Non-maximum suppression")
    img = _sheep()
    cats = D.categories("faster_rcnn")
    raw = D.raw_detections("sheep", img)
    sheep = np.array([cats[l] == "sheep" for l in raw["labels"]])
    b, s = raw["boxes"][sheep], raw["scores"][sheep]
    t = st.slider("NMS IoU threshold", 0.1, 0.95, 0.5, 0.05)
    keep, _ = M.nms(b, s, t)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.imshow(img); _clean(ax)
    for i in keep:
        _box(ax, b[i], C["green"], label=f"{s[i]:.2f}", fs=7)
    st.pyplot(fig)
    st.write(f"{len(b)} raw boxes → {len(keep)} after NMS")
