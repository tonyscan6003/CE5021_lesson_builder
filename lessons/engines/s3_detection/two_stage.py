"""Lesson 3.3 — Two-stage object detection with region proposals.

Figures
    rcnn_family           R-CNN -> Fast R-CNN -> Faster R-CNN: the problems and solutions
    rcnn_cost             CNN computation per image and SVM feature storage, computed
    spp_layer             spatial pyramid pooling of real VGG-16 feature maps of two sizes
    class_activation_map  real CAMs from ResNet-50 (GAP + one dense layer)
    roi_projection        proposals projected onto a real conv5_3 map; quantisation error
    roi_pooling           two RoIs of different size pooled to the same 7 x 7 output
    rpn_proposals         anchors, the RPN's proposals and the final detections of a
                          real Faster R-CNN
    rpn_recall            RPN vs selective search recall on lesson 3.2's 13 objects
    two_vs_single_stage   COCO mAP (torchvision metadata) vs inference time measured here

Real networks via engines/_torchvis.py (VGG-16 feature maps) and
engines/_torchdet.py (ResNet-50, Faster R-CNN and the detector zoo), cached in
data/_torch/. Costs from engines/_arch.py.

Source: reference/legacy_md/lesson_3_3_obj_det_partA_w_reg_Prop + lesson_3_3_script.
"""
from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt

from .._common import BuildContext, Figure, C, SERIES, save_mpl, load_image
from .. import _arch as A
from .. import _detmetrics as M
from .. import _torchdet as D
from .. import _torchvis as TV
from . import region_proposals as RP

FIGURES = ["rcnn_family", "rcnn_cost", "spp_layer", "class_activation_map",
           "roi_projection", "roi_pooling", "rpn_proposals", "rpn_recall",
           "two_vs_single_stage"]

STRIDE = 16           # VGG-16 conv5_3: four 2x2 poolings


def _box(ax, b, col, lw=2.0, ls="-", label=None, fs=8):
    ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col,
                               lw=lw, ls=ls))
    if label:
        ax.text(b[0] + 1, b[1] - 2, label, fontsize=fs, va="bottom",
                bbox=dict(fc=col, ec="none", pad=1.0, alpha=0.9))


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _stage(ax, x, y, w, h, text, col, fs=8.5):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=col, alpha=0.55, ec=C["ink"], lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.3))


# ---------------------------------------------------------------------------

def fig_rcnn_family(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(14, 6.2))
    rows = [
        ("R-CNN (2014)", [("image", C["light"]), ("selective\nsearch", C["orange"]),
                          ("warp each\nregion", C["light"]), ("CNN per\nregion", C["blue"]),
                          ("SVM per class\n+ box regressor", C["purple"])],
         "C1: CNN features    P2: regions of different sizes    P3: overlapping regions "
         "processed many times    P4: proposals trained separately"),
        ("Fast R-CNN (2015)", [("image", C["light"]), ("CNN once\nper image", C["blue"]),
                               ("RoI projection\n(selective search)", C["orange"]),
                               ("RoI pooling\n7 × 7", C["sky"]),
                               ("dense layers:\nclass + box", C["purple"])],
         "C2: RoI pooling (one level of SPP)    C3: RoI projection    "
         "multi-task loss, end-to-end training of the second stage"),
        ("Faster R-CNN (2015)", [("image", C["light"]), ("CNN once\nper image", C["blue"]),
                                 ("region proposal\nnetwork", C["green"]),
                                 ("RoI pooling\n7 × 7", C["sky"]),
                                 ("dense layers:\nclass + box", C["purple"])],
         "C4: region proposal network shares the convolution layers — no external proposals"),
    ]
    for r, (name, stages, note) in enumerate(rows):
        y = 4.2 - r * 2.0
        ax.text(-0.2, y + 0.45, name, ha="right", va="center", fontsize=10.5, weight="bold")
        for k, (txt, col) in enumerate(stages):
            _stage(ax, k * 2.6, y, 2.0, 0.9, txt, col)
            if k:
                _arrow(ax, k * 2.6 - 0.58, y + 0.45, k * 2.6 - 0.02, y + 0.45)
        ax.text(0, y - 0.28, note, fontsize=9, color=C["red"], va="top")
    ax.set_xlim(-3.3, 13); ax.set_ylim(-0.4, 5.4); ax.axis("off")
    png = save_mpl(fig, ctx, "rcnn_family")
    return Figure("rcnn_family",
                  "The R-CNN family of two-stage detectors and the problems solved at each "
                  "step. **R-CNN** used a pre-trained CNN as a feature extractor (C1), with "
                  "selective search for the proposals and an SVM per class, but each "
                  "region had to be warped to the input size of the network (P2), "
                  "overlapping regions were processed by the CNN many times (P3), and each "
                  "stage was trained separately (P4). **Fast R-CNN** processes the image "
                  "once and projects the proposals onto the feature maps (C3), converts "
                  "each region to a fixed size with RoI pooling (C2), and predicts the "
                  "class and box with dense layers trained by a multi-task loss. **Faster "
                  "R-CNN** replaces selective search with a region proposal network that "
                  "shares the convolution layers (C4).",
                  png=png)


def fig_rcnn_cost(ctx: BuildContext) -> Figure:
    n_prop, n_img = 2000, 5011                       # proposals per image; VOC 2007 trainval
    vgg = A.vgg16()
    conv224 = sum(l.macs for l in vgg.layers if l.kind == "conv")
    head = sum(l.macs for l in vgg.layers if l.kind == "fc" and "softmax" not in l.name)
    alex = A.alexnet()
    alex_conv = sum(l.macs for l in alex.layers if l.kind == "conv")
    alex_fc = sum(l.macs for l in alex.layers if l.kind == "fc" and "softmax" not in l.name)
    scale = 600 * 1000 / 224 ** 2                     # Fast R-CNN's image size vs 224²
    rows = [("R-CNN, AlexNet", n_prop * (alex_conv + alex_fc), 0),
            ("R-CNN, VGG-16", n_prop * (conv224 + head), 0),
            ("Fast R-CNN, VGG-16", conv224 * scale, n_prop * head)]
    store = n_prop * 4096 * 4 * n_img                 # fc7 features, float32
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    y = np.arange(len(rows))
    ax.barh(y, [r[1] / 1e12 for r in rows], color=C["blue"], label="convolution layers")
    ax.barh(y, [r[2] / 1e12 for r in rows], left=[r[1] / 1e12 for r in rows], color=C["orange"],
            label="dense layers per region")
    for k, r in enumerate(rows):
        ax.text((r[1] + r[2]) / 1e12 + 0.3, k, f"{(r[1] + r[2]) / 1e12:.1f} T", va="center",
                fontsize=9.5)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel(f"multiply-accumulates per image (×10¹²), {n_prop} proposals")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("(a) computation per image", fontsize=10.5)
    ax = axes[1]; ax.axis("off")
    ax.text(0.02, 0.85, "(b) R-CNN: features stored to train the SVMs", fontsize=10.5,
            transform=ax.transAxes)
    ax.text(0.02, 0.62, f"{n_prop:,} proposals × 4,096 values × 4 bytes\n"
            f"× {n_img:,} training images (Pascal VOC 2007)", fontsize=10,
            transform=ax.transAxes, va="top")
    ax.text(0.02, 0.22, f"= {store / 1e9:,.0f} GB", fontsize=15, color=C["red"],
            transform=ax.transAxes, weight="bold")
    png = save_mpl(fig, ctx, "rcnn_cost")
    ratio = (rows[1][1]) / (rows[2][1] + rows[2][2])
    return Figure("rcnn_cost",
                  "Why R-CNN is slow, calculated from the layer specifications of Section 2. "
                  f"(a) In R-CNN each of approximately {n_prop:,} warped proposals is passed "
                  "through the whole CNN, so the convolution layers are applied thousands "
                  "of times per image. In Fast R-CNN the convolution layers are applied "
                  "once, to the whole image (at 600 × 1000 pixels), and only the dense "
                  "layers are applied to each region; with VGG-16 the computation is "
                  f"reduced by a factor of approximately {ratio:.0f}. (b) To train the "
                  "SVMs and box regressors of R-CNN, the CNN features of every proposal in "
                  f"every training image must be stored, approximately {store / 1e9:,.0f} GB "
                  "for Pascal VOC 2007, which agrees with the hundreds of gigabytes "
                  "reported.",
                  png=png)


def _conv5(img):
    """Real VGG-16 conv5_3 feature maps (512, h/16, w/16) of an image."""
    import torch
    with torch.no_grad():
        return TV.features(TV.to_tensor(img), ["conv5_3"])["conv5_3"][0].cpu().numpy()


def spp_pool(fmap, levels=(4, 2, 1)):
    """Spatial pyramid pooling: max over an n × n grid of bins at each level,
    whatever the map's size; returns the concatenated vector and the bin edges."""
    c, h, w = fmap.shape
    out, edges = [], {}
    for n in levels:
        ys = np.floor(np.arange(n + 1) * h / n).astype(int)
        xs = np.floor(np.arange(n + 1) * w / n).astype(int)
        edges[n] = (ys, xs)
        for i in range(n):
            for j in range(n):
                out.append(fmap[:, ys[i]:max(ys[i + 1], ys[i] + 1),
                                xs[j]:max(xs[j + 1], xs[j] + 1)].max((1, 2)))
    return np.concatenate(out), edges


def fig_spp_layer(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=800)
    crops = [cv2.resize(img[180:340, 40:340], (224, 224)), img[120:440, 0:800][:, :640]]
    res = TV.cached("spp_conv5_sheep", dict(img="sheep", crops=[c.shape for c in crops]),
                    lambda: {f"f{k}": _conv5(c).astype(np.float16) for k, c in enumerate(crops)})
    fmaps = [res["f0"].astype(np.float32), res["f1"].astype(np.float32)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[1.2, 1, 1, 1]))
    for r, (crop, fm) in enumerate(zip(crops, fmaps)):
        vec, edges = spp_pool(fm)
        axes[r, 0].imshow(crop); _clean(axes[r, 0])
        axes[r, 0].set_title(f"input {crop.shape[1]} × {crop.shape[0]}", fontsize=10)
        act = fm.mean(0)
        for ax, n in zip(axes[r, 1:], (4, 2, 1)):
            ax.imshow(act, cmap="viridis"); _clean(ax)
            ys, xs = edges[n]
            for y in ys:
                ax.axhline(y - 0.5, color="white", lw=1.2)
            for x in xs:
                ax.axvline(x - 0.5, color="white", lw=1.2)
            ax.set_title(f"conv5_3 {fm.shape[2]} × {fm.shape[1]} × {fm.shape[0]}: "
                         f"{n} × {n} bins", fontsize=9.5)
        axes[r, 3].text(1.05, 0.5, f"→ {len(vec):,}\nvalues", transform=axes[r, 3].transAxes,
                        fontsize=11, va="center", color=C["red"])
    assert len(spp_pool(fmaps[0])[0]) == len(spp_pool(fmaps[1])[0]) == 21 * 512
    png = save_mpl(fig, ctx, "spp_layer")
    return Figure("spp_layer",
                  "**Spatial pyramid pooling** applied to real feature maps. Convolution "
                  "layers accept inputs of any size, but the size of the output feature "
                  "map changes with the input, while dense layers require a fixed-length "
                  "input. Here two inputs of different sizes give VGG-16 conv5_3 feature "
                  f"maps of {fmaps[0].shape[2]} × {fmaps[0].shape[1]} and "
                  f"{fmaps[1].shape[2]} × {fmaps[1].shape[1]} (the mean over the 512 "
                  "channels is shown). In SPP, max pooling is applied over a 4 × 4, a "
                  "2 × 2 and a 1 × 1 grid of bins; the size of the bins changes with the "
                  "size of the map, so that the number of bins does not. Each bin gives "
                  "one value per channel, so both inputs produce a vector of "
                  "(16 + 4 + 1) × 512 = 10,752 values, and the region proposals no longer "
                  "need to be warped.",
                  png=png)


def fig_class_activation_map(ctx: BuildContext) -> Figure:
    keys = ["sheep", "zebra", "cow_field", "macaw"]
    icats = D.categories("resnet50")
    fig, axes = plt.subplots(1, len(keys), figsize=(15.5, 3.2))
    shapes = None
    for ax, key in zip(axes, keys):
        img = load_image(key, max_side=448)
        r = D.class_activation_map(key, img)
        cam = np.maximum(r["cam"], 0)
        cam = cv2.resize(cam / (cam.max() + 1e-9), (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_CUBIC)
        shapes = r["fmap_shape"]
        ax.imshow(img); ax.imshow(cam, cmap="jet", alpha=0.45); _clean(ax)
        ax.set_title(f"predicted class: '{icats[int(r['cls'])]}'", fontsize=10)
    png = save_mpl(fig, ctx, "class_activation_map")
    return Figure("class_activation_map",
                  "**Class activation maps** computed with ResNet-50, an ImageNet classifier "
                  "whose last convolution layer is followed by global average pooling and "
                  "a single dense layer. The class score is a weighted sum of the pooled "
                  "feature maps, so applying the same class weights to the feature maps "
                  f"themselves ({shapes[0]:,} maps of {shapes[1]} × {shapes[2]} here) "
                  "shows where in the image the evidence for the predicted class lies "
                  "(the whole photograph is used as the input, at 448 pixels). "
                  "Although the network was trained with image labels only, the maps "
                  "highlight the objects: the network has learned to locate objects "
                  "without being trained to do so. This spatial information in the feature "
                  "maps is what makes region of interest projection, and the region "
                  "proposal network of Faster R-CNN, possible.",
                  png=png)


def fig_roi_projection(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=800)
    res = TV.cached("conv5_sheep800", dict(img="sheep", shape=img.shape),
                    lambda: dict(f=_conv5(img).astype(np.float16)))
    fm = res["f"].astype(np.float32)
    gt = RP.GT["sheep"] * 2                                   # 800 px coordinates
    proj = gt / STRIDE
    q = np.stack([np.floor(proj[:, 0]), np.floor(proj[:, 1]),
                  np.ceil(proj[:, 2]), np.ceil(proj[:, 3])], 1)
    back = q * STRIDE
    err = np.abs(back - gt).max()
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.6))
    ax = axes[0]
    ax.imshow(img); _clean(ax)
    for k, b in enumerate(gt[:4]):
        _box(ax, b, SERIES[k], lw=2)
    ax.set_ylim(img.shape[0] * 0.66, img.shape[0] * 0.2)
    ax.set_title(f"proposals on the {img.shape[1]} × {img.shape[0]} image", fontsize=10)
    ax = axes[1]
    ax.imshow(fm.mean(0), cmap="viridis"); _clean(ax)
    for k, (p, b) in enumerate(zip(proj[:4], q[:4])):
        _box(ax, p - 0.5, SERIES[k], lw=1.2, ls="--")
        _box(ax, b - 0.5, SERIES[k], lw=2)
    ax.set_ylim(fm.shape[1] * 0.66, fm.shape[1] * 0.2)
    ax.set_title(f"projected onto conv5_3, {fm.shape[2]} × {fm.shape[1]} (stride {STRIDE}): "
                 "exact (dashed) and rounded to cells (solid)", fontsize=10)
    png = save_mpl(fig, ctx, "roi_projection")
    return Figure("roi_projection",
                  "**Region of interest projection** in Fast R-CNN, using the real conv5_3 "
                  "feature maps of VGG-16 (mean over the channels). The convolution "
                  "layers are applied once to the whole image, and the proposals are "
                  "projected onto the feature maps instead of being cropped from the "
                  f"image. As the feature map is reduced by a stride of {STRIDE}, each "
                  f"proposal's coordinates are divided by {STRIDE}. The projected "
                  "coordinates are not whole numbers (dashed), and in RoI pooling they "
                  "are rounded to whole feature-map cells (solid); for these sheep the "
                  f"rounding moves the box edges by up to {err:.0f} pixels in the image. "
                  "The main advantage is that the convolution is computed once per image "
                  "rather than once per proposal.",
                  png=png)


def roi_pool(fmap, box, out=7):
    """RoI pooling (Fast R-CNN): the box, already in feature-map cells, is
    divided into an out × out grid of bins with whole-cell boundaries, and the
    maximum is taken in each bin."""
    x1, y1, x2, y2 = [int(v) for v in box]
    ys = np.floor(np.linspace(y1, y2, out + 1)).astype(int)
    xs = np.floor(np.linspace(x1, x2, out + 1)).astype(int)
    res = np.zeros((out, out))
    for i in range(out):
        for j in range(out):
            res[i, j] = fmap[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)].max()
    return res, ys, xs


def fig_roi_pooling(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=800)
    fm = TV.cached("conv5_sheep800", dict(img="sheep", shape=img.shape),
                   lambda: dict(f=_conv5(img).astype(np.float16)))["f"].astype(np.float32)
    gt = RP.GT["sheep"] * 2
    q = np.stack([np.floor(gt[:, 0] / STRIDE), np.floor(gt[:, 1] / STRIDE),
                  np.ceil(gt[:, 2] / STRIDE), np.ceil(gt[:, 3] / STRIDE)], 1)
    ch = int(np.argmax([fm[c][int(q[0, 1]):int(q[0, 3]), int(q[0, 0]):int(q[0, 2])].mean()
                        for c in range(fm.shape[0])]))
    picks = [0, 2]                                   # the largest and a small sheep
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.6), gridspec_kw=dict(width_ratios=[1.3, 1, 1, 1]))
    outs = []
    for k, i in enumerate(picks):
        b = q[i]
        pooled, ys, xs = roi_pool(fm[ch], b)
        outs.append(pooled)
        ax = axes[2 * k] if k == 0 else axes[2]
        x1, y1, x2, y2 = b.astype(int)
        ax.imshow(fm[ch][y1:y2, x1:x2], cmap="viridis", extent=(x1 - 0.5, x2 - 0.5, y2 - 0.5, y1 - 0.5))
        for y in ys:
            ax.axhline(y - 0.5, color="white", lw=0.9)
        for x in xs:
            ax.axvline(x - 0.5, color="white", lw=0.9)
        _clean(ax)
        ax.set_title(f"sheep {i + 1}: RoI of {x2 - x1} × {y2 - y1} cells\nin 7 × 7 bins",
                     fontsize=9.5)
        ax2 = axes[1] if k == 0 else axes[3]
        ax2.imshow(pooled, cmap="viridis"); _clean(ax2)
        ax2.set_title("→ max in each bin: 7 × 7", fontsize=9.5)
    assert outs[0].shape == outs[1].shape == (7, 7)
    png = save_mpl(fig, ctx, "roi_pooling")
    return Figure("roi_pooling",
                  "**RoI pooling**, a single level of spatial pyramid pooling, applied to "
                  f"one channel ({ch}) of the real conv5_3 feature map. Each projected "
                  "region of interest is divided into a 7 × 7 grid of bins, whose size "
                  "depends on the size of the region, and the maximum in each bin is "
                  "taken. A large and a small region therefore both give a 7 × 7 output "
                  "per channel (7 × 7 × 512 for VGG-16), which is flattened and passed to "
                  "the dense layers that predict the class and refine the box. When a "
                  "region is smaller than 7 cells, some bins contain only a single cell; "
                  "the bin boundaries are also rounded to whole cells, a quantisation that "
                  "RoIAlign removes (Lesson 3.6).",
                  png=png)


def anchors_at(cx, cy, scales=(128, 256, 512), ratios=(0.5, 1.0, 2.0)):
    """The 9 anchors of the original Faster R-CNN at one position: 3 scales
    (box areas of scale²) × 3 aspect ratios (height / width)."""
    out = []
    for s in scales:
        for r in ratios:
            w, h = s / np.sqrt(r), s * np.sqrt(r)
            out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(out)


def fig_rpn_proposals(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=800)
    cats = D.categories("faster_rcnn")
    props = D.rpn_proposals("sheep", img)["boxes"]
    det = D.detect("sheep", img)
    keep = det["scores"] >= 0.5
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.7))
    for ax in axes:
        ax.imshow(img); _clean(ax)
    an = anchors_at(250, 250, scales=(64, 128, 256))
    for k, a in enumerate(an):
        _box(axes[0], a, SERIES[k // 3], lw=1.3)
    axes[0].plot(250, 250, "o", color="white", ms=5)
    axes[0].set_title("(a) 9 anchor boxes at one position:\n3 sizes × 3 aspect ratios",
                      fontsize=10)
    for b in props[:50]:
        _box(axes[1], b, C["orange"], lw=0.8)
    axes[1].set_title(f"(b) stage 1: the 50 highest-scoring of\n{len(props)} RPN proposals",
                      fontsize=10)
    for b, l, s in zip(det["boxes"][keep], det["labels"][keep], det["scores"][keep]):
        _box(axes[2], b, C["green"], lw=1.6, label=f"{cats[l]} {s:.2f}", fs=6.5)
    axes[2].set_title(f"(c) stage 2: classes and refined boxes,\n{keep.sum()} detections "
                      "above 0.5", fontsize=10)
    for ax in axes:
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0] * 0.75, 0)
    png = save_mpl(fig, ctx, "rpn_proposals")
    return Figure("rpn_proposals",
                  "Inside a real Faster R-CNN (torchvision's pre-trained model). (a) The "
                  "**region proposal network** (RPN) is a small convolutional network that "
                  "slides over the shared feature maps. At every position it scores a set "
                  "of reference boxes, the **anchors**, of several sizes and aspect "
                  "ratios, for objectness (object or background), and predicts an "
                  "adjustment to each anchor's box. (b) The best-scoring adjusted anchors, "
                  "after overlapping ones are removed, are the proposals: they are "
                  "computed from the same feature maps as the detections, so no external "
                  "algorithm such as selective search is needed. (c) In the second stage "
                  "each proposal is RoI-pooled, classified, and its box refined. The "
                  "original training used an alternating scheme: the RPN is trained, then "
                  "the Fast R-CNN part with its proposals, then the shared convolution "
                  "layers are fixed and each part is fine-tuned in turn.",
                  png=png)


def fig_rpn_recall(ctx: BuildContext) -> Figure:
    ks = np.unique(np.round(np.logspace(0, np.log10(1000), 28)).astype(int))
    curves = {"region proposal network": [], "selective search": []}
    n_gt = 0
    for key, gt in RP.GT.items():
        img = RP.photo(key)
        n_gt += len(gt)
        curves["region proposal network"].append(
            RP.recall_at(D.rpn_proposals(f"{key}_400", img)["boxes"], gt, ks) * len(gt))
        curves["selective search"].append(RP.recall_at(RP.ss(key)[3], gt, ks) * len(gt))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    res = {}
    for (name, rows), col in zip(curves.items(), [C["green"], C["blue"]]):
        r = np.sum(rows, 0) / n_gt
        res[name] = r
        ax.semilogx(ks, r, color=col, lw=2.2, label=name)
    ax.set_xlabel("number of proposals per image"); ax.set_ylabel("recall (IoU > 0.5)")
    ax.set_ylim(0, 1.03); ax.grid(alpha=0.3); ax.legend(fontsize=9.5, loc="lower right")
    ax.set_title(f"proposal recall on lesson 3.2's {n_gt} annotated objects", fontsize=10.5)
    png = save_mpl(fig, ctx, "rpn_recall")
    k10 = int(np.argmin(np.abs(ks - 10)))
    return Figure("rpn_recall",
                  "The proposals of the region proposal network compared with selective "
                  "search, on the photographs and hand-drawn ground truth of Lesson 3.2. "
                  f"With {ks[k10]} proposals per image the RPN covers "
                  f"{res['region proposal network'][k10]:.0%} of the objects, against "
                  f"{res['selective search'][k10]:.0%} for selective search: as the "
                  "proposals are learned from data, far fewer are needed for a high "
                  "recall, which reduces the work of the second stage. (Faster R-CNN "
                  "passes 1,000 proposals to the second stage at test time.)",
                  png=png)


def fig_two_vs_single_stage(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=800)
    t = D.time_detectors("sheep", img)
    names, secs, dev = list(t["names"]), t["seconds"], str(t["device"])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for n, s in zip(names, secs):
        two = D.ZOO[n][2] == 2
        ax.scatter(s * 1000, D.coco_map(n), s=110, marker="s" if two else "o",
                   color=C["blue"] if two else C["orange"], edgecolor=C["ink"], zorder=3)
        ax.annotate(n, (s * 1000, D.coco_map(n)),
                    xytext=(7, -11) if n == "RetinaNet" else (7, 4), textcoords="offset points",
                    fontsize=9)
    ax.scatter([], [], marker="s", color=C["blue"], label="two-stage")
    ax.scatter([], [], marker="o", color=C["orange"], label="single-stage")
    ax.set_xscale("log")
    ax.set_xlabel(f"inference time per image, measured on this machine ({dev.upper()}), ms")
    ax.set_ylabel("COCO val2017 box mAP")
    ax.legend(fontsize=9.5, loc="lower right"); ax.grid(alpha=0.3, which="both")
    ax.set_title("accuracy against speed for torchvision's pre-trained detectors", fontsize=10.5)
    png = save_mpl(fig, ctx, "two_vs_single_stage")
    best = names[int(np.argmax([D.coco_map(n) for n in names]))]
    fast = names[int(np.argmin(secs))]
    return Figure("two_vs_single_stage",
                  "Two-stage and single-stage detectors compared. The mAP values are those "
                  "published for torchvision's pre-trained weights (COCO val2017), and the "
                  f"inference time was measured for one {img.shape[1]} × {img.shape[0]} "
                  f"image on the {dev.upper()} used to build this page. The region proposal "
                  "network is functionally equivalent to a single-stage detector: both "
                  "predict boxes and scores directly from the feature maps. The second "
                  "stage of a two-stage detector refines the boxes and classes, which gives "
                  f"more accurate detection — the most accurate model here is {best} — at "
                  f"the cost of a slower frame rate; the fastest is {fast}. Modern "
                  "single-stage detectors have closed much of the gap in accuracy while "
                  "keeping a higher frame rate, which makes them the default choice in many "
                  "applications. Both families depend on the backbone network as well as "
                  "on the number of stages.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_rcnn_family(ctx), fig_rcnn_cost(ctx), fig_spp_layer(ctx),
            fig_class_activation_map(ctx), fig_roi_projection(ctx), fig_roi_pooling(ctx),
            fig_rpn_proposals(ctx), fig_rpn_recall(ctx), fig_two_vs_single_stage(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("3.3 Region proposal network")
    key = st.selectbox("image", list(RP.GT))
    img = RP.photo(key)
    props = D.rpn_proposals(f"{key}_400", img)["boxes"]
    k = st.slider("proposals shown", 1, 200, 20)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(img); _clean(ax)
    for b in props[:k]:
        _box(ax, b, C["orange"], lw=0.9)
    for g in RP.GT[key]:
        _box(ax, g, C["sky"], lw=2)
    st.pyplot(fig)
