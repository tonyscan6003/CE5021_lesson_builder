"""Lesson 3.1 — Introduction to object detection.

Figures
    detection_task        classification vs localisation vs detection vs instance
                          segmentation, from real pre-trained models on one photo
    window_pyramid        a fixed window over a 4-level pyramid: the ball fits at level 0,
                          the dog only at level 3
    sliding_window        windows x scales x pyramid levels counted for a real image
    proposal_pipeline     region proposals warped and classified by a real ResNet-50
    detection_datasets    Pascal VOC, ImageNet, MS COCO and Open Images compared
    iou_boxes             intersection over union, three overlaps (interactive)
    tp_fp_fn              real Faster R-CNN detections scored against hand-drawn ground
                          truth at three threshold settings
    pr_curve              a ranked list of detections -> zig-zag PR curve ->
                          interpolated curve -> average precision
    mean_average_precision  AP per class and mAP, at IoU 0.5 and 0.75
    detection_errors      real false positives and false negatives from Faster R-CNN
    coco_error_analysis   nested PR curves as error sources are removed (simulated)
    detection_augmentation  flip, scale, crop and rotation with the boxes transformed

Real models (engines/_torchdet.py, cached in data/_torch/): ResNet-50 for
classification, Faster R-CNN v2 for detection, Mask R-CNN v2 for instance masks,
all torchvision's pre-trained weights. Metrics are engines/_detmetrics.py.

Source: reference/legacy_md/lesson_3_1_introduction + lesson_3_1_script.
"""
from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       load_image, slider_frames_layout)
from .. import _arch as A
from .. import _detmetrics as M
from .. import _torchdet as D

FIGURES = ["detection_task", "window_pyramid", "sliding_window", "proposal_pipeline", "detection_datasets",
           "iou_boxes", "tp_fp_fn", "pr_curve", "mean_average_precision",
           "detection_errors", "coco_error_analysis", "detection_augmentation"]

SIDE = 800          # every photo is processed at 800 px on its longer side

# Ground truth for the sheep photo ("7 sau på voll": seven sheep), drawn by hand
# at SIDE px and checked against close-ups; independent of any detector.
SHEEP_GT = np.array([[69, 214, 183, 316], [200, 218, 312, 298], [398, 189, 446, 250],
                     [507, 139, 572, 195], [527, 180, 590, 253], [584, 155, 661, 207],
                     [670, 144, 752, 204]], float)


def _photo(key):
    return load_image(key, max_side=SIDE)


def _cats():
    return D.categories("faster_rcnn")


def _box(ax, b, col, lw=2.0, ls="-", label=None, fs=8, txt="k"):
    ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col,
                               lw=lw, ls=ls))
    if label:
        ax.text(b[0] + 1, b[1] - 2, label, fontsize=fs, color=txt, va="bottom",
                bbox=dict(fc=col, ec="none", pad=1.0, alpha=0.9))


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_detection_task(ctx: BuildContext) -> Figure:
    # (a) and (b): one dominant object, as classification and localisation assume
    ram = _photo("ram")
    # (c) and (d): several objects
    img = _photo("sheep")
    icats, cats = D.categories("resnet50"), _cats()
    cls = D.classify("ram", ram)
    rdet = D.detect("ram", ram)
    det = D.detect("sheep", img)
    msk = D.instance_masks("sheep", img)
    keep = det["scores"] >= 0.5
    rh, rw = ram.shape[:2]
    sh = img.shape[0] * (0.66 - 0.12)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4),
                             gridspec_kw=dict(width_ratios=[rw / rh, rw / rh,
                                                            img.shape[1] / sh, img.shape[1] / sh],
                                              wspace=0.06))
    for ax, im in zip(axes, (ram, ram, img)):
        ax.imshow(im); _clean(ax)
    axes[0].set_title(f"(a) classification: one\nlabel for the whole image,\n"
                      f"'{icats[cls['labels'][0]]}' (ResNet-50)", fontsize=9.5)
    top = int(np.argmax(rdet["scores"]))
    _box(axes[1], rdet["boxes"][top], C["green"], lw=2.2,
         label=f"{cats[rdet['labels'][top]]} {rdet['scores'][top]:.2f}")
    axes[1].set_title("(b) localisation: a bounding\nbox around the main object\n(Faster R-CNN)",
                      fontsize=9.5)
    for b, l, s in zip(det["boxes"][keep], det["labels"][keep], det["scores"][keep]):
        _box(axes[2], b, C["green"], lw=1.6, label=f"{cats[l]} {s:.2f}", fs=6.5)
    axes[2].set_title(f"(c) detection: every object,\n{keep.sum()} boxes with classes\n"
                      "(Faster R-CNN)", fontsize=9.5)
    over = img.astype(float).copy()
    for i, m in enumerate(msk["masks"]):
        col = np.array(plt.matplotlib.colors.to_rgb(SERIES[i % len(SERIES)])) * 255
        over[m] = 0.45 * over[m] + 0.55 * col
    axes[3].imshow(over.astype(np.uint8)); _clean(axes[3])
    axes[3].set_title(f"(d) instance segmentation: a mask\nfor each of the {len(msk['masks'])} "
                      "sheep\n(Mask R-CNN)", fontsize=9.5)
    for ax in axes[2:]:
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0] * 0.66, img.shape[0] * 0.12)
    png = save_mpl(fig, ctx, "detection_task")
    return Figure("detection_task",
                  "Four recognition tasks, each produced by a pre-trained network. (a) In image "
                  "classification a single label is assigned to the whole image, which should "
                  "contain one dominant object; the ImageNet classifier gives "
                  f"'{icats[cls['labels'][0]]}' as the most probable of its 1,000 classes. (b) In "
                  "**localisation** a bounding box is placed around the main object in the "
                  "image, the object that defines the class of the image. (c) In object "
                  "detection every object in an image is found, and a class, a bounding box "
                  f"and a confidence are output for each: {keep.sum()} boxes with a confidence "
                  "above 0.5 are output for this photograph, including two false positives that "
                  "are discussed below. (d) In instance segmentation the pixels of each object "
                  "are also identified, with a separate mask for each instance. Object detection "
                  "is more difficult than classification and localisation, as both the class "
                  "and the location of an unknown number of objects must be predicted.",
                  png=png)


def sliding_window_count(h, w, win=128, stride=32, scale=1.25, ratios=(1.0,)):
    """Windows evaluated by a sliding-window detector over an image pyramid."""
    levels = []
    s = 1.0
    while min(h / s, w / s) >= win:
        hh, ww = int(h / s), int(w / s)
        n = ((hh - win) // stride + 1) * ((ww - win) // stride + 1)
        levels.append((s, hh, ww, n * len(ratios)))
        s *= scale
    return levels


def fig_sliding_window(ctx: BuildContext) -> Figure:
    img = _photo("sheep")
    h, w = img.shape[:2]
    win, stride, ratios = 128, 32, (0.5, 1.0, 2.0)
    levels = sliding_window_count(h, w, win, stride, ratios=ratios)
    total = sum(l[3] for l in levels)
    macs = A.resnet(50).macs
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.0), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax = axes[0]
    ax.imshow(img); _clean(ax)
    for x in range(0, w - win + 1, stride * 4):
        for y in range(0, h - win + 1, stride * 4):
            ax.add_patch(plt.Rectangle((x, y), win, win, fill=False, ec="white", lw=0.6,
                                       alpha=0.7))
    ax.add_patch(plt.Rectangle((stride * 6, stride * 6), win, win, fill=False,
                               ec=C["red"], lw=2.2))
    ax.set_title(f"(a) a {win} × {win} window moved in steps of {stride} px "
                 f"(every 4th position drawn)", fontsize=9.5)
    ax = axes[1]
    names = [f"level {i}\n{hh}×{ww}" for i, (_, hh, ww, _) in enumerate(levels)]
    ax.bar(range(len(levels)), [l[3] for l in levels], color=C["blue"])
    for i, l in enumerate(levels):
        ax.text(i, l[3], f"{l[3]:,}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(f"windows × {len(ratios)} aspect ratios")
    ax.set_title(f"(b) image pyramid, scale 1/1.25 per level: {total:,} windows,\n"
                 f"{total * macs / 1e12:.0f} T multiply-accumulates with ResNet-50 "
                 "on each", fontsize=9.5)
    png = save_mpl(fig, ctx, "sliding_window")
    return Figure("sliding_window",
                  "The naive approach to object detection: a pre-trained image classifier "
                  "is applied to every position of a sliding window. (a) A window of "
                  f"{win} × {win} pixels is moved across the {w} × {h} image in steps of "
                  f"{stride} pixels. (b) To detect objects of different sizes, the image is "
                  "repeatedly reduced in an image pyramid, and to detect objects of "
                  f"different shapes, {len(ratios)} aspect ratios are used at each position. "
                  f"In total {total:,} windows must be classified. With ResNet-50, which "
                  f"requires {macs / 1e9:.1f} G multiply-accumulates per window, approximately "
                  f"{total * macs / 1e12:.0f} × 10¹² multiply-accumulates are required for "
                  "this one image, which is far too slow for practical use.",
                  png=png)


def fig_window_pyramid(ctx: BuildContext) -> Figure:
    """A fixed-size window moved over every level of an image pyramid: each
    object is only matched by the window at the level where its size fits."""
    img = load_image("dog_ball", max_side=SIDE)
    det = D.detect("dog_ball", img)
    cats, icats = _cats(), D.categories("resnet50")
    names = [cats[l] for l in det["labels"]]
    objs = {"ball": det["boxes"][names.index("sports ball")], "dog": det["boxes"][names.index("dog")]}
    win, stride, factor, L = 128, 16, 1.6, 4
    levels, best = [], {}
    for k in range(L):
        sc = factor ** -k
        im = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)),
                        interpolation=cv2.INTER_AREA)
        h, w = im.shape[:2]
        xs, ys = np.arange(0, w - win + 1, stride), np.arange(0, h - win + 1, stride)
        wins = np.array([[x, y, x + win, y + win] for y in ys for x in xs], float)
        levels.append((sc, im, wins))
        for name, b in objs.items():
            v = M.iou(wins, b[None] * sc)[:, 0]
            i = int(np.argmax(v))
            if name not in best or v[i] > best[name][1]:
                best[name] = (k, float(v[i]), wins[i])
    # the claim of the figure: the ball fits the window at full size, the dog only when reduced
    assert best["ball"][0] == 0 and best["dog"][0] == L - 1, best
    assert best["ball"][1] > 0.5 and best["dog"][1] > 0.5, best
    total = sum(len(l[2]) for l in levels)
    widths = [l[1].shape[1] for l in levels]
    fig = plt.figure(figsize=(16, 5.0))
    gs = fig.add_gridspec(1, L, width_ratios=widths, wspace=0.08)
    for k, (sc, im, wins) in enumerate(levels):
        ax = fig.add_subplot(gs[k])
        ax.imshow(im); _clean(ax)
        for sp in ax.spines.values():
            sp.set_visible(False)
        # every 4th window position (in both directions), faint
        for wb in wins:
            if (wb[0] / stride) % 4 == 0 and (wb[1] / stride) % 4 == 0:
                ax.add_patch(plt.Rectangle(wb[:2], win, win, fill=False, ec="white", lw=0.5,
                                           alpha=0.5))
        ax.add_patch(plt.Rectangle((2, 2), win, win, fill=False, ec=C["yellow"], lw=2.2))
        for name, (kk, v, wb) in best.items():
            if kk == k:
                col = C["green"]
                ax.add_patch(plt.Rectangle(wb[:2], win, win, fill=False, ec=col, lw=2.6))
                ax.text(wb[0] + 2, wb[3] - 3, f"{name}: IoU {v:.2f}", fontsize=8, va="bottom",
                        bbox=dict(fc=col, ec="none", pad=1.0, alpha=0.9))
        ax.set_title(f"level {k}: {im.shape[1]} × {im.shape[0]}\n{len(wins):,} window positions",
                     fontsize=10)
    png = save_mpl(fig, ctx, "window_pyramid")
    (kb, vb, _), (kd, vd, _) = best["ball"], best["dog"]
    return Figure("window_pyramid",
                  "**Naive object detection** with a sliding window and an image pyramid, "
                  "on a real photograph. A window of fixed size, "
                  f"{win} × {win} pixels (yellow), is moved over the image in steps of "
                  f"{stride} pixels (every fourth position is drawn), and the region under "
                  "each position is passed to an image classifier. The ball is roughly the "
                  f"size of the window at full resolution (level 0, best IoU {vb:.2f}), but "
                  "the dog is far larger than the window, and no position covers it. The "
                  f"image is therefore reduced repeatedly, by a factor of {factor} at each "
                  "level, and the window is moved over every level of the resulting **image "
                  f"pyramid**: only at level {kd} has the dog become small enough to fit the "
                  f"window (IoU {vd:.2f}). The green windows are the positions that best fit "
                  f"each object (boxes from Faster R-CNN). In total {total:,} window "
                  "positions must be classified for this one image, almost all of them "
                  "containing no object.",
                  png=png)


def fig_proposal_pipeline(ctx: BuildContext) -> Figure:
    img = _photo("sheep")
    icats = D.categories("resnet50")
    rng = np.random.default_rng(3)
    # example proposals: jittered boxes around three objects and two background regions
    props = [SHEEP_GT[0] + [-6, -4, 8, 3], SHEEP_GT[2] + [-5, -8, 6, 4],
             SHEEP_GT[6] + [-4, -6, 5, 6], np.array([40, 30, 170, 120.0]),
             np.array([300, 330, 470, 460.0])]
    crops, labels = [], []
    for i, b in enumerate(props):
        x1, y1, x2, y2 = np.round(b).astype(int)
        crop = cv2.resize(img[y1:y2, x1:x2], (224, 224), interpolation=cv2.INTER_AREA)
        r = D.classify(f"sheep_proposal{i}", crop)
        crops.append(crop)
        labels.append((icats[r["labels"][0]], float(r["scores"][0])))
    fig = plt.figure(figsize=(14, 3.6))
    gs = fig.add_gridspec(2, 7, width_ratios=[2.6, 0.35] + [1] * 5, height_ratios=[1, 0.5],
                          hspace=0.05, wspace=0.08)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img); _clean(ax)
    for i, b in enumerate(props):
        _box(ax, b, SERIES[i % len(SERIES)], lw=2, label=str(i + 1), fs=8)
    ax.set_title("(a) region proposals: regions likely\nto contain an object", fontsize=9.5)
    ax = fig.add_subplot(gs[:, 1]); ax.axis("off")
    ax.annotate("", xy=(1.0, 0.5), xytext=(0.0, 0.5), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.8, color=C["ink"]))
    ax.text(0.5, 0.58, "warp to\n224 × 224", ha="center", fontsize=8.5, transform=ax.transAxes)
    for i, (crop, (lab, p)) in enumerate(zip(crops, labels)):
        ax = fig.add_subplot(gs[0, 2 + i]); ax.imshow(crop); _clean(ax)
        ax.set_title(f"region {i + 1}", fontsize=9)
        ax = fig.add_subplot(gs[1, 2 + i]); ax.axis("off")
        ax.text(0.5, 0.95, f"ResNet-50:\n'{lab}'\np = {p:.2f}", ha="center", va="top",
                fontsize=9, transform=ax.transAxes)
    png = save_mpl(fig, ctx, "proposal_pipeline")
    return Figure("proposal_pipeline",
                  "Object detection with region proposals and a pre-trained image "
                  "classifier, the approach used in R-CNN. (a) Instead of every window "
                  "position, a small number of regions likely to contain an object are "
                  "proposed (the five regions here are drawn by hand to illustrate the "
                  "idea; proposal algorithms are described in the next lesson). Each "
                  "region is warped to the fixed input size of the classifier and "
                  "classified. The regions containing a sheep are classified as "
                  f"'{labels[0][0]}', the closest of the 1000 ImageNet classes, by "
                  "ResNet-50. The background regions are also given labels: the barn is "
                  f"labelled '{labels[3][0]}' with a confidence of {labels[3][1]:.2f}, higher "
                  f"than that of {sum(l[1] < labels[3][1] for l in labels[:3])} of the "
                  "sheep. An ImageNet classifier has no 'background' class, "
                  "so it cannot reject a region, and a detector must learn one, as R-CNN "
                  "does. Three problems can be seen: the proposals are found "
                  "separately from the classifier, so the system is not trained end to "
                  "end; the regions have different sizes and shapes and must be warped; "
                  "and overlapping regions are processed by the classifier several times.",
                  png=png)


DATASETS = [
    # name, years, classes, images, objects / annotation, notes
    ("Pascal VOC", "2005–2012", "20", "11,530", "27,450 objects", "boxes, segmentation"),
    ("ImageNet (detection)", "2014–", "200", "≈ 450,000 train", "—",
     "boxes; mostly one centred object"),
    ("MS COCO", "2014–", "91 (80 used)", "≈ 330,000", "1.5 million objects, 7.3 per image",
     "boxes, instance masks"),
    ("Open Images (V6)", "2016–", "≈ 6,000", "≈ 9 million", "boxes for 600 classes",
     "boxes, instance masks"),
]


def fig_detection_datasets(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(12.5, 2.6))
    ax.axis("off")
    hdr = ["dataset", "years", "classes", "images", "annotated objects", "annotation"]
    tbl = ax.table(cellText=[list(r) for r in DATASETS], colLabels=hdr, loc="center",
                   cellLoc="left", colWidths=[0.17, 0.1, 0.11, 0.15, 0.25, 0.22])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["light"])
        if r == 0:
            cell.set_text_props(weight="bold")
    png = save_mpl(fig, ctx, "detection_datasets")
    return Figure("detection_datasets",
                  "The main object detection datasets, with the figures given in the "
                  "lecture slides. The annotation of a detection dataset is time "
                  "consuming, as a bounding box must be drawn around every object. Pascal "
                  "VOC has a small number of classes. The ImageNet detection task "
                  "contains many images of a single centred object, as the dataset was "
                  "assembled for classification. MS COCO contains real-world scenes with "
                  "several objects at different positions and scales, and is the standard "
                  "benchmark for object detection. MS COCO and Open Images also include "
                  "instance segmentation masks.",
                  png=png)


def iou_parts(p, g):
    ix1, iy1 = max(p[0], g[0]), max(p[1], g[1])
    ix2, iy2 = min(p[2], g[2]), min(p[3], g[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    ap, ag = (p[2] - p[0]) * (p[3] - p[1]), (g[2] - g[0]) * (g[3] - g[1])
    return inter, ap + ag - inter, (ix1, iy1, ix2, iy2)


def fig_iou_boxes(ctx: BuildContext) -> Figure:
    gt = (20, 20, 70, 60)
    preds = [(25, 24, 72, 62), (40, 35, 95, 80), (75, 30, 110, 70)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, p in zip(axes, preds):
        inter, union, ib = iou_parts(p, gt)
        for b, col, nm in [(gt, C["green"], "ground truth"), (p, C["blue"], "prediction")]:
            ax.add_patch(plt.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fc=col, alpha=0.18,
                                       ec=col, lw=2, label=nm))
        if inter:
            ax.add_patch(plt.Rectangle(ib[:2], ib[2] - ib[0], ib[3] - ib[1], fc=C["red"],
                                       alpha=0.45, ec="none", label="intersection"))
        ax.set_xlim(10, 115); ax.set_ylim(90, 10); ax.set_aspect("equal")
        ax.set_title(f"IoU = {inter} / {union} = {inter / union:.2f}", fontsize=10.5)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8, loc="lower left")
    png = save_mpl(fig, ctx, "iou_boxes")

    frames, xs = [], list(range(0, 61, 4))
    for dx in xs:
        p = (20 + dx, 24, 70 + dx, 64)
        inter, union, ib = iou_parts(p, gt)
        shapes = [dict(type="rect", x0=gt[0], x1=gt[2], y0=gt[1], y1=gt[3],
                       line=dict(color=C["green"], width=3), fillcolor="rgba(0,158,115,0.15)"),
                  dict(type="rect", x0=p[0], x1=p[2], y0=p[1], y1=p[3],
                       line=dict(color=C["blue"], width=3), fillcolor="rgba(0,114,178,0.15)")]
        if inter:
            shapes.append(dict(type="rect", x0=ib[0], x1=ib[2], y0=ib[1], y1=ib[3],
                               line=dict(width=0), fillcolor="rgba(213,94,0,0.45)"))
        frames.append(go.Frame(name=str(dx), layout=go.Layout(
            shapes=shapes, title=f"shift {dx} px: intersection {inter}, union {union}, "
                                 f"IoU = {inter / union:.2f}")))
    fig2 = go.Figure(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(opacity=0)))
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(xs, "prediction shifted by (px): "),
                       shapes=frames[0].layout.shapes, title=frames[0].layout.title.text,
                       showlegend=False)
    fig2.update_xaxes(range=[10, 140], constrain="domain")
    fig2.update_yaxes(range=[80, 10], scaleanchor="x")
    return Figure("iou_boxes",
                  "**Intersection over Union** (IoU), the measure of how well a predicted "
                  "bounding box matches the ground truth: the area of the intersection "
                  "of the two boxes divided by the area of their union. The IoU is 1 for "
                  "a perfect match, decreases as the overlap is reduced, and is 0 when the "
                  "boxes do not overlap. The union is calculated as the sum of the two "
                  "box areas minus the intersection. In the interactive version the "
                  "prediction is moved across the ground truth with the slider.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def _score(boxes, scores, gt, t1, t2):
    order, is_tp, best, matched = M.match(boxes, scores, gt, t_conf=t1, t_iou=t2)
    tp, fp, fn = int(is_tp.sum()), int((~is_tp).sum()), int((~matched).sum())
    return order, is_tp, best, matched, tp, fp, fn


def fig_tp_fp_fn(ctx: BuildContext) -> Figure:
    img = _photo("sheep")
    det = D.detect("sheep", img)
    cats = _cats()
    # a detection of another class cannot match a sheep: treat it as IoU 0
    sheep = np.array([cats[l] == "sheep" for l in det["labels"]])
    boxes = det["boxes"].copy()
    boxes[~sheep] = [-10, -10, -9, -9]
    settings = [(0.5, 0.5), (0.2, 0.5), (0.5, 0.95)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    res = []
    for ax, (t1, t2) in zip(axes, settings):
        order, is_tp, best, matched, tp, fp, fn = _score(boxes, det["scores"], SHEEP_GT, t1, t2)
        res.append((tp, fp, fn))
        ax.imshow(img); _clean(ax)
        for g, m in zip(SHEEP_GT, matched):
            if not m:
                _box(ax, g, C["orange"], lw=2.2, ls="--", label="FN", fs=7)
        for i, t, bi in zip(order, is_tp, best):
            b = det["boxes"][i]
            lab = f"{'TP' if t else 'FP'} {cats[det['labels'][i]]} {det['scores'][i]:.2f}"
            _box(ax, b, C["green"] if t else C["red"], lw=1.6, label=lab, fs=6)
        p = tp / max(tp + fp, 1); r = tp / len(SHEEP_GT)
        ax.set_title(f"T1 = {t1}, T2 = {t2}: TP {tp}, FP {fp}, FN {fn}\n"
                     f"precision {p:.2f}, recall {r:.2f}", fontsize=10)
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0] * 0.64, img.shape[0] * 0.12)
    # the claims of the caption
    assert res[0][2] == 0 and res[1][1] > res[0][1] and res[2][2] > 0, res
    png = save_mpl(fig, ctx, "tp_fp_fn")
    (tp0, fp0, fn0), (tp1, fp1, fn1), (tp2, fp2, fn2) = res
    return Figure("tp_fp_fn",
                  "Real detections from Faster R-CNN scored against ground truth drawn by "
                  "hand for the seven sheep. A prediction with a confidence below the "
                  "threshold T1 is ignored. A prediction above T1 is a **true positive** "
                  "(TP) if its IoU with a ground-truth box exceeds the threshold T2, and a "
                  "**false positive** (FP) otherwise; an object that is not detected by "
                  "any prediction is a **false negative** (FN, dashed). Left: with "
                  f"T1 = 0.5 and T2 = 0.5, all seven sheep are found, and {fp0} false "
                  "positives are produced: a box around a single head where two sheep "
                  "overlap, and a 'bird' on a white object on a post beside the barn. Centre: when T1 is "
                  f"reduced to 0.2, more low-confidence predictions are kept and the false "
                  f"positives increase to {fp1}, so the precision falls. Right: the boxes "
                  "of this detector are very accurate, with IoU values above 0.9, but when "
                  "T2 is raised to 0.95, the strictest threshold used in the COCO "
                  f"benchmark, {fn2} boxes no longer count as matches, giving {fp2} false "
                  f"positives in total and {fn2} false negatives. The choice of both "
                  "thresholds therefore affects the balance "
                  "between precision and recall.",
                  png=png)


def simulated_detections(n_gt, n_bg, quality, seed):
    """A detector's ranked output for one class over a test set, for teaching
    the PR curve: each object is detected with probability `quality`, with an
    IoU and a confidence that both increase with detection quality; background
    false positives and a few duplicates have lower confidence."""
    rng = np.random.default_rng(seed)
    rows = []                                   # (confidence, IoU, gt index or -1)
    for g in range(n_gt):
        if rng.random() < quality:
            q = rng.beta(4 * quality + 1, 2)
            rows.append((0.25 + 0.7 * q + rng.normal(0, 0.12), 0.3 + 0.65 * q, g))
            if rng.random() < 0.15:               # a duplicate of the same object
                rows.append((0.2 + 0.5 * rng.random(), 0.3 + 0.4 * rng.random(), g))
    for _ in range(n_bg):
        rows.append((0.1 + 0.8 * rng.beta(2, 2.5), 0.0, -1))
    rows = sorted(((float(np.clip(c, 0.01, 0.99)), float(i), g) for c, i, g in rows),
                  reverse=True)
    return rows


def rank_and_score(rows, n_gt, t_iou=0.5):
    """TP/FP down a ranked list: TP if IoU > T2 and the object is not yet found."""
    found, tp = set(), []
    for conf, iou, g in rows:
        ok = g >= 0 and iou > t_iou and g not in found
        if ok:
            found.add(g)
        tp.append(ok)
    tp = np.array(tp)
    p, r = M.pr_curve(tp, n_gt)
    return tp, p, r


def fig_pr_curve(ctx: BuildContext) -> Figure:
    n_gt = 10
    rows = simulated_detections(n_gt, 7, 0.85, seed=3)
    tp, p, r = rank_and_score(rows, n_gt)
    assert np.sum(np.diff(p) > 0) >= 3, "the example must show a zig-zag"
    pi = M.interpolate(p)
    ap = M.average_precision(p, r)
    fig = plt.figure(figsize=(15, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    cells = [[str(k + 1), f"{c:.2f}", f"{i:.2f}" if g >= 0 else "—", "TP" if t else "FP",
              f"{int(tp[:k + 1].sum())}", f"{int((~tp[:k + 1]).sum())}", f"{pp:.2f}", f"{rr:.2f}"]
             for k, ((c, i, g), t, pp, rr) in enumerate(zip(rows, tp, p, r))]
    tbl = ax.table(cellText=cells, colLabels=["rank", "conf", "IoU", "", "ΣTP", "ΣFP", "P", "R"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.2); tbl.scale(1, 1.08)
    for (rr_, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["light"])
        if rr_ == 0:
            cell.set_text_props(weight="bold")
        elif cc == 3:
            cell.set_text_props(color=C["green"] if tp[rr_ - 1] else C["red"], weight="bold")
    ax.set_title(f"(a) detections ranked by confidence\n{n_gt} objects, T2 = 0.5", fontsize=10)
    ax = fig.add_subplot(gs[1])
    ax.plot(r, p, "-o", color=C["blue"], ms=3.5, lw=1.4, label="precision–recall")
    ax.step(np.concatenate([[0], r]), np.concatenate([[pi[0]], pi]), where="pre",
            color=C["red"], lw=2, label="interpolated")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.legend(fontsize=8.5, loc="lower left")
    ax.set_title("(b) zig-zag curve and interpolated precision", fontsize=10)
    ax = fig.add_subplot(gs[2])
    rr = np.concatenate([[0], r])
    for k in range(len(r)):
        if rr[k + 1] > rr[k]:
            ax.add_patch(plt.Rectangle((rr[k], 0), rr[k + 1] - rr[k], pi[k],
                                       fc=C["sky"], ec=C["blue"], lw=0.8, alpha=0.6))
    ax.step(rr, np.concatenate([[pi[0]], pi]), where="pre", color=C["red"], lw=2)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.set_xlabel("recall"); ax.set_ylabel("interpolated precision")
    ax.set_title(f"(c) average precision = shaded area = {ap:.3f}", fontsize=10)
    png = save_mpl(fig, ctx, "pr_curve")
    return Figure("pr_curve",
                  "From a ranked list of detections to average precision, for one class. "
                  "(a) The predictions of a detector over a test set containing "
                  f"{n_gt} objects are ranked by confidence, and each is labelled TP or FP "
                  "with T2 = 0.5 (a second detection of an object that has already been "
                  "found is a false positive). Moving down the list is equivalent to "
                  "reducing the confidence threshold T1: the cumulative precision and "
                  "recall are calculated at each row. (b) As the threshold is reduced, "
                  "the recall increases monotonically while the precision moves up and "
                  "down, giving the characteristic zig-zag curve. The interpolated "
                  r"precision $p_{interp}(r)$ is the maximum precision at any recall of "
                  r"$r$ or greater. (c) The average precision is the area under the "
                  "interpolated curve, the sum of the rectangles defined by each "
                  f"increase in recall, here AP = {ap:.3f}. (The detections are simulated "
                  "for a clear example; the calculation is the one used by the Pascal "
                  "VOC benchmark.)",
                  png=png)


def fig_mean_average_precision(ctx: BuildContext) -> Figure:
    classes = [("sheep", 0.9, 14, 4), ("bird", 0.6, 12, 9), ("bicycle", 0.75, 10, 6),
               ("chair", 0.45, 16, 12), ("person", 0.85, 20, 6)]
    aps = {0.5: [], 0.75: []}
    for k, (name, q, n_gt, n_bg) in enumerate(classes):
        rows = simulated_detections(n_gt, n_bg, q, seed=20 + k)
        for t in aps:
            _, p, r = rank_and_score(rows, n_gt, t)
            aps[t].append(M.average_precision(p, r))
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    x = np.arange(len(classes))
    for j, (t, col) in enumerate([(0.5, C["blue"]), (0.75, C["orange"])]):
        ax.bar(x + (j - 0.5) * 0.38, aps[t], 0.36, color=col,
               label=f"IoU > {t}:  mAP = {np.mean(aps[t]):.2f}")
        ax.axhline(np.mean(aps[t]), color=col, ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in classes])
    ax.set_ylabel("average precision"); ax.set_ylim(0, 1.05); ax.legend(fontsize=9)
    ax.set_title("AP per class and mAP, at two IoU thresholds T2", fontsize=10.5)
    assert np.mean(aps[0.75]) < np.mean(aps[0.5])
    png = save_mpl(fig, ctx, "mean_average_precision")
    m50, m75 = np.mean(aps[0.5]), np.mean(aps[0.75])
    return Figure("mean_average_precision",
                  "Average precision is defined for one class. With several classes, the "
                  "**mean average precision** (mAP) is the mean of the AP values of the "
                  "individual classes. The AP varies between classes: classes that are "
                  "small, often occluded or similar in appearance to other classes, such "
                  "as the chair here, are detected less well. The AP and mAP also depend "
                  "on the IoU threshold T2 at which the PR curves were obtained: when T2 "
                  f"is raised from 0.5 to 0.75 the mAP falls from {m50:.2f} to {m75:.2f}, "
                  "as fewer predictions are placed accurately enough to count as true "
                  "positives. The IoU threshold must therefore always be stated when "
                  "detectors are compared. (Simulated detections, scored as in Figure 8.)",
                  png=png)


def fig_detection_errors(ctx: BuildContext) -> Figure:
    cats = _cats()
    sheep, horses, cows = _photo("sheep"), _photo("horses"), _photo("dairy_cows")
    ds, dh, dc = D.detect("sheep", sheep), D.detect("horses", horses), D.detect("dairy_cows", cows)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))

    def show(ax, img, det, pick, crop, title):
        ax.imshow(img); _clean(ax)
        for i in pick:
            b, l, s = det["boxes"][i], det["labels"][i], det["scores"][i]
            col = C["red"] if s >= 0.5 else C["orange"]
            _box(ax, b, col, lw=2, label=f"{cats[l]} {s:.2f}", fs=8)
        ax.set_xlim(crop[0], crop[2]); ax.set_ylim(crop[3], crop[1])
        ax.set_title(title, fontsize=10)

    # (a) localisation error: a box around one head where two sheep overlap
    m_s = D.iou(ds["boxes"], SHEEP_GT).max(1)
    loc = [i for i, (l, s, v) in enumerate(zip(ds["labels"], ds["scores"], m_s))
           if cats[l] == "sheep" and s >= 0.5 and 0.05 < v < 0.5]
    for g in SHEEP_GT[3:5]:
        _box(axes[0, 0], g, "white", lw=1.4, ls="--")
    show(axes[0, 0], sheep, ds, loc, (480, 120, 620, 280),
         "(a) false positive, localisation: correct class,\nbox around one head only (IoU "
         f"{m_s[loc[0]]:.2f})")
    # (b) similarity error: a horse also labelled as elephant and cow
    sim = [i for i, (l, s) in enumerate(zip(dh["labels"], dh["scores"]))
           if cats[l] != "horse" and s >= 0.25]
    show(axes[0, 1], horses, dh, sim, (420, 0, horses.shape[1], horses.shape[0]),
         "(b) false positives, similarity: the horse is\nalso detected as "
         + " and ".join(f"'{cats[dh['labels'][i]]}'" for i in sim))
    # (c) background error: a 'bird' on the roof of the barn
    bg = [i for i, (l, s) in enumerate(zip(ds["labels"], ds["scores"]))
          if cats[l] != "sheep" and s >= 0.5]
    show(axes[1, 0], sheep, ds, bg, (60, 20, 260, 150),
         f"(c) false positive, background: '{cats[ds['labels'][bg[0]]]}' "
         f"{ds['scores'][bg[0]]:.2f} on a\nwhite object on a post beside the barn")
    # (d) false negatives: small distant cows below T1 = 0.5; truncated cow at the border
    small = [i for i, (l, s, b) in enumerate(zip(dc["labels"], dc["scores"], dc["boxes"]))
             if cats[l] == "cow" and 0.25 <= s < 0.5 and b[0] > 300]
    trunc = [i for i, (l, s, b) in enumerate(zip(dc["labels"], dc["scores"], dc["boxes"]))
             if b[0] < 5 and s >= 0.4]
    show(axes[1, 1], cows, dc, small, (300, 215, 700, 325),
         "(d) false negatives, small objects: distant cows\ndetected below T1 = 0.5 "
         f"(confidence {min(dc['scores'][small]):.2f}–{max(dc['scores'][small]):.2f}) "
         "are rejected")
    png = save_mpl(fig, ctx, "detection_errors")
    tr = [f"'{cats[dc['labels'][i]]}' {dc['scores'][i]:.2f}" for i in trunc]
    return Figure("detection_errors",
                  "Real errors made by Faster R-CNN (red: confidence above 0.5; orange: "
                  "below; dashed white: ground truth). The types of error that reduce the average precision were "
                  "identified by Hoiem et al. False positives are caused by (a) "
                  "**localisation** errors, in which the object is correctly classified but "
                  "the box is inaccurate; (b) **similarity** errors, in which an object is "
                  "confused with a similar class, which occurs easily with animals; and "
                  "(c) confusion with the **background**, in which an object is detected "
                  "in a pattern of the background. (d) False negatives are typically "
                  "caused by object size: the distant cows are found, but with "
                  "confidences below the threshold, so they are missed. Objects that are "
                  "occluded or **truncated** by the border of the image are also difficult; "
                  "in the same photograph, a cow cut off at the left edge is labelled "
                  + " and ".join(tr) + ".",
                  png=png)


def coco_analysis(seed=11):
    """Nested PR curves of the COCO analyze() function, on a simulated detector.

    Each simulated detection has a confidence, an IoU with its best ground-truth
    match and an error type (ok, loc, sim, oth, bg). The conditions are relaxed
    one at a time, as in the COCO API: C75 and C50 are ordinary PR curves at
    IoU 0.75 and 0.5; Loc counts localisation errors (IoU 0.1-0.5) as matches;
    Sim and Oth remove false positives confused with a similar class / any
    class; BG removes background false positives; FN removes all false
    negatives, so the curve reaches precision 1 at recall 1.
    """
    rng = np.random.default_rng(seed)
    n_gt = 200
    rows = []
    for g in range(n_gt):
        if rng.random() < 0.82:
            q = rng.beta(4, 2)
            kind = rng.choice(["ok", "loc"], p=[0.8, 0.2])
            iou = 0.5 + 0.48 * q if kind == "ok" else 0.12 + 0.36 * rng.random()
            rows.append((0.3 + 0.65 * q + rng.normal(0, 0.05), iou, kind, g))
    for kind, n in [("sim", 18), ("oth", 12), ("bg", 45)]:
        for _ in range(n):
            rows.append((0.1 + 0.7 * rng.beta(1.6, 3), 0.0, kind, -1))
    rows.sort(key=lambda r: -r[0])
    curves = {}
    for name in ["C75", "C50", "Loc", "Sim", "Oth", "BG", "FN"]:
        tp = []
        found = set()
        for conf, iou, kind, g in rows:
            if name in ("Sim", "Oth", "BG", "FN") and kind == "sim":
                continue                         # ignored, not counted as FP
            if name in ("Oth", "BG", "FN") and kind == "oth":
                continue
            if name in ("BG", "FN") and kind == "bg":
                continue
            thr = {"C75": 0.75, "C50": 0.5}.get(name, 0.1)
            ok = g >= 0 and iou > thr and g not in found
            if ok:
                found.add(g)
            tp.append(ok)
        tp = np.array(tp)
        n = n_gt if name != "FN" else int(tp.sum())       # FN: missed objects removed
        p, r = M.pr_curve(tp, n)
        curves[name] = (r, M.interpolate(p), M.average_precision(p, r))
    return curves


def fig_coco_error_analysis(ctx: BuildContext) -> Figure:
    curves = coco_analysis()
    cols = {"C75": C["light"], "C50": C["sky"], "Loc": C["blue"], "Sim": C["red"],
            "Oth": C["green"], "BG": C["purple"], "FN": C["orange"]}
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    names = list(curves)
    prev_r, prev_p = np.array([0, 1]), np.array([0, 0])
    for name in names:
        r, p, ap = curves[name]
        rr = np.concatenate([[0], r, [r[-1], 1.0]]); pp = np.concatenate([[p[0]], p, [0, 0]])
        ax.fill_between(rr, 0, pp, step="pre", color=cols[name], alpha=0.9, zorder=-names.index(name),
                        label=f"[{ap:.3f}] {name}", ec=C["grey"], lw=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.legend(fontsize=9, loc="lower left", title="[AP] condition", title_fontsize=9)
    ax.set_title("Error analysis in the style of the COCO API (simulated detector)",
                 fontsize=10.5)
    aps = [curves[n][2] for n in names]
    assert all(b >= a - 1e-9 for a, b in zip(aps, aps[1:])) and abs(aps[-1] - 1) < 1e-9, aps
    png = save_mpl(fig, ctx, "coco_error_analysis")
    gain = {n: curves[n][2] - curves[m][2] for m, n in zip(names, names[1:])}
    big = sorted(gain, key=gain.get, reverse=True)[:3]
    return Figure("coco_error_analysis",
                  "The analysis function of the COCO API, which is used to diagnose the "
                  "sources of error in a detector, applied to a simulated detector. A "
                  "series of precision–recall curves is produced with progressively "
                  "relaxed conditions, and the AP of each is given in brackets: C75 and "
                  "C50 are the PR curves at IoU thresholds of 0.75 and 0.5; Loc ignores "
                  "localisation errors (IoU 0.1); Sim removes false positives confused "
                  "with a similar class (same supercategory); Oth removes all class "
                  "confusions; BG removes all background false positives; and FN removes "
                  "all false negatives, so that the AP becomes 1. The increase in area at "
                  "each step shows how much the AP would improve if that source of error "
                  "were removed. For this detector the largest gains come from "
                  + ", ".join(big[:-1]) + f" and {big[-1]}, which indicates where the "
                  "effort to improve it should be directed.",
                  png=png)


def transform_boxes(boxes, Mat):
    """Apply a 2 × 3 affine matrix to boxes: transform the four corners and take
    their axis-aligned bounding box (the only possible box after a rotation)."""
    out = []
    for x1, y1, x2, y2 in boxes:
        pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]]) @ Mat.T
        out.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
    return np.array(out)


def fig_detection_augmentation(ctx: BuildContext) -> Figure:
    img = _photo("sheep")
    h, w = img.shape[:2]
    cases = [("original", np.array([[1, 0, 0], [0, 1, 0]], float), 1.0),
             ("horizontal flip", np.array([[-1, 0, w], [0, 1, 0]], float), 1.0),
             ("scale 1.4 and crop", np.array([[1.4, 0, -0.2 * w], [0, 1.4, -0.3 * h]], float), 1.0),
             ("rotation 10°", cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0), 1.0),
             ("brightness × 0.6", np.array([[1, 0, 0], [0, 1, 0]], float), 0.6)]
    fig, axes = plt.subplots(1, len(cases), figsize=(16, 2.9))
    for ax, (name, Mat, gain) in zip(axes, cases):
        out = cv2.warpAffine(img, Mat, (w, h), borderMode=cv2.BORDER_CONSTANT)
        out = np.clip(out.astype(float) * gain, 0, 255).astype(np.uint8)
        bx = transform_boxes(SHEEP_GT, Mat)
        bx[:, [0, 2]] = bx[:, [0, 2]].clip(0, w); bx[:, [1, 3]] = bx[:, [1, 3]].clip(0, h)
        ax.imshow(out); _clean(ax)
        for b in bx:
            if (b[2] - b[0]) > 4 and (b[3] - b[1]) > 4:
                _box(ax, b, C["green"], lw=1.4)
        ax.set_title(name, fontsize=10)
    png = save_mpl(fig, ctx, "detection_augmentation")
    return Figure("detection_augmentation",
                  "Data augmentation for object detection. Photometric operations such as "
                  "a change of brightness or contrast leave the bounding boxes unchanged, "
                  "but every geometric operation must also be applied to the boxes. "
                  "Here each box is transformed with the same matrix as the image: its "
                  "four corners are transformed and the box enclosing them is taken. A "
                  "flip and a change of scale give exact boxes, and boxes are clipped at "
                  "the border of a crop, where an object may be lost altogether. After a "
                  "rotation the enclosing box is larger than the object, which is one "
                  "reason why simple policies of scaling, translation, flipping and random "
                  "cropping are used in many detection systems.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_detection_task(ctx), fig_window_pyramid(ctx), fig_sliding_window(ctx), fig_proposal_pipeline(ctx),
            fig_detection_datasets(ctx), fig_iou_boxes(ctx), fig_tp_fp_fn(ctx),
            fig_pr_curve(ctx), fig_mean_average_precision(ctx), fig_detection_errors(ctx),
            fig_coco_error_analysis(ctx), fig_detection_augmentation(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("3.1 Object detection metrics")
    img = _photo("sheep")
    det = D.detect("sheep", img)
    cats = _cats()
    t1 = st.slider("confidence threshold T1", 0.05, 0.95, 0.5, 0.05)
    t2 = st.slider("IoU threshold T2", 0.1, 0.95, 0.5, 0.05)
    boxes = det["boxes"].copy()
    boxes[[cats[l] != "sheep" for l in det["labels"]]] = [-10, -10, -9, -9]
    order, is_tp, best, matched, tp, fp, fn = _score(boxes, det["scores"], SHEEP_GT, t1, t2)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.imshow(img); _clean(ax)
    for g, m in zip(SHEEP_GT, matched):
        if not m:
            _box(ax, g, C["orange"], ls="--", label="FN")
    for i, t in zip(order, is_tp):
        _box(ax, det["boxes"][i], C["green"] if t else C["red"],
             label=f"{cats[det['labels'][i]]} {det['scores'][i]:.2f}", fs=6)
    st.pyplot(fig)
    st.write(f"TP {tp}, FP {fp}, FN {fn}: precision {tp / max(tp + fp, 1):.2f}, "
             f"recall {tp / len(SHEEP_GT):.2f}")
