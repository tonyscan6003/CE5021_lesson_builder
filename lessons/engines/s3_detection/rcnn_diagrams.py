"""Lesson 3.3 — the slides' architecture diagrams, redrawn with real data.

Figures (built by two_stage.build, in lesson order)
    rcnn_pipeline          slide 2: proposals + image classifier, on the slides' cyclist
    transfer_features      slide 3: engineered vs learned features for the SVM
    rcnn_training          slide 4: training the CNN, training the SVMs, inference
    rcnn_variants          slide 5: SVM + regressor vs fully connected heads
    spp_diagram            slide 6: warping vs SPP, and the SPP layer as volumes
    fast_rcnn_pipeline     slide 7: Fast R-CNN, and RoI pooling to 7 x 7 x C
    cam_mechanism          slide 8: how a class activation map is formed (real values)
    faster_rcnn_pipeline   slide 10: Faster R-CNN with its region proposal network
    stage_comparison       slide 11: two-stage vs single-stage

Photos, proposals, detections and feature values are real (selective search
from lesson 3.2, torchvision's pre-trained networks via engines/_torchdet.py);
the drawing primitives are engines/_blocks.py and _rcnn_draw.py.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog

from .._common import BuildContext, Figure, C, SERIES, save_mpl, load_image
from .. import _blocks as B
from .. import _detmetrics as M
from .. import _torchdet as D
from .._volumes import OBLIQUE, volume
from . import _rcnn_draw as R
from .region_proposals import selective_search

FS = 9


@lru_cache(maxsize=None)
def cyclist():
    """The slides' cyclist at 400 px, its selective-search proposals, and
    Faster R-CNN's person and bicycle boxes (used only to choose which
    proposals to show)."""
    img = load_image("cyclist", max_side=400)
    boxes = selective_search(img)[3]
    det = D.detect("cyclist", load_image("cyclist", max_side=800))
    cats = D.categories("faster_rcnn")
    big = {}
    for b, l, s in zip(det["boxes"] / 2, det["labels"], det["scores"]):
        name = cats[l]
        if name in ("person", "bicycle") and s > 0.9:
            if name not in big or (b[2] - b[0]) * (b[3] - b[1]) > \
                    (big[name][2] - big[name][0]) * (big[name][3] - big[name][1]):
                big[name] = b
    return img, boxes, big


def _crop(img, b, n=224):
    x1, y1, x2, y2 = np.round(b).astype(int)
    return cv2.resize(img[max(y1, 0):y2, max(x1, 0):x2], (n, n), interpolation=cv2.INTER_AREA)


def _best(boxes, target):
    v = M.iou(boxes, target[None])[:, 0]
    return int(np.argmax(v)), float(v.max())


def fig_rcnn_pipeline(ctx: BuildContext) -> Figure:
    img, boxes, big = cyclist()
    icats = D.categories("resnet50")
    k = 40
    chosen, labels = [], []
    for name in ("person", "bicycle"):
        i, v = _best(boxes, big[name])
        r = D.classify(f"cyclist_ss{i}", _crop(img, boxes[i]))
        chosen.append(boxes[i])
        labels.append(f"'{icats[r['labels'][0]]}' {r['scores'][0]:.2f}")
    fig, ax = B.canvas(17.2, 3.4, (17, 3.4))
    y, h = 0.55, 2.2
    p0 = B.photo(ax, img, 0.1, y, h, label="input image", fs=FS)
    B.arrow(ax, p0[2] + 0.08, y + h / 2, p0[2] + 0.42, y + h / 2)
    B.block(ax, p0[2] + 0.48, y + 0.6, 1.35, 1.0, "region proposal\ngenerator\n(selective search)",
            fc="#dae8fc", fs=FS)
    x1 = p0[2] + 1.9
    B.arrow(ax, x1, y + h / 2, x1 + 0.35, y + h / 2)
    p1 = B.photo(ax, img, x1 + 0.42, y, h, boxes=boxes[:k], box_col="#3c78d8", lw=0.7,
                 label=f"region proposals (first {k} of {len(boxes)})", fs=FS)
    B.arrow(ax, p1[2] + 0.08, y + h / 2, p1[2] + 0.42, y + h / 2)
    B.block(ax, p1[2] + 0.48, y + 0.6, 1.35, 1.0, "image classifier\n(ResNet-50, each\nregion warped)",
            fc="#dae8fc", fs=FS)
    x2 = p1[2] + 1.9
    B.arrow(ax, x2, y + h / 2, x2 + 0.35, y + h / 2)
    p2 = B.photo(ax, img, x2 + 0.42, y, h, boxes=chosen, box_col=[C["red"], C["green"]], lw=1.8,
                 box_labels=labels, label="labelled regions (objects)", fs=FS)
    ax.set_xlim(0, p2[2] + 0.1)
    png = save_mpl(fig, ctx, "rcnn_pipeline")
    return Figure("rcnn_pipeline",
                  "The basic deep learning object detection pipeline, with region proposals "
                  "and a pre-trained image classifier, run on the photograph used in the "
                  "lecture slides. Selective search (Lesson 3.2) proposes "
                  f"{len(boxes)} regions; each is warped to the input size of an image "
                  "classifier and classified. The two regions shown on the right are the "
                  "proposals that best fit the cyclist and the bicycle, with the label and "
                  "probability given by ResNet-50, an ImageNet classifier: the regions are "
                  "found, but an ImageNet classifier gives its closest class rather than "
                  "'person' or 'bicycle', and has no 'background' class with which to reject "
                  "the other proposals.",
                  png=png)


def _patches():
    sheep = load_image("sheep", max_side=800)
    friends = load_image("friends", max_side=800)
    tanker = load_image("tanker", max_side=800)
    city = load_image("liege_panorama", max_side=1200)
    return [("barn", sheep[0:70, 0:105]), ("forest", friends[0:260, 250:510]),
            ("water", tanker[300:438, 0:300]), ("city", city[100:300, 500:800])]


def fig_transfer_features(ctx: BuildContext) -> Figure:
    pats = _patches()
    fig, ax = B.canvas(15.2, 4.4, (15, 4.4))
    th = 0.85
    for i, (name, p) in enumerate(pats):
        x0 = 0.15 + (i % 2) * 1.35
        y0 = 2.35 - (i // 2) * 1.2
        q = cv2.resize(p, (int(th * 120), int(th * 100)))
        B.photo(ax, q, x0, y0, th, label=f"'{name}'", fs=FS)
    ax.text(1.4, 4.2, "image patches and labels", ha="center", fontsize=FS)
    hogs, cnns = [], []
    for name, p in pats:
        g = cv2.cvtColor(cv2.resize(p, (64, 64)), cv2.COLOR_RGB2GRAY)
        hogs.append(hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)))
        cnns.append(D.cnn_features(f"patch_{name}", cv2.resize(p, (224, 224)))["vec"])
    rows = [(3.25, "representation:\nengineered features\n(HOG, 9 orientations)", "#fff2cc",
             hogs, "classical"),
            (1.05, "pre-trained CNN\n(ResNet-50, ImageNet):\nlearned features", "#d9ead3",
             cnns, "R-CNN")]
    for yc, text, fc, vecs, tag in rows:
        B.arrow(ax, 2.95, 2.3, 3.55, yc)
        B.block(ax, 3.6, yc - 0.55, 2.4, 1.1, text, fc=fc, fs=FS)
        B.arrow(ax, 6.05, yc, 6.6, yc)
        ins = ax.inset_axes([6.7, yc - 0.6, 3.3, 1.2], transform=ax.transData)
        V = np.array(vecs)
        sel = np.sort(np.argsort(-V.mean(0))[:32])   # the 32 most active features
        for k, v in enumerate(V[:, sel]):
            ins.plot(np.arange(32), v / (np.abs(v).max() + 1e-9) + 2.3 * k,
                     color=SERIES[k], lw=1.1)
        ins.set_xticks([]); ins.set_yticks([])
        for s in ins.spines.values():
            s.set_color(C["light"])
        ax.text(8.35, yc + 0.68, f"feature vectors ({len(vecs[0]):,} values; the 32 most "
                "active shown)",
                ha="center", fontsize=FS - 0.5)
        B.arrow(ax, 10.1, yc, 10.65, yc)
        B.block(ax, 10.7, yc - 0.4, 1.2, 0.8, "SVM", fc="#f4cccc", fs=FS)
        B.arrow(ax, 11.95, yc, 12.5, yc)
        ax.text(12.6, yc, f"class label\n({tag})", va="center", fontsize=FS)
    B.line_arrow(ax, 4.8, 2.5, 4.8, 1.65, col=C["grey"], style="<->", lw=1.4)
    ax.text(4.95, 2.07, "replaced by", fontsize=FS - 0.5, color=C["grey"], va="center")
    png = save_mpl(fig, ctx, "transfer_features")
    return Figure("transfer_features",
                  "**The use of convolutional features**, one of the first applications "
                  "of transfer learning. In a classical recognition system (top), image "
                  "patches are converted into feature vectors by an engineered "
                  f"representation, here HOG with {len(hogs[0])} values per patch, and the "
                  "vectors are classified by a support vector machine. In R-CNN (bottom), "
                  "the engineered representation is replaced by a CNN pre-trained on image "
                  f"classification, whose {len(cnns[0]):,} learned features per patch are "
                  "passed to the SVM. Both sets of feature vectors are computed for the "
                  "four patches shown. As the SVM is trained separately from the CNN, the "
                  "system is not trained end to end.",
                  png=png)


def fig_rcnn_training(ctx: BuildContext) -> Figure:
    img, boxes, big = cyclist()
    sheep = load_image("sheep", max_side=400)
    sheep_ss = selective_search(sheep)[3]
    van = load_image("minivan", max_side=400)
    vdet = D.detect("minivan", load_image("minivan", max_side=800))
    vbox = vdet["boxes"][int(np.argmax((vdet["boxes"][:, 2] - vdet["boxes"][:, 0]) *
                                       (vdet["scores"] > 0.9)))] / 2
    van_ss = selective_search(van)[3]
    v = M.iou(van_ss, vbox[None])[:, 0]
    negs = van_ss[(v >= 0.2) & (v <= 0.5)][:3]
    cows = load_image("dairy_cows", max_side=400)
    cows_ss = selective_search(cows)[3]
    cdet = D.detect("dairy_cows", load_image("dairy_cows", max_side=800))
    cats = D.categories("faster_rcnn")
    ck = cdet["scores"] > 0.95
    fig, ax = B.canvas(16.5, 10.0, (16, 9.6))

    # (a) training the CNN
    ya = 7.0
    ax.text(0.1, ya + 2.75, "(a) training the CNN", fontsize=10.5, weight="bold")
    ax.text(0.1, ya + 2.35, "generate region proposals for the training data", fontsize=FS,
            color=C["red"])
    p = B.photo(ax, sheep, 0.1, ya + 1.15, 1.1, boxes=sheep_ss[:12], lw=0.6)
    p2 = B.photo(ax, img, p[2] + 0.15, ya + 1.15, 1.1, boxes=boxes[:12], lw=0.6)
    B.line_arrow(ax, 1.2, ya + 0.95, 2.4, ya + 0.35, col=C["red"], ls="--")
    ax.text(1.0, ya + 0.45, "warp to n × n", fontsize=FS, color=C["red"], ha="right")
    groups = [("sheep", [_crop(sheep, RP_GT(i)) for i in (0, 1, 2)]),
              ("person", [_crop(img, big["person"])]),
              ("bicycle", [_crop(img, big["bicycle"])]),
              ("background", [_crop(sheep, np.array([150, 230, 260, 300.])),
                              _crop(img, np.array([330, 0, 400, 60.]))])]
    x = 2.5
    for name, crops in groups:
        for j, c in enumerate(crops):
            B.photo(ax, c, x + j * 0.12, ya - 0.2 + j * 0.12, 0.75)
        ax.text(x + 0.38, ya - 0.3, name, ha="center", va="top", fontsize=FS)
        x += 1.1
    ax.text(p2[2] + 0.3, ya + 1.25, "N object categories + background (warped regions, n × n)",
            fontsize=FS, color=C["red"])
    B.arrow(ax, x, ya + 0.2, x + 0.5, ya + 0.2)
    cx = x + 0.6
    for k, (ln, sd) in enumerate([(0.25, 1.5), (0.35, 1.2), (0.45, 0.95), (0.55, 0.75)]):
        cx = volume(ax, cx, ya + 0.2, ln, sd, "#eeeeee", lw=0.8) + 0.12
    for k in range(3):
        B.bar(ax, cx + k * 0.25, ya + 0.2, 1.0 - k * 0.25, w=0.12, fc="#cfe2f3")
    for k in range(4):
        ax.text(cx + 0.8, ya + 0.55 - k * 0.24, f"P{k}", fontsize=FS - 0.5, va="center")
    ax.text(cx - 0.6, ya + 1.3, "fine-tune a pre-trained CNN\nfor N + 1 categories",
            ha="center", fontsize=FS, color=C["red"])

    # (b) training the SVMs and box regressor
    yb = 4.25
    ax.text(0.1, yb + 1.3, "(b) training the SVMs and the bounding-box regressor",
            fontsize=10.5, weight="bold")
    B.photo(ax, _crop(van, vbox), 0.3, yb, 0.95, label="'car':\nground truth", fs=FS)
    ax.plot([1.55, 1.8], [yb - 0.4, yb + 1.05], color=C["grey"], lw=1, ls="--")
    for j, nb in enumerate(negs):
        B.photo(ax, _crop(van, nb), 2.0 + j * 1.05, yb, 0.95)
    ax.text(2.0 + len(negs) * 0.52, yb - 0.08,
            "'not car': proposals overlapping the car by IoU 0.2–0.5", ha="center", va="top",
            fontsize=FS)
    x = 2.0 + len(negs) * 1.05 + 0.2
    B.arrow(ax, x, yb + 0.47, x + 0.5, yb + 0.47)
    B.block(ax, x + 0.55, yb, 1.6, 0.95, "trained\nconvolution layers", fc="#ffffff", fs=FS)
    B.arrow(ax, x + 2.2, yb + 0.47, x + 2.75, yb + 0.47)
    ax.text(x + 2.47, yb + 0.25, "features", ha="center", va="top", fontsize=FS - 0.5)
    B.block(ax, x + 2.8, yb, 1.0, 0.95, "SVM", fc="#d9ead3", fs=FS)
    B.block(ax, x + 3.8, yb, 1.6, 0.95, "bounding-box\nregression", fc="#fce5cd", fs=FS)

    # (c) inference
    yc = 0.75
    ax.text(0.1, yc + 2.0, "(c) inference", fontsize=10.5, weight="bold")
    q = B.photo(ax, cows, 0.1, yc, 1.4, label="input image", fs=FS)
    B.arrow(ax, q[2] + 0.08, yc + 0.7, q[2] + 0.5, yc + 0.7)
    ax.text(q[2] + 0.29, yc + 1.45, "region\nproposals", ha="center", va="bottom", fontsize=FS - 0.5)
    q2 = B.photo(ax, cows, q[2] + 0.55, yc, 1.4, boxes=cows_ss[:25], lw=0.6)
    B.arrow(ax, q2[2] + 0.08, yc + 0.7, q2[2] + 0.5, yc + 0.7)
    ax.text(q2[2] + 0.29, yc + 1.0, "warp to\nn × n", ha="center", va="bottom", fontsize=FS - 0.5)
    for j in range(4):
        ax.add_patch(plt.Rectangle((q2[2] + 0.6 + j * 0.1, yc + 0.3 + j * 0.1), 0.75, 0.75,
                                   fc="white", ec=C["red"], lw=1))
    x = q2[2] + 1.75
    B.arrow(ax, x, yc + 0.7, x + 0.45, yc + 0.7)
    B.block(ax, x + 0.5, yc + 0.22, 1.6, 0.95, "trained\nconvolution layers", fc="#ffffff", fs=FS)
    B.arrow(ax, x + 2.15, yc + 0.7, x + 2.7, yc + 0.7)
    B.block(ax, x + 2.75, yc + 0.22, 1.0, 0.95, "SVM", fc="#d9ead3", fs=FS)
    B.block(ax, x + 3.75, yc + 0.22, 1.6, 0.95, "bounding-box\nregression", fc="#fce5cd", fs=FS)
    B.arrow(ax, x + 5.4, yc + 0.7, x + 5.95, yc + 0.7)
    B.photo(ax, cows, x + 6.0, yc, 1.4, boxes=cdet["boxes"][ck] / 2, box_col=C["yellow"]
            if "yellow" in C else C["orange"], lw=1.2, label="label and bounding box\nfor each object",
            fs=FS)
    ax.set_xlim(0, 16.6)
    png = save_mpl(fig, ctx, "rcnn_training")
    return Figure("rcnn_training",
                  "The **R-CNN** multi-stage pipeline, each stage of which must be trained "
                  "separately, illustrated with real selective-search proposals. (a) Region "
                  "proposals are generated for the training images and warped to the "
                  "fixed n × n input size, and the warped regions of the N object categories, "
                  "plus background, are used to fine-tune a pre-trained CNN (AlexNet in the "
                  "original work) for the new categories. (b) The fully connected part is "
                  "then removed and the convolution layers are used as a feature generator. "
                  "For each class, an SVM is trained on the features of the ground-truth "
                  "regions as positives and of proposals that overlap them only slightly "
                  "(IoU 0.2–0.5, here three real proposals around the car) as negatives, and "
                  "a bounding-box regressor is trained on the same features. (c) At "
                  "inference, proposals are generated for a new image, warped, passed "
                  "through the convolution layers, and classified and refined by the SVMs "
                  "and the regressor. Every proposal is processed by the CNN, which is slow, "
                  "and the features of all training proposals must be stored to train the "
                  "SVMs. (The final boxes shown are those of torchvision's Faster R-CNN, "
                  "as no SVMs were trained here.)",
                  png=png)


def RP_GT(i):
    from .region_proposals import GT
    return GT["sheep"][i]


def _warp_row(ax, img, prop, x, ymid, fs=FS, label="region\nproposals"):
    p = B.photo(ax, img, x, ymid - 0.6, 1.2, boxes=prop, lw=0.7)
    ax.text((p[0] + p[2]) / 2, ymid - 0.68, label, ha="center", va="top", fontsize=fs - 0.5)
    B.arrow(ax, p[2] + 0.06, ymid, p[2] + 0.4, ymid)
    ax.text(p[2] + 0.23, ymid + 0.2, "warp to\nn × n", ha="center", va="bottom", fontsize=fs - 0.5)
    for j in range(4):
        ax.add_patch(plt.Rectangle((p[2] + 0.48 + j * 0.09, ymid - 0.42 + j * 0.09), 0.62, 0.62,
                                   fc="white", ec=C["red"], lw=1))
    x = p[2] + 1.45
    B.arrow(ax, x, ymid, x + 0.4, ymid)
    B.block(ax, x + 0.45, ymid - 0.45, 1.2, 0.9, "conv layers", fc=B.CONV_FC, fs=fs)
    return x + 1.7


def fig_rcnn_variants(ctx: BuildContext) -> Figure:
    img, boxes, _ = cyclist()
    fig, ax = B.canvas(11.5, 4.9, (12, 4.9))
    y1, y2 = 3.6, 1.0
    x = _warp_row(ax, img, boxes[:12], 0.1, y1)
    B.arrow(ax, x, y1, x + 0.4, y1)
    B.block(ax, x + 0.45, y1 - 0.05, 1.3, 0.55, "SVM", fc="#dae8fc", fs=FS)
    B.block(ax, x + 0.45, y1 - 0.6, 1.3, 0.55, "bounding-box\nregression", fc="#f8cecc", fs=FS - 0.5)
    for dy, t in [(0.22, "class"), (-0.32, "bounding box")]:
        B.arrow(ax, x + 1.8, y1 + dy, x + 2.15, y1 + dy, width=0.1)
        ax.text(x + 2.22, y1 + dy, t, va="center", fontsize=FS)
    ax.text(0.1, y1 + 1.05, "(a) R-CNN with an SVM classifier and a bounding-box regressor",
            fontsize=10, weight="bold")
    x = _warp_row(ax, img, boxes[:12], 0.1, y2)
    B.arrow(ax, x, y2, x + 0.4, y2)
    B.bar(ax, x + 0.45, y2, 0.9, fc=B.FC_FC); B.bar(ax, x + 0.75, y2, 0.9, fc=B.FC_FC)
    ax.text(x + 0.68, y2 - 0.52, "FC layers", ha="center", va="top", fontsize=FS - 0.5)
    R.heads(ax, x + 0.95, y2, fs=FS)
    ax.text(0.1, y2 + 1.05, "(b) R-CNN with a fully connected neural network classifier and "
            "bounding-box regressor", fontsize=10, weight="bold")
    png = save_mpl(fig, ctx, "rcnn_variants")
    return Figure("rcnn_variants",
                  "Towards end-to-end learning. R-CNN and its variants are conceptually "
                  "two-stage detectors: the first stage consists of the region proposals "
                  "and the convolution layers, which convert each proposal into a feature "
                  "vector, and the second stage predicts the class and a refined bounding box "
                  "from that vector. (a) In the original R-CNN, the second stage is an SVM "
                  "and a bounding-box regressor. (b) The first step towards end-to-end "
                  "training is to replace them with fully connected layers, which perform "
                  "the same functions but can be trained together with the convolution "
                  "layers. In both cases a fixed-size vector is still required for each "
                  "proposal, so the proposals must still be warped to n × n before the "
                  "convolution layers.",
                  png=png)


def fig_spp_diagram(ctx: BuildContext) -> Figure:
    img, boxes, _ = cyclist()
    fig, ax = B.canvas(15.5, 8.6, (15, 8.4))
    # (a) and (b): warping vs SPP
    for yc, spp, title in [(7.3, False, "(a) region-proposal detection with warped regions"),
                           (5.0, True, "(b) region-proposal detection with spatial pyramid pooling")]:
        ax.text(0.1, yc + 0.95, title, fontsize=10, weight="bold")
        p = B.photo(ax, img, 0.1, yc - 0.55, 1.1, boxes=boxes[:10], lw=0.6)
        x = p[2] + 0.06
        if not spp:
            B.arrow(ax, x, yc, x + 0.4, yc)
            B.block(ax, x + 0.45, yc - 0.3, 0.8, 0.6, "warp", fc="#f4cccc", fs=FS)
            x += 1.3
            ax.text(x - 0.05, yc + 0.42, "fixed input size", fontsize=FS - 0.5, color=C["red"],
                    ha="center")
        else:
            ax.text(x + 0.75, yc + 0.42, "any input size", fontsize=FS - 0.5, color=C["green"],
                    ha="center")
        B.arrow(ax, x, yc, x + 0.4, yc)
        B.block(ax, x + 0.45, yc - 0.35, 1.1, 0.7, "conv layers", fc=B.CONV_FC, fs=FS)
        x += 1.6
        if spp:
            B.arrow(ax, x, yc, x + 0.4, yc)
            B.block(ax, x + 0.45, yc - 0.35, 1.3, 0.7, "spatial pyramid\npooling", fc="#fff2cc",
                    fs=FS - 0.5)
            x += 1.8
            ax.text(x - 0.05, yc + 0.42, "fixed output size", fontsize=FS - 0.5, color=C["green"],
                    ha="center")
        B.arrow(ax, x, yc, x + 0.4, yc)
        B.bar(ax, x + 0.45, yc, 0.7, fc=B.FC_FC); B.bar(ax, x + 0.7, yc, 0.7, fc=B.FC_FC)
        ax.text(x + 0.6, yc - 0.42, "FC", ha="center", va="top", fontsize=FS - 0.5)
        B.arrow(ax, x + 0.92, yc, x + 1.3, yc)
        ax.text(x + 1.38, yc, "class and\nbounding box", va="center", fontsize=FS)

    # (c) the SPP layer as volumes
    ax.text(0.1, 3.85, "(c) the spatial pyramid pooling layer (C channels)", fontsize=10,
            weight="bold")
    L, S = 1.0, 2.2
    ym = 2.0
    right = volume(ax, 0.5, ym, L, S, "#d9ead3", lw=1.0, alpha=0.95)
    ax.text(0.5 + L / 2, ym - S / 2 - 0.1, "last feature map\nH × W × C", ha="center", va="top",
            fontsize=FS)
    levels = [(4, 3.3, "pool window H/4 × W/4"), (2, 2.0, "pool window H/2 × W/2"),
              (1, 0.7, "pool window H × W")]
    vx = right + 2.4
    seg = []
    for n, yl, txt in levels:
        B.arrow(ax, right + 0.1, ym + (yl - ym) * 0.4, vx - 0.15, yl, width=0.12)
        ax.text((right + vx) / 2 - 0.1, (ym + yl) / 2 + 0.2, txt, fontsize=FS - 0.5,
                ha="center", rotation=0)
        sd = 0.75
        r2 = volume(ax, vx, yl, L, sd, "#e1d5e7", lw=1.0, alpha=0.95)
        B.cap_grid(ax, vx + L, yl, sd, n)
        ax.text(r2 + 0.1, yl, f"{n} × {n} × C", va="center", fontsize=FS)
        B.arrow(ax, r2 + 0.95, yl, vx + 4.1, ym + (yl - ym) * 0.3, width=0.1)
        seg.append(n * n)
    bx = vx + 4.2
    tot = sum(seg)
    yb = ym - 1.3
    for k, (n, col) in enumerate(zip(seg, [C["purple"], C["sky"], C["green"]])):
        h = 2.6 * n / tot
        ax.add_patch(plt.Rectangle((bx, yb), 0.25, h, fc=col, alpha=0.6, ec=C["ink"]))
        ax.text(bx + 0.35, yb + h / 2, f"{n} × C", va="center", fontsize=FS - 0.5)
        yb += h
    ax.text(bx + 0.12, ym - 1.38, f"concatenate:\n(16 + 4 + 1) × C = 21C values\n"
            "(10,752 for C = 512)", ha="center", va="top", fontsize=FS)
    png = save_mpl(fig, ctx, "spp_diagram")
    return Figure("spp_diagram",
                  "**Spatial pyramid pooling** (SPP). (a) The size limitation arises "
                  "from the fully connected layers, not the convolution layers, but in "
                  "R-CNN it was met by warping every proposal to a fixed input size. (b) "
                  "With an SPP layer between the convolution layers and the fully connected "
                  "layers, a region of any size can be processed, as the SPP layer always "
                  "outputs a vector of the same length. (c) The SPP layer applies max "
                  "pooling at three levels. The pooling window is scaled with the size of "
                  "the feature map, H/4 × W/4, H/2 × W/2 and H × W, so that the levels always "
                  "give 4 × 4, 2 × 2 and 1 × 1 bins per channel; these are flattened and "
                  "concatenated into one vector of 21C values, which is passed to the fully "
                  "connected layers. Figure 7 applies this to real feature maps.",
                  png=png)


def fig_fast_rcnn_pipeline(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=800)
    from .region_proposals import GT
    gt = GT["sheep"] * 2
    big = np.array([500, 130, 765, 262.])      # the group of four sheep on the right
    small = gt[2]
    fig, ax = B.canvas(13.5, 7.2, (13.5, 7.2))
    ym = 5.0
    roi = R.roi_fraction(big, img.shape)
    p = R.backbone(ax, img, 0.1, ym, box=big, fmap_roi=roi, label="input image\n(selective search)")
    R.projection_arrow(ax, p["img"], big, img.shape, p["cap_x"], ym, roi, top=ym + 1.55)
    R.roi_head(ax, p["fmap_right"] + 0.1, ym)
    ax.add_patch(plt.Circle((p["cap_x"] + 0.7, ym), 1.3, fill=False, ec=C["red"], lw=1.4))
    ax.text(0.1, ym + 1.85, "(a) Fast R-CNN", fontsize=10.5, weight="bold")
    # (b) RoI pooling of two different RoIs to the same 7 x 7 x C
    ax.text(0.1, 2.75, "(b) RoI pooling: one level of pooling, with the window adjusted so the "
            "output is always 7 × 7 × C", fontsize=10.5, weight="bold")
    for k, (b, name) in enumerate([(big, "large region"), (small, "small region")]):
        x0 = 1.0 + k * 6.3
        rf = R.roi_fraction(b, img.shape)
        r = B.fmap(ax, x0, 1.3, R.FMAP_LEN, R.FMAP_SIDE, roi=rf, fs=FS,
                   label="convolution output\nfeature map")
        ax.text(x0 - 0.1, 2.25, f"projected {name}", ha="right", fontsize=FS)
        B.arrow(ax, r + 0.15, 1.3, r + 1.1, 1.3)
        ax.text(r + 0.62, 1.55, "pooling", ha="center", fontsize=FS)
        r2 = volume(ax, r + 1.2, 1.3, R.POOL_LEN * 1.6, 1.0, "#e1d5e7", lw=1.0, alpha=0.95)
        B.cap_grid(ax, r + 1.2 + R.POOL_LEN * 1.6, 1.3, 1.0, 7, col=C["grey"], lw=0.5)
        ax.text(r + 1.2 + R.POOL_LEN * 0.8, 0.68, "7 × 7 × C", ha="center", va="top", fontsize=FS)
    png = save_mpl(fig, ctx, "fast_rcnn_pipeline")
    return Figure("fast_rcnn_pipeline",
                  "**Fast R-CNN.** (a) The whole image is processed once by the convolution "
                  "layers, and each region proposal from selective search is projected onto "
                  "the output feature maps (**RoI projection**, red). The projected region is "
                  "converted to a fixed size by **RoI pooling** (circled), the result is passed "
                  "through fully connected layers to give an RoI feature vector, and two "
                  "branches predict the class probabilities (softmax) and the bounding box. "
                  "The SVMs and the separate regressor of R-CNN are replaced by these layers, "
                  "which are trained together with a multi-task loss, so the second stage is "
                  "trained end to end. (b) RoI pooling is a single level of spatial pyramid "
                  "pooling: whatever the size of the projected region, the pooling window is "
                  "adjusted so that the output is always 7 × 7 per channel. The regions are "
                  "those around the group of four sheep and around a single small sheep "
                  "(feature maps drawn in the "
                  "volume notation of Lesson 1.3, with the spatial extent on the end face).",
                  png=png)


def fig_cam_mechanism(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=448)
    icats = D.categories("resnet50")
    r = D.cam_parts("sheep", img)
    maps, w, g, ch = r["maps"], r["weights"], r["pooled"], r["channels"]
    n, cls = int(r["n"]), icats[int(r["cls"])]
    cam = np.maximum(r["cam"], 0)
    fig, ax = B.canvas(15.5, 5.6, (15, 5.5))
    ym = 4.4
    p = B.photo(ax, img, 0.1, ym - 0.75, 1.5, label="input image", fs=FS)
    B.arrow(ax, p[2] + 0.06, ym, p[2] + 0.4, ym)
    B.block(ax, p[2] + 0.45, ym - 0.45, 1.3, 0.9, "conv layers\n(ResNet-50)", fc=B.CONV_FC, fs=FS)
    x = p[2] + 1.85
    B.arrow(ax, x, ym, x + 0.35, ym)
    mx = x + 0.45
    vm = maps.max()
    for k in range(3):
        ax.imshow(maps[k], cmap="viridis", vmin=0, vmax=vm,
                  extent=(mx + k * 0.22, mx + k * 0.22 + 1.1, ym - 0.75 + k * 0.18,
                          ym + 0.35 + k * 0.18), zorder=2 + k)
    ax.text(mx + 0.75, ym - 0.95, f"last feature maps\n({n:,} maps of {maps.shape[2]} × "
            f"{maps.shape[1]})", ha="center", va="top", fontsize=FS)
    x = mx + 1.75
    B.arrow(ax, x, ym, x + 0.45, ym)
    ax.text(x + 0.22, ym + 0.2, "GAP", ha="center", va="bottom", fontsize=FS)
    vx = x + 0.55
    for k in range(5):
        col = SERIES[k] if k < 3 else C["light"]
        ax.add_patch(plt.Rectangle((vx, ym + 0.6 - k * 0.3), 0.3, 0.26, fc=col, ec=C["grey"],
                                   alpha=0.7))
        lab = f"{g[k]:.2f}" if k < 3 else ("⋮" if k == 3 else "")
        ax.text(vx + 0.38, ym + 0.73 - k * 0.3, lab, va="center", fontsize=FS - 0.5)
    for k in range(3):
        y0_ = ym + 0.73 - k * 0.3
        ax.plot([vx + 1.75, vx + 2.6], [y0_, ym], color=SERIES[k], lw=1.2)
        ax.text(vx + 0.85, y0_, f"× w{k + 1} = {w[k]:.2f}", fontsize=FS - 0.5,
                color=SERIES[k], va="center")
    ax.add_patch(plt.Circle((vx + 2.75, ym), 0.15, fc="white", ec=C["ink"]))
    ax.text(vx + 2.75, ym, "Σ", ha="center", va="center", fontsize=FS)
    ax.text(vx + 3.0, ym, f"class score → '{cls}'", va="center", fontsize=FS)
    ax.text(vx + 0.15, ym - 0.95, "pooled values", ha="center", va="top", fontsize=FS - 0.5)
    # the map: the same weights applied to the maps themselves
    yb = 1.3
    ax.text(0.1, 3.0, "class activation map: the same weights applied to the feature maps",
            fontsize=10, weight="bold")
    x = 0.4
    for k in range(3):
        ax.text(x, yb, f"{w[k]:.2f} ×", va="center", fontsize=10, color=SERIES[k])
        ax.imshow(maps[k], cmap="viridis", vmin=0, vmax=vm, extent=(x + 0.75, x + 2.15, yb - 0.5,
                                                                    yb + 0.5))
        ax.text(x + 1.45, yb - 0.6, f"map {ch[k]}", ha="center", va="top", fontsize=FS - 0.5)
        ax.text(x + 2.3, yb, "+", va="center", fontsize=13)
        x += 2.65
    ax.text(x, yb, f"…  (all {n:,} maps)", va="center", fontsize=10)
    x += 2.2
    ax.text(x, yb, "=", va="center", fontsize=13)
    camr = cv2.resize(cam / cam.max(), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    h = 1.6
    wd = h * img.shape[1] / img.shape[0]
    ax.imshow(img, extent=(x + 0.4, x + 0.4 + wd, yb - h / 2, yb + h / 2))
    ax.imshow(camr, cmap="jet", alpha=0.45, extent=(x + 0.4, x + 0.4 + wd, yb - h / 2, yb + h / 2))
    ax.text(x + 0.4 + wd / 2, yb - h / 2 - 0.08, f"class activation map for '{cls}'",
            ha="center", va="top", fontsize=FS)
    ax.set_xlim(0, x + 0.6 + wd)
    png = save_mpl(fig, ctx, "cam_mechanism")
    return Figure("cam_mechanism",
                  "How a **class activation map** is formed, with the real values of "
                  "ResNet-50 for the sheep photograph. Each of the "
                  f"{n:,} feature maps of the last convolution layer is reduced to a single "
                  "value by global average pooling (GAP), and the single fully connected "
                  "layer computes the score of each class as a weighted sum of these values. "
                  f"For the predicted class, '{cls}', the three maps that contribute most to "
                  "the score are shown with their pooled values and their weights. Applying "
                  "the same weights to the feature maps themselves, rather than to their "
                  "averages, and summing over all the maps gives the class activation map, "
                  "which shows where in the image the evidence for the class lies. It is "
                  "obtained without any training for localisation.",
                  png=png)


def _big_proposal(img):
    props = D.rpn_proposals("sheep", img)["boxes"][:30]
    return props[int(np.argmax((props[:, 2] - props[:, 0]) * (props[:, 3] - props[:, 1])))]


def _faster(ax, img, box, x, ym, fs=FS, brackets=False):
    """Faster R-CNN: backbone -> maps -> RPN -> RoI projection -> head."""
    p = R.backbone(ax, img, x, ym, label="input image", fs=fs)
    B.arrow(ax, p["fmap_right"] + 0.08, ym, p["fmap_right"] + 0.45, ym)
    f2 = p["fmap_right"] + 0.5
    roi = R.roi_fraction(box, img.shape)
    r2 = B.fmap(ax, f2, ym, R.FMAP_LEN, R.FMAP_SIDE, roi=roi, fs=fs)
    rx = p["fmap_x"] + 0.1
    top = ym + R.FMAP_SIDE / 2 + OBLIQUE[1] * R.FMAP_SIDE
    B.arrow(ax, rx + 0.3, top + 0.05, rx + 0.3, top + 0.45, width=0.12)
    B.block(ax, rx - 0.05, top + 0.5, 0.75, 0.55, "RPN", fc=B.CONV_FC, fs=fs)
    tx, ty = B.cap_point(f2 + R.FMAP_LEN, ym, R.FMAP_SIDE, (roi[0] + roi[2]) / 2, roi[1])
    ax.plot([rx + 0.7, tx], [top + 0.78, top + 0.78], color=C["red"], lw=1.6)
    B.line_arrow(ax, tx, top + 0.78, tx, ty + 0.02)
    ax.text((rx + 0.7 + tx) / 2 + 0.1, top + 0.85, "RoI projection", ha="center", va="bottom",
            fontsize=fs, color=C["red"])
    h = R.roi_head(ax, r2 + 0.1, ym, fs=fs)
    if brackets:
        B.bracket(ax, p["img"][0], f2 + 0.9, top + 1.2, "1st stage")
        B.bracket(ax, h["pool"] - 0.1, h["end"] - 0.6, ym - 1.45, "2nd stage", above=False)
    return p


def fig_faster_rcnn_pipeline(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=800)
    box = _big_proposal(img)
    fig, ax = B.canvas(14.5, 4.4, (14.5, 4.4))
    _faster(ax, img, box, 0.1, 1.8)
    png = save_mpl(fig, ctx, "faster_rcnn_pipeline")
    return Figure("faster_rcnn_pipeline",
                  "**Faster R-CNN** eliminates the external region proposal generator. "
                  "A small **region proposal network** (RPN) takes the output feature maps "
                  "of the backbone as its input and predicts the proposals, which are "
                  "projected onto the same feature maps (red) and processed by the Fast "
                  "R-CNN head as before. The RPN is convolutional and shares the convolution "
                  "layers of the feature-generating network, so the whole detector can be "
                  "trained end to end. The region drawn on the feature maps is the largest of "
                  "the 30 highest-scoring proposals of torchvision's Faster R-CNN for this "
                  "image.",
                  png=png)


def fig_stage_comparison(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=800)
    fig, ax = B.canvas(14.5, 9.0, (14.5, 8.4))
    _faster(ax, img, _big_proposal(img), 0.1, 5.5, brackets=True)
    ax.text(0.1, 8.75, "(a) Faster R-CNN: a two-stage detector", fontsize=10.5, weight="bold")
    ym = 1.4
    ax.text(0.1, 3.0, "(b) a single-stage detector", fontsize=10.5, weight="bold")
    p = R.backbone(ax, img, 0.1, ym, label="input image", fs=FS)
    B.arrow(ax, p["fmap_right"] + 0.08, ym, p["fmap_right"] + 0.45, ym)
    B.block(ax, p["fmap_right"] + 0.5, ym - 0.35, 1.3, 0.7, "predictor head", fc=B.CONV_FC, fs=FS)
    R.heads(ax, p["fmap_right"] + 1.85, ym, fs=FS, fc_layer=False)
    png = save_mpl(fig, ctx, "stage_comparison")
    return Figure("stage_comparison",
                  "Two-stage and single-stage detectors compared. (a) In Faster R-CNN, the "
                  "first stage predicts bounding boxes and confidences with the region "
                  "proposal network, and the boxes are projected onto the output feature "
                  "maps; in the second stage each region is converted to a vector by RoI "
                  "pooling, and refined boxes and classes are predicted. (b) A single-stage "
                  "detector predicts the bounding boxes, classes and confidences directly "
                  "from the output feature maps with a predictor head. The region proposal "
                  "network is functionally equivalent to a single-stage detector; the "
                  "additional stage gives more accurate box estimation and classification, "
                  "at the cost of a slower frame rate, as measured in the next figure.",
                  png=png)
