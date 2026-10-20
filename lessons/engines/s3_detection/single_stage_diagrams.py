"""Lesson 3.4 — the slides' diagrams, redrawn with real data.

Figures (built by single_stage.build, in lesson order)
    detection_system     slide 3a: classifier conv layers + prediction head
    head_sliding         slide 3b: a 3 x 3 head sliding over a real ResNet-50 map
    gt_assignment        slide 4c: anchors at the object's cell, IoU computed, > 0.5 kept
    box_dimensions       slide 6: anchor, prediction and ground truth with their dimensions
    scale_grids          slide 9: a large object on a coarse grid, a small one on a fine grid
    ssd_architecture     slide 10: SSD300 as volumes, shapes read from torchvision's model
    fpn_detector         slide 11: ResNet-50 + FPN + shared head (RetinaNet), as volumes

The slides' own photographs are used (the cat and the dog with a ball, all from Wikimedia Commons with their credits). Boxes come from
torchvision's Faster R-CNN (used as the ground truth, as noted in the
captions); anchor matching uses engines/_detmetrics.iou.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .._common import BuildContext, Figure, C, SERIES, save_mpl, load_image
from .. import _blocks as B
from .. import _detmetrics as M
from .. import _torchdet as D
from .._volumes import OBLIQUE, volume
from . import _rcnn_draw as R

FS = 9


def _det_boxes(key, side, names):
    """Faster R-CNN's best box for each requested class name, at `side` px."""
    full = load_image(key, max_side=800)
    det = D.detect(key, full)
    cats = D.categories("faster_rcnn")
    s = side / max(full.shape[:2])
    out = []
    for name in names:
        idx = [i for i, l in enumerate(det["labels"]) if cats[l] == name]
        i = max(idx, key=lambda j: det["scores"][j])
        out.append(det["boxes"][i] * s)
    return out


def anchors_at(cx, cy, sizes, ratios=(0.5, 1.0, 2.0)):
    """Anchors centred at (cx, cy): for each size (sqrt of the area) and ratio h/w."""
    out = []
    for s in sizes:
        for r in ratios:
            w, h = s / np.sqrt(r), s * np.sqrt(r)
            out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(out)


def _grid(ax, x0, y0, w, h, nx, ny, col=C["grey"], lw=0.6, alpha=1.0):
    for k in range(nx + 1):
        ax.plot([x0 + k * w / nx] * 2, [y0, y0 + h], color=col, lw=lw, alpha=alpha, zorder=3)
    for k in range(ny + 1):
        ax.plot([x0, x0 + w], [y0 + k * h / ny] * 2, color=col, lw=lw, alpha=alpha, zorder=3)


def _rect(ax, b, col, lw=1.6, ls="-", z=4, label=None, fs=8):
    ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col,
                               lw=lw, ls=ls, zorder=z))
    if label:
        ax.text(b[0] + 2, b[1] + 2, label, fontsize=fs, va="top", color=col, zorder=z + 1,
                bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8))


# ---------------------------------------------------------------------------

def fig_detection_system(ctx: BuildContext) -> Figure:
    img = load_image("ragdoll", max_side=400)
    cat = _det_boxes("ragdoll", 400, ["cat"])[0]
    fig, ax = B.canvas(13.0, 3.3, (13, 3.3))
    ym = 1.65
    p = B.photo(ax, img, 0.1, ym - 1.0, 2.0, boxes=[cat], box_col=C["green"], lw=1.6,
                label="input image", fs=FS)
    B.arrow(ax, p[2] + 0.08, ym, p[2] + 0.45, ym)
    B.block(ax, p[2] + 0.5, ym - 0.5, 1.7, 1.0, "image classifier CNN\nconvolution layers",
            fc="#e1d5e7", fs=FS)
    x = p[2] + 2.28
    B.arrow(ax, x, ym, x + 0.37, ym)
    fx = x + 0.45
    right = B.fmap(ax, fx, ym, 0.55, 1.5, label="output feature maps", fs=FS)
    B.cap_region(ax, fx + 0.55, ym, 1.5, (0.1, 0.35, 0.35, 0.55), C["red"], fill=C["red"],
                 ls="-")
    B.arrow(ax, right + 0.1, ym, right + 0.47, ym)
    B.block(ax, right + 0.55, ym - 0.45, 1.4, 0.9, "prediction head\n(3 × 3 convolutions)",
            fc=B.CONV_FC, fs=FS)
    ax.text(right + 1.25, ym - 0.6, "applied convolutionally\nat every position", ha="center",
            va="top", fontsize=FS - 0.5)
    hx = right + 2.0
    for dy, t in [(0.45, "class label"), (0.0, "bounding box"), (-0.45, "confidence")]:
        B.arrow(ax, hx, ym + dy * 0.3, hx + 0.45, ym + dy, width=0.1)
        ax.text(hx + 0.55, ym + dy, t, va="center", fontsize=FS)
    ax.set_xlim(0, hx + 2.0)
    png = save_mpl(fig, ctx, "detection_system")
    return Figure("detection_system",
                  "A basic single-stage object detection system. The input image is passed "
                  "once through the convolution layers of a pre-trained image classifier. A "
                  "**prediction head**, a small convolutional network, is added to the output "
                  "feature maps and applied convolutionally at every position (one position "
                  "is marked in red on the feature maps). At each position it predicts a "
                  "class label, a bounding box and a confidence. The photograph is the one "
                  "used in the lecture slides; the box around the cat is Faster R-CNN's "
                  "detection.",
                  png=png)


def fig_head_sliding(ctx: BuildContext) -> Figure:
    img = load_image("ragdoll", max_side=256)
    r = D.cam_parts("ragdoll_256", img, side=256)
    act = np.maximum(r["cam"], 0)
    act = act / act.max()
    ny, nx = act.shape
    cy, cx = np.unravel_index(np.argmax(act), act.shape)
    icats = D.categories("resnet50")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for k, ax in enumerate(axes):
        ax.imshow(act, cmap="turbo", vmin=0, vmax=1.2, extent=(0, nx, ny, 0), alpha=0.85)
        _grid(ax, 0, 0, nx, ny, nx, ny, col="k", lw=0.8)
        ax.set_xlim(-0.2, nx + 0.2); ax.set_ylim(ny + 0.2, -0.2); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax = axes[0]
    ax.add_patch(plt.Rectangle((0, 0), 3, 3, fill=False, ec=C["green"], lw=3, zorder=5))
    for (x0, y0, x1, y1) in [(3.1, 1.5, 4.3, 1.5), (1.5, 3.1, 1.5, 4.3)]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=3, color=C["green"]), zorder=6)
    ax.set_title(f"(a) the 3 × 3 prediction head is moved over the {nx} × {ny} output "
                 "feature map", fontsize=10)
    ax = axes[1]
    ax.add_patch(plt.Rectangle((cx - 1, cy - 1), 3, 3, fill=False, ec=C["green"], lw=3, zorder=5))
    ax.add_patch(plt.Rectangle((cx, cy), 1, 1, fill=False, ec="white", lw=2.2, zorder=6))
    ax.text(cx + 2.2, cy + 0.5, "object present:\npredict class,\nbox, confidence",
            fontsize=FS, va="center", color="k",
            bbox=dict(fc="white", ec=C["green"], pad=3))
    ax.set_title("(b) where the activation is high, an object is detected", fontsize=10)
    png = save_mpl(fig, ctx, "head_sliding")
    return Figure("head_sliding",
                  "The prediction head applied convolutionally. The colours show a real "
                  f"activation map of ResNet-50 for the photograph of Figure 2 (its class "
                  f"activation map for '{icats[int(r['cls'])]}', on the {nx} × {ny} grid of "
                  "the last convolution layer). (a) The 3 × 3 head is moved over every "
                  "position of the output feature map. (b) At each position it predicts a "
                  "confidence that an object is present in that region of the feature map, "
                  "together with the class and the bounding box. Each cell has a large "
                  "receptive field, so the features at a cell are computed from a large region "
                  "of the image and a small kernel is sufficient. The coarse grid limits how "
                  "precisely objects can be located, which is addressed by the multi-scale "
                  "feature maps described later in this lesson.",
                  png=png)


def fig_gt_assignment(ctx: BuildContext) -> Figure:
    side = 400
    img = load_image("ragdoll", max_side=side)
    gt = _det_boxes("ragdoll", side, ["cat"])[0]
    s = 50                                        # grid cell size in pixels: an 8 x 6 grid
    h, w = img.shape[:2]
    nx, ny = w // s, h // s
    gcx, gcy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    cx, cy = (int(gcx // s) + 0.5) * s, (int(gcy // s) + 0.5) * s
    an = anchors_at(cx, cy, sizes=[1.9 * s], ratios=(0.5, 1.0, 2.0))
    v = M.iou(an, gt[None])[:, 0]
    assert (v > 0.5).sum() >= 1 and (v <= 0.5).sum() >= 1, v
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.imshow(img, alpha=0.35, extent=(0, w, h, 0))
    _grid(ax, 0, 0, nx * s, ny * s, nx, ny, col="k", lw=0.8)
    _rect(ax, gt, C["green"], lw=2.6, label="ground truth")
    names = ["wide (h/w = 0.5)", "square", "tall (h/w = 2)"]
    for b, val, nm in zip(an, v, names):
        pos = val > 0.5
        _rect(ax, b, C["red"], lw=2.4 if pos else 1.3, ls="-" if pos else (0, (4, 3)))
        ax.text(b[2] + 3, b[1] + 10 if nm != "wide (h/w = 0.5)" else b[3] - 4,
                f"{nm}: IoU {val:.2f}" + ("  → positive" if pos else ""), fontsize=FS,
                color=C["red"], va="center",
                bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
    ax.add_patch(plt.Circle((cx, cy), 5, fc="none", ec=C["red"], lw=1.5, zorder=6))
    ax.set_xlim(0, nx * s); ax.set_ylim(ny * s, 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("anchors at the cell containing the centre of the object; the ground truth "
                 "is assigned to\nevery anchor whose IoU with it exceeds 0.5", fontsize=10)
    png = save_mpl(fig, ctx, "gt_assignment")
    pos = [n.split(" (")[0] for n, val in zip(names, v) if val > 0.5]
    return Figure("gt_assignment",
                  "**Ground-truth assignment** with anchor boxes, worked on the slide's "
                  "photograph and an 8 × 6 grid. Three anchors of the same area and different "
                  "aspect ratios are centred on the cell that contains the centre of the "
                  "object, and the IoU of each with the ground-truth box is calculated. The "
                  f"ground truth is assigned to every anchor whose IoU exceeds 0.5, here the "
                  + " and the ".join(pos) + f" anchors (IoU up to {v.max():.2f}), which are "
                  "trained to predict the offset and scale of the box from the anchor; the "
                  "wide anchor is a negative example for the confidence. Assigning an object "
                  "to several anchors increases the number of positive examples. The calculation of the "
                  "IoU of every anchor with every object adds computation during training.",
                  png=png)


def fig_box_dimensions(ctx: BuildContext) -> Figure:
    from .single_stage import _retina, SHEEP_GT
    img = load_image("sheep", max_side=800)
    r = _retina()
    g = SHEEP_GT[0]
    v = D.iou(r["anchors"], g[None])[:, 0]
    cand = np.where(v >= 0.5)[0]
    a = r["anchors"][cand[int(np.argmin(v[cand]))]]
    props = D.rpn_proposals("sheep", img)["boxes"]
    pv = D.iou(props, g[None])[:, 0]
    p = props[int(np.argmin(np.abs(pv - 0.8)))]       # a real RPN prediction, IoU ~ 0.8
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    ax.imshow(img, alpha=0.25)
    boxes = [(g, C["green"], "-", "gt"), (p, C["red"], "-", ""), (a, C["blue"], (0, (5, 3)), "a")]
    for b, col, ls, sub in boxes:
        _rect(ax, b, col, lw=2.2, ls=ls)
    pad = 28
    lo = np.min([g, p, a], 0); hi = np.max([g, p, a], 0)
    ax.set_xlim(lo[0] - pad * 2, hi[2] + pad * 2); ax.set_ylim(hi[3] + pad * 2, lo[1] - pad * 2)

    def dim_h(b, y, col, text):
        ax.annotate("", xy=(b[2], y), xytext=(b[0], y),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.4))
        ax.text((b[0] + b[2]) / 2, y - 3, text, ha="center", va="bottom", color=col, fontsize=12)

    def dim_v(b, x, col, text, side="left"):
        ax.annotate("", xy=(x, b[3]), xytext=(x, b[1]),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.4))
        ax.text(x + (-4 if side == "left" else 4), (b[1] + b[3]) / 2, text, ha="right" if
                side == "left" else "left", va="center", color=col, fontsize=12)
    dim_h(g, lo[1] - pad * 1.2, C["green"], "$w^{gt}$")
    dim_v(g, lo[0] - pad * 1.2, C["green"], "$h^{gt}$")
    dim_h(p, hi[3] + pad * 1.2, C["red"], "")
    ax.text((p[0] + p[2]) / 2, hi[3] + pad * 1.2 + 4, "$w$", ha="center", va="top", color=C["red"],
            fontsize=12)
    dim_v(p, hi[2] + pad * 1.2, C["red"], "$h$", side="right")
    dim_h(a, (a[1] + a[3]) / 2 + 0.38 * (a[3] - a[1]), C["blue"], "$w_a$")
    dim_v(a, a[0] + 0.12 * (a[2] - a[0]), C["blue"], "", side="right")
    ax.text(a[0] + 0.12 * (a[2] - a[0]) + 4, a[1] + 0.25 * (a[3] - a[1]), "$h_a$",
            color=C["blue"], fontsize=12)
    for b, col, lab, (ox, oy, ha) in [(g, C["green"], "$x^{gt}, y^{gt}$", (-6, -2, "right")),
                                      (p, C["red"], "$x, y$", (6, 8, "left")),
                                      (a, C["blue"], "$x_a, y_a$", (6, -6, "left"))]:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        ax.plot(cx, cy, "o", color=col, ms=7, mec="white", zorder=7)
        ax.text(cx + ox, cy + oy, lab, color=col, fontsize=12, va="center", ha=ha, zorder=7,
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.75))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("predicted box (red), anchor box (blue, dashed) and ground-truth box (green)",
                 fontsize=10)
    png = save_mpl(fig, ctx, "box_dimensions")
    return Figure("box_dimensions",
                  "The quantities used in **bounding-box prediction**, drawn to scale for one of "
                  "the sheep: the anchor box (blue, centre $x_a, y_a$, width $w_a$, height "
                  "$h_a$), centred on the current cell; the predicted box (red, $x, y, w, h$); "
                  "and the ground-truth box (green, $x^{gt}, y^{gt}, w^{gt}, h^{gt}$). The "
                  "prediction head predicts the displacement of the box centre from the anchor "
                  "centre and the scaling of its width and height, and is trained to match the "
                  "displacement and scaling of the ground truth from the same anchor. The "
                  "anchor is a real RetinaNet anchor matched to this sheep, the predicted box "
                  "is a real prediction of Faster R-CNN's region proposal network (IoU "
                  f"{float(D.iou(p[None], g[None])[0, 0]):.2f} with the ground truth), and "
                  "the ground truth was drawn by hand.",
                  png=png)


RETINA_SIZES = {3: 32, 4: 64, 5: 128, 6: 256, 7: 512}


def _level_anchors(cx, cy, level):
    base = RETINA_SIZES[level]
    return anchors_at(cx, cy, [base * 2 ** (k / 3) for k in range(3)])


def fig_scale_grids(ctx: BuildContext) -> Figure:
    side = 512
    img = load_image("dog_ball", max_side=side)
    dog, ball = _det_boxes("dog_ball", side, ["dog", "sports ball"])
    h, w = img.shape[:2]
    cases = [(6, dog, "large object (the dog) on a coarse feature map"),
             (4, ball, "small object (the ball) on a fine feature map")]
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    for ax, (lv, obj, title) in zip(axes, cases):
        s = 2 ** lv
        ax.imshow(img)
        _grid(ax, 0, 0, w, h, int(np.ceil(w / s)), int(np.ceil(h / s)), col="white",
              lw=0.5 if lv == 4 else 1.0, alpha=0.8)
        gcx, gcy = (obj[0] + obj[2]) / 2, (obj[1] + obj[3]) / 2
        cx, cy = (int(gcx // s) + 0.5) * s, (int(gcy // s) + 0.5) * s
        allv = M.iou(_level_anchors(cx, cy, lv), obj[None])[:, 0]
        k = int(np.argmax(allv)) // 3                  # the best of the three sizes
        size = RETINA_SIZES[lv] * 2 ** (k / 3)
        an = anchors_at(cx, cy, [size])
        v = M.iou(an, obj[None])[:, 0]
        other = ball if obj is dog else dog
        vo = M.iou(_level_anchors(*(((np.array([(other[0] + other[2]) / 2,
                                                  (other[1] + other[3]) / 2]) // s) + 0.5) * s),
                                  lv), other[None])[:, 0]
        res[lv] = (float(allv.max()), int((allv > 0.5).sum()), float(vo.max()))
        _rect(ax, obj, C["green"], lw=2.6)
        for b, val in zip(an, v):
            _rect(ax, b, C["red"], lw=2.2 if val > 0.5 else 1.0, ls="-" if val > 0.5 else (0, (4, 3)))
        ax.plot(cx, cy, "o", color=C["red"], ms=6, mec="white")
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"({'a' if lv == 6 else 'b'}) {title}\nstride {s} (P{lv}), anchors "
                     f"{RETINA_SIZES[lv]}–{RETINA_SIZES[lv] * 2 ** (2 / 3):.0f} px: best IoU "
                     f"{res[lv][0]:.2f}", fontsize=10)
    assert res[6][0] > 0.5 and res[4][0] > 0.5 and res[6][2] < 0.5 and res[4][2] < 0.5, res
    png = save_mpl(fig, ctx, "scale_grids")
    return Figure("scale_grids",
                  "The **scale problem**, on the photograph used in the slides, with "
                  "RetinaNet's anchor sizes (for clarity, the three aspect ratios are drawn "
                  "for the best-matching anchor size only). (a) A coarse feature map (stride 64) has a large "
                  "receptive field and large anchors, which match the dog (best IoU "
                  f"{res[6][0]:.2f}); the best of its anchors for the ball reaches only "
                  f"{res[6][2]:.2f}. (b) A fine feature map (stride 16) has small anchors, "
                  f"which match the ball (best IoU {res[4][0]:.2f}), while its best anchor "
                  f"for the dog reaches only {res[4][2]:.2f}. A detector that predicts from a "
                  "single coarse feature map therefore finds only large objects. Each feature "
                  "map of a multi-scale detector is given a nominal scale and anchors sized to "
                  "match, so that small objects are localised in the fine maps and large "
                  "objects in the coarse maps. (Dog and ball boxes: Faster R-CNN's "
                  "detections.)",
                  png=png)


def fig_ssd_architecture(ctx: BuildContext) -> Figure:
    r = D.ssd_default_boxes()
    shapes, chans, k = r["shapes"], r["channels"], r["per_cell"]
    ncls = len(D.categories("ssd300"))
    total = int(sum(h * w * kk for (h, w), kk in zip(shapes, k)))
    fig, ax = B.canvas(15.5, 5.6, (15.5, 5.6))
    ym = 4.0
    side_of = lambda n: 0.5 + 2.2 * np.sqrt(n / 300)
    from .._volumes import chan_len
    photo = __import__("cv2").resize(load_image("dog_ball", max_side=600), (300, 300))
    pp = B.photo(ax, photo, 0.2, ym - 0.9, 1.8, label="input 300 × 300 × 3", fs=FS)
    right = pp[2]
    B.arrow(ax, right + 0.08, ym, right + 0.45, ym)
    B.block(ax, right + 0.5, ym - 0.55, 1.5, 1.1, "VGG-16\nbase network\n(conv layers)",
            fc=B.CONV_FC, fs=FS)
    x = right + 2.1
    names = ["VGG-16 conv4_3", "VGG-16 fc7\n(as conv)", "extra conv", "extra conv",
             "extra conv", "extra conv"]
    tops = []
    for i, ((h, w), c) in enumerate(zip(shapes, chans)):
        B.arrow(ax, x, ym, x + 0.32, ym)
        x += 0.4
        sd = side_of(h)
        ln = 0.25 + 0.35 * chan_len(c) / chan_len(1024)
        rr = volume(ax, x, ym, ln, sd, "#dae8fc", lw=1.0)
        ax.text(x + ln / 2, ym + sd / 2 + OBLIQUE[1] * sd + 0.08, f"{h} × {w} × {c}",
                ha="center", va="bottom", fontsize=FS - 0.5)
        ax.text(x + ln / 2, ym - sd / 2 - 0.08, names[i], ha="center", va="top", fontsize=FS - 1)
        tops.append((x + ln / 2, ym - sd / 2 - 0.55))
        x = rr + 0.15
    yb = 1.75
    ybus = 0.95
    bx = tops[-1][0] + 1.2
    for (cx, cy), kk, (h, w) in zip(tops, k, shapes):
        B.line_arrow(ax, cx, cy, cx, yb + 0.57, col=C["grey"], lw=1.2)
        B.block(ax, cx - 0.55, yb - 0.1, 1.1, 0.65, f"3 × 3 conv\n{kk} × ({ncls} + 4)",
                fc="#fce5cd", fs=FS - 1)
        ax.plot([cx, cx], [yb - 0.12, ybus], color=C["grey"], lw=1.0)
    ax.plot([tops[0][0], bx], [ybus, ybus], color=C["grey"], lw=1.0)
    B.line_arrow(ax, bx - 0.3, ybus, bx, ybus, col=C["grey"], lw=1.0)
    B.block(ax, bx, ybus - 0.52, 1.6, 1.05, f"detections:\n{total:,} boxes\n→ NMS", fc="#f4cccc",
            fs=FS)
    ax.text(0.2, 0.05, f"boxes per image = Σ (map size × default boxes per cell) = "
            + " + ".join(f"{h}²×{kk}" for (h, w), kk in zip(shapes, k)) + f" = {total:,}",
            fontsize=FS)
    ax.set_xlim(0, bx + 1.8)
    png = save_mpl(fig, ctx, "ssd_architecture")
    return Figure("ssd_architecture",
                  "The **Single Shot MultiBox Detector** (SSD300), drawn in the volume "
                  "notation with every shape read from torchvision's model. VGG-16 is the "
                  "base network, with its fully connected layer fc7 converted into a "
                  "convolution, and further convolution layers progressively downsample the "
                  "output to produce feature maps at six scales. At every position of every "
                  "map, a 3 × 3 convolution predicts, for each default box, the "
                  f"{ncls} class scores (80 COCO classes, background and unused label "
                  "indices) and four box values. In total "
                  f"{total:,} boxes are predicted for each image, and non-maximum "
                  "suppression is applied to the result. In the published SSD300 the finest "
                  "prediction map is taken from VGG-16's conv4_3, with L2 normalisation, and "
                  "the remaining maps from conv5_3 onwards.",
                  png=png)


def fig_fpn_detector(ctx: BuildContext) -> Figure:
    H, W = 800, 1216                              # RetinaNet's input for the sheep photo
    from .._volumes import chan_len
    side_of = lambda stride: 0.35 + 0.85 * np.sqrt(8 / stride)
    fig, ax = B.canvas(15.0, 10.4, (14.5, 10.0))
    bottom = [("C2", "res2", 256, 4), ("C3", "res3", 512, 8), ("C4", "res4", 1024, 16),
              ("C5", "res5", 2048, 32)]
    ys = {"C2": 1.1, "C3": 3.0, "C4": 4.8, "C5": 6.5}
    xb = 2.0
    cap = {}
    for name, stage, c, s in bottom:
        y, sd = ys[name], side_of(s)
        ln = 0.25 + 0.55 * chan_len(c) / chan_len(2048)
        rr = volume(ax, xb, y, ln, sd, "#fff2cc", lw=1.0)
        ax.text(xb - 0.12, y, f"{name} ({stage})\n{H // s} × {W // s} × {c}", ha="right",
                va="center", fontsize=FS - 0.5)
        cap[name] = (rr, y, sd)
    names = [n for n, *_ in bottom]
    for a_, b_ in zip(names, names[1:]):
        B.arrow(ax, xb + 0.15, ys[a_] + cap[a_][2] / 2 + 0.25, xb + 0.15,
                ys[b_] - cap[b_][2] / 2 - 0.05, width=0.1)
    ax.text(xb + 0.5, 8.2, "bottom-up\npathway:\nResNet-50", ha="center", fontsize=FS)
    xp = 6.4
    pys = {"P3": 3.0, "P4": 4.8, "P5": 6.5, "P6": 7.95, "P7": 9.2}
    tops = {}
    for name, s in [("P3", 8), ("P4", 16), ("P5", 32), ("P6", 64), ("P7", 128)]:
        y, sd = pys[name], side_of(s)
        rr = volume(ax, xp, y, 0.3, sd, "#d5e8d4", lw=1.0)
        ax.text(rr + 0.1, y, f"{name}\n{-(-H // s)} × {-(-W // s)} × 256", va="center",
                fontsize=FS - 0.5)
        tops[name] = (rr, y, sd)
    for cname, pname in [("C3", "P3"), ("C4", "P4"), ("C5", "P5")]:
        rr, y, sd = cap[cname]
        B.line_arrow(ax, rr + 0.05, y, xp - 0.05, pys[pname], col=C["orange"], lw=1.5)
        ax.text((rr + xp) / 2, y + 0.1, "1 × 1 conv (lateral)", ha="center", va="bottom",
                fontsize=FS - 1, color=C["orange"])
    for a_, b_ in [("P5", "P4"), ("P4", "P3")]:
        B.line_arrow(ax, xp - 0.25, pys[a_] - tops[a_][2] / 2, xp - 0.25,
                     pys[b_] + tops[b_][2] / 2 + 0.1, col=C["red"], lw=1.5)
        ax.text(xp - 0.35, (pys[a_] + pys[b_]) / 2 + 0.35, "×2 upsample, add", ha="right",
                va="center", fontsize=FS - 1, color=C["red"])
    for a_, b_ in [("P5", "P6"), ("P6", "P7")]:
        B.line_arrow(ax, xp - 0.25, pys[a_] + tops[a_][2] / 2 + 0.3, xp - 0.25,
                     pys[b_] - tops[b_][2] / 2, col=C["grey"], lw=1.3)
        ax.text(xp - 0.35, (pys[a_] + pys[b_]) / 2 + 0.25, "3 × 3 conv, stride 2", ha="right",
                va="center", fontsize=FS - 1, color=C["grey"])
    ax.text(xp + 0.3, 10.15, "top-down pyramid", ha="center", fontsize=FS)
    hx, hy = 10.4, 5.6
    for name, (rr, y, sd) in tops.items():
        ax.plot([rr + 1.7, hx - 0.2], [y, hy], color=C["light"], lw=0.9, zorder=0)
    for dy, title, out in [(1.0, "class subnet", "W × H × KA"), (-1.0, "box subnet", "W × H × 4A")]:
        y = hy + dy
        for j in range(4):
            volume(ax, hx + j * 0.42, y, 0.22, 0.75, "#e1d5e7", lw=0.8)
        ax.text(hx + 0.85, y + 0.65, f"{title}: 4 × (3 × 3 conv, 256)", ha="center",
                fontsize=FS - 0.5)
        B.arrow(ax, hx + 1.85, y, hx + 2.3, y)
        volume(ax, hx + 2.35, y, 0.35, 0.75, "#f4cccc", lw=0.8)
        ax.text(hx + 2.95, y, out, va="center", fontsize=FS)
    ax.text(hx + 1.6, hy - 1.75, "shared prediction head: the same weights\nat every level; "
            "A = 9 anchors, K = 91 classes", ha="center", va="top", fontsize=FS - 0.5)
    png = save_mpl(fig, ctx, "fpn_detector")
    return Figure("fpn_detector",
                  "A detector with a **feature pyramid network**, as in RetinaNet, with the "
                  f"shapes for a {W} × {H} input. The bottom-up path is the ResNet-50 "
                  "backbone, whose fine feature maps (C3) have poor semantic information. A "
                  "parallel top-down pyramid is built from the last, semantically strong map: "
                  "each level is upsampled by two and added to the corresponding backbone map, "
                  "reduced to 256 channels by a 1 × 1 **lateral** convolution, which combines "
                  "semantic information with fine spatial detail. Two further levels, P6 and "
                  "P7, are created by stride-2 convolutions. All levels have the same depth, "
                  "256, so that one prediction head, a class subnet and a box subnet each of "
                  "four 3 × 3 convolutions, can be shared between them; at each position it "
                  "outputs K class scores and four box values for each of the A anchors.",
                  png=png)
