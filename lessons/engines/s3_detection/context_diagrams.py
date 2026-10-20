"""Lesson 3.6 — the slides' diagrams, redrawn with real data.

Figures (built by context_instance.build, in lesson order)
    psp_architecture      slide 4: PSPNet: backbone -> pooled 1/2/3/6 -> 1 x 1 conv ->
                          upsample -> concatenate -> final conv
    crf_graph             slide 5a: a fully connected CRF over real pixels, edge strength
                          = the pairwise weight used in dense_crf
    deeplab_pipeline      slide 6: input -> atrous DCNN -> coarse score map -> bilinear
                          interpolation -> CRF -> output, with DeepLabV3's real maps
    instance_approaches   slide 7: segmentation proposals classified; a semantic map cut
                          into instances by detection boxes
    mask_rcnn_pipeline    slide 8a: Faster R-CNN with RoIAlign and a mask branch
    roi_grids             slide 8b,c: RoIPool vs RoIAlign on the sheep photo's feature grid
"""
from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt

from .._common import BuildContext, Figure, C, SERIES, save_mpl, load_image
from .. import _blocks as B
from .. import _torchdet as D
from .._volumes import OBLIQUE, volume
from . import _rcnn_draw as R

FS = 9


def fig_psp_architecture(ctx: BuildContext) -> Figure:
    from .context_instance import _sheep400
    img = _sheep400()
    fig, ax = B.canvas(15.5, 6.0, (15.5, 5.8))
    ym = 3.0
    p = B.photo(ax, img, 0.1, ym - 0.75, 1.5, label="input image", fs=FS)
    B.arrow(ax, p[2] + 0.06, ym, p[2] + 0.4, ym)
    B.block(ax, p[2] + 0.45, ym - 0.45, 1.5, 0.9, "backbone CNN\n(ResNet, dilated)",
            fc=B.CONV_FC, fs=FS)
    x = p[2] + 2.05
    B.arrow(ax, x, ym, x + 0.35, ym)
    fx = x + 0.42
    fr = volume(ax, fx, ym, 0.9, 2.0, "#dae8fc", lw=1.0)
    ax.text(fx + 0.6, ym - 1.15, "feature map\n1/8 of the input,\n2,048 channels", ha="center",
            va="top", fontsize=FS - 0.5)
    cols = [C["red"], C["orange"], C["sky"], C["green"]]
    ys = [ym + 2.0, ym + 0.75, ym - 0.5, ym - 1.75]
    px = fr + 1.0
    for n, y, col in zip((1, 2, 3, 6), ys, cols):
        B.arrow(ax, fr + 0.08, ym + (y - ym) * 0.25, px - 0.08, y, width=0.1)
        sd = 0.55
        r1 = volume(ax, px, y, 0.6, sd, col, lw=0.9, alpha=0.75)
        B.cap_grid(ax, px + 0.6, y, sd, n, col="k", lw=0.5)
        ax.text(px + 0.3, y - sd / 2 - 0.06, f"{n} × {n}", ha="center", va="top", fontsize=FS - 1)
        B.arrow(ax, r1 + 0.06, y, r1 + 0.42, y, width=0.1)
        r2 = volume(ax, r1 + 0.48, y, 0.18, sd, col, lw=0.9, alpha=0.75)
        B.arrow(ax, r2 + 0.06, y, r2 + 0.42, y, width=0.1)
        r3 = volume(ax, r2 + 0.48, y, 0.18, 1.2, col, lw=0.9, alpha=0.75)
        B.arrow(ax, r3 + 0.06, y, r3 + 0.55, ym + (y - ym) * 0.25, width=0.1)
        cx_end = r3 + 0.6
    for x_, t_ in [(px + 0.3, "pooling"), (px + 1.05, "1 × 1 conv\n(2,048 → 512)"),
                   (px + 1.75, "upsample")]:
        ax.text(x_, 5.75, t_, ha="center", va="top", fontsize=FS - 0.5)
    cx = cx_end
    r = volume(ax, cx, ym, 0.9, 2.0, "#dae8fc", lw=1.0)
    for k, col in enumerate(cols):
        volume(ax, cx + 0.9 + k * 0.12, ym, 0.12, 2.0, col, lw=0.6, alpha=0.75)
    ax.text(cx + 0.75, ym - 1.15, "concatenate:\n2,048 + 4 × 512\n= 4,096 channels",
            ha="center", va="top", fontsize=FS - 0.5)
    x = cx + 1.75
    B.arrow(ax, x, ym, x + 0.4, ym)
    B.block(ax, x + 0.45, ym - 0.35, 0.95, 0.7, "final\nconv", fc=B.CONV_FC, fs=FS)
    B.arrow(ax, x + 1.45, ym, x + 1.85, ym)
    ax.text(x + 1.95, ym, "pixel-level\nprediction", va="center", fontsize=FS)
    ax.set_xlim(0, x + 3.2)
    png = save_mpl(fig, ctx, "psp_architecture")
    return Figure("psp_architecture",
                  "The **pyramid scene parsing network** (PSPNet). A ResNet backbone, with "
                  "dilated convolutions so that the feature map is 1/8 of the size of the "
                  "input, produces the output feature map. It is pooled to 1 × 1, 2 × 2, "
                  "3 × 3 and 6 × 6 bins, as in spatial pyramid pooling, so that information "
                  "about features at different spatial positions and scales is retained. The "
                  "depth of each pooled map is reduced by a 1 × 1 convolution, each is "
                  "upsampled to the size of the feature map, and they are concatenated with the "
                  "original features (channel counts as in PSPNet). A final convolution layer "
                  "produces the pixel-level prediction. The next figure applies this pooling to a real "
                  "feature map.",
                  png=png)


def fig_crf_graph(ctx: BuildContext) -> Figure:
    from .context_instance import crf_inputs, CRF_W
    small, P = crf_inputs()
    H_, W_ = small.shape[:2]
    # 9 pixels: 4 on the largest sheep, 5 on the grass around it
    from scipy import ndimage
    cats = D.categories("deeplab"); sheep = cats.index("sheep")
    lab = P.argmax(0) == sheep
    comp, n = ndimage.label(lab)
    big = 1 + int(np.argmax(ndimage.sum(lab, comp, range(1, n + 1))))
    ys, xs = np.where(comp == big)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    inside = [(cy - 1, cx - 2), (cy - 1, cx + 2), (cy + 2, cx - 2), (cy + 2, cx + 2)]
    inside = [(y, x) for y, x in inside if comp[y, x] == big] or [(cy, cx)]
    outside = [(y0 - 4, x0 - 2), (y0 - 4, x1 + 2), (y1 + 4, cx), (cy, x0 - 6), (cy, x1 + 6)]
    pts = [(min(max(y, 0), H_ - 1), min(max(x, 0), W_ - 1)) for y, x in inside + outside]
    pts = pts[:9]
    col = np.array([small[y, x].astype(float) for y, x in pts])
    pos = np.array([(x, y) for y, x in pts], float)
    K = len(pts)
    w = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                dp = ((pos[i] - pos[j]) ** 2).sum(); dc = ((col[i] - col[j]) ** 2).sum()
                w[i, j] = (CRF_W["w_app"] * np.exp(-dp / (2 * CRF_W["th_a"] ** 2) - dc / (2 * CRF_W["th_b"] ** 2))
                           + CRF_W["w_sm"] * np.exp(-dp / (2 * CRF_W["th_g"] ** 2)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1, 1.2]))
    ax = axes[0]
    ax.imshow(small)
    for k, (y, x) in enumerate(pts):
        ax.plot(x, y, "o", ms=9, mfc="none", mec="white", mew=2)
        ax.text(x + 1.2, y - 1.2, str(k + 1), color="white", fontsize=9, weight="bold")
    ax.set_ylim(min(max(p[0] for p in pts) + 6, H_), max(min(p[0] for p in pts) - 6, 0))
    ax.set_xlim(max(min(p[1] for p in pts) - 8, 0), min(max(p[1] for p in pts) + 8, W_))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"(a) {K} pixels of the image used in the next figure: {len(inside)} on a sheep, "
                 f"{K - len(inside)} around it", fontsize=10)
    ax = axes[1]
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False) + np.pi / 2
    node = np.stack([np.cos(ang), np.sin(ang)], 1)
    wl = np.log10(w + 1e-12)
    lo, hi = wl[w > 0].min(), wl.max()
    for i in range(K):
        for j in range(i + 1, K):
            s = (wl[i, j] - lo) / (hi - lo + 1e-9)
            ax.plot(*zip(node[i], node[j]), color=C["ink"], lw=0.3 + 3.2 * s, alpha=0.15 + 0.8 * s)
    for k in range(K):
        ax.add_patch(plt.Circle(node[k], 0.11, fc=col[k] / 255, ec="k", lw=1.2, zorder=3))
        ax.text(*(node[k] * 1.24), str(k + 1), ha="center", va="center", fontsize=10)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(b) fully connected CRF: every pair of pixels is joined; line weight = "
                 "pairwise\nweight (log scale), large for pixels that are close and similar "
                 "in colour", fontsize=10)
    png = save_mpl(fig, ctx, "crf_graph")
    ni = len(inside)
    wool = np.mean([w[i, j] for i in range(ni) for j in range(ni) if i != j])
    cross = np.mean([w[i, j] for i in range(ni) for j in range(ni, K)])
    assert wool > 10 * cross, (wool, cross)
    return Figure("crf_graph",
                  "A **fully connected conditional random field**. Each pixel is a node of a "
                  "graph, and every pair of nodes is joined by an edge, so the label of each "
                  "pixel depends on all the others. (a) Nine pixels of the reduced sheep image "
                  f"of the next figure, {len(inside)} on a sheep and the rest around it, mostly on the "
                  "grass. (b) The graph between "
                  "them, with the colour of each node taken from its pixel and the thickness "
                  "of each edge set by the CRF's actual pairwise weight: a Gaussian of the "
                  "distance between the pixels and of the difference in their colours. Pixels "
                  "that are close together and similar in colour, such as the pixels of white "
                  "wool, are strongly connected and are encouraged to take the same label "
                  f"(here the mean weight between the four pixels on the sheep is "
                  f"{wool / cross:.0f} times that between them and the others); pixels of different colours are only weakly "
                  "connected. With every "
                  "pixel of an image as a node the model is intractable to learn exactly, and "
                  "efficient approximate inference (the dense CRF) is used.",
                  png=png)


def fig_deeplab_pipeline(ctx: BuildContext) -> Figure:
    from .context_instance import crf_result, _sheep400, CRF_SIZE
    img = _sheep400()
    cats = D.categories("deeplab"); sheep = cats.index("sheep")
    L8 = D.semantic_scores("sheep_400", img)["logits8"]
    P8 = np.exp(L8 - L8.max(0)); P8 /= P8.sum(0)
    small, Q, ref = crf_result()
    W_, H_ = CRF_SIZE
    up = Q[0][sheep]
    after = Q[-1][sheep]
    out = np.zeros(small.shape, np.uint8)
    out[after > 0.5] = (200, 30, 30)
    fig, ax = B.canvas(14.5, 6.6, (14.5, 6.4))
    h = 1.7
    row1, row2 = 4.2, 0.9
    B.photo(ax, small, 0.4, row1, h, fs=FS)
    ax.text(0.4 + h * W_ / H_ / 2, row1 + h + 0.12, "input", ha="center", fontsize=10)
    x = 0.4 + h * W_ / H_ + 0.4
    B.arrow(ax, x, row1 + h / 2, x + 0.6, row1 + h / 2)
    B.block(ax, x + 0.7, row1 + 0.1, 2.4, h - 0.2, "DCNN with atrous\n(dilated) convolution\n"
            "(ResNet-50, ASPP)", fc="#dae8fc", fs=FS)
    ax.text(x + 1.9, row1 + h + 0.12, "DCNN", ha="center", fontsize=10)
    x += 3.5
    B.arrow(ax, x, row1 + h / 2, x + 0.6, row1 + h / 2)
    x += 0.7
    ax.imshow(P8[sheep], cmap="jet", vmin=0, vmax=1, extent=(x, x + h * W_ / H_, row1, row1 + h),
              interpolation="nearest")
    ax.text(x + h * W_ / H_ / 2, row1 + h + 0.12, f"'sheep' coarse score map\n({P8.shape[2]} × "
            f"{P8.shape[1]}, stride 8)", ha="center", fontsize=10)
    xc = x + h * W_ / H_ / 2
    B.arrow(ax, xc, row1 - 0.1, xc, row2 + h + 0.55)
    ax.imshow(up, cmap="jet", vmin=0, vmax=1, extent=(x, x + h * W_ / H_, row2, row2 + h))
    ax.text(xc, row2 + h + 0.12, "bilinear interpolation", ha="center", fontsize=10)
    B.arrow(ax, x - 0.1, row2 + h / 2, x - 0.7, row2 + h / 2)
    x2 = x - 0.8 - h * W_ / H_
    ax.imshow(after, cmap="jet", vmin=0, vmax=1, extent=(x2, x2 + h * W_ / H_, row2, row2 + h))
    ax.text(x2 + h * W_ / H_ / 2, row2 + h + 0.12, "fully connected CRF\n(5 mean-field iterations)",
            ha="center", fontsize=10)
    B.arrow(ax, x2 - 0.1, row2 + h / 2, x2 - 0.7, row2 + h / 2)
    x3 = x2 - 0.8 - h * W_ / H_
    ax.imshow(out, extent=(x3, x3 + h * W_ / H_, row2, row2 + h))
    ax.text(x3 + h * W_ / H_ / 2, row2 + h + 0.12, "final output", ha="center", fontsize=10)
    ax.set_xlim(0, x + h * W_ / H_ + 0.4); ax.set_ylim(0.5, 6.6)
    png = save_mpl(fig, ctx, "deeplab_pipeline")
    return Figure("deeplab_pipeline",
                  "The **DeepLab** system, run with the real outputs of DeepLabV3 for the sheep "
                  "photograph. The DCNN, a backbone with atrous (dilated) convolutions, "
                  "produces a coarse score map for each class at 1/8 of the input resolution "
                  "(the 'sheep' map is shown). It is upsampled by bilinear interpolation, "
                  "without a learned decoder or skip connections, and a fully connected CRF "
                  "then sharpens the boundaries to follow the image, giving the final output. "
                  "The CRF step is the one computed in Figure 7 (as in DeepLab v1 and v2; "
                  "torchvision's DeepLabV3 itself has no CRF).",
                  png=png)


def fig_instance_approaches(ctx: BuildContext) -> Figure:
    from .region_proposals import selective_search
    img = load_image("friends", max_side=400)
    cats_d, cats_s = D.categories("faster_rcnn"), D.categories("deeplab")
    det = D.detect("friends", load_image("friends", max_side=800))
    people = [b / 2 for b, l, s in zip(det["boxes"], det["labels"], det["scores"])
              if cats_d[l] == "person" and s > 0.9]
    sem = cv2.resize(D.semantic("friends", load_image("friends", max_side=800))["labels"],
                     (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    person = sem == cats_s.index("person")
    labels, merges, Rg, boxes = selective_search(img)
    icats = D.categories("resnet50")
    fig, ax = B.canvas(15.5, 8.4, (15.5, 8.2))
    # (a) segmentation proposals -> classifier -> instances
    ya = 5.0
    ax.text(0.1, 8.15, "(a) instance segmentation with segmentation proposals", fontsize=10.5,
            weight="bold")
    h = 2.4
    p = B.photo(ax, img, 0.2, ya, h, label="input image", fs=FS)
    regs = []
    for b in people[:3]:
        best = int(np.argmax(D.iou(boxes, b[None])[:, 0]))
        regs.append(best)
    from .region_proposals import labels_after
    B.arrow(ax, p[2] + 0.08, ya + h / 2, p[2] + 0.5, ya + h / 2)
    x = p[2] + 0.6
    for k, i in enumerate(regs):
        b = boxes[i].astype(int)
        crop = img[b[1]:b[3], b[0]:b[2]]
        sz = 0.95
        B.photo(ax, crop, x + k * 1.0, ya + 1.25 - (k % 2) * 1.1, sz)
    ax.text(x + 1.4, ya - 0.08, "segmentation proposals\n(selective search regions)",
            ha="center", va="top", fontsize=FS)
    x += 3.2
    B.arrow(ax, x, ya + h / 2, x + 0.45, ya + h / 2)
    B.block(ax, x + 0.5, ya + h / 2 - 0.45, 1.6, 0.9, "DCNN classifier", fc=B.CONV_FC, fs=FS)
    x += 2.2
    B.arrow(ax, x, ya + h / 2, x + 0.45, ya + h / 2)
    over = img.astype(float).copy()
    for k, i in enumerate(regs):
        b = boxes[i].astype(int)
        m = np.zeros(img.shape[:2], bool); m[b[1]:b[3], b[0]:b[2]] = True
        m &= person
        c = np.array(plt.matplotlib.colors.to_rgb(SERIES[k])) * 255
        over[m] = 0.4 * over[m] + 0.6 * c
    B.photo(ax, over.astype(np.uint8), x + 0.55, ya, h, label="instance segmentation", fs=FS)
    # (b) semantic segmentation cut by detection boxes
    yb = 0.6
    ax.text(0.1, 3.75, "(b) instance segmentation by cutting a semantic segmentation with "
            "object proposals", fontsize=10.5, weight="bold")
    sem_img = img.astype(float).copy()
    sem_img[person] = 0.4 * sem_img[person] + 0.6 * np.array([200, 60, 160])
    p = B.photo(ax, sem_img.astype(np.uint8), 0.2, yb, h, label="semantic segmentation "
                "(DeepLabV3): one 'person' region", fs=FS)
    B.arrow(ax, p[2] + 0.08, yb + h / 2, p[2] + 0.5, yb + h / 2)
    p2 = B.photo(ax, sem_img.astype(np.uint8), p[2] + 0.6, yb, h, boxes=people,
                 box_col=[SERIES[k % 6] for k in range(len(people))], lw=1.6,
                 label="cut with person boxes (Faster R-CNN)", fs=FS)
    B.arrow(ax, p2[2] + 0.08, yb + h / 2, p2[2] + 0.5, yb + h / 2)
    cut = img.astype(float).copy()
    owner = np.full(img.shape[:2], -1)
    overlap = np.zeros(img.shape[:2], int)
    for k, b in enumerate(people):
        m = np.zeros(img.shape[:2], bool)
        m[int(b[1]):int(b[3]), int(b[0]):int(b[2])] = True
        m &= person
        overlap += m
        owner[m & (owner < 0)] = k
    for k in range(len(people)):
        m = owner == k
        c = np.array(plt.matplotlib.colors.to_rgb(SERIES[k % 6])) * 255
        cut[m] = 0.4 * cut[m] + 0.6 * c
    cut[overlap > 1] = (0, 0, 0)
    amb = (overlap > 1).sum() / max(person.sum(), 1)
    B.photo(ax, cut.astype(np.uint8), p2[2] + 0.6, yb, h,
            label=f"instances; black: in two boxes ({amb:.0%} of the person pixels)", fs=FS)
    ax.set_xlim(0, 15.5)
    png = save_mpl(fig, ctx, "instance_approaches")
    return Figure("instance_approaches",
                  "Two earlier approaches to instance segmentation, in which segmentation is "
                  "performed before recognition. (a) Segmentation proposals, here regions "
                  "from selective search (Lesson 3.2) that best fit the people, are each "
                  "classified by a deep network, and the pixels of each region that is "
                  "recognised are labelled as a separate instance (here the person pixels "
                  "inside each region's box are coloured). (b) A semantic segmentation is performed first, in which all the "
                  "people share one label, and object proposals, here Faster R-CNN's person "
                  "boxes, are used to cut the semantic region into instances. Where the boxes "
                  f"overlap, {amb:.0%} of the person pixels here, it is ambiguous which "
                  "instance a pixel belongs to. Both approaches are cumbersome and cannot "
                  "easily be trained end to end, which Mask R-CNN addresses.",
                  png=png)


def fig_mask_rcnn_pipeline(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=800)
    m = D.mask_head_outputs("sheep", img)
    i = 0
    box = m["boxes"][i]
    fig, ax = B.canvas(15.5, 4.6, (15.5, 4.6))
    ym = 1.9
    p = R.backbone(ax, img, 0.1, ym, label="input image")
    B.arrow(ax, p["fmap_right"] + 0.08, ym, p["fmap_right"] + 0.45, ym)
    f2 = p["fmap_right"] + 0.5
    roi = R.roi_fraction(box, img.shape)
    r2 = B.fmap(ax, f2, ym, R.FMAP_LEN, R.FMAP_SIDE, roi=roi, fs=FS)
    top = ym + R.FMAP_SIDE / 2 + OBLIQUE[1] * R.FMAP_SIDE
    rx = p["fmap_x"] + 0.1
    B.arrow(ax, rx + 0.3, top + 0.05, rx + 0.3, top + 0.4, width=0.12)
    B.block(ax, rx - 0.05, top + 0.45, 0.75, 0.5, "RPN", fc=B.CONV_FC, fs=FS)
    tx, ty = B.cap_point(f2 + R.FMAP_LEN, ym, R.FMAP_SIDE, (roi[0] + roi[2]) / 2, roi[1])
    ax.plot([rx + 0.7, tx], [top + 0.7, top + 0.7], color=C["red"], lw=1.6)
    B.line_arrow(ax, tx, top + 0.7, tx, ty + 0.02)
    ax.text((rx + tx) / 2 + 0.3, top + 0.77, "RoI projection", ha="center", fontsize=FS,
            color=C["red"])
    B.arrow(ax, r2 + 0.1, ym, r2 + 0.45, ym)
    ra = volume(ax, r2 + 0.5, ym, 0.4, 0.75, "#f4cccc", lw=1.2)
    ax.text(r2 + 0.7, ym + 0.75, "RoIAlign\n14 × 14 × 256", ha="center", va="bottom", fontsize=FS)
    # box / class branch
    B.line_arrow(ax, ra + 0.05, ym + 0.2, ra + 0.6, ym + 1.4, col=C["red"])
    ax.text(ra + 0.7, ym + 1.45, "bounding box and class\n(as in Faster R-CNN)", fontsize=FS,
            va="center")
    # mask branch
    B.arrow(ax, ra + 0.06, ym, ra + 0.4, ym)
    x = ra + 0.45
    for k in range(4):
        x = volume(ax, x, ym, 0.12, 0.75, "#e1d5e7", lw=0.8) + 0.05
    ax.text(x - 0.45, ym - 0.6, "4 × (3 × 3 conv)", ha="center", va="top", fontsize=FS - 0.5)
    B.arrow(ax, x + 0.03, ym, x + 0.4, ym)
    x2 = volume(ax, x + 0.45, ym, 0.12, 1.2, "#e1d5e7", lw=0.8)
    ax.text(x + 0.55, ym - 0.75, "transpose conv\n28 × 28", ha="center", va="top", fontsize=FS - 0.5)
    B.arrow(ax, x2 + 0.05, ym, x2 + 0.4, ym)
    small = m["small"][i].astype(np.float32)
    ax.imshow(small, cmap="gray", vmin=0, vmax=1, extent=(x2 + 0.45, x2 + 1.65, ym - 0.6, ym + 0.6))
    ax.text(x2 + 1.05, ym - 0.7, f"binary mask, {small.shape[1]} × {small.shape[0]}\n(class "
            "'sheep' of 91)", ha="center", va="top", fontsize=FS - 0.5)
    ax.set_xlim(0, x2 + 2.3)
    png = save_mpl(fig, ctx, "mask_rcnn_pipeline")
    return Figure("mask_rcnn_pipeline",
                  "**Mask R-CNN**: Faster R-CNN with a mask branch. The region proposal network "
                  "proposes regions, which are projected onto the output feature maps, and "
                  "**RoIAlign** converts each region to a fixed-size feature map without "
                  "quantisation. One branch predicts the bounding box and class as in Faster "
                  "R-CNN; the mask branch, a small fully convolutional network, keeps the "
                  "spatial layout and predicts an m × m binary mask for each class. The mask "
                  "shown is Mask R-CNN's real 28 × 28 output for the region of the largest "
                  "sheep (the shapes are those of torchvision's model).",
                  png=png)


def fig_roi_grids(ctx: BuildContext) -> Figure:
    img = load_image("sheep", max_side=800)
    s = 32                                         # a coarse feature grid, stride 32
    h, w = img.shape[:2]
    box = np.array([505.0, 135.0, 668.0, 258.0])   # a proposal around two sheep (image px)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6))
    crop = (420, 70, 760, 330)
    for ax, mode in zip(axes, ("pool", "align")):
        ax.imshow(img, extent=(0, w, h, 0))
        for x in range(0, w + 1, s):
            ax.axvline(x, color="k", lw=0.5, alpha=0.7)
        for y in range(0, h + 1, s):
            ax.axhline(y, color="k", lw=0.5, alpha=0.7)
        ax.add_patch(plt.Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False,
                                   ec=C["red"], lw=1.4, ls="--"))
        if mode == "pool":
            q = np.array([np.floor(box[0] / s), np.floor(box[1] / s), np.ceil(box[2] / s),
                          np.ceil(box[3] / s)]) * s
            ax.add_patch(plt.Rectangle(q[:2], q[2] - q[0], q[3] - q[1], fill=False,
                                       ec=C["red"], lw=2.6))
            ax.plot([(q[0] + q[2]) / 2] * 2, [q[1], q[3]], color=C["red"], lw=1.6)
            ax.plot([q[0], q[2]], [(q[1] + q[3]) / 2] * 2, color=C["red"], lw=1.6)
            err = np.abs(q - box).max()
            ax.set_title(f"(b) RoI pooling: the proposal (dashed) is quantised to whole cells "
                         f"(solid),\nmoving its edges by up to {err:.0f} pixels; 2 × 2 bins "
                         "shown", fontsize=10)
        else:
            ax.add_patch(plt.Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False,
                                       ec=C["red"], lw=2.6))
            bw, bh = (box[2] - box[0]) / 2, (box[3] - box[1]) / 2
            ax.plot([box[0] + bw] * 2, [box[1], box[3]], color=C["red"], lw=1.6)
            ax.plot([box[0], box[2]], [box[1] + bh] * 2, color=C["red"], lw=1.6)
            for i in range(2):
                for j in range(2):
                    for a in range(2):
                        for b_ in range(2):
                            ax.plot(box[0] + j * bw + (b_ + 0.5) * bw / 2,
                                    box[1] + i * bh + (a + 0.5) * bh / 2, "o", color=C["blue"],
                                    ms=6, mec="white")
            ax.set_title("(c) RoIAlign: no quantisation; features are interpolated bilinearly "
                         "at\nregularly spaced points (blue) in each bin; 2 × 2 bins shown",
                         fontsize=10)
        ax.set_xlim(crop[0], crop[2]); ax.set_ylim(crop[3], crop[1])
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "roi_grids")
    return Figure("roi_grids",
                  "RoI pooling and RoIAlign on the photograph, with a grid of feature-map cells "
                  f"at a stride of {s} pixels drawn over the image. (b) In RoI pooling, as in "
                  "Faster R-CNN, the region proposal is quantised to the cells of the output "
                  "feature map and the features are pooled within bins of the region, giving "
                  "a fixed-size output for the classifier and box predictor; the quantisation "
                  "misaligns the pooled features with the proposed region. (c) In RoIAlign, "
                  "used in Mask R-CNN, the proposal is not quantised, and the feature map is "
                  "interpolated at points within each bin, which preserves the spatial "
                  "correspondence between the features and the region. The next figure "
                  "computes both on a small feature map.",
                  png=png)
