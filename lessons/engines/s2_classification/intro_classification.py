"""Lesson 2.1 — Introduction to image classification.

Figures
    classification_task     one dominant subject per image -> one label
    recognition_challenges  viewpoint, illumination, scale, occlusion, clutter,
                            deformation and intra-class variation
    ilsvrc_progress         ImageNet LSVRC winning top-5 error by year, with depth
    binary_metrics          pedestrian confusion matrix -> accuracy, precision, recall
    multiclass_metrics      one-vs-rest TP/FP/FN/TN per class, macro and weighted means
    top1_top5               the same test image scored by two classifiers
    imbalance_and_shift     per-class recall falls with class frequency; accuracy falls
                            under domain shift (both measured on real digits)

Source: reference/legacy_md/lesson_2_1_introduction + lesson_2_1_script.
"""
from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       load_image, use_source, sync_matrix)

FIGURES = ["classification_task", "recognition_challenges", "ilsvrc_progress",
           "binary_metrics", "multiclass_metrics", "top1_top5",
           "imbalance_and_shift"]

# ImageNet LSVRC classification winners (top-5 error, %), with depth. These are
# the published challenge results, not computed here; 2010-11 entries were not
# neural networks. Human reference: Russakovsky et al. 2015 (Karpathy's study).
ILSVRC = [
    (2010, "NEC-UIUC", 28.2, None),
    (2011, "XRCE", 25.8, None),
    (2012, "AlexNet", 16.4, 8),
    (2013, "ZFNet", 11.7, 8),
    (2014, "GoogLeNet", 6.7, 22),
    (2015, "ResNet", 3.57, 152),
    (2016, "Trimps-Soushen", 2.99, None),
    (2017, "SENet", 2.25, 152),
]
HUMAN_TOP5 = 5.1

# Slide 10's three-class example (rows = actual, columns = predicted). The
# counts are those of Grasple exercise 107739, so the figure and the online
# exercise agree.
MC_CLASSES = ["cat", "dog", "mouse"]
MC_MATRIX = np.array([[19, 2, 3],
                      [3, 17, 7],
                      [9, 6, 31]])

# Slide 9's binary example.
BIN = dict(TP=46, FP=7, FN=12, TN=135)


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def one_vs_rest(M, i):
    """TP, FP, FN, TN for class i of a confusion matrix (rows actual, cols predicted)."""
    tp = M[i, i]
    fp = M[:, i].sum() - tp          # predicted i, actually something else
    fn = M[i, :].sum() - tp          # actually i, predicted something else
    tn = M.sum() - tp - fp - fn
    return int(tp), int(fp), int(fn), int(tn)


def metrics(tp, fp, fn, tn):
    return dict(accuracy=(tp + tn) / (tp + fp + fn + tn),
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0)


def per_class_table(M):
    rows = []
    for i in range(len(M)):
        tp, fp, fn, tn = one_vs_rest(M, i)
        rows.append(dict(tp=tp, fp=fp, fn=fn, tn=tn, support=int(M[i].sum()),
                         **metrics(tp, fp, fn, tn)))
    return rows


def _digits_split(seed=0):
    from sklearn.datasets import load_digits
    d = load_digits()
    X, y = d.data / 16.0, d.target
    idx = np.random.default_rng(seed).permutation(len(y))
    return X[idx[:1200]], y[idx[:1200]], X[idx[1200:]], y[idx[1200:]]


def imbalance_experiment(seed=0):
    """Train on digits with deliberately skewed class counts; measure per-class recall."""
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr, Xte, yte = _digits_split(seed)
    rng = np.random.default_rng(seed)
    # class k keeps a fraction that falls geometrically: 100% for 0 down to ~3% for 9
    keep_frac = 0.68 ** np.arange(10)
    keep = np.concatenate([rng.permutation(np.where(ytr == k)[0])
                           [:max(3, int(keep_frac[k] * (ytr == k).sum()))]
                           for k in range(10)])
    clf = LogisticRegression(max_iter=3000, C=0.5).fit(Xtr[keep], ytr[keep])
    pred = clf.predict(Xte)
    counts = np.array([(ytr[keep] == k).sum() for k in range(10)])
    recall = np.array([(pred[yte == k] == k).mean() for k in range(10)])
    return counts, recall


def shift_experiment(seed=0):
    """Train on clean digits; test with increasing contrast reduction and noise,
    a stand-in for 'trained on day, tested at night'."""
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr, Xte, yte = _digits_split(seed)
    clf = LogisticRegression(max_iter=3000, C=0.5).fit(Xtr, ytr)
    rng = np.random.default_rng(seed + 1)
    levels = np.linspace(0, 1, 9)
    acc = []
    for s in levels:
        Xs = Xte * (1 - 0.75 * s) + rng.normal(0, 0.25 * s, Xte.shape)
        acc.append((clf.predict(np.clip(Xs, 0, 1)) == yte).mean())
    return levels, np.array(acc)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_classification_task(ctx: BuildContext) -> Figure:
    items = [("lion", "lion"), ("macaw", "macaw"), ("zebra", "zebra"),
             ("minivan", "minivan")]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.1),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.15]))
    for ax, (key, label) in zip(axes, items):
        img = load_image(key, max_side=260)
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"label: {label}", fontsize=10, color=C["green"])
    img = load_image("workbench", max_side=260)
    axes[4].imshow(img); axes[4].set_xticks([]); axes[4].set_yticks([])
    axes[4].set_title("label: ?  (no single subject)", fontsize=10, color=C["red"])
    for s in axes[4].spines.values():
        s.set_edgecolor(C["red"]); s.set_linewidth(2)
    png = save_mpl(fig, ctx, "classification_task")
    return Figure("classification_task",
                  "In image classification, one label is assigned to a whole image according "
                  "to its content. To avoid ambiguity, the images should have a single "
                  "subject that dominates the frame, as in the first four images. The last "
                  "image, a workbench covered in many objects, has no such subject and "
                  "cannot be described by a single label; this is a task for object "
                  "detection, which is described in Section 3.",
                  png=png)


def fig_recognition_challenges(ctx: BuildContext) -> Figure:
    base = load_image("labrador", max_side=300)
    h, w = base.shape[:2]

    def persp(img):
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[0.18 * w, 0.06 * h], [0.92 * w, 0], [w, h], [0.05 * w, 0.92 * h]])
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                                   borderMode=cv2.BORDER_REFLECT_101)

    def dark(img):
        f = (img / 255.0) ** 2.6 * 0.55
        return (np.clip(f, 0, 1) * 255).astype(np.uint8)

    def small(img):
        out = np.full_like(img, 0)
        out[:] = cv2.resize(img, (16, 12), interpolation=cv2.INTER_AREA).mean((0, 1))
        s = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        out[int(h * 0.55):int(h * 0.55) + s.shape[0], int(w * 0.6):int(w * 0.6) + s.shape[1]] = s
        return out

    def occlude(img):
        out = img.copy()
        cv2.rectangle(out, (int(0.05 * w), int(0.10 * h)), (int(0.55 * w), int(0.70 * h)),
                      (110, 110, 110), -1)
        return out

    clutter_bg = load_image("workbench", max_side=300)
    clutter_bg = cv2.resize(clutter_bg, (w, h))

    def clutter(img):
        out = clutter_bg.copy()
        s = cv2.resize(img, (w // 2, h // 2))
        out[h // 4:h // 4 + s.shape[0], w // 3:w // 3 + s.shape[1]] = s
        return out

    panels = [("original", base), ("viewpoint", persp(base)),
              ("illumination", dark(base)), ("scale", small(base)),
              ("occlusion", occlude(base)), ("background clutter", clutter(base)),
              ("intra-class variation", load_image("agathe_cat", max_side=300)),
              ("intra-class variation", load_image("cat", max_side=300))]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.2))
    for ax, (name, img) in zip(axes.ravel(), panels):
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=10)
    axes[1, 2].set_title("intra-class variation:\ntwo cats", fontsize=10)
    axes[1, 3].set_title("… that share little\nat the pixel level", fontsize=10)
    png = save_mpl(fig, ctx, "recognition_challenges")
    return Figure("recognition_challenges",
                  "Difficulties in classification. The same dog is shown under synthetic "
                  "changes of viewpoint, illumination and scale, partly occluded, and "
                  "against a cluttered background. To a human observer it is clearly the "
                  "same animal throughout, but the pixel values have changed almost "
                  "completely. Intra-class variation is shown at the bottom right, with two "
                  "cats that share a label but little else. Real images also contain "
                  "deformable objects, whose shape changes with pose. Deep networks are able "
                  "to address these difficulties because features that are robust to all of "
                  "them are learned from many labelled examples.",
                  png=png)


def fig_ilsvrc_progress(ctx: BuildContext) -> Figure:
    years = [r[0] for r in ILSVRC]
    err = [r[2] for r in ILSVRC]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    cols = [C["grey"] if r[3] is None and r[0] < 2012 else C["blue"] for r in ILSVRC]
    bars = ax.bar(years, err, color=cols, edgecolor=C["ink"], lw=0.8, width=0.7)
    for b, (yr, name, e, depth) in zip(bars, ILSVRC):
        ax.text(b.get_x() + b.get_width() / 2, e + 0.5, f"{e:g}%", ha="center",
                fontsize=9, weight="bold")
        lab = name + (f"\n{depth} layers" if depth else "")
        ax.text(b.get_x() + b.get_width() / 2, -1.2, lab, ha="center", va="top",
                fontsize=8.2)
    ax.axhline(HUMAN_TOP5, color=C["red"], ls="--", lw=1.5)
    ax.text(2017.45, HUMAN_TOP5 + 0.45, f"human ≈ {HUMAN_TOP5}%", color=C["red"],
            fontsize=9, ha="right")
    ax.annotate("AlexNet: the first deep CNN,\n~10 points better than the\n"
                "best classical approach", xy=(2012, 16.4), xytext=(2013.4, 22),
                fontsize=8.8, arrowprops=dict(arrowstyle="->", color=C["grey"]))
    ax.set_xticks(years); ax.set_xticklabels([str(y) for y in years])
    ax.tick_params(axis="x", pad=34)
    ax.set_ylabel("top-5 error (%)")
    ax.set_ylim(0, 31)
    ax.set_title("ImageNet Large Scale Visual Recognition Challenge: winning top-5 "
                 "error by year (grey = not a neural network)", fontsize=10.5)
    png = save_mpl(fig, ctx, "ilsvrc_progress")
    return Figure("ilsvrc_progress",
                  "The ImageNet challenge was compiled to promote progress in image "
                  "classification. In 2012 AlexNet, the first deep convolutional network to "
                  "be entered, outperformed the best classical approach by approximately ten "
                  "percentage points, and this result marked the start of the deep learning "
                  "era. The error was then approximately halved every two years until 2015, "
                  "when the human estimate was surpassed by ResNet and the depth increased "
                  "from 22 layers to 152; after this the improvements level off. The "
                  "published challenge results are shown. AlexNet, VGG, GoogLeNet and ResNet "
                  "are described in the next three lessons.",
                  png=png)


def fig_binary_metrics(ctx: BuildContext) -> Figure:
    tp, fp, fn, tn = BIN["TP"], BIN["FP"], BIN["FN"], BIN["TN"]
    m = metrics(tp, fp, fn, tn)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2),
                             gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    M = np.array([[tp, fn], [fp, tn]])
    labels = [["TP", "FN"], ["FP", "TN"]]
    tint = [[C["green"], C["orange"]], [C["red"], C["sky"]]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, fc=tint[i][j], alpha=0.35,
                                       ec=C["ink"], lw=1.2))
            ax.text(j + 0.5, 1.5 - i + 0.12, labels[i][j], ha="center", fontsize=11,
                    weight="bold")
            ax.text(j + 0.5, 1.5 - i - 0.15, str(M[i, j]), ha="center", fontsize=13)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["pedestrian", "no pedestrian"])
    ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(["pedestrian", "no pedestrian"])
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    for s in ax.spines.values():
        s.set_visible(False)

    ax = axes[1]
    ax.axis("off")
    lines = [
        ("accuracy", "(TP + TN) / all", f"({tp} + {tn}) / {tp + fp + fn + tn}",
         m["accuracy"], "how often is the classifier right?"),
        ("precision", "TP / (TP + FP)", f"{tp} / ({tp} + {fp})", m["precision"],
         "when it says 'pedestrian', how often is it right?"),
        ("recall", "TP / (TP + FN)", f"{tp} / ({tp} + {fn})", m["recall"],
         "of the real pedestrians, how many did it find?"),
    ]
    for k, (name, formula, sub, val, q) in enumerate(lines):
        y = 0.86 - k * 0.31
        ax.text(0.0, y, name, fontsize=12, weight="bold")
        ax.text(0.0, y - 0.09, q, fontsize=9, color=C["grey"], style="italic")
        ax.text(0.38, y, f"= {formula} = {sub} = {val:.3f}", fontsize=10.5,
                family="monospace")
    png = save_mpl(fig, ctx, "binary_metrics")
    return Figure("binary_metrics",
                  "A confusion matrix for a binary pedestrian classifier and the three "
                  "metrics calculated from it. There are four outcomes: a pedestrian that is "
                  "correctly detected is a **true positive**; a pedestrian reported where "
                  "there is none is a **false positive**; a pedestrian that is missed is a "
                  "**false negative**; and background that is correctly identified is a "
                  "**true negative**. Accuracy is the fraction of all predictions that are "
                  "correct. Precision is a measure of how reliable a positive prediction is, "
                  "and a high precision indicates few false positives. Recall is the "
                  "proportion of the actual positives that are found, and a high recall "
                  "indicates few missed detections.",
                  png=png)


def fig_multiclass_metrics(ctx: BuildContext) -> Figure:
    M = MC_MATRIX
    rows = per_class_table(M)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.35]))
    roles = {"TP": C["green"], "FP": C["red"], "FN": C["orange"], "TN": C["sky"]}
    for ax, i in zip(axes[:3], range(3)):
        for r in range(3):
            for c in range(3):
                role = ("TP" if r == i and c == i else "FP" if c == i else
                        "FN" if r == i else "TN")
                ax.add_patch(plt.Rectangle((c, 2 - r), 1, 1, fc=roles[role], alpha=0.35,
                                           ec=C["ink"], lw=1))
                ax.text(c + 0.5, 2.5 - r + 0.1, str(M[r, c]), ha="center", fontsize=11)
                ax.text(c + 0.5, 2.5 - r - 0.22, role, ha="center", fontsize=7.5,
                        color=C["grey"])
        ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.set_aspect("equal")
        ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(MC_CLASSES, fontsize=9)
        ax.set_yticks([2.5, 1.5, 0.5]); ax.set_yticklabels(MC_CLASSES, fontsize=9)
        ax.xaxis.tick_top()
        ax.set_title(f"one vs rest: {MC_CLASSES[i]}", fontsize=10, pad=22)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("actual")
    fig.text(0.04, 0.03, "rows: actual class    columns: predicted class", fontsize=9,
             color=C["grey"])

    ax = axes[3]
    ax.axis("off")
    hdr = ["class", "TP", "FP", "FN", "TN", "acc", "prec", "rec"]
    cells = [[MC_CLASSES[i], r["tp"], r["fp"], r["fn"], r["tn"],
              f"{r['accuracy']:.2f}", f"{r['precision']:.2f}", f"{r['recall']:.2f}"]
             for i, r in enumerate(rows)]
    sup = np.array([r["support"] for r in rows])
    macro =[np.mean([r[k] for r in rows]) for k in ("accuracy", "precision", "recall")]
    wtd = [np.sum([r[k] * s for r, s in zip(rows, sup)]) / sup.sum()
           for k in ("accuracy", "precision", "recall")]
    cells.append(["macro", "", "", "", "", *[f"{v:.2f}" for v in macro]])
    cells.append(["weighted", "", "", "", "", *[f"{v:.2f}" for v in wtd]])
    tbl = ax.table(cellText=cells, colLabels=hdr, loc="center", cellLoc="center",
                   colWidths=[0.22] + [0.1] * 4 + [0.12] * 3)
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.55)
    for (r_, c_), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["light"])
        if r_ == 0 or r_ > 3:
            cell.set_text_props(weight="bold")
    png = save_mpl(fig, ctx, "multiclass_metrics")
    return Figure("multiclass_metrics",
                  "In the multi-class case each metric is calculated one class at a time, "
                  "with that class treated as positive and every other class as negative. "
                  "The actual labels are given by the rows and the predictions by the "
                  "columns. For a given class, the true positive is the diagonal cell; the "
                  "false positives are the remaining cells of its **column** (predicted as "
                  "this class but actually another class); the false negatives are the "
                  "remaining cells of its **row** (actually this class but predicted as "
                  "another class); and all other cells are true negatives. The per-class "
                  "values are then combined: in the **macro** average every class is "
                  "weighted equally, so the average is reduced by a poorly performing class, "
                  "while in the **weighted** average each class is weighted by its number of "
                  "examples. The two averages are equal only when the classes are balanced. "
                  "These counts match the online exercise, so the table also serves as its "
                  "answer key.",
                  png=png)


def fig_top1_top5(ctx: BuildContext) -> Figure:
    img = load_image("lion", max_side=320)
    classes = ["lion", "tiger", "cheetah", "leopard", "cougar", "jaguar", "lynx"]
    pa = np.array([0.62, 0.14, 0.09, 0.06, 0.04, 0.03, 0.02])
    pb = np.array([0.21, 0.27, 0.24, 0.12, 0.08, 0.05, 0.03])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9),
                             gridspec_kw=dict(width_ratios=[0.95, 1, 1]))
    axes[0].imshow(img); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("test image — ground truth: lion", fontsize=10)
    for ax, p, name in [(axes[1], pa, "classifier A"), (axes[2], pb, "classifier B")]:
        order = np.argsort(p)[::-1]
        top5 = set(order[:5])
        cols = [C["green"] if classes[i] == "lion" else
                (C["sky"] if i in top5 else C["light"]) for i in order]
        ax.barh(range(len(p)), p[order], color=cols, edgecolor=C["grey"])
        ax.set_yticks(range(len(p))); ax.set_yticklabels([classes[i] for i in order])
        ax.invert_yaxis()
        ax.axhline(4.5, color=C["grey"], ls=":", lw=1)
        ax.text(p.max() * 0.98, 4.35, "top 5 above this line", ha="right", fontsize=8,
                color=C["grey"])
        t1 = classes[order[0]] == "lion"
        t5 = "lion" in [classes[i] for i in order[:5]]
        ax.set_title(f"{name}:  top-1 {'✓' if t1 else '✗'}   top-5 {'✓' if t5 else '✗'}",
                     fontsize=10.5, color=C["green"] if t1 else C["red"])
        ax.set_xlabel("output probability")
    png = save_mpl(fig, ctx, "top1_top5")
    return Figure("top1_top5",
                  "Top-1 and top-5 accuracy, the two metrics used for ImageNet. A probability "
                  "is output by a deep classifier for every class. For **top-1** accuracy, "
                  "the highest probability must correspond to the ground truth: classifier A "
                  "passes, but classifier B ranks tiger above lion and fails. For **top-5** "
                  "accuracy, the true label need only appear among the five highest "
                  "probabilities, in any order, so both classifiers pass. With 1000 classes, "
                  "many of which are very similar, top-1 is a stringent test, and for this "
                  "reason top-5 was introduced; the top-5 accuracy of a classifier is always "
                  "at least as high as its top-1 accuracy. (The probabilities shown are an "
                  "illustrative example, not the output of a network.)",
                  png=png)


def fig_imbalance_and_shift(ctx: BuildContext) -> Figure:
    counts, recall = imbalance_experiment()
    levels, acc = shift_experiment()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.scatter(counts, recall, s=60, color=C["blue"], zorder=3)
    for k, (n, r) in enumerate(zip(counts, recall)):
        ax.annotate(str(k), (n, r), xytext=(5, 4), textcoords="offset points",
                    fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("training examples of the class (log scale)")
    ax.set_ylabel("recall on the test set")
    ax.set_ylim(0, 1.05)
    rho = np.corrcoef(np.log(counts), recall)[0, 1]
    ax.set_title(f"(a) under-represented classes are recognised worst\n"
                 f"(correlation of log count with recall: {rho:.2f})", fontsize=10)

    ax = axes[1]
    ax.plot(levels, acc, color=C["red"], lw=2.2, marker="o")
    ax.set_xlabel("severity of the shift (contrast loss + noise)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"(b) domain shift: same classifier, different conditions\n"
                 f"({acc[0]:.0%} on the training domain → {acc[-1]:.0%} at the worst)",
                 fontsize=10)
    png = save_mpl(fig, ctx, "imbalance_and_shift")
    return Figure("imbalance_and_shift",
                  "Two failure modes, both **measured** on a classifier trained on "
                  "handwritten digits. (a) **Imbalance and confusion.** The training set is "
                  "reduced so that each digit has fewer examples than the previous one; the "
                  "classes with the fewest training examples are recognised least well, as "
                  "there was less opportunity for the features that distinguish them to be "
                  "learned. This is how bias is introduced into a trained network: the "
                  "performance is poorest for whatever is under-represented in the training "
                  "data. (b) **Robustness to domain shift.** The performance of a classifier "
                  "depends on the match between its training and test conditions. When the "
                  "same classifier is tested on images with reduced contrast and added "
                  "noise, representing a network trained on daytime scenes and deployed at "
                  "night, its accuracy is steadily reduced, although the digits themselves "
                  "are unchanged.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_classification_task(ctx), fig_recognition_challenges(ctx),
            fig_ilsvrc_progress(ctx), fig_binary_metrics(ctx),
            fig_multiclass_metrics(ctx), fig_top1_top5(ctx),
            fig_imbalance_and_shift(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("2.1 Introduction to image classification")
    tab1, tab2 = st.tabs(["Confusion matrix", "Top-k accuracy"])

    with tab1:
        st.markdown("Edit the confusion matrix (rows = actual, columns = predicted).")
        txt = st.text_area("matrix", "\n".join(" ".join(map(str, r)) for r in MC_MATRIX),
                           height=110)
        try:
            M = np.array([[int(v) for v in r.split()] for r in txt.strip().splitlines()])
            assert M.shape[0] == M.shape[1]
        except Exception:
            st.error("Enter a square matrix of integers."); return
        rows = per_class_table(M)
        st.dataframe([{"class": i, "TP": r["tp"], "FP": r["fp"], "FN": r["fn"],
                       "TN": r["tn"], "accuracy": round(r["accuracy"], 3),
                       "precision": round(r["precision"], 3),
                       "recall": round(r["recall"], 3)} for i, r in enumerate(rows)],
                     width="stretch")
        sup = np.array([r["support"] for r in rows])
        c = st.columns(2)
        c[0].metric("macro recall", f"{np.mean([r['recall'] for r in rows]):.3f}")
        c[1].metric("weighted recall",
                    f"{np.sum([r['recall'] * s for r, s in zip(rows, sup)]) / sup.sum():.3f}")

    with tab2:
        k = st.slider("k", 1, 7, 5)
        txt = st.text_input("output probabilities", "0.21, 0.27, 0.24, 0.12, 0.08, 0.05, 0.03")
        true = st.number_input("index of the true class", 0, 6, 0)
        try:
            p = np.array([float(v) for v in txt.split(",")])
        except ValueError:
            st.error("Use comma-separated numbers."); return
        order = np.argsort(p)[::-1]
        hit = int(true) in order[:k]
        st.write(f"ranking: {list(order)}")
        (st.success if hit else st.error)(
            f"top-{k}: the true class is ranked {list(order).index(int(true)) + 1} → "
            f"{'correct' if hit else 'wrong'}")
