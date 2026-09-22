"""Lesson 4.5 — The Viola-Jones face detector.

Figures
    haar_features       the basic rectangle features and scaled / shifted copies inside a 24x24 window
    integral_image      column then row accumulation; a rectangle sum from four values
    feature_on_face     two classic features (eyes vs cheeks, eyes vs nose bridge) evaluated on a face
    adaboost            weak classifiers (decision stumps) combined by AdaBoost on 2D data
    cascade             the attention cascade: most windows rejected early
    face_detection      OpenCV Haar cascade detections on photographs
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from .._common import BuildContext, Figure, C, SERIES, save_mpl, load_image, sync_matrix
from .machine_learning import moons, decision_regions

FIGURES = ["haar_features", "integral_image", "feature_on_face", "adaboost", "cascade", "face_detection"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def integral(img):
    return img.cumsum(axis=0).cumsum(axis=1)


def rect_sum(ii, r0, c0, h, w):
    """Sum of img[r0:r0+h, c0:c0+w] from the integral image (A − B − C + D)."""
    A = ii[r0 + h - 1, c0 + w - 1]
    B = ii[r0 - 1, c0 + w - 1] if r0 > 0 else 0
    Cc = ii[r0 + h - 1, c0 - 1] if c0 > 0 else 0
    D = ii[r0 - 1, c0 - 1] if r0 > 0 and c0 > 0 else 0
    return A - B - Cc + D


def face_crop(key="jimmy", size=96):
    rgb = load_image(key, max_side=900)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    fs = cas.detectMultiScale(g, 1.1, 5)
    x, y, w, h = max(fs, key=lambda r: r[2] * r[3])
    return cv2.resize(g[y:y + h, x:x + w], (size, size), interpolation=cv2.INTER_AREA)


def detect_faces(g, scale=1.1, neighbours=5):
    cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return cas.detectMultiScale(g, scale, neighbours)


def draw_feature(ax, rects, title):
    """rects: list of (x, y, w, h, sign) with sign +1 white, -1 black, on a unit-ish grid."""
    for x, y, w, h, s in rects:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="white" if s > 0 else "black", ec="k", lw=1))
    ax.set_xlim(-0.2, 4.2); ax.set_ylim(4.2, -0.2); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=8)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_haar_features(ctx: BuildContext) -> Figure:
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.3, 2.6])
    feats = [([(0, 1, 2, 1, -1), (2, 1, 2, 1, 1)], "two-rectangle (vertical edge)"),
             ([(0, 0.5, 4, 1.5, -1), (0, 2, 4, 1.5, 1)], "two-rectangle (horizontal edge)"),
             ([(0, 1, 1.33, 2, 1), (1.33, 1, 1.33, 2, -1), (2.66, 1, 1.33, 2, 1)], "three-rectangle (line)"),
             ([(0, 0, 2, 2, 1), (2, 0, 2, 2, -1), (0, 2, 2, 2, -1), (2, 2, 2, 2, 1)], "four-rectangle (diagonal)"),
             ([(0.5, 0, 3, 1.33, 1), (0.5, 1.33, 3, 1.33, -1), (0.5, 2.66, 3, 1.33, 1)], "three-rectangle (horizontal line)"),
             ([(1, 0.5, 1, 3, -1), (2, 0.5, 1, 3, 1)], "two-rectangle, narrow"),
             ([(0, 0, 4, 2, -1), (0, 2, 4, 2, 1)], "two-rectangle, full window"),
             ([(1.5, 1.5, 1, 1, -1), (2.5, 1.5, 1, 1, 1)], "two-rectangle, small")]
    for i, (rects, t) in enumerate(feats):
        ax = fig.add_subplot(gs[i // 4, i % 4]); draw_feature(ax, rects, t)
    ax = fig.add_subplot(gs[:, 5])
    ax.set_xlim(-0.5, 24.5); ax.set_ylim(24.5, -0.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for k in range(25):
        ax.axhline(k, color=C["light"], lw=0.5); ax.axvline(k, color=C["light"], lw=0.5)
    ax.add_patch(plt.Rectangle((0, 0), 24, 24, fill=False, ec=C["ink"], lw=1.5))
    # a few two-rectangle features at different scales and positions, drawn crisply
    for x, y, w, h, vertical in [(2, 2, 3, 6, True), (13, 1, 5, 4, False), (4, 12, 2, 3, True),
                                 (9, 9, 6, 6, True), (15, 16, 4, 3, False), (1, 19, 3, 2, True)]:
        if vertical:                                    # black | white, side by side
            ax.add_patch(plt.Rectangle((x, y), w, h, fc="black", ec="k", lw=0.8))
            ax.add_patch(plt.Rectangle((x + w, y), w, h, fc="white", ec="k", lw=0.8))
        else:                                           # black over white
            ax.add_patch(plt.Rectangle((x, y), w, h, fc="black", ec="k", lw=0.8))
            ax.add_patch(plt.Rectangle((x, y + h), w, h, fc="white", ec="k", lw=0.8))
    ax.set_title("six of the features that fit inside the 24 × 24\ndetector window: over 180,000 in total", fontsize=9)
    png = save_mpl(fig, ctx, "haar_features")
    return Figure("haar_features",
                  "Haar-like features. Each feature is the sum of the pixels under its "
                  "white rectangles minus the sum under its black rectangles, so it "
                  "responds to an edge, a line or a diagonal pattern of contrast. Every "
                  "feature type can be scaled and shifted anywhere inside the 24 × 24 "
                  "detector window, giving a very large pool of candidate features.",
                  png=png)


def fig_integral_image(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(7)
    img = rng.integers(0, 6, (5, 5))
    col = img.cumsum(axis=0); ii = integral(img)
    r0, c0, h, w = 1, 1, 3, 3
    s = rect_sum(ii, r0, c0, h, w)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    sync_matrix(axes[0], img, fmt="{:.0f}", title="image", highlight=(r0, c0, h, w))
    sync_matrix(axes[1], col, fmt="{:.0f}", cmap="Greens", title="step 1: accumulate down each column")
    sync_matrix(axes[2], ii, fmt="{:.0f}", cmap="Purples", title="step 2: accumulate along each row = integral image")
    for (rr, cc), lab in [((r0 + h - 1, c0 + w - 1), "A"), ((r0 - 1, c0 + w - 1), "B"), ((r0 + h - 1, c0 - 1), "C"), ((r0 - 1, c0 - 1), "D")]:
        axes[2].add_patch(plt.Rectangle((cc - 0.5, rr - 0.5), 1, 1, fill=False, ec=C["red"], lw=2)); axes[2].text(cc + 0.32, rr - 0.28, lab, color=C["red"], fontsize=9, weight="bold")
    axes[3].axis("off")
    A, B, Cc, D = ii[r0 + h - 1, c0 + w - 1], ii[r0 - 1, c0 + w - 1], ii[r0 + h - 1, c0 - 1], ii[r0 - 1, c0 - 1]
    axes[3].text(0, 0.85, "sum of the red 3 × 3 region:", fontsize=10)
    axes[3].text(0, 0.65, f"A − B − C + D = {A} − {B} − {Cc} + {D} = {s}", fontsize=12)
    axes[3].text(0, 0.45, f"direct sum of the 9 pixels: {img[r0:r0 + h, c0:c0 + w].sum()} ✓", fontsize=10)
    axes[3].text(0, 0.2, "Any rectangle, whatever its size,\ncosts four lookups.", fontsize=9, color=C["grey"], va="top")
    png = save_mpl(fig, ctx, "integral_image")
    return Figure("integral_image",
                  "The integral image (summed-area table). Each entry holds the sum of all "
                  "pixels above and to the left, computed once by accumulating down the "
                  "columns and then along the rows. The sum of any rectangle is then "
                  "A − B − C + D from its four corner entries, so large features cost no "
                  "more than small ones.", png=png)


def fig_feature_on_face(ctx: BuildContext) -> Figure:
    f = face_crop("jimmy", 96).astype(float)
    ii = integral(f)
    mean = lambda r0, c0, h, w: rect_sum(ii, r0, c0, h, w) / (h * w)  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(f, cmap="gray"); axes[0].set_title("face, 96 × 96")
    # feature 1: eye band (dark) above the cheek band (bright)
    y1, hh, x1, ww = 26, 12, 14, 68
    axes[1].imshow(f, cmap="gray")
    axes[1].add_patch(plt.Rectangle((x1, y1), ww, hh, fc="black", alpha=0.45))
    axes[1].add_patch(plt.Rectangle((x1, y1 + hh), ww, hh, fc="white", alpha=0.45))
    m_eye, m_cheek = mean(y1, x1, hh, ww), mean(y1 + hh, x1, hh, ww)
    axes[1].set_title(f"mean intensity: eyes {m_eye:.0f}, cheeks {m_cheek:.0f}\nfeature = cheeks − eyes = {m_cheek - m_eye:+.0f}", fontsize=9)
    # feature 2: the two eyes (dark) either side of the nose bridge (bright)
    y2, hh2, x2, ww2 = 26, 12, 22, 17
    axes[2].imshow(f, cmap="gray")
    for k, s in enumerate([-1, 1, -1]):
        axes[2].add_patch(plt.Rectangle((x2 + k * ww2, y2), ww2, hh2, fc="black" if s < 0 else "white", alpha=0.45))
    m_bridge = mean(y2, x2 + ww2, hh2, ww2)
    m_eyes = 0.5 * (mean(y2, x2, hh2, ww2) + mean(y2, x2 + 2 * ww2, hh2, ww2))
    axes[2].set_title(f"mean intensity: eyes {m_eyes:.0f}, bridge {m_bridge:.0f}\nfeature = bridge − eyes = {m_bridge - m_eyes:+.0f}", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "feature_on_face")
    return Figure("feature_on_face",
                  "Interpreting the selected features. The first two features chosen by "
                  "AdaBoost in the original paper exploit two properties of faces: the "
                  "eye region is darker than the cheeks and the top of the nose, and the "
                  "eyes are darker than the bridge of the nose between them. Both come "
                  "out positive here. Mean intensities are shown so that the two "
                  "rectangles are comparable; each is four integral-image lookups.",
                  png=png)


def fig_adaboost(ctx: BuildContext) -> Figure:
    X, y = moons(300, 0.25, 1)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, n in zip(axes, [1, 3, 10, 50]):
        clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=n, random_state=0).fit(X, y)
        xx, yy, Z = decision_regions(clf, X)
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[C["sky"], "#f6c8b0"], alpha=0.5)
        for c in (0, 1):
            ax.plot(X[y == c, 0], X[y == c, 1], "o", color=SERIES[c], ms=3)
        ax.set_title(f"{n} weak classifier{'s' if n > 1 else ''} combined: accuracy {clf.score(X, y):.2f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "adaboost")
    return Figure("adaboost",
                  "AdaBoost. Each weak classifier here is a single threshold on one "
                  "feature, a straight cut of the plane that on its own is barely better "
                  "than chance. Boosting trains them in sequence, each concentrating on "
                  "the points the previous ones got wrong, and combines their weighted "
                  "votes into an accurate classifier. Viola-Jones uses the same idea with "
                  "one rectangle feature per weak classifier.", png=png)


def fig_cascade(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 3.6)); ax.set_xlim(0, 14); ax.set_ylim(0.2, 3.6); ax.axis("off")
    stages = [("stage 1\n1 feature", 1.0), ("stage 2\n10 features", 0.5), ("stage 3\n25 features", 0.2), ("…", 0.05), ("stage 38\n~200 features", 0.002)]
    x = 0.3
    ax.text(0.3, 3.3, "every window of the image", fontsize=9, weight="bold", color=C["grey"])
    for i, (name, frac) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((x, 1.6), 1.8, 1.2, boxstyle="round,pad=0.02", fc="#e8f1f8", ec=C["grey"]))
        ax.text(x + 0.9, 2.2, name, ha="center", va="center", fontsize=8)
        ax.text(x + 0.9, 3.0, f"{frac * 100:g}% of windows arrive", ha="center", fontsize=7, color=C["grey"])
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.85, 2.2), (x + 2.35, 2.2), arrowstyle="->", mutation_scale=12, color=C["ink"]))
            ax.text(x + 2.1, 2.35, "face?", ha="center", fontsize=7, color=C["green"])
        ax.add_patch(FancyArrowPatch((x + 0.9, 1.55), (x + 0.9, 0.9), arrowstyle="->", mutation_scale=12, color=C["red"]))
        ax.text(x + 1.0, 1.15, "reject: not a face", fontsize=7, color=C["red"])
        x += 2.4
    ax.add_patch(FancyBboxPatch((x, 1.6), 1.4, 1.2, boxstyle="round,pad=0.02", fc="#e6f4ee", ec=C["grey"])); ax.text(x + 0.7, 2.2, "face", ha="center", va="center", fontsize=9, weight="bold")
    png = save_mpl(fig, ctx, "cascade")
    return Figure("cascade",
                  "The attention cascade. Weak classifiers are grouped into stages of "
                  "increasing size; the first stages use the features with the lowest "
                  "false-negative rate, so they almost never reject a real face but "
                  "discard most background windows immediately. Only windows that pass "
                  "every stage are declared faces, so the average window costs very few "
                  "feature evaluations.", png=png)


def fig_face_detection(ctx: BuildContext) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, key in zip(axes, ["jimmy", "friends"]):
        rgb = load_image(key, max_side=900); g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        fs = detect_faces(g, 1.1, 5); out = rgb.copy()
        for (x, y, w, h) in fs:
            cv2.rectangle(out, (x, y), (x + w, y + h), (213, 94, 0), 3)
        ax.imshow(out); ax.set_title(f"OpenCV Haar cascade: {len(fs)} faces"); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "face_detection")
    return Figure("face_detection",
                  "The trained Viola-Jones cascade shipped with OpenCV applied as a "
                  "sliding-window detector over an image pyramid. Training is slow, but "
                  "the integral image, the small number of selected features and the "
                  "cascade make detection fast enough for early digital cameras and "
                  "phones. Faces that are small, turned away or partly hidden are missed.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_haar_features(ctx), fig_integral_image(ctx), fig_feature_on_face(ctx), fig_adaboost(ctx),
            fig_cascade(ctx), fig_face_detection(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("4.5 Viola-Jones")
    tab1, tab2 = st.tabs(["Face detection", "AdaBoost"])
    with tab1:
        key = st.selectbox("image", ["jimmy", "friends", "faces_crumb", "crowd"])
        rgb = load_image(key, max_side=900); g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sc = st.slider("pyramid scale factor", 1.02, 1.5, 1.1, 0.01); nb = st.slider("minimum neighbours", 1, 10, 5)
        fs = detect_faces(g, sc, nb); out = rgb.copy()
        for (x, y, w, h) in fs:
            cv2.rectangle(out, (x, y), (x + w, y + h), (213, 94, 0), 3)
        st.image(out, caption=f"{len(fs)} faces", width="stretch")
    with tab2:
        n = st.slider("number of weak classifiers", 1, 100, 5)
        X, y = moons(300, 0.25, 1)
        clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=n, random_state=0).fit(X, y)
        xx, yy, Z = decision_regions(clf, X)
        fig, ax = plt.subplots(figsize=(6, 4.5)); ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=[C["sky"], "#f6c8b0"], alpha=0.5)
        for c in (0, 1):
            ax.plot(X[y == c, 0], X[y == c, 1], "o", color=SERIES[c], ms=3)
        ax.set_title(f"training accuracy {clf.score(X, y):.2f}"); ax.set_xticks([]); ax.set_yticks([]); st.pyplot(fig)
