"""Lesson 5.1 — Introduction to facial recognition.

Figures
    identification_verification  a probe against a gallery (identification) versus a 1:1 check (verification)
    fr_challenges                many identities, few examples per identity, high inter-personal similarity
    intra_personal               intrinsic (expression, ageing) and extrinsic (pose, glasses, illumination) variation
    constrained_dataset          the Olivetti gallery: centred frontal views under constant illumination
    unconstrained                faces found in the wild at many scales and poses (Haar cascade on a group photo)
    fr_pipeline                  detect, align, represent, recognise
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from .._common import BuildContext, Figure, C, save_mpl, load_image
from ._faces_common import olivetti, illumination_gradient, as_image

FIGURES = ["identification_verification", "fr_challenges", "intra_personal",
           "constrained_dataset", "unconstrained", "fr_pipeline"]


def box(ax, xy, w, h, text, fc="#e8f1f8", fs=8.5):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=C["grey"], lw=1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=13, color=C["ink"], lw=1.2))


def show_face(ax, vec, title=None, fs=8):
    ax.imshow(as_image(vec), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fs)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_identification_verification(ctx: BuildContext) -> Figure:
    images, X, y = olivetti()
    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(2, 7, height_ratios=[1, 1], hspace=0.35)
    # identification: one probe against a gallery of identities
    ax = fig.add_subplot(gs[0, 0]); show_face(ax, X[y == 3][7], "probe\n(unknown)")
    for j, ident in enumerate([0, 1, 3, 5, 7]):
        ax = fig.add_subplot(gs[0, j + 2]); show_face(ax, X[y == ident][0], f"gallery id {ident}" + ("  ✓ match" if ident == 3 else ""))
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.annotate("", xy=(1.0, 0.5), xytext=(0.0, 0.5), arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.5))
    ax.text(0.5, 0.62, "compare\n1 : N", ha="center", fontsize=8)
    # verification: probe against a single claimed identity
    ax = fig.add_subplot(gs[1, 0]); show_face(ax, X[y == 12][4], "probe\nclaims id 12")
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    ax.annotate("", xy=(1.0, 0.5), xytext=(0.0, 0.5), arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=1.5))
    ax.text(0.5, 0.62, "compare\n1 : 1", ha="center", fontsize=8)
    ax = fig.add_subplot(gs[1, 2]); show_face(ax, X[y == 12][0], "stored id 12\n→ accept")
    ax = fig.add_subplot(gs[1, 3]); show_face(ax, X[y == 30][2], "impostor probe")
    ax = fig.add_subplot(gs[1, 4]); ax.axis("off")
    ax.annotate("", xy=(1.0, 0.5), xytext=(0.0, 0.5), arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=1.5))
    ax = fig.add_subplot(gs[1, 5]); show_face(ax, X[y == 12][0], "stored id 12\n→ reject")
    ax = fig.add_subplot(gs[1, 6]); ax.axis("off")
    ax.text(0, 0.95, "identification (top)", fontsize=9, weight="bold", va="top")
    ax.text(0, 0.80, "label a probe by searching a gallery.\nClosed set: the identity is always there.\nOpen set: it may not be, so the system\nmust be able to answer \"unknown\".", fontsize=7.5, va="top")
    ax.text(0, 0.42, "verification (bottom)", fontsize=9, weight="bold", va="top")
    ax.text(0, 0.27, "check a claimed identity, 1 : 1.\nOnly meaningful as an open-set task,\nsince impostors must be rejected.", fontsize=7.5, va="top")
    png = save_mpl(fig, ctx, "identification_verification")
    return Figure("identification_verification",
                  "The two facial-recognition tasks. In identification a probe image of an "
                  "unknown person is compared against a gallery of many identities and a "
                  "label is assigned. In verification the purported identity of a person "
                  "is checked against stored examples of who they claim to be, a 1:1 "
                  "comparison that must accept the genuine user and reject impostors.",
                  png=png)


def fig_fr_challenges(ctx: BuildContext) -> Figure:
    images, X, y = olivetti()
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    same = y[:, None] == y[None, :]
    # the two closest images of *different* people: small inter-personal difference
    Dd = np.where(same, np.inf, D)
    i, j = np.unravel_index(np.argmin(Dd), Dd.shape)
    d_diff = float(Dd[i, j])
    # the two most distant images of one of those same people: large intra-personal difference
    worst = int(y[i])
    m = y == worst
    idx = np.flatnonzero(m)
    sub = D[np.ix_(idx, idx)]
    p, q = np.unravel_index(np.argmax(sub), sub.shape)
    p, q = int(idx[p]), int(idx[q])
    d_same = float(D[p, q])
    fig = plt.figure(figsize=(12.5, 2.9))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 0.3, 1, 1, 2.1],
                          height_ratios=[1, 0.22], hspace=0.05)
    ax = fig.add_subplot(gs[0, 0]); show_face(ax, X[i], f"identity {y[i]}")
    ax = fig.add_subplot(gs[0, 1]); show_face(ax, X[j], f"identity {y[j]}")
    ax = fig.add_subplot(gs[0, 3]); show_face(ax, X[p], f"identity {worst}")
    ax = fig.add_subplot(gs[0, 4]); show_face(ax, X[q], f"identity {worst}")
    ax = fig.add_subplot(gs[1, 0:2]); ax.axis("off")
    ax.text(0.5, 1.0, f"different people, distance {d_diff:.1f}", ha="center", va="top", fontsize=8.5, color=C["red"])
    ax = fig.add_subplot(gs[1, 3:5]); ax.axis("off")
    ax.text(0.5, 1.0, f"same person, distance {d_same:.1f}", ha="center", va="top", fontsize=8.5, color=C["blue"])
    ax = fig.add_subplot(gs[:, 5]); ax.axis("off")
    ax.text(0, 1.0, "why facial recognition is harder than object detection", fontsize=9.5, weight="bold", va="top")
    ax.text(0, 0.86, "•  very many classes: one per identity\n"
                     "•  very few examples per identity, sometimes one\n"
                     "•  in open-set use, many unknown identities to reject\n"
                     "•  small inter-personal differences: different people can look alike\n"
                     "•  large intra-personal differences: one person can look very\n    different between images", fontsize=8, va="top")
    ax.text(0, 0.24, f"Two images of different people are closer in raw pixels (distance\n{d_diff:.1f}) than two images of identity {worst} are to each other (distance\n{d_same:.1f}). A raw-pixel nearest-neighbour classifier confuses them.", fontsize=8, va="top", color=C["grey"])
    png = save_mpl(fig, ctx, "fr_challenges")
    return Figure("fr_challenges",
                  "Facial recognition is a sub-task of object detection, but a harder one. "
                  "There are many identities and few examples of each, and faces are "
                  "separated only by small inter-personal differences while images of one "
                  "person vary a great deal. In this dataset two images of different people "
                  "are closer in raw pixel distance than two images of the same person, "
                  "which is exactly why a good representation is needed.", png=png)


def fig_intra_personal(ctx: BuildContext) -> Figure:
    images, X, y = olivetti()
    ident = 7
    faces = X[y == ident]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    # intrinsic: expression changes within the dataset
    for j in range(4):
        show_face(axes[0, j], faces[j], f"expression / pose {j + 1}")
    axes[0, 4].axis("off")
    axes[0, 4].text(0, 0.9, "intrinsic factors", fontsize=10, weight="bold", va="top")
    axes[0, 4].text(0, 0.72, "due to the person:\nexpression, ageing,\nfacial hair", fontsize=8.5, va="top")
    # extrinsic: synthetic illumination and occlusion applied to one face
    show_face(axes[1, 0], faces[0], "reference")
    show_face(axes[1, 1], illumination_gradient(faces[0][None], 0.9)[0], "illumination from the left")
    show_face(axes[1, 2], illumination_gradient(faces[0][None], 0.9, "right")[0], "illumination from the right")
    occ = as_image(faces[0]).copy(); occ[18:30, 8:56] = 0.15
    axes[1, 3].imshow(occ, cmap="gray", vmin=0, vmax=1); axes[1, 3].set_xticks([]); axes[1, 3].set_yticks([]); axes[1, 3].set_title("occlusion (dark glasses)", fontsize=8)
    axes[1, 4].axis("off")
    axes[1, 4].text(0, 0.9, "extrinsic factors", fontsize=10, weight="bold", va="top")
    axes[1, 4].text(0, 0.72, "due to the imaging:\npose and camera angle,\nglasses and hats,\nillumination", fontsize=8.5, va="top")
    png = save_mpl(fig, ctx, "intra_personal")
    return Figure("intra_personal",
                  "Intra-personal variation: the ways images of one person differ. "
                  "Intrinsic factors belong to the person, such as expression, ageing and "
                  "facial hair (top row, four images of one identity). Extrinsic factors "
                  "come from the imaging conditions: pose, paraphernalia such as glasses, "
                  "and illumination. The illumination ramps and the occlusion in the "
                  "bottom row are applied synthetically to one image so that the effect "
                  "can be seen in isolation.", png=png)


def fig_constrained_dataset(ctx: BuildContext) -> Figure:
    images, X, y = olivetti()
    fig, axes = plt.subplots(4, 10, figsize=(12, 5.4))
    for r in range(4):
        for c in range(10):
            show_face(axes[r, c], X[y == r][c])
        axes[r, 0].set_ylabel(f"id {r}", fontsize=8)
    fig.suptitle("Olivetti dataset: 4 of the 40 identities, 10 images each", fontsize=10)
    png = save_mpl(fig, ctx, "constrained_dataset")
    return Figure("constrained_dataset",
                  "A constrained dataset. The Olivetti faces are centred frontal views at "
                  "a fixed size, with limited pose and expression variation and nearly "
                  "constant illumination, and there are only 40 identities with ten "
                  "images each. Constraining the images this way let the early methods "
                  "of the next lessons succeed; their performance is far less robust when "
                  "illumination or pose varies. Many real applications, such as access "
                  "control, are themselves constrained.", png=png)


def fig_unconstrained(ctx: BuildContext) -> Figure:
    rgb = load_image("jimmy", max_side=900)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cas.detectMultiScale(g, 1.08, 5)
    out = rgb.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (213, 94, 0), 4)
    sizes = sorted(int(w) for (_, _, w, _) in faces)
    fig = plt.figure(figsize=(11, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.55], height_ratios=[1.0, 0.9], wspace=0.1)
    ax = fig.add_subplot(gs[:, 0]); ax.imshow(out)
    ax.set_title(f"faces must first be found: {len(faces)} detections, "
                 f"{sizes[0]} to {sizes[-1]} px wide", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    crops = sorted(faces, key=lambda r: -r[2])[:3]
    sub = gs[0, 1].subgridspec(1, 3, wspace=0.1)
    for k, (x, y, w, h) in enumerate(crops):
        a = fig.add_subplot(sub[0, k])
        a.imshow(cv2.resize(g[y:y + h, x:x + w], (64, 64)), cmap="gray")
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{w} px crop → 64 × 64" if k == 1 else f"{w} px crop", fontsize=8)
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    ax.text(0, 1.0, "unconstrained recognition (\"in the wild\")", fontsize=9.5, weight="bold", va="top")
    ax.text(0, 0.82, "•  faces must be detected first\n•  wide variation in scale and pose\n•  varying illumination and expression\n•  occlusion and motion blur\n•  faces are rarely centred or aligned", fontsize=8, va="top")
    ax.text(0.53, 0.82, "Compare these ragged crops with the\nOlivetti gallery above: they differ in\nscale, tilt, centring and lighting.\nAlignment is the step that turns one\ninto the other.", fontsize=8, va="top", color=C["grey"])
    png = save_mpl(fig, ctx, "unconstrained")
    return Figure("unconstrained",
                  "Unconstrained facial recognition. Faces must first be located in the "
                  "image, and they appear at very different scales, poses and lighting. "
                  "The crops taken straight from the detector vary in size, tilt and "
                  "centring, unlike the neatly aligned images of a constrained dataset.",
                  png=png)


def fig_fr_pipeline(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(13, 2.5)); ax.set_xlim(0, 13); ax.set_ylim(0, 2.5); ax.axis("off")
    steps = [("1. detect\nlocalise every face\nin the image", "#e8f1f8"),
             ("2. align\nfind facial landmarks\nand warp to a centred view", "#fdf1e3"),
             ("3. represent\na discriminative, lower-dimensional\nvector: PCA (eigenfaces),\nLDA (fisherfaces)", "#e6f4ee"),
             ("4. recognise\nnearest neighbour in the\ngallery, or a classifier", "#f3e8f5")]
    x = 0.15
    for k, (text, fc) in enumerate(steps):
        box(ax, (x, 0.8), 2.6, 1.5, text, fc=fc, fs=8)
        if k < len(steps) - 1:
            arrow(ax, (x + 2.68, 1.55), (x + 3.18, 1.55))
        x += 3.2
    ax.text(0.15, 0.35, "Lessons 5.2 to 5.5 are about step 3: finding a representation in which images of "
                        "one person are close together and images of different people are far apart.",
            fontsize=8.5, color=C["grey"])
    png = save_mpl(fig, ctx, "fr_pipeline")
    return Figure("fr_pipeline",
                  "The stages of a modern facial-recognition system. Faces are localised, "
                  "aligned to a canonical centred view using facial landmarks, converted "
                  "into a discriminative representation, and finally matched, often by a "
                  "simple nearest-neighbour search. The rest of this section concerns the "
                  "representation stage.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_identification_verification(ctx), fig_fr_challenges(ctx), fig_intra_personal(ctx),
            fig_constrained_dataset(ctx), fig_unconstrained(ctx), fig_fr_pipeline(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("5.1 Facial recognition: the data")
    images, X, y = olivetti()
    st.markdown("The Olivetti gallery: 40 identities, 10 images each, 64 × 64 pixels.")
    ident = st.slider("identity", 0, 39, 7)
    faces = X[y == ident]
    st.image([as_image(f) for f in faces], width=90, caption=[f"image {i}" for i in range(10)], clamp=True)
    st.subheader("Distances in raw pixel space")
    c1, c2 = st.columns(2)
    other = c2.slider("compare with identity", 0, 39, (ident + 1) % 40)
    k = c1.slider("image of this identity", 0, 9, 0)
    d_same = np.linalg.norm(faces[k] - faces[(k + 5) % 10])
    d_diff = np.linalg.norm(faces[k] - X[y == other][0])
    st.write(f"distance to another image of identity {ident}: **{d_same:.1f}**")
    st.write(f"distance to identity {other}: **{d_diff:.1f}**")
    st.caption("When the second number is the smaller of the two, a nearest-neighbour "
               "classifier on raw pixels would get this pair wrong. A better "
               "representation is what the following lessons build.")
