"""Lesson 1.5 — Binary images and morphology.

Figures
    binary_threshold     greyscale image thresholded to a binary mask
    structuring_elements cross, diamond, disc and square structuring elements
    fit_hit_miss         fit, hit and miss of a structuring element on a small binary image
    erosion_dilation     erosion and dilation on a 9x9 grid and on shapes (interactive)
    opening_closing      opening removes small objects, closing fills gaps
    distance_transform   L1 and L2 distance transforms of a binary image
    connected_components row-scan labelling step by step, then labelling a photo mask
    grey_morphology      greyscale erosion, dilation, opening and closing
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import ndimage

from .._common import (BuildContext, Figure, C, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, load_image, sync_matrix)
from .._vendor import _morph_utils as mu

FIGURES = ["binary_threshold", "structuring_elements", "fit_hit_miss", "erosion_dilation",
           "opening_closing", "distance_transform", "connected_components", "grey_morphology"]

BIN_CMAP = ListedColormap(["#1f2933", "#ffffff"])


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def se_disc(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def teaching_blob(seed: int = 4, size: int = 9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return mu.make_blob(rng, size=size)


def classify_position(img, se, r, c) -> str:
    """'fit', 'hit' or 'miss' for the SE centred at (r, c)."""
    h, w = img.shape
    ph, pw = se.shape[0] // 2, se.shape[1] // 2
    ones = []
    for i in range(se.shape[0]):
        for j in range(se.shape[1]):
            if se[i, j]:
                rr, cc = r + i - ph, c + j - pw
                ones.append(bool(0 <= rr < h and 0 <= cc < w and img[rr, cc]))
    if all(ones):
        return "fit"
    if any(ones):
        return "hit"
    return "miss"


def shapes_image(n: int = 220, seed: int = 0) -> np.ndarray:
    """Synthetic binary image: large discs, small specks, and a bar with gaps."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.zeros((n, n), np.uint8)
    for cx, cy, r in [(60, 60, 34), (150, 70, 28), (100, 150, 38)]:
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2] = 1
    for _ in range(40):
        cx, cy = rng.integers(5, n - 5, 2)
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < rng.integers(1, 3) ** 2] = 1
    img[190:205, 20:200] = 1
    for gx in [60, 110, 160]:
        img[190:205, gx:gx + 4] = 0
    img[100:106, 150:160] = 0          # a small hole in the large disc
    return img


def holed_shapes_image(n: int = 220, seed: int = 0, n_pepper: int = 130) -> np.ndarray:
    """The same shapes, but damaged by holes instead of specks.

    Closing fills holes, so demonstrating it needs an image that has some. This
    drops the background specks, which are the opening story, and instead knocks
    random black pixels out of the objects plus one hole large enough to see.
    Every hole is narrower than the disc structuring element used in the figure,
    so closing can fill all of them.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.zeros((n, n), np.uint8)
    # the lower disc sits higher than in shapes_image so that dilating by the
    # structuring element does not bridge it to the bar, which would leave the
    # two merged after the erosion and muddle the point of the figure
    for cx, cy, r in [(60, 60, 34), (150, 70, 28), (100, 136, 38)]:
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2] = 1
    img[190:205, 20:200] = 1
    for gx in [60, 110, 160]:
        img[190:205, gx:gx + 4] = 0    # gaps in the bar
    # keep the pepper away from the edges, so every hole is enclosed by
    # foreground rather than becoming a notch in an object's outline
    interior = cv2.erode(img, se_disc(3))
    inside = np.flatnonzero(interior.ravel())
    for i in rng.choice(inside, size=min(n_pepper, len(inside)), replace=False):
        r, c = divmod(int(i), n)
        img[r:r + rng.integers(1, 3), c:c + rng.integers(1, 3)] = 0
    img[132:140, 96:104] = 0           # one hole big enough to see, 8 x 8
    return img


def label_steps(img: np.ndarray):
    """Two-pass connected components (8-neighbours) returning the first-pass labels,
    the equivalences recorded, and the final relabelled image."""
    h, w = img.shape
    labels = np.zeros((h, w), int)
    parent = {}
    nxt = 1
    for r in range(h):
        for c in range(w):
            if not img[r, c]:
                continue
            nb = []
            for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and labels[rr, cc]:
                    nb.append(labels[rr, cc])
            if not nb:
                labels[r, c] = nxt; parent[nxt] = nxt; nxt += 1
            else:
                m = min(nb); labels[r, c] = m
                for l in nb:
                    ra, rb = _find(parent, l), _find(parent, m)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    final = labels.copy()
    roots = {}
    for l in range(1, nxt):
        roots[l] = _find(parent, l)
    uniq = {r: i + 1 for i, r in enumerate(sorted(set(roots.values())))}
    for l, r in roots.items():
        final[labels == l] = uniq[r]
    equiv = sorted({(l, roots[l]) for l in roots if roots[l] != l})
    return labels, equiv, final


def _find(parent, x):
    while parent[x] != x:
        x = parent[x]
    return x


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_binary_threshold(ctx: BuildContext) -> Figure:
    g = load_image("sheep", gray=True, max_side=640)
    t = 150
    b = (g > t).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[1.2, 1, 1.2]))
    axes[0].imshow(g, vmin=0, vmax=255); axes[0].set_title("(a) greyscale image")
    axes[1].hist(g.ravel(), bins=64, color=C["grey"]); axes[1].axvline(t, color=C["red"])
    axes[1].set_title(f"(b) histogram, threshold t = {t}"); axes[1].set_xlabel("intensity")
    axes[2].imshow(b, cmap=BIN_CMAP); axes[2].set_title("(c) binary image: 1 where I > t")
    for ax in (axes[0], axes[2]):
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "binary_threshold")
    return Figure("binary_threshold",
                  "A binary image is made from a greyscale one by thresholding: pixels above "
                  "the threshold become 1 (white), the rest 0 (black). OpenCV stores the "
                  "white pixels as 255. The result is a mask that separates bright "
                  "foreground objects, here the sheep, from the background.", png=png)


def fig_structuring_elements(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(0)
    ses = [("cross 3×3", mu.make_se(rng, 3, "cross")[0]), ("diamond 5×5", mu.make_se(rng, 5, "diamond")[0]),
           ("disc 7×7", se_disc(3)), ("square 5×5", np.ones((5, 5), int))]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for ax, (t, se) in zip(axes, ses):
        sync_matrix(ax, se, fmt="{:.0f}", cmap="Oranges", title=t)
        c = se.shape[0] // 2
        ax.add_patch(plt.Rectangle((c - 0.5, c - 0.5), 1, 1, fill=False, ec=C["blue"], lw=2))
    png = save_mpl(fig, ctx, "structuring_elements")
    return Figure("structuring_elements",
                  "Structuring elements: small binary arrays that define the neighbourhood "
                  "over which a morphological operation is applied. The origin (blue) is "
                  "placed on the pixel being processed. Larger elements act more strongly on "
                  "object boundaries.", png=png)


def fig_fit_hit_miss(ctx: BuildContext) -> Figure:
    img = teaching_blob(4)
    se = np.ones((3, 3), int)
    ero = mu.erode(img, se)
    dil = mu.dilate(img, se)
    # one example of each kind, with the whole element inside the grid;
    # prefer a miss far from the blob and a hit on its boundary
    cands = {"fit": [], "hit": [], "miss": []}
    for r in range(1, 8):
        for c in range(1, 8):
            cands[classify_position(img, se, r, c)].append((r, c))
    examples = {"fit": cands["fit"][len(cands["fit"]) // 2] if cands["fit"] else (4, 4),
                "hit": cands["hit"][0] if cands["hit"] else (4, 4),
                "miss": max(cands["miss"], key=lambda p: p[0] + p[1]) if cands["miss"] else (7, 7)}
    colours = {"fit": C["green"], "hit": C["blue"], "miss": C["red"]}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    sync_matrix(axes[0], se, cmap="Oranges", fmt="{:.0f}", title="3 × 3 structuring element")
    for ax, (t, im) in zip(axes[1:], [("image with fit / hit / miss", img), ("erosion: 1 only where fit", ero),
                                       ("dilation: 1 where fit or hit", dil)]):
        ax.imshow(im, cmap=BIN_CMAP, vmin=0, vmax=1)
        ax.set_xticks(np.arange(-0.5, 9, 1), minor=True); ax.set_yticks(np.arange(-0.5, 9, 1), minor=True)
        ax.grid(which="minor", color=C["light"], lw=0.6); ax.tick_params(which="both", length=0)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_title(t)
        for k, (r, c) in examples.items():
            ax.add_patch(plt.Rectangle((c - 1.5, r - 1.5), 3, 3, fill=False, ec=colours[k], lw=2.2))
            if t.startswith("image"):
                ax.text(c, r, k, color=colours[k], ha="center", va="center", fontsize=8, weight="bold",
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    png = save_mpl(fig, ctx, "fit_hit_miss")
    return Figure("fit_hit_miss",
                  "Fit, hit and miss. With the structuring element centred on a pixel: a "
                  "fit (green) has every 1 of the element over a 1 of the image; a hit "
                  "(blue) has some overlap; a miss (red) has none. Erosion outputs 1 only "
                  "for a fit, so objects shrink; dilation outputs 1 for a fit or a hit, so "
                  "objects grow.", png=png)


def fig_erosion_dilation(ctx: BuildContext) -> Figure:
    img = shapes_image()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, (t, im) in zip(axes, [("original", img), ("erosion (disc r = 5)", cv2.erode(img, se_disc(5))),
                                  ("dilation (disc r = 5)", cv2.dilate(img, se_disc(5)))]):
        ax.imshow(im, cmap=BIN_CMAP, vmin=0, vmax=1); ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "erosion_dilation")

    radii = [1, 2, 3, 5, 8, 12]
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("erosion", "dilation"))
    frames = []
    for r in radii:
        e = cv2.erode(img, se_disc(r)) * 255; d = cv2.dilate(img, se_disc(r)) * 255
        frames.append(go.Frame(name=str(r), data=[go.Image(source=to_data_uri(e)), go.Image(source=to_data_uri(d))],
                               traces=[0, 1]))
    fig2.add_trace(frames[3].data[0], row=1, col=1); fig2.add_trace(frames[3].data[1], row=1, col=2)
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(radii, "disc radius = "), title="Structuring element size")
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    return Figure("erosion_dilation",
                  "Erosion removes a layer of pixels from every boundary: thin specks vanish, "
                  "the hole widens and the gaps in the bar grow. Dilation adds a layer: "
                  "specks grow, the hole and the gaps close. Move the slider to change the "
                  "structuring element size.",
                  png=png, html=plotly_html(fig2, height=420), kind="plotly")


def fig_opening_closing(ctx: BuildContext) -> Figure:
    """The two operations need different damage to show what they repair:
    opening removes specks added to the background, closing fills holes
    knocked out of the objects."""
    se = se_disc(5)
    speckled = shapes_image()                  # white specks on the background
    holed = holed_shapes_image()               # black pixel noise inside the objects
    opened = cv2.morphologyEx(speckled, cv2.MORPH_OPEN, se)
    closed = cv2.morphologyEx(holed, cv2.MORPH_CLOSE, se)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))
    row1 = [("(a) specks on the background", speckled),
            ("(b) erosion", cv2.erode(speckled, se)),
            ("(c) then dilation = opening", opened)]
    row2 = [("(d) holes inside the objects", holed),
            ("(e) dilation", cv2.dilate(holed, se)),
            ("(f) then erosion = closing", closed)]
    for ax, (t, im) in zip(axes.ravel(), row1 + row2):
        ax.imshow(im, cmap=BIN_CMAP, vmin=0, vmax=1)
        ax.set_title(t, fontsize=9.5); ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("opening", fontsize=11, weight="bold")
    axes[1, 0].set_ylabel("closing", fontsize=11, weight="bold")
    png = save_mpl(fig, ctx, "opening_closing")
    return Figure("opening_closing",
                  "Opening and closing repair opposite kinds of damage, so each row "
                  "starts from the image its operation is meant to clean up. Opening "
                  "(top) is erosion followed by dilation: the erosion deletes the specks "
                  "scattered over the background and the dilation returns the discs and "
                  "the bar to their original size, so the specks never come back. Closing "
                  "(bottom) is dilation followed by erosion, applied to objects riddled "
                  "with black pixel noise: the dilation fills the holes and the gaps in "
                  "the bar, and the erosion shrinks the result back, leaving the objects "
                  "solid and the right size. Both use the same disc structuring element, "
                  "and a hole or speck is only removed if it is smaller than that "
                  "element.", png=png)


def fig_distance_transform(ctx: BuildContext) -> Figure:
    n = 160
    yy, xx = np.mgrid[0:n, 0:n]
    img = (((xx - 60) ** 2 + (yy - 80) ** 2 < 45 ** 2) | ((xx - 110) ** 2 + (yy - 80) ** 2 < 40 ** 2)).astype(np.uint8)
    l1 = ndimage.distance_transform_cdt(img, metric="taxicab")
    l2 = ndimage.distance_transform_edt(img)
    small = np.zeros((7, 7), np.uint8); small[1:6, 1:6] = 1; small[1, 1] = 0; small[5, 5] = 0
    s1 = ndimage.distance_transform_cdt(small, metric="taxicab")
    s2 = ndimage.distance_transform_edt(small)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    sync_matrix(axes[0, 0], small, cmap="Greys", fmt="{:.0f}", title="binary image")
    sync_matrix(axes[0, 1], s1, cmap="viridis", fmt="{:.0f}", title="L1 (Manhattan) distance", text_color="white")
    sync_matrix(axes[0, 2], s2, cmap="viridis", fmt="{:.2f}", title="L2 (Euclidean) distance", fontsize=8,
                text_color="white")
    axes[1, 0].imshow(img, cmap=BIN_CMAP, vmin=0, vmax=1); axes[1, 0].set_title("two overlapping discs")
    axes[1, 1].imshow(l1, cmap="viridis"); axes[1, 1].set_title("L1 distance transform")
    axes[1, 2].imshow(l2, cmap="viridis"); axes[1, 2].set_title("L2 distance transform")
    for ax in axes[1]:
        ax.set_xticks([]); ax.set_yticks([])
    for ax, d in zip(axes[1, 1:], [l1, l2]):
        r, c = np.unravel_index(np.argmax(d), d.shape)
        ax.plot(c, r, "r+", ms=12, mew=2)
    png = save_mpl(fig, ctx, "distance_transform")
    return Figure("distance_transform",
                  "The distance transform gives every foreground pixel its distance to the "
                  "nearest background pixel. Top: a small example with the L1 (Manhattan) "
                  "and L2 (Euclidean) metrics. Bottom: on overlapping discs the brightest "
                  "values mark the disc centres (red crosses), which is why the transform "
                  "is used for centre finding and skeletons.", png=png)


def fig_connected_components(ctx: BuildContext) -> Figure:
    img = np.array([[0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 1, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0]])
    first, equiv, final = label_steps(img)
    g = load_image("sheep", gray=True, max_side=640)
    mask = (g > 175).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se_disc(3))
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.6]))
    sync_matrix(axes[0], img, cmap="Greys", fmt="{:.0f}", title="(a) binary image")
    lab_str = lambda v: "" if v == 0 else chr(64 + int(v))  # noqa: E731
    for ax, (t, L) in zip(axes[1:3], [("(b) first pass: provisional labels", first), ("(c) after merging equivalences", final)]):
        ax.imshow(np.where(L > 0, L, np.nan), cmap="tab10", vmin=1, vmax=10)
        for i in range(7):
            for j in range(7):
                if L[i, j]:
                    ax.text(j, i, lab_str(L[i, j]), ha="center", va="center", fontsize=9, weight="bold")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t)
    eq_txt = ", ".join(f"{lab_str(a)}≡{lab_str(b)}" for a, b in equiv) or "none"
    axes[1].set_xlabel(f"equivalences noted: {eq_txt}", fontsize=9)
    axes[3].imshow(np.where(lab > 0, lab, np.nan), cmap="tab20", interpolation="nearest")
    axes[3].set_title(f"(d) sheep mask: {n} components"); axes[3].set_xticks([]); axes[3].set_yticks([])
    png = save_mpl(fig, ctx, "connected_components")
    return Figure("connected_components",
                  "Connected components. (a) A binary image is scanned row by row. (b) Each "
                  "foreground pixel takes the label of an already-labelled neighbour above "
                  "or to the left (8-neighbours), or a new label if there is none; where two "
                  "different labels meet, one is chosen and the equivalence is noted. (c) A "
                  "second pass merges equivalent labels. (d) The same algorithm labels every "
                  "separate blob in the thresholded sheep mask.", png=png)


def fig_grey_morphology(ctx: BuildContext) -> Figure:
    g = load_image("cat", gray=True, max_side=480)
    se = se_disc(3)
    panels = [("original", g), ("greyscale erosion (min)", cv2.erode(g, se)),
              ("greyscale dilation (max)", cv2.dilate(g, se)),
              ("opening (erode, dilate)", cv2.morphologyEx(g, cv2.MORPH_OPEN, se)),
              ("closing (dilate, erode)", cv2.morphologyEx(g, cv2.MORPH_CLOSE, se))]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    for ax, (t, im) in zip(axes, panels):
        ax.imshow(im, vmin=0, vmax=255); ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "grey_morphology")
    return Figure("grey_morphology",
                  "Greyscale morphology replaces the fit/hit rules with minimum and maximum "
                  "over the structuring element. Erosion darkens pixels surrounded by darker "
                  "neighbours; dilation brightens pixels surrounded by brighter ones. Opening "
                  "removes small bright spots, closing removes small dark spots, and both "
                  "leave the image closer to the original than erosion or dilation alone.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_binary_threshold(ctx), fig_structuring_elements(ctx), fig_fit_hit_miss(ctx),
            fig_erosion_dilation(ctx), fig_opening_closing(ctx), fig_distance_transform(ctx),
            fig_connected_components(ctx), fig_grey_morphology(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("1.5 Binary images and morphology")
    tab1, tab2, tab3 = st.tabs(["Fit / hit / miss on a grid", "Operations on shapes", "Threshold and label"])
    with tab1:
        seed = st.number_input("blob seed", 0, 999, 4)
        shape = st.selectbox("structuring element", ["square 3×3", "cross 3×3", "diamond 5×5", "oval 5×5"])
        rng = np.random.default_rng(0)
        se = {"square 3×3": np.ones((3, 3), int), "cross 3×3": mu.make_se(rng, 3, "cross")[0],
              "diamond 5×5": mu.make_se(rng, 5, "diamond")[0], "oval 5×5": mu.make_se(rng, 5, "oval")[0]}[shape]
        img = teaching_blob(int(seed))
        c1, c2 = st.columns(2)
        r = c1.slider("row", 0, 8, 4); c = c2.slider("column", 0, 8, 4)
        kind = classify_position(img, se, r, c)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
        sync_matrix(axes[0], se, cmap="Oranges", fmt="{:.0f}", title="structuring element")
        for ax, (t, im) in zip(axes[1:], [(f"image: {kind} at ({r},{c})", img), ("erosion", mu.erode(img, se)),
                                           ("dilation", mu.dilate(img, se))]):
            ax.imshow(im, cmap=BIN_CMAP, vmin=0, vmax=1); ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
            k = se.shape[0]
            ax.add_patch(plt.Rectangle((c - k / 2, r - k / 2), k, k, fill=False,
                                       ec={"fit": C["green"], "hit": C["blue"], "miss": C["red"]}[kind], lw=2))
        st.pyplot(fig)
    with tab2:
        c1, c2, c3 = st.columns(3)
        src = c1.radio("input image", ["specks on the background", "holes inside the objects"])
        op = c2.selectbox("operation", ["erosion", "dilation", "opening", "closing"])
        rad = c3.slider("disc radius", 1, 15, 5)
        img = holed_shapes_image() if src.startswith("holes") else shapes_image()
        se = se_disc(rad)
        out = {"erosion": cv2.erode, "dilation": cv2.dilate,
               "opening": lambda i, s: cv2.morphologyEx(i, cv2.MORPH_OPEN, s),
               "closing": lambda i, s: cv2.morphologyEx(i, cv2.MORPH_CLOSE, s)}[op](img, se)
        a, b = st.columns(2)
        a.image(img * 255, caption="input", width="stretch")
        b.image(out * 255, caption=op, width="stretch")
        st.caption("Opening is best seen on the speckled image and closing on the holed "
                   "one, because each repairs the kind of damage the other cannot. "
                   "Raising the disc radius removes larger specks and fills larger holes, "
                   "and past a point it starts destroying the objects themselves.")
    with tab3:
        g = load_image("sheep", gray=True, max_side=640)
        t = st.slider("threshold", 0, 255, 150)
        rad = st.slider("opening radius (0 = none)", 0, 6, 2)
        mask = (g > t).astype(np.uint8)
        if rad:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se_disc(rad))
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        c1, c2 = st.columns(2)
        c1.image(mask * 255, caption="mask", width="stretch")
        fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(np.where(lab > 0, lab, np.nan), cmap="tab20"); ax.axis("off")
        ax.set_title(f"{n} connected components"); c2.pyplot(fig)
