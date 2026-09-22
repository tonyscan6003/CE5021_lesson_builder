"""Lesson 4.4 — Template matching.

Figures
    template_search     a template slid over the image; sum of squared differences map and its minimum
    correlation_vs_ncc  plain cross-correlation gives false maxima in bright regions; NCC finds the eye
    shape_matching      edge outlines of a template and a scene: matching by shape rather than pixels
    shortest_distance   the shortest distance between two shapes is the same for very different overlaps
    chamfer_matching    edges -> inverted -> distance transform -> correlate with a binary ring template
    hausdorff           directed Hausdorff distances between two point sets, step by step
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import cdist

from .._common import BuildContext, Figure, C, save_mpl, load_image, norm01, sync_matrix
from .._vendor import hausdorff as hd_gen

FIGURES = ["template_search", "correlation_vs_ncc", "shape_matching", "shortest_distance",
           "chamfer_matching", "hausdorff"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def face_and_eye(max_side=640):
    """Portrait photo and a template cut around one eye found by the Haar cascade."""
    rgb = load_image("faces_crumb", max_side=max_side)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    fs = face.detectMultiScale(g, 1.1, 5)
    if len(fs):
        x, y, w, h = max(fs, key=lambda r: r[2] * r[3])
        es = eyes.detectMultiScale(g[y:y + h // 2, x:x + w], 1.1, 5)
        if len(es):
            ex, ey, ew, eh = sorted(es, key=lambda r: r[0])[0]
            return g, g[y + ey:y + ey + eh, x + ex:x + ex + ew]
    h, w = g.shape
    return g, g[int(0.35 * h):int(0.45 * h), int(0.35 * w):int(0.48 * w)]


def owl_and_head():
    g = load_image("owl", gray=True, max_side=520)
    h, w = g.shape
    return g, g[int(0.18 * h):int(0.42 * h), int(0.38 * w):int(0.62 * w)]


def match_map(img, tpl, method):
    return cv2.matchTemplate(img.astype(np.float32), tpl.astype(np.float32), method)


def ring_template(r_out, r_in):
    n = 2 * r_out + 3
    yy, xx = np.mgrid[0:n, 0:n] - n // 2
    d = np.hypot(xx, yy)
    return ((d <= r_out) & (d >= r_in)).astype(np.float32)


def directed_hausdorff(A, B):
    D = cdist(A, B)
    mins = D.min(axis=1)
    return mins.max(), mins, D.argmin(axis=1)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_template_search(ctx: BuildContext) -> Figure:
    g, tpl = owl_and_head()
    sq = match_map(g, tpl, cv2.TM_SQDIFF)
    ys, xs = np.unravel_index(np.argmin(sq), sq.shape)
    th, tw = tpl.shape
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), gridspec_kw=dict(width_ratios=[0.6, 1.4, 1.4, 1.4]))
    axes[0].imshow(tpl, vmin=0, vmax=255); axes[0].set_title(f"template {tw} × {th}")
    out = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    for (px, py), col in [((40, 60), (120, 120, 120)), ((200, 40), (120, 120, 120)), ((xs, ys), (213, 94, 0))]:
        cv2.rectangle(out, (px, py), (px + tw, py + th), col, 2)
    axes[1].imshow(out); axes[1].set_title("template moved over every position (three shown)")
    axes[2].imshow(np.sqrt(sq), cmap="viridis"); axes[2].plot(xs, ys, "r*", ms=12); axes[2].set_title("Euclidean distance at each position (dark = similar)")
    axes[3].imshow(g, vmin=0, vmax=255); axes[3].add_patch(plt.Rectangle((xs, ys), tw, th, fill=False, ec=C["red"], lw=2)); axes[3].set_title("best match: the minimum distance")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "template_search")
    return Figure("template_search",
                  "Template matching. A template (the owl's head) is moved systematically "
                  "over the image, like a kernel in convolution, and a similarity score is "
                  "computed at every position. With the Euclidean distance between "
                  "template and image pixels the position with the lowest score is the "
                  "best match.", png=png)


def fig_correlation_vs_ncc(ctx: BuildContext) -> Figure:
    g, tpl = face_and_eye()
    cc = match_map(g, tpl, cv2.TM_CCORR)
    ncc = match_map(g, tpl, cv2.TM_CCOEFF_NORMED)
    th, tw = tpl.shape
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), gridspec_kw=dict(width_ratios=[1.2, 0.5, 1.2, 1.2]))
    axes[0].imshow(g, vmin=0, vmax=255); axes[0].set_title("image")
    axes[1].imshow(tpl, vmin=0, vmax=255); axes[1].set_title("eye template")
    for ax, m, name in [(axes[2], cc, "cross-correlation: bright regions score high"), (axes[3], ncc, "normalised cross-correlation: peak on the eye")]:
        ax.imshow(m, cmap="magma"); ys, xs = np.unravel_index(np.argmax(m), m.shape)
        ax.add_patch(plt.Rectangle((xs, ys), tw, th, fill=False, ec=C["sky"], lw=2)); ax.set_title(name, fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "correlation_vs_ncc")
    return Figure("correlation_vs_ncc",
                  "Correlation-based similarity. Plain cross-correlation (the template "
                  "multiplied with the image at each shift and summed, without reversal) "
                  "peaks wherever the image is bright, giving false maxima. Normalised "
                  "cross-correlation subtracts the local mean and divides by the standard "
                  "deviation at every position, so the peak lands on the matching eye. "
                  "The normalisation makes it more expensive to compute.", png=png)


def fig_shape_matching(ctx: BuildContext) -> Figure:
    n = 200
    def shape(angle, cx, cy, s=1.0):
        img = np.zeros((n, n), np.uint8)
        pts = np.array([[-40, -25], [40, -25], [40, 25], [-40, 25]], float) * s
        a = np.deg2rad(angle); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        pts = (pts @ R.T + [cx, cy]).astype(np.int32)
        cv2.fillPoly(img, [pts], 255); cv2.circle(img, (int(cx), int(cy)), 12, 0, -1)
        return img
    tpl = shape(0, 100, 100); scene = shape(12, 110, 95, 1.05)
    scene = cv2.GaussianBlur(scene, (0, 0), 1.2)
    e_t = cv2.Canny(tpl, 50, 150); e_s = cv2.Canny(scene, 50, 150)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    for ax, im, t in zip(axes, [tpl, e_t, scene, e_s], ["template", "template edges", "scene: rotated 12°, scaled 1.05", "scene edges"]):
        ax.imshow(im, cmap="gray"); ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "shape_matching")
    return Figure("shape_matching",
                  "Shape-based matching. Correlation on pixels is not robust to even small "
                  "changes of orientation or appearance: the peak collapses. Matching the "
                  "edge outline of the template against the edge image of the scene "
                  "instead, with a distance between binary shapes as the score, tolerates "
                  "such changes.", png=png)


def _poly(pts, n=80):
    pts = np.asarray(pts, float)
    return np.vstack([np.linspace(pts[i], pts[(i + 1) % len(pts)], n) for i in range(len(pts))])


def _rect(w, h, angle, tx, ty):
    p = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], float)
    a = np.deg2rad(angle); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return _poly(p @ R.T + [tx, ty])


def _shortest_and_hausdorff(P, Q):
    D = cdist(P, Q)
    return D.min(), max(D.min(axis=1).max(), D.min(axis=0).max())


def fig_shortest_distance(ctx: BuildContext) -> Figure:
    A = _poly([[0, 0], [6, 0], [6, 4], [0, 4]])
    B_near = _rect(4.8, 2.8, 0, 3, 2)                 # nested inside A with a uniform gap
    d_target = _shortest_and_hausdorff(A, B_near)[0]
    lo, hi = 6.0, 20.0                                 # slide a rotated copy out to the same shortest distance
    for _ in range(50):
        mid = (lo + hi) / 2
        if cdist(A, _rect(4.8, 2.8, 60, mid, 2)).min() < d_target:
            lo = mid
        else:
            hi = mid
    B_far = _rect(4.8, 2.8, 60, lo, 2)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw=dict(width_ratios=[0.9, 1.7, 1.1]))
    for ax, B, name in [(axes[0], B_near, "(a) outlines correspond well"), (axes[1], B_far, "(b) same shortest distance, no correspondence")]:
        s, h = _shortest_and_hausdorff(A, B)
        ax.plot(A[:, 0], A[:, 1], ".", color=C["red"], ms=2.5, label="shape A")
        ax.plot(B[:, 0], B[:, 1], ".", color=C["blue"], ms=2.5, label="shape B")
        D = cdist(A, B); i, j = np.unravel_index(np.argmin(D), D.shape)
        ax.plot([A[i, 0], B[j, 0]], [A[i, 1], B[j, 1]], color=C["ink"], lw=2.5, label=f"shortest = {s:.2f}")
        k = int(np.argmax(D.min(axis=1))); m = int(np.argmin(D[k]))
        ax.plot([A[k, 0], B[m, 0]], [A[k, 1], B[m, 1]], color=C["orange"], lw=2, ls="--", label=f"Hausdorff = {h:.2f}")
        ax.set_title(name, fontsize=9)
        pts = np.vstack([A, B])
        ax.set_aspect("equal")
        ax.set_xlim(pts[:, 0].min() - 0.7, pts[:, 0].max() + 0.7)
        ax.set_ylim(pts[:, 1].min() - 2.4, pts[:, 1].max() + 0.7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7.5, loc="lower center", ncol=2, columnspacing=1.0)
    img = np.zeros((70, 90), np.uint8)
    cv2.polylines(img, [np.array([[14, 50], [56, 50], [68, 22], [26, 22]], np.int32)], True, 1, 1)
    dt = ndimage.distance_transform_edt(1 - img)
    ys, xs = np.mgrid[12:62:8, 10:82:9]
    axes[2].imshow(dt, cmap="viridis"); axes[2].contour(img, levels=[0.5], colors=[C["blue"]], linewidths=1.5)
    axes[2].scatter(xs, ys, c=[C["red"]], s=12)
    axes[2].set_title(f"(c) distance transform of B; read off at A's\npixels (red), the minimum is {dt[ys, xs].min():.1f}", fontsize=9)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    png = save_mpl(fig, ctx, "shortest_distance")
    return Figure("shortest_distance",
                  "The shortest distance is a poor measure of how well two shapes match. "
                  "In (a) the outlines nearly coincide and in (b) one is rotated and "
                  "pushed away, yet both have exactly the same shortest distance (solid "
                  "line), because only the single closest pair of points is used. The "
                  "furthest-nearest pair (dashed), which is the Hausdorff distance, "
                  "separates the two cases immediately. Shortest distances are cheap to "
                  "get: the distance transform of B gives the distance from every pixel "
                  "to B, so the values at A's pixels are all the distances at once (c).",
                  png=png)


CHAMFER_RADIUS = 150


def chamfer_pipeline(key="plate_dutch", max_side=420, r=CHAMFER_RADIUS):
    """Edges -> inverted -> distance transform -> correlate with a ring template."""
    rgb = load_image(key, max_side=max_side)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.5), 60, 140)
    dt = ndimage.distance_transform_edt(edges == 0).astype(np.float32)
    tpl = ring_template(r, r - 4)
    score = cv2.matchTemplate(dt, tpl, cv2.TM_CCORR) / tpl.sum()
    y, x = np.unravel_index(np.argmin(score), score.shape)
    centre = (int(x + tpl.shape[1] // 2), int(y + tpl.shape[0] // 2))
    return rgb, edges, dt, tpl, score, centre


def fig_chamfer_matching(ctx: BuildContext) -> Figure:
    r = CHAMFER_RADIUS
    rgb, edges, dt, tpl, score, centre = chamfer_pipeline()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes[0, 0].imshow(rgb); axes[0, 0].set_title("image: find the circular plate", fontsize=9)
    axes[0, 1].imshow(edges, cmap="gray"); axes[0, 1].set_title("1. Canny edges after blurring", fontsize=9)
    axes[0, 2].imshow(dt, cmap="viridis"); axes[0, 2].set_title("2. distance transform of the inverted edges\n(the chamfer image)", fontsize=9)
    axes[1, 0].imshow(tpl, cmap="gray"); axes[1, 0].set_title(f"3. binary template: ring of radius {r} px", fontsize=9)
    im = axes[1, 1].imshow(score, cmap="viridis")
    ys, xs = np.unravel_index(np.argmin(score), score.shape)
    axes[1, 1].plot(xs, ys, "r*", ms=14)
    axes[1, 1].set_title(f"4. mean distance under the template;\nglobal minimum {score.min():.2f} px (star)", fontsize=9)
    out = rgb.copy(); cv2.circle(out, centre, r, (213, 94, 0), 3)
    axes[1, 2].imshow(out); axes[1, 2].set_title("5. the template at the global minimum", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "chamfer_matching")
    return Figure("chamfer_matching",
                  "Chamfer matching. Edges are detected and inverted, and the distance "
                  "transform gives every pixel its distance to the nearest edge; the "
                  "result is called the chamfer image because its edges become gentle "
                  "slopes. A binary template, here a ring with the radius of the plate, "
                  "is placed at every position and the chamfer values under it are "
                  "averaged, which can be done as a correlation. The global minimum of "
                  "that score is where the template best fits the edges.", png=png)


def fig_hausdorff(ctx: BuildContext) -> Figure:
    p = hd_gen.generate_params(seed=4)
    A = np.array(p["A"], float); B = np.array(p["B"], float)
    hAB, minsA, nnA = directed_hausdorff(A, B)
    hBA, minsB, nnB = directed_hausdorff(B, A)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (P, Q, mins, nn, name, col) in zip(axes[:2], [(A, B, minsA, nnA, "h(A → B)", C["red"]), (B, A, minsB, nnB, "h(B → A)", C["blue"])]):
        ax.plot(*np.vstack([A, A[:1]]).T, "-o", color=C["red"], ms=5, label="A"); ax.plot(*np.vstack([B, B[:1]]).T, "-s", color=C["blue"], ms=5, label="B")
        for i, (pt, j) in enumerate(zip(P, nn)):
            ax.plot([pt[0], Q[j, 0]], [pt[1], Q[j, 1]], color=C["grey"], lw=0.8)
        k = int(np.argmax(mins)); ax.plot([P[k, 0], Q[nn[k], 0]], [P[k, 1], Q[nn[k], 1]], color=col, lw=2.5)
        ax.set_title(f"{name} = max over points of the nearest distance = {mins.max():.2f}", fontsize=9); ax.set_aspect("equal"); ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    axes[2].axis("off")
    axes[2].text(0, 0.9, "directed distances are not symmetric:", fontsize=10)
    axes[2].text(0.05, 0.74, f"h(A → B) = {hAB:.2f}", fontsize=11, color=C["red"]); axes[2].text(0.05, 0.62, f"h(B → A) = {hBA:.2f}", fontsize=11, color=C["blue"])
    axes[2].text(0, 0.42, f"Hausdorff distance H(A, B) = max(h(A→B), h(B→A)) = {max(hAB, hBA):.2f}", fontsize=10, weight="bold")
    axes[2].text(0, 0.2, "H focuses on the single worst outlier point; in practice a\npercentile (e.g. 75%) of the nearest distances is more robust.", fontsize=8, color=C["grey"], va="top")
    png = save_mpl(fig, ctx, "hausdorff")
    return Figure("hausdorff",
                  "The Hausdorff distance. For every point of A the distance to the "
                  "nearest point of B is found (grey lines) and the largest of these is "
                  "the directed distance h(A → B); h(B → A) is defined the other way "
                  "round and is generally different. The Hausdorff distance is the larger "
                  "of the two. It separates overlapping from non-overlapping shapes far "
                  "better than the shortest distance, and can be computed from distance "
                  "transforms.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_template_search(ctx), fig_correlation_vs_ncc(ctx), fig_shape_matching(ctx),
            fig_shortest_distance(ctx), fig_chamfer_matching(ctx), fig_hausdorff(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("4.4 Template matching")
    tab1, tab2 = st.tabs(["Correlation methods", "Chamfer matching"])
    with tab1:
        g, tpl = face_and_eye()
        method = st.selectbox("similarity", ["squared difference", "cross-correlation", "normalised cross-correlation"])
        m = {"squared difference": cv2.TM_SQDIFF, "cross-correlation": cv2.TM_CCORR, "normalised cross-correlation": cv2.TM_CCOEFF_NORMED}[method]
        score = match_map(g, tpl, m)
        best = np.unravel_index(np.argmin(score) if m == cv2.TM_SQDIFF else np.argmax(score), score.shape)
        c1, c2 = st.columns(2)
        c1.image(norm01(score), caption=method, width="stretch")
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB); cv2.rectangle(out, (best[1], best[0]), (best[1] + tpl.shape[1], best[0] + tpl.shape[0]), (86, 180, 233), 2)
        c2.image(out, caption="best match", width="stretch")
    with tab2:
        key = st.selectbox("image", ["plate_dutch", "fruit", "rubiks_cube"])
        r = st.slider("ring radius (px)", 15, 180, CHAMFER_RADIUS if key == "plate_dutch" else 40)
        rgb, edges, dt, tpl, score, centre = chamfer_pipeline(key, 420, r)
        out = rgb.copy(); cv2.circle(out, centre, r, (213, 94, 0), 3)
        c1, c2, c3 = st.columns(3)
        c1.image(norm01(dt), caption="chamfer image", width="stretch")
        c2.image(norm01(score), caption=f"chamfer score (dark = good), min {score.min():.2f} px", width="stretch")
        c3.image(out, caption="template at the global minimum", width="stretch")
