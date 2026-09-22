"""Lesson 2.2 — The Canny algorithm and the Hough transform.

Figures
    sobel_vs_canny     thick, uneven Sobel edges versus thin, binary Canny edges
    canny_stages       smoothing, gradient, non-maximum suppression, hysteresis
    nms_patch          non-maximum suppression on a 5x5 magnitude grid
    hysteresis         strong, weak-connected and weak-isolated pixels
    hough_parameter    image points become lines (m, b) or sinusoids (rho, theta) in parameter space
    hough_accumulator  voting: the accumulator peak gives the line through the points
    hough_photo        Hough lines on the Canny edges of a photograph (interactive)
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .._common import (BuildContext, Figure, C, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, load_image, sync_matrix, to_uint8)
from .._vendor import canny_nms as nms_gen

FIGURES = ["sobel_vs_canny", "canny_stages", "nms_patch", "hysteresis",
           "hough_parameter", "hough_accumulator", "hough_photo"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def sobel_mag_dir(g):
    gx = cv2.Sobel(g.astype(float), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g.astype(float), cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def non_max_suppress(mag, ang):
    """Keep a pixel only if it is a maximum along its (quantised) gradient direction."""
    q = ((np.rad2deg(ang) % 180) + 22.5) // 45 % 4         # 0:0°, 1:45°, 2:90°, 3:135°
    out = np.zeros_like(mag)
    offs = {0: ((0, -1), (0, 1)), 1: ((-1, 1), (1, -1)), 2: ((-1, 0), (1, 0)), 3: ((-1, -1), (1, 1))}
    h, w = mag.shape
    for d, ((r1, c1), (r2, c2)) in offs.items():
        m = q == d
        a = np.zeros_like(mag); b = np.zeros_like(mag)
        a[1:-1, 1:-1] = mag[1 + r1:h - 1 + r1, 1 + c1:w - 1 + c1]
        b[1:-1, 1:-1] = mag[1 + r2:h - 1 + r2, 1 + c2:w - 1 + c2]
        keep = m & (mag >= a) & (mag >= b)
        out[keep] = mag[keep]
    return out


def hough_lines(points, n_theta=180, rho_res=1.0):
    """Vote every point into a (rho, theta) accumulator."""
    thetas = np.linspace(0, np.pi, n_theta, endpoint=False)
    pts = np.asarray(points, float)
    diag = np.hypot(pts[:, 0].max(), pts[:, 1].max()) + 1
    rhos = np.arange(-diag, diag + rho_res, rho_res)
    acc = np.zeros((len(rhos), n_theta), int)
    for x, y in pts:
        r = x * np.cos(thetas) + y * np.sin(thetas)
        idx = np.round((r + diag) / rho_res).astype(int)
        acc[idx, np.arange(n_theta)] += 1
    return acc, rhos, thetas


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_sobel_vs_canny(ctx: BuildContext) -> Figure:
    g = load_image("trevi", gray=True, max_side=520)
    mag, _ = sobel_mag_dir(cv2.GaussianBlur(g, (0, 0), 1.0))
    edges = cv2.Canny(g, 80, 180)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].imshow(g, vmin=0, vmax=255); axes[0].set_title("image")
    axes[1].imshow(mag, vmin=0, vmax=np.percentile(mag, 99)); axes[1].set_title("Sobel magnitude: thick, many weak edges")
    axes[2].imshow(edges, cmap="gray"); axes[2].set_title("Canny: one-pixel edges, single intensity")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "sobel_vs_canny")
    return Figure("sobel_vs_canny",
                  "Simple derivative kernels return edges several pixels wide, of varying "
                  "strength and including many minor edges. The Canny algorithm returns "
                  "thin, single-pixel edges at one intensity and discards the weak ones.",
                  png=png)


def fig_canny_stages(ctx: BuildContext) -> Figure:
    g = load_image("trevi", gray=True, max_side=520)
    crop = g[80:330, 150:450]
    sm = cv2.GaussianBlur(crop, (0, 0), 1.4)
    mag, ang = sobel_mag_dir(sm)
    thin = non_max_suppress(mag, ang)
    final = cv2.Canny(sm, 60, 150)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3))
    panels = [("1. image", crop, 255), ("2. Gaussian smoothing", sm, 255), ("3. Sobel magnitude", mag, np.percentile(mag, 99)),
              ("4. non-maximum suppression", thin, np.percentile(mag, 99)), ("5. hysteresis thresholding", final, 255)]
    for ax, (t, im, v) in zip(axes, panels):
        ax.imshow(im, vmin=0, vmax=v); ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "canny_stages")
    return Figure("canny_stages",
                  "The Canny pipeline. The image is smoothed to suppress noise, Sobel "
                  "gradients give a magnitude and direction, non-maximum suppression keeps "
                  "only the pixels that are local maxima along the gradient direction "
                  "(thinning the ridges), and hysteresis thresholding keeps strong edges "
                  "and weak edges connected to them.", png=png)


def fig_nms_patch(ctx: BuildContext) -> Figure:
    p = nms_gen.generate_params(seed=7)
    c = nms_gen.compute(p)
    mags = np.array(p["mags"]); dirs = np.array(p["dirs"]); res = np.array(c["result"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    sync_matrix(axes[0], mags, title="gradient magnitude |G|")
    sync_matrix(axes[1], mags, title="gradient direction (arrows)", cmap="Greys")
    for r in range(5):
        for cc in range(5):
            a = np.deg2rad(dirs[r, cc])
            axes[1].annotate("", xy=(cc + 0.35 * np.cos(a), r - 0.35 * np.sin(a)), xytext=(cc - 0.35 * np.cos(a), r + 0.35 * np.sin(a)),
                             arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.2))
    sync_matrix(axes[2], res, cmap="Greens", title="kept after NMS (interior pixels)")
    png = save_mpl(fig, ctx, "nms_patch")
    return Figure("nms_patch",
                  "Non-maximum suppression on a 5 × 5 block. Each interior pixel is compared "
                  "with its two neighbours along its gradient direction (arrows); it "
                  "survives only if its magnitude is at least as large as both. Ridges "
                  "several pixels wide are reduced to their crest.", png=png)


def fig_hysteresis(ctx: BuildContext) -> Figure:
    grid = np.array([[5, 8, 60, 90, 9], [7, 55, 88, 50, 8], [45, 70, 12, 10, 6], [9, 48, 7, 40, 30], [6, 8, 9, 35, 44]])
    tmax, tmin = 80, 25
    strong = grid >= tmax; weak = (grid >= tmin) & (grid < tmax)
    # connected weak pixels: grow from strong through weak (8-connectivity)
    keep = strong.copy()
    for _ in range(10):
        grown = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & weak
        keep = keep | grown
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    sync_matrix(axes[0], grid, title=f"magnitudes; T_max = {tmax}, T_min = {tmin}")
    cls = np.where(strong, 2, np.where(weak, 1, 0))
    sync_matrix(axes[1], cls, cmap="YlOrRd", fmt="{:.0f}", title="2 = strong, 1 = weak, 0 = below T_min")
    sync_matrix(axes[2], keep.astype(int), cmap="Greens", fmt="{:.0f}", title="final edge: strong + connected weak")
    png = save_mpl(fig, ctx, "hysteresis")
    return Figure("hysteresis",
                  "Hysteresis thresholding. Pixels above T_max are strong edges and are "
                  "kept; pixels below T_min are discarded; pixels in between are kept only "
                  "if they connect to a strong pixel. The weak pixels at the bottom right "
                  "form a chain that never touches a strong pixel, so they are removed.",
                  png=png)


def fig_hough_parameter(ctx: BuildContext) -> Figure:
    pts = [(1, 1), (3, 2)]
    m = np.linspace(-2, 2, 200)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    for (x, y), col in zip(pts, [C["blue"], C["orange"]]):
        ax.plot(x, y, "o", color=col, ms=8)
        for mm in np.linspace(-1.5, 2.5, 7):
            ax.plot([-1, 5], [y + mm * (-1 - x), y + mm * (5 - x)], color=col, lw=0.6, alpha=0.5)
    ax.plot([-1, 5], [1 + 0.5 * (-2), 1 + 0.5 * 4], color=C["green"], lw=2, label="line through both: y = 0.5x + 0.5")
    ax.set_xlim(-1, 5); ax.set_ylim(-1, 4); ax.set_title("(a) image space: candidate lines through each point"); ax.legend(fontsize=7); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax = axes[1]
    for (x, y), col in zip(pts, [C["blue"], C["orange"]]):
        ax.plot(m, y - m * x, color=col, label=f"point ({x},{y}): b = {y} − {x}m")
    ax.plot(0.5, 0.5, "o", color=C["green"], ms=9); ax.set_title("(b) parameter space (m, b): each point is a line"); ax.legend(fontsize=7); ax.set_xlabel("m"); ax.set_ylabel("b")
    ax = axes[2]
    th = np.linspace(0, np.pi, 300)
    for (x, y), col in zip(pts, [C["blue"], C["orange"]]):
        ax.plot(th, x * np.cos(th) + y * np.sin(th), color=col, label=f"point ({x},{y})")
    ax.set_title("(c) polar parameters (θ, ρ): each point is a sinusoid"); ax.legend(fontsize=7); ax.set_xlabel("θ (rad)"); ax.set_ylabel("ρ = x cos θ + y sin θ")
    png = save_mpl(fig, ctx, "hough_parameter")
    return Figure("hough_parameter",
                  "The Hough transform. (a) Many lines pass through each edge point. (b) In "
                  "the (m, b) parameter space every candidate line through a point maps "
                  "to one point, so a point in the image becomes a line in parameter "
                  "space, and the two lines intersect at the parameters of the line "
                  "through both image points (green). (c) With the bounded polar "
                  "parameters (ρ, θ) each image point becomes a sinusoid instead.", png=png)


def fig_hough_accumulator(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(1)
    xs = np.linspace(5, 95, 12); ys = 0.6 * xs + 10 + rng.normal(0, 0.8, xs.size)
    noise = rng.uniform(5, 95, (8, 2))
    pts = np.vstack([np.column_stack([xs, ys]), noise])
    acc, rhos, thetas = hough_lines(pts, n_theta=180, rho_res=1.0)
    i, j = np.unravel_index(np.argmax(acc), acc.shape)
    rho, th = rhos[i], thetas[j]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].plot(pts[:, 0], pts[:, 1], "o", color=C["blue"]); axes[0].set_title("(a) edge points (12 collinear + 8 noise)"); axes[0].set_aspect("equal")
    for x, y in pts:
        axes[1].plot(thetas, x * np.cos(thetas) + y * np.sin(thetas), lw=0.6, color=C["blue"], alpha=0.5)
    axes[1].plot(th, rho, "*", color=C["red"], ms=12); axes[1].set_title("(b) sinusoids in (θ, ρ); peak marked"); axes[1].set_xlabel("θ"); axes[1].set_ylabel("ρ")
    axes[2].imshow(acc, aspect="auto", cmap="magma", extent=[0, np.pi, rhos[-1], rhos[0]]); axes[2].set_title(f"(c) accumulator votes; max {acc.max()} at θ = {th:.2f}, ρ = {rho:.0f}")
    axes[2].set_xlabel("θ"); axes[2].set_ylabel("ρ")
    xx = np.array([0, 100]); axes[0].plot(xx, (rho - xx * np.cos(th)) / np.sin(th), color=C["red"], label="line from the peak"); axes[0].legend(fontsize=7)
    png = save_mpl(fig, ctx, "hough_accumulator")
    return Figure("hough_accumulator",
                  "Voting. Every edge point votes along its sinusoid in a quantised (ρ, θ) "
                  "accumulator; the cell with most votes is where the sinusoids of the "
                  "collinear points cross, and it gives the line (red) that fits them. "
                  "Noise points add scattered votes that never accumulate.", png=png)


def fig_hough_photo(ctx: BuildContext) -> Figure:
    rgb = load_image("office_facade", max_side=560)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(g, 80, 200)

    def draw(threshold):
        out = rgb.copy()
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
        n = 0 if lines is None else len(lines)
        for l in (lines if lines is not None else [])[:200]:
            rho, th = l[0]
            a, b = np.cos(th), np.sin(th); x0, y0 = a * rho, b * rho
            cv2.line(out, (int(x0 + 2000 * (-b)), int(y0 + 2000 * a)), (int(x0 - 2000 * (-b)), int(y0 - 2000 * a)), (213, 94, 0), 1)
        return out, n

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].imshow(edges, cmap="gray"); axes[0].set_title("Canny edges")
    for ax, thr in zip(axes[1:], [220, 120]):
        out, n = draw(thr); ax.imshow(out); ax.set_title(f"Hough lines, vote threshold {thr}: {n} lines")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "hough_photo")

    thrs = [300, 250, 200, 160, 120, 90]
    frames = [go.Frame(name=str(t), data=[go.Image(source=to_data_uri(draw(t)[0], "jpg"))],
                       layout=go.Layout(title=f"vote threshold {t}: {draw(t)[1]} lines")) for t in thrs]
    fig2 = go.Figure(data=frames[2].data, frames=frames)
    fig2.update_layout(**slider_frames_layout(thrs, "threshold = "), title=frames[2].layout.title.text)
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    return Figure("hough_photo",
                  "Lines from edges. The Hough transform is applied to the Canny edge map of "
                  "a building; lines whose accumulator cell exceeds a vote threshold are "
                  "drawn. Lowering the threshold returns more, and less reliable, lines. "
                  "Because whole lines are fitted, gaps in the edge map no longer matter.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_sobel_vs_canny(ctx), fig_canny_stages(ctx), fig_nms_patch(ctx), fig_hysteresis(ctx),
            fig_hough_parameter(ctx), fig_hough_accumulator(ctx), fig_hough_photo(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("2.2 Canny and Hough")
    tab1, tab2 = st.tabs(["Canny thresholds", "Hough lines"])
    with tab1:
        key = st.selectbox("image", ["trevi", "notre_dame_b", "office_facade", "galvani"])
        g = load_image(key, gray=True, max_side=520)
        s = st.slider("Gaussian σ", 0.5, 4.0, 1.4, 0.1)
        lo, hi = st.slider("hysteresis thresholds (T_min, T_max)", 0, 400, (60, 150))
        sm = cv2.GaussianBlur(g, (0, 0), s)
        c1, c2 = st.columns(2)
        c1.image(g, caption="image", width="stretch"); c2.image(cv2.Canny(sm, lo, hi), caption="Canny edges", width="stretch")
    with tab2:
        key = st.selectbox("photo", ["office_facade", "notre_dame_b", "trevi"])
        rgb = load_image(key, max_side=560); g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(g, 80, 200)
        thr = st.slider("vote threshold", 50, 400, 150, 10)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, thr)
        out = rgb.copy()
        for l in (lines if lines is not None else [])[:300]:
            rho, th = l[0]; a, b = np.cos(th), np.sin(th); x0, y0 = a * rho, b * rho
            cv2.line(out, (int(x0 + 2000 * (-b)), int(y0 + 2000 * a)), (int(x0 - 2000 * (-b)), int(y0 - 2000 * a)), (213, 94, 0), 1)
        c1, c2 = st.columns(2)
        c1.image(edges, caption="Canny edges", width="stretch"); c2.image(out, caption=f"{0 if lines is None else len(lines)} lines", width="stretch")
