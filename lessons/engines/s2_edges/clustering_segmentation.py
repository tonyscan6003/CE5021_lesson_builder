"""Lesson 2.5 — Clustering-based segmentation: k-means and mean shift.

Figures
    kmeans_steps        one k-means iteration at a time on 2D points (interactive)
    kmeans_colour       k-means on pixel colours for several k; a poor segmentation example
    parzen_density      histogram versus Parzen (kernel) density estimate (interactive bandwidth)
    mean_shift_path     the mean-shift window climbing to a density maximum; basins of attraction
    mean_shift_segment  mean-shift segmentation in Luv, and with spatial coordinates added
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans, MeanShift
from skimage.segmentation import mark_boundaries

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout, load_image, to_uint8)

FIGURES = ["kmeans_steps", "kmeans_colour", "parzen_density", "mean_shift_path", "mean_shift_segment"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def blobs_2d(seed=0, n=60):
    rng = np.random.default_rng(seed)
    centres = np.array([[2, 2], [7, 6], [3, 7.5]])
    pts = np.vstack([c + rng.normal(0, 0.8, (n, 2)) for c in centres])
    return pts


def kmeans_iterations(pts, k=3, seed=None, n_iter=6):
    """Run k-means from random initial centroids and record every iteration.

    With seed=None the run whose final clustering has the lowest total
    within-cluster distance (over 20 random starts) is returned, so the
    teaching example does not get stuck in a local minimum."""
    def run(s):
        rng = np.random.default_rng(s)
        cen = pts[rng.choice(len(pts), k, replace=False)].copy()
        hist = []
        for _ in range(n_iter):
            d = np.linalg.norm(pts[:, None, :] - cen[None, :, :], axis=2)
            lab = d.argmin(axis=1)
            hist.append((cen.copy(), lab.copy()))
            cen = np.array([pts[lab == j].mean(axis=0) if np.any(lab == j) else cen[j] for j in range(k)])
        d = np.linalg.norm(pts[:, None, :] - cen[None, :, :], axis=2)
        return hist, d.min(axis=1).sum()
    if seed is not None:
        return run(seed)[0]
    return min((run(s) for s in range(20)), key=lambda t: t[1])[0]


def kmeans_image(rgb, k, seed=0):
    px = rgb.reshape(-1, 3).astype(float)
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(px)
    return km.cluster_centers_[km.labels_].reshape(rgb.shape).astype(np.uint8), km.labels_.reshape(rgb.shape[:2])


def parzen(samples, xs, h):
    return np.mean(np.exp(-(xs[:, None] - samples[None, :]) ** 2 / (2 * h ** 2)), axis=1) / (np.sqrt(2 * np.pi) * h)


def mean_shift_track(pts, start, h=1.2, n_iter=25):
    x = np.array(start, float); path = [x.copy()]
    for _ in range(n_iter):
        w = np.exp(-np.sum((pts - x) ** 2, axis=1) / (2 * h ** 2))
        new = (w[:, None] * pts).sum(axis=0) / w.sum()
        path.append(new.copy())
        if np.linalg.norm(new - x) < 1e-3:
            break
        x = new
    return np.array(path)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_kmeans_steps(ctx: BuildContext) -> Figure:
    pts = blobs_2d()
    hist = kmeans_iterations(pts)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, i in zip(axes, [0, 1, 2, 5]):
        cen, lab = hist[i]
        for j in range(3):
            ax.plot(pts[lab == j, 0], pts[lab == j, 1], ".", color=SERIES[j], ms=5)
            ax.plot(cen[j, 0], cen[j, 1], "X", color=SERIES[j], ms=12, mec="k")
        ax.set_title("initial centroids" if i == 0 else f"iteration {i}"); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "kmeans_steps")
    fig2 = go.Figure()
    frames = []
    for i, (cen, lab) in enumerate(hist):
        data = []
        for j in range(3):
            data.append(go.Scatter(x=pts[lab == j, 0], y=pts[lab == j, 1], mode="markers", marker=dict(color=SERIES[j], size=6), name=f"cluster {j+1}"))
            data.append(go.Scatter(x=[cen[j, 0]], y=[cen[j, 1]], mode="markers", marker=dict(color=SERIES[j], size=16, symbol="x", line=dict(width=1, color="black")), showlegend=False))
        frames.append(go.Frame(name=str(i), data=data, traces=list(range(6)), layout=go.Layout(title=f"k-means iteration {i}")))
    for tr in frames[0].data:
        fig2.add_trace(tr)
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(list(range(len(hist))), "iteration "), title="k-means iteration 0")
    return Figure("kmeans_steps",
                  "k-means with k = 3. Each iteration assigns every point to its nearest "
                  "centroid (crosses) and then moves each centroid to the mean of its "
                  "points; the process converges in a few steps. The number of clusters "
                  "must be chosen in advance.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def fig_kmeans_colour(ctx: BuildContext) -> Figure:
    fruit = load_image("fruit", max_side=360)
    cow = load_image("cow_field", max_side=360)
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    axes[0, 0].imshow(fruit); axes[0, 0].set_title("image")
    for ax, k in zip(axes[0, 1:], [2, 4, 8]):
        ax.imshow(kmeans_image(fruit, k)[0]); ax.set_title(f"k = {k} clusters in RGB")
    axes[1, 0].imshow(cow); axes[1, 0].set_title("image")
    q, lab = kmeans_image(cow, 3)
    axes[1, 1].imshow(q); axes[1, 1].set_title("k = 3 in RGB")
    axes[1, 2].imshow(lab, cmap="tab10"); axes[1, 2].set_title("cluster labels")
    axes[1, 3].imshow(mark_boundaries(cow, lab, color=(1, 1, 0))); axes[1, 3].set_title("cow and shadow share a cluster")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "kmeans_colour")
    return Figure("kmeans_colour",
                  "k-means on pixel colours. Top: the fruit image quantised to 2, 4 and 8 "
                  "colour clusters; the clustering runs in RGB feature space and ignores "
                  "where pixels are. Bottom: the black patches of the cow and the dark "
                  "grass are assigned to the same cluster, so separate objects merge; "
                  "k-means alone gives poor segmentations.", png=png)


def fig_parzen_density(ctx: BuildContext) -> Figure:
    rng = np.random.default_rng(2)
    samples = np.concatenate([rng.normal(2, 0.5, 40), rng.normal(5, 0.8, 60), rng.normal(8.5, 0.4, 25)])
    xs = np.linspace(0, 11, 500)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    axes[0].hist(samples, bins=15, density=True, color=C["light"], ec=C["grey"]); axes[0].plot(samples, np.zeros_like(samples) - 0.01, "|", color=C["ink"])
    axes[0].set_title("(a) histogram estimate of the density")
    for ax, h in zip(axes[1:], [0.25, 0.7]):
        ax.plot(samples, np.zeros_like(samples) - 0.01, "|", color=C["ink"])
        for s in samples[::6]:
            ax.plot(xs, np.exp(-(xs - s) ** 2 / (2 * h ** 2)) / (np.sqrt(2 * np.pi) * h) / len(samples), color=C["orange"], lw=0.5)
        ax.plot(xs, parzen(samples, xs, h), color=C["blue"], lw=2); ax.set_title(f"(b) Parzen window, bandwidth h = {h}")
    for ax in axes:
        ax.set_xlabel("feature value"); ax.set_ylim(-0.03, 0.6)
    png = save_mpl(fig, ctx, "parzen_density")
    hs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
    fig2 = go.Figure()
    frames = [go.Frame(name=str(h), data=[go.Scatter(x=xs, y=parzen(samples, xs, h), line=dict(color=C["blue"], width=2))],
                       layout=go.Layout(title=f"Parzen density estimate, bandwidth h = {h}")) for h in hs]
    fig2.add_trace(frames[3].data[0]); fig2.add_trace(go.Scatter(x=samples, y=np.zeros_like(samples), mode="markers", marker=dict(symbol="line-ns-open", color="black"), showlegend=False))
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(hs, "h = "), title=frames[3].layout.title.text, showlegend=False, xaxis_title="feature value")
    return Figure("parzen_density",
                  "Estimating a probability density from samples. (a) A histogram gives a "
                  "crude estimate. (b) A Parzen window places a kernel, here a Gaussian of "
                  "bandwidth h, on every sample and sums them (equivalent to a convolution); "
                  "a small h follows the samples closely, a large h smooths them.",
                  png=png, html=plotly_html(fig2, height=380), kind="plotly")


def fig_mean_shift_path(ctx: BuildContext) -> Figure:
    pts = blobs_2d(seed=5, n=80)
    h = 1.2
    xs = np.linspace(-1, 10, 120); ys = np.linspace(-1, 10.5, 120)
    X, Y = np.meshgrid(xs, ys)
    dens = np.zeros_like(X)
    for p in pts:
        dens += np.exp(-((X - p[0]) ** 2 + (Y - p[1]) ** 2) / (2 * h ** 2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].contourf(X, Y, dens, levels=12, cmap="Blues"); axes[0].plot(pts[:, 0], pts[:, 1], ".", color=C["ink"], ms=3)
    for start in [(0.5, 5), (8.5, 9), (5, 0.5), (5.5, 4.5)]:
        path = mean_shift_track(pts, start, h)
        axes[0].plot(path[:, 0], path[:, 1], "-o", color=C["red"], ms=4, lw=1.2)
        axes[0].plot(path[-1, 0], path[-1, 1], "*", color=C["orange"], ms=14, mec="k")
    axes[0].set_title("Parzen density (contours) and mean-shift paths from four starts"); axes[0].set_xticks([]); axes[0].set_yticks([])
    rng = np.random.default_rng(1)
    starts = rng.uniform([-0.5, -0.5], [9.5, 10], (150, 2))
    modes = []
    cols = []
    for s in starts:
        m = mean_shift_track(pts, s, h)[-1]
        for i, mm in enumerate(modes):
            if np.linalg.norm(mm - m) < 0.5:
                cols.append(i); break
        else:
            modes.append(m); cols.append(len(modes) - 1)
    axes[1].contour(X, Y, dens, levels=8, colors=[C["light"]])
    axes[1].scatter(starts[:, 0], starts[:, 1], c=[SERIES[c % 6] for c in cols], s=18)
    for i, m in enumerate(modes):
        axes[1].plot(m[0], m[1], "*", color=SERIES[i % 6], ms=16, mec="k")
    axes[1].set_title(f"basins of attraction: {len(modes)} modes found from 150 random starts"); axes[1].set_xticks([]); axes[1].set_yticks([])
    png = save_mpl(fig, ctx, "mean_shift_path")
    return Figure("mean_shift_path",
                  "Mean shift. Left: from a starting point, the weighted mean of the "
                  "samples inside a Gaussian window is computed and the window moves "
                  "there; the mean-shift vector always points up the density gradient, so "
                  "the window climbs to a local maximum (star). Right: starting many "
                  "windows at random and colouring each start by the mode it reaches "
                  "reveals the basin of attraction of each cluster, without choosing the "
                  "number of clusters in advance.", png=png)


def fig_mean_shift_segment(ctx: BuildContext) -> Figure:
    rgb = load_image("fruit", max_side=160)
    luv = cv2.cvtColor(rgb, cv2.COLOR_RGB2Luv).reshape(-1, 3).astype(float)
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    spatial = np.column_stack([yy.ravel(), xx.ravel()]).astype(float)
    ms1 = MeanShift(bandwidth=14, bin_seeding=True).fit(luv)
    lab1 = ms1.labels_.reshape(h, w)
    feat = np.column_stack([luv, spatial * 0.6])
    ms2 = MeanShift(bandwidth=20, bin_seeding=True).fit(feat)
    lab2 = ms2.labels_.reshape(h, w)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].imshow(rgb); axes[0].set_title("image")
    for ax, lab, t in zip(axes[1:], [lab1, lab2], ["mean shift on (L, u, v)", "mean shift on (L, u, v, x, y)"]):
        mean_col = np.zeros_like(rgb)
        for l in np.unique(lab):
            mean_col[lab == l] = rgb[lab == l].mean(axis=0)
        ax.imshow(mean_col, interpolation="nearest"); ax.set_title(f"{t}: {lab.max() + 1} regions")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "mean_shift_segment")
    return Figure("mean_shift_segment",
                  "Mean-shift segmentation with scikit-learn; each region is filled with its "
                  "mean colour. Middle: clustering the pixels in the Luv colour space (a "
                  "luminance channel and two colour channels) groups similar colours "
                  "wherever they are. Right: adding the pixel coordinates as two more "
                  "features, with a product of a colour kernel and a spatial kernel, keeps "
                  "regions spatially compact. The bandwidth must be chosen by trial and "
                  "error.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_kmeans_steps(ctx), fig_kmeans_colour(ctx), fig_parzen_density(ctx),
            fig_mean_shift_path(ctx), fig_mean_shift_segment(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("2.5 Clustering-based segmentation")
    tab1, tab2, tab3 = st.tabs(["k-means on colours", "Parzen window", "Mean shift"])
    with tab1:
        key = st.selectbox("image", ["fruit", "cow_field", "macaw", "balloons"])
        rgb = load_image(key, max_side=360)
        k = st.slider("k", 2, 12, 4)
        q, lab = kmeans_image(rgb, k)
        c1, c2 = st.columns(2)
        c1.image(rgb, caption="image", width="stretch"); c2.image(q, caption=f"k = {k}", width="stretch")
    with tab2:
        rng = np.random.default_rng(2)
        samples = np.concatenate([rng.normal(2, 0.5, 40), rng.normal(5, 0.8, 60), rng.normal(8.5, 0.4, 25)])
        h = st.slider("bandwidth h", 0.05, 2.0, 0.4, 0.05)
        xs = np.linspace(0, 11, 500)
        fig, ax = plt.subplots(figsize=(7, 3)); ax.hist(samples, bins=20, density=True, color=C["light"]); ax.plot(xs, parzen(samples, xs, h), color=C["blue"], lw=2)
        st.pyplot(fig)
    with tab3:
        key = st.selectbox("photo", ["fruit", "cow_field", "balloons"])
        rgb = load_image(key, max_side=140)
        bw = st.slider("bandwidth", 5, 40, 14)
        use_xy = st.checkbox("include spatial coordinates", True)
        luv = cv2.cvtColor(rgb, cv2.COLOR_RGB2Luv).reshape(-1, 3).astype(float)
        h_, w_ = rgb.shape[:2]; yy, xx = np.mgrid[0:h_, 0:w_]
        feat = np.column_stack([luv, np.column_stack([yy.ravel(), xx.ravel()]) * 0.6]) if use_xy else luv
        lab = MeanShift(bandwidth=bw, bin_seeding=True).fit(feat).labels_.reshape(h_, w_)
        mean_col = np.zeros_like(rgb)
        for l in np.unique(lab):
            mean_col[lab == l] = rgb[lab == l].mean(axis=0)
        st.image(mean_col, caption=f"{lab.max() + 1} regions, each filled with its mean colour", width=420)
