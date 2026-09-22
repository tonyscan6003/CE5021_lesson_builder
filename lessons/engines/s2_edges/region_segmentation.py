"""Lesson 2.4 — Region-based segmentation: thresholding, Otsu, watershed, graph merging.

Figures
    threshold_segment    histogram-based threshold separating a foreground object (interactive)
    otsu                 between-class variance for every threshold and the Otsu optimum
    threshold_problems   histograms without a clear valley; small objects defeat Otsu
    image_surface        a greyscale image viewed as a relief surface (interactive)
    watershed_flood      flooding a 1D profile; watershed on touching fruit with and without markers
    graph_merge          Felzenszwalb graph-based segmentation at three scales
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed, felzenszwalb, mark_boundaries
from skimage.feature import peak_local_max

from .._common import (BuildContext, Figure, C, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, load_image, to_uint8)

FIGURES = ["threshold_segment", "otsu", "threshold_problems", "image_surface", "watershed_flood", "graph_merge"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def between_class_variance(g: np.ndarray):
    """Otsu's between-class variance for every threshold t in 0..254."""
    hist = np.bincount(g.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    levels = np.arange(256)
    w0 = np.cumsum(p)
    mu_cum = np.cumsum(p * levels)
    mu_T = mu_cum[-1]
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = mu_cum / w0
        mu1 = (mu_T - mu_cum) / w1
        sb = w0 * w1 * (mu0 - mu1) ** 2
    sb[~np.isfinite(sb)] = 0
    return sb[:255]


def sheep_scene():
    rgb = load_image("sheep", max_side=640)
    return rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def fruit_gray(max_side=420):
    return load_image("fruit", gray=True, max_side=max_side)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_threshold_segment(ctx: BuildContext) -> Figure:
    rgb, ch = sheep_scene()
    t = 160
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1, 1.2]))
    axes[0].imshow(rgb); axes[0].set_title("(a) image")
    axes[1].imshow(ch, vmin=0, vmax=255); axes[1].set_title("(b) greyscale")
    axes[2].hist(ch.ravel(), bins=64, color=C["grey"]); axes[2].axvline(t, color=C["red"]); axes[2].set_title(f"(c) histogram, t = {t}"); axes[2].set_xlabel("intensity")
    axes[3].imshow(ch > t, cmap="gray"); axes[3].set_title("(d) foreground mask")
    for ax in (axes[0], axes[1], axes[3]):
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "threshold_segment")

    ts = [100, 120, 140, 160, 180, 200, 220]
    frames = [go.Frame(name=str(tt), data=[go.Image(source=to_data_uri(to_uint8((ch > tt) * 255)))],
                       layout=go.Layout(title=f"threshold t = {tt}")) for tt in ts]
    fig2 = go.Figure(data=frames[3].data, frames=frames)
    fig2.update_layout(**slider_frames_layout(ts, "t = "), title=frames[3].layout.title.text)
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    return Figure("threshold_segment",
                  "Segmentation by thresholding. The histogram of the greyscale sheep photo "
                  "has a large mode for the grass and a smaller one for the bright sheep; "
                  "a threshold in the valley between them separates the two into a binary "
                  "mask. The method can be applied to a greyscale image or to any colour "
                  "channel. Move the slider to see how sensitive the mask is to t.",
                  png=png, html=plotly_html(fig2, height=420), kind="plotly")


def sheep_crop():
    """Close crop of two sheep so that foreground and background have comparable pixel counts."""
    _, ch = sheep_scene()
    h, w = ch.shape
    return ch[int(0.38 * h):int(0.72 * h), int(0.04 * w):int(0.42 * w)]


def fig_otsu(ctx: BuildContext) -> Figure:
    ch = sheep_crop()
    sb = between_class_variance(ch)
    t_opt = int(np.argmax(sb))
    t_sk = threshold_otsu(ch)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), gridspec_kw=dict(width_ratios=[1, 1.2, 1.2, 1]))
    axes[0].imshow(ch, vmin=0, vmax=255); axes[0].set_title("crop of the sheep image"); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].hist(ch.ravel(), bins=128, color=C["grey"]); axes[1].axvline(t_opt, color=C["red"], label=f"Otsu t = {t_opt}"); axes[1].legend(fontsize=8)
    axes[1].set_title("histogram"); axes[1].set_xlabel("intensity")
    axes[2].plot(sb, color=C["blue"]); axes[2].axvline(t_opt, color=C["red"]); axes[2].set_title("between-class variance σ²_B(t)"); axes[2].set_xlabel("threshold t")
    axes[3].imshow(ch > t_opt, cmap="gray"); axes[3].set_title(f"Otsu mask (skimage gives {t_sk})"); axes[3].set_xticks([]); axes[3].set_yticks([])
    png = save_mpl(fig, ctx, "otsu")
    return Figure("otsu",
                  "Otsu's method tries every threshold and picks the one that best "
                  "separates the two classes: it maximises the between-class variance "
                  "w₀ w₁ (μ₀ − μ₁)², which is equivalent to minimising the spread within "
                  "the foreground and background. On this crop, where sheep and grass have "
                  "comparable numbers of pixels, the optimum falls in the valley between "
                  "the two histogram modes.", png=png)


def fig_threshold_problems(ctx: BuildContext) -> Figure:
    cow = load_image("cow_field", max_side=420); cg = cv2.cvtColor(cow, cv2.COLOR_RGB2GRAY)
    ball = load_image("balloons", max_side=420); bg = cv2.cvtColor(ball, cv2.COLOR_RGB2GRAY)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), gridspec_kw=dict(width_ratios=[1.2, 1, 1.2]))
    for r, (rgb, g, name) in enumerate([(cow, cg, "cow: no clear valley"), (ball, bg, "balloons: small foreground")]):
        t = threshold_otsu(g)
        axes[r, 0].imshow(rgb); axes[r, 0].set_title(name)
        axes[r, 1].hist(g.ravel(), bins=64, color=C["grey"]); axes[r, 1].axvline(t, color=C["red"]); axes[r, 1].set_title(f"histogram, Otsu t = {t}")
        axes[r, 2].imshow(g > t, cmap="gray"); axes[r, 2].set_title("Otsu mask")
        for ax in (axes[r, 0], axes[r, 2]):
            ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "threshold_problems")
    return Figure("threshold_problems",
                  "Where thresholding fails. Most images have no obvious valley in the "
                  "histogram, and the cow's patches split between foreground and background "
                  "(top). Otsu assumes comparable numbers of foreground and background "
                  "pixels, so with small objects such as balloons it splits the sky instead "
                  "(bottom). Thresholding is rarely sufficient on its own.", png=png)


def fig_image_surface(ctx: BuildContext) -> Figure:
    g = fruit_gray(420)
    small = cv2.resize(g, (96, 72), interpolation=cv2.INTER_AREA).astype(float)
    Y, X = np.mgrid[0:small.shape[0], 0:small.shape[1]]
    fig = plt.figure(figsize=(11, 4))
    ax = fig.add_subplot(1, 2, 1); ax.imshow(g, vmin=0, vmax=255); ax.set_title("greyscale image"); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1, 2, 2, projection="3d"); ax.plot_surface(X, -Y, small, cmap="viridis", linewidth=0)
    ax.set_title("the same image as a surface: bright = high"); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    png = save_mpl(fig, ctx, "image_surface")
    fig2 = go.Figure(go.Surface(z=small[::-1], colorscale="Viridis", showscale=False))
    fig2.update_layout(title="Drag to rotate: intensity as height", scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_title="intensity"))
    return Figure("image_surface",
                  "A greyscale image can be read as a topographic surface in which bright "
                  "pixels are hills and dark pixels are valleys. The watershed algorithm "
                  "is defined on this relief.",
                  png=png, html=plotly_html(fig2, height=480), kind="plotly")


def fig_watershed_flood(ctx: BuildContext) -> Figure:
    x = np.linspace(0, 10, 500)
    prof = 3 + np.sin(x) + 0.6 * np.sin(2.3 * x + 1) + 0.4 * np.sin(5 * x)
    g = fruit_gray(420)
    sm = cv2.GaussianBlur(g, (0, 0), 2)
    grad = cv2.morphologyEx(sm, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ws_plain = watershed(grad)
    _, mask = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) // 255
    dist = ndimage.distance_transform_edt(mask)
    coords = peak_local_max(dist, min_distance=18, labels=mask)
    markers = np.zeros_like(mask, int); markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    ws_mark = watershed(-dist, markers, mask=mask)
    fig = plt.figure(figsize=(14, 6.6))
    gs = fig.add_gridspec(2, 4)
    for j, lvl in enumerate([2.0, 2.6, 3.2, 3.8]):
        ax = fig.add_subplot(gs[0, j]); ax.plot(x, prof, color=C["ink"]); ax.fill_between(x, prof, lvl, where=prof < lvl, color=C["sky"], alpha=0.7)
        ax.set_ylim(1.5, 5); ax.set_title(f"water level {lvl}"); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 0]); ax.imshow(g, vmin=0, vmax=255); ax.set_title("touching fruit", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 1]); ax.imshow(ws_plain % 20, cmap="tab20"); ax.set_title(f"no markers: {ws_plain.max()} regions", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 2]); ax.imshow(dist, cmap="viridis"); ax.plot(coords[:, 1], coords[:, 0], "r+", ms=8); ax.set_title("distance transform, markers at peaks", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 3]); ax.imshow(mark_boundaries(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), ws_mark, color=(1, 0.3, 0))); ax.set_title(f"with markers: {ws_mark.max()} regions", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "watershed_flood")
    return Figure("watershed_flood",
                  "Watershed. Top: the water table rises from every local minimum of the "
                  "surface; where two pools would meet a dam (a boundary) is built. Bottom: "
                  "flooding the gradient image of touching fruit from all its minima "
                  "over-segments it badly. Using markers, here the peaks of the distance "
                  "transform of a thresholded mask, only the chosen basins are flooded and "
                  "each fruit becomes one region.", png=png)


def fig_graph_merge(ctx: BuildContext) -> Figure:
    rgb = load_image("cycle_track", max_side=400)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    axes[0].imshow(rgb); axes[0].set_title("image")
    for ax, sc in zip(axes[1:], [50, 300, 1500]):
        seg = felzenszwalb(rgb, scale=sc, sigma=0.8, min_size=30)
        ax.imshow(mark_boundaries(rgb, seg, color=(1, 1, 0))); ax.set_title(f"scale = {sc}: {seg.max() + 1} regions")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "graph_merge")
    return Figure("graph_merge",
                  "Graph-based split and merge (Felzenszwalb and Huttenlocher, 2004). Every "
                  "pixel starts as its own region; edges between neighbouring pixels are "
                  "visited in order of increasing weight and regions are merged unless the "
                  "difference between them is large compared with their internal "
                  "differences. A scale parameter controls how readily regions merge; the "
                  "result tends to over-segment but is a good basis for further merging.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_threshold_segment(ctx), fig_otsu(ctx), fig_threshold_problems(ctx), fig_image_surface(ctx),
            fig_watershed_flood(ctx), fig_graph_merge(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("2.4 Region-based segmentation")
    tab1, tab2, tab3 = st.tabs(["Threshold and Otsu", "Watershed", "Graph-based"])
    with tab1:
        key = st.selectbox("image", ["sheep", "cow_field", "balloons", "starfish", "fruit"])
        rgb = load_image(key, max_side=480)
        chan = st.radio("channel", ["grey", "R", "G", "B"], horizontal=True)
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if chan == "grey" else rgb[:, :, "RGB".index(chan)]
        t_otsu = int(threshold_otsu(g))
        t = st.slider("threshold", 0, 255, t_otsu)
        st.caption(f"Otsu threshold for this channel: {t_otsu}")
        c1, c2 = st.columns(2)
        c1.image(g, caption=chan, width="stretch"); c2.image(to_uint8((g > t) * 255), caption=f"mask at t = {t}", width="stretch")
        fig, ax = plt.subplots(figsize=(7, 2.2)); ax.plot(between_class_variance(g), color=C["blue"]); ax.axvline(t, color=C["red"]); ax.set_title("between-class variance"); st.pyplot(fig)
    with tab2:
        g = fruit_gray(420); sm = cv2.GaussianBlur(g, (0, 0), 2)
        md = st.slider("marker minimum distance", 5, 60, 18)
        _, mask = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) // 255
        dist = ndimage.distance_transform_edt(mask)
        coords = peak_local_max(dist, min_distance=md, labels=mask)
        markers = np.zeros_like(mask, int); markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        ws = watershed(-dist, markers, mask=mask)
        st.image(to_uint8(mark_boundaries(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), ws, color=(1, 0.3, 0))), caption=f"{ws.max()} regions from {len(coords)} markers", width="stretch")
    with tab3:
        key = st.selectbox("photo", ["cycle_track", "cow_field", "fruit", "trevi"])
        rgb = load_image(key, max_side=400)
        sc = st.select_slider("scale", [20, 50, 100, 300, 800, 1500], value=300)
        seg = felzenszwalb(rgb, scale=sc, sigma=0.8, min_size=30)
        st.image(to_uint8(mark_boundaries(rgb, seg, color=(1, 1, 0))), caption=f"{seg.max() + 1} regions", width="stretch")
