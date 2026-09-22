"""Lesson 3.3 — Keypoint descriptors and matching.

Figures
    gradient_patch         gradients in an 8x8 patch, four 4x4 cells, four 8-bin histograms, one 32-vector
    characteristic_orientation  a patch and its rotated copy give different histograms until normalised
    ratio_test             nearest / second-nearest distance ratio separates good from bad matches
    matching_scaled        matches between an image and a scaled, rotated copy (interactive ratio)
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, load_image)
from ._sift_common import sift_keypoints, ratio_matches, draw_matches, gradients, orientation_histogram

FIGURES = ["gradient_patch", "characteristic_orientation", "ratio_test", "matching_scaled"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def galvani_patch(size=8, sigma=2.0):
    """An 8x8 patch around a strong keypoint on the statue, lightly smoothed for clean gradients."""
    g = load_image("galvani", gray=True, max_side=520).astype(float)
    g = cv2.GaussianBlur(g, (0, 0), sigma)
    kp, _ = sift_keypoints(g.astype(np.uint8), n=50)
    kp = sorted(kp, key=lambda k: -k.response)[0]
    x, y = int(kp.pt[0]), int(kp.pt[1])
    return g[y - size // 2 - 1:y + size // 2 + 1, x - size // 2 - 1:x + size // 2 + 1], (x, y)


def cell_histograms(mag, ang, cell=4, n_bins=8):
    """Histograms for each cell x cell block of an (interior) patch."""
    h, w = mag.shape
    hists = []
    for r in range(0, h, cell):
        for c in range(0, w, cell):
            hists.append(orientation_histogram(mag[r:r + cell, c:c + cell], ang[r:r + cell, c:c + cell], n_bins))
    return hists


def transformed_pair(key="butterfly", angle=35, scale=0.6, max_side=560):
    g = load_image(key, gray=True, max_side=max_side)
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return g, cv2.warpAffine(g, M, (w, h), borderValue=128), M


def match_correctness(kp1, kp2, matches, M, tol=4.0):
    """Fraction of matches whose keypoint maps within tol pixels of its partner under M."""
    ok = []
    for m in matches:
        p = np.array([*kp1[m.queryIdx].pt, 1.0])
        q = M @ p
        ok.append(np.linalg.norm(q - np.array(kp2[m.trainIdx].pt)) < tol)
    return np.array(ok, bool)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_gradient_patch(ctx: BuildContext) -> Figure:
    big, (x, y) = galvani_patch()
    mag, ang = gradients(big)
    mag, ang, patch = mag[1:-1, 1:-1], ang[1:-1, 1:-1], big[1:-1, 1:-1]
    hists = cell_histograms(mag, ang)
    vec = np.concatenate(hists)
    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1])
    g = load_image("galvani", gray=True, max_side=520)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(g, vmin=0, vmax=255); ax.add_patch(plt.Rectangle((x - 4, y - 4), 8, 8, fill=False, ec=C["red"], lw=1.5))
    ax.set_xlim(x - 40, x + 40); ax.set_ylim(y + 40, y - 40); ax.set_title("(a) 8 × 8 patch at a keypoint"); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(patch, cmap="gray")
    yy, xx = np.mgrid[0:8, 0:8]
    ax.quiver(xx, yy, np.cos(np.deg2rad(ang)) * mag, -np.sin(np.deg2rad(ang)) * mag, color=C["orange"], scale=mag.max() * 12, width=0.012)
    ax.axhline(3.5, color=C["sky"], lw=1.5); ax.axvline(3.5, color=C["sky"], lw=1.5); ax.set_title("(b) gradient at every pixel, split into 2 × 2 cells"); ax.set_xticks([]); ax.set_yticks([])
    for k, hst in enumerate(hists):
        ax = fig.add_subplot(gs[0, 2] if k < 2 else gs[0, 3]) if False else fig.add_subplot(gs[1, k])
        ax.bar(np.arange(8), hst, color=SERIES[k % 4]); ax.set_xticks(range(8)); ax.set_xticklabels([f"{45 * i}°" for i in range(8)], fontsize=6)
        ax.set_title(f"(c) histogram of cell {k + 1}", fontsize=9)
    ax = fig.add_subplot(gs[0, 2:]); ax.bar(np.arange(32), vec, color=[SERIES[i // 8 % 4] for i in range(32)])
    ax.set_title("(d) descriptor vector: four 8-bin histograms concatenated, length 32"); ax.set_xlabel("element")
    png = save_mpl(fig, ctx, "gradient_patch")
    return Figure("gradient_patch",
                  "Building a gradient-based descriptor. (b) The gradient magnitude and "
                  "angle are computed at every pixel of an 8 × 8 patch around the "
                  "keypoint, and the patch is divided into four 4 × 4 cells. (c) In each "
                  "cell an 8-bin orientation histogram is formed by adding each pixel's "
                  "magnitude to the bin of its angle. (d) Concatenating the four histograms "
                  "gives a 32-element descriptor vector.", png=png)


def fig_characteristic_orientation(ctx: BuildContext) -> Figure:
    g = load_image("plate_dutch", gray=True, max_side=640).astype(float)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    kp, _ = sift_keypoints(g.astype(np.uint8), n=40)
    kp = sorted(kp, key=lambda k: -k.response)[0]
    x, y = int(kp.pt[0]), int(kp.pt[1])
    R = 24
    patch = g[y - R:y + R, x - R:x + R]
    rot = cv2.warpAffine(g, cv2.getRotationMatrix2D((x, y), 60, 1.0), g.shape[::-1])[y - R:y + R, x - R:x + R]

    def dominant(p):
        m, a = gradients(p)
        yy, xx = np.mgrid[-R:R, -R:R]
        w = np.exp(-(xx ** 2 + yy ** 2) / (2 * (0.5 * R) ** 2))
        h = orientation_histogram(m[1:-1, 1:-1], a[1:-1, 1:-1], 36, w[1:-1, 1:-1])
        return h, (np.argmax(h) + 0.5) * 10

    h1, d1 = dominant(patch); h2, d2 = dominant(rot)
    n1 = cv2.warpAffine(patch, cv2.getRotationMatrix2D((R, R), d1, 1.0), (2 * R, 2 * R))
    n2 = cv2.warpAffine(rot, cv2.getRotationMatrix2D((R, R), d2, 1.0), (2 * R, 2 * R))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for r, (p, hst, d, nrm, name) in enumerate([(patch, h1, d1, n1, "original patch"), (rot, h2, d2, n2, "patch rotated by 60°")]):
        axes[r, 0].imshow(p, cmap="gray"); axes[r, 0].set_title(f"{name}"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].annotate("", xy=(R + 18 * np.cos(np.deg2rad(d)), R - 18 * np.sin(np.deg2rad(d))), xytext=(R, R), arrowprops=dict(arrowstyle="->", color=C["red"], lw=2))
        axes[r, 1].bar(np.arange(36) * 10, hst, width=9, color=C["blue"]); axes[r, 1].axvline(d, color=C["red"]); axes[r, 1].set_title(f"orientation histogram: peak at {d:.0f}°"); axes[r, 1].set_xlabel("gradient angle")
        axes[r, 2].imshow(nrm, cmap="gray"); axes[r, 2].set_title("rotated back by the peak angle"); axes[r, 2].set_xticks([]); axes[r, 2].set_yticks([])
    png = save_mpl(fig, ctx, "characteristic_orientation")
    return Figure("characteristic_orientation",
                  "Rotation invariance. The same keypoint in an image and in a rotated copy "
                  "gives different gradient histograms, and would give different "
                  "descriptors. The peak of the orientation histogram defines a "
                  "characteristic orientation (red arrow); rotating each patch by its own "
                  "peak angle aligns them, so the descriptor computed afterwards is the "
                  "same in both images.", png=png)


def fig_ratio_test(ctx: BuildContext) -> Figure:
    g, t, M = transformed_pair()
    k1, d1 = sift_keypoints(g); k2, d2 = sift_keypoints(t)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = bf.knnMatch(d1, d2, k=2)
    ratios = np.array([m.distance / max(n.distance, 1e-6) for m, n in pairs])
    firsts = [m for m, n in pairs]
    ok = match_correctness(k1, k2, firsts, M)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].hist(ratios[ok], bins=40, range=(0, 1), color=C["green"], alpha=0.8, label=f"correct matches ({ok.sum()})")
    axes[0].hist(ratios[~ok], bins=40, range=(0, 1), color=C["red"], alpha=0.6, label=f"wrong matches ({(~ok).sum()})")
    axes[0].axvline(0.7, color="k", ls="--"); axes[0].text(0.71, axes[0].get_ylim()[1] * 0.9, "ratio = 0.7", fontsize=8)
    axes[0].set_xlabel("distance to nearest / distance to second nearest"); axes[0].legend(fontsize=8); axes[0].set_title("Lowe's ratio test separates the two populations")
    ths = np.linspace(0.3, 1.0, 36)
    prec = [ok[ratios < th].mean() if (ratios < th).any() else 1 for th in ths]
    nmat = [(ratios < th).sum() for th in ths]
    ax2 = axes[1]; ax2.plot(ths, prec, color=C["green"], label="fraction of accepted matches that are correct"); ax2.set_ylabel("precision"); ax2.set_xlabel("ratio threshold")
    ax3 = ax2.twinx(); ax3.plot(ths, nmat, color=C["grey"], ls="--", label="matches accepted"); ax3.set_ylabel("matches accepted")
    ax2.set_title("threshold trade-off"); ax2.legend(fontsize=7, loc="lower left"); ax3.legend(fontsize=7, loc="lower right")
    png = save_mpl(fig, ctx, "ratio_test")
    return Figure("ratio_test",
                  "Brute-force matching compares every descriptor of one image with every "
                  "descriptor of the other. Because the second image is a known "
                  "transformation of the first, each match can be checked. Correct "
                  "matches have a nearest neighbour much closer than the second nearest "
                  "(small ratio); wrong matches have two neighbours at similar distances. "
                  "Accepting matches only when the ratio is below about 0.7 keeps nearly "
                  "all the correct ones and removes most of the wrong ones.", png=png)


def fig_matching_scaled(ctx: BuildContext) -> Figure:
    g, t, M = transformed_pair()
    k1, d1 = sift_keypoints(g); k2, d2 = sift_keypoints(t)
    good, pairs = ratio_matches(d1, d2, 0.7)
    good = sorted(good, key=lambda m: m.distance)
    ok = match_correctness(k1, k2, good, M)
    vis = draw_matches(g, k1, t, k2, good, max_draw=70)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(vis); ax.set_title(f"SIFT keypoints matched between the image and a copy rotated 35° and scaled 0.6: {len(good)} matches, {ok.mean() * 100:.0f}% correct")
    ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "matching_scaled")
    ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    frames = []
    for r in ratios:
        gd, _ = ratio_matches(d1, d2, r); gd = sorted(gd, key=lambda m: m.distance)
        okr = match_correctness(k1, k2, gd, M)
        vis = draw_matches(g, k1, t, k2, gd, max_draw=120)
        frames.append(go.Frame(name=str(r), data=[go.Image(source=to_data_uri(vis, "jpg"))],
                               layout=go.Layout(title=f"ratio < {r}: {len(gd)} matches, {okr.mean() * 100:.0f}% correct (120 drawn)")))
    fig2 = go.Figure(data=frames[2].data, frames=frames)
    fig2.update_layout(**slider_frames_layout(ratios, "ratio threshold = "), title=frames[2].layout.title.text)
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    return Figure("matching_scaled",
                  "Feature matching between an image and a scaled, rotated copy of itself "
                  "using SIFT descriptors and the ratio test. Almost every accepted match "
                  "joins the same physical point in the two images. Raise the ratio "
                  "threshold with the slider to admit more, and increasingly wrong, matches.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_gradient_patch(ctx), fig_characteristic_orientation(ctx), fig_ratio_test(ctx), fig_matching_scaled(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("3.3 Descriptors and matching")
    key = st.selectbox("image", ["butterfly", "galvani", "plate_dutch", "trevi_detail"])
    c1, c2, c3 = st.columns(3)
    ang = c1.slider("rotation (°)", -90, 90, 35); sc = c2.slider("scale", 0.3, 1.5, 0.6, 0.05); ratio = c3.slider("ratio threshold", 0.4, 1.0, 0.7, 0.05)
    g, t, M = transformed_pair(key, ang, sc)
    k1, d1 = sift_keypoints(g); k2, d2 = sift_keypoints(t)
    good, _ = ratio_matches(d1, d2, ratio); good = sorted(good, key=lambda m: m.distance)
    ok = match_correctness(k1, k2, good, M)
    st.image(draw_matches(g, k1, t, k2, good, max_draw=100), caption=f"{len(good)} matches, {ok.mean() * 100 if len(ok) else 0:.0f}% correct", width="stretch")
