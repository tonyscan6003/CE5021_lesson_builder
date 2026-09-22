"""Lesson 1.1 — Digital images: sampling, quantisation, Bayer colour capture, RGB.

Figures
    sample_quantise   photo re-sampled on coarser pixel grids and fewer bits (interactive)
    bayer_demosaic    RGGB colour filter array, raw sensor values, demosaiced result
    resolution_pixels the same scene recorded with four times fewer pixels
    rgb_channels      a colour image and its three 8-bit channels
    additive_mixing   red, green and blue light adding to secondary colours and white
    rgb_colour_space  pixels of a photo plotted inside the RGB cube (interactive)
    display_rgb       an LCD monitor and its red, green and blue sub-pixels
    camera_effects    rolling shutter, sensor gain noise and JPEG compression artefacts
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .._common import (BuildContext, Figure, C, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, load_image,
                       upscale_nearest)

FIGURES = ["sample_quantise", "bayer_demosaic", "resolution_pixels", "rgb_channels",
           "additive_mixing", "rgb_colour_space", "display_rgb", "camera_effects"]

GRID_SIZES = [8, 16, 32, 64, 128, 256]
BIT_DEPTHS = [1, 2, 3, 4, 8]


# ---------------------------------------------------------------------------
# Core computations (also used by demo())
# ---------------------------------------------------------------------------

def centre_square(img: np.ndarray, frac: float = 1.0, cx: float = 0.5, cy: float = 0.5) -> np.ndarray:
    """Square crop of side frac*min(h, w) centred at fractional position (cx, cy)."""
    h, w = img.shape[:2]
    side = int(min(h, w) * frac)
    y0 = int(np.clip(cy * h - side / 2, 0, h - side))
    x0 = int(np.clip(cx * w - side / 2, 0, w - side))
    return img[y0:y0 + side, x0:x0 + side]


def cat_square(size: int = 256) -> np.ndarray:
    """Square greyscale crop of the cat photo, `size` pixels a side."""
    g = load_image("cat", gray=True, max_side=1024)
    crop = centre_square(g, frac=0.85, cx=0.5, cy=0.45)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def resample_grid(img: np.ndarray, n: int) -> np.ndarray:
    """Sample the scene on an n x n pixel grid (area average, like a sensor)."""
    return cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)


def quantise(img: np.ndarray, bits: int) -> np.ndarray:
    """Quantise 8-bit values to `bits` bits and stretch back to 0..255 for display."""
    levels = 2 ** bits
    q = np.floor(img.astype(np.float64) / 256.0 * levels)
    return np.round(q * (255.0 / (levels - 1))).astype(np.uint8)


def make_bayer_mosaic(rgb: np.ndarray):
    """Return (raw sensor values, colour-coded CFA view) for an RGGB Bayer filter."""
    raw = np.zeros(rgb.shape[:2], np.uint8)
    raw[0::2, 0::2] = rgb[0::2, 0::2, 0]   # R
    raw[0::2, 1::2] = rgb[0::2, 1::2, 1]   # G
    raw[1::2, 0::2] = rgb[1::2, 0::2, 1]   # G
    raw[1::2, 1::2] = rgb[1::2, 1::2, 2]   # B
    cfa = np.zeros_like(rgb)
    cfa[0::2, 0::2, 0] = raw[0::2, 0::2]
    cfa[0::2, 1::2, 1] = raw[0::2, 1::2]
    cfa[1::2, 0::2, 1] = raw[1::2, 0::2]
    cfa[1::2, 1::2, 2] = raw[1::2, 1::2]
    return raw, cfa


def demosaic(raw: np.ndarray) -> np.ndarray:
    # OpenCV names the pattern by the 2x2 block at (1,1); RGGB -> "BG".
    return cv2.cvtColor(raw, cv2.COLOR_BayerBG2RGB)


def macaw_crop(size: int = 192) -> np.ndarray:
    rgb = load_image("macaw", max_side=640)
    crop = centre_square(rgb, frac=0.5, cx=0.5, cy=0.4)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def jpeg_roundtrip(rgb: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out, len(buf)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_sample_quantise(ctx: BuildContext) -> Figure:
    base = cat_square(256)
    grids = [16, 64, 256]
    bits = [1, 3, 8]
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.4))
    for r, n in enumerate(grids):
        for c_, b in enumerate(bits):
            im = quantise(resample_grid(base, n), b)
            ax = axes[r, c_]
            ax.imshow(upscale_nearest(im, 256 // n), vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{b}-bit ({2**b} levels)")
            if c_ == 0:
                ax.set_ylabel(f"{n} × {n} pixels", fontsize=10)
    png = save_mpl(fig, ctx, "sample_quantise")

    frames = []
    for n in GRID_SIZES:
        traces = []
        for b in BIT_DEPTHS:
            im = upscale_nearest(quantise(resample_grid(base, n), b), 256 // n)
            traces.append(go.Image(source=to_data_uri(im), visible=(b == 8)))
        frames.append(go.Frame(name=str(n), data=traces))
    fig2 = go.Figure(data=frames[-1].data, frames=frames)
    buttons = []
    for i, b in enumerate(BIT_DEPTHS):
        vis = [j == i for j in range(len(BIT_DEPTHS))]
        buttons.append(dict(label=f"{b}-bit", method="restyle", args=[{"visible": vis}]))
    layout = slider_frames_layout(GRID_SIZES, "pixel grid: ", extra_updatemenus=[
        dict(type="buttons", direction="right", buttons=buttons, active=4,
             x=0.5, xanchor="center", y=1.12, yanchor="top")])
    fig2.update_layout(**layout, title="Sampling grid (slider) and quantisation (buttons)")
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    return Figure("sample_quantise",
                  "Sampling and quantisation of the same scene. Rows use a coarser "
                  "pixel grid (fewer samples), columns use fewer bits per pixel "
                  "(fewer grey levels). Fine detail needs many pixels; smooth "
                  "tones need enough levels to avoid false contours.",
                  png=png, html=plotly_html(fig2, height=520), kind="plotly")


def fig_bayer_demosaic(ctx: BuildContext) -> Figure:
    rgb = macaw_crop(192)
    raw, cfa = make_bayer_mosaic(rgb)
    out = demosaic(raw)
    z = 8
    yz, xz = 80, 90
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3))
    axes[0].imshow(rgb); axes[0].set_title("(a) Scene")
    axes[1].imshow(upscale_nearest(cfa[yz:yz + z, xz:xz + z], 20))
    axes[1].set_title("(b) Bayer filter, 8 × 8 close-up")
    axes[2].imshow(raw, vmin=0, vmax=255); axes[2].set_title("(c) Raw sensor values")
    axes[3].imshow(out); axes[3].set_title("(d) After demosaicing")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].add_patch(plt.Rectangle((xz, yz), z, z, fill=False, ec="white", lw=1.5))
    png = save_mpl(fig, ctx, "bayer_demosaic")
    return Figure("bayer_demosaic",
                  "Colour capture with a single sensor. (b) An RGGB Bayer colour "
                  "filter array over the white square in (a): every pixel records "
                  "one colour, and half of them are green. (c) The sensor itself "
                  "only sees one number per pixel. (d) Demosaicing interpolates the "
                  "two missing colours at each pixel to form the RGB image.", png=png)


def fig_resolution_pixels(ctx: BuildContext) -> Figure:
    rgb = load_image("lion", max_side=1024)
    h, w = rgb.shape[:2]
    crop = rgb[int(h * 0.1):int(h * 0.1) + 480, int(w * 0.3):int(w * 0.3) + 440]
    small = cv2.resize(crop, (220, 240), interpolation=cv2.INTER_AREA)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    axes[0].imshow(crop); axes[0].set_title("440 × 480 pixels")
    axes[1].imshow(upscale_nearest(small, 2)); axes[1].set_title("220 × 240 pixels (shown at the same size)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "resolution_pixels")
    return Figure("resolution_pixels",
                  "The same scene recorded with four times fewer pixels. Halving the "
                  "pixel count along each axis loses the finest detail such as "
                  "individual hairs. The pixel count bounds the resolution; the lens "
                  "decides whether that bound is reached.", png=png)


def fig_rgb_channels(ctx: BuildContext) -> Figure:
    rgb = load_image("sacre_coeur", max_side=480)
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6))
    axes[0, 0].imshow(rgb); axes[0, 0].set_title("RGB image")
    axes[1, 0].axis("off")
    for k, name in enumerate("RGB"):
        ch = rgb[:, :, k]
        axes[0, k + 1].imshow(ch, vmin=0, vmax=255)
        axes[0, k + 1].set_title(f"{name} channel (8-bit)")
        tint = np.zeros_like(rgb); tint[:, :, k] = ch
        axes[1, k + 1].imshow(tint)
        axes[1, k + 1].set_title(f"{name} channel as light")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "rgb_channels")
    return Figure("rgb_channels",
                  "A colour image is three 8-bit channels. Top row: each channel "
                  "shown as a greyscale intensity map. Bottom row: the same channel "
                  "displayed as the amount of red, green or blue light a display "
                  "pixel would emit.", png=png)


def fig_additive_mixing(ctx: BuildContext) -> Figure:
    n = 300
    yy, xx = np.mgrid[0:n, 0:n]
    r = n * 0.28
    centres = [(n * 0.40, n * 0.36), (n * 0.60, n * 0.36), (n * 0.50, n * 0.62)]
    img = np.zeros((n, n, 3))
    for k, (cx, cy) in enumerate(centres):
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2, k] = 1.0
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(img)
    for (cx, cy), lab in zip(centres, ["R", "G", "B"]):
        ax.text(cx, cy - r * 0.55, lab, color="white", ha="center", fontsize=14, weight="bold")
    ax.text(n * 0.50, n * 0.33, "yellow", color="black", ha="center", fontsize=9)
    ax.text(n * 0.62, n * 0.56, "cyan", color="black", ha="center", fontsize=9)
    ax.text(n * 0.38, n * 0.56, "magenta", color="black", ha="center", fontsize=9)
    ax.text(n * 0.50, n * 0.47, "white", color="black", ha="center", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Additive RGB colour model")
    png = save_mpl(fig, ctx, "additive_mixing")
    return Figure("additive_mixing",
                  "The additive RGB colour model: overlapping red, green and blue "
                  "light produces yellow, cyan and magenta, and all three at full "
                  "strength give white.", png=png)


def fig_rgb_colour_space(ctx: BuildContext) -> Figure:
    rgb = load_image("macaw", max_side=240)
    px = rgb.reshape(-1, 3)
    rng = np.random.default_rng(0)
    sel = px[rng.choice(len(px), 4000, replace=False)]
    cols = [f"rgb({r},{g},{b})" for r, g, b in sel]

    fig = plt.figure(figsize=(5, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2], c=sel / 255.0, s=4, depthshade=False)
    ax.set_xlim(0, 255); ax.set_ylim(0, 255); ax.set_zlim(0, 255)
    ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
    ax.set_title("Pixels of a photo inside the RGB cube")
    png = save_mpl(fig, ctx, "rgb_colour_space")

    fig2 = go.Figure(go.Scatter3d(
        x=sel[:, 0], y=sel[:, 1], z=sel[:, 2], mode="markers",
        marker=dict(size=2.5, color=cols), hovertemplate="R=%{x} G=%{y} B=%{z}<extra></extra>"))
    fig2.update_layout(scene=dict(
        xaxis=dict(title="R", range=[0, 255]), yaxis=dict(title="G", range=[0, 255]),
        zaxis=dict(title="B", range=[0, 255])),
        title="Drag to rotate: 4000 pixels sampled from the macaw photo")
    return Figure("rgb_colour_space",
                  "The RGB colour space as a cube with 256 levels along each axis. "
                  "Each dot is one pixel of the macaw photo placed at its (R, G, B) "
                  "coordinates and painted in its own colour. Even a colourful photo "
                  "occupies only a small part of the 16.8 million available colours.",
                  png=png, html=plotly_html(fig2, height=520), kind="plotly")


def fig_display_rgb(ctx: BuildContext) -> Figure:
    monitor = load_image("lcd_monitor", max_side=1024)
    sub = load_image("lcd_subpixels", max_side=800)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1.2, 1]))
    axes[0].imshow(monitor); axes[0].set_title("(a) LCD monitor")
    axes[1].imshow(sub); axes[1].set_title("(b) macro photograph of LCD sub-pixels")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "display_rgb")
    return Figure("display_rgb",
                  "Displays are RGB devices: each screen pixel is a triplet of red, "
                  "green and blue light-emitting sub-pixels whose intensities are set "
                  "by the three channel values. Viewed from a distance the triplet "
                  "blends into a single colour.", png=png)


def fig_camera_effects(ctx: BuildContext) -> Figure:
    heli = load_image("helicopter", max_side=800)
    noise = load_image("sensor_noise")
    macaw = macaw_crop(256)
    jpg, nbytes = jpeg_roundtrip(macaw, 6)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].imshow(heli); axes[0].set_title("(a) Rolling shutter: bent rotor blades")
    axes[1].imshow(noise); axes[1].set_title("(b) Sensor gain: ISO 100 (left) vs ISO 3200 (right)")
    axes[2].imshow(jpg); axes[2].set_title(f"(c) JPEG at quality 6 ({nbytes // 1024} kB): blocking")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "camera_effects")
    return Figure("camera_effects",
                  "Cameras are not perfect. (a) An electronic rolling shutter reads "
                  "the sensor line by line, so fast-moving rotor blades appear bent. "
                  "(b) Raising the sensor gain to shoot in low light amplifies noise. "
                  "(c) Aggressive JPEG compression leaves 8 × 8 blocking artefacts.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_sample_quantise(ctx), fig_bayer_demosaic(ctx), fig_resolution_pixels(ctx),
            fig_rgb_channels(ctx), fig_additive_mixing(ctx), fig_rgb_colour_space(ctx),
            fig_display_rgb(ctx), fig_camera_effects(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("1.1 Digital images: sampling and quantisation")
    st.markdown("A digital image is a **spatially sampled** and **quantised** "
                "record of the light reaching the sensor. Change the number of "
                "pixels and the number of bits per pixel and watch what is lost.")
    base = cat_square(256)
    col1, col2 = st.columns(2)
    n = col1.select_slider("Pixel grid (n × n)", options=GRID_SIZES, value=64)
    b = col2.select_slider("Bits per pixel", options=BIT_DEPTHS, value=8)
    im = quantise(resample_grid(base, n), b)
    c1, c2 = st.columns(2)
    c1.image(base, caption="Scene (256 × 256, 8-bit)", width="stretch")
    c2.image(upscale_nearest(im, 256 // n),
             caption=f"{n} × {n} pixels, {b}-bit ({2**b} grey levels)", width="stretch")
    st.caption(f"Storage: {n*n*b/8/1024:.1f} kB versus {256*256/1024:.0f} kB for the scene.")

    st.subheader("Bayer colour filter and demosaicing")
    rgb = macaw_crop(192)
    raw, cfa = make_bayer_mosaic(rgb)
    out = demosaic(raw)
    zoom = st.slider("Close-up size (pixels)", 4, 32, 8, step=4)
    c1, c2, c3 = st.columns(3)
    c1.image(rgb, caption="Scene", width="stretch")
    c2.image(upscale_nearest(cfa[:zoom, :zoom], 192 // zoom), caption="Colour filter array (close-up)",
             width="stretch")
    c3.image(out, caption="Demosaiced from one value per pixel", width="stretch")
    err = np.abs(out.astype(int) - rgb.astype(int)).mean()
    st.caption(f"Mean absolute error of the demosaiced image: {err:.1f} grey levels.")

    st.subheader("JPEG compression")
    q = st.slider("JPEG quality", 1, 100, 10)
    jpg, nbytes = jpeg_roundtrip(macaw_crop(256), q)
    c1, c2 = st.columns(2)
    c1.image(macaw_crop(256), caption="original (uncompressed 192 kB)", width="stretch")
    c2.image(jpg, caption=f"quality {q}: {nbytes / 1024:.1f} kB", width="stretch")
