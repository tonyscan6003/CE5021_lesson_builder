"""Lesson 1.7 — The 2D discrete Fourier transform.

Figures
    dft_basis_2d      2D basis functions (corrugations) for pairs of frequencies (k, l)
    spectrum_2d       an image, its raw spectrum and the centred (fftshift) spectrum
    leakage_2d        crosshatch pattern: leakage lines, reduced by a Hann window
    magnitude_phase   reconstruction from magnitude only and from phase only
    common_dfts       square, Gaussian and circle and their transforms
    freq_filter       ideal circular low/high-pass vs Gaussian filters in the frequency domain (interactive)
    periodic_noise    removing a sinusoidal interference pattern with a notch filter
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .._common import (BuildContext, Figure, C, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, load_image, to_uint8, norm01)

FIGURES = ["dft_basis_2d", "spectrum_2d", "leakage_2d", "magnitude_phase", "common_dfts",
           "freq_filter", "periodic_noise"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def log_spectrum(img: np.ndarray, shift: bool = True) -> np.ndarray:
    F = np.fft.fft2(img.astype(float))
    if shift:
        F = np.fft.fftshift(F)
    return np.log1p(np.abs(F))


def radial_mask(shape, radius, high=False, gaussian=False) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (yy - h / 2) ** 2 + (xx - w / 2) ** 2
    if gaussian:
        m = np.exp(-d2 / (2 * radius ** 2))
    else:
        m = (d2 <= radius ** 2).astype(float)
    return 1 - m if high else m


def filter_freq(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    F = np.fft.fftshift(np.fft.fft2(img.astype(float)))
    out = np.fft.ifft2(np.fft.ifftshift(F * mask)).real
    return np.clip(out, 0, 255)


def square_image(max_side=320):
    g = load_image("sacre_coeur", gray=True, max_side=max_side)
    h, w = g.shape
    s = min(h, w)
    return g[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_dft_basis_2d(ctx: BuildContext) -> Figure:
    M = 32
    yy, xx = np.mgrid[0:M, 0:M]
    fig, axes = plt.subplots(4, 4, figsize=(7.5, 7.8))
    for k in range(4):
        for l in range(4):
            b = np.cos(2 * np.pi * (k * xx + l * yy) / M)
            axes[l, k].imshow(b, cmap="gray", vmin=-1, vmax=1)
            axes[l, k].set_title(f"k = {k}, l = {l}", fontsize=9); axes[l, k].set_xticks([]); axes[l, k].set_yticks([])
    fig.suptitle("real part of the basis function e^{j2π(kx/M + ly/N)}", fontsize=10)
    png = save_mpl(fig, ctx, "dft_basis_2d")
    return Figure("dft_basis_2d",
                  "2D DFT basis functions. Each pair of frequencies (k, l) is a corrugation: "
                  "k sets the variation along x, l along y. With l = 0 the stripes are "
                  "vertical, with k = 0 horizontal, and other pairs give tilted stripes of "
                  "higher frequency. An image is reconstructed as a weighted sum of these.",
                  png=png)


def fig_spectrum_2d(ctx: BuildContext) -> Figure:
    g = square_image(320)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    axes[0].imshow(g, vmin=0, vmax=255); axes[0].set_title("(a) image")
    axes[1].imshow(log_spectrum(g, shift=False), cmap="magma"); axes[1].set_title("(b) log |F(k, l)|: (0,0) at top-left")
    axes[2].imshow(log_spectrum(g, shift=True), cmap="magma"); axes[2].set_title("(c) after fftshift: (0,0) at the centre")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "spectrum_2d")
    return Figure("spectrum_2d",
                  "The 2D spectrum of an image is the same size as the image. (b) As "
                  "computed, the zero frequency sits at the top-left corner and the four "
                  "quadrants mirror each other. (c) Swapping the quadrants (fftshift) puts "
                  "the low frequencies, where most of an image's energy lies, in the "
                  "centre, giving the familiar bright central spot. Magnitudes are shown on "
                  "a log scale.", png=png)


def fig_leakage_2d(ctx: BuildContext) -> Figure:
    n = 128
    yy, xx = np.mgrid[0:n, 0:n]
    cross = np.cos(2 * np.pi * (5.5 * xx + 3.5 * yy) / n) + np.cos(2 * np.pi * (-4.5 * xx + 6.5 * yy) / n)
    w = np.outer(np.hanning(n), np.hanning(n))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    axes[0].imshow(cross, cmap="gray"); axes[0].set_title("(a) crosshatch pattern")
    axes[1].imshow(log_spectrum(cross), cmap="magma"); axes[1].set_title("(b) spectrum: leakage lines")
    axes[2].imshow(cross * w, cmap="gray"); axes[2].set_title("(c) Hann-windowed pattern")
    axes[3].imshow(log_spectrum(cross * w), cmap="magma"); axes[3].set_title("(d) spectrum after windowing")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "leakage_2d")
    return Figure("leakage_2d",
                  "Spectral leakage in 2D. Two diagonal corrugations that do not fit a whole "
                  "number of cycles into the image (a) produce a spectrum (b) with horizontal "
                  "and vertical streaks radiating from each frequency. A Hann window (c) "
                  "tapers the image edges and isolates the two frequencies and the DC term "
                  "(d), at the cost of darkening the image borders.", png=png)


def fig_magnitude_phase(ctx: BuildContext) -> Figure:
    g = square_image(320).astype(float)
    F = np.fft.fft2(g)
    mag_only = np.fft.ifft2(np.abs(F)).real
    phase_only = np.fft.ifft2(np.exp(1j * np.angle(F))).real
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    axes[0].imshow(g, vmin=0, vmax=255); axes[0].set_title("(a) image")
    axes[1].imshow(np.fft.fftshift(np.log1p(np.abs(mag_only))), cmap="gray"); axes[1].set_title("(b) magnitude only (phase set to 0)")
    axes[2].imshow(norm01(np.clip(phase_only, np.percentile(phase_only, 1), np.percentile(phase_only, 99))), cmap="gray")
    axes[2].set_title("(c) phase only (magnitude set to 1)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "magnitude_phase")
    return Figure("magnitude_phase",
                  "Magnitude versus phase. Reconstructing from the magnitudes alone (b) gives "
                  "a symmetric blob with no trace of the scene; reconstructing from the "
                  "phases alone (c) preserves the edges and layout of the building. The "
                  "phase carries the structural information of an image.", png=png)


def fig_common_dfts(ctx: BuildContext) -> Figure:
    n = 128
    yy, xx = np.mgrid[0:n, 0:n]
    cx = cy = n // 2
    square = ((np.abs(xx - cx) < 6) & (np.abs(yy - cy) < 6)).astype(float)
    gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 4.0 ** 2))
    circle = (((xx - cx) ** 2 + (yy - cy) ** 2) < 7 ** 2).astype(float)
    fig = plt.figure(figsize=(12, 7.5))
    for j, (name, im) in enumerate([("square", square), ("Gaussian", gauss), ("circle", circle)]):
        F = np.abs(np.fft.fftshift(np.fft.fft2(im)))
        ax = fig.add_subplot(3, 3, j + 1); ax.imshow(im, cmap="gray"); ax.set_title(f"{name} (image domain)"); ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(3, 3, j + 4); ax.imshow(np.log1p(F), cmap="magma"); ax.set_title(f"|DFT| of the {name}"); ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(3, 3, j + 7, projection="3d")
        sl = slice(cx - 24, cx + 24)
        ax.plot_surface(xx[sl, sl], yy[sl, sl], F[sl, sl], cmap="viridis", linewidth=0)
        ax.set_title(f"|DFT| of the {name}, 3D"); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    png = save_mpl(fig, ctx, "common_dfts")
    return Figure("common_dfts",
                  "Common transforms. A small feature in the image domain gives a wide "
                  "response in the frequency domain. The sharp-edged square and circle "
                  "produce extensive ringing, the square along the k and l axes and the "
                  "circle as concentric rings (a Bessel function), whereas the Gaussian "
                  "transforms to another, wider Gaussian with no ringing.", png=png)


def fig_freq_filter(ctx: BuildContext) -> Figure:
    g = square_image(320)
    lp = filter_freq(g, radial_mask(g.shape, 20))
    hp = filter_freq(g, radial_mask(g.shape, 20, high=True))
    gw = filter_freq(g, radial_mask(g.shape, 60, gaussian=True))
    gn = filter_freq(g, radial_mask(g.shape, 12, gaussian=True))
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    panels = [("image", g), ("ideal low-pass, r = 20", lp), ("ideal high-pass, r = 20", hp),
              ("log spectrum with r = 20 circle", None),
              ("Gaussian LP σ = 60 (light)", gw), ("Gaussian LP σ = 12 (heavy)", gn),
              ("ideal LP: ringing (crop)", lp[100:200, 100:200]), ("Gaussian LP: no ringing (crop)", gn[100:200, 100:200])]
    for ax, (t, im) in zip(axes.ravel(), panels):
        if im is None:
            ax.imshow(log_spectrum(g), cmap="magma")
            ax.add_patch(plt.Circle((g.shape[1] / 2, g.shape[0] / 2), 20, fill=False, ec=C["green"], lw=2))
        elif t.startswith("ideal high"):
            ax.imshow(np.abs(im - np.mean(im)), vmin=0, vmax=60)
        else:
            ax.imshow(im, vmin=0, vmax=255)
        ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "freq_filter")

    radii = [5, 10, 20, 40, 80, 160]
    fig2 = make_subplots(rows=1, cols=3, subplot_titles=("ideal low-pass", "ideal high-pass (magnitude)", "Gaussian low-pass"))
    frames = []
    for r in radii:
        a = filter_freq(g, radial_mask(g.shape, r))
        b = np.abs(filter_freq(g, radial_mask(g.shape, r, high=True)) - 128)
        c = filter_freq(g, radial_mask(g.shape, r, gaussian=True))
        frames.append(go.Frame(name=str(r), data=[go.Image(source=to_data_uri(to_uint8(a), "jpg")),
                                                  go.Image(source=to_data_uri(to_uint8(np.clip(b * 3, 0, 255)), "jpg")),
                                                  go.Image(source=to_data_uri(to_uint8(c), "jpg"))], traces=[0, 1, 2]))
    for i, tr in enumerate(frames[2].data):
        fig2.add_trace(tr, row=1, col=i + 1)
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(radii, "radius / σ = "), title="Frequency-domain filtering")
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    return Figure("freq_filter",
                  "Filtering in the frequency domain. Multiplying the centred spectrum by a "
                  "disc keeps the low frequencies (blur) and by the inverse disc keeps the "
                  "high frequencies (edges), but the sharp disc edge produces ringing in the "
                  "image, since the inverse transform of a circle is a Bessel function. A "
                  "Gaussian mask blurs without ringing: a wide Gaussian removes little, a "
                  "narrow one removes a lot.",
                  png=png, html=plotly_html(fig2, height=380), kind="plotly")


def fig_periodic_noise(ctx: BuildContext) -> Figure:
    g = square_image(320).astype(float)
    n = g.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    fx, fy = 22, 14
    noise = 40 * np.sin(2 * np.pi * (fx * xx + fy * yy) / n)
    noisy = np.clip(g + noise, 0, 255)
    F = np.fft.fftshift(np.fft.fft2(noisy))
    mask = np.ones_like(g)
    for sx, sy in [(fx, fy), (-fx, -fy)]:
        mask[((xx - n / 2 - sx) ** 2 + (yy - n / 2 - sy) ** 2) < 4 ** 2] = 0
    clean = np.clip(np.fft.ifft2(np.fft.ifftshift(F * mask)).real, 0, 255)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    axes[0].imshow(noisy, vmin=0, vmax=255); axes[0].set_title("(a) image with periodic interference")
    axes[1].imshow(np.log1p(np.abs(F)), cmap="magma"); axes[1].set_title("(b) spectrum: two bright peaks")
    for sx, sy in [(fx, fy), (-fx, -fy)]:
        axes[1].add_patch(plt.Circle((n / 2 + sx, n / 2 + sy), 8, fill=False, ec=C["green"], lw=1.5))
    axes[2].imshow(np.log1p(np.abs(F * mask)), cmap="magma"); axes[2].set_title("(c) notch filter applied")
    axes[3].imshow(clean, vmin=0, vmax=255); axes[3].set_title("(d) inverse DFT")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "periodic_noise")
    return Figure("periodic_noise",
                  "Removing periodic noise. A sinusoidal interference pattern (a) is hard to "
                  "remove by convolution but appears in the spectrum as a single pair of "
                  "peaks (b). Zeroing those peaks with a notch filter (c) and inverting (d) "
                  "removes the stripes while leaving the rest of the image intact.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_dft_basis_2d(ctx), fig_spectrum_2d(ctx), fig_leakage_2d(ctx), fig_magnitude_phase(ctx),
            fig_common_dfts(ctx), fig_freq_filter(ctx), fig_periodic_noise(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("1.7 The 2D DFT")
    tab1, tab2, tab3 = st.tabs(["Basis functions", "Frequency-domain filtering", "Notch filter"])
    with tab1:
        c1, c2 = st.columns(2)
        k = c1.slider("k (x frequency)", 0, 8, 2); l = c2.slider("l (y frequency)", 0, 8, 1)
        M = 64; yy, xx = np.mgrid[0:M, 0:M]
        st.image(to_uint8((np.cos(2 * np.pi * (k * xx + l * yy) / M) + 1) / 2), caption=f"k = {k}, l = {l}", width=320)
    with tab2:
        g = square_image(320)
        kind = st.radio("filter", ["ideal low-pass", "ideal high-pass", "Gaussian low-pass"], horizontal=True)
        r = st.slider("radius / σ (pixels)", 2, 160, 20)
        mask = radial_mask(g.shape, r, high=(kind == "ideal high-pass"), gaussian=(kind == "Gaussian low-pass"))
        out = filter_freq(g, mask)
        if kind == "ideal high-pass":
            out = np.clip(np.abs(out - out.mean()) * 3, 0, 255)
        c1, c2, c3 = st.columns(3)
        c1.image(g, caption="image", width="stretch")
        c2.image(to_uint8(norm01(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g.astype(float)))) * mask))), caption="spectrum × mask", width="stretch")
        c3.image(to_uint8(out), caption=kind, width="stretch")
    with tab3:
        g = square_image(320).astype(float); n = g.shape[0]
        yy, xx = np.mgrid[0:n, 0:n]
        fx = st.slider("interference frequency x", 5, 60, 22); fy = st.slider("interference frequency y", 0, 60, 14)
        rad = st.slider("notch radius", 1, 12, 4)
        noisy = np.clip(g + 40 * np.sin(2 * np.pi * (fx * xx + fy * yy) / n), 0, 255)
        F = np.fft.fftshift(np.fft.fft2(noisy)); mask = np.ones_like(g)
        for sx, sy in [(fx, fy), (-fx, -fy)]:
            mask[((xx - n / 2 - sx) ** 2 + (yy - n / 2 - sy) ** 2) < rad ** 2] = 0
        clean = np.clip(np.fft.ifft2(np.fft.ifftshift(F * mask)).real, 0, 255)
        c1, c2, c3 = st.columns(3)
        c1.image(to_uint8(noisy), caption="noisy", width="stretch")
        c2.image(to_uint8(norm01(np.log1p(np.abs(F * mask)))), caption="notched spectrum", width="stretch")
        c3.image(to_uint8(clean), caption="restored", width="stretch")
