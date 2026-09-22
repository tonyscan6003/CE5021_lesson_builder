"""Lesson 1.3 — Image sampling, aliasing and interpolation.

Figures
    aliasing_1d        a sine above fs/2 is sampled and read back as a low frequency (interactive)
    image_frequency    low- and high-frequency image content with intensity profiles
    image_aliasing     moiré from downsampling without a low-pass filter (interactive)
    digital_spectrum   one set of samples represents many frequencies (image frequencies)
    interp_kernels     replication, linear and sinc interpolation kernels and their spectra
    signal_resampling  interpolate then resample; residual image frequencies alias
    interp_methods     nearest, bilinear, bicubic and Lanczos resizing of an image and an edge (interactive)
    bilinear_area      the area-weighting view of bilinear interpolation
"""
from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .._common import (BuildContext, Figure, C, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, load_image,
                       upscale_nearest)
from .._vendor import bi_lin

FIGURES = ["aliasing_1d", "image_frequency", "image_aliasing", "moire_photo", "digital_spectrum",
           "interp_kernels", "signal_resampling", "interp_methods", "bilinear_area"]

CV_METHODS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "Lanczos (windowed sinc)": cv2.INTER_LANCZOS4,
}


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def alias_frequency(f: float, fs: float = 1.0) -> float:
    """Frequency that samples of a sine at f (cycles per sample period) appear to have."""
    r = f % fs
    return min(r, fs - r)


def aliasing_curves(f1: float, fs: float = 1.0, n_samples: int = 16, n_fine: int = 2000):
    t = np.linspace(0, n_samples, n_fine)
    ts = np.arange(n_samples + 1)
    fa = alias_frequency(f1, fs)
    x_true = np.cos(2 * np.pi * f1 * t)
    x_alias = np.cos(2 * np.pi * fa * t)
    samples = np.cos(2 * np.pi * f1 * ts)
    return t, ts, x_true, x_alias, samples, fa


def kernel_family(up: int = 16, half_width: int = 4):
    """Box, triangle and windowed sinc kernels on an `up`-times finer grid."""
    u = np.arange(-half_width * up, half_width * up + 1) / up
    box = ((u >= -0.5) & (u < 0.5)).astype(float)
    tri = np.clip(1 - np.abs(u), 0, None)
    sinc = np.sinc(u) * np.hanning(len(u))
    return u, {"replication (box)": box, "linear (triangle)": tri, "sinc": sinc}


def reconstruct(samples: np.ndarray, kernel: np.ndarray, up: int = 16) -> np.ndarray:
    """Zero-stuff the samples by `up` and convolve with the interpolation kernel."""
    z = np.zeros(len(samples) * up)
    z[::up] = samples
    return np.convolve(z, kernel, mode="same")


def stripe_pattern(n: int = 256, period: float = 5.0, angle_deg: float = 20.0) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    a = np.deg2rad(angle_deg)
    s = 0.5 + 0.5 * np.cos(2 * np.pi * (xx * np.cos(a) + yy * np.sin(a)) / period)
    return (s * 255).astype(np.uint8)


def downsample(img: np.ndarray, factor: int, prefilter: bool) -> np.ndarray:
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if prefilter else cv2.INTER_NEAREST
    return cv2.resize(img, (w // factor, h // factor), interpolation=interp)


def edge_profile(method: int, up: int = 8):
    """Upscale a 1-pixel-wide step edge and return the interpolated row profile."""
    row = np.zeros((4, 16), np.float32)
    row[:, 8:] = 1.0
    big = cv2.resize(row, (16 * up, 4 * up), interpolation=method)
    return big[2 * up]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_aliasing_1d(ctx: BuildContext) -> Figure:
    f1 = 0.9
    t, ts, x_true, x_alias, samples, fa = aliasing_curves(f1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    ax = axes[0]
    ax.plot(t, x_true, color=C["red"], lw=1, label=f"signal, f = {f1:.1f} fs")
    ax.plot(t, x_alias, color=C["blue"], lw=2, label=f"alias, f = {fa:.1f} fs")
    ax.plot(ts, samples, "ko", ms=5, label="samples")
    ax.set_title("(a) Sampling a sine above fs/2"); ax.set_xlabel("time (sample periods)")
    ax.set_ylim(-1.15, 1.75)
    ax.legend(fontsize=8, loc="upper center", ncol=3, columnspacing=0.8, handlelength=1.5)
    ax = axes[1]
    ax.axvline(0.5, color=C["grey"], ls="--"); ax.text(0.51, 1.05, "fs/2", color=C["grey"])
    ax.stem([f1], [1], linefmt=C["red"], markerfmt="o", basefmt=" ")
    ax.stem([fa], [1], linefmt=C["blue"], markerfmt="o", basefmt=" ")
    ax.annotate("", xy=(fa, 0.6), xytext=(f1, 0.6),
                arrowprops=dict(arrowstyle="->", color=C["grey"], connectionstyle="arc3,rad=0.3"))
    ax.text(0.5, 0.65, "mirrored about fs/2", ha="center", fontsize=8, color=C["grey"])
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.2); ax.set_xlabel("frequency (× fs)")
    ax.set_title("(b) Aliasing in the frequency domain")
    ax = axes[2]
    f = np.linspace(0, 1, 400)
    lp = 1 / np.sqrt(1 + (f / 0.4) ** 8)
    ax.plot(f, lp, color=C["green"], label="low-pass filter")
    ax.axvline(0.5, color=C["grey"], ls="--")
    ax.stem([f1], [1], linefmt=C["red"] + "40", markerfmt=" ", basefmt=" ")
    ax.stem([f1], [np.interp(f1, f, lp)], linefmt=C["red"], markerfmt="o", basefmt=" ")
    ax.stem([fa], [np.interp(f1, f, lp)], linefmt=C["blue"], markerfmt="o", basefmt=" ")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.2); ax.set_xlabel("frequency (× fs)")
    ax.set_title("(c) Low-pass filtering before sampling"); ax.legend(fontsize=8)
    png = save_mpl(fig, ctx, "aliasing_1d")

    # interactive: slider over f1
    fs_list = [round(v, 2) for v in np.arange(0.05, 0.96, 0.05)]
    fig2 = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                         subplot_titles=("time domain", "frequency domain"))
    frames = []
    for f1_ in fs_list:
        t, ts, x_true, x_alias, samples, fa = aliasing_curves(f1_, n_fine=400)
        frames.append(go.Frame(name=str(f1_), data=[
            go.Scatter(x=t, y=x_true, line=dict(color=C["red"], width=1), name="signal"),
            go.Scatter(x=t, y=x_alias, line=dict(color=C["blue"], width=2.5), name="alias"),
            go.Scatter(x=ts, y=samples, mode="markers", marker=dict(color="black", size=8), name="samples"),
            go.Bar(x=[f1_], y=[1], width=0.02, marker_color=C["red"], name="signal f"),
            go.Bar(x=[fa], y=[1], width=0.02, marker_color=C["blue"], name="alias f"),
        ], traces=[0, 1, 2, 3, 4],
            layout=go.Layout(title=f"f = {f1_:.2f} fs  →  samples look like f = {fa:.2f} fs")))
    for tr in frames[0].data[:3]:
        fig2.add_trace(tr, row=1, col=1)
    for tr in frames[0].data[3:]:
        fig2.add_trace(tr, row=1, col=2)
    fig2.frames = frames
    fig2.add_vline(x=0.5, line_dash="dash", line_color=C["grey"], row=1, col=2)
    fig2.update_layout(**slider_frames_layout(fs_list, "signal frequency (× fs) = "),
                       title=frames[0].layout.title.text, barmode="overlay")
    fig2.update_xaxes(title_text="time (sample periods)", row=1, col=1)
    fig2.update_xaxes(title_text="frequency (× fs)", range=[0, 1], row=1, col=2)
    fig2.update_yaxes(range=[0, 1.2], row=1, col=2)
    return Figure("aliasing_1d",
                  "Aliasing. (a) A sinusoid at 0.9 of the sample rate is sampled once "
                  "per period marker; the samples are indistinguishable from a sinusoid "
                  "at 0.1 fs. (b) Frequencies above fs/2 fold back, mirrored about "
                  "fs/2. (c) A low-pass filter applied before sampling attenuates the "
                  "high frequency so that the alias it produces is small.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def _crop_frac(img, y0, x0, h, w):
    H, W = img.shape[:2]
    return img[int(y0 * H):int(y0 * H) + h, int(x0 * W):int(x0 * W) + w]


def fig_image_frequency(ctx: BuildContext) -> Figure:
    sail = load_image("sail", gray=True, max_side=1280)
    sky = _crop_frac(sail, 0.05, 0.6, 200, 260)
    zebra = _crop_frac(load_image("zebra", gray=True, max_side=1280), 0.30, 0.45, 200, 260)
    brick = _crop_frac(load_image("brickwork", gray=True, max_side=1280), 0.14, 0.25, 200, 260)
    panels = [("(a) sky: low frequency", sky), ("(b) zebra stripes: high frequency", zebra),
              ("(c) brickwork: high frequency", brick)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), gridspec_kw=dict(height_ratios=[2, 1]))
    for c_, (title, im) in enumerate(panels):
        r = im.shape[0] // 2
        axes[0, c_].imshow(im, vmin=0, vmax=255); axes[0, c_].set_title(title)
        axes[0, c_].axhline(r, color=C["red"], lw=1)
        axes[0, c_].set_xticks([]); axes[0, c_].set_yticks([])
        axes[1, c_].plot(im[r].astype(float), color=C["red"], lw=0.9)
        axes[1, c_].set_ylim(0, 255); axes[1, c_].set_xlabel("column"); axes[1, c_].set_ylabel("intensity")
    png = save_mpl(fig, ctx, "image_frequency")
    return Figure("image_frequency",
                  "Frequency in an image is the rate of change of intensity with "
                  "position. Regions of near-constant intensity (sky) contain only low "
                  "spatial frequencies; edges and repeating patterns (stripes, bricks) "
                  "contain high spatial frequencies. Each lower panel plots the "
                  "intensity along the orange row.", png=png)


def brick_square(size: int = 512) -> np.ndarray:
    """Square crop of the brick courses only (no railings) from the Kilkenny wall photo."""
    g = load_image("brickwork", gray=True, max_side=1280)
    h, w = g.shape
    y0, x0, s = int(0.12 * h), int(0.25 * w), int(0.42 * h)
    return cv2.resize(g[y0:y0 + s, x0:x0 + s], (size, size), interpolation=cv2.INTER_AREA)


def fig_image_aliasing(ctx: BuildContext) -> Figure:
    brick = brick_square(512)
    stripes = stripe_pattern(256, period=5.0, angle_deg=20)
    f = 4
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for r, (name, im) in enumerate([("brickwork", brick), ("stripes, period 5 px", stripes)]):
        nn = downsample(im, f, prefilter=False)
        ar = downsample(im, f, prefilter=True)
        axes[r, 0].imshow(im, vmin=0, vmax=255); axes[r, 0].set_title(f"{name}: {im.shape[1]} px wide")
        axes[r, 1].imshow(upscale_nearest(nn, f), vmin=0, vmax=255)
        axes[r, 1].set_title(f"÷{f} by dropping pixels (aliased)")
        axes[r, 2].imshow(upscale_nearest(ar, f), vmin=0, vmax=255)
        axes[r, 2].set_title(f"÷{f} with low-pass averaging first")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "image_aliasing")

    factors = [1, 2, 3, 4, 5, 6, 8]
    small = brick[128:384, 128:384]      # 256 px crop keeps the embedded page small
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("drop pixels (nearest)", "average then sample (area)"))
    frames = []
    for fct in factors:
        a = upscale_nearest(downsample(small, fct, False), fct)
        b = upscale_nearest(downsample(small, fct, True), fct)
        frames.append(go.Frame(name=str(fct), data=[go.Image(source=to_data_uri(a, "jpg")),
                                                    go.Image(source=to_data_uri(b, "jpg"))], traces=[0, 1]))
    fig2.add_trace(frames[0].data[0], row=1, col=1); fig2.add_trace(frames[0].data[1], row=1, col=2)
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(factors, "downsample factor ÷"),
                       title="Reducing the brick image without and with a pre-filter")
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    return Figure("image_aliasing",
                  "Aliasing in images. Reducing an image by keeping every fourth pixel "
                  "samples the fine pattern too coarsely, so the bricks and stripes "
                  "turn into wide, false moiré ripples (middle). Averaging over the "
                  "pixels being merged acts as the low-pass filter and removes the "
                  "ripples (right). The camera's optical low-pass filter plays the "
                  "same role for the sensor.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def fig_moire_photo(ctx: BuildContext) -> Figure:
    """The same brick wall at two sampling rates: moiré appears only in the
    undersampled version, which is the point of the figure."""
    full = load_image("moire_bricks")
    small = load_image("moire_bricks_small")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4),
                             gridspec_kw=dict(width_ratios=[1, 1], wspace=0.06))
    axes[0].imshow(full)
    axes[0].set_title(f"(a) densely sampled: {full.shape[1]} × {full.shape[0]}\n"
                      "every brick course is resolved", fontsize=9)
    # shown at the same physical size, so the comparison is of sampling rate,
    # not of how large the picture is on the page
    axes[1].imshow(small, interpolation="nearest")
    axes[1].set_title(f"(b) sparsely sampled: {small.shape[1]} × {small.shape[0]}\n"
                      "broad bands appear that are not on the wall", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "moire_photo")
    return Figure("moire_photo",
                  "Aliasing in a real photograph. The left image resolves every "
                  "course of brickwork. The right image is the same wall sampled on a "
                  "much coarser grid, too coarse for the brick spacing, and the "
                  "aliased detail reappears as broad curved bands that are not on the "
                  "wall at all. The bands are in the pixel data, not in the display: "
                  "enlarging the right image does not remove them.", png=png)


def fig_digital_spectrum(ctx: BuildContext) -> Figure:
    f_lo = 0.15
    t = np.linspace(0, 12, 2000)
    ts = np.arange(13)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
    ax = axes[0]
    ax.plot(t, np.cos(2 * np.pi * f_lo * t), color=C["blue"], lw=2, label=f"f = {f_lo} fs")
    ax.plot(t, np.cos(2 * np.pi * (1 - f_lo) * t), color=C["red"], lw=0.8, label=f"f = {1-f_lo:.2f} fs")
    ax.plot(t, np.cos(2 * np.pi * (1 + f_lo) * t), color=C["orange"], lw=0.8, label=f"f = {1+f_lo:.2f} fs")
    ax.plot(ts, np.cos(2 * np.pi * f_lo * ts), "ko", ms=5, label="samples")
    ax.set_title("(a) One set of samples, many sinusoids"); ax.set_xlabel("time (sample periods)")
    ax.set_ylim(-1.15, 1.75)
    ax.legend(fontsize=8, ncol=4, loc="upper center", columnspacing=0.8, handlelength=1.5)
    ax = axes[1]
    for m in range(0, 3):
        for sgn in (+1, -1):
            f = m + sgn * f_lo
            if 0 <= f <= 2.5:
                ax.stem([f], [1], linefmt=C["blue"] if m == 0 else C["grey"], markerfmt="o", basefmt=" ")
    for m in (0.5, 1.0, 1.5, 2.0):
        ax.axvline(m, color=C["light"], ls="--")
    ax.text(f_lo, 1.08, "signal", ha="center", fontsize=8, color=C["blue"])
    ax.text(1.0, 1.08, "image frequencies", ha="center", fontsize=8, color=C["grey"])
    ax.text(2.0, 1.08, "image frequencies", ha="center", fontsize=8, color=C["grey"])
    ax.set_xlim(0, 2.5); ax.set_ylim(0, 1.25); ax.set_xlabel("frequency (× fs)")
    ax.set_title("(b) Spectrum of the sampled signal repeats every fs")
    png = save_mpl(fig, ctx, "digital_spectrum")
    return Figure("digital_spectrum",
                  "A consequence of sampling. (a) The samples of a 0.15 fs sinusoid "
                  "lie exactly on sinusoids at 0.85 fs, 1.15 fs and so on. (b) The "
                  "spectrum of a sampled signal is therefore periodic in fs: copies of "
                  "the signal frequency, called image frequencies, appear around every "
                  "multiple of the sample rate.", png=png)


def fig_interp_kernels(ctx: BuildContext) -> Figure:
    up = 16
    u, kernels = kernel_family(up=up, half_width=4)
    n_s = 12
    ts = np.arange(n_s)
    samples = np.cos(2 * np.pi * 0.12 * ts + 0.6)
    tf = np.arange(n_s * up) / up
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.6))
    for c_, (name, k) in enumerate(kernels.items()):
        ax = axes[0, c_]
        ax.plot(u, k, color=C["blue"]); ax.set_title(f"{name} kernel")
        ax.set_xlabel("sample periods"); ax.axhline(0, color="k", lw=0.5)
        ax = axes[1, c_]
        rec = reconstruct(samples, k, up)
        ax.plot(tf, rec, color=C["orange"], label="reconstruction")
        ax.plot(ts, samples, "ko", ms=4, label="samples")
        ax.set_title("reconstructed signal"); ax.set_xlabel("time (sample periods)")
        ax.legend(fontsize=8, loc="upper right")
        ax = axes[2, c_]
        nfft = 4096
        H = np.abs(np.fft.rfft(k, nfft)); H /= H.max()
        f = np.fft.rfftfreq(nfft, d=1 / up)
        ax.plot(f, H, color=C["green"])
        for m in (0.5, 1.0, 2.0):
            ax.axvline(m, color=C["light"], ls="--")
        ax.text(0.5, 1.02, "fs/2", fontsize=8, color=C["grey"], ha="center")
        ax.text(1.0, 1.02, "fs", fontsize=8, color=C["grey"], ha="center")
        ax.text(2.0, 1.02, "2fs", fontsize=8, color=C["grey"], ha="center")
        ax.set_xlim(0, 2.5); ax.set_ylim(0, 1.15)
        ax.set_title("frequency response"); ax.set_xlabel("frequency (× fs)")
    png = save_mpl(fig, ctx, "interp_kernels")
    return Figure("interp_kernels",
                  "Interpolation as convolution. Reconstructing a continuous signal "
                  "from samples means convolving the samples with an interpolation "
                  "kernel (top). Replication holds each value, linear interpolation "
                  "joins them with straight lines, the sinc reproduces the sinusoid "
                  "exactly (middle). In the frequency domain (bottom) only the sinc "
                  "cuts off at fs/2; the box and triangle let the image frequencies "
                  "around fs and 2fs partially through, which shows up as steps and "
                  "corners in the reconstruction.", png=png)


def fig_signal_resampling(ctx: BuildContext) -> Figure:
    up = 16
    u, kernels = kernel_family(up=up, half_width=4)
    n_s = 16
    ts = np.arange(n_s)
    f_sig = 0.22
    samples = np.cos(2 * np.pi * f_sig * ts)
    tf = np.arange(n_s * up) / up
    new_rate = 0.6          # resample at 0.6 fs
    t_new = np.arange(0, n_s, 1 / new_rate)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (name, k) in zip(axes[:2], list(kernels.items())[0::2]):
        rec = reconstruct(samples, k, up)
        new = np.interp(t_new, tf, rec)
        ax.plot(tf, rec, color=C["orange"], lw=1, label=f"{name.split(' ')[0]} reconstruction")
        ax.plot(ts, samples, "ko", ms=4, label="original samples")
        ax.plot(t_new, new, "s", color=C["blue"], ms=5, label=f"new samples at {new_rate} fs")
        ax.set_title(f"resampling after {name.split(' ')[0]} interpolation")
        ax.set_xlabel("time (original sample periods)"); ax.legend(fontsize=7, loc="lower left")
    ax = axes[2]
    truth = np.cos(2 * np.pi * f_sig * t_new)
    for name, k, col, off in [("box", kernels["replication (box)"], C["orange"], -0.15),
                              ("sinc", kernels["sinc"], C["green"], 0.15)]:
        rec = reconstruct(samples, k, up)
        new = np.interp(t_new, tf, rec)
        ax.bar(t_new + off, new - truth, width=0.3, color=col, label=f"after {name} interpolation")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time (original sample periods)"); ax.set_ylabel("new sample − true value")
    ax.set_title("error of the new samples"); ax.legend(fontsize=7)
    png = save_mpl(fig, ctx, "signal_resampling")
    return Figure("signal_resampling",
                  "Changing the sample rate: interpolate, then take samples at the "
                  "new rate. With replication (left) the reconstruction still contains "
                  "image-frequency content, so the new samples (squares) miss the true "
                  "sinusoid; the right panel shows their errors. With sinc "
                  "interpolation (middle) the new samples lie on the true curve and the "
                  "errors are negligible. Reducing the sample rate is the more "
                  "dangerous direction.", png=png)


def detail_crop(size: int = 32) -> np.ndarray:
    """A high-contrast `size` x `size` crop from the Sphinx head (highest local std)."""
    g = load_image("sphinx", gray=True, max_side=640)
    h, w = g.shape
    best, best_s = None, -1.0
    for y in range(int(0.10 * h), int(0.45 * h), size // 2):
        for x in range(int(0.30 * w), int(0.60 * w), size // 2):
            s = g[y:y + size, x:x + size].std()
            if s > best_s:
                best, best_s = (y, x), s
    y, x = best
    return g[y:y + size, x:x + size]


def fig_interp_methods(ctx: BuildContext) -> Figure:
    crop = detail_crop(32)
    up = 8
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2), gridspec_kw=dict(height_ratios=[2, 1]))
    for c_, (name, m) in enumerate(CV_METHODS.items()):
        big = cv2.resize(crop, None, fx=up, fy=up, interpolation=m)
        axes[0, c_].imshow(big, vmin=0, vmax=255); axes[0, c_].set_title(f"{name}, ×{up}")
        axes[0, c_].set_xticks([]); axes[0, c_].set_yticks([])
        prof = edge_profile(m, up)
        axes[1, c_].plot(np.arange(len(prof)) / up, prof, color=C["blue"])
        axes[1, c_].set_ylim(-0.25, 1.25); axes[1, c_].set_xlim(4, 12)
        axes[1, c_].set_title("step edge enlarged ×8", fontsize=9); axes[1, c_].set_xlabel("original pixels")
    axes[1, 3].annotate("ringing (Gibbs)", xy=(7.6, -0.08), xytext=(4.6, -0.2), fontsize=8,
                        arrowprops=dict(arrowstyle="->", color=C["red"]), color=C["red"])
    png = save_mpl(fig, ctx, "interp_methods")

    fig2 = go.Figure()
    for i, (name, m) in enumerate(CV_METHODS.items()):
        big = cv2.resize(crop, None, fx=up, fy=up, interpolation=m)
        fig2.add_trace(go.Image(source=to_data_uri(big), visible=(i == 0), name=name))
    buttons = [dict(label=name, method="update",
                    args=[{"visible": [j == i for j in range(len(CV_METHODS))]}, {"title": f"{name}, ×{up}"}])
               for i, name in enumerate(CV_METHODS)]
    fig2.update_layout(updatemenus=[dict(type="buttons", direction="right", buttons=buttons,
                                         x=0.5, xanchor="center", y=1.12, yanchor="top")],
                       title=f"nearest, ×{up}")
    fig2.update_xaxes(visible=False); fig2.update_yaxes(visible=False)
    return Figure("interp_methods",
                  "Enlarging a 32 × 32 crop eight times with four interpolation "
                  "methods. Nearest-neighbour replicates pixels and looks blocky; "
                  "bilinear uses 4 neighbours, bicubic 16, and the Lanczos filter "
                  "is a truncated sinc. The lower row enlarges a single step edge: the "
                  "truncated sinc overshoots on both sides of the edge, the Gibbs "
                  "phenomenon, visible as faint ripples next to sharp edges.",
                  png=png, html=plotly_html(fig2, height=470), kind="plotly")


def fig_bilinear_area(ctx: BuildContext) -> Figure:
    params = {"xi": 1.4, "yj": 0.6, "I00": 10, "I20": 20, "I02": 5, "I22": 15}
    ctx.fig_dir.mkdir(parents=True, exist_ok=True)
    left = ctx.fig_dir / "_bilinear_grid_tmp.png"
    bi_lin._make_plot(params, shade=True, save_path=str(left))
    grid_img = plt.imread(left)
    left.unlink()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw=dict(width_ratios=[1, 1.1]))
    axes[0].imshow(grid_img); axes[0].axis("off")
    axes[0].set_title("(a) one new pixel inside a 2 × 2 cell of old pixels")
    ax = axes[1]
    n_old, factor = 5, 2.6
    for r in range(n_old):
        for c_ in range(n_old):
            ax.plot(c_, r, "o", color=C["grey"], ms=9)
    new = np.arange(0, n_old - 1 + 1e-9, 1 / factor)
    for r in new:
        for c_ in new:
            ax.plot(c_, r, "o", mfc="none", mec="crimson", ms=6, mew=1.3)
    ax.set_xlim(-0.5, n_old - 0.5); ax.set_ylim(n_old - 0.5, -0.5); ax.set_aspect("equal")
    ax.set_xticks(range(n_old)); ax.set_yticks(range(n_old))
    ax.set_title(f"(b) resizing by {factor}×: new samples (red) on the old grid (grey)")
    ax.grid(True, color=C["light"])
    png = save_mpl(fig, ctx, "bilinear_area")
    c = bi_lin.compute(params)
    return Figure("bilinear_area",
                  "Bilinear interpolation. (a) The new pixel (triangle) takes a weighted "
                  "sum of its four neighbours; each weight is the area of the rectangle "
                  "diagonally opposite that neighbour, so the nearest pixel gets the "
                  f"largest share. Here I = {c['M_interp']:.2f}. (b) When an image is "
                  "resized, every new sample position falls inside some cell of the "
                  "old grid and is interpolated in the same way.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_aliasing_1d(ctx), fig_image_frequency(ctx), fig_image_aliasing(ctx),
            fig_moire_photo(ctx), fig_digital_spectrum(ctx), fig_interp_kernels(ctx), fig_signal_resampling(ctx),
            fig_interp_methods(ctx), fig_bilinear_area(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("1.3 Sampling, aliasing and interpolation")
    tab1, tab2, tab3, tab4 = st.tabs(["1D aliasing", "Image aliasing", "Interpolation methods",
                                      "Bilinear calculator"])
    with tab1:
        f1 = st.slider("signal frequency (× sample rate)", 0.02, 1.5, 0.9, 0.02)
        t, ts, x_true, x_alias, samples, fa = aliasing_curves(f1)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, x_true, color=C["red"], lw=1, label=f"signal f = {f1:.2f} fs")
        ax.plot(t, x_alias, color=C["blue"], lw=2, label=f"what the samples show: f = {fa:.2f} fs")
        ax.plot(ts, samples, "ko", label="samples"); ax.legend(fontsize=8); ax.set_xlabel("sample periods")
        st.pyplot(fig)
        st.markdown("Above **0.5 fs** the samples describe a *lower* frequency mirrored "
                    "about fs/2; above fs the pattern repeats.")
    with tab2:
        src = st.radio("image", ["brickwork", "stripes"], horizontal=True)
        im = brick_square(512) if src == "brickwork" else stripe_pattern(256, 5.0, 20)
        fct = st.slider("downsample factor", 1, 8, 4)
        c1, c2 = st.columns(2)
        c1.image(upscale_nearest(downsample(im, fct, False), fct), caption="drop pixels (nearest)",
                 width="stretch")
        c2.image(upscale_nearest(downsample(im, fct, True), fct), caption="average first (area)",
                 width="stretch")
    with tab3:
        crop = detail_crop(32)
        up = st.slider("enlargement factor", 2, 12, 8)
        cols = st.columns(len(CV_METHODS))
        for col, (name, m) in zip(cols, CV_METHODS.items()):
            col.image(cv2.resize(crop, None, fx=up, fy=up, interpolation=m), caption=name,
                      width="stretch")
        fig, ax = plt.subplots(figsize=(8, 2.6))
        for name, m in CV_METHODS.items():
            prof = edge_profile(m, up)
            ax.plot(np.arange(len(prof)) / up, prof, label=name)
        ax.set_xlim(4, 12); ax.legend(fontsize=8); ax.set_title("step edge profile after enlargement")
        st.pyplot(fig)
    with tab4:
        c1, c2 = st.columns(2)
        xi = c1.slider("x position in the 2×2 cell", 0.0, 2.0, 1.4, 0.1)
        yj = c2.slider("y position (downwards)", 0.0, 2.0, 0.6, 0.1)
        I = [st.number_input(lab, value=v) for lab, v in
             zip(["I(0,0) top-left", "I(2,0) top-right", "I(0,2) bottom-left", "I(2,2) bottom-right"],
                 [10, 20, 5, 15])]
        params = dict(xi=xi, yj=yj, I00=I[0], I20=I[1], I02=I[2], I22=I[3])
        c = bi_lin.compute(params)
        st.write("areas opposite each corner:", c["areas"])
        st.latex(rf"I({xi},{yj}) = \frac{{\sum_c I_c A_c}}{{4}} = {c['M_interp']:.3f}")
