"""Lesson 1.6 — 1D Fourier series and the discrete Fourier transform.

Figures
    periodic_signals     periodic waveforms and their periods
    sinusoid_params      amplitude, frequency and phase of a sinusoid (interactive)
    iq_components        a phase-shifted sinusoid as in-phase and quadrature parts
    complex_phasor       e^{jωt} as circular motion; positive and negative frequencies (interactive)
    fourier_series       partial sums of the square-wave series and the Gibbs overshoot (interactive)
    frequency_spectrum   trigonometric and exponential-form spectra of the square wave
    dft_16               16-point DFT of a sine and a square wave; mirror frequencies
    dft_leakage          non-integer cycles cause spectral leakage; a window reduces it
    convolution_theorem  filtering by multiplying spectra equals convolving in time
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .._common import BuildContext, Figure, C, save_mpl, plotly_html, slider_frames_layout

FIGURES = ["periodic_signals", "sinusoid_params", "iq_components", "complex_phasor",
           "fourier_series", "frequency_spectrum", "dft_16", "dft_leakage", "convolution_theorem"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def square_partial_sum(t: np.ndarray, K: int, T: float = 1.0) -> np.ndarray:
    """Fourier series of a ±1 square wave, odd harmonics up to K."""
    x = np.zeros_like(t)
    for k in range(1, K + 1, 2):
        x += (4 / (np.pi * k)) * np.sin(2 * np.pi * k * t / T)
    return x


def square_coeffs(K: int):
    ks = np.arange(0, K + 1)
    bk = np.where(ks % 2 == 1, 4 / (np.pi * np.maximum(ks, 1)), 0.0)
    bk[0] = 0
    return ks, bk


def dft_mag(x: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.fft(x))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_periodic_signals(ctx: BuildContext) -> Figure:
    t = np.linspace(0, 4, 2000)
    sine = np.sin(2 * np.pi * t)
    pulse = ((t % 1.5) < 0.4).astype(float)
    tri = np.abs(((t % 1.0) - 0.5)) * 2 - 0.5 + 0.3 * np.sin(2 * np.pi * 4 * t)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, (y, T, name) in zip(axes, [(sine, 1.0, "sine, T₁ = 1"), (pulse, 1.5, "pulse train, T₂ = 1.5"),
                                       (tri, 1.0, "irregular periodic, T₃ = 1")]):
        ax.plot(t, y, color=C["blue"])
        for m in range(1, 4):
            ax.axvline(m * T, color=C["light"], ls="--")
        ax.annotate("", xy=(T, ax.get_ylim()[1] * 0.9), xytext=(0, ax.get_ylim()[1] * 0.9),
                    arrowprops=dict(arrowstyle="<->", color=C["red"]))
        ax.text(T / 2, ax.get_ylim()[1] * 0.95, "T", color=C["red"], ha="center", fontsize=9)
        ax.set_title(name); ax.set_xlabel("t")
    png = save_mpl(fig, ctx, "periodic_signals")
    return Figure("periodic_signals",
                  "Periodic signals repeat with some period T: x(t) = x(t + T). The shape "
                  "need not be a sinusoid, and a periodic waveform can contain components "
                  "that vary much faster than its period, as in the third example.", png=png)


def fig_sinusoid_params(ctx: BuildContext) -> Figure:
    t = np.linspace(0, 2, 1000)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].plot(t, np.sin(2 * np.pi * t), label="A = 1")
    axes[0].plot(t, 2 * np.sin(2 * np.pi * t), label="A = 2"); axes[0].set_title("amplitude A")
    axes[1].plot(t, np.sin(2 * np.pi * t), label="f = 1 Hz")
    axes[1].plot(t, np.sin(2 * np.pi * 3 * t), label="f = 3 Hz"); axes[1].set_title("frequency f (ω = 2πf)")
    axes[2].plot(t, np.sin(2 * np.pi * t), label="φ = 0")
    axes[2].plot(t, np.sin(2 * np.pi * t + np.pi / 2), label="φ = +π/2 (advanced = cos)")
    axes[2].plot(t, np.sin(2 * np.pi * t - np.pi / 4), label="φ = −π/4 (delayed)"); axes[2].set_title("phase φ")
    for ax in axes:
        ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=7, loc="upper right"); ax.set_xlabel("t (s)")
        ax.set_ylim(-2.3, 2.9)
    png = save_mpl(fig, ctx, "sinusoid_params")

    fig2 = go.Figure()
    frames = []
    phis = [round(p, 2) for p in np.arange(-np.pi, np.pi + 0.01, np.pi / 8)]
    for p in phis:
        frames.append(go.Frame(name=str(p), data=[go.Scatter(x=t, y=np.sin(2 * np.pi * t + p), line=dict(color=C["blue"]))],
                               layout=go.Layout(title=f"x(t) = sin(2πt + φ), φ = {p:+.2f} rad")))
    fig2.add_trace(go.Scatter(x=t, y=np.sin(2 * np.pi * t), line=dict(color=C["blue"])))
    fig2.add_trace(go.Scatter(x=t, y=np.sin(2 * np.pi * t), line=dict(color=C["grey"], dash="dot"), name="φ = 0"))
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(phis, "φ = "), title="x(t) = sin(2πt + φ), φ = 0.00 rad",
                       showlegend=False, yaxis=dict(range=[-1.2, 1.2]), xaxis_title="t (s)")
    return Figure("sinusoid_params",
                  "A sinusoid A sin(ωt + φ) has three parameters: amplitude A, angular "
                  "frequency ω = 2πf in radians per second (period 1/f), and phase φ in "
                  "radians. A positive phase advances the wave (shifts it left); a phase of "
                  "+π/2 turns a sine into a cosine.",
                  png=png, html=plotly_html(fig2, height=380), kind="plotly")


def fig_iq_components(ctx: BuildContext) -> Figure:
    t = np.linspace(0, 2, 1000)
    A, phi = 1.5, np.pi / 3
    a, b = A * np.sin(phi), A * np.cos(phi)      # A sin(ωt+φ) = a cos ωt + b sin ωt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    axes[0].plot(t, A * np.sin(2 * np.pi * t + phi), color=C["ink"], lw=2, label=f"A sin(ωt + φ), A = {A}, φ = π/3")
    axes[0].plot(t, a * np.cos(2 * np.pi * t), color=C["blue"], label=f"in-phase: {a:.2f} cos ωt")
    axes[0].plot(t, b * np.sin(2 * np.pi * t), color=C["orange"], label=f"quadrature: {b:.2f} sin ωt")
    axes[0].legend(fontsize=7); axes[0].set_xlabel("t"); axes[0].set_title("one sinusoid = cosine part + sine part")
    axes[1].axis("off")
    axes[1].text(0.02, 0.85, r"$A\sin(\omega t+\varphi)=a\cos\omega t+b\sin\omega t$", fontsize=13)
    axes[1].text(0.02, 0.68, r"$a = A\sin\varphi, \quad b = A\cos\varphi$", fontsize=12)
    axes[1].text(0.02, 0.45, "The phase is absorbed into two amplitudes:\n"
                 "a cos ωt is the in-phase term, b sin ωt the quadrature term\n"
                 "(90° out of phase with the cosine).", fontsize=10, va="top")
    png = save_mpl(fig, ctx, "iq_components")
    return Figure("iq_components",
                  "In-phase and quadrature components. Any phase-shifted sinusoid can be "
                  "written as a cosine plus a sine of the same frequency with no phase "
                  "terms, only two amplitudes. This is the form the Fourier series uses.",
                  png=png)


def fig_complex_phasor(ctx: BuildContext) -> Figure:
    th = np.linspace(0, 2 * np.pi, 400)
    t0 = 0.9
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    ax = axes[0]
    ax.plot(np.cos(th), np.sin(th), color=C["light"])
    ax.annotate("", xy=(np.cos(t0), np.sin(t0)), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=C["blue"], lw=2))
    ax.plot([np.cos(t0), np.cos(t0)], [0, np.sin(t0)], color=C["orange"], ls="--")
    ax.plot([0, np.cos(t0)], [0, 0], color=C["green"], lw=3)
    ax.plot([0, 0], [0, np.sin(t0)], color=C["orange"], lw=3)
    ax.text(np.cos(t0) / 2, -0.15, "cos ωt (real)", color=C["green"], fontsize=8, ha="center")
    ax.text(-0.45, np.sin(t0) / 2, "sin ωt\n(imag)", color=C["orange"], fontsize=8, ha="center")
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_title("(a) e^{jωt} = cos ωt + j sin ωt"); ax.set_xlabel("real"); ax.set_ylabel("imaginary")
    ax = axes[1]
    ax.plot(np.cos(th), np.sin(th), color=C["light"])
    for sgn, col, lab in [(1, C["blue"], "e^{+jωt} (anticlockwise)"), (-1, C["red"], "e^{−jωt} (clockwise)")]:
        ax.annotate("", xy=(np.cos(t0), sgn * np.sin(t0)), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=2))
        ax.text(np.cos(t0) + 0.05, sgn * np.sin(t0), lab, color=col, fontsize=8, va="center")
    ax.annotate("", xy=(2 * np.cos(t0), 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=C["green"], lw=2))
    ax.text(np.cos(t0), -0.2, "sum = 2 cos ωt", color=C["green"], fontsize=8, ha="center")
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 2.3); ax.set_ylim(-1.3, 1.3)
    ax.set_title("(b) positive and negative frequency"); ax.set_xlabel("real")
    ax = axes[2]
    tt = np.linspace(0, 2, 500)
    ax.plot(tt, np.cos(2 * np.pi * tt), color=C["green"], lw=2, label="cos ωt = (e^{jωt} + e^{−jωt}) / 2")
    ax.plot(tt, np.sin(2 * np.pi * tt), color=C["orange"], lw=2, label="sin ωt = (e^{jωt} − e^{−jωt}) / 2j")
    ax.legend(fontsize=7); ax.set_xlabel("t"); ax.set_title("(c) Euler's identity")
    png = save_mpl(fig, ctx, "complex_phasor")

    ts = [round(v, 2) for v in np.linspace(0, 1, 17)]
    fig2 = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6],
                         subplot_titles=("phasors in the complex plane", "real parts: cos ωt"))
    frames = []
    for tv in ts:
        ang = 2 * np.pi * tv
        frames.append(go.Frame(name=str(tv), data=[
            go.Scatter(x=[0, np.cos(ang)], y=[0, np.sin(ang)], mode="lines+markers", line=dict(color=C["blue"], width=3), name="e^{+jωt}"),
            go.Scatter(x=[0, np.cos(ang)], y=[0, -np.sin(ang)], mode="lines+markers", line=dict(color=C["red"], width=3), name="e^{−jωt}"),
            go.Scatter(x=[0, 2 * np.cos(ang)], y=[0, 0], mode="lines+markers", line=dict(color=C["green"], width=3), name="sum"),
            go.Scatter(x=tt, y=2 * np.cos(2 * np.pi * tt), line=dict(color=C["green"]), name="2 cos ωt"),
            go.Scatter(x=[tv], y=[2 * np.cos(ang)], mode="markers", marker=dict(color=C["green"], size=12), name="now"),
        ], traces=[0, 1, 2, 3, 4]))
    for i, tr in enumerate(frames[3].data):
        fig2.add_trace(tr, row=1, col=1 if i < 3 else 2)
    fig2.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), line=dict(color=C["light"]), showlegend=False), row=1, col=1)
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(ts, "t = "), title="Counter-rotating phasors add to a real cosine")
    fig2.update_xaxes(range=[-2.3, 2.3], row=1, col=1); fig2.update_yaxes(range=[-1.3, 1.3], scaleanchor="x", row=1, col=1)
    fig2.update_xaxes(title_text="t", row=1, col=2)
    return Figure("complex_phasor",
                  "Complex sinusoids. (a) e^{jωt} traces a circle in the complex plane: its "
                  "real part is cos ωt and its imaginary part sin ωt. (b) e^{−jωt} rotates "
                  "the other way, so the pair can be thought of as a positive and a negative "
                  "frequency; their imaginary parts cancel and their sum is a real cosine. "
                  "(c) Euler's identity states the same thing algebraically.",
                  png=png, html=plotly_html(fig2, height=420), kind="plotly")


def fig_fourier_series(ctx: BuildContext) -> Figure:
    t = np.linspace(-1, 1, 4000)
    Ks = [1, 3, 5, 15, 51]
    fig, axes = plt.subplots(1, len(Ks), figsize=(14, 2.9), sharey=True)
    sq = np.sign(np.sin(2 * np.pi * t))
    for ax, K in zip(axes, Ks):
        ax.plot(t, sq, color=C["light"], lw=1)
        ax.plot(t, square_partial_sum(t, K), color=C["blue"], lw=1.2)
        ax.set_title(f"harmonics up to k = {K}"); ax.set_xlabel("t / T"); ax.set_ylim(-1.4, 1.4)
    axes[-1].annotate("Gibbs overshoot ≈ 9%", xy=(0.02, 1.18), xytext=(0.2, 1.3), fontsize=8,
                      arrowprops=dict(arrowstyle="->", color=C["red"]), color=C["red"])
    png = save_mpl(fig, ctx, "fourier_series")

    Ks2 = [1, 3, 5, 7, 9, 15, 21, 31, 51, 101]
    t2 = np.linspace(-1, 1, 800)
    fig2 = go.Figure()
    frames = [go.Frame(name=str(K), data=[go.Scatter(x=t2, y=square_partial_sum(t2, K), line=dict(color=C["blue"]))],
                       layout=go.Layout(title=f"square wave from odd harmonics 1 … {K}")) for K in Ks2]
    fig2.add_trace(frames[1].data[0])
    fig2.add_trace(go.Scatter(x=t2, y=np.sign(np.sin(2 * np.pi * t2)), line=dict(color=C["grey"], dash="dot"), name="square wave"))
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(Ks2, "highest harmonic k = "), title=frames[1].layout.title.text,
                       showlegend=False, yaxis=dict(range=[-1.4, 1.4]), xaxis_title="t / T")
    return Figure("fourier_series",
                  "Fourier series of a square wave. Adding odd harmonics k = 1, 3, 5, … of "
                  "the fundamental frequency 2π/T, each with amplitude 4/(πk), builds up the "
                  "square wave. More terms narrow the ripples, but the overshoot at each "
                  "edge, the Gibbs phenomenon, never disappears.",
                  png=png, html=plotly_html(fig2, height=380), kind="plotly")


def fig_frequency_spectrum(ctx: BuildContext) -> Figure:
    ks, bk = square_coeffs(9)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    axes[0].stem(ks, bk, linefmt=C["blue"], markerfmt="o", basefmt=" ")
    axes[0].set_title("trigonometric form: |coefficient| at k·2π/T"); axes[0].set_xlabel("k"); axes[0].set_ylabel("magnitude √(a²+b²)")
    kk = np.arange(-9, 10)
    ck = np.where(kk % 2 != 0, 2 / (np.pi * np.abs(np.where(kk == 0, 1, kk))), 0.0)
    axes[1].stem(kk, ck, linefmt=C["orange"], markerfmt="o", basefmt=" ")
    axes[1].set_title("exponential form: |c_k| at ±k"); axes[1].set_xlabel("k"); axes[1].set_ylabel("|c_k|")
    for ax in axes:
        ax.set_ylim(0, 1.5)
    png = save_mpl(fig, ctx, "frequency_spectrum")
    return Figure("frequency_spectrum",
                  "Frequency spectrum of the square wave. Left: the trigonometric series "
                  "has only odd harmonics with magnitude 4/(πk); the sine coefficients b_k "
                  "carry everything and the a_k are zero. Right: the exponential form "
                  "spreads each line over a positive and a negative frequency of half the "
                  "magnitude, so the total power is unchanged. Either spectrum describes "
                  "the signal completely.", png=png)


def fig_dft_16(ctx: BuildContext) -> Figure:
    N = 16
    n = np.arange(N)
    x = np.cos(2 * np.pi * 2 * n / N)
    X = dft_mag(x)
    sq = np.sign(np.sin(2 * np.pi * n / N + 1e-9))
    SQ = dft_mag(sq)
    tt = np.linspace(0, N, 800)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    axes[0, 0].plot(tt, np.cos(2 * np.pi * 2 * tt / N), color=C["light"]); axes[0, 0].stem(n, x, linefmt=C["blue"], markerfmt="o", basefmt=" ")
    axes[0, 0].set_title("(a) 16 samples of a sine, 2 cycles"); axes[0, 0].set_xlabel("n")
    axes[0, 1].stem(n, X, linefmt=C["blue"], markerfmt="o", basefmt=" "); axes[0, 1].set_title("(a) |X[k]|: peaks at k = 2 and k = 14")
    axes[0, 1].axvline(8, color=C["grey"], ls="--"); axes[0, 1].text(8.2, X.max() * 0.9, "N/2", color=C["grey"]); axes[0, 1].set_xlabel("k")
    ax = axes[0, 2]
    th = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(th), np.sin(th), color=C["light"])
    for k, col, lab, lw in [(-2, C["red"], "k = −2, clockwise", 5), (14, C["blue"], "k = 14, anticlockwise", 2)]:
        ang = 2 * np.pi * k * 1 / N
        ax.annotate("", xy=(np.cos(ang), np.sin(ang)), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=lw))
        ax.plot([], [], color=col, lw=lw, label=lab)
    ax.plot(np.cos(2 * np.pi * 14 / N), np.sin(2 * np.pi * 14 / N), "ko", label="position after n = 1")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal"); ax.set_title("(a) after one sample, k = 14 and k = −2 coincide"); ax.set_xticks([]); ax.set_yticks([])
    axes[1, 0].stem(n, sq, linefmt=C["orange"], markerfmt="o", basefmt=" "); axes[1, 0].set_title("(b) 16 samples of a square wave"); axes[1, 0].set_xlabel("n")
    axes[1, 1].stem(n, SQ, linefmt=C["orange"], markerfmt="o", basefmt=" "); axes[1, 1].set_title("(b) |X[k]|: odd harmonics only"); axes[1, 1].set_xlabel("k")
    axes[1, 1].axvline(8, color=C["grey"], ls="--")
    rec = np.fft.ifft(np.fft.fft(sq)).real
    axes[1, 2].stem(n, rec, linefmt=C["green"], markerfmt="o", basefmt=" "); axes[1, 2].set_title("(b) inverse DFT reproduces the samples"); axes[1, 2].set_xlabel("n")
    png = save_mpl(fig, ctx, "dft_16")
    return Figure("dft_16",
                  "The discrete Fourier transform of N = 16 samples. (a) A sine with two "
                  "cycles in the window gives a peak at k = 2 and a mirror peak at k = 14: "
                  "the DFT uses only positive k, and k = 14 is the alias of the negative "
                  "frequency k = −2, since after one sample both phasors reach the same "
                  "point. (b) A square wave gives the same odd-harmonic pattern as its "
                  "Fourier series, and the inverse DFT returns the samples exactly.", png=png)


def fig_dft_leakage(ctx: BuildContext) -> Figure:
    N = 16
    n = np.arange(N)
    x_int = np.cos(2 * np.pi * 2 * n / N)
    x_frac = np.cos(2 * np.pi * 2.5 * n / N)
    w = np.hanning(N)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6))
    for r, (x, name) in enumerate([(x_int, "2 cycles in the window"), (x_frac, "2.5 cycles in the window")]):
        axes[r, 0].stem(n, x, linefmt=C["blue"], markerfmt="o", basefmt=" "); axes[r, 0].set_title(name); axes[r, 0].set_xlabel("n")
        axes[r, 1].stem(n, dft_mag(x), linefmt=C["blue"], markerfmt="o", basefmt=" "); axes[r, 1].set_title("|X[k]| rectangular window"); axes[r, 1].set_xlabel("k")
        axes[r, 2].stem(n, dft_mag(x * w), linefmt=C["green"], markerfmt="o", basefmt=" "); axes[r, 2].set_title("|X[k]| with a Hann window"); axes[r, 2].set_xlabel("k")
        for ax in axes[r, 1:]:
            ax.set_ylim(0, 9)
    png = save_mpl(fig, ctx, "dft_leakage")
    return Figure("dft_leakage",
                  "Spectral leakage. The DFT assumes the N samples are one period of a "
                  "periodic signal. When a whole number of cycles fits (top) the energy "
                  "lands in one bin. With 2.5 cycles (bottom) the implied signal has a jump "
                  "at the window edge and the energy leaks into many bins. A Hann window "
                  "tapers the ends and confines the leakage to neighbouring bins, at the "
                  "cost of a lower peak.", png=png)


def fig_convolution_theorem(ctx: BuildContext) -> Figure:
    N = 64
    n = np.arange(N)
    x = np.sign(np.sin(2 * np.pi * 2 * n / N + 1e-9))
    X = np.fft.fft(x)
    cutoff = 7
    H = np.zeros(N); H[:cutoff + 1] = 1; H[N - cutoff:] = 1            # mirrored brick wall
    y_freq = np.fft.ifft(X * H).real
    h = np.fft.ifft(H).real                                             # the equivalent kernel
    y_time = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(h)))      # circular convolution x ⊛ h
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.8))
    axes[0, 0].stem(n, x, linefmt=C["orange"], markerfmt=" ", basefmt=" "); axes[0, 0].set_title("x[n]: square wave")
    axes[0, 1].stem(n, np.abs(X), linefmt=C["blue"], markerfmt="o", basefmt=" ")
    axes[0, 1].plot(n, H * np.abs(X).max(), color=C["green"], lw=2, label="brick-wall low-pass H[k]")
    axes[0, 1].set_title("|X[k]| and the filter (mirrored about N/2)"); axes[0, 1].legend(fontsize=7)
    axes[0, 2].stem(n, np.abs(X * H), linefmt=C["blue"], markerfmt="o", basefmt=" "); axes[0, 2].set_title("|X[k] · H[k]|")
    axes[1, 0].stem(n, np.fft.fftshift(h), linefmt=C["green"], markerfmt="o", basefmt=" "); axes[1, 0].set_title("h[n] = IDFT of H: a sinc-like kernel")
    axes[1, 1].plot(n, y_freq, color=C["blue"], lw=2, label="IDFT of X·H")
    axes[1, 1].plot(n, y_time, color=C["red"], ls="--", label="x ⊛ h in the time domain")
    axes[1, 1].set_title(f"identical: max difference {np.abs(y_freq - y_time).max():.1e}"); axes[1, 1].legend(fontsize=7)
    axes[1, 2].axis("off")
    axes[1, 2].text(0, 0.7, "Convolution theorem\n\nDFT(x ∗ h) = DFT(x) · DFT(h)\n\nFiltering can be done by\nmultiplying spectra instead of\nconvolving samples.", fontsize=11, va="top")
    for ax in axes.ravel()[:5]:
        ax.set_xlabel("n" if ax in (axes[0, 0], axes[1, 0], axes[1, 1]) else "k")
    png = save_mpl(fig, ctx, "convolution_theorem")
    return Figure("convolution_theorem",
                  "The convolution theorem. Multiplying the DFT of a square wave by a "
                  "brick-wall low-pass response (mirrored about N/2) and inverting gives a "
                  "smoothed waveform. Convolving the square wave directly with the kernel "
                  "whose DFT is that response gives exactly the same waveform. Filtering in "
                  "the frequency domain is therefore an alternative to convolution.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_periodic_signals(ctx), fig_sinusoid_params(ctx), fig_iq_components(ctx),
            fig_complex_phasor(ctx), fig_fourier_series(ctx), fig_frequency_spectrum(ctx),
            fig_dft_16(ctx), fig_dft_leakage(ctx), fig_convolution_theorem(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("1.6 Fourier series and the DFT")
    tab1, tab2, tab3 = st.tabs(["Fourier series", "DFT and leakage", "Convolution theorem"])
    with tab1:
        K = st.slider("highest odd harmonic", 1, 101, 5, 2)
        t = np.linspace(-1, 1, 3000)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, np.sign(np.sin(2 * np.pi * t)), color=C["light"]); ax.plot(t, square_partial_sum(t, K), color=C["blue"])
        ax.set_title(f"{(K + 1) // 2} terms"); st.pyplot(fig)
    with tab2:
        N = st.select_slider("N samples", [16, 32, 64], value=16)
        cyc = st.slider("cycles in the window", 0.5, 8.0, 2.0, 0.25)
        win = st.checkbox("apply Hann window")
        n = np.arange(N); x = np.cos(2 * np.pi * cyc * n / N)
        if win:
            x = x * np.hanning(N)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].stem(n, x, linefmt=C["blue"], markerfmt="o", basefmt=" "); axes[0].set_title("samples")
        axes[1].stem(n, dft_mag(x), linefmt=C["blue"], markerfmt="o", basefmt=" "); axes[1].set_title("|X[k]|")
        axes[1].axvline(N / 2, color=C["grey"], ls="--"); st.pyplot(fig)
        st.caption("Non-integer cycle counts leak into neighbouring bins; the peak above N/2 is the mirror (negative) frequency.")
    with tab3:
        N = 64; n = np.arange(N)
        x = np.sign(np.sin(2 * np.pi * 2 * n / N + 1e-9))
        cutoff = st.slider("low-pass cutoff bin", 1, 31, 7)
        H = np.zeros(N); H[:cutoff + 1] = 1; H[N - cutoff:] = 1
        y = np.fft.ifft(np.fft.fft(x) * H).real
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].stem(n, np.abs(np.fft.fft(x)), linefmt=C["blue"], markerfmt="o", basefmt=" ")
        axes[0].plot(n, H * 40, color=C["green"]); axes[0].set_title("spectrum and filter")
        axes[1].plot(n, x, color=C["light"]); axes[1].plot(n, y, color=C["blue"]); axes[1].set_title("filtered waveform")
        st.pyplot(fig)
