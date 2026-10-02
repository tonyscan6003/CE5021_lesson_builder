"""Lesson 1.3 — Advanced convolution.

Figures
    multi_channel_conv  the slide's RGB example: per-channel 2D convolutions, summed
    multi_kernel_conv   C-channel input convolved with N kernels gives N feature maps
    conv_volumes        the compact volume notation and what it implies about shapes
    conv_cost           parameters and multiplications for one convolution layer
    one_by_one_conv     1x1 convolution mixes channels and leaves the spatial size alone
    bottleneck_conv     the slide's worked bottleneck example (interactive)

Source: reference/legacy_md/lesson_1_3_advanced_convolution + lesson_1_3_script.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import convolve2d

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout, sync_matrix)
from .._volumes import (OBLIQUE, chan_len as _chan_len, spatial_side as _spatial_side,
                        volume as _volume, kernel_cube as _kernel_cube)

FIGURES = ["multi_channel_conv", "multi_kernel_conv", "conv_volumes",
           "conv_cost", "one_by_one_conv", "bottleneck_conv"]

# Slide 4: a 3x3x3 input convolved with a 2x2x3 kernel, valid padding, stride 1.
MC_INPUT = np.array([[[1, 4, 4], [2, 4, 4], [2, 4, 5]],          # R
                     [[2, 2, 2], [4, 5, 2], [2, 4, 0]],          # G
                     [[3, 2, 4], [0, 4, 3], [0, 5, 1]]])         # B
MC_KERNEL = np.array([[[1, -2], [0, -2]],
                      [[-2, 0], [2, 1]],
                      [[-1, 0], [2, -2]]])

# Slide 10: the bottleneck worked example.
BN = dict(m=32, n=32, C=24, N=12, k=3, P=12)


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def multi_channel_conv(x, k):
    """Convolve each channel with the matching kernel channel and sum the results."""
    per_channel = [convolve2d(x[c], k[c], mode="valid") for c in range(x.shape[0])]
    return np.stack(per_channel), sum(per_channel)


def conv_layer_cost(m, n, c, k, N, out_m=None, out_n=None, bias=True):
    """Parameters and multiplications for one convolution layer.

    The number of kernel positions equals the spatial size of the output, which
    is the shortcut the lesson uses to avoid needing stride and padding.
    """
    out_m = m if out_m is None else out_m
    out_n = n if out_n is None else out_n
    weights = N * k * k * c
    return dict(weights=weights, biases=N if bias else 0,
                params=weights + (N if bias else 0),
                positions=out_m * out_n,
                mults=weights * out_m * out_n)


def bottleneck_cost(m, n, c, k, N, p):
    """Multiplications with and without a 1x1 bottleneck, assuming 'same' padding."""
    plain = conv_layer_cost(m, n, c, k, N)["mults"]
    reduce_ = conv_layer_cost(m, n, c, 1, p)["mults"]
    expand = conv_layer_cost(m, n, p, k, N)["mults"]
    return dict(plain=plain, reduce=reduce_, expand=expand,
                total=reduce_ + expand, saved=plain - (reduce_ + expand))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_multi_channel_conv(ctx: BuildContext) -> Figure:
    per, total = multi_channel_conv(MC_INPUT, MC_KERNEL)
    names = ["R", "G", "B"]
    cmaps = ["Reds", "Greens", "Blues"]
    fig, axes = plt.subplots(3, 5, figsize=(13, 7.4),
                             gridspec_kw=dict(width_ratios=[1.1, 0.92, 0.92, 1.1, 1.35]))
    for c_ in range(3):
        sync_matrix(axes[c_, 0], MC_INPUT[c_], title=f"input channel {names[c_]}",
                    cmap=cmaps[c_])
        sync_matrix(axes[c_, 1], MC_KERNEL[c_], title=f"kernel channel {names[c_]}",
                    cmap="Oranges")
        sync_matrix(axes[c_, 2], MC_KERNEL[c_][::-1, ::-1],
                    title="flipped", cmap="Oranges")
        sync_matrix(axes[c_, 3], per[c_], title=f"convolution result {names[c_]}",
                    cmap="Purples")
        axes[c_, 4].axis("off")
    # the summed output, in the right-hand column beside the three results
    out_ax = fig.add_subplot(1, 5, 5)
    out_ax.set_position(axes[1, 4].get_position())
    sync_matrix(out_ax, total, title="output feature map\n(the three results summed)",
                cmap="Greens", fontsize=13)
    fig.suptitle("Each input channel is convolved with its own kernel channel, "
                 "and the three results are summed into one feature map:\n"
                 f"{per[0].tolist()} + {per[1].tolist()} + {per[2].tolist()} "
                 f"= {total.tolist()}", fontsize=10)
    png = save_mpl(fig, ctx, "multi_channel_conv")
    return Figure("multi_channel_conv",
                  "Convolution with a multi-channel input. The kernel must have the "
                  "same number of channels as the input — here a 3 × 3 × 3 RGB input "
                  "and a 2 × 2 × 3 kernel with valid padding and stride 1. Each "
                  "channel of the input is convolved with the matching channel of the "
                  "kernel, giving three 2 × 2 results, and those are summed "
                  f"element-wise into a single 2 × 2 output feature map, "
                  f"{total.tolist()}. One kernel always produces exactly one output "
                  "feature map, however many input channels there are.", png=png)


def fig_multi_kernel_conv(ctx: BuildContext) -> Figure:
    c, N = 5, 4
    # One spatial end-cap size shared by the input, every feature map and the
    # stacked output, because they all have the same spatial extent. Only the
    # lengths differ: C for the input, 1 per feature map, N for the output. The
    # kernels are the odd ones out - full length, but only k x k spatially.
    # Vertical pitch must clear the end-cap side PLUS its oblique rise, or the
    # stacked feature maps run into each other.
    side, k_side = 0.80, 0.30
    pitch = side * (1 + OBLIQUE[1]) + 0.18
    fig, ax = plt.subplots(figsize=(12.5, 5.8))

    y_in = 2.6
    x_in = _volume(ax, 0.2, y_in, _chan_len(c), side, C["light"],
                   f"input  m × n × C   (C = {c})")
    _kernel_cube(ax, 0.2, y_in - side * 0.24, c, side, frac=0.33)

    x_k, x_fm = 2.5, 5.3
    ys = [y_in + pitch * (N - 1) / 2 - i * pitch for i in range(N)]
    for i, y in enumerate(ys):
        col = SERIES[i % len(SERIES)]
        # each kernel is as long as the input is deep: it must span every channel
        x_ke = _volume(ax, x_k, y, _chan_len(c), k_side, col, lw=1.0)
        ax.annotate("", xy=(x_k - 0.14, y), xytext=(x_in + 0.12, y_in + 0.22),
                    arrowprops=dict(arrowstyle="->", color=C["grey"], lw=0.9,
                                    alpha=0.45))
        ax.annotate("", xy=(x_fm - 0.16, y), xytext=(x_ke + 0.12, y),
                    arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.1))
        # one output feature map: same spatial size, exactly 1 channel deep
        _volume(ax, x_fm, y, _chan_len(1), side, col, lw=1.0, alpha=0.85)
    y_head = ys[0] + side / 2 + 0.55
    ax.text(x_k + _chan_len(c) / 2, y_head,
            f"N = {N} kernels, each k × k × C\n(full depth, small spatially)",
            ha="center", va="bottom", fontsize=9.5)
    ax.text(x_fm + 0.35, y_head, f"N = {N} output feature maps\n(1 channel each)",
            ha="center", va="bottom", fontsize=9.5)

    ax.annotate("", xy=(x_fm + 1.85, y_in), xytext=(x_fm + 1.05, y_in),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.6))
    ax.text(x_fm + 1.45, y_in + 0.16, "stack", ha="center", fontsize=9)
    x_out = _volume(ax, x_fm + 2.0, y_in, _chan_len(N), side, C["sky"],
                    f"output  m′ × n′ × N   (N = {N})")
    ax.set_xlim(-0.2, x_out + 0.4)
    ax.set_ylim(ys[-1] - side / 2 - 0.55, y_head + 0.85)
    ax.axis("off")
    png = save_mpl(fig, ctx, "multi_kernel_conv")
    return Figure("multi_kernel_conv",
                  f"A convolution layer uses several kernels in parallel. Every one of "
                  f"the N = {N} kernels has the same number of channels as the input "
                  f"(C = {c}) — which is why each kernel is drawn the same length as "
                  "the input volume — and is convolved with the whole input in the way "
                  "the previous figure shows, producing one 2D feature map each. Those "
                  "feature maps are one channel deep, so they draw as thin slices; "
                  "stacking them gives an output N channels long. This is the rule to "
                  "remember: the kernel's channel count is set by the input, and the "
                  "output's channel count is set by the number of kernels. The spatial "
                  "size follows the same arithmetic as plain 2D convolution.", png=png)


def fig_conv_volumes(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    # A typical stride-2 stage: spatial size halves while the channel count
    # doubles, so the box gets longer as its end cap shrinks.
    ymid, in_side, out_side = 2.5, _spatial_side(32), _spatial_side(16)
    x_in = _volume(ax, 0.3, ymid, _chan_len(64), in_side, C["light"],
                   "input   m × n × C")
    # The kernel spans every input channel, so it runs the full length of the
    # input volume and covers only a small spatial patch. Drawn dashed, inside.
    _kernel_cube(ax, 0.3, ymid - in_side * 0.18, 64, in_side, frac=0.32,
                 label="one kernel, k × k × C")
    ax.annotate("", xy=(x_in + 1.25, ymid), xytext=(x_in + 0.18, ymid),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.8))
    ax.text(x_in + 0.71, ymid + 0.20, "N kernels\nof size k × k × C",
            ha="center", fontsize=9.5)
    x_out = _volume(ax, x_in + 1.4, ymid, _chan_len(256), out_side, C["sky"],
                    "output   m′ × n′ × N")
    ax.text(x_out / 2, 0.72,
            "The x axis is the channel axis: a deeper volume is a longer box, and the "
            "end cap is the spatial size.\n"
            "Read the shapes off it — C input channels ⇒ every kernel has C channels;  "
            "N output channels ⇒ there are N kernels.\n"
            "So the layer holds N × k × k × C weights (+ N biases).",
            ha="center", va="top", fontsize=9.5)
    ax.set_xlim(-0.2, x_out + 0.4); ax.set_ylim(-0.35, 4.3); ax.axis("off")
    png = save_mpl(fig, ctx, "conv_volumes")
    return Figure("conv_volumes",
                  "Real networks have hundreds of feature maps, so drawing each one is "
                  "impractical. Architecture diagrams instead show a whole stack as a "
                  "single volume. Note the orientation, which is the standard one: the "
                  "horizontal axis is the channel axis, so a volume with more channels "
                  "is a longer box, and the square end cap is the spatial size m × n. "
                  "Going deeper into a network, volumes typically get longer and their "
                  "end caps smaller. The convolution is marked by a dashed cube inside "
                  "the input: it runs the full length of the volume, because a kernel "
                  "must span every input channel, but covers only a k × k patch "
                  "spatially. Nothing is lost by the compact notation — the input "
                  "length tells you how many channels each kernel has and the output "
                  "length tells you how many kernels there are, so the layer's shape "
                  "and cost can be recovered from the diagram alone.", png=png)


def fig_conv_cost(ctx: BuildContext) -> Figure:
    m = n = 32
    k, c, N = 3, 24, 12
    r = conv_layer_cost(m, n, c, k, N)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    rows = [("kernels", f"N = {N}"),
            ("channels per kernel", f"C = {c}"),
            ("kernel spatial size", f"k × k = {k} × {k}"),
            ("weights", f"N·k·k·C = {N}·{k}·{k}·{c} = {r['weights']:,}"),
            ("biases", f"N = {r['biases']}"),
            ("parameters", f"{r['params']:,}"),
            ("kernel positions", f"m′·n′ = {m}·{n} = {r['positions']:,}"),
            ("multiplications", f"weights × positions = {r['mults']:,}")]
    for i, (a, b) in enumerate(rows):
        y = 1 - i * 0.125
        weight = "bold" if a in ("parameters", "multiplications") else "normal"
        ax.text(0.0, y, a, fontsize=10, fontweight=weight)
        ax.text(0.46, y, b, fontsize=10, family="monospace", fontweight=weight,
                color=C["red"] if weight == "bold" else C["ink"])
    ax.set_xlim(-0.03, 1.35); ax.set_ylim(-0.05, 1.12); ax.axis("off")
    ax.set_title(f"one layer: {m}×{n}×{c} input, {N} kernels of {k}×{k}, "
                 f"'same' padding", fontsize=10)

    ax = axes[1]
    ks = np.arange(1, 12)
    for cc, col in zip((12, 24, 48), SERIES):
        ax.plot(ks, [conv_layer_cost(m, n, cc, int(kk), N)["mults"] / 1e6 for kk in ks],
                color=col, lw=2, marker="o", ms=3.5, label=f"C = {cc}")
    ax.scatter([k], [r["mults"] / 1e6], color=C["red"], zorder=5, s=50)
    ax.set_xlabel("kernel size k")
    ax.set_ylabel("multiplications (millions)")
    ax.set_title("cost grows with k², not k", fontsize=10)
    ax.set_xticks(ks); ax.legend(fontsize=9)
    png = save_mpl(fig, ctx, "conv_cost")
    return Figure("conv_cost",
                  "Counting the cost of one convolution layer. The parameters live in "
                  "the kernels: N kernels of k × k × C, plus one bias per output "
                  "feature map. For the multiplications, the number of kernel "
                  "positions is just the spatial size of the output — a shortcut that "
                  "avoids having to work back through the stride and padding. "
                  "Multiplications are counted rather than additions because they cost "
                  "more. Because the count goes as k², small kernels are strongly "
                  "preferred: at C = 24 here, a 5 × 5 kernel costs nearly three times "
                  "a 3 × 3 one.", png=png)


def fig_one_by_one_conv(ctx: BuildContext) -> Figure:
    c = 6
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    side = _spatial_side(32)
    for ax, N, title in [(axes[0], 1, "(a) one 1 × 1 kernel → one feature map"),
                         (axes[1], 4, "(b) N 1 × 1 kernels → N feature maps")]:
        ymid = 2.0
        x_in = _volume(ax, 0.2, ymid, _chan_len(c), side, C["light"],
                       f"input  m × n × C   (C = {c})")
        # a 1x1 kernel: still the full depth of the input, but a single pixel
        # of spatial extent, so it draws as a long thin rod inside the volume
        _kernel_cube(ax, 0.2, ymid - side * 0.22, c, side, frac=0.13,
                     label=f"1 × 1 × {c}")
        ax.annotate("", xy=(x_in + 1.05, ymid), xytext=(x_in + 0.15, ymid),
                    arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.6))
        ax.text(x_in + 0.60, ymid + 0.18,
                f"{N} kernel{'s' if N > 1 else ''}\n1 × 1 × {c}",
                ha="center", fontsize=9)
        x_out = _volume(ax, x_in + 1.2, ymid, _chan_len(N), side, C["sky"],
                        f"output  m × n × {N}")
        ax.set_xlim(-0.2, x_out + 0.35); ax.set_ylim(0.35, 3.5); ax.axis("off")
        ax.set_title(title, fontsize=10)
    png = save_mpl(fig, ctx, "one_by_one_conv")
    return Figure("one_by_one_conv",
                  "A 1 × 1 convolution is an ordinary convolution with the smallest "
                  "possible kernel. It still spans the full depth of the input, which "
                  "the dashed cube shows clearly: the kernel runs the whole length of "
                  "the input volume but is only one pixel across, so it computes a "
                  "weighted sum across channels at each pixel independently. Two "
                  "properties make it useful: it needs no padding and leaves the "
                  "spatial size unchanged — the end caps are the same in the input and "
                  "the output — and it has the smallest parameter count of any kernel. "
                  "What it changes is the number of channels, which is the length of "
                  "the box, and that is exactly what a bottleneck layer needs.",
                  png=png)


def fig_bottleneck_conv(ctx: BuildContext) -> Figure:
    b = BN
    r = bottleneck_cost(b["m"], b["n"], b["C"], b["k"], b["N"], b["P"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6),
                             gridspec_kw=dict(height_ratios=[1.25, 1]))

    ax = axes[0]
    side, ymid = _spatial_side(b["m"]), 1.9
    x = _volume(ax, 0.2, ymid, _chan_len(b["C"]), side, C["light"],
                f"input\n{b['m']}×{b['n']}×{b['C']}")
    _kernel_cube(ax, 0.2, ymid - side * 0.22, b["C"], side, frac=0.26)
    ax.annotate("", xy=(x + 1.0, ymid), xytext=(x + 0.14, ymid),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.6))
    ax.text(x + 0.57, ymid + 0.16, f"{b['N']} × {b['k']}×{b['k']}",
            ha="center", fontsize=9)
    xa = _volume(ax, x + 1.15, ymid, _chan_len(b["N"]), side, C["sky"],
                 f"output\n{b['m']}×{b['n']}×{b['N']}")
    ax.text(xa / 2, 0.45, f"(a) plain:  {r['plain']:,} multiplications", ha="center",
            va="top", fontsize=10.5, color=C["red"])

    x0 = xa + 1.5
    x = _volume(ax, x0, ymid, _chan_len(b["C"]), side, C["light"],
                f"input\n{b['m']}×{b['n']}×{b['C']}")
    _kernel_cube(ax, x0, ymid - side * 0.22, b["C"], side, frac=0.12)
    ax.annotate("", xy=(x + 0.78, ymid), xytext=(x + 0.12, ymid),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.6))
    ax.text(x + 0.45, ymid + 0.16, "1×1", ha="center", fontsize=8.5)
    x = _volume(ax, x + 0.92, ymid, _chan_len(b["P"]), side, C["orange"],
                f"bottleneck\n{b['m']}×{b['n']}×{b['P']}")
    ax.annotate("", xy=(x + 0.78, ymid), xytext=(x + 0.12, ymid),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.6))
    ax.text(x + 0.45, ymid + 0.16, f"{b['k']}×{b['k']}", ha="center", fontsize=8.5)
    xb = _volume(ax, x + 0.92, ymid, _chan_len(b["N"]), side, C["sky"],
                 f"output\n{b['m']}×{b['n']}×{b['N']}")
    ax.text((x0 + xb) / 2, 0.45,
            f"(b) bottleneck:  {r['reduce']:,} + {r['expand']:,} = {r['total']:,}",
            ha="center", va="top", fontsize=10.5, color=C["green"])
    ax.set_xlim(-0.3, xb + 0.4); ax.set_ylim(-0.35, 3.3); ax.axis("off")

    ax = axes[1]
    labels = ["plain\n3×3", "bottleneck\n1×1 stage", "bottleneck\n3×3 stage"]
    vals = [r["plain"], r["reduce"], r["expand"]]
    bars = ax.bar(labels, np.array(vals) / 1e6,
                  color=[C["red"], C["orange"], C["orange"]])
    ax.bar(["bottleneck\ntotal"], [r["total"] / 1e6], color=C["green"])
    for b_, v in zip(list(bars) + list(ax.patches[3:4]), vals + [r["total"]]):
        ax.text(b_.get_x() + b_.get_width() / 2, v / 1e6 + 0.04, f"{v / 1e6:.2f}M",
                ha="center", fontsize=9.5)
    ax.set_ylabel("multiplications (millions)")
    ax.set_title(f"saving {r['saved']:,} multiplications "
                 f"({100 * r['saved'] / r['plain']:.0f}%)", fontsize=10.5)
    png = save_mpl(fig, ctx, "bottleneck_conv")

    # interactive: sweep the bottleneck width P
    ps = list(range(2, b["C"] + 1, 2))
    fig2 = go.Figure()
    costs = [bottleneck_cost(b["m"], b["n"], b["C"], b["k"], b["N"], p) for p in ps]
    fig2.add_trace(go.Scatter(x=ps, y=[c_["total"] / 1e6 for c_ in costs],
                              mode="lines+markers", name="with bottleneck",
                              line=dict(color=C["green"], width=3)))
    fig2.add_trace(go.Scatter(x=ps, y=[r["plain"] / 1e6] * len(ps), mode="lines",
                              name="plain 3×3", line=dict(color=C["red"], dash="dash")))
    fig2.update_layout(title=f"bottleneck width P vs cost "
                             f"({b['m']}×{b['n']}×{b['C']} input, {b['N']} kernels of "
                             f"{b['k']}×{b['k']})",
                       xaxis_title="bottleneck channels P",
                       yaxis_title="multiplications (millions)")
    return Figure("bottleneck_conv",
                  f"The worked bottleneck example from the slides. A "
                  f"{b['m']} × {b['n']} × {b['C']} input convolved with {b['N']} "
                  f"kernels of {b['k']} × {b['k']} at 'same' padding costs "
                  f"{r['plain']:,} multiplications (a). Inserting a 1 × 1 convolution "
                  f"that cuts the channels from {b['C']} to {b['P']} first (b) costs "
                  f"{r['reduce']:,} for the 1 × 1 stage plus {r['expand']:,} for the "
                  f"{b['k']} × {b['k']} stage, {r['total']:,} in total — a saving of "
                  f"{100 * r['saved'] / r['plain']:.0f}%. The expensive "
                  f"{b['k']} × {b['k']} kernels now see far fewer channels. The "
                  "interactive version sweeps the bottleneck width so you can see "
                  "where the saving disappears.",
                  png=png, html=plotly_html(fig2, height=420), kind="plotly")


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_multi_channel_conv(ctx), fig_multi_kernel_conv(ctx), fig_conv_volumes(ctx),
            fig_conv_cost(ctx), fig_one_by_one_conv(ctx), fig_bottleneck_conv(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("1.3 Advanced convolution")
    tab1, tab2, tab3 = st.tabs(["Multi-channel", "Layer cost", "Bottleneck"])

    with tab1:
        st.markdown("Each input channel is convolved with the matching kernel channel; "
                    "the results are summed into one feature map.")
        per, total = multi_channel_conv(MC_INPUT, MC_KERNEL)
        fig, axes = plt.subplots(2, 3, figsize=(9, 5.4))
        for c_ in range(3):
            sync_matrix(axes[0, c_], MC_INPUT[c_], title=f"input {'RGB'[c_]}",
                        cmap=["Reds", "Greens", "Blues"][c_])
            sync_matrix(axes[1, c_], per[c_], title=f"conv result {'RGB'[c_]}",
                        cmap="Purples")
        st.pyplot(fig)
        fig2, ax = plt.subplots(figsize=(3, 2.6))
        sync_matrix(ax, total, title="summed output", cmap="Greens")
        st.pyplot(fig2)

    with tab2:
        c = st.columns(5)
        m = c[0].number_input("m", 1, 1024, 32)
        n = c[1].number_input("n", 1, 1024, 32)
        ch = c[2].number_input("C", 1, 2048, 24)
        k = c[3].number_input("k", 1, 15, 3)
        N = c[4].number_input("N", 1, 2048, 12)
        r = conv_layer_cost(int(m), int(n), int(ch), int(k), int(N))
        st.latex(rf"\text{{weights}} = N k^2 C = {N}\cdot{k}^2\cdot{ch} = {r['weights']:,}")
        st.latex(rf"\text{{multiplications}} = {r['weights']:,} \times {m}\cdot{n}"
                 rf" = {r['mults']:,}")
        st.write(f"parameters including biases: {r['params']:,}")
        st.caption("Assumes 'same' padding, so the output spatial size equals the input.")

    with tab3:
        c = st.columns(3)
        ch = c[0].number_input("input channels C", 2, 2048, 24)
        N = c[1].number_input("kernels N", 1, 2048, 12)
        k = c[2].number_input("kernel k", 2, 11, 3)
        p = st.slider("bottleneck channels P", 1, int(ch), min(12, int(ch)))
        r = bottleneck_cost(32, 32, int(ch), int(k), int(N), int(p))
        c1, c2, c3 = st.columns(3)
        c1.metric("plain", f"{r['plain'] / 1e6:.2f}M")
        c2.metric("bottleneck", f"{r['total'] / 1e6:.2f}M")
        c3.metric("saved", f"{r['saved'] / 1e6:.2f}M",
                  f"{100 * r['saved'] / r['plain']:.0f}%")
        if r["saved"] <= 0:
            st.warning("With P this large the bottleneck costs more than it saves.")
        ps = list(range(1, int(ch) + 1))
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.plot(ps, [bottleneck_cost(32, 32, int(ch), int(k), int(N), q)["total"] / 1e6
                     for q in ps], color=C["green"], lw=2, label="with bottleneck")
        ax.axhline(r["plain"] / 1e6, color=C["red"], ls="--", label="plain")
        ax.axvline(p, color=C["grey"], lw=1)
        ax.set_xlabel("bottleneck channels P")
        ax.set_ylabel("multiplications (millions)")
        ax.legend(fontsize=9)
        st.pyplot(fig)
