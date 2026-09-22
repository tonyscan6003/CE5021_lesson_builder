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


def _depth(channels: int) -> float:
    """Visual cuboid depth for a channel count.

    Compressed logarithmically: a 256-channel volume should read as deeper than
    a 4-channel one without being 64 times the size of the whole figure.
    """
    c = max(int(channels), 1)
    return 0.28 + 0.85 * np.log10(c) / np.log10(256)


def _volume(ax, x0, ytop, w, h, channels, face, label=None, lw=1.3, alpha=0.95):
    """Draw a feature-map volume as a cuboid; return its right-hand extent.

    (x0, ytop) is the top-left of the *front* face, which is what the caller
    lines things up against; the cuboid extends up and to the right from there.
    """
    d = _depth(channels)
    dx, dy = d * 0.72, d * 0.52
    ybot = ytop - h
    ax.add_patch(plt.Polygon([(x0, ytop), (x0 + dx, ytop + dy),
                              (x0 + w + dx, ytop + dy), (x0 + w, ytop)],
                             fc=face, ec=C["ink"], lw=lw, alpha=alpha * 0.72))
    ax.add_patch(plt.Polygon([(x0 + w, ytop), (x0 + w + dx, ytop + dy),
                              (x0 + w + dx, ybot + dy), (x0 + w, ybot)],
                             fc=face, ec=C["ink"], lw=lw, alpha=alpha * 0.52))
    ax.add_patch(plt.Rectangle((x0, ybot), w, h, fc=face, ec=C["ink"],
                               lw=lw, alpha=alpha))
    if label:
        ax.text(x0 + w / 2, ybot - 0.18, label, ha="center", va="top", fontsize=9.5)
    return x0 + w + dx


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
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    x_in = _volume(ax, 0.2, 3.3, 1.3, 2.2, c, C["light"],
                   f"input  m × n × C\n(C = {c})")
    ys = [3.55 - i * 0.92 for i in range(N)]
    x_ker = 3.0
    for i, y in enumerate(ys):
        _volume(ax, x_ker, y, 0.42, 0.42, 2, SERIES[i % len(SERIES)], None, lw=1.0)
        ax.annotate("", xy=(x_ker - 0.12, y - 0.21), xytext=(x_in + 0.15, 2.2),
                    arrowprops=dict(arrowstyle="->", color=C["grey"], lw=0.9, alpha=0.5))
        ax.annotate("", xy=(4.95, y - 0.21), xytext=(4.15, y - 0.21),
                    arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.1))
        ax.add_patch(plt.Rectangle((5.05, y - 0.42), 0.85, 0.42,
                                   fc=SERIES[i % len(SERIES)], ec=C["ink"],
                                   lw=1.1, alpha=0.8))
    ax.text(x_ker + 0.4, 4.35, f"N = {N} kernels,\neach k × k × C", ha="center",
            fontsize=9.5)
    ax.text(5.5, 4.35, f"N = {N} output\nfeature maps", ha="center", fontsize=9.5)
    ax.annotate("", xy=(7.0, 2.1), xytext=(6.2, 2.1),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.6))
    ax.text(6.6, 2.3, "stack", ha="center", fontsize=9)
    x_out = _volume(ax, 7.2, 3.3, 1.1, 2.2, N, C["sky"],
                    f"output  m′ × n′ × N\n(N = {N})")
    ax.set_xlim(-0.2, x_out + 0.4); ax.set_ylim(0.0, 4.9); ax.axis("off")
    png = save_mpl(fig, ctx, "multi_kernel_conv")
    return Figure("multi_kernel_conv",
                  f"A convolution layer uses several kernels in parallel. Every one of "
                  f"the N = {N} kernels has the same number of channels as the input "
                  f"(C = {c}) and is convolved with the whole input in the way the "
                  "previous figure shows, producing one 2D feature map each. Stacking "
                  "them gives an output with N channels. This is the rule to remember: "
                  "the kernel's channel count is set by the input, and the output's "
                  "channel count is set by the number of kernels. The spatial size "
                  "follows the same arithmetic as plain 2D convolution.", png=png)


def fig_conv_volumes(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x_in = _volume(ax, 0.3, 3.5, 1.9, 2.4, 32, C["light"], "input   m × n × C")
    # kernel projected onto the input face, as the slides draw it
    ax.add_patch(plt.Rectangle((0.55, 1.55), 0.55, 0.55, fc=C["orange"],
                               ec=C["ink"], lw=1.4, alpha=0.9))
    ax.text(0.83, 1.40, "k × k", ha="center", va="top", fontsize=8.5, color=C["red"])
    ax.annotate("", xy=(x_in + 1.1, 2.3), xytext=(x_in + 0.15, 2.3),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.8))
    ax.text(x_in + 0.62, 2.5, "N kernels\nof size k × k × C", ha="center", fontsize=9.5)
    x_out = _volume(ax, x_in + 1.25, 3.5, 1.4, 2.4, 96, C["sky"],
                    "output   m′ × n′ × N")
    ax.text((x_out) / 2, 0.45,
            "Read off the shapes:  C input channels ⇒ every kernel has C channels.\n"
            "N output channels ⇒ there are N kernels.  So the layer holds "
            "N × k × k × C weights (+ N biases).",
            ha="center", va="top", fontsize=9.5)
    ax.set_xlim(-0.2, x_out + 0.4); ax.set_ylim(-0.5, 4.6); ax.axis("off")
    png = save_mpl(fig, ctx, "conv_volumes")
    return Figure("conv_volumes",
                  "Real networks have hundreds of feature maps, so drawing each one is "
                  "impractical. Architecture diagrams instead show a whole stack as a "
                  "single volume whose depth is the channel count, often with a small "
                  "k × k square projected onto the input face to mark a convolution. "
                  "Nothing is lost: the input depth tells you how many channels each "
                  "kernel must have and the output depth tells you how many kernels "
                  "there are, so the layer's shape and cost can be recovered from the "
                  "diagram alone.", png=png)


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
    for ax, N, title in [(axes[0], 1, "(a) one 1 × 1 kernel → one feature map"),
                         (axes[1], 4, "(b) N 1 × 1 kernels → N feature maps")]:
        x_in = _volume(ax, 0.2, 2.9, 1.2, 1.9, c, C["light"],
                       f"input  m × n × C\n(C = {c})")
        ax.add_patch(plt.Rectangle((0.42, 1.55), 0.18, 0.18, fc=C["orange"],
                                   ec=C["ink"], lw=1.3))
        ax.text(0.51, 1.42, "1×1", ha="center", va="top", fontsize=8, color=C["red"])
        ax.annotate("", xy=(x_in + 1.15, 1.9), xytext=(x_in + 0.15, 1.9),
                    arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.6))
        ax.text(x_in + 0.65, 2.1, f"{N} kernel{'s' if N > 1 else ''}\n1 × 1 × {c}",
                ha="center", fontsize=9)
        x_out = _volume(ax, x_in + 1.3, 2.9, 1.2, 1.9, N, C["sky"],
                        f"output  m × n × {N}")
        ax.set_xlim(-0.2, x_out + 0.35); ax.set_ylim(0.2, 3.7); ax.axis("off")
        ax.set_title(title, fontsize=10)
    png = save_mpl(fig, ctx, "one_by_one_conv")
    return Figure("one_by_one_conv",
                  "A 1 × 1 convolution is an ordinary convolution with the smallest "
                  "possible kernel. It still spans the full depth of the input, so "
                  "each kernel is 1 × 1 × C and computes a weighted sum across "
                  "channels at every pixel independently. Two properties make it "
                  "useful: it needs no padding and leaves the spatial size unchanged, "
                  "and it has the smallest parameter count of any kernel. What it "
                  "changes is the number of channels — which is exactly what a "
                  "bottleneck layer needs.", png=png)


def fig_bottleneck_conv(ctx: BuildContext) -> Figure:
    b = BN
    r = bottleneck_cost(b["m"], b["n"], b["C"], b["k"], b["N"], b["P"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6),
                             gridspec_kw=dict(height_ratios=[1.25, 1]))

    ax = axes[0]
    x = _volume(ax, 0.2, 2.6, 1.0, 1.7, b["C"], C["light"],
                f"input\n{b['m']}×{b['n']}×{b['C']}")
    ax.annotate("", xy=(x + 1.15, 1.75), xytext=(x + 0.15, 1.75),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.6))
    ax.text(x + 0.65, 1.95, f"{b['N']} × {b['k']}×{b['k']}", ha="center", fontsize=9)
    xa = _volume(ax, x + 1.3, 2.6, 1.0, 1.7, b["N"], C["sky"],
                 f"output\n{b['m']}×{b['n']}×{b['N']}")
    ax.text(xa / 2, -0.15, f"(a) plain:  {r['plain']:,} multiplications", ha="center",
            va="top", fontsize=10.5, color=C["red"])

    x0 = xa + 1.4
    x = _volume(ax, x0, 2.6, 1.0, 1.7, b["C"], C["light"],
                f"input\n{b['m']}×{b['n']}×{b['C']}")
    ax.annotate("", xy=(x + 0.75, 1.75), xytext=(x + 0.12, 1.75),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.6))
    ax.text(x + 0.44, 1.95, "1×1", ha="center", fontsize=8.5)
    x = _volume(ax, x + 0.9, 2.6, 0.9, 1.7, b["P"], C["orange"],
                f"bottleneck\n{b['m']}×{b['n']}×{b['P']}")
    ax.annotate("", xy=(x + 0.75, 1.75), xytext=(x + 0.12, 1.75),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.6))
    ax.text(x + 0.44, 1.95, f"{b['k']}×{b['k']}", ha="center", fontsize=8.5)
    xb = _volume(ax, x + 0.9, 2.6, 1.0, 1.7, b["N"], C["sky"],
                 f"output\n{b['m']}×{b['n']}×{b['N']}")
    ax.text((x0 + xb) / 2, -0.15,
            f"(b) bottleneck:  {r['reduce']:,} + {r['expand']:,} = {r['total']:,}",
            ha="center", va="top", fontsize=10.5, color=C["green"])
    ax.set_xlim(-0.3, xb + 0.4); ax.set_ylim(-0.9, 3.6); ax.axis("off")

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
