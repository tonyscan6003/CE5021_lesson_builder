"""Lesson 1.1 — 2D convolution.

Figures
    conv2d_flip_slide  flip the kernel, pad, slide, multiply and sum (interactive)
    padding_modes      valid / same / full padding, plus an output-size calculator
    stride_grid        stride > 1 downsamples: kernel positions and output size
    pooling_ops        max and average pooling, and why pooling has no parameters
    corr_vs_conv       correlation and convolution coincide for a symmetric kernel

Source: reference/legacy_md/lesson_1_1_convolution + lesson_1_1_script.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import convolve2d, correlate2d

from .._common import (BuildContext, Figure, C, save_mpl, plotly_html,
                       slider_frames_layout, sync_matrix)

FIGURES = ["conv2d_flip_slide", "padding_modes", "stride_grid",
           "pooling_ops", "corr_vs_conv"]

# Fixed teaching examples ----------------------------------------------------
# The slides work a 4x4 image with a 3x3 kernel; the kernel is asymmetric on
# purpose so that the flip is visible in the result.
IMG_2D = np.array([[1, 2, 0, 1],
                   [0, 1, 3, 1],
                   [2, 1, 0, 2],
                   [1, 0, 1, 1]])
KER_2D = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])

# Slide 6/9: a 6x6 input with a 2x2 kernel and stride 2 gives a 3x3 output.
IMG_STRIDE = np.array([[3, 1, 2, 4, 1, 0],
                       [0, 2, 1, 3, 2, 1],
                       [1, 4, 0, 2, 3, 2],
                       [2, 0, 3, 1, 0, 4],
                       [4, 2, 1, 0, 2, 1],
                       [1, 3, 2, 4, 1, 3]])
KER_STRIDE = np.array([[1, 0], [0, 1]])

# Slide 7: pooling, 2x2 window with stride 2 on a 4x4 map.
POOL_IN = np.array([[7, 4, 8, 2],
                    [3, 7, 1, 6],
                    [7, 8, 3, 3],
                    [6, 8, 2, 5]])


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def conv2d_steps(img, ker, pad=1, stride=1):
    """Padded image, flipped kernel and the per-position products of a convolution."""
    kf = ker[::-1, ::-1]
    P = np.pad(img, pad)
    kr, kc = ker.shape
    oh = (P.shape[0] - kr) // stride + 1
    ow = (P.shape[1] - kc) // stride + 1
    out = np.zeros((oh, ow), int)
    steps = []
    for i in range(oh):
        for j in range(ow):
            patch = P[i * stride:i * stride + kr, j * stride:j * stride + kc]
            prod = patch * kf
            out[i, j] = prod.sum()
            steps.append(dict(i=i, j=j, patch=patch.copy(), prod=prod, val=int(out[i, j])))
    if stride == 1 and pad == kr // 2 and kr % 2 == 1:
        assert np.array_equal(out, convolve2d(img, ker, mode="same"))
    return P, kf, out, steps


def output_size(i, k, p, s):
    """The general convolution arithmetic relationship (Dumoulin & Visin, eq. 6)."""
    return (i + 2 * p - k) // s + 1


def pool2d(x, size=2, stride=2, how="max"):
    oh = (x.shape[0] - size) // stride + 1
    ow = (x.shape[1] - size) // stride + 1
    out = np.zeros((oh, ow), float)
    for i in range(oh):
        for j in range(ow):
            win = x[i * stride:i * stride + size, j * stride:j * stride + size]
            out[i, j] = win.max() if how == "max" else win.mean()
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_conv2d_flip_slide(ctx: BuildContext) -> Figure:
    P, kf, out, steps = conv2d_steps(IMG_2D, KER_2D, pad=1)
    kr, kc = KER_2D.shape
    show = steps[5]                                   # output position (1,1)
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.8, 1, 1.4]))
    sync_matrix(axes[0], KER_2D, title="(a) kernel h", cmap="Oranges")
    sync_matrix(axes[1], kf, title="(b) flipped kernel h'", cmap="Oranges")
    sync_matrix(axes[2], P.astype(float), title="(c) zero-padded image, window at (1,1)",
                highlight=(show["i"], show["j"], kr, kc))
    sync_matrix(axes[3], show["prod"], title="(d) patch × h'", cmap="Purples")
    partial = np.full(out.shape, np.nan)
    for s in steps[:6]:
        partial[s["i"], s["j"]] = s["val"]
    sync_matrix(axes[4], partial, title="(e) output so far", cmap="Greens")
    terms = " + ".join(f"{a}·{b}" for a, b in zip(show["patch"].ravel(), kf.ravel()) if b != 0)
    axes[2].set_xlabel(f"output(1,1) = {terms} = {show['val']}", fontsize=9.5)
    png = save_mpl(fig, ctx, "conv2d_flip_slide")

    fig2 = make_subplots(rows=1, cols=3, column_widths=[0.42, 0.29, 0.29],
                         subplot_titles=("zero-padded image (window in red)",
                                         "patch × flipped kernel", "output"))
    frames = []
    vals = np.full(out.shape, np.nan)
    for s in steps:
        vals[s["i"], s["j"]] = s["val"]
        shape = dict(type="rect", xref="x", yref="y",
                     x0=s["j"] - 0.5, x1=s["j"] + kc - 0.5,
                     y0=s["i"] - 0.5, y1=s["i"] + kr - 0.5,
                     line=dict(color=C["red"], width=3))
        terms = " + ".join(f"{a}·{b}" for a, b in zip(s["patch"].ravel(), kf.ravel()) if b != 0)
        frames.append(go.Frame(name=f"({s['i']},{s['j']})", data=[
            go.Heatmap(z=P, text=P, texttemplate="%{text}", colorscale="Blues", showscale=False),
            go.Heatmap(z=s["prod"], text=s["prod"], texttemplate="%{text}",
                       colorscale="Purples", showscale=False, zmid=0),
            go.Heatmap(z=vals.copy(), text=vals.copy(), texttemplate="%{text}",
                       colorscale="Greens", showscale=False, zmid=0),
        ], traces=[0, 1, 2], layout=go.Layout(
            shapes=[shape], title=f"output({s['i']},{s['j']}) = {terms} = {s['val']}")))
    for c_, tr in enumerate(frames[0].data, start=1):
        fig2.add_trace(tr, row=1, col=c_)
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout([f.name for f in frames], "position "),
                       shapes=frames[0].layout.shapes, title=frames[0].layout.title.text)
    for c_ in (1, 2, 3):
        fig2.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=c_)
        fig2.update_xaxes(showticklabels=False, row=1, col=c_)
    return Figure("conv2d_flip_slide",
                  "2D convolution of a 4 × 4 image with a 3 × 3 kernel using one ring of "
                  "zero padding, so that the output is the same size as the input. (a) The "
                  "kernel is flipped horizontally and vertically to give (b). (c) The flipped "
                  "kernel is placed over each 3 × 3 window of the padded image, and (d) the "
                  "values are multiplied element-wise and summed to give one output pixel "
                  "(e). Every position can be stepped through in the interactive version.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def fig_padding_modes(ctx: BuildContext) -> Figure:
    i, k = 4, 3
    modes = [("valid: p = 0", 0), ("same: p = ⌊k/2⌋ = 1", 1), ("full: p = k − 1 = 2", 2)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (name, p) in zip(axes, modes):
        n = i + 2 * p
        o = output_size(i, k, p, 1)
        for r in range(n):
            for c_ in range(n):
                inside = p <= r < p + i and p <= c_ < p + i
                ax.add_patch(plt.Rectangle((c_, n - 1 - r), 1, 1,
                                           fc=C["light"] if inside else "white",
                                           ec=C["grey"], lw=0.8,
                                           hatch=None if inside else "///"))
        ax.add_patch(plt.Rectangle((0, n - k), k, k, fill=False, ec=C["red"], lw=2.5))
        ax.add_patch(plt.Rectangle((n - k, 0), k, k, fill=False, ec=C["orange"],
                                   lw=2.5, ls="--"))
        ax.set_xlim(-0.2, n + 0.2); ax.set_ylim(-0.2, n + 0.2); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{name}\ninput {i}×{i}, kernel {k}×{k} → output {o}×{o}", fontsize=9.5)
    fig.suptitle("Grey = image pixels, hatched = zero padding; "
                 "red = first kernel position, orange = last", fontsize=9)
    png = save_mpl(fig, ctx, "padding_modes")

    widget = """
<div class="widget conv-calc">
  <b>Output-size calculator</b> &nbsp; <span class="muted">o = ⌊(i + 2p − k)/s⌋ + 1</span>
  <div class="widget-row">
    <label>input i <input type="number" id="cc-i" value="8" min="1"></label>
    <label>kernel k <input type="number" id="cc-k" value="3" min="1"></label>
    <label>padding p <input type="number" id="cc-p" value="0" min="0"></label>
    <label>stride s <input type="number" id="cc-s" value="1" min="1"></label>
  </div>
  <div id="cc-out" class="widget-out"></div>
</div>
<script>
(function(){
  const ids=["cc-i","cc-k","cc-p","cc-s"], out=document.getElementById("cc-out");
  function upd(){
    const [i,k,p,s]=ids.map(x=>parseInt(document.getElementById(x).value||"0"));
    const num=i+2*p-k;
    if(num<0||s<1){out.textContent="kernel is larger than the padded input";return;}
    const o=Math.floor(num/s)+1;
    let mode="custom padding"; if(p===0) mode="valid";
    if(k%2===1&&p===Math.floor(k/2)&&s===1) mode="same";
    if(p===k-1&&s===1) mode="full";
    out.innerHTML=`o = ⌊(${i} + 2·${p} − ${k}) / ${s}⌋ + 1 = ⌊${num}/${s}⌋ + 1 = <b>${o}</b>`
      + ` &nbsp; → output ${o} × ${o} &nbsp; <span class="muted">(${mode})</span>`;
  }
  ids.forEach(x=>document.getElementById(x).addEventListener("input",upd)); upd();
})();
</script>
"""
    return Figure("padding_modes",
                  "The three common padding choices for a 4 × 4 image and a 3 × 3 kernel at "
                  "unit stride. With valid padding only real pixels are used and the output "
                  "is smaller than the input; with same padding ⌊k/2⌋ rings are added so that "
                  "the size is unchanged; with full padding k − 1 rings are added, so that "
                  "the kernel just overlaps the image at the extremes, as required by the "
                  "mathematical definition. Other sizes can be calculated with the "
                  "calculator.",
                  png=png, html=widget, kind="widget")


def fig_stride_grid(ctx: BuildContext) -> Figure:
    k = 2
    n = IMG_STRIDE.shape[0]
    strides = (1, 2, 3)
    # Real convolutions, so the output values on the page are the ones a student
    # gets by working the same example by hand.
    outs = {}
    for s in strides:
        _, kflip, out, _ = conv2d_steps(IMG_STRIDE, KER_STRIDE, pad=0, stride=s)
        assert out.shape == (output_size(n, k, 0, s),) * 2
        outs[s] = out

    # Both rows share one cell size and one axis extent, so the output really
    # does look smaller rather than being rescaled to fill its panel.
    lim = (-0.6, n - 0.4)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.8))
    for col, s in enumerate(strides):
        o = outs[s].shape[0]
        ax = axes[0, col]
        sync_matrix(ax, IMG_STRIDE, title=f"input 6×6, stride s = {s}", fontsize=9)
        # One ring per kernel position, on the position's top-left cell. Drawing
        # every kernel outline instead overlaps into an unreadable mess at
        # stride 1, which is exactly the case this figure has to keep legible.
        ax.scatter([oj * s for oi in range(o) for oj in range(o)],
                   [oi * s for oi in range(o) for oj in range(o)],
                   s=150, marker="o", facecolors="none", edgecolors=C["red"],
                   linewidths=1.8, zorder=4)
        ax.add_patch(plt.Rectangle((-0.5, -0.5), k, k, fill=False,
                                   ec=C["red"], lw=2.4, zorder=5))
        ax.set_xlim(*lim); ax.set_ylim(lim[1], lim[0])
        ax.set_xlabel(f"{o * o} kernel positions", fontsize=9.5)

        ax = axes[1, col]
        # Formula in the title, not an xlabel: the axes span the full 6-cell
        # extent, so an xlabel would float well below the smaller output block.
        sync_matrix(ax, outs[s], cmap="Greens", fontsize=9,
                    title=f"output {o}×{o}     o = ⌊(6 − 2)/{s}⌋ + 1 = {o}")
        ax.set_xlim(*lim); ax.set_ylim(lim[1], lim[0])
    png = save_mpl(fig, ctx, "stride_grid")

    # Interactive: one slider over stride, input fixed and output shrinking.
    fig2 = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                         subplot_titles=("input 6 × 6 — rings mark kernel positions",
                                         "output"))

    def _traces(s):
        o = outs[s].shape[0]
        return [
            go.Heatmap(z=IMG_STRIDE, text=IMG_STRIDE, texttemplate="%{text}",
                       colorscale="Blues", showscale=False, xgap=1, ygap=1,
                       hoverinfo="skip", xaxis="x", yaxis="y"),
            go.Scatter(x=[oj * s for oi in range(o) for oj in range(o)],
                       y=[oi * s for oi in range(o) for oj in range(o)],
                       mode="markers", hoverinfo="skip", showlegend=False,
                       marker=dict(size=15, color="rgba(0,0,0,0)",
                                   line=dict(color=C["red"], width=2)),
                       xaxis="x", yaxis="y"),
            go.Heatmap(z=outs[s], text=outs[s], texttemplate="%{text}",
                       colorscale="Greens", showscale=False, xgap=1, ygap=1,
                       hoverinfo="skip", xaxis="x2", yaxis="y2"),
        ]

    frames = []
    for s in strides:
        o = outs[s].shape[0]
        frames.append(go.Frame(name=str(s), data=_traces(s), traces=[0, 1, 2],
                               layout=go.Layout(shapes=[dict(
                                   type="rect", xref="x", yref="y",
                                   x0=-0.5, x1=k - 0.5, y0=-0.5, y1=k - 0.5,
                                   line=dict(color=C["red"], width=3))],
                                   title=f"stride s = {s}:  "
                                         f"o = ⌊(6 − 2)/{s}⌋ + 1 = {o}  →  "
                                         f"output {o} × {o} "
                                         f"({o * o} values from {o * o} positions)")))
    for tr, (r_, c_) in zip(_traces(strides[0]), [(1, 1), (1, 1), (1, 2)]):
        fig2.add_trace(tr, row=r_, col=c_)
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(strides, "stride s = "),
                       shapes=frames[0].layout.shapes, showlegend=False,
                       title=frames[0].layout.title.text)
    # Identical range and square cells on both panels: that is what makes the
    # output visibly shrink instead of being stretched to fill its own panel.
    for c_, anchor in ((1, "x"), (2, "x2")):
        fig2.update_xaxes(range=[-0.5, n - 0.5], showticklabels=False,
                          showgrid=False, zeroline=False, row=1, col=c_)
        fig2.update_yaxes(range=[n - 0.5, -0.5], showticklabels=False,
                          showgrid=False, zeroline=False,
                          scaleanchor=anchor, scaleratio=1, row=1, col=c_)
    return Figure("stride_grid",
                  "Stride is the distance moved by the kernel between positions. The same "
                  "6 × 6 input is convolved with the 2 × 2 kernel "
                  f"{KER_STRIDE.tolist()} at three strides, with no padding. In the top row, "
                  "each ring marks the top-left cell of one kernel position and the bold "
                  "square is the kernel at the first position, so the number of positions is "
                  "reduced from 25 to 9 to 4. In the bottom row the resulting output is drawn "
                  "at the same cell size, so the reduction from 5 × 5 to 3 × 3 to 2 × 2 is "
                  "shown to scale. A larger stride therefore results in both a smaller output "
                  "and less computation, and is used in convolutional networks to downsample "
                  "without a separate pooling layer. In the interactive version the stride is "
                  "set with the slider.",
                  png=png, html=plotly_html(fig2, height=470), kind="plotly")


def fig_pooling_ops(ctx: BuildContext) -> Figure:
    mx = pool2d(POOL_IN, 2, 2, "max")
    av = pool2d(POOL_IN, 2, 2, "mean")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6),
                             gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    sync_matrix(axes[0], POOL_IN, title="input 4×4, 2×2 window, stride 2")
    for bi in range(2):
        for bj in range(2):
            axes[0].add_patch(plt.Rectangle((bj * 2 - 0.5, bi * 2 - 0.5), 2, 2,
                                            fill=False, ec=C["red"], lw=2))
    sync_matrix(axes[1], mx, title="(a) max pooling", cmap="Oranges", fmt="{:g}")
    sync_matrix(axes[2], av, title="(b) average pooling", cmap="Greens", fmt="{:.2f}")
    png = save_mpl(fig, ctx, "pooling_ops")
    return Figure("pooling_ops",
                  "In pooling, a window is moved over the input in the same stepwise manner "
                  "as in convolution, but there is no kernel and no multiplication: each "
                  "window is replaced by the maximum (a) or the mean (b) of the values within "
                  "it. With a 2 × 2 window and stride 2, each spatial dimension is halved. As "
                  "there is no kernel, a pooling layer has no learnable parameters. Max "
                  "pooling is also non-linear, whereas convolution is linear.",
                  png=png)


def fig_corr_vs_conv(ctx: BuildContext) -> Figure:
    img = np.array([[1, 2, 0], [0, 1, 3], [2, 1, 0]])
    k_asym = np.array([[1, 2], [0, -1]])
    k_sym = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2))
    for r, (ker, name) in enumerate([(k_asym, "asymmetric kernel"),
                                     (k_sym, "symmetric kernel")]):
        sync_matrix(axes[r, 0], img, title="image")
        sync_matrix(axes[r, 1], ker, title=f"h, {name}", cmap="Oranges")
        sync_matrix(axes[r, 2], correlate2d(img, ker, mode="valid"),
                    title="cross-correlation (no flip)", cmap="Greens")
        sync_matrix(axes[r, 3], convolve2d(img, ker, mode="valid"),
                    title="convolution (kernel flipped)", cmap="Purples")
    png = save_mpl(fig, ctx, "corr_vs_conv")
    return Figure("corr_vs_conv",
                  "In cross-correlation the kernel is applied as given, whereas in "
                  "convolution it is first flipped along both axes. With an asymmetric "
                  "kernel (top) the two results differ. With a symmetric kernel (bottom) the "
                  "flip has no effect and the results are identical. Correlation is a measure "
                  "of the similarity between the image and the kernel. In deep learning "
                  "frameworks the operation is labelled \"convolution\" but correlation is "
                  "almost always computed; as the kernel weights are learned, it makes no "
                  "difference whether they are stored flipped.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_conv2d_flip_slide(ctx), fig_padding_modes(ctx), fig_stride_grid(ctx),
            fig_pooling_ops(ctx), fig_corr_vs_conv(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("1.1 2D convolution")
    tab1, tab2, tab3 = st.tabs(["Convolution", "Output size", "Pooling"])

    with tab1:
        st.markdown("Edit the image and kernel; the flipped kernel slides over the "
                    "padded input.")
        c1, c2 = st.columns([2, 1])
        img_txt = c1.text_area("image (one row per line)",
                               "\n".join(" ".join(map(str, r)) for r in IMG_2D), height=120)
        ker_txt = c2.text_area("kernel",
                               "\n".join(" ".join(map(str, r)) for r in KER_2D), height=120)
        try:
            img = np.array([[int(v) for v in r.split()] for r in img_txt.strip().splitlines()])
            ker = np.array([[int(v) for v in r.split()] for r in ker_txt.strip().splitlines()])
        except ValueError:
            st.error("Use whitespace-separated integers."); return
        c1, c2 = st.columns(2)
        mode = c1.radio("padding", ["valid", "same", "full"], horizontal=True)
        stride = c2.slider("stride", 1, 3, 1)
        p = {"valid": 0, "same": ker.shape[0] // 2, "full": ker.shape[0] - 1}[mode]
        P, kf, out, steps = conv2d_steps(img, ker, pad=p, stride=stride)
        idx = st.slider("step", 0, len(steps) - 1, 0)
        s = steps[idx]
        fig, axes = plt.subplots(1, 4, figsize=(11, 3))
        sync_matrix(axes[0], kf, title="flipped kernel", cmap="Oranges")
        sync_matrix(axes[1], P, title="padded input",
                    highlight=(s["i"] * stride, s["j"] * stride, *ker.shape))
        sync_matrix(axes[2], s["prod"], title="products", cmap="Purples")
        partial = np.full(out.shape, np.nan)
        for t in steps[:idx + 1]:
            partial[t["i"], t["j"]] = t["val"]
        sync_matrix(axes[3], partial, title="output", cmap="Greens")
        st.pyplot(fig)
        st.write(f"output({s['i']},{s['j']}) = {s['val']} · "
                 f"full output size {out.shape[0]} × {out.shape[1]}")

    with tab2:
        st.markdown("The general relationship "
                    r"$o = \lfloor (i + 2p - k)/s \rfloor + 1$.")
        c = st.columns(4)
        i = c[0].number_input("input i", 1, 512, 8)
        k = c[1].number_input("kernel k", 1, 31, 3)
        p = c[2].number_input("padding p", 0, 16, 0)
        s = c[3].number_input("stride s", 1, 8, 1)
        if i + 2 * p - k < 0:
            st.error("The kernel is larger than the padded input.")
        else:
            o = output_size(int(i), int(k), int(p), int(s))
            st.latex(rf"o = \left\lfloor \frac{{{i} + 2\cdot{p} - {k}}}{{{s}}} \right\rfloor"
                     rf" + 1 = {o}")
            named = []
            if p == 0:
                named.append("valid")
            if k % 2 == 1 and p == k // 2 and s == 1:
                named.append("same")
            if p == k - 1 and s == 1:
                named.append("full")
            st.write("Named case:", ", ".join(named) if named else "none of valid/same/full")

    with tab3:
        txt = st.text_area("input (one row per line)",
                           "\n".join(" ".join(map(str, r)) for r in POOL_IN), height=120)
        try:
            x = np.array([[float(v) for v in r.split()] for r in txt.strip().splitlines()])
        except ValueError:
            st.error("Use whitespace-separated numbers."); return
        c1, c2 = st.columns(2)
        size = c1.slider("window", 2, 4, 2)
        stride = c2.slider("stride", 1, 4, 2)
        if min(x.shape) < size:
            st.error("The window is larger than the input."); return
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        sync_matrix(axes[0], x, title="input", fmt="{:g}")
        sync_matrix(axes[1], pool2d(x, size, stride, "max"), title="max", cmap="Oranges",
                    fmt="{:g}")
        sync_matrix(axes[2], pool2d(x, size, stride, "mean"), title="average",
                    cmap="Greens", fmt="{:.2f}")
        st.pyplot(fig)
        st.info("Pooling has no learnable parameters.")
