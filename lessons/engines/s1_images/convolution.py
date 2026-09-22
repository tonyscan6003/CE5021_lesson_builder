"""Lesson 1.2 — Image processing with convolution.

Figures
    kernel_gallery     common kernels applied to a photo
    conv1d_slide       1D convolution: kernel reversal and shift for every n (interactive)
    corr_vs_conv       cross-correlation and convolution differ unless the kernel is symmetric
    conv2d_flip_slide  2D convolution: flip the kernel, slide it, multiply and sum (interactive)
    padding_modes      valid / same / full padding and output size, plus a calculator widget
"""
from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import convolve2d, correlate2d

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout, load_image, sync_matrix)

FIGURES = ["kernel_gallery", "conv1d_slide", "corr_vs_conv",
           "conv2d_flip_slide", "padding_modes"]

# Fixed teaching examples ----------------------------------------------------
X_VALS, X_START = np.array([1, 2, 3, 1]), 0        # x[n], n = 0..3
H_VALS, H_START = np.array([1, 0, 2]), -1          # h[n], n = -1..1

IMG_2D = np.array([[1, 2, 0, 1],
                   [0, 1, 3, 1],
                   [2, 1, 0, 2],
                   [1, 0, 1, 1]])
KER_2D = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])                    # asymmetric on purpose

KERNELS = {
    "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
    "Box blur 5×5": np.ones((5, 5)) / 25,
    "Gaussian 5×5 (σ=1)": (lambda k: k @ k.T)(cv2.getGaussianKernel(5, 1.0)),
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    "Laplacian (edges)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
    "Sobel x (vertical edges)": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float),
}


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def conv1d_tables(x_vals, x_start, h_vals, h_start):
    """All the intermediate rows of a 1D convolution written in the dummy variable k."""
    x_end = x_start + len(x_vals) - 1
    h_end = h_start + len(h_vals) - 1
    n_range = list(range(x_start + h_start, x_end + h_end + 1))
    k_range = list(range(n_range[0] - 1, n_range[-1] + 2))

    def x_at(k):
        i = k - x_start
        return int(x_vals[i]) if 0 <= i < len(x_vals) else 0

    def h_at(m):
        i = m - h_start
        return int(h_vals[i]) if 0 <= i < len(h_vals) else 0

    x_row = [x_at(k) for k in k_range]
    rows = {}
    for n in n_range:
        h_row = [h_at(n - k) for k in k_range]
        prod = [a * b for a, b in zip(x_row, h_row)]
        rows[n] = dict(h=h_row, prod=prod, total=int(sum(prod)))
    return dict(k=k_range, x=x_row, n_range=n_range, rows=rows,
                h_rev=[h_at(-k) for k in k_range], h_k=[h_at(k) for k in k_range])


def conv2d_steps(img, ker, pad=1):
    """Padded image, flipped kernel and per-position products for 'same' convolution."""
    kf = ker[::-1, ::-1]
    P = np.pad(img, pad)
    kr, kc = ker.shape
    out = np.zeros((P.shape[0] - kr + 1, P.shape[1] - kc + 1), int)
    steps = []
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = P[i:i + kr, j:j + kc]
            prod = patch * kf
            out[i, j] = prod.sum()
            steps.append(dict(i=i, j=j, patch=patch.copy(), prod=prod, val=int(out[i, j])))
    assert np.array_equal(out, convolve2d(img, ker, mode="same"))
    return P, kf, out, steps


def output_size(i, k, p, s):
    return (i + 2 * p - k) // s + 1


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_kernel_gallery(ctx: BuildContext) -> Figure:
    g = load_image("sphinx", gray=True, max_side=520)
    g = g[40:300, 120:460].astype(np.float32)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for ax, (name, ker) in zip(axes.ravel(), KERNELS.items()):
        out = convolve2d(g, ker, mode="same", boundary="symm")
        if name.startswith(("Laplacian", "Sobel")):
            out = np.abs(out)
            ax.imshow(out, vmin=0, vmax=np.percentile(out, 99))
        else:
            ax.imshow(out, vmin=0, vmax=255)
        rows = ["[" + " ".join(f"{v:g}" for v in r) + "]" for r in ker]
        if ker.shape[0] > 3:
            rows = [f"{ker.shape[0]}×{ker.shape[1]}, all {ker[0,0]:.2g}"] if name.startswith("Box") \
                else [f"{ker.shape[0]}×{ker.shape[1]} Gaussian"]
        ax.set_title(name + "\n" + "\n".join(rows), fontsize=8.5, family="monospace")
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "kernel_gallery")
    return Figure("kernel_gallery",
                  "The same greyscale image convolved with six small kernels. A "
                  "kernel is a small array of weights; changing the weights turns "
                  "the same operation into a blur, a sharpener or an edge detector. "
                  "Edge outputs are shown as magnitudes.", png=png)


def fig_conv1d_slide(ctx: BuildContext) -> Figure:
    T = conv1d_tables(X_VALS, X_START, H_VALS, H_START)
    k = T["k"]
    n_range = T["n_range"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharex=True)
    axes = axes.ravel()

    def stems(ax, vals, color, label, offset=0.0, w=0.25):
        ax.bar(np.array(k) + offset, vals, width=w, color=color, label=label)

    ax = axes[0]
    stems(ax, T["x"], C["grey"], "x[k]", -0.15)
    stems(ax, T["h_k"], C["blue"], "h[k]", 0.15)
    ax.set_title("x[k] and kernel h[k]"); ax.legend(fontsize=8)
    ax = axes[1]
    stems(ax, T["x"], C["grey"], "x[k]", -0.15)
    stems(ax, T["h_rev"], C["orange"], "h[−k] (reversed)", 0.15)
    ax.set_title("Reverse the kernel: h[−k]"); ax.legend(fontsize=8)
    for ax, n in zip(axes[2:], n_range):
        row = T["rows"][n]
        stems(ax, T["x"], C["grey"], "x[k]", -0.15)
        stems(ax, row["h"], C["orange"], f"h[{n}−k]", 0.15)
        for kk, p in zip(k, row["prod"]):
            if p != 0:
                ax.text(kk, max(T["x"][k.index(kk)], row["h"][k.index(kk)]) + 0.15, f"{p}",
                        ha="center", fontsize=8, color=C["red"])
        terms = " + ".join(f"{a}·{b}" for a, b in zip(T["x"], row["h"]) if b != 0)
        ax.set_title(f"n = {n}:  (x∗h)[{n}] = {terms} = {row['total']}", fontsize=8.5)
        ax.legend(fontsize=8, loc="upper right")
    for ax in axes:
        ax.set_xticks(k); ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(0, 4.2)
    for ax in axes[4:]:
        ax.set_xlabel("k")
    png = save_mpl(fig, ctx, "conv1d_slide")

    # interactive: slider over n
    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=("x[k]", "reversed and shifted kernel h[n−k]",
                                         "products x[k]·h[n−k]"))
    frames = []
    for n in n_range:
        row = T["rows"][n]
        frames.append(go.Frame(name=str(n), data=[
            go.Bar(x=k, y=T["x"], marker_color=C["grey"]),
            go.Bar(x=k, y=row["h"], marker_color=C["orange"]),
            go.Bar(x=k, y=row["prod"], marker_color=C["red"]),
        ], traces=[0, 1, 2], layout=go.Layout(
            title=f"n = {n}:  (x∗h)[{n}] = Σ x[k]·h[{n}−k] = {row['total']}")))
    for tr in frames[0].data:
        fig2.add_trace(tr, row=len(fig2.data) + 1, col=1)
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(n_range, "n = "), showlegend=False,
                       title=frames[0].layout.title.text)
    fig2.update_xaxes(tickmode="array", tickvals=k, title_text="k", row=3, col=1)
    fig2.update_yaxes(range=[-0.2, 6.5], row=3, col=1)
    for r in (1, 2):
        fig2.update_yaxes(range=[0, 3.5], row=r, col=1)
    return Figure("conv1d_slide",
                  f"1D convolution of x[n] = {X_VALS.tolist()} (n = 0…3) with "
                  f"h[n] = {H_VALS.tolist()} (n = −1…1). Top left: both sequences in "
                  "the dummy variable k. Top second: the reversed kernel h[−k]. The "
                  "remaining panels shift the reversed kernel to each output index n, "
                  "multiply overlapping samples (red numbers) and sum.",
                  png=png, html=plotly_html(fig2, height=600), kind="plotly")


def fig_corr_vs_conv(ctx: BuildContext) -> Figure:
    img = np.array([[1, 2, 0], [0, 1, 3], [2, 1, 0]])
    k_asym = np.array([[1, 2], [0, -1]])
    k_sym = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2))
    for r, (ker, name) in enumerate([(k_asym, "asymmetric kernel"), (k_sym, "symmetric kernel")]):
        sync_matrix(axes[r, 0], img, title="image")
        sync_matrix(axes[r, 1], ker, title=f"h, {name}", cmap="Oranges")
        sync_matrix(axes[r, 2], correlate2d(img, ker, mode="valid"),
                    title="cross-correlation (no flip)", cmap="Greens")
        sync_matrix(axes[r, 3], convolve2d(img, ker, mode="valid"),
                    title="convolution (kernel flipped)", cmap="Purples")
    png = save_mpl(fig, ctx, "corr_vs_conv")
    return Figure("corr_vs_conv",
                  "Cross-correlation slides the kernel as given; convolution first "
                  "flips it in both axes. With an asymmetric kernel (top) the two "
                  "results differ. With a symmetric kernel (bottom) the flip changes "
                  "nothing and the results are identical, which is why most image "
                  "processing libraries quietly compute correlation.", png=png)


def fig_conv2d_flip_slide(ctx: BuildContext) -> Figure:
    P, kf, out, steps = conv2d_steps(IMG_2D, KER_2D, pad=1)
    kr, kc = KER_2D.shape
    show = steps[5]  # position (1,1)
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.8, 1, 1.4]))
    sync_matrix(axes[0], KER_2D, title="(a) kernel h", cmap="Oranges")
    sync_matrix(axes[1], kf, title="(b) flipped kernel h'", cmap="Oranges")
    Pn = P.astype(float)
    sync_matrix(axes[2], Pn, title="(c) zero-padded image, window at (1,1)",
                highlight=(show["i"], show["j"], kr, kc))
    sync_matrix(axes[3], show["prod"], title="(d) patch × h'", cmap="Purples")
    partial = np.full(out.shape, np.nan)
    for s in steps[:6]:
        partial[s["i"], s["j"]] = s["val"]
    sync_matrix(axes[4], partial, title="(e) output so far", cmap="Greens")
    terms = " + ".join(f"{a}·{b}" for a, b in zip(show["patch"].ravel(), kf.ravel()) if b != 0)
    axes[2].set_xlabel(f"output(1,1) = {terms} = {show['val']}", fontsize=9.5)
    png = save_mpl(fig, ctx, "conv2d_flip_slide")

    # interactive: slider over all output positions
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
            go.Heatmap(z=s["prod"], text=s["prod"], texttemplate="%{text}", colorscale="Purples",
                       showscale=False, zmid=0),
            go.Heatmap(z=vals.copy(), text=vals.copy(), texttemplate="%{text}",
                       colorscale="Greens", showscale=False, zmid=0),
        ], traces=[0, 1, 2], layout=go.Layout(shapes=[shape],
            title=f"output({s['i']},{s['j']}) = {terms} = {s['val']}")))
    for c_, tr in enumerate(frames[0].data, start=1):
        fig2.add_trace(tr, row=1, col=c_)
    fig2.frames = frames
    fig2.update_layout(**slider_frames_layout([f.name for f in frames], "position "),
                       shapes=frames[0].layout.shapes, title=frames[0].layout.title.text)
    for c_ in (1, 2, 3):
        fig2.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=c_)
        fig2.update_xaxes(showticklabels=False, row=1, col=c_)
    return Figure("conv2d_flip_slide",
                  "2D convolution of a 4 × 4 image with a 3 × 3 kernel using one ring "
                  "of zero padding (\"same\" output size). (a) The kernel is flipped "
                  "horizontally and vertically to give (b). (c) The flipped kernel is "
                  "placed over each 3 × 3 window of the padded image, (d) multiplied "
                  "element-wise and summed, giving one output pixel (e).",
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
                ax.add_patch(plt.Rectangle((c_, n - 1 - r), 1, 1, fc=C["light"] if inside else "white",
                                           ec=C["grey"], lw=0.8, hatch=None if inside else "///"))
        # first and last kernel positions
        ax.add_patch(plt.Rectangle((0, n - k), k, k, fill=False, ec=C["red"], lw=2.5))
        ax.add_patch(plt.Rectangle((n - k, 0), k, k, fill=False, ec=C["orange"], lw=2.5, ls="--"))
        ax.set_xlim(-0.2, n + 0.2); ax.set_ylim(-0.2, n + 0.2); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{name}\ninput {i}×{i}, kernel {k}×{k} → output {o}×{o}", fontsize=9.5)
    fig.suptitle("Grey = image pixels, hatched = zero padding; red = first kernel position, "
                 "orange = last", fontsize=9)
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
    let mode="custom padding"; if(p===0) mode="valid"; if(k%2===1&&p===Math.floor(k/2)&&s===1) mode="same";
    if(p===k-1&&s===1) mode="full";
    out.innerHTML=`o = ⌊(${i} + 2·${p} − ${k}) / ${s}⌋ + 1 = ⌊${num}/${s}⌋ + 1 = <b>${o}</b> &nbsp; → output ${o} × ${o} &nbsp; <span class="muted">(${mode})</span>`;
  }
  ids.forEach(x=>document.getElementById(x).addEventListener("input",upd)); upd();
})();
</script>
"""
    return Figure("padding_modes",
                  "The three common padding choices for a 4 × 4 image and 3 × 3 kernel "
                  "with unit stride. Valid padding uses only real pixels and shrinks the "
                  "output; same padding adds ⌊k/2⌋ rings to keep the size; full padding "
                  "adds k − 1 rings so the kernel just overlaps the image at the "
                  "extremes, matching the mathematical definition. Use the calculator "
                  "below to try other sizes.", png=png, html=widget, kind="widget")


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_kernel_gallery(ctx), fig_conv1d_slide(ctx), fig_corr_vs_conv(ctx),
            fig_conv2d_flip_slide(ctx), fig_padding_modes(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("1.2 Convolution")
    tab1, tab2, tab3 = st.tabs(["1D convolution", "2D convolution", "Kernels on an image"])

    with tab1:
        st.markdown("Enter a sequence and a kernel, then step through every output index n.")
        c1, c2 = st.columns(2)
        xs = c1.text_input("x[n] values", "1, 2, 3, 1")
        xstart = c1.number_input("x starts at n =", value=0, step=1)
        hs = c2.text_input("h[n] values", "1, 0, 2")
        hstart = c2.number_input("h starts at n =", value=-1, step=1)
        try:
            xv = np.array([int(v) for v in xs.split(",")])
            hv = np.array([int(v) for v in hs.split(",")])
        except ValueError:
            st.error("Use comma-separated integers."); return
        T = conv1d_tables(xv, int(xstart), hv, int(hstart))
        n = st.select_slider("output index n", options=T["n_range"], value=T["n_range"][0])
        row = T["rows"][n]
        fig, axes = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
        axes[0].bar(T["k"], T["x"], color=C["grey"]); axes[0].set_title("x[k]")
        axes[1].bar(T["k"], row["h"], color=C["orange"]); axes[1].set_title(f"h[{n}−k]")
        axes[2].bar(T["k"], row["prod"], color=C["red"]); axes[2].set_title("products")
        axes[2].set_xticks(T["k"]); axes[2].set_xlabel("k")
        st.pyplot(fig)
        st.latex(rf"(x*h)[{n}] = \sum_k x[k]\,h[{n}-k] = {row['total']}")
        st.write("Full result:", {n_: T["rows"][n_]["total"] for n_ in T["n_range"]})

    with tab2:
        st.markdown("Edit the image and kernel; the flipped kernel slides over the padded image.")
        c1, c2 = st.columns([2, 1])
        img_txt = c1.text_area("image (rows separated by newlines)",
                               "\n".join(" ".join(map(str, r)) for r in IMG_2D), height=120)
        ker_txt = c2.text_area("kernel", "\n".join(" ".join(map(str, r)) for r in KER_2D), height=120)
        try:
            img = np.array([[int(v) for v in r.split()] for r in img_txt.strip().splitlines()])
            ker = np.array([[int(v) for v in r.split()] for r in ker_txt.strip().splitlines()])
        except ValueError:
            st.error("Use whitespace-separated integers."); return
        pad = st.radio("padding", ["valid (0)", "same", "full"], horizontal=True)
        p = {"valid (0)": 0, "same": ker.shape[0] // 2, "full": ker.shape[0] - 1}[pad]
        P, kf, out, steps = conv2d_steps(img, ker, pad=p) if p else conv2d_steps_valid(img, ker)
        idx = st.slider("step", 0, len(steps) - 1, 0)
        s = steps[idx]
        fig, axes = plt.subplots(1, 4, figsize=(11, 3))
        sync_matrix(axes[0], kf, title="flipped kernel", cmap="Oranges")
        sync_matrix(axes[1], P, title="padded image", highlight=(s["i"], s["j"], *ker.shape))
        sync_matrix(axes[2], s["prod"], title="products", cmap="Purples")
        partial = np.full(out.shape, np.nan)
        for t in steps[:idx + 1]:
            partial[t["i"], t["j"]] = t["val"]
        sync_matrix(axes[3], partial, title="output", cmap="Greens")
        st.pyplot(fig)
        st.write(f"output({s['i']},{s['j']}) = {s['val']}")

    with tab3:
        g = load_image("sphinx", gray=True, max_side=640).astype(np.float32)
        name = st.selectbox("kernel", list(KERNELS))
        ker = KERNELS[name]
        out = convolve2d(g, ker, mode="same", boundary="symm")
        c1, c2 = st.columns(2)
        c1.image(g / 255, caption="input", width="stretch")
        if name.startswith(("Laplacian", "Sobel")):
            out = np.abs(out); out = np.clip(out / np.percentile(out, 99), 0, 1)
        else:
            out = np.clip(out / 255, 0, 1)
        c2.image(out, caption=name, width="stretch")
        st.code(np.array2string(ker, precision=3, suppress_small=True))


def conv2d_steps_valid(img, ker):
    """Valid-padding variant of conv2d_steps (no zero ring)."""
    kf = ker[::-1, ::-1]
    kr, kc = ker.shape
    out = np.zeros((img.shape[0] - kr + 1, img.shape[1] - kc + 1), int)
    steps = []
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = img[i:i + kr, j:j + kc]
            prod = patch * kf
            out[i, j] = prod.sum()
            steps.append(dict(i=i, j=j, patch=patch.copy(), prod=prod, val=int(out[i, j])))
    return img, kf, out, steps
