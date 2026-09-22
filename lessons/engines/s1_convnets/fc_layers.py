"""Lesson 1.2 — Fully connected layers and classifiers.

Figures
    single_neuron        weights, bias, sum and activation with numbers flowing through
    activation_functions sigmoid, tanh, ReLU and leaky ReLU with their derivatives
    fc_network           a one-hidden-layer network, parameters counted per layer
    nn_classifiers       binary (sigmoid) and multi-class (softmax) output layers
    softmax_activation   the slide worked example, plus a live logits widget
    fc_param_count       flatten n x m x C then connect to N neurons (interactive)

Source: reference/legacy_md/lesson_1_2_FC_layers + lesson_1_2_script.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout)

FIGURES = ["single_neuron", "activation_functions", "fc_network",
           "nn_classifiers", "softmax_activation", "fc_param_count"]

# Slide 2: a three-input neuron.
NEURON_X = np.array([2.6, -0.7, -1.6])
NEURON_W = np.array([0.3, 0.8, 0.9])
NEURON_B = 1.9

# Slide 7: the softmax example worked on the slide.
SOFTMAX_Z = np.array([4.0, 3.0, 3.0, 2.0])


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    e = np.exp(np.asarray(z, float) - np.max(z))     # shift for numerical stability
    return e / e.sum()


ACTIVATIONS = {
    "sigmoid": (lambda z: sigmoid(z), lambda z: sigmoid(z) * (1 - sigmoid(z))),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "ReLU": (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
    "leaky ReLU (α=0.1)": (lambda z: np.where(z > 0, z, 0.1 * z),
                           lambda z: np.where(z > 0, 1.0, 0.1)),
}


def fc_params(n, m, channels, neurons):
    """Weights and biases when an n x m x C feature map is flattened into N neurons."""
    inputs = n * m * channels
    return dict(inputs=inputs, weights=inputs * neurons, biases=neurons,
                total=inputs * neurons + neurons)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_single_neuron(ctx: BuildContext) -> Figure:
    x, w, b = NEURON_X, NEURON_W, NEURON_B
    z = float(x @ w + b)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ys = [2.0, 1.0, 0.0]
    for xi, wi, y in zip(x, w, ys):
        ax.text(0.0, y, f"$x_{{{ys.index(y) + 1}}}$ = {xi:g}", ha="right", va="center",
                fontsize=11)
        ax.annotate("", xy=(2.55, 1.0), xytext=(0.15, y),
                    arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.4))
        ax.text(1.35, (y + 1.0) / 2 + 0.11, f"× {wi:g}", ha="center", fontsize=10,
                color=C["blue"])
    ax.add_patch(plt.Circle((2.85, 1.0), 0.3, fc=C["light"], ec=C["ink"], lw=1.5))
    ax.text(2.85, 1.0, "Σ", ha="center", va="center", fontsize=15)
    ax.text(2.85, 0.5, f"bias b = {b:g}", ha="center", fontsize=10, color=C["orange"])
    ax.annotate("", xy=(2.85, 0.78), xytext=(2.85, 0.62),
                arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.4))
    ax.annotate("", xy=(3.85, 1.0), xytext=(3.2, 1.0),
                arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.4))
    ax.text(3.5, 1.16, f"z = {z:.3f}", ha="center", fontsize=10, color=C["red"])
    ax.add_patch(plt.Rectangle((3.85, 0.65), 0.75, 0.7, fc="white", ec=C["ink"], lw=1.5))
    ax.text(4.22, 1.0, "σ", ha="center", va="center", fontsize=15)
    ax.annotate("", xy=(5.4, 1.0), xytext=(4.6, 1.0),
                arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.4))
    ax.text(5.5, 1.0, f"y = σ(z)", ha="left", va="center", fontsize=11)
    ax.text(2.9, -0.55,
            "z = " + " + ".join(f"({xi:g})({wi:g})" for xi, wi in zip(x, w))
            + f" + {b:g} = {z:.3f}\n"
            f"linear σ(z) = z = {z:.3f}          "
            f"sigmoid σ(z) = 1/(1 + e^−z) = {sigmoid(z):.2f}",
            ha="center", va="center", fontsize=10.5, family="monospace")
    ax.set_xlim(-1.1, 6.6); ax.set_ylim(-1.1, 2.5); ax.axis("off")
    png = save_mpl(fig, ctx, "single_neuron")
    return Figure("single_neuron",
                  f"A single artificial neuron with inputs {x.tolist()} and weights "
                  f"{w.tolist()}. Each input is multiplied by its own weight, the "
                  f"products are summed, the bias b = {b:g} is added, and the "
                  "activation function σ is applied to the result. With a linear "
                  f"activation the output is just z = {z:.3f}; with a sigmoid it is "
                  f"{sigmoid(z):.2f}. The weights and the bias are the neuron's "
                  "learnable parameters — four of them here.", png=png)


def fig_activation_functions(ctx: BuildContext) -> Figure:
    z = np.linspace(-6, 6, 601)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (name, (f, df)), col in zip(ACTIVATIONS.items(), SERIES):
        # leaky ReLU coincides exactly with ReLU for z > 0, so dash it to keep
        # both visible rather than having one drawn on top of the other
        style = dict(ls="--", lw=2.4) if name.startswith("leaky") else dict(lw=2)
        axes[0].plot(z, f(z), color=col, label=name, **style)
        axes[1].plot(z, df(z), color=col, label=name, **style)
    for zz in (-6, 6):
        axes[0].axvspan(zz - 0 if zz > 0 else -6, 6 if zz > 0 else -4, alpha=0)
    axes[0].axvspan(-6, -4, color=C["red"], alpha=0.07)
    axes[0].axvspan(4, 6, color=C["red"], alpha=0.07)
    axes[0].text(-5.0, 0.72, "saturation", fontsize=8.5, color=C["red"], ha="center")
    axes[0].text(5.0, 0.72, "saturation", fontsize=8.5, color=C["red"], ha="center")
    axes[0].set_title("activation σ(z)")
    axes[1].set_title("derivative σ′(z) — the factor backpropagation multiplies by")
    for ax in axes:
        ax.axhline(0, color=C["grey"], lw=0.7)
        ax.axvline(0, color=C["grey"], lw=0.7)
        ax.set_xlabel("z"); ax.legend(fontsize=9)
    axes[0].set_ylim(-1.6, 3.2); axes[1].set_ylim(-0.1, 1.15)
    png = save_mpl(fig, ctx, "activation_functions")
    return Figure("activation_functions",
                  "The activation function is what makes a network non-linear; without "
                  "it a stack of layers would collapse into a single linear map. "
                  "Sigmoid and tanh were used in early networks and are still used "
                  "where a bounded output is wanted, but they are slow to compute and "
                  "they saturate: for |z| beyond about 4 the derivative (right) is "
                  "near zero, so almost no gradient flows back and learning stalls. "
                  "ReLU and leaky ReLU are fast and do not saturate for positive z, "
                  "which is why they are the default in deep networks. Leaky ReLU "
                  "keeps a small gradient for negative z so a unit cannot go "
                  "permanently silent.", png=png)


def fig_fc_network(ctx: BuildContext) -> Figure:
    layers = [(3, "input\n3 values"), (4, "hidden\n4 neurons"), (2, "output\n2 neurons")]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    pos = []
    for li, (n, _) in enumerate(layers):
        ys = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
        pos.append([(li * 2.4, y) for y in ys])
    for a, b in zip(pos, pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=C["light"], lw=0.9, zorder=1)
    for li, ((n, label), ps) in enumerate(zip(layers, pos)):
        col = C["grey"] if li == 0 else (C["blue"] if li == 1 else C["orange"])
        # scatter, not plt.Circle: marker size is in display units, so the
        # neurons stay round whatever the data aspect ratio turns out to be.
        ax.scatter([p[0] for p in ps], [p[1] for p in ps], s=460,
                   facecolors="white", edgecolors=col, linewidths=2, zorder=2)
        ax.text(ps[0][0], max(p[1] for p in ps) + 0.62, label, ha="center",
                fontsize=10, color=col)
    total = 0
    for li in (1, 2):
        n_in, n_out = layers[li - 1][0], layers[li][0]
        p = n_in * n_out + n_out
        total += p
        ax.text((li * 2.4 - 1.2), -2.35,
                f"{n_in}×{n_out} weights + {n_out} biases = {p}", ha="center", fontsize=9.5)
    ax.text(2.4, -2.95, f"total {total} parameters", ha="center", fontsize=10.5,
            color=C["red"])
    ax.set_xlim(-1.0, 5.8); ax.set_ylim(-3.3, 2.6); ax.axis("off")
    png = save_mpl(fig, ctx, "fc_network")
    return Figure("fc_network",
                  "A one-hidden-layer fully connected network. \"Fully connected\" "
                  "means every neuron in a layer receives every output of the previous "
                  "layer, so each layer contributes (inputs × neurons) weights plus one "
                  "bias per neuron. The whole network is a mathematical function "
                  "mapping the inputs to the outputs, and the parameters of that "
                  "function are what training adjusts.", png=png)


def fig_nn_classifiers(ctx: BuildContext) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    z = np.linspace(-6, 6, 400)
    axes[0].plot(z, sigmoid(z), color=C["blue"], lw=2)
    axes[0].axhline(0.5, color=C["grey"], ls="--", lw=1)
    axes[0].fill_between(z, 0, sigmoid(z), where=sigmoid(z) >= 0.5, color=C["blue"],
                         alpha=0.12)
    axes[0].annotate("confident: pedestrian", xy=(3.2, 0.96), xytext=(-1.2, 1.12),
                     fontsize=9, arrowprops=dict(arrowstyle="->", color=C["grey"]))
    axes[0].annotate("confident: background", xy=(-3.2, 0.04), xytext=(-5.9, 0.34),
                     fontsize=9, arrowprops=dict(arrowstyle="->", color=C["grey"]))
    axes[0].set_title("(a) binary classifier: one sigmoid output")
    axes[0].set_xlabel("z (output-neuron sum)")
    axes[0].set_ylabel("P(pedestrian)")
    axes[0].set_ylim(-0.05, 1.25)

    classes = ["cat", "dog", "horse", "bird", "sheep"]
    p = softmax(np.array([1.3, 1.5, 3.9, 2.9, 1.1]))
    bars = axes[1].bar(classes, p, color=[C["orange"] if v == p.max() else C["light"]
                                          for v in p], edgecolor=C["grey"])
    for b_, v in zip(bars, p):
        axes[1].text(b_.get_x() + b_.get_width() / 2, v + 0.015, f"{v:.2f}",
                     ha="center", fontsize=9.5)
    axes[1].set_title(f"(b) multi-class: softmax over {len(classes)} outputs "
                      f"(Σ = {p.sum():.2f})")
    axes[1].set_ylabel("probability")
    axes[1].set_ylim(0, max(p) * 1.28)
    png = save_mpl(fig, ctx, "nn_classifiers")
    return Figure("nn_classifiers",
                  "A neural network is a probabilistic classifier: it outputs the "
                  "probability that the input belongs to a class rather than assigning "
                  "a label directly. (a) For a binary task one output neuron with a "
                  "sigmoid gives a value between 0 and 1. (b) For a multi-class task "
                  "the last layer has one neuron per class and softmax turns their "
                  "sums into probabilities that add to 1; the predicted class is the "
                  "largest. Note that the last layer does not use an ordinary "
                  "activation function — softmax acts across all the neurons at once, "
                  "not on each independently.", png=png)


def fig_softmax_activation(ctx: BuildContext) -> Figure:
    z = SOFTMAX_Z
    e = np.exp(z)
    p = e / e.sum()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    idx = np.arange(len(z))
    axes[0].bar(idx, z, color=C["grey"])
    axes[0].set_title("$z_i$ (sums from the last layer)")
    axes[1].bar(idx, e, color=C["blue"])
    for i, v in zip(idx, e):
        axes[1].text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=9)
    axes[1].set_title(f"$e^{{z_i}}$   (sum = {e.sum():.1f})")
    axes[1].set_ylim(0, e.max() * 1.22)
    axes[2].bar(idx, p, color=C["orange"])
    for i, v in zip(idx, p):
        axes[2].text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    axes[2].set_title(f"$p_i = e^{{z_i}} / \\sum_j e^{{z_j}}$   (sum = {p.sum():.2f})")
    axes[2].set_ylim(0, p.max() * 1.25)
    for ax in axes:
        ax.set_xticks(idx); ax.set_xticklabels([f"$z_{i+1}$" for i in idx])
    png = save_mpl(fig, ctx, "softmax_activation")

    widget = """
<div class="widget conv-calc">
  <b>Softmax calculator</b> &nbsp;
  <span class="muted">p<sub>i</sub> = e<sup>z<sub>i</sub></sup> / Σ<sub>j</sub> e<sup>z<sub>j</sub></sup></span>
  <div class="widget-row">
    <label>logits z (comma separated)
      <input type="text" id="sm-z" value="4, 3, 3, 2" size="24"></label>
  </div>
  <div id="sm-out" class="widget-out"></div>
</div>
<script>
(function(){
  const inp=document.getElementById("sm-z"), out=document.getElementById("sm-out");
  function upd(){
    const z=inp.value.split(",").map(s=>parseFloat(s.trim())).filter(v=>!isNaN(v));
    if(!z.length){out.textContent="enter some numbers";return;}
    const m=Math.max(...z), e=z.map(v=>Math.exp(v-m)), s=e.reduce((a,b)=>a+b,0);
    const p=e.map(v=>v/s);
    out.innerHTML="<table class='sm-table'><tr><th>i</th>"
      + z.map((_,i)=>`<th>${i+1}</th>`).join("") + "</tr>"
      + "<tr><td>z<sub>i</sub></td>" + z.map(v=>`<td>${v}</td>`).join("") + "</tr>"
      + "<tr><td>p<sub>i</sub></td>" + p.map(v=>`<td><b>${v.toFixed(3)}</b></td>`).join("")
      + "</tr></table>"
      + `<span class="muted">Σ p<sub>i</sub> = ${p.reduce((a,b)=>a+b,0).toFixed(3)}`
      + ` &nbsp;·&nbsp; predicted class: ${p.indexOf(Math.max(...p))+1}</span>`;
  }
  inp.addEventListener("input",upd); upd();
})();
</script>
"""
    return Figure("softmax_activation",
                  "The softmax calculation worked through for the four logits on the "
                  f"slide, z = {[int(v) for v in z]}. Exponentiating gives "
                  f"{', '.join(f'{v:.1f}' for v in e)}, which sum to {e.sum():.1f}; "
                  "dividing each by that sum gives "
                  f"{', '.join(f'{v:.2f}' for v in p)}. The probabilities add to 1, "
                  "which is the check worth doing every time. Note that equal logits "
                  "give equal probabilities, and that the exponential exaggerates "
                  "differences — a logit one unit higher is about 2.7 times more "
                  "likely. Try your own values below.",
                  png=png, html=widget, kind="widget")


def fig_fc_param_count(ctx: BuildContext) -> Figure:
    n = m = 7
    channels = 4
    neurons = 10
    r = fc_params(n, m, channels, neurons)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw=dict(width_ratios=[1, 1.15]))

    ax = axes[0]
    for c_ in range(channels):
        off = c_ * 0.34
        ax.add_patch(plt.Rectangle((off, -off), 2.0, 2.0, fc=C["light"],
                                   ec=C["grey"], lw=1.2, alpha=0.95))
    ax.text(1.0 + 0.34 * (channels - 1) / 2, 2.55,
            f"feature map {n} × {m} × {channels}", ha="center", fontsize=10)
    ax.annotate("", xy=(4.3, 0.55), xytext=(3.3, 0.55),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.6))
    ax.text(3.8, 0.78, "flatten", ha="center", fontsize=9.5)
    ax.add_patch(plt.Rectangle((4.45, -0.7), 0.42, 2.5, fc=C["blue"], ec=C["ink"],
                               lw=1.2, alpha=0.65))
    ax.text(4.66, 2.0, f"vector\n{r['inputs']} × 1", ha="center", fontsize=9.5)
    ax.set_xlim(-0.4, 5.6); ax.set_ylim(-1.5, 3.1); ax.axis("off")
    ax.set_title("(a) a feature map is flattened first", fontsize=10.5)

    ax = axes[1]
    ns = np.arange(1, 65)
    for ch, col in zip((1, 4, 16), SERIES):
        ax.plot(ns, [fc_params(n, m, ch, int(v))["total"] for v in ns], color=col, lw=2,
                label=f"C = {ch} channels")
    ax.scatter([neurons], [r["total"]], color=C["red"], zorder=5, s=45)
    top = fc_params(n, m, 16, 64)["total"]
    ax.annotate(f"{n}×{m}×{channels} → {neurons} neurons\n"
                f"{r['weights']:,} weights + {r['biases']} biases = {r['total']:,}",
                xy=(neurons, r["total"]), xytext=(16, top * 0.74), fontsize=9,
                ha="left", arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.2))
    ax.set_xlabel("neurons N in the fully connected layer")
    ax.set_ylabel("parameters")
    ax.set_title("(b) parameters grow as n·m·C·N + N", fontsize=10.5)
    ax.legend(fontsize=9)
    png = save_mpl(fig, ctx, "fc_param_count")

    widget = """
<div class="widget conv-calc">
  <b>Fully connected parameter count</b> &nbsp;
  <span class="muted">params = (n · m · C) · N + N</span>
  <div class="widget-row">
    <label>n <input type="number" id="fc-n" value="7" min="1"></label>
    <label>m <input type="number" id="fc-m" value="7" min="1"></label>
    <label>channels C <input type="number" id="fc-c" value="4" min="1"></label>
    <label>neurons N <input type="number" id="fc-N" value="10" min="1"></label>
  </div>
  <div id="fc-out" class="widget-out"></div>
</div>
<script>
(function(){
  const ids=["fc-n","fc-m","fc-c","fc-N"], out=document.getElementById("fc-out");
  const fmt=v=>v.toLocaleString("en-GB");
  function upd(){
    const [n,m,c,N]=ids.map(x=>parseInt(document.getElementById(x).value||"0"));
    if([n,m,c,N].some(v=>!v||v<1)){out.textContent="all values must be at least 1";return;}
    const inp=n*m*c, w=inp*N, t=w+N;
    out.innerHTML=`flattened input = ${n}·${m}·${c} = <b>${fmt(inp)}</b><br>`
      + `weights = ${fmt(inp)} · ${N} = ${fmt(w)} &nbsp; biases = ${N}<br>`
      + `total = <b>${fmt(t)}</b> parameters`;
  }
  ids.forEach(x=>document.getElementById(x).addEventListener("input",upd)); upd();
})();
</script>
"""
    return Figure("fc_param_count",
                  "Fully connected layers are usually where a network's parameters "
                  "are. A feature map of size n × m × C must first be flattened into "
                  "a vector of n·m·C values; connecting that to N neurons needs "
                  "n·m·C·N weights plus one bias per neuron. (a) The flattening step. "
                  "(b) The count grows linearly in the number of neurons and in the "
                  f"number of channels — the marked point is {n} × {m} × {channels} "
                  f"into {neurons} neurons, which is already {r['total']:,} parameters. "
                  "Use the calculator to see how quickly this gets out of hand for a "
                  "realistic image size.", png=png, html=widget, kind="widget")


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_single_neuron(ctx), fig_activation_functions(ctx), fig_fc_network(ctx),
            fig_nn_classifiers(ctx), fig_softmax_activation(ctx), fig_fc_param_count(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("1.2 Fully connected layers and classifiers")
    tab1, tab2, tab3 = st.tabs(["Neuron", "Softmax", "Parameter count"])

    with tab1:
        st.markdown("Set the inputs, weights and bias and watch the output change.")
        c = st.columns(3)
        x = np.array([c[i].number_input(f"x{i+1}", value=float(NEURON_X[i]), step=0.1)
                      for i in range(3)])
        c = st.columns(3)
        w = np.array([c[i].number_input(f"w{i+1}", value=float(NEURON_W[i]), step=0.1)
                      for i in range(3)])
        b = st.number_input("bias b", value=float(NEURON_B), step=0.1)
        z = float(x @ w + b)
        name = st.selectbox("activation", list(ACTIVATIONS))
        f, df = ACTIVATIONS[name]
        st.latex(rf"z = \sum_i x_i w_i + b = {z:.3f} \qquad y = \sigma(z) = {float(f(z)):.3f}")
        zz = np.linspace(-6, 6, 400)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.plot(zz, f(zz), color=C["blue"], lw=2)
        ax.scatter([z], [float(f(z))], color=C["red"], zorder=5, s=60)
        ax.axvline(z, color=C["red"], ls="--", lw=1)
        ax.set_xlabel("z"); ax.set_title(name)
        st.pyplot(fig)
        if abs(z) > 4 and name in ("sigmoid", "tanh"):
            st.warning(f"σ′(z) = {float(df(z)):.4f} — this unit is saturated, so very "
                       "little gradient flows back through it.")

    with tab2:
        txt = st.text_input("logits z (comma separated)", "4, 3, 3, 2")
        try:
            z = np.array([float(v) for v in txt.split(",")])
        except ValueError:
            st.error("Use comma-separated numbers."); return
        p = softmax(z)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        axes[0].bar(range(len(z)), z, color=C["grey"]); axes[0].set_title("logits z")
        axes[1].bar(range(len(p)), p, color=C["orange"]); axes[1].set_title("softmax p")
        st.pyplot(fig)
        st.write({f"class {i+1}": round(float(v), 4) for i, v in enumerate(p)})
        st.success(f"Σ p = {p.sum():.4f} · predicted class {int(p.argmax()) + 1}")

    with tab3:
        c = st.columns(4)
        n = c[0].number_input("n", 1, 1024, 7)
        m = c[1].number_input("m", 1, 1024, 7)
        ch = c[2].number_input("channels C", 1, 1024, 4)
        N = c[3].number_input("neurons N", 1, 4096, 10)
        r = fc_params(int(n), int(m), int(ch), int(N))
        st.latex(rf"({n} \cdot {m} \cdot {ch}) \cdot {N} + {N} = {r['total']:,}")
        st.write(f"flattened input {r['inputs']:,} · weights {r['weights']:,} "
                 f"· biases {r['biases']:,}")
        st.caption("A 256 × 256 RGB image into 100 neurons needs about 19.7 million "
                   "parameters, which is why images are not fed to fully connected "
                   "layers directly.")
