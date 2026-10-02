"""Lesson 2.4 — Transfer learning.

Figures
    transfer_pipeline     feature extractor + classifier; reuse the extractor with
                          a new head (MobileNetV2's numbers, computed)
    transfer_detection    a pre-trained extractor feeding class and box heads
    pretrain_convergence  pre-trained vs from scratch, with all the target data
                          (measured): faster, but not better in the end
    data_size_sweep       final accuracy against target-set size, for a closely
                          related task and for new classes (measured)
    decision_grid         when to use transfer learning, frozen or fine-tuned
    pretrained_models     ready-made backbones: size with and without the top

The experiments are real: a small network (engines/_tinynet.py) is pre-trained
on one part of the digits data and transferred to another, averaged over
several random subsets. They are small and quick, but they reproduce both the
benefit of pre-training and the limits that He et al. and Raghu et al. found.

Source: reference/legacy_md/lesson_2_4_transfer_learning + lesson_2_4_script.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .._common import BuildContext, Figure, C, SERIES, save_mpl, plotly_html, load_image, use_source
from .. import _arch as A
from .. import _tinynet as T

FIGURES = ["transfer_pipeline", "transfer_detection", "pretrain_convergence",
           "data_size_sweep", "decision_grid", "pretrained_models"]

MODES = [("scratch", "trained from scratch", C["red"]),
         ("finetune", "pre-trained, fine-tuned", C["green"]),
         ("frozen", "pre-trained, frozen extractor", C["blue"])]
SIZES = [6, 10, 20, 50, 150, None]          # None = the whole target pool
SEEDS = range(6)
EPOCHS = 60


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _setup(task):
    """Source and target data for one transfer task.

    'related': source = digit identity (10 classes); target = odd/even parity of
               *different* images. The target needs exactly the features the
               source learned — the favourable case.
    'new':     source = digits 0-4; target = digits 5-9. Same kind of image,
               new classes — features only partly reusable.
    """
    if task in _CACHE:
        return _CACHE[task]
    Xtr, ytr, Xte, yte = T.digits()
    rng = np.random.default_rng(7)
    if task == "related":
        perm = rng.permutation(len(ytr))
        S, P = perm[: len(ytr) // 2], perm[len(ytr) // 2:]
        Xs, ys, ks = Xtr[S], ytr[S], 10
        Xt, yt, Xtt, ytt, kt = Xtr[P], ytr[P] % 2, Xte, yte % 2, 2
    else:
        s, st = ytr < 5, yte < 5
        Xs, ys, ks = Xtr[s], ytr[s], 5
        Xt, yt, Xtt, ytt, kt = Xtr[~s], ytr[~s] - 5, Xte[~st], yte[~st] - 5, 5
    pre = T.MLP(64, 64, 2, ks, seed=1)
    T.train(pre, Xs, ys, epochs=40, lr=0.05)
    _CACHE[task] = (pre, Xt, yt, Xtt, ytt, kt)
    return _CACHE[task]


def run(task, mode, n, seed, epochs=EPOCHS, lr=0.03):
    pre, Xt, yt, Xtt, ytt, kt = _setup(task)
    idx = np.random.default_rng(seed).permutation(len(yt))[: (n or len(yt))]
    net = T.MLP(64, 64, 2, kt, seed=seed + 100)
    freeze = ()
    if mode != "scratch":
        for i in range(len(net.W) - 1):                 # copy the feature extractor
            net.W[i], net.b[i] = pre.W[i].copy(), pre.b[i].copy()
        if mode == "frozen":
            freeze = tuple(range(len(net.W) - 1))       # train the new head only
    return T.train(net, Xt[idx], yt[idx], Xtt, ytt, epochs=epochs, lr=lr,
                   batch=min(32, len(idx)), freeze=freeze, seed=seed)


def sweep(task):
    key = ("sweep", task)
    if key not in _CACHE:
        out = {}
        for mode, _, _ in MODES:
            out[mode] = np.array([[run(task, mode, n, s)["test_acc"][-1] for s in SEEDS]
                                  for n in SIZES])
        _CACHE[key] = out
    return _CACHE[key]


def pool_size(task):
    return len(_setup(task)[2])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, txt, fc, fs=9, alpha=0.55, **kw):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, alpha=alpha, ec=C["ink"], lw=1.1, **kw))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)


def fig_transfer_pipeline(ctx: BuildContext) -> Figure:
    net = A.mobilenet_v2()
    head = sum(l.params for l in net.layers if l.stage == "classifier")
    backbone = net.params - head
    new_head = 1280 * 5 + 5
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.2))
    ax = axes[0]
    _box(ax, 0.0, 0.3, 1.6, 0.9, "ImageNet\n1.28 M images", C["light"])
    _box(ax, 2.0, 0.3, 4.2, 0.9, f"feature extractor (convolution layers)\n"
                                 f"{backbone:,} parameters", C["blue"])
    _box(ax, 6.6, 0.3, 2.6, 0.9, f"classifier: 1000 classes\n{head:,} parameters",
         C["purple"])
    for x0, x1 in [(1.6, 2.0), (6.2, 6.6)]:
        ax.annotate("", xy=(x1 - 0.02, 0.75), xytext=(x0 + 0.02, 0.75),
                    arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.4))
    ax.set_xlim(-0.2, 9.4); ax.set_ylim(0, 1.6); ax.axis("off")
    ax.set_title("(a) source: a network pre-trained on a large dataset", fontsize=10.5,
                 loc="left")

    ax = axes[1]
    _box(ax, 0.0, 0.3, 1.6, 0.9, "your images\n(e.g. 5 classes)", C["light"])
    _box(ax, 2.0, 0.3, 4.2, 0.9, "the same feature extractor, copied:\nfrozen, or "
                                 "fine-tuned with a small learning rate", C["blue"])
    _box(ax, 6.6, 0.3, 2.6, 0.9, f"new classifier: 5 classes\n{new_head:,} parameters",
         C["green"])
    for x0, x1 in [(1.6, 2.0), (6.2, 6.6)]:
        ax.annotate("", xy=(x1 - 0.02, 0.75), xytext=(x0 + 0.02, 0.75),
                    arrowprops=dict(arrowstyle="->", color=C["grey"], lw=1.4))
    ax.annotate("", xy=(4.1, 1.22), xytext=(4.1, 1.6),
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.8))
    ax.text(4.25, 1.42, "copy the learned weights", fontsize=9, color=C["red"])
    ax.text(7.9, 0.12, "replaced, trained from random", ha="center", fontsize=8.5,
            color=C["grey"], va="top")
    ax.set_xlim(-0.2, 9.4); ax.set_ylim(-0.25, 1.6); ax.axis("off")
    ax.set_title(f"(b) target: with the extractor frozen, only {new_head:,} of "
                 f"{backbone + new_head:,} parameters are trained", fontsize=10.5, loc="left",
                 pad=16)
    png = save_mpl(fig, ctx, "transfer_pipeline")
    return Figure("transfer_pipeline",
                  "Transfer learning relies on a network splitting into a **feature "
                  "extractor** — the convolution layers — and a **classifier**. "
                  "(a) Trained on a large dataset such as ImageNet, the extractor has "
                  "learned many generic image features: lines, edges and blobs in the "
                  "early layers, more specific and complex features deeper in. (b) For "
                  "a new task, the extractor is copied and only a new classifier is "
                  "attached. In the simplest case the extractor is used as a fixed "
                  "feature extractor and only the classifier is trained; usually the "
                  "extractor is also **fine-tuned**, training both together. The "
                  "numbers are MobileNetV2's, computed from its architecture: freezing "
                  f"the extractor leaves {new_head:,} parameters to train for a "
                  "5-class problem, which is why so little data and computation is "
                  "needed.", png=png)


def fig_transfer_detection(ctx: BuildContext) -> Figure:
    img = load_image("cat", max_side=300)
    rng = np.random.default_rng(1)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(13, 4.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1.3, 0.25, 1.4], wspace=0.15)
    ax = fig.add_subplot(gs[0])
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("input", fontsize=10)

    # a coarse 'activation' map: a blob over the cat's location, for illustration
    ax = fig.add_subplot(gs[1])
    gy, gx = np.mgrid[0:7, 0:7]
    act = np.exp(-((gx - 3.2) ** 2 + (gy - 3.4) ** 2) / 4.0) + 0.12 * rng.random((7, 7))
    ax.imshow(act, cmap="magma"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("pre-trained extractor: final 7 × 7 feature map\n(illustrative) — "
                 "where it fires encodes where the cat is", fontsize=9.5)

    ax = fig.add_subplot(gs[3])
    _box(ax, 0.0, 1.5, 2.6, 0.8, "class head\n→ 'cat', p = 0.97", C["purple"])
    _box(ax, 0.0, 0.2, 2.6, 0.8, "box head\n→ (x, y, w, h)", C["green"])
    ax.set_xlim(-0.4, 2.9); ax.set_ylim(0, 2.6); ax.axis("off")
    ax.set_title("two new heads, trained on the\ntarget task", fontsize=10)
    png = save_mpl(fig, ctx, "transfer_detection")
    return Figure("transfer_detection",
                  "Transfer learning can change the task as well as the data. Here a "
                  "feature extractor pre-trained for classification is reused for "
                  "object detection. When an image passes through it, certain feature "
                  "maps in the last layers activate — not uniformly, but in a pattern "
                  "that carries spatial information about where the object is. That "
                  "is enough to train two heads: a class predictor, and a bounding-box "
                  "predictor that outputs the object's coordinates. (The feature map "
                  "shown is illustrative, not a network's output.) The extractor is "
                  "usually fine-tuned and the whole network trained end to end. "
                  "Detection data must say where every object is, so it is expensive "
                  "to label and scarce — which usually makes transfer learning "
                  "necessary to train a detector at all. Section 3 builds on this.",
                  png=png)


CONV_LR = 0.005   # same for every method; low enough that convergence is visible


def fig_pretrain_convergence(ctx: BuildContext) -> Figure:
    use_source("sklearn_digits")
    task = "related"
    hist = {m: [run(task, m, None, s, lr=CONV_LR) for s in SEEDS] for m, _, _ in MODES}
    mean = {m: np.array([h["test_acc"] for h in hist[m]]).mean(0) for m in hist}
    ep = hist["scratch"][0]["epoch"]
    meet = next((e for e, a, b in zip(ep, mean["scratch"], mean["finetune"]) if a >= b), None)
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    for mode, label, col in MODES:
        acc = np.array([h["test_acc"] for h in hist[mode]])
        ax.plot(ep, acc.mean(0), color=col, lw=2.2,
                label=f"{label}: final {acc[:, -1].mean():.1%}")
        ax.fill_between(ep, acc.min(0), acc.max(0), color=col, alpha=0.12)
    if meet:
        ax.axvline(meet, color=C["grey"], ls=":", lw=1.2)
        ax.text(meet * 1.05, 0.78, f"scratch catches\nup at epoch {meet}", fontsize=8.5,
                color=C["grey"])
    ax.set_xscale("log")
    ax.set_xlabel("epoch (log scale)"); ax.set_ylabel("test accuracy")
    ax.set_ylim(0.55, 1.0)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(f"all {pool_size(task)} target images · after 1 epoch: from scratch "
                 f"{mean['scratch'][0]:.0%}, fine-tuned {mean['finetune'][0]:.0%}",
                 fontsize=10.5)
    png = save_mpl(fig, ctx, "pretrain_convergence")
    return Figure("pretrain_convergence",
                  "Pre-training's main benefit is speed, measured here. A small network "
                  "is pre-trained to recognise handwritten digits, then trained to tell "
                  "odd from even digits on a separate set of images, using all the "
                  "available target data — either from random weights or starting "
                  "from the pre-trained feature extractor. All runs use the same "
                  f"learning rate ({CONV_LR}); there are {len(SEEDS)} of each, the line "
                  "is the mean and the band the range. After one epoch the pre-trained "
                  f"network is {100 * (mean['finetune'][0] - mean['scratch'][0]):.0f} "
                  "points ahead. But the network trained from scratch catches up"
                  + (f" by epoch {meet}" if meet else "") + ", and their final accuracy "
                  "is the same. This is exactly the finding of He and colleagues: with "
                  "enough target data, pre-training converges faster but does not "
                  "reach a better final accuracy. The frozen extractor ends lowest, "
                  "because freezing stops the features being adapted to the new task.",
                  png=png)


def fig_data_size_sweep(ctx: BuildContext) -> Figure:
    use_source("sklearn_digits")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    panels = [("related", "(a) closely related task: digit identity → odd or even"),
              ("new", "(b) new classes: digits 0–4 → digits 5–9")]
    for ax, (task, title) in zip(axes, panels):
        res = sweep(task)
        ns = [n or pool_size(task) for n in SIZES]
        for mode, label, col in MODES:
            m, s = res[mode].mean(1), res[mode].std(1)
            ax.errorbar(ns, m, yerr=s, color=col, lw=2, marker="o", ms=5, capsize=3,
                        label=label)
        ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("labelled target examples (log scale)")
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("final test accuracy")
    axes[0].legend(fontsize=8.8, loc="lower right")
    rel, new = sweep("related"), sweep("new")
    png = save_mpl(fig, ctx, "data_size_sweep")
    return Figure("data_size_sweep",
                  "Whether transfer learning helps depends on the target data and on "
                  f"how related the tasks are, measured over {len(SEEDS)} random "
                  "subsets at each size. (a) **A closely related task.** The source "
                  "network learned digit identity; the target is whether a digit is "
                  "odd or even, on images the source never saw. With only 6 labelled "
                  f"examples the pre-trained network reaches {rel['finetune'][0].mean():.0%} "
                  f"against {rel['scratch'][0].mean():.0%} from scratch, and even a "
                  "**frozen** extractor does as well as fine-tuning — the most "
                  "favourable case on the slides. As data grows, scratch catches up "
                  "and with all of it slightly overtakes. (b) **New classes.** The "
                  "source learned digits 0–4; the target is 5–9. Now pre-training gives "
                  "no advantage at any size — with the fewest examples, fine-tuning is "
                  "even slightly worse than starting from scratch — and the frozen "
                  "extractor is clearly worst: the features are only partly the right "
                  "ones. "
                  "This is the point of Raghu et al. — transfer helps less than is "
                  "often assumed when the domains or tasks differ — and of the slide's "
                  "advice that less similar domains need fine-tuning.", png=png)


def fig_decision_grid(ctx: BuildContext) -> Figure:
    cells = {
        (0, 0): ("similar domain · small target set",
                 "transfer learning: clearly better than scratch\n"
                 "frozen extractor may be enough", C["green"], "panel (a), left"),
        (0, 1): ("similar domain · large target set",
                 "same final accuracy either way\n"
                 "transfer saves training time", C["sky"], "convergence figure"),
        (1, 0): ("dissimilar domain · small target set",
                 "fine-tune: frozen features are wrong\n"
                 "transfer slightly better; risk of overfitting", C["orange"],
                 "panel (b), left"),
        (1, 1): ("dissimilar domain · large target set",
                 "training from scratch does as well\n"
                 "consider a smaller, lighter model", C["light"], "panel (b), right"),
    }
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for (r, c), (head, body, col, ev) in cells.items():
        x, y = 1.0 + c * 4.6, 2.5 - r * 2.3
        ax.add_patch(plt.Rectangle((x, y), 4.4, 2.1, fc=col, alpha=0.4, ec=C["ink"], lw=1.1))
        ax.text(x + 2.2, y + 1.7, head, ha="center", fontsize=9.5, weight="bold")
        ax.text(x + 2.2, y + 1.0, body, ha="center", va="center", fontsize=9.2)
        ax.text(x + 2.2, y + 0.22, f"measured: {ev}", ha="center", fontsize=8,
                color=C["grey"], style="italic")
    ax.text(3.2, 4.85, "small target dataset", ha="center", fontsize=10)
    ax.text(7.8, 4.85, "large target dataset", ha="center", fontsize=10)
    ax.text(0.85, 3.55, "similar\nsource and\ntarget", ha="right", va="center", fontsize=10)
    ax.text(0.85, 1.25, "dissimilar\n(e.g. medical\nimages)", ha="right", va="center",
            fontsize=10)
    ax.set_xlim(-0.6, 10.3); ax.set_ylim(0.0, 5.15); ax.axis("off")
    png = save_mpl(fig, ctx, "decision_grid")
    return Figure("decision_grid",
                  "When to use transfer learning, summarising the lesson's "
                  "recommendations against the similarity of source and target and "
                  "the size of the target dataset. The most favourable case is a "
                  "similar domain with little target data: transfer learning will be "
                  "much better than training from scratch, and if the classes are "
                  "similar it can work well without fine-tuning at all. With a large "
                  "target dataset the final accuracy is the same either way, but "
                  "transfer learning saves training time. With a dissimilar domain, "
                  "such as medical images, fine-tuning is necessary, a small target "
                  "set risks overfitting, and with a medium-sized dataset training "
                  "from scratch performs as well — so it is worth asking whether a "
                  "much lighter model, trained from scratch, would do. Limited "
                  "computation is a separate reason to prefer transfer learning in "
                  "any cell. The italic notes point to where this lesson's experiments "
                  "show each behaviour.", png=png)


def fig_pretrained_models(ctx: BuildContext) -> Figure:
    rows = []
    for name, fn in [("MobileNetV2", A.mobilenet_v2), ("DenseNet-121", A.densenet121),
                     ("ResNet-50", lambda: A.resnet(50)), ("VGG-16", A.vgg16)]:
        net = fn()
        top = sum(l.params for l in net.layers if l.kind == "fc")
        rows.append((name, net.params, net.params - top))
    fig, ax = plt.subplots(figsize=(11, 3.9))
    y = np.arange(len(rows))
    full = np.array([r[1] for r in rows]) / 1e6
    notop = np.array([r[2] for r in rows]) / 1e6
    ax.barh(y + 0.19, full, 0.36, color=C["purple"], label="whole network")
    ax.barh(y - 0.19, notop, 0.36, color=C["blue"], label="feature extractor only "
                                                           "(classifier removed)")
    for i, (n, f, nt) in enumerate(rows):
        ax.text(f / 1e6 + 1.5, i + 0.19, f"{f:,}", va="center", fontsize=8.5)
        ax.text(nt / 1e6 + 1.5, i - 0.19, f"{nt:,}", va="center", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("parameters (millions)"); ax.set_xlim(0, 165)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("ready-made ImageNet backbones, with and without their classifier — "
                 "trainable parameters, computed from each architecture", fontsize=10)
    png = save_mpl(fig, ctx, "pretrained_models")
    vgg = rows[-1]
    return Figure("pretrained_models",
                  "Frameworks provide standard image recognition models with ImageNet "
                  "weights ready to use: Keras has an applications module containing "
                  "ResNet50, VGG16, DenseNet and MobileNetV2 among others. Setting "
                  "**include_top=False** removes the fully connected classification "
                  "part, leaving only the feature extractor, to which a new classifier "
                  "is attached; the extractor's layers can then be frozen or fine-tuned "
                  "together with it. The input_shape argument fixes the size of the "
                  "model's tensors. Note how much of each model the top is: most of "
                  "VGG-16's parameters are in its classifier, so without the top it "
                  f"shrinks from {vgg[1]:,} to {vgg[2]:,}, whereas the backbones that "
                  "end in global average pooling barely change. The counts are "
                  "trainable parameters, computed from each architecture. If you check "
                  "them with Keras's model.summary(), VGG-16 matches exactly but the "
                  "others come out somewhat larger: Keras also counts batch norm's "
                  "running mean and variance (Lesson 2.3) as non-trainable parameters, "
                  "and its ResNet50 gives each convolution a bias.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    A.verify()
    return [fig_transfer_pipeline(ctx), fig_transfer_detection(ctx),
            fig_pretrain_convergence(ctx), fig_data_size_sweep(ctx),
            fig_decision_grid(ctx), fig_pretrained_models(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("2.4 Transfer learning")
    c = st.columns(3)
    task = c[0].radio("target task", ["related", "new"],
                      format_func=lambda t: {"related": "odd / even (related)",
                                             "new": "digits 5–9 (new classes)"}[t])
    n = c[1].select_slider("labelled target examples", [6, 10, 20, 50, 150, 300, 600], 20)
    seed = c[2].number_input("random subset", 0, 99, 0)
    if st.button("train all three"):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for mode, label, col in MODES:
            h = run(task, mode, n, int(seed))
            ax.plot(h["epoch"], h["test_acc"], color=col, lw=2,
                    label=f"{label}: {h['test_acc'][-1]:.1%}")
        ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1)
        ax.legend(fontsize=8.5)
        st.pyplot(fig)
        st.caption("Try several subsets: with few examples, results vary a lot from "
                   "one random subset to the next.")
