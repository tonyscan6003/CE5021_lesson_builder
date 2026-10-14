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
                  "Transfer learning is based on the division of a network into a **feature "
                  "extractor**, the convolution layers, and a **classifier**. (a) When "
                  "trained on a large dataset such as ImageNet, many generic image features "
                  "are learned by the extractor: lines, edges and blobs in the early layers, "
                  "and more specific and complex features in the deeper layers. (b) For a "
                  "new task the extractor is copied and a new classifier is attached. In the "
                  "simplest case the extractor is used as a fixed feature extractor and only "
                  "the classifier is trained; usually the extractor is also **fine-tuned**, "
                  "with both trained together. The values are those of MobileNetV2, "
                  "calculated from its architecture: when the extractor is frozen, "
                  f"{new_head:,} parameters remain to be trained for a 5-class problem, "
                  "which is why so little data and computation are required.",
                  png=png)


def fig_transfer_detection(ctx: BuildContext) -> Figure:
    img = load_image("cat", max_side=300)
    rng = np.random.default_rng(1)
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(13, 4.3))
    YM = 2.35

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.5))

    # input image
    iw = 3.3
    ih = iw * h / w
    ax.imshow(img, extent=(0, iw, YM - ih / 2, YM + ih / 2))
    ax.text(iw / 2, YM + ih / 2 + 0.12, "input", ha="center", va="bottom", fontsize=10)
    arrow(iw + 0.1, YM, iw + 0.75, YM)

    # the pre-trained feature extractor: convolution layers, spatial size shrinking
    x0, x1 = iw + 0.8, iw + 3.9
    ax.add_patch(plt.Polygon([(x0, YM - 1.35), (x1, YM - 0.7), (x1, YM + 0.7),
                              (x0, YM + 1.35)], fc=C["sky"], alpha=0.6, ec=C["ink"], lw=1.2))
    ax.text((x0 + x1) / 2, YM, "pre-trained\nfeature extractor\n(convolution layers,\n"
            "usually fine-tuned)", ha="center", va="center", fontsize=9.5)
    arrow(x1 + 0.1, YM, x1 + 0.75, YM)

    # a coarse 'activation' map: a blob over the cat's location, for illustration
    fx, fs = x1 + 0.8, 2.5
    gy, gx = np.mgrid[0:7, 0:7]
    act = np.exp(-((gx - 3.2) ** 2 + (gy - 3.4) ** 2) / 4.0) + 0.12 * rng.random((7, 7))
    ax.imshow(act, cmap="magma", extent=(fx, fx + fs, YM - fs / 2, YM + fs / 2))
    ax.text(fx + fs / 2, YM + fs / 2 + 0.12, "final 7 × 7 feature map (illustrative):\n"
            "where it fires encodes where the cat is", ha="center", va="bottom",
            fontsize=9.5)

    # two new heads, both fed by the same feature map
    hx, hw, hh = fx + fs + 1.3, 3.0, 0.9
    _box(ax, hx, YM + 0.45, hw, hh, "class head\n→ 'cat', p = 0.97", C["purple"], fs=9.5)
    _box(ax, hx, YM - 0.45 - hh, hw, hh, "box head\n→ (x, y, w, h)", C["green"], fs=9.5)
    arrow(fx + fs + 0.1, YM + 0.1, hx - 0.05, YM + 0.45 + hh / 2)
    arrow(fx + fs + 0.1, YM - 0.1, hx - 0.05, YM - 0.45 - hh / 2)
    ax.text(hx + hw / 2, YM + 0.45 + hh + 0.12, "two new heads, trained on the target task",
            ha="center", va="bottom", fontsize=9.5)
    ax.set_xlim(-0.1, hx + hw + 0.1); ax.set_ylim(YM - 1.75, YM + 2.0)
    ax.set_aspect("equal"); ax.axis("off")
    png = save_mpl(fig, ctx, "transfer_detection")
    return Figure("transfer_detection",
                  "In transfer learning the task can be changed as well as the data. Here a "
                  "feature extractor pre-trained for classification is reused for object "
                  "detection. When the input image is passed through the extractor, certain feature "
                  "maps in the last layers are activated, not uniformly but in a pattern "
                  "that contains spatial information about the location of the object. This "
                  "is sufficient for two heads to be trained: a class predictor, and a "
                  "bounding-box predictor that outputs the coordinates of the object. (The "
                  "feature map shown is illustrative, not the output of a network.) The "
                  "extractor is usually fine-tuned and the whole network trained end to end. "
                  "The location of every object must be recorded in detection data, so the "
                  "data is expensive to label and scarce, and transfer learning is therefore "
                  "usually necessary for a detector to be trained. This is developed further "
                  "in Section 3.",
                  png=png)


CONV_LR = 0.005   # same for every method; low enough that convergence is visible


def fig_pretrain_convergence(ctx: BuildContext) -> Figure:
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
                  "The main benefit of pre-training is speed, as measured here. A small "
                  "network is pre-trained to recognise handwritten digits, and is then "
                  "trained to distinguish odd from even digits on a separate set of images, "
                  "using all of the available target data, either from random weights or "
                  "from the pre-trained feature extractor. The same learning rate "
                  f"({CONV_LR}) is used for all runs; there are {len(SEEDS)} runs of each, "
                  "with the line showing the mean and the band the range. After one epoch "
                  "the pre-trained network is "
                  f"{100 * (mean['finetune'][0] - mean['scratch'][0]):.0f} points ahead. "
                  "However, the network trained from scratch catches up"
                  + (f" by epoch {meet}" if meet else "") + ", and the final accuracy is "
                  "the same. This agrees with the finding of He and colleagues: with "
                  "sufficient target data, pre-training results in faster convergence but "
                  "not in a better final accuracy. The lowest final accuracy is obtained "
                  "with the frozen extractor, as the features cannot be adapted to the new "
                  "task.",
                  png=png)


def fig_data_size_sweep(ctx: BuildContext) -> Figure:
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
                  "The benefit of transfer learning depends on the amount of target data and "
                  f"on how closely the tasks are related, measured over {len(SEEDS)} random "
                  "subsets at each size. (a) **A closely related task.** Digit identity was "
                  "learned by the source network; the target task is to decide whether a "
                  "digit is odd or even, on images not seen by the source network. With only "
                  "6 labelled examples, the pre-trained network reaches "
                  f"{rel['finetune'][0].mean():.0%} compared with "
                  f"{rel['scratch'][0].mean():.0%} from scratch, and a **frozen** extractor "
                  "performs as well as fine-tuning, which is the most favourable case "
                  "described in the lecture slides. As the amount of data increases, the "
                  "network trained from scratch catches up, and with all of the data it "
                  "performs slightly better. (b) **New classes.** Digits 0–4 were learned by "
                  "the source network, and the target is digits 5–9. In this case "
                  "pre-training gives no advantage at any size; with the fewest examples, "
                  "fine-tuning is slightly worse than training from scratch, and the frozen "
                  "extractor performs worst, as the features are only partly suitable. This "
                  "agrees with the finding of Raghu et al. that transfer learning is less "
                  "beneficial than is often assumed when the domains or tasks differ, and "
                  "with the recommendation that fine-tuning is required for less similar "
                  "domains.",
                  png=png)


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
                  "The recommendations of this lesson on the use of transfer learning, "
                  "arranged by the similarity of the source and target and by the size of "
                  "the target dataset. The most favourable case is a similar domain with "
                  "little target data: transfer learning performs much better than training "
                  "from scratch, and if the classes are similar, good results can be "
                  "obtained without fine-tuning. With a large target dataset the final "
                  "accuracy is the same in both cases, but training time is saved with "
                  "transfer learning. With a dissimilar domain, such as medical images, "
                  "fine-tuning is necessary, there is a risk of overfitting with a small "
                  "target dataset, and with a medium-sized dataset equivalent performance is "
                  "obtained by training from scratch; in this case a much lighter model "
                  "trained from scratch should be considered. Limited computational "
                  "resources are a separate reason to use transfer learning in any case. "
                  "The italic notes indicate where each behaviour is shown in the "
                  "experiments of this lesson.",
                  png=png)


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
                  "Standard image recognition models with ImageNet weights are provided by "
                  "the frameworks: the torchvision.models module contains ResNet-50, VGG-16, "
                  "DenseNet and MobileNetV2, among others. Each model consists of a feature "
                  "extractor and a classifier; when the final classification layer is "
                  "replaced, a new classifier is attached to the feature extractor, and the "
                  "layers of the extractor can then be frozen or fine-tuned together with "
                  "the new classifier. Most of the parameters of VGG-16 are in its "
                  f"classifier, so without the classifier it is reduced from {vgg[1]:,} to "
                  f"{vgg[2]:,} parameters, whereas the networks that end in global average "
                  "pooling are changed very little. The counts are trainable parameters, "
                  "calculated from each architecture, and are the same as those obtained "
                  "from the torchvision models by summing p.numel() over model.parameters().",
                  png=png)


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
