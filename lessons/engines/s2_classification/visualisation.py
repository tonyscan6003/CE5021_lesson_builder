"""Lesson 2.5 — Feature visualisation and adversarial attack.

Figures
    reconstruction          inversion: what survives of an image at increasing depth
    activation_maximisation the input that most excites one unit / one channel, five layers
    class_templates         activation maximisation on a trained classifier's outputs
    style_transfer          Gram-matrix texture synthesis and full style transfer
    style_transfer_loop     how content and style losses combine into style transfer
    deep_dream              gradient ascent on a photo, amplifying what a layer sees
    adversarial_fgsm        the fast gradient sign attack, and why it is not just noise
    patch_attack            changing only a small patch — the stop-sign stickers
    pose_generalisation     accuracy as test digits are rotated away from upright

Genuine trained models are used throughout:
  * engines/_torchvis.py — VGG-16 with its ImageNet weights (PyTorch), for
    reconstruction, activation maximisation, style transfer and deep dream.
    Results are cached in data/_torch/ and recomputed when the settings change.
  * a softmax classifier TRAINED on the sklearn digits. The adversarial and
    generalisation experiments attack and test this real trained model.

Source: reference/legacy_md/Lesson_2_5_visualisation + lesson_2_5_script.
"""
from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

from .._common import BuildContext, Figure, C, SERIES, save_mpl, plotly_html, load_image, use_source
from .. import _torchvis as TV

# channels of the five style layers conv1_1 ... conv5_1 of VGG-16
TV_STYLE_CHANNELS = (64, 128, 256, 512, 512)

FIGURES = ["reconstruction", "activation_maximisation", "class_templates",
           "style_transfer", "style_transfer_loop", "deep_dream",
           "adversarial_fgsm", "patch_attack", "pose_generalisation"]

def _square(img, n):
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)


# -- the trained classifier ------------------------------------------------------

_CLF: dict = {}


def digits_classifier():
    """Multinomial logistic regression trained on the digits, pixels in [0, 1]."""
    if not _CLF:
        from sklearn.datasets import load_digits
        from sklearn.linear_model import LogisticRegression
        d = load_digits()
        X, y = d.data / 16.0, d.target
        idx = np.random.default_rng(0).permutation(len(y))
        tr, te = idx[:1200], idx[1200:]
        clf = LogisticRegression(max_iter=5000, C=1.0).fit(X[tr], y[tr])
        _CLF.update(clf=clf, Xte=X[te], yte=y[te], W=clf.coef_, b=clf.intercept_)
    return _CLF


def _probs(X, W, b):
    z = X @ W.T + b
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def input_gradient(X, y, W, b):
    """d(cross-entropy)/dx for softmax regression: W^T (p - onehot)."""
    p = _probs(X, W, b)
    p[np.arange(len(y)), y] -= 1.0
    return p @ W


def fgsm(X, y, eps, W, b):
    return np.clip(X + eps * np.sign(input_gradient(X, y, W, b)), 0, 1)


def patch_attack(x, y, W, b, size=3, steps=40, alpha=0.25):
    """Change only a size x size patch (any values in [0, 1]), at the most effective
    position, to make the classifier wrong. Projected gradient ascent on the loss."""
    best = None
    for r in range(0, 9 - size):
        for c in range(0, 9 - size):
            m = np.zeros((8, 8)); m[r:r + size, c:c + size] = 1
            m = m.ravel()
            xa = x.copy()
            for _ in range(steps):
                g = input_gradient(xa[None], np.array([y]), W, b)[0]
                xa = np.clip(xa + alpha * np.sign(g) * m, 0, 1)
            p = _probs(xa[None], W, b)[0]
            score = p[y]
            if best is None or score < best[0]:
                best = (score, xa, m, p)
    return best


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

REC_LAYERS = ["conv1_2", "conv2_2", "conv3_3", "conv4_3", "conv5_3"]


def fig_reconstruction(ctx: BuildContext) -> Figure:
    img = _square(load_image("owl"), 224)
    # the smoothness penalty must grow with depth: deep features constrain the
    # pixels less, and without it the deep reconstructions drown in noise
    tv = {"conv1_2": 1.0, "conv2_2": 2.0, "conv3_3": 4.0, "conv4_3": 10.0, "conv5_3": 10.0}
    cfg = dict(image="owl", size=224, layers=REC_LAYERS, iters=500, tv=tv)
    rec = TV.cached("reconstruction", cfg, lambda: {
        l: TV.reconstruct(img, l, iters=cfg["iters"], tv_weight=tv[l]) for l in REC_LAYERS})
    g = lambda a: a.astype(float).mean(2).ravel()
    corr = [np.corrcoef(g(img), g(rec[l]))[0, 1] for l in REC_LAYERS]
    fig, axes = plt.subplots(1, 6, figsize=(14, 2.9))
    axes[0].imshow(img); axes[0].set_title("target image", fontsize=10)
    for ax, l, cc in zip(axes[1:], REC_LAYERS, corr):
        ax.imshow(rec[l])
        n = 224 // 2 ** (int(l[4]) - 1)
        ax.set_title(f"from {l}: {n}×{n} maps\ncorrelation {cc:.2f}", fontsize=9.5)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "reconstruction")
    return Figure("reconstruction",
                  "**Reconstruction** (inversion) with VGG-16 trained on ImageNet. The "
                  "target image is passed through the network and its features at one layer "
                  "are stored. Starting from random noise, the *pixels* are then adjusted by "
                  "gradient descent until the same features are produced, with a smoothness "
                  "(total variation) penalty used as a regulariser so that the result looks "
                  "natural. From the early layers the reconstruction is almost exact, as many "
                  "low-level features are retained. From deeper layers the fine detail and "
                  "the exact colours are lost, and an increasingly abstract version of the "
                  "image is recovered, in which the parts of the owl that the network uses "
                  "for recognition, such as the eyes and the texture of the feathers, are "
                  "retained in approximately the right places. The correlation with the "
                  "target decreases with depth accordingly.",
                  png=png)


AM_LAYERS = {"conv1_2": 25, "conv2_2": 120, "conv3_3": 140, "conv4_3": 260, "conv5_3": 280}


def fig_activation_maximisation(ctx: BuildContext) -> Figure:
    size = 224
    cfg = dict(layers=AM_LAYERS, size=size, steps=384)
    res = TV.cached("activation_maximisation", cfg, lambda: {
        f"{l}_{mode}": TV.activation_max(l, ch, unit=(mode == "unit"), size=size,
                                         steps=cfg["steps"])
        for l, ch in AM_LAYERS.items() for mode in ("unit", "channel")})
    fig, axes = plt.subplots(2, len(AM_LAYERS), figsize=(14, 6.1))
    for j, (l, ch) in enumerate(AM_LAYERS.items()):
        stride = 2 ** (int(l[4]) - 1)
        rf = TV.RF[l]
        # the maximised unit is the centre of the feature map; its receptive
        # field is centred on the input pixels that unit's position covers
        c = (size // stride) // 2 * stride + (stride - 1) / 2
        ax = axes[0, j]
        ax.imshow(res[f"{l}_unit"])
        ax.add_patch(plt.Rectangle((c - rf / 2, c - rf / 2), rf, rf, fill=False,
                                   ec="white", lw=1.4, ls="--"))
        ax.set_title(f"{l}, channel {ch}\nreceptive field {rf} × {rf}", fontsize=9.5)
        axes[1, j].imshow(res[f"{l}_channel"])
    axes[0, 0].set_ylabel("one unit\n(centre of the map)", fontsize=9.5)
    axes[1, 0].set_ylabel("whole channel", fontsize=9.5)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "activation_maximisation")
    return Figure("activation_maximisation",
                  "**Activation maximisation** with VGG-16 trained on ImageNet: the input "
                  "image to which a neuron responds most strongly. Starting from a "
                  "near-blank image, the pixels are changed by **gradient ascent**, as the "
                  "quantity is to be maximised. As many unnatural noise patterns would also "
                  "excite the neuron, regularisation is required; here the image is "
                  "randomly shifted at every step and a smoothness penalty is applied, so "
                  "that only patterns to which the neuron responds robustly are retained. "
                  "Top row: one unit, the centre of one feature map, is maximised in each of "
                  "five layers, starting from a grey image. Only the pixels within the "
                  "receptive field of the unit (dashed, calculated with the formula of "
                  "Lesson 1.4) can affect it, so the pattern is confined to the receptive "
                  "field, apart from a faint halo caused by the random shifts, and "
                  "**increases in size with depth**; the rest of the image remains grey. Bottom row: the same channels maximised over the whole feature map, "
                  "with the image optimised in a smoothed frequency representation, which "
                  "shows the feature itself. Early layers respond to colours and oriented edges, the middle "
                  "layers to textures and repeated patterns, and the deepest layers to "
                  "complex structures resembling parts of objects.",
                  png=png)


def fig_class_templates(ctx: BuildContext) -> Figure:
    m = digits_classifier()
    W = m["W"]
    acc = (m["clf"].predict(m["Xte"]) == m["yte"]).mean()
    fig, axes = plt.subplots(2, 10, figsize=(13, 3.3))
    for k in range(10):
        lim = np.abs(W[k]).max()
        axes[0, k].imshow(W[k].reshape(8, 8), cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[0, k].set_title(f"'{k}' neuron", fontsize=9)
        xbest = (W[k] > 0).astype(float)           # the [0,1] image maximising w·x
        axes[1, k].imshow(xbest.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        for r in range(2):
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
    axes[0, 0].set_ylabel("weights", fontsize=9)
    axes[1, 0].set_ylabel("best input", fontsize=9)
    fig.suptitle(f"a classifier trained on handwritten digits ({acc:.1%} test accuracy): "
                 "what each output neuron responds to", fontsize=10.5)
    png = save_mpl(fig, ctx, "class_templates")
    return Figure("class_templates",
                  "Activation maximisation on a **trained** model, a classifier for "
                  "handwritten digits with one output neuron per class. For a single layer "
                  "such as this, the result can be obtained directly from the weights (top "
                  "row: red pixels increase the score and blue pixels decrease it), and the "
                  "input image that maximises each neuron consists of the pixels with "
                  "positive weight (bottom row). Some of these, such as '0' and '7', "
                  "resemble the digit. Others are only fragments, and the locations where "
                  "ink must be *absent* (the blue weights, such as the empty centre of the "
                  "'0') are as important to each neuron as those where it should be "
                  "present, as this is what distinguishes its class from the others. As "
                  "discussed in the lecture slide on visualisation for understanding, a "
                  "class may be recognised by a network from properties that co-occur with "
                  "it, such as the texture of clouds or the sky behind a building, rather "
                  "than from the object itself, which can cause confusion when the network "
                  "is deployed.",
                  png=png)


ST_CONTENT = "sacre_coeur"


def fig_style_transfer(ctx: BuildContext) -> Figure:
    content = load_image(ST_CONTENT, max_side=320)
    style = load_image("starry_night", max_side=320)
    # style weights β relative to a content weight α = 1
    BETAS = (1e6, 1e7)
    cfg = dict(content=ST_CONTENT, style="starry_night", side=320, iters=500,
               betas=BETAS, tv=1.0)
    res = TV.cached("style_transfer", cfg, lambda: {
        "texture": TV.style_transfer(None, style, size=content.shape[:2], iters=cfg["iters"],
                                     style_weight=BETAS[0], tv_weight=cfg["tv"]),
        **{f"transfer_{b:g}": TV.style_transfer(content, style, iters=cfg["iters"],
                                                style_weight=b, tv_weight=cfg["tv"])
           for b in BETAS}})
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.1))
    panels = [(content, "(a) content image"), (style, "(b) style image"),
              (res["texture"], "(c) style only:\ntexture synthesis"),
              (res[f"transfer_{BETAS[0]:g}"], f"(d) style transfer\nβ/α = $10^{{{np.log10(BETAS[0]):.0f}}}$"),
              (res[f"transfer_{BETAS[1]:g}"], f"(e) more weight on style\nβ/α = $10^{{{np.log10(BETAS[1]):.0f}}}$")]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im); ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "style_transfer")
    nch = sum(TV_STYLE_CHANNELS)
    return Figure("style_transfer",
                  "Style transfer with VGG-16 trained on ImageNet. The **Gram matrix** of a "
                  "layer is the inner product of every pair of its feature maps, summed over "
                  "all positions: entry (i, j) is large when features i and j tend to be "
                  "active together anywhere in the image. As the sum is taken over positions, "
                  "information about *where* features occur is discarded and only "
                  "information about *which features co-occur* is retained, which "
                  "corresponds to style, or texture. Gram matrices are taken from five "
                  f"layers, conv1_1 to conv5_1, {nch:,} rows and columns in total, so that "
                  "texture is captured at small scales in the early layers and at larger "
                  "scales in the deeper layers. (c) When a noise image is adjusted until "
                  "only its Gram matrices match those of the style image, the brush strokes, "
                  "swirls and colours of the painting are reproduced but its layout is not. "
                  "(d) When a content loss on a deep layer (conv4_2) of the content image is "
                  "added, the noise image converges to the building of the content image "
                  "painted in the style of the artwork. (e) The balance is set by the ratio "
                  "of the style weight β to the content weight α: when β/α is increased "
                  "tenfold, the style dominates and the building is only partly preserved. "
                  "All of the generated images are produced from random noise, as described "
                  "in the steps below.",
                  png=png)


def fig_style_transfer_loop(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(13, 5.0))

    def box(x, y, w, h, txt, col, fs=9):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=col, alpha=0.5, ec=C["ink"], lw=1.1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)

    def arr(x0, y0, x1, y1, col=C["grey"]):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4))

    box(0.0, 3.6, 1.8, 0.8, "content image", C["light"])
    box(0.0, 0.5, 1.8, 0.8, "style image", C["light"])
    box(0.0, 2.05, 1.8, 0.8, "generated image\n(starts as noise)", C["yellow"])
    for y in (4.0, 0.9, 2.45):
        arr(1.85, y, 2.55, y)
    box(2.6, 0.35, 2.6, 4.2, "pre-trained network\n(weights fixed)", C["blue"], fs=10)
    box(5.9, 3.55, 2.4, 0.9, "deep-layer features\n(conv4): content", C["sky"])
    box(5.9, 0.4, 2.4, 0.9, "Gram matrices of\nevery layer: style", C["orange"])
    box(5.9, 2.0, 2.4, 0.9, "features and Gram\nmatrices of generated", C["yellow"])
    for y0, y1 in [(4.0, 4.0), (0.85, 0.85), (2.45, 2.45)]:
        arr(5.25, y0, 5.85, y1)
    box(9.1, 2.95, 2.7, 0.85, "content loss\n‖F_gen − F_content‖²", C["green"])
    box(9.1, 1.1, 2.7, 0.85, "style loss\nΣ ‖G_gen − G_style‖²", C["green"])
    arr(8.35, 3.9, 9.05, 3.5); arr(8.35, 2.6, 9.05, 3.25)
    arr(8.35, 0.85, 9.05, 1.4); arr(8.35, 2.3, 9.05, 1.7)
    box(12.3, 1.75, 2.0, 1.4, "L = α·content\n  + β·style", C["red"], fs=10)
    arr(11.85, 3.35, 12.25, 2.8); arr(11.85, 1.5, 12.25, 2.1)
    # feedback path routed beneath the diagram, so it crosses nothing
    ax.plot([13.3, 13.3, -0.15, -0.15], [1.72, -0.25, -0.25, 2.45], color=C["red"], lw=1.8)
    ax.annotate("", xy=(-0.02, 2.45), xytext=(-0.16, 2.45),
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.8))
    ax.text(6.6, -0.55, "gradient descent updates the generated image's pixels — "
                        "never the network's weights", ha="center", va="top", fontsize=10,
            color=C["red"])
    ax.set_xlim(-0.5, 14.6); ax.set_ylim(-1.0, 4.8); ax.axis("off")
    png = save_mpl(fig, ctx, "style_transfer_loop")
    return Figure("style_transfer_loop",
                  "Style transfer. The content image is passed through a pre-trained network "
                  "and a deep feature map is stored (conv4 in the original work); the "
                  "content, that is, the objects and their arrangement, is retained in the "
                  "deep layers but fine detail is not. The style image is passed through the "
                  "network and the Gram matrices of all its convolution layers are stored, "
                  "capturing texture at small scales in the early layers and at larger "
                  "scales in the deep layers. A random-noise image is then passed through "
                  "the network, and its features and Gram matrices are compared with the "
                  "stored values in a content loss and a style loss. The pixels of the noise "
                  "image are adjusted by gradient descent to minimise both losses together, "
                  "and the image converges to one with the content of one image and the "
                  "style of the other. The balance is set by the weights α and β. As in "
                  "reconstruction, the image is optimised rather than the network.",
                  png=png)


DD_LAYERS = ["conv4_3", "conv5_3"]


def fig_deep_dream(ctx: BuildContext) -> Figure:
    src = load_image("pyramids", max_side=448)
    cfg = dict(image="pyramids", side=448, layers=DD_LAYERS, octaves=4, steps=24)
    res = TV.cached("deep_dream", cfg, lambda: {
        l: TV.deep_dream(src, l, octaves=cfg["octaves"], steps=cfg["steps"])
        for l in DD_LAYERS})
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    axes[0].imshow(src); axes[0].set_title("content image", fontsize=10)
    for ax, l in zip(axes[1:], DD_LAYERS):
        ax.imshow(res[l]); ax.set_title(f"deep dream on {l}", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "deep_dream")
    return Figure("deep_dream",
                  "**Deep dream** with VGG-16 trained on ImageNet: activation maximisation "
                  "started from a photograph rather than noise. The network responds to "
                  "the features already present in the photograph, and these responses are "
                  "amplified by gradient ascent, so that the image is modified towards "
                  "whatever the layer detects in it. The ascent is repeated at four scales "
                  "(octaves), from coarse to fine, so that features of every size are "
                  "amplified. With conv4_3, dog-like faces and eyes appear over the pyramids "
                  "and the sky is covered in repeated patterns; with the deepest layer, "
                  "conv5_3, larger insect-like and bird-like creatures emerge from the "
                  "outlines of the pyramids. Many ImageNet classes are breeds of dog, birds "
                  "and insects, and these results reveal the associations learned by the "
                  "network between image features and the objects of those classes.",
                  png=png)


def fig_adversarial_fgsm(ctx: BuildContext) -> Figure:
    use_source("sklearn_digits")
    m = digits_classifier()
    W, b, X, y = m["W"], m["b"], m["Xte"], m["yte"]
    p0 = _probs(X, W, b)
    ok = np.where(p0.argmax(1) == y)[0]
    # the confidently classified example that needs the smallest attack
    eps_grid = np.round(np.arange(0.01, 0.31, 0.01), 2)
    cand = ok[np.argsort(-p0[ok, y[ok]])[:60]]
    best = None
    for i in cand:
        for e in eps_grid:
            pa = _probs(fgsm(X[i:i + 1], y[i:i + 1], e, W, b), W, b)[0]
            if pa.argmax() != y[i] and pa.max() >= 0.8:     # wrong AND confident
                if best is None or e < best[1]:
                    best = (i, e, pa)
                break
    i, e, pa = best
    xa = fgsm(X[i:i + 1], y[i:i + 1], e, W, b)[0]

    rng = np.random.default_rng(0)
    eps_c = np.linspace(0, 0.3, 16)
    acc_fgsm = [(_probs(fgsm(X, y, ee, W, b), W, b).argmax(1) == y).mean() for ee in eps_c]
    acc_rand = [(_probs(np.clip(X + ee * rng.choice([-1, 1], X.shape), 0, 1), W, b)
                 .argmax(1) == y).mean() for ee in eps_c]

    fig = plt.figure(figsize=(13.5, 4.3))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.15, 2.2], wspace=0.25)
    for k, (img, title, cmap) in enumerate(
            [(X[i], f"original: '{y[i]}' ({p0[i, y[i]]:.0%})", "gray_r"),
             (np.sign(input_gradient(X[i:i + 1], y[i:i + 1], W, b)[0]),
              f"+ {e:.2f} × sign(gradient)", "RdBu_r"),
             (xa, f"adversarial: '{pa.argmax()}' ({pa.max():.0%})", "gray_r")]):
        ax = fig.add_subplot(gs[k])
        ax.imshow(img.reshape(8, 8), cmap=cmap, vmin=-1 if cmap == "RdBu_r" else 0, vmax=1)
        ax.set_title(title, fontsize=9.5, color=C["red"] if k == 2 else C["ink"])
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[4])
    ax.plot(eps_c, acc_fgsm, color=C["red"], lw=2.2, marker="o", ms=3.5,
            label="adversarial (FGSM) perturbation")
    ax.plot(eps_c, acc_rand, color=C["grey"], lw=2.2, marker="o", ms=3.5,
            label="random ±ε noise, same size")
    ax.set_xlabel("perturbation size ε (pixel range 0–1)")
    ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8.8, loc="lower left")
    k10 = int(np.argmin(np.abs(eps_c - 0.1)))
    ax.set_title(f"at ε = 0.1: {acc_rand[k10]:.0%} with random noise, "
                 f"{acc_fgsm[k10]:.0%} with the attack", fontsize=10)
    png = save_mpl(fig, ctx, "adversarial_fgsm")
    return Figure("adversarial_fgsm",
                  "An **adversarial attack** on a trained classifier. In the **fast gradient "
                  "sign method**, the gradient of the loss with respect to the *input "
                  "pixels* is calculated, which gives the direction in which each pixel "
                  "should move to increase the error of the network, and a small step ε in "
                  "the sign of this gradient is added: x' = x + ε·sign(∇ₓL). Left: a digit "
                  f"that is correctly classified with {p0[i, y[i]]:.0%} confidence becomes, "
                  f"after a change of at most {e:.2f} per pixel, a '{pa.argmax()}' with "
                  f"{pa.max():.0%} confidence, which is incorrect and more confident than "
                  "many correct predictions. Right: random noise of exactly the same size has "
                  "little effect on accuracy, whereas the same perturbation budget directed "
                  "by the gradient reduces the accuracy greatly. Adversarial examples are "
                  "therefore not random noise, but are carefully chosen. Each output depends "
                  "on many inputs through many multiplications, so many small, coordinated "
                  "changes accumulate to a large change at the output. Deep networks are "
                  "affected in the same way as this simple classifier, attacks designed "
                  "against one network often transfer to others, and the development of "
                  "robust networks is an active research area.",
                  png=png)


def fig_patch_attack(ctx: BuildContext) -> Figure:
    use_source("sklearn_digits")
    m = digits_classifier()
    W, b, X, y = m["W"], m["b"], m["Xte"], m["yte"]
    pred = _probs(X, W, b).argmax(1)
    ok = np.where(pred == y)[0][:60]
    results = [(i, patch_attack(X[i], y[i], W, b)) for i in ok]
    flips = [(i, r) for i, r in results if r[3].argmax() != y[i]]
    rate = len(flips) / len(results)
    med_conf = float(np.median([r[3].max() for _, r in flips])) if flips else 0.0
    show = flips[:4]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for j, (i, (score, xa, mask, p)) in enumerate(show):
        axes[0, j].imshow(X[i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        axes[0, j].set_title(f"'{y[i]}' — correct", fontsize=10)
        axes[1, j].imshow(xa.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        r, c = np.argwhere(mask.reshape(8, 8))[0]
        axes[1, j].add_patch(plt.Rectangle((c - 0.5, r - 0.5), 3, 3, fill=False,
                                           ec=C["red"], lw=2))
        axes[1, j].set_title(f"→ '{p.argmax()}' ({p.max():.0%})", fontsize=10,
                             color=C["red"])
        for rr in range(2):
            axes[rr, j].set_xticks([]); axes[rr, j].set_yticks([])
    axes[0, 0].set_ylabel("original", fontsize=10)
    axes[1, 0].set_ylabel("one 3 × 3 patch\nchanged", fontsize=10)
    fig.suptitle(f"changing only 9 of 64 pixels, in one 3 × 3 patch, fools the "
                 f"classifier on {rate:.0%} of the {len(results)} digits tried",
                 fontsize=10.5)
    png = save_mpl(fig, ctx, "patch_attack")
    return Figure("patch_attack",
                  "As the attack in the previous figure adds noise across the whole image, "
                  "which cannot be done to a real scene, adversarial examples may appear to "
                  "be of little practical relevance. However, the perturbation does not need "
                  "to cover the whole image. Here only one 3 × 3 patch, 9 of the 64 pixels, "
                  "may be changed by the attack, to any values, in the manner of a sticker, "
                  "and the position and values are found by gradient ascent on the loss. "
                  f"This is sufficient to fool the trained classifier on {rate:.0%} of the "
                  f"digits tested, with a median confidence in the incorrect answer of "
                  f"{med_conf:.0%}. This is the digital equivalent of the physical attack in "
                  "the lecture slides, in which black and white tape placed at chosen points "
                  "on a stop sign caused a network to classify it as a speed-limit sign over "
                  "a wide range of distances and viewpoints, with clear implications for "
                  "safety-critical systems such as self-driving cars.",
                  png=png)


def fig_pose_generalisation(ctx: BuildContext) -> Figure:
    use_source("sklearn_digits")
    m = digits_classifier()
    W, b, X, y = m["W"], m["b"], m["Xte"], m["yte"]
    angles = np.arange(0, 181, 15)
    acc = []
    for a in angles:
        R = np.array([ndimage.rotate(x.reshape(8, 8), a, reshape=False, order=1,
                                     mode="constant").ravel() for x in X])
        acc.append((_probs(np.clip(R, 0, 1), W, b).argmax(1) == y).mean())
    idx = int(np.where(y == 7)[0][0])
    fig = plt.figure(figsize=(13, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.22)
    ax = fig.add_subplot(gs[0])
    ax.plot(angles, acc, color=C["red"], lw=2.2, marker="o")
    ax.axhline(0.1, color=C["grey"], ls=":", lw=1.2)
    ax.text(178, 0.12, "chance", ha="right", fontsize=8.5, color=C["grey"])
    ax.set_xlabel("rotation of the test digits (degrees)")
    ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.02); ax.set_xticks(angles[::2])
    ax.set_title(f"upright: {acc[0]:.0%}   ·   at 45°: {acc[3]:.0%}   ·   "
                 f"upside down: {acc[-1]:.0%}", fontsize=10.5)
    sub = gs[1].subgridspec(1, 4, wspace=0.1)
    for k, a in enumerate((0, 30, 60, 90)):
        axk = fig.add_subplot(sub[k])
        R = ndimage.rotate(X[idx].reshape(8, 8), a, reshape=False, order=1, mode="constant")
        p = _probs(np.clip(R.ravel(), 0, 1)[None], W, b)[0]
        axk.imshow(R, cmap="gray_r", vmin=0, vmax=1)
        axk.set_title(f"{a}° → '{p.argmax()}'", fontsize=9.5,
                      color=C["green"] if p.argmax() == y[idx] else C["red"])
        axk.set_xticks([]); axk.set_yticks([])
    png = save_mpl(fig, ctx, "pose_generalisation")
    return Figure("pose_generalisation",
                  "Lack of generalisation, measured. No attack is applied to this "
                  "classifier; its test digits are rotated, a change of pose that would "
                  "hardly be noticed by a human observer. As all of the training digits are "
                  "upright, the accuracy decreases rapidly with angle and reaches the chance "
                  "level at approximately 75°; the partial recovery near 180° occurs only "
                  "because some digits, such as 0, 1 and 8, appear similar when upside down. "
                  "The same failure has been shown for deep ImageNet networks, in which a "
                  "school bus, a scooter or a fire truck is recognised in the poses common "
                  "in the training data and misclassified when the pose changes. A network "
                  "may generalise well within its training distribution and still behave "
                  "unexpectedly outside it; for example, a pedestrian lying in the road, "
                  "seen from an angle absent from the training data, could cause the kind of "
                  "error that makes deployment in safety-critical applications dangerous.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_reconstruction(ctx), fig_activation_maximisation(ctx),
            fig_class_templates(ctx), fig_style_transfer(ctx),
            fig_style_transfer_loop(ctx), fig_deep_dream(ctx),
            fig_adversarial_fgsm(ctx), fig_patch_attack(ctx),
            fig_pose_generalisation(ctx)]


# ---------------------------------------------------------------------------
# Streamlit demo
# ---------------------------------------------------------------------------

def demo() -> None:
    import streamlit as st
    st.title("2.5 Visualisation and adversarial attack")
    tab1, tab2 = st.tabs(["Adversarial attack", "Reconstruction"])

    with tab1:
        m = digits_classifier()
        W, b, X, y = m["W"], m["b"], m["Xte"], m["yte"]
        i = st.slider("test digit", 0, len(y) - 1, 0)
        eps = st.slider("ε", 0.0, 0.4, 0.1, 0.01)
        mode = st.radio("perturbation", ["FGSM", "random ±ε"], horizontal=True)
        x = X[i:i + 1]
        if mode == "FGSM":
            xa = fgsm(x, y[i:i + 1], eps, W, b)
        else:
            xa = np.clip(x + eps * np.random.default_rng(i).choice([-1, 1], x.shape), 0, 1)
        p0, pa = _probs(x, W, b)[0], _probs(xa, W, b)[0]
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(x.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        axes[0].set_title(f"'{p0.argmax()}' {p0.max():.0%}")
        axes[1].imshow((xa - x).reshape(8, 8), cmap="RdBu_r", vmin=-0.4, vmax=0.4)
        axes[1].set_title("change")
        axes[2].imshow(xa.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        axes[2].set_title(f"'{pa.argmax()}' {pa.max():.0%}")
        for a in axes:
            a.axis("off")
        st.pyplot(fig)
        st.caption(f"true label: {y[i]}")

    with tab2:
        key = st.selectbox("image", ["owl", "lion", "zebra", "sphinx"])
        layer = st.select_slider("VGG-16 layer", REC_LAYERS, value="conv3_3")
        st.caption(f"running on {TV.device()}")
        if st.button("reconstruct from noise"):
            img = _square(load_image(key), 160)
            with st.spinner("optimising the image..."):
                r = TV.reconstruct(img, layer, iters=150)
            c1, c2 = st.columns(2)
            c1.image(img, caption="target")
            c2.image(r, caption=f"from {layer} features")
