"""Lesson 2.5 — Feature visualisation and adversarial attack.

Figures
    reconstruction          inversion: what survives of an image at increasing depth
    activation_maximisation the input that most excites one unit, at four depths
    class_templates         activation maximisation on a trained classifier's outputs
    texture_statistics      style as Gram matrices; textures re-synthesised from them
    style_transfer_loop     how content and style losses combine into style transfer
    deep_dream              gradient ascent on a photo, amplifying what a layer sees
    adversarial_fgsm        the fast gradient sign attack, and why it is not just noise
    patch_attack            changing only a small patch — the stop-sign stickers
    pose_generalisation     accuracy as test digits are rotated away from upright

Two genuine models are used, and the captions say which:
  * engines/_fixednet.py — a FIXED network (a Gabor filter bank, ReLU and
    pooling) with an exact input gradient. The visualisation algorithms run on
    it for real; what they reveal is how depth and pooling change what a layer
    retains, not what a trained CNN has learned.
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
from .. import _fixednet as FN

FIGURES = ["reconstruction", "activation_maximisation", "class_templates",
           "texture_statistics", "style_transfer_loop", "deep_dream",
           "adversarial_fgsm", "patch_attack", "pose_generalisation"]

DEPTHS = (0, 2, 3, 4)


def receptive_field(depth, k=FN.BANK.shape[-1]):
    """conv k x k (stride 1) then average pooling 2^d at stride 2^d (lesson 1.4)."""
    p = 2 ** depth
    r = 1
    r = p * r + (p - p)          # back through the pooling layer (k = s = p)
    r = 1 * r + (k - 1)          # back through the convolution (stride 1)
    return r


def _square(img, n):
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)


def _texture_crop(key, frac, n=64):
    im = load_image(key)
    h, w = im.shape[:2]
    y0, y1, x0, x1 = frac
    return _square(im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)], n).transpose(2, 0, 1) / 255.0


def synthesise_texture(style, depths=(0, 1, 2), iters=150, seed=0):
    """Gatys-style texture synthesis: match only the Gram matrices (plus mean colour),
    starting from noise. No content term, so the layout is free to change."""
    gs = {d: FN.gram(FN.features(style, d)[0]) for d in depths}
    gss = {d: (g ** 2).sum() for d, g in gs.items()}
    mu = style.mean((1, 2), keepdims=True)

    def fg(x):
        tot, grad = 0.0, np.zeros_like(x)
        for d, G0 in gs.items():
            phi, cache = FN.features(x, d)
            Fm = phi.reshape(phi.shape[0], -1)
            dG = Fm @ Fm.T / Fm.shape[1] - G0
            tot += (dG ** 2).sum() / gss[d]
            grad += FN.backward(((4.0 / gss[d]) * (dG @ Fm) / Fm.shape[1]).reshape(phi.shape),
                                cache)
        dm = x.mean((1, 2), keepdims=True) - mu      # Gabor filters are zero-mean, so
        tot += 5 * (dm ** 2).sum()                   # colour must be matched separately
        grad += 10 * dm / (x.shape[1] * x.shape[2])
        t, gt = FN.tv(x)
        return tot + 2e-5 * t, grad + 2e-5 * gt     # just enough to suppress pixel noise

    x0 = np.random.default_rng(seed).uniform(0.3, 0.7, style.shape)
    return FN.optimise(fg, x0, iters)[0]


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

def fig_reconstruction(ctx: BuildContext) -> Figure:
    img = _square(load_image("owl", gray=True), 96) / 255.0
    recs = [FN.reconstruct(img, d, iters=200) for d in DEPTHS]
    corr = [np.corrcoef(img.ravel(), r.ravel())[0, 1] for r in recs]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.3))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("target image", fontsize=10)
    for ax, d, r, cc in zip(axes[1:], DEPTHS, recs, corr):
        ax.imshow(r, cmap="gray", vmin=0, vmax=1)
        fm = 96 // 2 ** d
        ax.set_title(f"from depth {d}: {fm}×{fm} maps\ncorrelation {cc:.2f}", fontsize=9.5)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "reconstruction")
    return Figure("reconstruction",
                  "**Reconstruction** (inversion): what information do a layer's "
                  "feature maps keep about the image? The target is passed through the "
                  "network and its features at one depth stored. Then, starting from "
                  "random noise, gradient descent adjusts the *pixels* until the noise "
                  "image produces the same features, with a smoothness penalty as the "
                  "regulariser that keeps the result natural-looking. From the earliest "
                  "features the reconstruction is almost exact. From deeper ones detail "
                  "is lost and only an increasingly abstract composition of the image "
                  "survives, as the falling correlation with the original shows. The "
                  "network here is a fixed one — oriented Gabor filters, ReLU and "
                  "pooling, where 'deeper' means more pooling — not a trained CNN, so "
                  "it shows the algorithm and the effect of depth honestly but not what "
                  "a trained network's deep layers encode. In a trained network the "
                  "deep layers keep *semantic* information about objects instead.",
                  png=png)


def fig_activation_maximisation(ctx: BuildContext) -> Figure:
    filters = [0, 2, 4, 6]            # four orientations, even phase, short wavelength
    n = 64
    fig, axes = plt.subplots(len(filters), len(DEPTHS), figsize=(10, 9.4))
    for j, d in enumerate(DEPTHS):
        rf = receptive_field(d)
        # the maximised unit is the centre of the POOLED map, which covers input
        # pixels cy*p .. cy*p+p-1, widened by the kernel's half-width each side
        p_, half = 2 ** d, FN.BANK.shape[-1] // 2
        left = (n // p_) // 2 * p_ - half - 0.5
        for i, f in enumerate(filters):
            x = FN.activation_max(n, d, f)
            ax = axes[i, j]
            ax.imshow(x, cmap="gray", vmin=x.min(), vmax=x.max())
            ax.add_patch(plt.Rectangle((left, left), rf, rf,
                                       fill=False, ec=C["red"], lw=1.2, ls="--"))
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, j].set_title(f"depth {d}\nreceptive field {rf} × {rf}", fontsize=10)
    for i, f in enumerate(filters):
        axes[i, 0].set_ylabel(f"filter {f // 2 + 1}", fontsize=9)
    png = save_mpl(fig, ctx, "activation_maximisation")
    return Figure("activation_maximisation",
                  "**Activation maximisation**: which input makes one neuron respond "
                  "most strongly? Starting from a near-blank image, **gradient ascent** "
                  "— ascent, because we are maximising — changes the pixels to increase "
                  "the output of one unit, here the centre of one feature map, while a "
                  "penalty on the pixel values keeps the image from running away to "
                  "extreme noise patterns that also excite the unit. Each row is a "
                  "different filter, each column a deeper layer of the same fixed "
                  "network. The preferred stimulus is that filter's oriented grating, "
                  "and it **grows with depth**, because the unit's receptive field "
                  "grows — the dashed square is the receptive field computed with "
                  "Lesson 1.4's formula. That growth is exactly what the slide's "
                  "VGG-16 visualisations show. In a trained network the content also "
                  "changes with depth, from edges and colour blobs to textures to "
                  "object parts; this fixed network has only edges to offer, so only "
                  "the size changes here.", png=png)


def fig_class_templates(ctx: BuildContext) -> Figure:
    use_source("sklearn_digits")
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
                  "Activation maximisation on a genuinely **trained** model, a "
                  "classifier for handwritten digits with one output neuron per class. "
                  "For a single layer like this the answer can be read straight off "
                  "the weights (top row: red pixels raise the score, blue lower it), "
                  "and the input image that maximises each neuron is just the pixels "
                  "with positive weight (bottom row). Some, like '0' and '7', look like "
                  "the digit. Others are only fragments, and every neuron cares as much "
                  "about where ink must be *absent* — the blue weights, such as the "
                  "empty centre of the '0' — as about where it should be, because that "
                  "is what separates its class from the others. "
                  "This is the point of the slide on visualisation for understanding: "
                  "a network may recognise a class by properties that merely co-occur "
                  "with it — the fluffiness of clouds, the sky behind a building — "
                  "rather than by the object itself, which can cause confusion when it "
                  "is deployed.", png=png)


def fig_texture_statistics(ctx: BuildContext) -> Figure:
    srcs = [("zebra", (0.25, 0.7, 0.3, 0.75)), ("brickwork", (0.0, 0.45, 0.0, 1.0)),
            ("macaw", (0.3, 0.9, 0.45, 0.95))]
    styles = [(k, _texture_crop(k, f)) for k, f in srcs]
    synths = [synthesise_texture(s) for _, s in styles]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7.0))
    for j, ((k, s), syn) in enumerate(zip(styles, synths)):
        axes[0, j].imshow(s.transpose(1, 2, 0))
        axes[0, j].set_title(f"texture: {k}", fontsize=10)
        axes[1, j].imshow(syn.transpose(1, 2, 0))
        axes[1, j].set_title("re-synthesised from noise\nby matching only its Gram matrices",
                             fontsize=9.5)
        for r in range(2):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    png = save_mpl(fig, ctx, "texture_statistics")
    return Figure("texture_statistics",
                  "What 'style' means to a network. The **Gram matrix** of a layer is "
                  "the inner product of every pair of its feature maps, summed over "
                  "all positions: entry (i, j) is large when features i and j tend to "
                  "fire together, anywhere in the image. Summing over positions throws "
                  "away *where* things are and keeps only *which features co-occur* — "
                  f"which is what a texture is. For this network the Gram matrix of a "
                  f"colour image is {3 * len(FN.BANK)} × {3 * len(FN.BANK)} — one row "
                  "and column per feature map — however large the image. The proof "
                  "that it captures the texture is the bottom row: starting "
                  "from noise and adjusting the pixels until the Gram matrices match "
                  "(at three depths), the stripes, bricks and feathers come back, with "
                  "the right orientations and colours but a completely different "
                  "layout. This runs for real on the lesson's fixed Gabor network, whose "
                  "features are good enough for texture. Full style transfer also needs "
                  "a content term from a *trained* network's deep layers, which this "
                  "network does not have — the next figure shows how the two are "
                  "combined.", png=png)


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
                  "Training style transfer. The content image is passed through a "
                  "pre-trained network and a deep feature map stored (conv4 in the "
                  "original work): deep layers keep the content — the objects and their "
                  "arrangement — but not fine detail. The style image is passed through "
                  "and the Gram matrices of all its convolution layers stored, capturing "
                  "texture at small scales in early layers and larger scales in deep "
                  "ones. A random-noise image is then passed through, and its features "
                  "and Gram matrices compared with those stored, in a content loss and "
                  "a style loss. Gradient descent adjusts the noise image's pixels to "
                  "minimise both together, and it converges to an image with the "
                  "content of one and the style of the other. The weights α and β set "
                  "the balance. As with reconstruction, it is the image that is "
                  "optimised, not the network.", png=png)


def fig_deep_dream(ctx: BuildContext) -> Figure:
    src = load_image("pyramids", gray=True)
    g = cv2.resize(src, (176, 128), interpolation=cv2.INTER_AREA) / 255.0
    pad = 12
    gp = np.pad(g, pad, mode="reflect")          # reflect-pad so the border does not
    outs = []                                    # get dreamed into a frame
    for d in (1, 3):
        outs.append(FN.deep_dream(gp, d, steps=40)[pad:-pad, pad:-pad])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].imshow(g, cmap="gray", vmin=0, vmax=1); axes[0].set_title("input photograph",
                                                                     fontsize=10)
    for ax, o, d in zip(axes[1:], outs, (1, 3)):
        ax.imshow(o, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"after 40 steps of gradient ascent on depth {d}", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "deep_dream")
    return Figure("deep_dream",
                  "**Deep dream** is activation maximisation started from a photograph "
                  "instead of noise. The photograph makes the network respond to the "
                  "features it already contains, and gradient ascent on those "
                  "responses amplifies them, pushing the image towards whatever the "
                  "layer sees in it. Run here on the lesson's fixed network, whose only "
                  "features are oriented edges, it can only amplify edges: the pyramids' "
                  "outlines are reinforced and haloed, more coarsely from the deeper, "
                  "more pooled layer. With a trained network, maximising many deep "
                  "layers at once amplifies learned object parts instead, which is "
                  "where the hallucinatory images come from — horizon lines become "
                  "towers and pagodas, leaves become birds and insects — revealing what "
                  "the network associates with those image features.", png=png)


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
                  "An **adversarial attack** on a real trained classifier. The **fast "
                  "gradient sign method** computes the gradient of the loss with "
                  "respect to the *input pixels* — the direction in which each pixel "
                  "should move to make the network more wrong — and adds a small step "
                  "ε in the sign of it: x' = x + ε·sign(∇ₓL). Left: a digit the "
                  f"classifier gets right with {p0[i, y[i]]:.0%} confidence becomes, "
                  f"after a change of at most {e:.2f} per pixel, a '{pa.argmax()}' "
                  f"with {pa.max():.0%} confidence — wrong, and more sure of itself than "
                  "many correct answers. Right: the crucial comparison. Random noise of "
                  "exactly the same size barely affects accuracy, while the same "
                  "budget aimed by the gradient destroys it. Adversarial examples are "
                  "not noise — they are carefully chosen. Each output depends on many "
                  "inputs through many multiplications, so many tiny, coordinated "
                  "changes add up to a large change at the output. Deep networks are "
                  "affected just as this simple classifier is, attacks crafted against "
                  "one network often transfer to others, and building robust networks "
                  "is an active research area.", png=png)


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
                  "It is tempting to dismiss adversarial examples because the attack "
                  "adds noise across the whole image, which nobody can do to a real "
                  "scene. But the perturbation does not need to cover the image. Here "
                  "the attack may change only one 3 × 3 patch — 9 of the 64 pixels — "
                  "to any values at all, like a sticker, and the position and values "
                  "are found by gradient ascent on the loss. That is enough to fool "
                  f"the trained classifier on {rate:.0%} of the digits tried, with a "
                  f"median confidence in the wrong answer of {med_conf:.0%}. This is "
                  "the digital analogue of the physical "
                  "attack on the slide, where black and white tape placed at chosen "
                  "points on a stop sign made a network read it as a speed-limit sign "
                  "across a wide range of distances and viewpoints — with obvious "
                  "implications for safety-critical systems such as self-driving cars.",
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
                  "Lack of generalisation, measured. Nobody is attacking this "
                  "classifier; its test digits are simply rotated, a change of pose a "
                  "person would barely notice. Because the training digits are all "
                  "upright, accuracy falls steeply with the angle and reaches chance by "
                  "about 75°; the partial recovery near 180° is only because some "
                  "digits, such as 0, 1 and 8, look much the same upside down. The "
                  "same failure has "
                  "been shown for deep ImageNet networks, which recognise a school bus, "
                  "a scooter or a fire truck in the poses common in their training "
                  "data and misclassify them when the pose changes. A network can "
                  "generalise well within its training distribution and still behave "
                  "unexpectedly outside it — a pedestrian lying in the road, seen from "
                  "an angle absent from the training data, is exactly the kind of "
                  "error that makes deployment in safety-critical applications "
                  "dangerous.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    FN.gradcheck()
    return [fig_reconstruction(ctx), fig_activation_maximisation(ctx),
            fig_class_templates(ctx), fig_texture_statistics(ctx),
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
        d = st.slider("depth (more pooling)", 0, 4, 2)
        if st.button("reconstruct from noise"):
            img = _square(load_image(key, gray=True), 96) / 255.0
            r = FN.reconstruct(img, d, iters=150)
            c1, c2 = st.columns(2)
            c1.image(img, caption="target", clamp=True)
            c2.image(r, caption=f"from depth {d} features", clamp=True)
