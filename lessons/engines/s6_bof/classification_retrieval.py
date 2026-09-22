"""Lesson 6.2 — Classification and image retrieval with the bag of features.

Figures
    bof_classification  the vocabulary-then-SVM pipeline, with a measured result on two scene classes
    spatial_pyramid     putting the spatial information back: grid levels, weights, vector length
    retrieval_task      a query image ranked against a database of credited images
    inverted_index      the word → image table, and the lookup for one query
    vocab_tree          hierarchical k-means: a vocabulary tree and the search cost
    bof_summary         what the representation is good and bad at, and the link to CNN filters
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.svm import SVC

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       slider_frames_layout, load_image, to_data_uri, norm01)
from ._bof_common import (ANIMALS, PLACES, DB_KEYS, vocabulary, describe_query,
                          rank_database, variant, dense_descriptors, term_frequency,
                          codeword_patches, informative_words)

FIGURES = ["bof_classification", "spatial_pyramid", "retrieval_task",
           "inverted_index", "vocab_tree", "bof_summary"]

PYRAMID_LEVELS = 3


def _box(ax, xy, w, h, text, fc="#e8f1f8", fs=8):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=fc, ec=C["grey"], lw=1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=13,
                                 color=C["ink"], lw=1.2))


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def scene_classification(k_words: int = 120):
    """Leave-one-out accuracy of an SVM on bag-of-features vectors, two scene classes."""
    keys = tuple(ANIMALS + PLACES)
    voc = vocabulary(keys, k_words=k_words)
    X = voc["tf"]
    y = np.r_[np.zeros(len(ANIMALS), int), np.ones(len(PLACES), int)]
    out = {}
    for name, clf in [("RBF kernel", SVC(kernel="rbf", C=10, gamma="scale")),
                      ("linear kernel", SVC(kernel="linear", C=10))]:
        s = cross_val_score(clf, X, y, cv=LeaveOneOut())
        out[name] = (s.mean(), int(s.sum()), len(s))
    return voc, X, y, out


def pyramid_histograms(gray, voc, levels: int = PYRAMID_LEVELS):
    """Per-cell term histograms for each pyramid level, with the Lazebnik weights."""
    kps, desc = dense_descriptors(gray, voc["step"])
    ids = voc["km"].predict(desc) if len(desc) else np.zeros(0, int)
    xy = np.array([k.pt for k in kps]) if kps else np.zeros((0, 2))
    h, w = gray.shape
    out = []
    for lv in range(levels):
        n = 2 ** lv
        cells = np.empty((n, n, voc["k_words"]))
        for r in range(n):
            for c in range(n):
                m = ((xy[:, 1] >= r * h / n) & (xy[:, 1] < (r + 1) * h / n) &
                     (xy[:, 0] >= c * w / n) & (xy[:, 0] < (c + 1) * w / n)) if len(xy) \
                    else np.zeros(0, bool)
                cells[r, c] = term_frequency(ids[m], voc["k_words"])
        # Lazebnik's weights, with L the index of the finest level: the coarsest
        # level gets 1/2**L and level l > 0 gets 1/2**(L - l + 1).
        L = levels - 1
        weight = 1 / 2 ** L if lv == 0 else 1 / 2 ** (L - lv + 1)
        out.append(dict(level=lv, n=n, cells=cells, weight=weight))
    return out, kps, ids


def pyramid_length(k_words: int, levels: int = PYRAMID_LEVELS) -> int:
    return k_words * sum(4 ** lv for lv in range(levels))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_bof_classification(ctx: BuildContext) -> Figure:
    voc, X, y, scores = scene_classification()
    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.62, 1], hspace=0.25)
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 13); ax.set_ylim(0, 2.3); ax.axis("off")
    ax.text(0.15, 2.2, "(a) the classification pipeline", fontsize=9.5, weight="bold",
            color=C["grey"])
    x = 0.15
    for text, fc in [("a large image dataset\n(no labels needed)", "#e8f1f8"),
                     ("describe features,\ncluster → vocabulary", "#e8f1f8"),
                     ("labelled training images\n→ term vectors", "#fdf1e3"),
                     ("train an SVM,\nusually an RBF kernel", "#e6f4ee"),
                     ("a new image → term\nvector → class label", "#f3e8f5")]:
        _box(ax, (x, 0.55), 2.15, 1.2, text, fc=fc, fs=8)
        if x < 10:
            _arrow(ax, (x + 2.22, 1.15), (x + 2.6, 1.15))
        x += 2.55
    ax.text(0.15, 0.2, "Building the vocabulary is the overhead the representation adds: it has "
                       "to be done before any classifier can be trained.", fontsize=8.5,
            color=C["grey"])
    sub = gs[1].subgridspec(2, 9, width_ratios=[1] * 8 + [2.0], wspace=0.08, hspace=0.3)
    for j, k in enumerate(ANIMALS[:8]):
        a = fig.add_subplot(sub[0, j]); a.imshow(load_image(k, max_side=180))
        a.set_xticks([]); a.set_yticks([])
        if j == 0:
            a.set_ylabel("class 0\nanimals", fontsize=8)
    for j, k in enumerate(PLACES[:8]):
        a = fig.add_subplot(sub[1, j]); a.imshow(load_image(k, max_side=180))
        a.set_xticks([]); a.set_yticks([])
        if j == 0:
            a.set_ylabel("class 1\nplaces and\nvehicles", fontsize=8)
    a = fig.add_subplot(sub[:, 8]); a.axis("off")
    a.text(0, 1.0, "(b) a measured result", fontsize=9.5, weight="bold", va="top")
    a.text(0, 0.86, f"{len(y)} credited images in two classes,\n"
                    f"a {voc['k_words']}-word vocabulary built from\nthem, "
                    "leave-one-out cross-validation:", fontsize=8.5, va="top")
    ys = 0.58
    for name, (acc, hit, n) in scores.items():
        a.text(0, ys, f"SVM, {name}: {100 * acc:.1f}%  ({hit}/{n})", fontsize=9.5,
               va="top", color=C["green"] if "RBF" in name else C["blue"])
        ys -= 0.12
    a.text(0, 0.30, "The RBF kernel does better because it can\nlearn a more complex "
                    "decision boundary in\nthe histogram space.", fontsize=8.5, va="top")
    a.text(0, 0.10, "This dataset is tiny. The point is the pipeline,\nnot the number: "
                    "on real benchmarks the bag of\nfeatures was the state of the art until "
                    "deep\nnetworks overtook it in 2012.", fontsize=8, va="top",
           color=C["grey"])
    png = save_mpl(fig, ctx, "bof_classification")
    return Figure("bof_classification",
                  "Image classification with the bag of features. The vocabulary is built "
                  "first, without labels; the labelled training images are then converted "
                  "to term vectors and a support vector machine is trained on them, "
                  "typically with a radial basis function kernel so that a more complex "
                  "decision boundary can be learnt. Here the two classes are animals and "
                  "built scenes, drawn from the credited images used throughout these "
                  "lessons. The dataset is far too small to be a benchmark, but the "
                  "measured accuracies show the pipeline working end to end.", png=png)


def fig_spatial_pyramid(ctx: BuildContext) -> Figure:
    voc = vocabulary(tuple(DB_KEYS), k_words=200)
    g = load_image("trevi", gray=True, max_side=360)
    rgb = load_image("trevi", max_side=360)
    levels, kps, ids = pyramid_histograms(g, voc)
    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.1], height_ratios=[1, 0.9],
                          hspace=0.35, wspace=0.25)
    for lv, info in enumerate(levels):
        ax = fig.add_subplot(gs[0, lv]); ax.imshow(rgb)
        n = info["n"]; h, w = g.shape
        for i in range(1, n):
            ax.axvline(i * w / n, color=C["yellow"], lw=1.6)
            ax.axhline(i * h / n, color=C["yellow"], lw=1.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"level {lv}: {n} × {n} = {n * n} cell"
                     f"{'s' if n > 1 else ''}\nweight {info['weight']:g}", fontsize=9)
    ax = fig.add_subplot(gs[:, 3]); ax.axis("off")
    ax.text(0, 1.0, "the spatial pyramid", fontsize=9.5, weight="bold", va="top")
    ax.text(0, 0.94, "The bag of features throws spatial\ninformation away. Lazebnik and "
                     "Schmid\nput some of it back by splitting the image\ninto a pyramid "
                     "of grids, forming a term\nhistogram in every cell and\nconcatenating "
                     "them all into one long\nvector.", fontsize=8.5, va="top")
    total = pyramid_length(voc["k_words"])
    ax.text(0, 0.62, f"with a {voc['k_words']}-word vocabulary\nand {PYRAMID_LEVELS} levels:\n"
                     + "\n".join(f"  level {i['level']}: {i['n'] * i['n']:2d} cells x "
                                 f"{voc['k_words']} = {i['n'] * i['n'] * voc['k_words']:5d}"
                                 for i in levels)
                     + f"\n  total term vector: {total} values",
            fontsize=7.5, va="top", family="monospace")
    ax.text(0, 0.30, "Cells at finer levels are weighted\nmore heavily, and the SVM uses the\n"
                     "pyramid match kernel rather than an\nRBF. The idea is still in use in "
                     "vision\nnetworks today.",
            fontsize=8.5, va="top", color=C["grey"])
    ax = fig.add_subplot(gs[1, 0:2])
    cells = levels[1]["cells"]
    for r in range(2):
        for c in range(2):
            off = (r * 2 + c) * voc["k_words"]
            ax.bar(np.arange(voc["k_words"]) + off, cells[r, c], width=1.0,
                   color=SERIES[r * 2 + c], label=f"cell ({r},{c})")
    for b in range(1, 4):
        ax.axvline(b * voc["k_words"], color=C["grey"], lw=0.8, ls=":")
    ax.set_xlabel("concatenated histograms of the four level-1 cells")
    ax.set_ylabel("term frequency"); ax.legend(fontsize=7.5, ncols=4)
    ax.set_title("(b) the four level-1 cells produce four histograms, laid end to end",
                 fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    lens = [voc["k_words"] * sum(4 ** i for i in range(L + 1)) for L in range(5)]
    ax.plot(range(1, 6), lens, "o-", color=C["blue"], ms=5)
    ax.set_yscale("log"); ax.set_ylim(lens[0] * 0.5, lens[-1] * 4)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("pyramid levels"); ax.set_ylabel("length of the term vector")
    ax.set_title("(c) the cost: the vector grows by a factor of four per level",
                 fontsize=9)
    for L, v in zip(range(1, 6), lens):
        ax.annotate(f"{v:,}", (L, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5)
    png = save_mpl(fig, ctx, "spatial_pyramid")
    return Figure("spatial_pyramid",
                  "The spatial pyramid. Because the bag of features is orderless, two very "
                  "different scenes made of the same parts get the same term vector. "
                  "Dividing the image into a pyramid of grids, forming a histogram in each "
                  "cell and concatenating them with weights that favour the finer levels "
                  "restores some spatial information and measurably improves "
                  "classification. The price is a term vector that grows by a factor of "
                  "four with every level added.", png=png)


def fig_retrieval_task(ctx: BuildContext) -> Figure:
    voc = vocabulary(tuple(DB_KEYS), k_words=200)
    qkey, kind = "notre_dame_a", "rotate"
    qrgb = variant(load_image(qkey, max_side=360), kind)
    q = describe_query(cv2.cvtColor(qrgb, cv2.COLOR_RGB2GRAY), voc)
    order, sim = rank_database(voc, q["tfidf"])
    # how well the whole database is retrieved under each change, for the right-hand panel
    stats = {}
    for k in ["crop", "rotate", "scale", "brighten"]:
        ranks = []
        for i, key in enumerate(voc["keys"]):
            v = describe_query(cv2.cvtColor(variant(load_image(key, max_side=360), k),
                                            cv2.COLOR_RGB2GRAY), voc)["tfidf"]
            o, _ = rank_database(voc, v)
            ranks.append(int(np.flatnonzero(o == i)[0]) + 1)
        ranks = np.array(ranks)
        stats[k] = (100 * (ranks == 1).mean(), 100 * (ranks <= 3).mean())
    fig = plt.figure(figsize=(13, 3.6))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 1.5], wspace=0.1)
    ax = fig.add_subplot(gs[0]); ax.imshow(qrgb); ax.set_anchor("N")
    ax.set_title(f"query\n({kind}d copy of a\ndatabase image)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for j in range(4):
        i = order[j]
        key = voc["keys"][i]
        ax = fig.add_subplot(gs[1 + j]); ax.imshow(load_image(key, max_side=300))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_anchor("N")
        right = key == qkey
        ax.set_title(f"rank {j + 1}: {key.replace('_', ' ')}\nsimilarity {sim[i]:.3f}",
                     fontsize=8.5, color=C["green"] if right else C["ink"])
        if right:
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(C["green"]); s.set_linewidth(3)
    ax = fig.add_subplot(gs[5]); ax.axis("off")
    ax.text(0, 1.0, "bag-of-features retrieval", fontsize=9.5, weight="bold", va="top")
    ax.text(0, 0.86, f"Database: {len(voc['keys'])} credited images, a\n"
                     f"{voc['k_words']}-word vocabulary, tf-idf weighted\n"
                     "vectors compared by cosine similarity.", fontsize=8.5, va="top")
    lines = "\n".join(f"{k:9s} top-1 {a:5.1f}%   top-3 {b:5.1f}%"
                      for k, (a, b) in stats.items())
    ax.text(0, 0.58, "every image queried by a changed copy:\n" + lines,
            fontsize=8.5, va="top", family="monospace")
    ax.text(0, 0.28, "The method is unsupervised, so it works on any\ndataset without "
                     "labels. The alternative, content-\nbased retrieval, classifies the "
                     "images first and\nreturns those with the same label, which needs\n"
                     "labelled training data.", fontsize=8.5, va="top")
    png = save_mpl(fig, ctx, "retrieval_task")

    frames = []
    for k in ["crop", "rotate", "scale", "brighten"]:
        vr = variant(load_image(qkey, max_side=360), k)
        v = describe_query(cv2.cvtColor(vr, cv2.COLOR_RGB2GRAY), voc)["tfidf"]
        o, s = rank_database(voc, v)
        top = o[:8]
        frames.append(go.Frame(name=k,
                               data=[go.Bar(x=[voc["keys"][i].replace("_", " ") for i in top],
                                            y=s[top],
                                            marker_color=[C["green"] if voc["keys"][i] == qkey
                                                          else C["blue"] for i in top])],
                               layout=go.Layout(title=f"query: {k} — correct image at rank "
                                                      f"{int(np.flatnonzero(o == voc['keys'].index(qkey))[0]) + 1}")))
    fig2 = go.Figure(data=frames[0].data, frames=frames)
    fig2.update_layout(**slider_frames_layout(["crop", "rotate", "scale", "brighten"],
                                              "change: "),
                       title=frames[0].layout.title.text,
                       yaxis_title="cosine similarity")
    return Figure("retrieval_task",
                  "Image retrieval with the bag of features. The query is a changed copy "
                  "of one database image, and the database is ranked by the cosine "
                  "similarity of the tf-idf weighted term vectors. Across the whole "
                  "database the correct image comes back at rank one for most changes of "
                  "crop, rotation, scale and brightness, which is the robustness the "
                  "orderless representation buys. Retrieval needs no labels at all, unlike "
                  "content-based retrieval, which classifies first and then matches labels.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def fig_inverted_index(ctx: BuildContext) -> Figure:
    voc = vocabulary(tuple(DB_KEYS), k_words=200)
    short = [k.replace("_", " ")[:13] for k in voc["keys"]]
    qkey = "sacre_coeur"
    qrgb = variant(load_image(qkey, max_side=320), "crop")
    q = describe_query(cv2.cvtColor(qrgb, cv2.COLOR_RGB2GRAY), voc)
    # six words the query actually contains, plus two it does not, so the lookup
    # shows both a hit and a miss
    pool = [int(w) for w in informative_words(voc, 40)]
    q_words = [w for w in pool if q["tf"][w] > 0][:6]
    misses = [w for w in pool if q["tf"][w] == 0][:2]
    words = sorted(q_words + misses, key=lambda w: -voc["df"][w])
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.62, 2.0, 1.05], wspace=0.15)
    ax = fig.add_subplot(gs[0]); ax.imshow(qrgb); ax.set_xticks([]); ax.set_yticks([])
    ax.set_anchor("N")
    ax.set_title("query image\n(a cropped copy of one\ndatabase image)", fontsize=9)
    ax = fig.add_subplot(gs[1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.0, 1.03, "the inverted file index: for each visual word, which database "
                       "images contain it", fontsize=9, va="bottom")
    row_h = 1.0 / (len(words) + 1)
    ax.text(0.005, 1 - row_h * 0.5, "visual word", fontsize=8.5, weight="bold", va="center")
    ax.text(0.17, 1 - row_h * 0.5, "images containing that word", fontsize=8.5,
            weight="bold", va="center")
    for r, wd in enumerate(words):
        y = 1 - row_h * (r + 1.5)
        hit = wd in q_words
        if hit:
            ax.add_patch(Rectangle((0, y - row_h * 0.45), 1, row_h * 0.9,
                                   color=C["yellow"], alpha=0.35, zorder=0))
        ax.text(0.005, y, f"word {wd:3d}", fontsize=8.5, va="center",
                family="monospace", weight="bold" if hit else "normal")
        imgs = [short[i] for i in np.flatnonzero(voc["tf"][:, wd] > 0)]
        txt = ", ".join(imgs[:8]) + (f"  (+{len(imgs) - 8} more)" if len(imgs) > 8 else "")
        ax.text(0.17, y, txt, fontsize=7.5, va="center",
                color=C["ink"] if hit else C["grey"])
    ax.text(0.0, -0.02, "highlighted rows are the words the query contains: only the images "
                        "listed on those rows need to be looked at.",
            fontsize=8, va="top", color=C["grey"])
    ax = fig.add_subplot(gs[2]); ax.axis("off")
    counts = np.zeros(len(voc["keys"]), int)
    present = np.flatnonzero(q["tf"] > 0)
    for wd in present:
        counts += (voc["tf"][:, wd] > 0).astype(int)
    order = np.argsort(-counts)
    ax.text(0, 1.0, "why an index at all", fontsize=9.5, weight="bold", va="top")
    ax.text(0, 0.87, f"Comparing a query against every vector in a\ndatabase of a million "
                     f"images is far too slow. The\nindex stores the same information the "
                     f"other way\nround, so only the images that share a word with\nthe "
                     f"query are ever touched.", fontsize=8.5, va="top")
    lines = "\n".join(f"{short[i]:12s} {counts[i]:3d} shared words" for i in order[:6])
    ax.text(0, 0.55, f"the query has {len(present)} words; candidates ranked\n"
                     f"by shared words:\n\n" + lines, fontsize=8.5, va="top",
            family="monospace")
    ax.text(0, 0.16, "The best candidate is the original of the query.\nThe short list is "
                     "then re-ranked with the full term\nvectors, and in Video Google also "
                     "checked for\nspatial consistency: the matching words should\nsit "
                     "close together.", fontsize=8.5, va="top", color=C["grey"])
    png = save_mpl(fig, ctx, "inverted_index")
    return Figure("inverted_index",
                  "The inverted file index. Matching a query term vector against every "
                  "vector in a large database is too slow, so the same information is "
                  "stored the other way round: for each visual word, a list of the images "
                  "that contain it. Looking up the words in the query gathers a short list "
                  "of candidate images that share features with it, and only those "
                  "candidates are then compared in full. Video Google added a spatial "
                  "consistency check on the matched words, which improves object retrieval "
                  "further.", png=png)


def fig_vocab_tree(ctx: BuildContext) -> Figure:
    voc = vocabulary(tuple(DB_KEYS[:16]), k_words=40)
    D = np.vstack([voc["descs"][k] for k in voc["keys"]])
    rng = np.random.default_rng(0)
    D = D[rng.choice(len(D), 4000, replace=False)]
    branch, depth = 3, 3
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1.0], wspace=0.2)
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # draw the tree, and split the descriptors the same way at each level
    def draw(x0, x1, y, level, subset, colour):
        cx = (x0 + x1) / 2
        if level == depth:                      # leaves: too many to label
            ax.add_patch(FancyBboxPatch((cx - 0.012, y - 0.025), 0.024, 0.05,
                                        boxstyle="round,pad=0.003", fc=colour,
                                        ec=C["grey"], lw=0.6))
            return
        ax.add_patch(FancyBboxPatch((cx - 0.028, y - 0.03), 0.056, 0.06,
                                    boxstyle="round,pad=0.005", fc=colour, ec=C["grey"]))
        ax.text(cx, y, f"{len(subset)}", ha="center", va="center", fontsize=7)
        km = KMeans(branch, n_init=4, random_state=0).fit(D[subset])
        lab = km.labels_
        span = (x1 - x0) / branch
        for b in range(branch):
            nx0 = x0 + b * span
            child = subset[lab == b]
            if not len(child):
                continue
            ax.plot([cx, nx0 + span / 2], [y - 0.035, y - 0.26 + 0.035],
                    color=C["grey"], lw=0.9)
            draw(nx0, nx0 + span, y - 0.26, level + 1, child,
                 ["#e8f1f8", "#fdf1e3", "#e6f4ee", "#f3e8f5"][level + 1])
    draw(0.02, 0.98, 0.93, 0, np.arange(len(D)), "#e8f1f8")
    ax.text(0.5, 0.02, f"branching factor {branch}, depth {depth}: each labelled box holds the "
                       f"number of\ndescriptors in that node; the {branch ** depth} small boxes "
                       "at the bottom are the visual words",
            fontsize=8, ha="center")
    ax.set_title("(a) hierarchical $k$-means over the descriptors", fontsize=9)
    ax = fig.add_subplot(gs[1])
    ds = np.arange(1, 9)
    words = branch ** ds
    flat = words
    tree = branch * ds
    ax.plot(ds, flat, "o-", color=C["red"], ms=4, label="flat vocabulary:\ncompare with every word")
    ax.plot(ds, tree, "s-", color=C["green"], ms=4, label="vocabulary tree:\n$k$ comparisons per level")
    ax.set_yscale("log"); ax.set_xlabel("tree depth $L$")
    ax.set_ylabel("descriptor comparisons to assign one feature")
    ax.set_title(f"(b) vocabulary of $k^L$ = ${branch}^L$ words", fontsize=9)
    ax.legend(fontsize=7.5)
    ax = fig.add_subplot(gs[2]); ax.axis("off")
    ax.text(0, 1.0, "scaling up retrieval", fontsize=9.5, weight="bold", va="top")
    ax.text(0, 0.86, "Video Google used 4,000 frames and a 10,000-word\nvocabulary, which is "
                     "not really large scale. Nister and\nStewenius represented the "
                     "vocabulary as a tree built\nby hierarchical $k$-means: cluster all "
                     "the descriptors\ninto $k$ groups, cluster each group into $k$ again, "
                     "and\nrepeat to the chosen depth.", fontsize=8.5, va="top")
    ax.text(0, 0.46, "A query descriptor is routed down the tree by matching\nthe cluster "
                     "centres at each level, so only $k$ comparisons\nper level are needed "
                     "instead of one per word. A\nsimilarity score is accumulated on the "
                     "way down using\nthe inverted files at each node.", fontsize=8.5, va="top")
    ax.text(0, 0.12, "That gives much larger vocabularies, searches only a\nsmall fraction of "
                     "the database, and reaches sub-second\nretrieval over a million images.",
            fontsize=8.5, va="top", color=C["grey"])
    png = save_mpl(fig, ctx, "vocab_tree")
    return Figure("vocab_tree",
                  "The vocabulary tree. Instead of one flat k-means over all the "
                  "descriptors, the descriptors are clustered into k groups, each group is "
                  "clustered into k again, and so on to a chosen depth, so the leaves form "
                  "a vocabulary of k to the power L words. A query feature is routed down "
                  "the tree by comparing it with only k centres at each level, which makes "
                  "the assignment logarithmic in the vocabulary size rather than linear. "
                  "Combined with inverted files at each node this supports sub-second "
                  "retrieval over a million images.", png=png)


def fig_bof_summary(ctx: BuildContext) -> Figure:
    voc = vocabulary(tuple(DB_KEYS), k_words=200)
    words = [int(w) for w in informative_words(voc, 6)]
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], hspace=0.35)
    ax = fig.add_subplot(gs[:, 0]); ax.axis("off")
    ax.text(0, 1.0, "the bag of features: what it gives you", fontsize=10, weight="bold",
            va="top")
    ax.text(0.01, 0.88, "•  a compact summary of the whole image content\n"
                        "•  tolerance of viewpoint, scale and illumination change\n"
                        "•  good results in practice, and the state of the art in image\n"
                        "    recognition before deep learning arrived in 2012\n"
                        "•  an unsupervised vocabulary: no labelling needed to build it\n"
                        "•  retrieval over very large databases, via inverted files and\n"
                        "    a vocabulary tree", fontsize=9, va="top", color=C["green"])
    ax.text(0, 0.42, "and what it costs you", fontsize=10, weight="bold", va="top")
    ax.text(0.01, 0.30, "•  no semantic meaning: the system has no idea what any\n"
                        "    visual word represents\n"
                        "•  spatial information is discarded, so it is unsuited to object\n"
                        "    detection and localisation; the spatial pyramid recovers\n"
                        "    some of it for classification\n"
                        "•  the vocabulary must be built before anything can be trained,\n"
                        "    and it may not transfer to images unlike the ones it came from",
            fontsize=9, va="top", color=C["red"])
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    sub = gs[0, 1].subgridspec(1, 6, wspace=0.1)
    for j, wd in enumerate(words):
        pats = codeword_patches(voc, wd, 1)
        if pats:
            a = fig.add_subplot(sub[0, j]); a.imshow(pats[0], cmap="gray")
            a.set_xticks([]); a.set_yticks([])
    ax.set_title("visual words from the vocabulary", fontsize=9, y=1.12)
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    sub = gs[1, 1].subgridspec(1, 6, wspace=0.1)
    yy, xx = np.mgrid[-1:1:32j, -1:1:32j]
    for j, (theta, freq) in enumerate([(0, 3), (45, 3), (90, 3), (135, 4), (0, 6), (60, 6)]):
        t = np.deg2rad(theta)
        u = xx * np.cos(t) + yy * np.sin(t)
        gab = np.exp(-(xx ** 2 + yy ** 2) / 0.6) * np.cos(2 * np.pi * freq * u / 2)
        a = fig.add_subplot(sub[0, j]); a.imshow(norm01(gab), cmap="gray")
        a.set_xticks([]); a.set_yticks([])
    ax.set_title("oriented filters of the kind a convolutional network\nlearns in its first "
                 "layer", fontsize=9, y=1.12)
    png = save_mpl(fig, ctx, "bof_summary")
    return Figure("bof_summary",
                  "The bag of features in summary. It gives a compact, viewpoint-tolerant "
                  "description of image content, needs no labels to build its vocabulary, "
                  "and scales to very large retrieval problems, but it assigns no meaning "
                  "to its visual words and throws away spatial layout. There is also a "
                  "family resemblance to what came next: the visual words of a bag of "
                  "features are edges, bars and textures at particular orientations, much "
                  "like the filters a convolutional network learns in its first layer.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_bof_classification(ctx), fig_spatial_pyramid(ctx), fig_retrieval_task(ctx),
            fig_inverted_index(ctx), fig_vocab_tree(ctx), fig_bof_summary(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("6.2 Classification and retrieval")
    tab1, tab2 = st.tabs(["Retrieval", "Classification"])
    voc = vocabulary(tuple(DB_KEYS), k_words=200)
    with tab1:
        c1, c2 = st.columns(2)
        qkey = c1.selectbox("query image", voc["keys"],
                            index=voc["keys"].index("notre_dame_a"))
        kind = c2.radio("change applied to the query",
                        ["crop", "rotate", "scale", "brighten", "none"], horizontal=True)
        qrgb = variant(load_image(qkey, max_side=360), kind)
        q = describe_query(cv2.cvtColor(qrgb, cv2.COLOR_RGB2GRAY), voc)
        order, sim = rank_database(voc, q["tfidf"])
        rank = int(np.flatnonzero(order == voc["keys"].index(qkey))[0]) + 1
        st.image(qrgb, width=220, caption=f"query: {qkey} ({kind})")
        st.write(f"the original is returned at rank **{rank}** of {len(voc['keys'])}")
        top = order[:6]
        st.image([load_image(voc["keys"][i], max_side=240) for i in top], width=160,
                 caption=[f"{j + 1}. {voc['keys'][i]}  ({sim[i]:.3f})"
                          for j, i in enumerate(top)])
        st.caption("Ranking is by cosine similarity of tf-idf weighted term vectors. "
                   "Rotation is the hardest of these changes, because dense sampling "
                   "positions no longer land on the same scene points.")
    with tab2:
        k_words = st.select_slider("vocabulary size", [40, 80, 120, 200], value=120)
        _, X, y, scores = scene_classification(k_words)
        for name, (acc, hit, n) in scores.items():
            st.write(f"SVM, {name}: **{100 * acc:.1f}%** ({hit}/{n}) "
                     "leave-one-out accuracy")
        st.caption("Two classes, animals against built scenes, from the credited images "
                   "used in these lessons. The dataset is tiny, so treat the numbers as a "
                   "demonstration of the pipeline rather than a benchmark.")
