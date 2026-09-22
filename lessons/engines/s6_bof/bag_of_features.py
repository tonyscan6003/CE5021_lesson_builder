"""Lesson 6.1 — The bag of features representation.

Figures
    bag_of_words        three toy documents as unordered word-count vectors
    tf_idf              the same documents weighted by term frequency-inverse document frequency
    bof_algorithm       the three stages: build the vocabulary, assign terms, form the term vector
    keypoint_vs_dense   detected keypoints move with viewpoint; dense sampling does not
    descriptor_clustering  real SIFT descriptors clustered by k-means into visual words
    visual_vocabulary   the patches belonging to each codeword, and the frequency problem
    bof_histogram       three images as term histograms, with raw, tf-idf and binary weighting
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from .._common import (BuildContext, Figure, C, SERIES, save_mpl, plotly_html,
                       load_image, norm01)
from ._bof_common import (DOCUMENTS, bag_of_words, vocabulary, dense_keypoints,
                          sparse_descriptors, dense_descriptors, codeword_patches,
                          informative_words, DB_KEYS)

FIGURES = ["bag_of_words", "tf_idf", "bof_algorithm", "keypoint_vs_dense",
           "descriptor_clustering", "visual_vocabulary", "bof_histogram"]


def _box(ax, xy, w, h, text, fc="#e8f1f8", fs=8):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=fc, ec=C["grey"], lw=1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=13,
                                 color=C["ink"], lw=1.2))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_bag_of_words(ctx: BuildContext) -> Figure:
    names, vocab, counts, tf, n_i, idf, w = bag_of_words(DOCUMENTS)
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1])
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    ax.text(0, 1.0, "three short documents", fontsize=10, weight="bold", va="top")
    y = 0.86
    for i, (name, text) in enumerate(DOCUMENTS.items()):
        ax.text(0, y, name, fontsize=9, weight="bold", color=SERIES[i], va="top")
        ax.text(0, y - 0.09, '"' + text + '"', fontsize=8.5, va="top", wrap=True)
        y -= 0.26
    ax.text(0, 0.10, "The bag of words keeps only which words occur and how often.\n"
                     "Word order and sentence structure are thrown away, which is\n"
                     "what makes the representation orderless.",
            fontsize=8.5, va="top", color=C["grey"])
    ax = fig.add_subplot(gs[1])
    x = np.arange(len(vocab)); wid = 0.27
    for i, name in enumerate(names):
        ax.bar(x + (i - 1) * wid, counts[i], wid, color=SERIES[i], label=name.split(":")[0])
    ax.set_xticks(x); ax.set_xticklabels(vocab, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("raw word count"); ax.legend(fontsize=8, ncols=3)
    ax.set_title("raw counts: the common words dominate every document", fontsize=9)
    png = save_mpl(fig, ctx, "bag_of_words")
    return Figure("bag_of_words",
                  "The bag of words. Similar documents can be matched by comparing how "
                  "often each word occurs, with no regard for word order. The problem is "
                  'visible immediately: "the", "and", "left" and "watched" appear in every '
                  "document, so they dominate the raw counts while carrying almost no "
                  "information about what each document is about.", png=png)


def fig_tf_idf(ctx: BuildContext) -> Figure:
    names, vocab, counts, tf, n_i, idf, w = bag_of_words(DOCUMENTS)
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95])
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(vocab)); wid = 0.27
    for i, name in enumerate(names):
        ax.bar(x + (i - 1) * wid, tf[i], wid, color=SERIES[i], label=name.split(":")[0])
    ax.set_xticks(x); ax.set_xticklabels(vocab, rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("term frequency  $n_{id}/n_d$"); ax.legend(fontsize=7.5, ncols=3)
    ax.set_title("(a) term frequency alone", fontsize=9)
    ax = fig.add_subplot(gs[1])
    for i, name in enumerate(names):
        ax.bar(x + (i - 1) * wid, w[i], wid, color=SERIES[i], label=name.split(":")[0])
    ax.set_xticks(x); ax.set_xticklabels(vocab, rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("tf-idf weight  $t_i$"); ax.legend(fontsize=7.5, ncols=3)
    ax.set_title("(b) after the inverse document frequency", fontsize=9)
    ax = fig.add_subplot(gs[2]); ax.axis("off")
    ax.text(0, 1.0, "term frequency-inverse document frequency", fontsize=9.5,
            weight="bold", va="top")
    ax.text(0.03, 0.88, r"$t_i = \dfrac{n_{id}}{n_d}\,\log\dfrac{N}{n_i}$", fontsize=15, va="top")
    ax.text(0, 0.64, "$n_{id}$  occurrences of word $i$ in document $d$\n"
                     "$n_d$    total words in document $d$\n"
                     "$N$      documents in the database\n"
                     "$n_i$     documents that contain word $i$", fontsize=8.5, va="top")
    rows = [(vocab[j], int(n_i[j]), idf[j]) for j in np.argsort(-n_i)[:4]]
    rows += [(vocab[j], int(n_i[j]), idf[j]) for j in np.argsort(w.max(axis=0))[::-1][:3]]
    lines = "\n".join(f"{v:<9s}  in {int(d)} of {len(names)} docs    idf = {f:.2f}"
                      for v, d, f in rows)
    ax.text(0, 0.42, lines, fontsize=7.5, va="top", family="monospace")
    ax.text(0, 0.06, "A word that appears in every document has an idf of exactly\n"
                     "zero, so it drops out. The discriminative words are the ones\n"
                     "that survive the weighting.", fontsize=8.5, va="top", color=C["grey"])
    png = save_mpl(fig, ctx, "tf_idf")
    return Figure("tf_idf",
                  "The tf-idf weighting. The term frequency alone still favours the common "
                  "words (a). Multiplying by the inverse document frequency, the logarithm "
                  "of the number of documents divided by the number that contain the word, "
                  "drives words present in every document to zero and leaves the "
                  'discriminative ones standing: "harbour" for the first document, '
                  '"heron" and "shore" for the second, "road" and "van" for the third (b).',
                  png=png)


def fig_bof_algorithm(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")
    ax.text(0.15, 3.45, "1. build the visual vocabulary   (unsupervised, done once)",
            fontsize=9, weight="bold", color=C["grey"])
    x = 0.15
    for text in ["all images in a\nlarge dataset",
                 "detect or sample\nfeatures",
                 "describe each patch\n(SIFT)",
                 "cluster the descriptors\n($k$-means)",
                 "cluster centres =\nvisual words"]:
        _box(ax, (x, 2.15), 2.15, 1.1, text, fc="#e8f1f8", fs=8)
        if x < 10:
            _arrow(ax, (x + 2.22, 2.7), (x + 2.6, 2.7))
        x += 2.55
    ax.text(0.15, 1.85, "2. assign terms      3. generate the term vector   (for any new image)",
            fontsize=9, weight="bold", color=C["grey"])
    x = 0.15
    for text, fc in [("a new image", "#fdf1e3"),
                     ("detect or sample\nfeatures", "#fdf1e3"),
                     ("describe each patch", "#fdf1e3"),
                     ("assign each descriptor to\nits nearest visual word", "#e6f4ee"),
                     ("count the words →\nterm histogram", "#f3e8f5")]:
        _box(ax, (x, 0.55), 2.15, 1.1, text, fc=fc, fs=8)
        if x < 10:
            _arrow(ax, (x + 2.22, 1.1), (x + 2.6, 1.1))
        x += 2.55
    ax.text(0.15, 0.2, "The vocabulary is built by clustering, so no image labels are needed for it: "
                       "that step is unsupervised learning.", fontsize=8.5, color=C["grey"])
    png = save_mpl(fig, ctx, "bof_algorithm")
    return Figure("bof_algorithm",
                  "The bag of features algorithm. A visual vocabulary is built once, by "
                  "describing features from every image in a large dataset and clustering "
                  "the descriptors; the cluster centres are the visual words, or codewords. "
                  "Any image can then be converted to a term vector: describe its features, "
                  "assign each descriptor to the nearest codeword, and count. Building the "
                  "vocabulary is unsupervised, so the training images need no labels.",
                  png=png)


def fig_keypoint_vs_dense(ctx: BuildContext) -> Figure:
    a = load_image("notre_dame_a", max_side=420)
    b = load_image("notre_dame_b", max_side=420)
    ga, gb = (cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in (a, b))
    ka, _ = sparse_descriptors(ga, 140)
    kb, _ = sparse_descriptors(gb, 140)
    dense = dense_keypoints(ga.shape, step=26)
    fig = plt.figure(figsize=(13, 4.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.08)
    for j, (im, kps, name) in enumerate([(a, ka, "(a) view 1: detected keypoints"),
                                         (b, kb, "(b) view 2: detected keypoints")]):
        ax = fig.add_subplot(gs[j]); ax.imshow(im)
        xs = [k.pt[0] for k in kps]; ys = [k.pt[1] for k in kps]
        ax.plot(xs, ys, "o", ms=2.5, color=C["red"], alpha=0.7)
        ax.set_title(f"{name}\n{len(kps)} points", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[2]); ax.imshow(a)
    ax.plot([k.pt[0] for k in dense], [k.pt[1] for k in dense], "s", ms=2.5,
            color=C["green"], alpha=0.9)
    ax.set_title(f"(c) view 1: dense sampling\n{len(dense)} points", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[3]); ax.axis("off")
    ax.text(0, 1.0, "which features go into the vocabulary?", fontsize=9.5,
            weight="bold", va="top")
    ax.text(0, 0.84, "•  keypoints: blobs or corners, sparse and\n    repeatable on rigid "
                     "structure, but they move\n    when the viewpoint, pose or lighting "
                     "changes,\n    so the same object yields different words\n"
                     "•  random or dense sampling: many more\n    features, and the same "
                     "positions every time,\n    which makes a shared set of words more\n"
                     "    likely for an object seen differently",
            fontsize=8.5, va="top")
    ax.text(0, 0.30, f"Here the two views of the same building share the\n"
                     f"broad structure, but the detector fires in different\n"
                     f"places. The dense grid in (c) covers the whole frame\n"
                     f"at a fixed spacing of {26} pixels.", fontsize=8.5, va="top")
    ax.text(0, 0.06, "The patch around each point is then described, and SIFT\n"
                     "is the descriptor that clusters best.", fontsize=8.5, va="top",
            color=C["grey"])
    png = save_mpl(fig, ctx, "keypoint_vs_dense")
    return Figure("keypoint_vs_dense",
                  "Choosing the features. Keypoint detectors give sparse, well-localised "
                  "points, but they respond to different parts of the scene when the "
                  "viewpoint changes, so the same object produces different visual words "
                  "in different images. Random or dense sampling produces far more "
                  "features at positions that do not depend on the image content, which "
                  "raises the chance that an object seen under a new pose or light still "
                  "shares words with the version in the vocabulary.", png=png)


def fig_descriptor_clustering(ctx: BuildContext) -> Figure:
    keys = tuple(DB_KEYS[:12])
    voc = vocabulary(keys, k_words=40)
    D = np.vstack([voc["descs"][k] for k in keys])
    lab = np.concatenate([voc["words"][k] for k in keys])
    p = PCA(2, random_state=0).fit(D)
    Z = p.transform(D)
    Ck = p.transform(voc["km"].cluster_centers_)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Z), size=min(3000, len(Z)), replace=False)
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 0.9])
    ax = fig.add_subplot(gs[0])
    ax.plot(Z[sub, 0], Z[sub, 1], ".", ms=2, color=C["light"])
    ax.plot(Ck[:, 0], Ck[:, 1], "X", ms=9, color=C["red"], mec="k", mew=0.6)
    ax.set_title(f"(a) {len(D):,} SIFT descriptors from {len(keys)} images,\n"
                 f"{voc['k_words']} $k$-means centres, shown on their first two\n"
                 "principal components", fontsize=9)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    show = [int(w) for w in informative_words(voc, 4)]
    ax = fig.add_subplot(gs[1])
    ax.plot(Z[sub, 0], Z[sub, 1], ".", ms=2, color=C["light"])
    for i, wd in enumerate(show):
        sel = np.flatnonzero(lab == wd)
        ax.plot(Z[sel, 0], Z[sel, 1], ".", ms=3, color=SERIES[i], label=f"word {wd}")
        ax.plot(*Ck[wd], "X", ms=11, color=SERIES[i], mec="k", mew=0.8)
    ax.set_title("(b) four clusters: each is one visual word", fontsize=9)
    ax.set_xlabel("PC 1"); ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[2]); ax.axis("off")
    sub2 = gs[2].subgridspec(4, 5, wspace=0.08, hspace=0.25)
    for i, wd in enumerate(show):
        pats = codeword_patches(voc, wd, 5)
        for j, pat in enumerate(pats):
            a = fig.add_subplot(sub2[i, j])
            a.imshow(pat, cmap="gray"); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_color(SERIES[i]); s.set_linewidth(2); s.set_visible(True)
            if j == 2 and i == 0:
                a.set_title("(c) the patches in each cluster", fontsize=9)
    png = save_mpl(fig, ctx, "descriptor_clustering")
    return Figure("descriptor_clustering",
                  "Clustering descriptors into visual words. Every descriptor from every "
                  "image in the dataset is a point in a 128-dimensional space, drawn here "
                  "on its first two principal components (a). k-means finds cluster "
                  "centres, and each centre is a visual word (b). Reading the patches that "
                  "belong to one cluster back out of the images shows what the word "
                  "represents: an edge at a particular orientation, a corner, a texture (c). "
                  "Clustering is what limits the representation to a manageable number of "
                  "distinct features.", png=png)


def fig_visual_vocabulary(ctx: BuildContext) -> Figure:
    voc = vocabulary(tuple(DB_KEYS), k_words=200)
    words = [int(w) for w in informative_words(voc, 7)]
    n_img = len(voc["keys"])
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 0.8], hspace=0.35)
    sub = gs[:, 0].subgridspec(4, 7, wspace=0.12, hspace=0.12)
    for i, wd in enumerate(words):
        pats = codeword_patches(voc, wd, 4, per_image=1)
        for j, pat in enumerate(pats[:4]):
            a = fig.add_subplot(sub[j, i])
            a.imshow(pat, cmap="gray"); a.set_xticks([]); a.set_yticks([])
            if j == 0:
                a.set_title(f"word {wd}", fontsize=7.5)
    fig.text(0.01, 1.04, f"(a) seven words from a {voc['k_words']}-word visual vocabulary built "
                         f"from {n_img} images. Each column is one word; the four patches "
                         "below it come from four different images.", fontsize=9, va="top")
    ax = fig.add_subplot(gs[0, 1])
    frac = np.sort(voc["df"] / n_img)[::-1]
    ax.plot(np.arange(1, len(frac) + 1), 100 * frac, color=C["blue"], lw=2)
    ax.axhspan(25, 70, color=C["green"], alpha=0.18)
    ax.text(len(frac) * 0.55, 48, "intermediate frequency:\nthe useful words",
            fontsize=8, color=C["green"], ha="center")
    ax.set_xlabel("visual word, ordered by document frequency", fontsize=8.5)
    ax.set_ylabel("% of images containing it", fontsize=8.5)
    ax.set_title("(b) the frequency problem", fontsize=9)
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    f = voc["df"] / n_img
    hi = int((f >= 0.5).sum()); lo = int((f <= 0.1).sum())
    ax.text(0, 1.0, f"The most common word here occurs in {100 * f.max():.0f}% of the images; "
                    f"{hi} of\nthe {voc['k_words']} words occur in more than half of them and {lo} in "
                    "a tenth or\nfewer. The frequent group, mostly plain edges and flat\n"
                    "regions, discriminates almost nothing; the rare group is too\n"
                    "rare to match on. The words in between do the work.",
            fontsize=8.5, va="top")
    ax.text(0, 0.34, "The vocabulary comes from clustering alone, so it is produced by\n"
                     "unsupervised learning: no image has to be labelled to create it.\n"
                     "Nothing in it carries a semantic meaning, though. The system\n"
                     "does not know what any word depicts.",
            fontsize=8.5, va="top", color=C["grey"])
    png = save_mpl(fig, ctx, "visual_vocabulary")
    return Figure("visual_vocabulary",
                  "A visual vocabulary. Each codeword is a cluster centre, and the patches "
                  "nearest to it show what it stands for: edges and lines at many "
                  "orientations, stripes, and a few more complex textures (a). Word "
                  "frequency is a real problem (b): words that appear in nearly every "
                  "image, typically simple edges, dominate the histograms while "
                  "discriminating almost nothing, and very rare words are of little use "
                  "either. The words of intermediate frequency do the work.", png=png)


def fig_bof_histogram(ctx: BuildContext) -> Figure:
    voc = vocabulary(tuple(DB_KEYS), k_words=200)
    picks = ["zebra", "trevi", "macaw"]
    idx = [voc["keys"].index(k) for k in picks]
    fig = plt.figure(figsize=(13, 6.0))
    gs = fig.add_gridspec(3, 3, width_ratios=[0.55, 1.5, 1.0], hspace=0.45, wspace=0.18)
    for r, (k, i) in enumerate(zip(picks, idx)):
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(load_image(k, max_side=260)); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(k.replace("_", " "), fontsize=9)
        tf = voc["tf"][i]
        ax = fig.add_subplot(gs[r, 1])
        ax.bar(np.arange(len(tf)), tf, width=1.0, color=SERIES[r])
        ax.set_ylabel("term frequency", fontsize=8.5)
        ax.set_title(f"raw term histogram: {int((tf > 0).sum())} of {voc['k_words']} "
                     f"words present", fontsize=9)
        if r == 2:
            ax.set_xlabel("visual word")
        ax = fig.add_subplot(gs[r, 2])
        w = voc["tfidf"][i]
        ax.bar(np.arange(len(w)), w, width=1.0, color=SERIES[r])
        ax.set_ylabel("tf-idf weight", fontsize=8.5)
        ax.set_title("after tf-idf weighting and normalisation", fontsize=9)
        if r == 2:
            ax.set_xlabel("visual word")
    png = save_mpl(fig, ctx, "bof_histogram")

    labels = ["raw counts", "tf-idf", "binary"]
    tf = voc["tf"][idx[0]]
    variants = [tf, voc["tfidf"][idx[0]], (tf > 0).astype(float)]
    frames = [go.Frame(name=lab, data=[go.Bar(x=np.arange(len(v)), y=v,
                                              marker_color=C["blue"])],
                       layout=go.Layout(title=f"{picks[0]}: {lab}"))
              for lab, v in zip(labels, variants)]
    fig2 = go.Figure(data=frames[0].data, frames=frames)
    from .._common import slider_frames_layout
    fig2.update_layout(**slider_frames_layout(labels, "histogram: "),
                       title=frames[0].layout.title.text,
                       xaxis_title="visual word", yaxis_title="weight")
    return Figure("bof_histogram",
                  "Three images as term vectors over the same 200-word vocabulary. Most "
                  "words are absent from any one image, so the histogram is sparse, and "
                  "with a large vocabulary it is sparser still. The histogram can be stored "
                  "as raw counts, as tf-idf weights, which suppress the words common to "
                  "every image, or as a purely binary vector, which is often used for very "
                  "long sparse vectors because the presence of a word is more informative "
                  "than its exact count.",
                  png=png, html=plotly_html(fig2, height=420), kind="plotly")


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_bag_of_words(ctx), fig_tf_idf(ctx), fig_bof_algorithm(ctx),
            fig_keypoint_vs_dense(ctx), fig_descriptor_clustering(ctx),
            fig_visual_vocabulary(ctx), fig_bof_histogram(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("6.1 Bag of features")
    tab1, tab2 = st.tabs(["Term vectors", "Visual vocabulary"])
    voc = vocabulary(tuple(DB_KEYS), k_words=200)
    with tab1:
        key = st.selectbox("image", voc["keys"], index=voc["keys"].index("zebra"))
        weighting = st.radio("histogram weighting", ["raw counts", "tf-idf", "binary"],
                             horizontal=True)
        i = voc["keys"].index(key)
        tf = voc["tf"][i]
        v = {"raw counts": tf, "tf-idf": voc["tfidf"][i],
             "binary": (tf > 0).astype(float)}[weighting]
        c1, c2 = st.columns([1, 2])
        c1.image(load_image(key, max_side=320), caption=key)
        fig, ax = plt.subplots(figsize=(7, 3.0))
        ax.bar(np.arange(len(v)), v, width=1.0, color=C["blue"])
        ax.set_xlabel("visual word"); ax.set_ylabel(weighting)
        c2.pyplot(fig)
        st.write(f"words present: **{int((tf > 0).sum())}** of {voc['k_words']}  ·  "
                 f"descriptors: **{len(voc['descs'][key])}**")
        st.caption("Switch the weighting to see the common words collapse under tf-idf "
                   "and the vector become a presence flag under binary weighting.")
    with tab2:
        n_img = len(voc["keys"])
        frac = voc["df"] / n_img
        band = st.slider("show words present in this fraction of the images",
                         0.0, 1.0, (0.25, 0.7), 0.05)
        cand = np.flatnonzero((frac >= band[0]) & (frac <= band[1]))
        st.write(f"**{len(cand)}** of {voc['k_words']} words fall in that band")
        if len(cand):
            order = cand[np.argsort(-voc["df"][cand])][:8]
            cols = st.columns(len(order))
            for col, wd in zip(cols, order):
                pats = codeword_patches(voc, int(wd), 3)
                if pats:
                    col.image([norm01(p) for p in pats], width=54,
                              caption=[f"word {wd}"] * len(pats), clamp=True)
        st.caption("Words present in almost every image are plain edges and flat "
                   "regions; words present in very few are too rare to match on. The "
                   "middle band is where the discriminative words live.")
