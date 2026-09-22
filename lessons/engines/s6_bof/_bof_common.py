"""Shared bag-of-features machinery for Section 6.

Everything here works on the credited Commons images in `lessons/data/`, loaded
through `load_image` so that each figure records its attributions. Nothing from
the lab datasets is used.

The vocabulary is expensive enough to be worth caching between figures, so
`vocabulary()` memoises on its parameters.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .._common import load_image, use_source

# Two scene categories used for the classification figure, and the wider set
# used as the retrieval database. All are credited sources in data/sources.yaml.
ANIMALS = ["cat", "agathe_cat", "kitten_hiding", "lion", "labrador", "macaw", "owl",
           "butterfly", "monarch", "alpaca", "sheep", "horses", "cow_field",
           "dairy_cows", "zebra", "starfish"]
PLACES = ["notre_dame_a", "notre_dame_b", "trevi", "trevi_detail", "sacre_coeur",
          "pyramids", "sphinx", "liege_panorama", "office_facade", "brickwork",
          "arlit_street", "cycle_track", "chairs", "workbench", "minivan", "tanker"]
DB_KEYS = ANIMALS + PLACES + ["helicopter", "balloons", "rubiks_cube", "fruit"]

MAX_SIDE = 360
STEP = 10           # dense sampling stride, pixels
PATCH = 16          # descriptor support radius, pixels


def _sift():
    return cv2.SIFT_create()


def dense_keypoints(shape, step: int = STEP, size: int = PATCH):
    h, w = shape[:2]
    return [cv2.KeyPoint(float(x), float(y), float(size))
            for y in range(size, h - size, step)
            for x in range(size, w - size, step)]


def dense_descriptors(gray, step: int = STEP, size: int = PATCH):
    """SIFT descriptors on a regular grid. Returns (keypoints, descriptors)."""
    kps = dense_keypoints(gray.shape, step, size)
    kps, desc = _sift().compute(gray, kps)
    if desc is None:
        return [], np.zeros((0, 128), np.float32)
    return list(kps), desc


def sparse_descriptors(gray, n: int | None = None):
    """SIFT keypoints found by the detector, strongest first."""
    kps, desc = _sift().detectAndCompute(gray, None)
    if desc is None:
        return [], np.zeros((0, 128), np.float32)
    order = np.argsort([-k.response for k in kps])
    if n:
        order = order[:n]
    return [kps[i] for i in order], desc[order]


@lru_cache(maxsize=4)
def _image_descriptors(keys: tuple[str, ...], step: int, max_side: int):
    grays, kps, descs = {}, {}, {}
    for k in keys:
        g = load_image(k, gray=True, max_side=max_side)
        grays[k] = g
        kps[k], descs[k] = dense_descriptors(g, step)
    return grays, kps, descs


def vocabulary(keys: tuple[str, ...] = tuple(DB_KEYS), k_words: int = 200,
               step: int = STEP, max_side: int = MAX_SIDE):
    """A visual vocabulary over the given credited images.

    Thin wrapper over the cached builder that re-registers every source image
    with the attribution tracker, so a figure gets its credits whether or not
    the vocabulary was already in the cache.
    """
    for k in keys:
        use_source(k)
    return _vocabulary(tuple(keys), k_words, step, max_side)


@lru_cache(maxsize=4)
def _vocabulary(keys: tuple[str, ...], k_words: int, step: int, max_side: int):
    """Build a visual vocabulary by k-means over dense SIFT descriptors.

    Returns a dict with the k-means model, the per-image keypoints and
    descriptors, the term-frequency matrix, the inverse document frequencies
    and the tf-idf matrix (L2-normalised rows).
    """
    grays, kps, descs = _image_descriptors(keys, step, max_side)
    stack = np.vstack([descs[k] for k in keys])
    km = MiniBatchKMeans(n_clusters=k_words, random_state=0, n_init=5,
                         batch_size=1024).fit(stack)
    words = {k: km.predict(descs[k]) if len(descs[k]) else np.zeros(0, int) for k in keys}
    tf = np.stack([term_frequency(words[k], k_words) for k in keys])
    df = (tf > 0).sum(axis=0)
    idf = np.log(len(keys) / np.maximum(df, 1))
    V = tf * idf
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    return dict(km=km, keys=list(keys), grays=grays, kps=kps, descs=descs,
                words=words, tf=tf, df=df, idf=idf, tfidf=V, k_words=k_words,
                step=step, max_side=max_side)


def term_frequency(word_ids, k_words: int) -> np.ndarray:
    h = np.bincount(np.asarray(word_ids, int), minlength=k_words).astype(float)
    return h / max(h.sum(), 1.0)


def describe_query(gray, voc) -> dict:
    """Term frequency, tf-idf vector and word ids for an image outside the database."""
    kps, desc = dense_descriptors(gray, voc["step"])
    ids = voc["km"].predict(desc) if len(desc) else np.zeros(0, int)
    tf = term_frequency(ids, voc["k_words"])
    v = tf * voc["idf"]
    v /= max(np.linalg.norm(v), 1e-9)
    return dict(kps=kps, desc=desc, ids=ids, tf=tf, tfidf=v)


def rank_database(voc, v) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity of a tf-idf query vector against the database, best first."""
    s = voc["tfidf"] @ v
    order = np.argsort(-s)
    return order, s


def variant(rgb, kind: str):
    """A geometric or photometric change applied to make a query image."""
    h, w = rgb.shape[:2]
    if kind == "crop":
        return rgb[int(0.12 * h):int(0.88 * h), int(0.12 * w):int(0.88 * w)]
    if kind == "rotate":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 18, 1.0)
        return cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if kind == "scale":
        return cv2.resize(rgb, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    if kind == "brighten":
        return np.clip(rgb.astype(float) * 0.65 + 35, 0, 255).astype(np.uint8)
    return rgb


def codeword_patches(voc, word: int, n: int = 6, size: int = PATCH,
                     per_image: int = 1):
    """The image patches whose descriptors are closest to one cluster centre.

    `per_image` caps how many patches may come from the same image, so the
    result shows the same visual word as it occurs across different pictures
    rather than several overlapping crops of one region.
    """
    centre = voc["km"].cluster_centers_[word]
    found = []
    for key in voc["keys"]:
        ids = voc["words"][key]
        sel = np.flatnonzero(ids == word)
        if not len(sel):
            continue
        d = np.linalg.norm(voc["descs"][key][sel] - centre, axis=1)
        taken = 0
        for i in sel[np.argsort(d)]:
            if taken >= per_image:
                break
            x, y = voc["kps"][key][i].pt
            g = voc["grays"][key]
            x0, y0 = int(round(x)) - size, int(round(y)) - size
            patch = g[y0:y0 + 2 * size, x0:x0 + 2 * size]
            if patch.shape == (2 * size, 2 * size):
                found.append((float(np.linalg.norm(voc["descs"][key][i] - centre)), patch))
                taken += 1
    found.sort(key=lambda t: t[0])
    return [p for _, p in found[:n]]


def informative_words(voc, n: int = 12, lo: float = 0.25, hi: float = 0.7):
    """Words of intermediate document frequency: common enough to recur, rare
    enough to discriminate. Returns word indices ordered by document frequency."""
    n_img = len(voc["keys"])
    frac = voc["df"] / n_img
    cand = np.flatnonzero((frac >= lo) & (frac <= hi))
    if len(cand) < n:
        cand = np.argsort(-np.abs(frac - 0.45))[::-1]
    counts = (voc["tf"][:, cand] > 0).sum(axis=0)
    return cand[np.argsort(-counts)[:n]]


# ---------------------------------------------------------------------------
# Bag of words: the toy document example used in lesson 6.1
# ---------------------------------------------------------------------------

DOCUMENTS = {
    "A: a harbour report": ("the ship left the harbour and the crew "
                            "watched the harbour lights fade"),
    "B: a wildlife note": ("the heron left the shore and the heron "
                           "watched the shore for fish"),
    "C: a traffic notice": ("the van left the road and the road crew "
                            "watched the van and the road"),
}


def bag_of_words(docs: dict[str, str]):
    """Vocabulary, raw counts, term frequencies and tf-idf for a set of documents.

    tf-idf is (n_id / n_d) * log(N / n_i), with n_i the number of documents in
    the database that contain word i, so a word present in every document gets
    a weight of exactly zero.
    """
    tokens = {k: v.split() for k, v in docs.items()}
    vocab = sorted({w for t in tokens.values() for w in t})
    names = list(docs)
    counts = np.array([[tokens[d].count(w) for w in vocab] for d in names], float)
    nd = counts.sum(axis=1, keepdims=True)
    tf = counts / nd
    n_i = (counts > 0).sum(axis=0).astype(float)   # documents containing each word
    N = len(names)
    idf = np.log(N / np.maximum(n_i, 1))
    return names, vocab, counts, tf, n_i, idf, tf * idf
