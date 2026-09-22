"""Shared Olivetti-faces helpers for the Section 5 engines.

The Olivetti set (40 identities x 10 images, 64 x 64 greyscale) is a
constrained facial-recognition dataset: centred frontal views under nearly
constant illumination. It is fetched by scikit-learn into lessons/data and
is credited under the key "olivetti".
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_olivetti_faces

from .._common import DATA_DIR, use_source

_CACHE: dict = {}


def olivetti():
    """(images (400, 64, 64) float 0..1, data (400, 4096), target (400,))."""
    use_source("olivetti")
    if "d" not in _CACHE:
        _CACHE["d"] = fetch_olivetti_faces(data_home=str(DATA_DIR / "_sklearn"), shuffle=False)
    d = _CACHE["d"]
    return d.images, d.data, d.target


def split_by_identity(n_train=7, seed=0):
    """Per-identity train/test split: the first n_train of each identity's 10 images."""
    _, X, y = olivetti()
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for ident in np.unique(y):
        idx = np.flatnonzero(y == ident)
        rng.shuffle(idx)
        tr += list(idx[:n_train]); te += list(idx[n_train:])
    return X[tr], y[tr], X[te], y[te]


def as_image(vec):
    return np.asarray(vec).reshape(64, 64)


def illumination_gradient(X, strength, side="left"):
    """Add a synthetic linear illumination ramp across each 64 x 64 face."""
    ramp = np.linspace(0, 1, 64)
    if side == "left":
        ramp = ramp[::-1]
    field = np.tile(ramp, (64, 1)).ravel()
    return np.clip(X + strength * (field - field.mean()), 0, 1)


def mean_face(X):
    return X.mean(axis=0)
