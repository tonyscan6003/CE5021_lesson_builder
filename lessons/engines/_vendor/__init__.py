"""Question generators vendored from the CE6021 question-bank repo.

Only `generate_params()` / `compute()` / `render()` are used, to produce the
worked numerical examples embedded in lessons. Files are copied verbatim
apart from the `..utils` import being redirected to `._utils`.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

TYPES = [
    "conv1d", "conv2d",
    "conv_arith1", "conv_arith2", "conv_arith3", "conv_arith4",
    "conv_arith5", "conv_arith6",
    "bi_lin", "sep_conv", "med_filt",
    "morph_opr1", "morph_opr2", "morph_opr3", "morph_opr4",
    "grad_dir", "canny_nms", "hys_thresh", "hough", "harris",
    "otsu_thresh", "kmeans_iter",
    "sift_bin_lin", "sift_orient",
    "iou", "objdet_pre", "con_mat", "hausdorff", "int_img", "haar_feature",
    "hog_calc", "hog_bilin", "pca_project",
]


def get(type_name: str):
    if type_name not in TYPES:
        raise KeyError(f"unknown worked-example type {type_name!r}; known: {TYPES}")
    return importlib.import_module(f"{__name__}.{type_name}")


def worked_example(type_name: str, seed: int, fig_dir: Path | None = None,
                   fig_prefix: str = "figures", **params) -> str:
    """Return question + full solution markdown for one generator instance.

    Generators that draw a figure write it to `fig_dir` and the image link is
    rewritten to `<fig_prefix>/<file>` so it resolves from the lesson page.
    """
    mod = get(type_name)
    p = mod.generate_params(seed=seed, **params)
    c = mod.compute(p)
    kwargs = {}
    if fig_dir is not None and "out_dir" in mod.render.__code__.co_varnames:
        stem = f"ex_{type_name}_{seed}"
        kwargs = dict(out_dir=str(fig_dir), img_stem=stem)
    md = mod.render(p, c, mode="full", **kwargs)
    if kwargs:
        md = re.sub(r"\]\((ex_[^)]+\.png)\)", rf"]({fig_prefix}/\1)", md)
    return md
