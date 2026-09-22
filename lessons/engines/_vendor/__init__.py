"""Question generators vendored from the question-bank repo (autoquiz).

Only `generate_params()` / `compute()` / `render()` are used, to produce the
worked numerical examples embedded in lessons. Files are copied verbatim
apart from the `..utils` import being redirected to `._utils`.

Re-vendor after an upstream change with:

    git clone https://github.com/tonyscan6003/autoquiz.git autoquiz
    for g in $(python -c 'from lessons.engines._vendor import TYPES; print(*TYPES)'); do
        sed 's/from \\.\\.utils import/from ._utils import/' \\
            autoquiz/question_tool/generators/$g.py > lessons/engines/_vendor/$g.py
    done
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

# Deep-learning generators: the worked examples for CE5021. None of them draws
# a figure, so `worked_example()` never needs a fig_dir for these.
DL_TYPES = [
    "single_neu", "softmax_act", "fc_param", "cross_ent",
    "conv_param1", "conv_param2", "mult_kernel_conv", "max_pool",
    "global_avg_pool", "rfield", "mat_conv",
    "bottleneck_conv", "inception_block", "depthwise_sep_conv",
    "bat_norm1", "bat_norm2", "dropout",
    "lr_schedule", "lr_schedule_cos",
    "dilated_conv", "transpose_conv1", "transpose_conv2",
]

# Classical machine-vision generators, kept from the CE6021 build. CE5021 uses
# the convolution-arithmetic and detection-metric ones; the rest are unused by
# the current lesson set but are cheap to keep and still pass their smoke test.
CV_TYPES = [
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

TYPES = DL_TYPES + CV_TYPES


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
