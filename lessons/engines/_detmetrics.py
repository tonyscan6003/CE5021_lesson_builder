"""Object-detection evaluation, as lesson 3.1 defines it, in NumPy.

    iou(a, b)                IoU matrix between two box sets (x1, y1, x2, y2)
    match(...)               greedy matching of ranked detections to ground truth
    pr_curve(tp, n_gt)       cumulative precision and recall down a ranked list
    interpolate(p)           p_interp(r) = max precision at any recall >= r
    average_precision(...)   area under the interpolated curve (all points, VOC)
    nms(boxes, scores, t)    non-maximum suppression

Used by 3.1 (metrics), 3.2 (proposal recall) and 3.4 (NMS).
"""
from __future__ import annotations

import numpy as np


def iou(a, b) -> np.ndarray:
    a, b = np.atleast_2d(np.asarray(a, float)), np.atleast_2d(np.asarray(b, float))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def match(boxes, scores, gt, t_conf=0.0, t_iou=0.5):
    """Classify detections as TP or FP, highest confidence first.

    A detection below the confidence threshold T1 is ignored. Otherwise it is a
    true positive when its IoU with a not-yet-matched ground-truth box exceeds
    T2, and a false positive if not (including a second detection of an
    object that is already matched). Unmatched ground truth = false negatives.

    Returns (order, is_tp, best_iou, gt_matched): `order` lists the kept
    detections by decreasing score; the other arrays follow that order, except
    gt_matched, which is one bool per ground-truth box.
    """
    scores = np.asarray(scores)
    order = [i for i in np.argsort(-scores, kind="stable") if scores[i] > t_conf]
    taken = np.zeros(len(gt), bool)
    M = iou(boxes, gt) if len(gt) and len(boxes) else np.zeros((len(boxes), len(gt)))
    is_tp, best = [], []
    for i in order:
        cand = np.where(~taken, M[i], -1.0) if len(gt) else np.array([-1.0])
        j = int(np.argmax(cand))
        best.append(float(M[i].max()) if len(gt) else 0.0)
        if len(gt) and cand[j] > t_iou:
            taken[j] = True
            is_tp.append(True)
        else:
            is_tp.append(False)
    return np.array(order, int), np.array(is_tp, bool), np.array(best), taken


def pr_curve(is_tp, n_gt):
    """Precision and recall after each detection of a ranked list."""
    tp = np.cumsum(is_tp)
    fp = np.cumsum(~np.asarray(is_tp))
    return tp / np.maximum(tp + fp, 1), tp / n_gt


def interpolate(precision):
    """p_interp(r) = max p(r') for r' >= r: a running maximum from the right."""
    return np.maximum.accumulate(np.asarray(precision)[::-1])[::-1]


def average_precision(precision, recall):
    """Area under the interpolated PR curve, summed over the recall steps:
    AP = sum_i (r_{i+1} - r_i) p_interp(r_{i+1}), starting from r_0 = 0."""
    r = np.concatenate([[0.0], recall])
    p = interpolate(precision)
    return float(np.sum((r[1:] - r[:-1]) * p))


def nms(boxes, scores, t_iou=0.5):
    """Keep the highest-scoring box, drop every box overlapping it by more than
    t_iou, and repeat. Returns kept indices and, for each step, what it did."""
    order = list(np.argsort(-np.asarray(scores), kind="stable"))
    keep, steps = [], []
    while order:
        i = order.pop(0)
        keep.append(i)
        ov = iou(boxes[[i]], boxes[order])[0] if order else np.array([])
        drop = [o for o, v in zip(order, ov) if v > t_iou]
        steps.append((i, drop))
        order = [o for o in order if o not in drop]
    return keep, steps
