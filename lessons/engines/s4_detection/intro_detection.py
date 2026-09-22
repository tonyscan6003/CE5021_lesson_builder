"""Lesson 4.1 — Introduction to image classification and object detection.

Figures
    classification_vs_detection  whole-image labels versus bounding boxes (OpenCV pretrained HOG people detector)
    feature_engineering          image -> representation -> classifier, for training and for testing
    confusion_binary             the 2x2 confusion matrix for pedestrian detection with accuracy, precision, recall
    recognition_challenges       viewpoint, illumination, scale, occlusion, clutter, deformation, intra-class variation
    sliding_window               a fixed detector window slid over the image (interactive)
    pyramid_for_scale            the same window on an image pyramid covers objects of different sizes
    iou_boxes                    intersection over union of a predicted and a ground-truth box (interactive)
    detection_pr                 TP, FP and FN detections at an IoU threshold, precision and recall
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import plotly.graph_objects as go

from .._common import (BuildContext, Figure, C, save_mpl, plotly_html,
                       slider_frames_layout, to_data_uri, load_image, sync_matrix)

FIGURES = ["classification_vs_detection", "feature_engineering", "confusion_binary", "recognition_challenges",
           "sliding_window", "pyramid_for_scale", "iou_boxes", "detection_pr"]


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------

def people_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def detect_people(rgb, scale=1.05, stride=8, group=True):
    """OpenCV's pretrained HOG pedestrian detector.

    group=False passes groupThreshold=0, which disables OpenCV's internal
    merging and returns the raw, overlapping window hits.
    """
    rects, w = people_detector().detectMultiScale(
        rgb, 0.0, (stride, stride), (0, 0), scale, 2.0 if group else 0.0)
    return [tuple(int(v) for v in r) for r in rects], list(np.asarray(w).ravel()) if len(w) else []


def iou(a, b):
    """IoU of boxes (x, y, w, h)."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(ax1, bx1)); ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0, inter, union


def draw_boxes(rgb, boxes, colour=(213, 94, 0), thickness=2):
    out = rgb.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, thickness)
    return out


def box(ax, xy, w, h, text, fc="#e8f1f8", fontsize=9):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=C["grey"], lw=1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=14, color=C["ink"], lw=1.2))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_classification_vs_detection(ctx: BuildContext) -> Figure:
    tiles = [("labrador", "dog"), ("agathe_cat", "cat"), ("tanker", "ship")]
    street = load_image("arlit_street", max_side=1200)
    boxes, _ = detect_people(street)
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.15, 2.2])
    for j, (key, lab) in enumerate(tiles):
        ax = fig.add_subplot(gs[j]); im = load_image(key, max_side=400)
        h, w = im.shape[:2]; s = min(h, w); im = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        ax.imshow(im); ax.set_title(f'classification: "{lab}"', fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[4]); ax.imshow(draw_boxes(street, boxes)); ax.set_title(f"detection: {len(boxes)} people localised with bounding boxes", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "classification_vs_detection")
    return Figure("classification_vs_detection",
                  "Two recognition tasks. Image classification assigns one label to the "
                  "whole image; object detection also localises each instance of an "
                  "object with a bounding box, here using OpenCV's pretrained pedestrian "
                  "detector. Detection is slower than classification because of the "
                  "localisation step.", png=png)


def fig_feature_engineering(ctx: BuildContext) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 3.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 3.6); ax.axis("off")
    for row, (y, phase, last) in enumerate([(2.2, "training", "classifier learns its parameters"), (0.6, "testing", "classifier predicts the label")]):
        ax.text(0.1, y + 0.9, phase, fontsize=10, weight="bold", color=C["grey"])
        box(ax, (0.2, y), 2.0, 0.8, "images" + ("\n+ labels" if row == 0 else ""))
        arrow(ax, (2.25, y + 0.4), (3.0, y + 0.4))
        box(ax, (3.0, y), 2.8, 0.8, "representation\n(feature engineering)", fc="#fdf1e3")
        arrow(ax, (5.85, y + 0.4), (6.6, y + 0.4))
        box(ax, (6.6, y), 2.0, 0.8, "feature vectors")
        arrow(ax, (8.65, y + 0.4), (9.4, y + 0.4))
        box(ax, (9.4, y), 2.4, 0.8, last, fc="#e6f4ee")
    png = save_mpl(fig, ctx, "feature_engineering")
    return Figure("feature_engineering",
                  "Recognition with machine learning. Images are first converted into an "
                  "abstract representation, a feature vector, that makes the classes "
                  "easier to separate; designing that representation is called feature "
                  "engineering. The classifier is trained on feature vectors with labels, "
                  "and test images must be converted to the same representation before "
                  "they are classified.", png=png)


def fig_confusion_binary(ctx: BuildContext) -> Figure:
    tp, fp, fn, tn = 82, 9, 14, 195
    M = np.array([[tp, fn], [fp, tn]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1, 1.3]))
    ax = axes[0]
    sync_matrix(ax, M, cmap="Blues", fmt="{:.0f}", fontsize=13)
    labels = [["true positive\n(TP)", "false negative\n(FN)"], ["false positive\n(FP)", "true negative\n(TN)"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.32, labels[i][j], ha="center", va="center", fontsize=7.5, color=C["grey"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["predicted:\npedestrian", "predicted:\nno pedestrian"], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["actual:\npedestrian", "actual:\nno pedestrian"], fontsize=8)
    ax.tick_params(length=0); ax.set_title("confusion matrix, 300 test images")
    ax = axes[1]; ax.axis("off")
    ax.text(0, 0.92, f"accuracy = (TP + TN) / (TP + TN + FP + FN) = ({tp} + {tn}) / 300 = {(tp + tn) / 300:.2f}", fontsize=10)
    ax.text(0, 0.68, f"precision = TP / (TP + FP) = {tp} / {tp + fp} = {tp / (tp + fp):.2f}", fontsize=10)
    ax.text(0.05, 0.56, "of the images predicted to contain a pedestrian, how many did?", fontsize=8, color=C["grey"])
    ax.text(0, 0.36, f"recall = TP / (TP + FN) = {tp} / {tp + fn} = {tp / (tp + fn):.2f}", fontsize=10)
    ax.text(0.05, 0.24, "of the images that contain a pedestrian, how many were found?", fontsize=8, color=C["grey"])
    ax.text(0, 0.04, "A false negative (a missed pedestrian) is far more dangerous than a false positive.", fontsize=8, color=C["red"])
    png = save_mpl(fig, ctx, "confusion_binary")
    return Figure("confusion_binary",
                  "Classification metrics for a binary pedestrian classifier. The "
                  "confusion matrix counts the four outcomes of comparing predictions with "
                  "the ground truth. Accuracy is the fraction of correct predictions; "
                  "precision measures how trustworthy a positive prediction is (few false "
                  "positives); recall measures how few positives are missed (few false "
                  "negatives).", png=png)


def fig_recognition_challenges(ctx: BuildContext) -> Figure:
    macaw = load_image("macaw", max_side=400)
    dark = np.clip(macaw.astype(float) * 0.35, 0, 255).astype(np.uint8)
    panels = [("viewpoint", [("notre_dame_a", None), ("notre_dame_b", None)]),
              ("illumination", [(None, macaw), (None, dark)]),
              ("scale", [("crowd", None), ("friends", None)]),
              ("occlusion", [("kitten_hiding", None)]),
              ("background clutter", [("workbench", None)]),
              ("deformation", [("horses", None)]),
              ("intra-class variation", [("chairs", None)])]
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.6))
    axes = axes.ravel()
    k = 0
    for title, items in panels:
        for j, (key, im) in enumerate(items):
            ax = axes[k]; k += 1
            suffix = f" ({'ab'[j]})" if len(items) > 1 else ""
            ax.imshow(im if im is not None else load_image(key, max_side=400))
            ax.set_title(title + suffix, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "recognition_challenges")
    return Figure("recognition_challenges",
                  "Why recognition is hard: the same object can look very different with "
                  "viewpoint and illumination; objects appear at different scales, "
                  "partly occluded and against cluttered backgrounds; many objects are "
                  "deformable; and members of one class (chairs) vary enormously. A "
                  "recognition system fails when one of these exceeds what it was "
                  "trained for.", png=png)


def fig_sliding_window(ctx: BuildContext) -> Figure:
    rgb = load_image("arlit_street", max_side=640)
    h, w = rgb.shape[:2]
    win_w, win_h = 64, 128
    step = 32
    positions = [(x, y) for y in range(0, h - win_h + 1, step) for x in range(0, w - win_w + 1, step)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, idx in zip(axes, [5, len(positions) // 2 + 3, len(positions) - 7]):
        x, y = positions[idx]; out = rgb.copy()
        for px, py in positions:
            cv2.rectangle(out, (px, py), (px + win_w, py + win_h), (200, 200, 200), 1)
        cv2.rectangle(out, (x, y), (x + win_w, y + win_h), (213, 94, 0), 3)
        ax.imshow(out); ax.set_title(f"window {idx + 1} of {len(positions)}: classify this {win_w} × {win_h} patch"); ax.set_xticks([]); ax.set_yticks([])
    png = save_mpl(fig, ctx, "sliding_window")
    sel = positions[::4]
    # The background image is encoded once and the window is a separate trace that
    # each frame moves, rather than re-encoding the whole picture per frame.
    grid = rgb.copy()
    for px, py in positions:
        cv2.rectangle(grid, (px, py), (px + win_w, py + win_h), (200, 200, 200), 1)
    base = go.Image(source=to_data_uri(grid, "jpg", quality=72))

    def window_trace(x, y):
        return go.Scatter(x=[x, x + win_w, x + win_w, x, x],
                          y=[y, y, y + win_h, y + win_h, y],
                          mode="lines", line=dict(color=C["red"], width=3),
                          hoverinfo="skip", showlegend=False)

    frames = [go.Frame(name=str(i), data=[window_trace(x, y)], traces=[1],
                       layout=go.Layout(title=f"window position {i + 1} of {len(sel)}"))
              for i, (x, y) in enumerate(sel)]
    fig2 = go.Figure(data=[base, window_trace(*sel[0])], frames=frames)
    fig2.update_layout(**slider_frames_layout(list(range(len(sel))), "position "),
                       title=frames[0].layout.title.text)
    fig2.update_xaxes(visible=False, range=[0, w])
    fig2.update_yaxes(visible=False, range=[h, 0], scaleanchor="x")
    return Figure("sliding_window",
                  "Sliding-window detection. A binary classifier trained on patches of a "
                  "fixed size, here 64 × 128 pixels, is applied at every position of a "
                  "window that steps across the image. Each position is a separate "
                  "classification, so a large image requires hundreds or thousands of "
                  "them, and only objects that fit the window are found.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def fig_pyramid_for_scale(ctx: BuildContext) -> Figure:
    rgb = load_image("friends", max_side=720)
    h, w = rgb.shape[:2]
    boxes_all = []
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), gridspec_kw=dict(width_ratios=[8, 5.6, 4, 2.8]))
    cur = rgb.copy(); f = 1.0
    for j, ax in enumerate(axes):
        out = cur.copy()
        win_w, win_h = 64, 128
        cv2.rectangle(out, (int(out.shape[1] * 0.42), int(out.shape[0] * 0.30)), (int(out.shape[1] * 0.42) + win_w, int(out.shape[0] * 0.30) + win_h), (213, 94, 0), 2)
        ax.imshow(out); ax.set_title(f"level {j}: {cur.shape[1]} px wide, scale {f:.2f}\nwindow covers {win_w / f:.0f} × {win_h / f:.0f} original px", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        cur = cv2.resize(cv2.GaussianBlur(cur, (0, 0), 0.8), None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA); f *= 0.7
    png = save_mpl(fig, ctx, "pyramid_for_scale")
    return Figure("pyramid_for_scale",
                  "Scale invariance from an image pyramid. The detector window has a fixed "
                  "size, so on the full image it fits only small, distant people. "
                  "Blurring and shrinking the image by a constant factor at each level "
                  "makes the same window cover a larger part of the scene, so large, "
                  "nearby objects are found at the smaller levels. More levels give a "
                  "better chance of finding every object at the cost of more "
                  "classifications.", png=png)


def fig_iou_boxes(ctx: BuildContext) -> Figure:
    gt = (30, 25, 50, 40)
    preds = [(35, 30, 50, 40), (55, 40, 50, 40), (32, 24, 60, 50)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, p in zip(axes, preds):
        v, inter, union = iou(gt, p)
        ax.add_patch(plt.Rectangle(gt[:2], gt[2], gt[3], fill=False, ec=C["green"], lw=2.5, label="ground truth"))
        ax.add_patch(plt.Rectangle(p[:2], p[2], p[3], fill=False, ec=C["red"], lw=2.5, label="prediction"))
        ix, iy = max(gt[0], p[0]), max(gt[1], p[1]); iw = max(0, min(gt[0] + gt[2], p[0] + p[2]) - ix); ih = max(0, min(gt[1] + gt[3], p[1] + p[3]) - iy)
        ax.add_patch(plt.Rectangle((ix, iy), iw, ih, fc=C["sky"], alpha=0.5, ec="none"))
        ax.set_xlim(0, 130); ax.set_ylim(100, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"intersection {inter} / union {union} = IoU {v:.2f}" + ("  ✓ TP" if v >= 0.5 else "  ✗ FP"), fontsize=9)
    axes[0].legend(fontsize=7, loc="lower right")
    png = save_mpl(fig, ctx, "iou_boxes")
    shifts = list(range(0, 61, 5))
    frames = []
    for s in shifts:
        p = (30 + s, 25 + s // 3, 50, 40); v, inter, union = iou(gt, p)
        ix, iy = max(gt[0], p[0]), max(gt[1], p[1]); iw = max(0, min(gt[0] + gt[2], p[0] + p[2]) - ix); ih = max(0, min(gt[1] + gt[3], p[1] + p[3]) - iy)
        frames.append(go.Frame(name=str(s), data=[
            go.Scatter(x=[p[0], p[0] + p[2], p[0] + p[2], p[0], p[0]], y=[p[1], p[1], p[1] + p[3], p[1] + p[3], p[1]], mode="lines", line=dict(color=C["red"], width=3), name="prediction"),
            go.Scatter(x=[ix, ix + iw, ix + iw, ix, ix], y=[iy, iy, iy + ih, iy + ih, iy], fill="toself", fillcolor="rgba(86,180,233,0.5)", line=dict(width=0), name="intersection")],
            traces=[1, 2], layout=go.Layout(title=f"IoU = {inter} / {union} = {v:.2f}" + (" → true positive at a 0.5 threshold" if v >= 0.5 else " → false positive"))))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[gt[0], gt[0] + gt[2], gt[0] + gt[2], gt[0], gt[0]], y=[gt[1], gt[1], gt[1] + gt[3], gt[1] + gt[3], gt[1]], mode="lines", line=dict(color=C["green"], width=3), name="ground truth"))
    fig2.add_trace(frames[1].data[0]); fig2.add_trace(frames[1].data[1]); fig2.frames = frames
    fig2.update_layout(**slider_frames_layout(shifts, "shift = "), title=frames[1].layout.title.text, xaxis=dict(range=[0, 140], scaleanchor="y"), yaxis=dict(range=[100, 0]))
    return Figure("iou_boxes",
                  "Intersection over union measures how well a predicted box fits the "
                  "ground-truth box: the area of overlap divided by the area of the "
                  "union. A detection is usually counted as a true positive when its IoU "
                  "exceeds a threshold such as 0.5. Drag the slider to move the "
                  "prediction and watch the IoU fall.",
                  png=png, html=plotly_html(fig2, height=430), kind="plotly")


def fig_detection_pr(ctx: BuildContext) -> Figure:
    rgb = load_image("arlit_street", max_side=1200)
    gt, _ = detect_people(rgb)                          # three people, used as the ground truth
    H, W = rgb.shape[:2]
    # Predictions: two good boxes (slightly offset), one spurious box on empty ground.
    # The third ground-truth person receives no prediction, so it is a false negative.
    spur_w, spur_h = int(gt[0][2] * 0.8), int(gt[0][3] * 0.8)
    preds = [(gt[0][0] + 10, gt[0][1] + 12, gt[0][2], gt[0][3]),
             (gt[1][0] - 8, gt[1][1] + 6, int(gt[1][2] * 1.05), gt[1][3]),
             (int(W * 0.05), H - spur_h - 10, spur_w, spur_h)]   # spurious, kept inside the frame
    thr = 0.5
    tp_boxes, fp_boxes, matched = [], [], set()
    for p in preds:
        best = max(((iou(g, p)[0], i) for i, g in enumerate(gt)), default=(0.0, -1))
        if best[0] >= thr and best[1] not in matched:
            tp_boxes.append(p); matched.add(best[1])
        else:
            fp_boxes.append(p)
    fn_boxes = [g for i, g in enumerate(gt) if i not in matched]
    TP, FP, FN = len(tp_boxes), len(fp_boxes), len(fn_boxes)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axes[0]; ax.imshow(rgb)
    for b in gt:
        ax.add_patch(plt.Rectangle(b[:2], b[2], b[3], fill=False, ec=C["green"], lw=2.5, ls="--"))
    for b in tp_boxes:
        ax.add_patch(plt.Rectangle(b[:2], b[2], b[3], fill=False, ec=C["blue"], lw=2.5))
    for b in fp_boxes:
        ax.add_patch(plt.Rectangle(b[:2], b[2], b[3], fill=False, ec=C["red"], lw=2.5))
    for b in fn_boxes:
        ax.text(b[0] + b[2] / 2, b[1] - 8, "missed (FN)", color=C["green"], fontsize=8, ha="center", weight="bold")
    ax.plot([], [], color=C["green"], ls="--", lw=2.5, label="ground truth")
    ax.plot([], [], color=C["blue"], lw=2.5, label="prediction counted as TP")
    ax.plot([], [], color=C["red"], lw=2.5, label="prediction counted as FP")
    ax.legend(fontsize=8, loc="upper left"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("matching predictions to ground truth by IoU")
    ax = axes[1]; ax.axis("off")
    ax.text(0, 0.92, f"{len(gt)} ground-truth people, {len(preds)} predictions, IoU threshold {thr}", fontsize=9)
    ax.text(0, 0.76, f"TP = {TP}    FP = {FP}    FN = {FN}", fontsize=13)
    ax.text(0, 0.56, f"precision = TP / (TP + FP) = {TP}/{TP + FP} = {TP / (TP + FP):.2f}", fontsize=11)
    ax.text(0, 0.42, f"recall = TP / (TP + FN) = {TP}/{TP + FN} = {TP / (TP + FN):.2f}", fontsize=11)
    ax.text(0, 0.18, "No true negatives are counted: the whole background\nwould be one, so accuracy is not used for detection.", fontsize=8, color=C["grey"], va="top")
    png = save_mpl(fig, ctx, "detection_pr")
    return Figure("detection_pr",
                  "Precision and recall for a detector. Every prediction is matched to a "
                  "ground-truth box by IoU: a good overlap counts as a true positive "
                  "(blue), a box on empty ground as a false positive (red), and a person "
                  "with no matching prediction as a false negative. True negatives are "
                  "not counted, because the entire background would be one, so accuracy "
                  "is meaningless for detection and precision and recall are used instead.",
                  png=png)


def build(ctx: BuildContext) -> list[Figure]:
    return [fig_classification_vs_detection(ctx), fig_feature_engineering(ctx), fig_confusion_binary(ctx),
            fig_recognition_challenges(ctx), fig_sliding_window(ctx), fig_pyramid_for_scale(ctx),
            fig_iou_boxes(ctx), fig_detection_pr(ctx)]


def demo() -> None:
    import streamlit as st
    st.title("4.1 Introduction to object detection")
    tab1, tab2 = st.tabs(["IoU", "Pretrained pedestrian detector"])
    with tab1:
        c1, c2 = st.columns(2)
        dx = c1.slider("prediction offset x", -50, 50, 5); dy = c2.slider("prediction offset y", -50, 50, 5)
        sw = c1.slider("prediction width", 20, 100, 50); sh = c2.slider("prediction height", 20, 100, 40)
        gt = (30, 25, 50, 40); p = (30 + dx, 25 + dy, sw, sh)
        v, inter, union = iou(gt, p)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.add_patch(plt.Rectangle(gt[:2], gt[2], gt[3], fill=False, ec=C["green"], lw=2.5)); ax.add_patch(plt.Rectangle(p[:2], p[2], p[3], fill=False, ec=C["red"], lw=2.5))
        ax.set_xlim(-30, 160); ax.set_ylim(130, -30); ax.set_aspect("equal"); ax.set_title(f"IoU = {inter}/{union} = {v:.2f}"); st.pyplot(fig)
    with tab2:
        key = st.selectbox("image", ["arlit_street", "friends", "crowd", "jimmy"])
        rgb = load_image(key, max_side=900)
        scale = st.slider("pyramid scale factor", 1.01, 1.5, 1.05, 0.01)
        stride = st.select_slider("window stride", [4, 8, 16], value=8)
        boxes, _ = detect_people(rgb, scale, stride)
        st.image(draw_boxes(rgb, boxes), caption=f"{len(boxes)} detections", width="stretch")
