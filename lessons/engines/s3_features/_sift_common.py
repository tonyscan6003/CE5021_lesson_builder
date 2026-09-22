"""Shared helpers for the Section 3 engines: SIFT via OpenCV and gradient tools."""
from __future__ import annotations

import cv2
import numpy as np

from .._common import load_image


def sift_keypoints(gray, n=0):
    sift = cv2.SIFT_create(nfeatures=n)
    return sift.detectAndCompute(gray, None)


def ratio_matches(d1, d2, ratio=0.7):
    """Brute-force matching with Lowe's ratio test. Returns (good, all_pairs)."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in pairs if m.distance < ratio * n.distance]
    return good, pairs


def draw_matches(img1, kp1, img2, kp2, matches, max_draw=80, colour=(213, 94, 0)):
    return cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_draw], None,
                           matchColor=colour, singlePointColor=(120, 120, 120),
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def gradients(patch):
    """Central-difference gradient magnitude and angle (degrees, 0..360) of a float patch."""
    p = patch.astype(float)
    gx = np.zeros_like(p); gy = np.zeros_like(p)
    gx[:, 1:-1] = p[:, 2:] - p[:, :-2]
    gy[1:-1, :] = p[2:, :] - p[:-2, :]
    return np.hypot(gx, gy), np.degrees(np.arctan2(gy, gx)) % 360


def orientation_histogram(mag, ang, n_bins=8, weights=None):
    w = mag if weights is None else mag * weights
    return np.histogram(ang.ravel(), bins=n_bins, range=(0, 360), weights=w.ravel())[0]


def pair_images(key_a, key_b, max_side=640):
    a = load_image(key_a, gray=True, max_side=max_side)
    b = load_image(key_b, gray=True, max_side=max_side)
    return a, b
