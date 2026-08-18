"""Confidence-biased RANSAC sampling and weighted eight-point refinement.

Thresholds are in resized image pixels. This does not change COLMAP's bundle
adjustment objective; confidence influences the matches entering its backend.
"""

import cv2
import numpy as np


def sampson_error(F, xy1, xy2):
    a = np.c_[xy1, np.ones(len(xy1))]
    b = np.c_[xy2, np.ones(len(xy2))]
    fa, fb = a @ F.T, b @ F
    numerator = np.sum(b * fa, axis=1) ** 2
    denominator = fa[:, 0] ** 2 + fa[:, 1] ** 2 + fb[:, 0] ** 2 + fb[:, 1] ** 2
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def weighted_eight_point(a, b, weights):
    def normalize(x):
        center = np.average(x, axis=0, weights=weights)
        scale = np.sqrt(2) / max(
            np.average(np.linalg.norm(x - center, axis=1), weights=weights), 1e-9
        )
        T = np.array([[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1]])
        return (np.c_[x, np.ones(len(x))] @ T.T)[:, :2], T

    a, ta = normalize(a)
    b, tb = normalize(b)
    x, y, u, v = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    A = np.c_[u * x, u * y, u, v * x, v * y, v, x, y, np.ones(len(a))]
    _, _, vt = np.linalg.svd(A * np.sqrt(weights[:, None]), full_matrices=len(a) == 8)
    F = vt[-1].reshape(3, 3)
    U, s, V = np.linalg.svd(F)
    s[-1] = 0
    F = tb.T @ (U @ np.diag(s) @ V) @ ta
    return F / max(np.linalg.norm(F), 1e-12)


def verify(xy1, xy2, confidence, threshold=1.5, iterations=600, seed=42, weighted=False):
    n = len(xy1)
    if n < 8:
        return None, np.zeros(n, bool), np.full(n, np.inf)
    cv2.setRNGSeed(seed)
    F, _ = cv2.findFundamentalMat(xy1, xy2, cv2.FM_RANSAC, threshold, 0.999, iterations)
    if F is None or F.shape != (3, 3):
        return None, np.zeros(n, bool), np.full(n, np.inf)
    weights = np.clip(confidence, 1e-4, 1)
    if weighted:
        rng = np.random.default_rng(seed)
        probabilities = weights / weights.sum()
        score = np.sum(weights[sampson_error(F, xy1, xy2) < threshold])
        for _ in range(iterations):
            ids = rng.choice(n, 8, replace=False, p=probabilities)
            try:
                candidate = weighted_eight_point(xy1[ids], xy2[ids], weights[ids])
            except np.linalg.LinAlgError:
                continue
            candidate_score = np.sum(weights[sampson_error(candidate, xy1, xy2) < threshold])
            if candidate_score > score:
                F, score = candidate, candidate_score
        mask = sampson_error(F, xy1, xy2) < threshold
        if mask.sum() >= 8:
            candidate = weighted_eight_point(xy1[mask], xy2[mask], weights[mask])
            if np.sum(weights[sampson_error(candidate, xy1, xy2) < threshold]) >= score:
                F = candidate
    errors = sampson_error(F, xy1, xy2)
    return F, errors < threshold, errors
