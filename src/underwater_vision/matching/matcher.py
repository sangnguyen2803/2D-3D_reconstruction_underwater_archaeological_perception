import cv2
import numpy as np


def match_local(a, b, ratio=0.8):
    if len(a) < 2 or len(b) < 2:
        return np.empty((0, 2), np.uint32), np.empty(0, np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(a, b, k=2)
    reverse = matcher.knnMatch(b, a, k=2)
    backward = {m.queryIdx: m.trainIdx for m, n in reverse if m.distance < ratio * n.distance}
    kept = [
        (m.queryIdx, m.trainIdx, m.distance)
        for m, n in forward
        if m.distance < ratio * n.distance and backward.get(m.trainIdx) == m.queryIdx
    ]
    pairs = np.array([[i, j] for i, j, _ in kept], dtype=np.uint32).reshape(-1, 2)
    scores = np.array([max(0, 1 - d / np.sqrt(2)) for _, _, d in kept], np.float32)
    return pairs, scores


def score_matches(
    pairs,
    local_scores,
    semantic_a=None,
    semantic_b=None,
    reliability_a=None,
    reliability_b=None,
    alpha=0.35,
):
    score = local_scores.copy()
    if semantic_a is not None:
        cosine = np.sum(semantic_a[pairs[:, 0]] * semantic_b[pairs[:, 1]], axis=1)
        score = (1 - alpha) * score + alpha * np.clip((cosine + 1) / 2, 0, 1)
    if reliability_a is not None:
        score *= reliability_a[pairs[:, 0]] * reliability_b[pairs[:, 1]]
    return np.clip(score, 0, 1)
