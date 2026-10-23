"""Global DINO retrieval and explicitly approximate reference co-frustum overlap."""

from numbers import Integral

import numpy as np


def retrieve_pairs(descriptors, top_k=12, sequential_window=0):
    for name, value in (("top_k", top_k), ("sequential_window", sequential_window)):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    d = np.asarray(descriptors, np.float64)
    if d.ndim != 2 or d.shape[1] == 0 or not np.isfinite(d).all():
        raise ValueError("Need finite nonzero global descriptors")
    # Scale before the L2 norm so very large or small descriptors retain their direction.
    magnitude = np.max(np.abs(d), axis=1, keepdims=True)
    if (magnitude == 0).any():
        raise ValueError("Need finite nonzero global descriptors")
    d = (d / magnitude).astype(np.float32)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    scores = d @ d.T
    np.fill_diagonal(scores, -np.inf)
    k = min(top_k, max(0, len(d) - 1))
    neighbors = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    pairs = {tuple(sorted((i, int(j)))) for i, row in enumerate(neighbors) for j in row}
    pairs.update(
        (i, j) for i in range(len(d)) for j in range(i + 1, min(len(d), i + sequential_window + 1))
    )
    return sorted(pairs), neighbors


def overlap_proxy(poses, K, width, height, depth, threshold=0.25):
    """Reference-camera co-frustum proxy at a triangulation-derived scene depth.

    Occlusion, relief, and actual image overlap are not observable from poses alone.
    Report this limitation instead of calling the matrix overlap ground truth.
    """
    poses = np.asarray(poses)
    inverse = np.linalg.inv(poses)
    yy, xx = np.meshgrid(
        np.linspace(height * 0.1, height * 0.9, 5), np.linspace(width * 0.1, width * 0.9, 5)
    )
    rays = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)] @ np.linalg.inv(K).T
    grid = rays * depth
    fraction = np.zeros((len(poses), len(poses)), np.float32)
    for i, pose in enumerate(poses):
        world = grid @ pose[:3, :3].T + pose[:3, 3]
        camera = np.einsum("nij,pj->npi", inverse[:, :3, :3], world) + inverse[:, None, :3, 3]
        z = camera[:, :, 2]
        pixels = camera @ K.T
        xy = pixels[:, :, :2] / np.maximum(z[:, :, None], 1e-9)
        inside = (
            (z > 0)
            & (xy[:, :, 0] >= 0)
            & (xy[:, :, 0] < width)
            & (xy[:, :, 1] >= 0)
            & (xy[:, :, 1] < height)
        )
        fraction[i] = inside.mean(1)
    result = np.minimum(fraction, fraction.T) >= threshold
    np.fill_diagonal(result, False)
    return result, fraction


def retrieval_metrics(neighbors, truth, window=6):
    """Evaluate distinct non-self neighbors; empty retrievals have zero precision."""
    if not isinstance(window, Integral) or isinstance(window, bool) or window < 0:
        raise ValueError("window must be a nonnegative integer")
    neighbors, truth = np.asarray(neighbors), np.asarray(truth, dtype=bool)
    if truth.ndim != 2 or truth.shape[0] != truth.shape[1]:
        raise ValueError("truth must be a square matrix")
    if (
        neighbors.ndim != 2
        or len(neighbors) != len(truth)
        or not np.issubdtype(neighbors.dtype, np.integer)
    ):
        raise ValueError("neighbors must be an integer matrix with one row per query")
    if ((neighbors < 0) | (neighbors >= len(truth))).any():
        raise ValueError("Neighbor indices must be within the truth matrix")
    if (neighbors == np.arange(len(neighbors))[:, None]).any() or (
        np.diff(np.sort(neighbors, axis=1), axis=1) == 0
    ).any():
        raise ValueError("Neighbors must be distinct and exclude self matches")
    if np.diag(truth).any():
        raise ValueError("truth must exclude self matches")
    hits, positives, sequential_hits, long_hits, long_positives = 0, 0, 0, 0, 0
    precision = []
    for i, row in enumerate(neighbors):
        sequential = np.arange(max(0, i - window), min(len(truth), i + window + 1))
        sequential = sequential[sequential != i]
        positive = np.flatnonzero(truth[i])
        long = positive[np.abs(positive - i) > window]
        hits += int(truth[i, row].sum())
        sequential_hits += int(truth[i, sequential].sum())
        positives += len(positive)
        long_hits += int(np.isin(row, long).sum())
        long_positives += len(long)
        precision.append(float(truth[i, row].mean()) if len(row) else 0.0)
    return {
        "queries": len(neighbors),
        "top_k": neighbors.shape[1],
        "directed_positive_pairs": positives,
        "recall": hits / max(positives, 1),
        "sequential_recall": sequential_hits / max(positives, 1),
        "mean_precision": float(np.mean(precision)) if precision else 0.0,
        "long_range_recall": long_hits / max(long_positives, 1),
        "long_range_positives": long_positives,
        "sequential_budget": 2 * window,
        "label_type": "reference_co_frustum_proxy_not_measured_overlap",
    }
