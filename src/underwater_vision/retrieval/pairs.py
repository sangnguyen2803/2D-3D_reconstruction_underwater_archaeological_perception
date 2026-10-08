"""Global DINO retrieval and explicitly approximate reference co-frustum overlap."""

import numpy as np


def retrieve_pairs(descriptors, top_k=12, sequential_window=0):
    d = np.asarray(descriptors, np.float32)
    if d.ndim != 2 or not np.isfinite(d).all() or (np.linalg.norm(d, axis=1) < 1e-9).any():
        raise ValueError("Need finite nonzero global descriptors")
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    scores = d @ d.T
    np.fill_diagonal(scores, -np.inf)
    k = min(top_k, len(d) - 1)
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
        precision.append(float(truth[i, row].mean()))
    return {
        "queries": len(neighbors),
        "top_k": neighbors.shape[1],
        "directed_positive_pairs": positives,
        "recall": hits / max(positives, 1),
        "sequential_recall": sequential_hits / max(positives, 1),
        "mean_precision": float(np.mean(precision)),
        "long_range_recall": long_hits / max(long_positives, 1),
        "long_range_positives": long_positives,
        "sequential_budget": 2 * window,
        "label_type": "reference_co_frustum_proxy_not_measured_overlap",
    }
