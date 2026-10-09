"""Paired camera-pose evaluation on identical registration populations."""

import numpy as np

from underwater_vision.evaluation.metrics import similarity_alignment


def aligned_squared_error(centers, reference, sample=None):
    selected = np.arange(len(centers)) if sample is None else np.asarray(sample)
    scale, rotation, offset = similarity_alignment(centers[selected], reference[selected])
    aligned = scale * (centers @ rotation.T) + offset
    return np.square(aligned - reference).sum(axis=1)


def compare_poses(first, second, reference, repeats=500, cluster_size=4, seed=42):
    """After minus before ATE with re-alignment in each temporal block draw.

    Adjacent cameras are kept together. This is conditional on the reconstructed
    camera networks and published BA reference, not new-scene uncertainty.
    Missing cameras remain a separate reported outcome.
    """
    names = sorted(set(first) & set(second) & set(reference))
    if len(names) < 8:
        return {"status": "insufficient_common_cameras", "common_cameras": len(names)}
    a, b, target = [
        np.array([np.asarray(poses[n])[:3, 3] for n in names])
        for poses in [first, second, reference]
    ]
    ea, eb = aligned_squared_error(a, target), aligned_squared_error(b, target)
    before, after = np.sqrt(ea.mean()), np.sqrt(eb.mean())
    clusters = [
        np.arange(start, min(start + cluster_size, len(names)))
        for start in range(0, len(names), cluster_size)
    ]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        ids = np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
        try:
            sampled_a = aligned_squared_error(a, target, ids)
            sampled_b = aligned_squared_error(b, target, ids)
        except ValueError:
            continue
        draws.append(float(np.sqrt(sampled_b[ids].mean()) - np.sqrt(sampled_a[ids].mean())))
    return {
        "status": "ok",
        "names": names,
        "common_cameras": len(names),
        "registered_before": len(first),
        "registered_after": len(second),
        "ate_before": float(before),
        "ate_after": float(after),
        "ate_delta": float(after - before),
        "relative_ate_reduction": float(1 - after / before) if before > 0 else None,
        "ci95_delta": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap_deltas": draws,
        "bootstrap_cluster_size": cluster_size,
        "bootstrap_valid_draws": len(draws),
        "interpretation": "Sim3 re-fitted per paired temporal-camera-cluster bootstrap draw; published BA camera agreement, not independent surface or new-scene accuracy",
    }
