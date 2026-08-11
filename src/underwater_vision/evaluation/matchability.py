"""Ignore-aware ranking, retention and paired image bootstrap."""

import numpy as np
from sklearn.metrics import average_precision_score

from underwater_vision.evaluation.uncertainty import binary_metrics


def metrics(labels, probability):
    result = binary_metrics(labels, probability)
    result.pop("threshold_curve", None)
    y, p = np.asarray(labels), np.asarray(probability)
    keep = y >= 0
    y, p = y[keep], p[keep]
    fractions = np.linspace(0.05, 1, 20)
    retained = np.maximum(1, np.ceil(fractions * len(y)).astype(int))
    if len(y):
        order = np.argsort(-p, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
        sizes = np.diff(np.r_[starts, len(y)])
        tied_positive = np.repeat(np.add.reduceat(y[order], starts) / sizes, sizes)
        kept_positive = np.cumsum(tied_positive)[retained - 1]
    else:
        kept_positive = np.zeros(len(fractions))
    result["retention"] = {
        "keypoint_fraction": fractions.tolist(),
        "positive_recall": (kept_positive / max(1, y.sum())).tolist(),
        "scope": "labeled keypoints only; ties averaged",
    }
    return result


def paired_ap_bootstrap(rows, first, second, repeats=500, seed=42):
    """Rows are independent image/pair clusters; all their keypoints stay paired.

    Sort once, then compute exact weighted AP for each cluster resample.
    Equal scores share one threshold; score quantization is never used.
    """
    if len(rows) < 2:
        return {"status": "insufficient_clusters", "clusters": len(rows)}
    ys, clusters, scores = [], [], {k: [] for k in [first, second]}
    for cluster, r in enumerate(rows):
        valid = np.asarray(r["labels"]) >= 0
        y = np.asarray(r["labels"])[valid]
        ys.append(y)
        clusters.append(np.full(len(y), cluster, np.int32))
        for key in [first, second]:
            p = np.asarray(r[key])[valid]
            scores[key].append(p)
    y = np.concatenate(ys)
    if len(np.unique(y)) < 2:
        return {"status": "insufficient_classes", "clusters": len(rows)}
    estimate = average_precision_score(y, np.concatenate(scores[first])) - average_precision_score(
        y, np.concatenate(scores[second])
    )
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(rows), np.full(len(rows), 1 / len(rows)), size=repeats)
    ap = {}
    cluster_ids = np.concatenate(clusters)
    for key in [first, second]:
        p = np.concatenate(scores[key])
        order = np.argsort(-p, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
        sorted_y, sorted_cluster = y[order], cluster_ids[order]
        estimates = []
        for weight in weights:
            sample_weight = weight[sorted_cluster]
            tp = np.add.reduceat(sample_weight * sorted_y, starts)
            total = np.add.reduceat(sample_weight, starts)
            precision = np.cumsum(tp) / np.maximum(np.cumsum(total), 1)
            estimates.append(float((precision * tp).sum() / max(tp.sum(), 1)))
        ap[key] = np.asarray(estimates)
    difference = ap[first] - ap[second]
    return {
        "status": "ok",
        "estimate": float(estimate),
        "ci95": np.quantile(difference, [0.025, 0.975]).tolist(),
        "clusters": len(rows),
        "bootstrap_repeats": repeats,
        "ci_method": "exact weighted AP, paired image/pair percentile bootstrap, tied scores pooled",
        "bootstrap_differences": difference.tolist(),
    }


def binned_analysis(labels, probability, statistics, xy, shape, extra=None):
    """Bins fitted on evaluation covariates only; never used to fit predictions."""
    h, w = shape
    result = {}
    variables = {
        "local_contrast": statistics[:, 2],
        "scale": statistics[:, 1],
        "frame_radius": np.sqrt(((xy[:, 0] - w / 2) / w) ** 2 + ((xy[:, 1] - h / 2) / h) ** 2),
    }
    variables.update(extra or {})
    for name, values in variables.items():
        edges = np.quantile(values, [0.25, 0.5, 0.75])
        groups = np.digitize(values, edges)
        result[name] = {
            "edges": edges.tolist(),
            "bins": [
                metrics(np.asarray(labels)[groups == b], np.asarray(probability)[groups == b])
                for b in range(4)
            ],
        }
    return result
