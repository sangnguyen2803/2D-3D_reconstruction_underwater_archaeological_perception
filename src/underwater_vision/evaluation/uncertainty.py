"""Tie-aware ranking and sparsification evaluated against held-out errors."""

import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def binary_metrics(labels, probability, threshold=0.5):
    y, p = np.asarray(labels), np.asarray(probability)
    known = y >= 0
    y, p = y[known], p[known]
    if not len(y) or len(np.unique(y)) < 2:
        return {"status": "insufficient_classes", "samples": len(y)}
    predicted = p >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    precision = tp / max(int(predicted.sum()), 1)
    recall = tp / max(int((y == 1).sum()), 1)
    pr, rc, thresholds = precision_recall_curve(y, p)
    bins = []
    ece = 0.0
    for left in np.linspace(0, 0.9, 10):
        ids = (p >= left) & (p < left + 0.1 if left < 0.9 else p <= 1)
        if ids.any():
            confidence, observed = float(p[ids].mean()), float(y[ids].mean())
            ece += ids.mean() * abs(confidence - observed)
            bins.append({"predicted": confidence, "observed": observed, "samples": int(ids.sum())})
    return {
        "status": "ok",
        "samples": len(y),
        "positive_fraction": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": float(ece),
        "threshold": float(threshold),
        "pr_curve": {
            "precision": pr[:: max(1, len(pr) // 200)].tolist(),
            "recall": rc[:: max(1, len(rc) // 200)].tolist(),
        },
        "calibration": bins,
        "threshold_curve": {
            "precision": pr[:-1].tolist(),
            "recall": rc[:-1].tolist(),
            "thresholds": thresholds.tolist(),
        },
    }


def choose_threshold(labels, probability):
    m = binary_metrics(labels, probability)
    if m["status"] != "ok":
        raise ValueError("Validation needs positive and negative labels")
    curve = m["threshold_curve"]
    p, r = np.array(curve["precision"]), np.array(curve["recall"])
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(curve["thresholds"][int(np.argmax(f1))])


def _retained_means(error, uncertainty, retained):
    # Average errors within tied uncertainty groups: do not let original order or
    # the oracle labels resolve ties, especially for a view-count-only baseline.
    order = np.argsort(uncertainty, kind="stable")
    u, e = uncertainty[order], error[order]
    starts = np.r_[0, np.flatnonzero(np.diff(u)) + 1]
    ends = np.r_[starts[1:], len(e)]
    tied = np.empty_like(e, dtype=float)
    for start, end in zip(starts, ends, strict=True):
        tied[start:end] = e[start:end].mean()
    sums = np.r_[0.0, np.cumsum(tied)]
    return sums[retained] / retained


def uncertainty_metrics(error, predicted_error, steps=100):
    e, u = np.asarray(error, float), np.asarray(predicted_error, float)
    if (
        e.shape != u.shape
        or len(e) < 3
        or not np.isfinite(e).all()
        or not np.isfinite(u).all()
        or (e < 0).any()
    ):
        raise ValueError("Need matching finite nonnegative error arrays with >=3 points")
    removed = np.linspace(0, 0.99, steps)
    retained = np.maximum(1, np.ceil(len(e) * (1 - removed)).astype(int))
    scale = max(e.mean(), 1e-12)
    actual = _retained_means(e, u, retained) / scale
    oracle = _retained_means(e, e, retained) / scale
    ause = float(trapezoid(actual - oracle, removed))
    rho = None if np.ptp(u) == 0 or np.ptp(e) == 0 else float(spearmanr(e, u).statistic)
    bins = []
    order = np.argsort(u, kind="stable")
    for ids in np.array_split(order, min(10, len(order))):
        bins.append(
            {
                "predicted_error_mean": float(u[ids].mean()),
                "observed_error_mean": float(e[ids].mean()),
                "samples": len(ids),
            }
        )
    return {
        "points": len(e),
        "ause": max(0.0, ause),
        "spearman_error": rho,
        "mean_error_px": float(e.mean()),
        "median_error_px": float(np.median(e)),
        "prediction_mae_px": float(np.mean(np.abs(u - e))),
        "normalization": "mean_error; fraction_removed_0_to_0.99; tie-averaged",
        "curve": {
            "fraction_removed": removed.tolist(),
            "retained_error_normalized": actual.tolist(),
            "oracle_error_normalized": oracle.tolist(),
        },
        "calibration": bins,
    }
