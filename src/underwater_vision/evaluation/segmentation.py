"""Hand-label segmentation metrics and paired, grouped bootstrap statistics."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from underwater_vision.semantic.schema import IGNORE, NAMES, validate_mask


def confusion(target, prediction):
    target, prediction = validate_mask(target), validate_mask(prediction)
    if target.shape != prediction.shape:
        raise ValueError("Prediction and hand mask sizes differ")
    valid = target != IGNORE
    # An abstained prediction on a labeled pixel counts as a miss (extra column).
    predicted = np.where(prediction[valid] == IGNORE, len(NAMES), prediction[valid])
    return np.bincount(
        target[valid].astype(int) * (len(NAMES) + 1) + predicted,
        minlength=len(NAMES) * (len(NAMES) + 1),
    ).reshape(len(NAMES), len(NAMES) + 1)


def segmentation_metrics(matrix):
    matrix = np.asarray(matrix)
    tp = np.diag(matrix[:, : len(NAMES)])
    union = matrix.sum(1) + matrix[:, : len(NAMES)].sum(0) - tp
    iou = np.divide(tp, union, out=np.full(len(NAMES), np.nan), where=union > 0)
    recall = np.divide(tp, matrix.sum(1), out=np.full(len(NAMES), np.nan), where=matrix.sum(1) > 0)
    return {
        "miou": float(np.nanmean(iou)) if np.isfinite(iou).any() else None,
        "pixel_accuracy": float(tp.sum() / matrix.sum()) if matrix.sum() else None,
        "per_class_iou": dict(
            zip(NAMES, [float(x) if np.isfinite(x) else None for x in iou], strict=True)
        ),
        "per_class_recall": dict(
            zip(NAMES, [float(x) if np.isfinite(x) else None for x in recall], strict=True)
        ),
        "labeled_pixels": int(matrix.sum()),
    }


def pixel_ece(logits, labels, temperature=1.0, bins=15):
    valid = labels != IGNORE
    if not valid.any():
        return None
    p = softmax(np.asarray(logits)[valid] / temperature, axis=-1)
    c, correct = p.max(-1), p.argmax(-1) == labels[valid]
    ece = 0.0
    for i in range(bins):
        member = (c >= i / bins) & (c < (i + 1) / bins if i < bins - 1 else c <= 1)
        if member.any():
            ece += member.mean() * abs(correct[member].mean() - c[member].mean())
    return float(ece)


def fit_temperature(logits, labels):
    valid = labels != IGNORE
    if not valid.any():
        raise ValueError("Temperature calibration requires hand-labeled validation pixels")
    x, y = np.asarray(logits)[valid], labels[valid].astype(int)

    def loss(log_t):
        p = softmax(x / np.exp(log_t), axis=-1)
        return -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()

    return float(
        np.exp(minimize_scalar(loss, bounds=(np.log(0.1), np.log(10)), method="bounded").x)
    )


def paired_bootstrap(groups, statistic, repeats=1000, seed=42):
    """Resample complete image/pair/track groups, never independent pixels."""
    estimate = float(statistic(groups)) if groups else np.nan
    if len(groups) < 2:
        return {
            "estimate": estimate if np.isfinite(estimate) else None,
            "ci95": None,
            "groups": len(groups),
        }
    rng = np.random.default_rng(seed)
    values = np.array(
        [
            statistic([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            for _ in range(repeats)
        ]
    )
    finite = values[np.isfinite(values)]
    return {
        "estimate": estimate if np.isfinite(estimate) else None,
        "ci95": np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None,
        "groups": len(groups),
        "repeats": repeats,
        "seed": seed,
    }
