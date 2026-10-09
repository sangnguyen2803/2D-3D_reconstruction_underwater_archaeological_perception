"""Match ranking with an identical budget and a sparse-pair coverage guard."""

import numpy as np


def endpoint_ranking(pairs, first_scores, second_scores, filter_scores=None):
    """Geometric mean of endpoint scores, optionally fused with a match scorer."""
    pairs = np.asarray(pairs)
    a, b = np.asarray(first_scores), np.asarray(second_scores)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or a.ndim != 1 or b.ndim != 1:
        raise ValueError("Expected match pairs and aligned endpoint score vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0):
        raise ValueError("Endpoint scores must be finite and nonnegative")
    scores = np.sqrt(a[pairs[:, 0]] * b[pairs[:, 1]])
    if filter_scores is not None:
        f = np.asarray(filter_scores)
        if f.shape != scores.shape or not np.isfinite(f).all() or np.any(f < 0):
            raise ValueError("Expected aligned finite nonnegative filter scores")
        scores = scores * f
    return scores


def budget_indices(candidate_features, scores, retention=0.9, minimum=128):
    """Rank only mutual ratio-0.8 matches; never delete all of a weak pair.

    All scorers receive exactly the same candidate pool and retained count.
    Keeping sparse pairs intact prevents a learned score from severing edges
    solely because their image texture differs from the training distribution.
    """
    x, scores = np.asarray(candidate_features), np.asarray(scores)
    if x.ndim != 2 or x.shape[1] < 3 or scores.shape != (len(x),):
        raise ValueError("Expected aligned candidate features and one score per match")
    if not 0 < retention <= 1 or minimum < 0 or not np.isfinite(scores).all():
        raise ValueError("Invalid match budget or nonfinite scores")
    eligible = np.flatnonzero((x[:, 1] < 0.8) & (x[:, 2] > 0.5))
    count = min(len(eligible), max(minimum, int(np.ceil(retention * len(eligible)))))
    ranking = np.argsort(-scores[eligible], kind="stable")[:count]
    return np.sort(eligible[ranking])
