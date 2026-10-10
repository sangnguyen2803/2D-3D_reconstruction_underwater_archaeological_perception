"""Bidirectional top-1 labels: RootSIFT compatibility, not general visibility."""

import numpy as np

LABEL_SPEC = "rootsift_reference_matchability_v3"


def eligible_pair(errors, positive=1.5, minimum=15):
    return int(np.sum(np.asarray(errors) <= positive)) >= minimum


def aggregate_labels(errors, positive=1.5, negative=4.0, n_min=2):
    """Columns are eligible neighboring views; NaN means no eligible candidate.

    A single compatible candidate is positive even with fewer than n_min views.
    Ineligible pairs must be removed BEFORE invoking this function.
    """
    if not 0 < positive < negative or n_min < 1:
        raise ValueError("Invalid thresholds or candidate count")
    e = np.asarray(errors, float)
    if e.ndim != 2:
        raise ValueError("Expected keypoints by eligible views")
    present = np.isfinite(e)
    good = present & (e <= positive)
    bad = present & (e >= negative)
    label = np.full(len(e), -1, np.int8)
    label[(present.sum(1) >= n_min) & (bad.sum(1) == present.sum(1))] = 0
    label[good.any(1)] = 1
    labeled = good.sum(1) + bad.sum(1)
    soft = np.divide(good.sum(1), labeled, out=np.zeros(len(e)), where=labeled > 0)
    soft[label < 0] = -1
    return label, soft.astype(np.float32)


def stronger_labels(l1, heldout_error, threshold=2.0):
    """L2 preserves L1 negatives; uncertified L1 positives become IGNORE.

    Missing triangulation is not evidence for a negative. Evaluate L1/L2/soft
    models against one common L1 test set, rather than their own training labels.
    """
    label = np.asarray(l1, np.int8).copy()
    e = np.asarray(heldout_error)
    label[(label == 1) & (~np.isfinite(e) | (e > threshold))] = -1
    return label


def label_counts(labels):
    return {
        name: int(np.sum(labels == v))
        for name, v in [("positive", 1), ("negative", 0), ("ignore", -1)]
    }


def same_slab_pairs(pairs, groups):
    role = {n: g for g, names in groups.items() for n in names}
    return [
        (a, b)
        for a, b in pairs
        if role.get(a) == role.get(b) and role.get(a) not in {None, "excluded"}
    ]
