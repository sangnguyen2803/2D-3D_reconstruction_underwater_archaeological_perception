"""Focused matchability core contracts, independent of pipeline integration."""

import numpy as np
import pytest

from underwater_vision.matchability.labels import (
    aggregate_labels,
    eligible_pair,
    same_slab_pairs,
    stronger_labels,
)


def test_positive_override_n_min_and_ambiguous_ignore():
    error = [[1, np.nan], [6, np.nan], [4, 5], [2, 7], [np.nan, np.nan], [1, 5]]
    labels, soft = aggregate_labels(error)
    assert labels.tolist() == [1, -1, 0, -1, -1, 1]
    assert soft.tolist() == [1, -1, 0, -1, -1, 0.5]


def test_ineligible_pairs_never_make_negatives():
    assert not eligible_pair([8] * 20)
    y, _ = aggregate_labels(np.empty((20, 0)))
    assert np.all(y == -1)


def test_eligibility_boundary_is_inclusive():
    assert eligible_pair([1.5] * 15)
    assert not eligible_pair([1.5] * 14)


def test_l2_unknown_track_is_not_negative():
    assert stronger_labels([1, 1, 1, 0, -1], [2, 3, np.nan, np.nan, 0]).tolist() == [
        1,
        -1,
        -1,
        0,
        -1,
    ]


def test_labels_reject_bad_thresholds():
    with pytest.raises(ValueError):
        aggregate_labels([[1]], positive=4, negative=1)


def test_no_split_straddling_pairs():
    groups = {"train": ["a", "b"], "validation": ["c"], "test": ["d", "e"], "excluded": ["x"]}
    assert same_slab_pairs([("a", "b"), ("a", "d"), ("d", "e"), ("x", "x")], groups) == [
        ("a", "b"),
        ("d", "e"),
    ]
