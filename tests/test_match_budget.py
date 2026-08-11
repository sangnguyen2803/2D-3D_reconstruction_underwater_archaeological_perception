import numpy as np
import pytest

from underwater_vision.matching.budget import budget_indices, endpoint_ranking


def test_sparse_pair_kept_without_nonmutual_or_ratio_failures():
    x = np.array([[0, 0.4, 1], [0, 0.9, 1], [0, 0.3, 0], [0, 0.7, 1]])
    np.testing.assert_array_equal(budget_indices(x, [0.1, 1, 1, 0.2]), [0, 3])


def test_different_scores_have_identical_budget_and_stable_ties():
    x = np.tile([0, 0.5, 1], (200, 1))
    a = budget_indices(x, np.arange(200), retention=0.9)
    b = budget_indices(x, np.ones(200), retention=0.9)
    assert len(a) == len(b) == 180
    np.testing.assert_array_equal(b, np.arange(180))
    np.testing.assert_array_equal(a, np.arange(20, 200))


def test_invalid_or_misaligned_score_rejected():
    with pytest.raises(ValueError):
        budget_indices(np.ones((2, 3)), [np.nan, 1])
    with pytest.raises(ValueError):
        budget_indices(np.ones((2, 3)), [1])


def test_endpoint_evidence_is_aligned_to_match_pairs():
    pairs = np.array([[1, 0], [0, 1]])
    np.testing.assert_allclose(endpoint_ranking(pairs, [0.25, 1], [0.25, 1]), [0.5, 0.5])
    np.testing.assert_allclose(
        endpoint_ranking(pairs, [0.25, 1], [0.25, 1], [0.2, 0.8]), [0.1, 0.4]
    )
    with pytest.raises(ValueError):
        endpoint_ranking(pairs, [0.25, -1], [0.25, 1])
