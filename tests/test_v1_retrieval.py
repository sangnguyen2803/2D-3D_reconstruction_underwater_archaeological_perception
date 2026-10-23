import numpy as np
import pytest

from underwater_vision.retrieval.pairs import retrieve_pairs


def test_retrieval_never_contains_self_pairs_or_duplicates():
    descriptors = np.random.default_rng(2).normal(size=(20, 8))
    pairs, neighbors = retrieve_pairs(descriptors, 3)
    assert all(a < b for a, b in pairs)
    assert len(set(pairs)) == len(pairs)
    assert all(i not in row for i, row in enumerate(neighbors))


@pytest.mark.parametrize("setting", ["top_k", "sequential_window"])
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_retrieval_rejects_invalid_budgets(setting, value):
    with pytest.raises(ValueError, match=f"{setting} must be a nonnegative integer"):
        retrieve_pairs(np.eye(3), **{setting: value})


@pytest.mark.parametrize("count", [0, 1])
def test_retrieval_supports_empty_and_single_image_collections(count):
    pairs, neighbors = retrieve_pairs(np.ones((count, 8)))
    assert pairs == []
    assert neighbors.shape == (count, 0)


def test_zero_retrieval_budget_preserves_sequential_pairs():
    pairs, neighbors = retrieve_pairs(np.eye(4), top_k=0, sequential_window=1)
    assert pairs == [(0, 1), (1, 2), (2, 3)]
    assert neighbors.shape == (4, 0)


def test_retrieval_is_invariant_to_extreme_descriptor_scales():
    descriptors = np.random.default_rng(7).normal(size=(5, 8))
    expected_pairs, expected_neighbors = retrieve_pairs(descriptors, top_k=2)
    scales = np.array([1e-300, 1e-100, 1, 1e100, 1e300])[:, None]
    with np.errstate(all="raise"):
        pairs, neighbors = retrieve_pairs(descriptors * scales, top_k=2)
    assert pairs == expected_pairs
    np.testing.assert_array_equal(neighbors, expected_neighbors)


@pytest.mark.parametrize(
    "descriptors",
    [np.zeros((2, 3)), np.ones((2, 0)), np.ones(3), [[np.nan]], [[np.inf]]],
)
def test_retrieval_rejects_invalid_descriptors(descriptors):
    with pytest.raises(ValueError, match="finite nonzero global descriptors"):
        retrieve_pairs(descriptors)
