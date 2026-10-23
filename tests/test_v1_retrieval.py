import numpy as np
import pytest

from underwater_vision.retrieval.pairs import retrieval_metrics, retrieve_pairs


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


@pytest.mark.parametrize("count", [0, 1, 3])
def test_empty_retrieval_metrics_are_finite(count):
    truth = ~np.eye(count, dtype=bool)
    with np.errstate(all="raise"):
        metrics = retrieval_metrics(np.empty((count, 0), dtype=int), truth)
    assert metrics["queries"] == count
    assert metrics["top_k"] == 0
    assert metrics["mean_precision"] == 0
    assert metrics["recall"] == 0
    assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))


def test_retrieval_metrics_count_known_positive_pairs():
    truth = np.zeros((4, 4), dtype=bool)
    truth[0, 1] = truth[1, 0] = truth[2, 3] = truth[3, 2] = True
    metrics = retrieval_metrics([[1], [0], [3], [2]], truth, window=0)
    assert metrics["directed_positive_pairs"] == 4
    assert metrics["recall"] == metrics["mean_precision"] == 1
    assert metrics["long_range_recall"] == 1
    assert metrics["sequential_recall"] == 0


@pytest.mark.parametrize(
    "neighbors",
    [
        [[1], [0]],
        [[1.0], [0.0], [1.0]],
        [[-1], [0], [1]],
        [[3], [0], [1]],
        [[0], [0], [1]],
        [[1, 1], [0, 2], [0, 1]],
    ],
)
def test_retrieval_metrics_reject_invalid_neighbors(neighbors):
    with pytest.raises(ValueError):
        retrieval_metrics(neighbors, ~np.eye(3, dtype=bool))


@pytest.mark.parametrize("window", [-1, 1.5, True])
def test_retrieval_metrics_reject_invalid_windows(window):
    with pytest.raises(ValueError, match="window must be a nonnegative integer"):
        retrieval_metrics([[1], [0]], ~np.eye(2, dtype=bool), window=window)


@pytest.mark.parametrize("truth", [np.zeros((2, 3)), np.eye(2, dtype=bool)])
def test_retrieval_metrics_reject_invalid_truth(truth):
    with pytest.raises(ValueError, match="truth must"):
        retrieval_metrics([[1], [0]], truth)
