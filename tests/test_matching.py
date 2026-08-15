import numpy as np

from underwater_vision.matching.matcher import match_local, score_matches


def test_mutual_matches_unique():
    a = np.eye(5, dtype=np.float32)
    b = a[[2, 4, 0, 1, 3]]
    pairs, scores = match_local(a, b)
    assert len(pairs) == 5
    assert len(set(pairs[:, 1])) == 5
    np.testing.assert_array_equal(a[pairs[:, 0]], b[pairs[:, 1]])
    assert np.all(scores == 1)


def test_reliability_downweights_correspondence():
    pairs = np.array([[0, 0], [1, 1]])
    scores = score_matches(
        pairs, np.ones(2), reliability_a=np.array([1, 0.2]), reliability_b=np.array([1, 0.3])
    )
    np.testing.assert_allclose(scores, [1, 0.06])
