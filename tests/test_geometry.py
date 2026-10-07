import numpy as np

from underwater_vision.geometry.verification import sampson_error, verify, weighted_eight_point


def scene(n=80):
    rng = np.random.default_rng(3)
    xyz = rng.uniform([-2, -1, 4], [2, 1, 8], (n, 3))
    a = xyz[:, :2] / xyz[:, 2:] * 500 + [320, 240]
    moved = xyz + [0.6, 0, 0]
    b = moved[:, :2] / moved[:, 2:] * 500 + [320, 240]
    return a, b


def test_normalized_eight_point():
    a, b = scene()
    F = weighted_eight_point(a, b, np.ones(len(a)))
    assert np.linalg.matrix_rank(F, tol=1e-8) == 2
    assert np.max(sampson_error(F, a, b)) < 1e-7


def test_weighted_ransac_rejects_outliers():
    a, b = scene()
    b[-20:] = np.random.default_rng(9).uniform(0, 500, (20, 2))
    confidence = np.r_[np.ones(60), np.full(20, 0.05)]
    _, mask, error = verify(a, b, confidence, weighted=True, iterations=100)
    assert mask[:60].sum() >= 58
    assert mask[-20:].sum() <= 2
    assert np.isfinite(error).all()


def test_too_few_matches():
    _, mask, errors = verify(np.zeros((4, 2)), np.zeros((4, 2)), np.ones(4))
    assert not mask.any()
    assert np.isinf(errors).all()
