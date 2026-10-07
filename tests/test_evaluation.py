import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from underwater_vision.evaluation.metrics import (
    dense_metrics,
    similarity_alignment,
    trajectory_metrics,
)


def test_alignment_scale_rotation_translation():
    xyz = np.random.default_rng(2).normal(size=(20, 3))
    R = Rotation.from_euler("xyz", [0.2, 0.1, -0.3]).as_matrix()
    target = 2.7 * (xyz @ R.T) + [4, 2, -1]
    s, r, t = similarity_alignment(xyz, target)
    np.testing.assert_allclose(s, 2.7)
    np.testing.assert_allclose(r, R, atol=1e-12)
    np.testing.assert_allclose(t, [4, 2, -1], atol=1e-12)


def test_trajectory_gauge_invariance():
    positions = np.random.default_rng(8).normal(size=(10, 3))
    R = Rotation.from_euler("y", 0.3).as_matrix()
    e, g = {}, {}
    for k, p in enumerate(positions):
        a, b = np.eye(4), np.eye(4)
        a[:3, 3] = p
        b[:3, :3] = R
        b[:3, 3] = 3 * R @ p + [1, 4, 2]
        e[str(k)], g[str(k)] = a, b
    m = trajectory_metrics(e, g)
    assert m["ate_rmse_reference_units"] < 1e-12
    assert m["rpe_rotation_median_degrees"] < 1e-10
    assert m["rpe_translation_rmse_reference_units"] < 1e-12


def test_dense_empty_and_identical():
    with pytest.raises(ValueError):
        dense_metrics([], [])
    points = np.random.default_rng(5).normal(size=(30, 3))
    assert dense_metrics(points, points)["fscore_0.01"] == 1
