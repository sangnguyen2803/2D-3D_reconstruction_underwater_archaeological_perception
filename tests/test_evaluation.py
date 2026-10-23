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


@pytest.mark.parametrize(
    "source,target",
    [
        (np.zeros(3), np.zeros(3)),
        (np.zeros((3, 2)), np.zeros((3, 2))),
        (np.zeros((3, 3)), np.zeros((4, 3))),
        (np.zeros((3, 3)), np.zeros((3, 3, 1))),
    ],
)
def test_alignment_rejects_mismatched_or_non_3d_centers(source, target):
    with pytest.raises(ValueError, match="matching shape"):
        similarity_alignment(source, target)


@pytest.mark.parametrize("side", ["source", "target"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_alignment_rejects_nonfinite_centers(side, value):
    clouds = {"source": np.eye(3), "target": np.eye(3)}
    clouds[side][0, 0] = value
    with pytest.raises(ValueError, match="centers must be finite"):
        similarity_alignment(**clouds)


@pytest.mark.parametrize("side", ["source", "target"])
@pytest.mark.parametrize("degenerate", [np.zeros((3, 3)), np.arange(9).reshape(3, 3)])
def test_alignment_rejects_degenerate_source_or_target(side, degenerate):
    clouds = {"source": np.eye(3), "target": np.eye(3)}
    clouds[side] = degenerate
    with pytest.raises(ValueError, match="non-collinear"):
        similarity_alignment(**clouds)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_alignment_rejects_insufficient_centers(count):
    with pytest.raises(ValueError, match="At least 3"):
        similarity_alignment(np.zeros((count, 3)), np.zeros((count, 3)))
