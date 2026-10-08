import numpy as np
import pytest

from underwater_vision.evaluation.uncertainty import uncertainty_metrics
from underwater_vision.geometry.reference import project
from underwater_vision.pointcloud.learned_confidence import holdout_observation


def scene():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)
    poses = [np.eye(4) for _ in range(4)]
    for i, pose in enumerate(poses):
        pose[0, 3] = i * 0.3
    point = np.array([0.2, 0.1, 4.0])
    xy = np.array([project(point, pose, K) for pose in poses])
    return K, poses, point, xy


def test_holdout_observation_never_enters_predictor_features():
    K, poses, _, xy = scene()
    nodes = [(str(i), 0) for i in range(4)]
    semantic = np.eye(4, dtype=np.float32)
    quality = np.full(4, 0.8)
    x, error, _, point, held = holdout_observation(nodes, poses, xy, semantic, quality, K)
    xy[held] += [8, 0]
    semantic[held] = 42
    quality[held] = 0.01
    other, changed_error, _, changed_point, _ = holdout_observation(
        nodes, poses, xy, semantic, quality, K
    )
    np.testing.assert_allclose(other, x)
    np.testing.assert_allclose(changed_point, point)
    assert error < 1e-8 and changed_error > 7


def test_ause_oracle_and_ties_do_not_depend_on_input_order():
    error = np.arange(1, 51, dtype=float)
    assert uncertainty_metrics(error, error)["ause"] < 1e-12
    original = uncertainty_metrics(error, np.ones(50))
    shuffled = uncertainty_metrics(error[::-1], np.ones(50))
    assert original["ause"] == pytest.approx(shuffled["ause"])
    np.testing.assert_allclose(original["curve"]["retained_error_normalized"], 1)
    assert uncertainty_metrics(error, -error)["ause"] > original["ause"]
