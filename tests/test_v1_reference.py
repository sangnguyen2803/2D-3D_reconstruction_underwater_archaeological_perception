import numpy as np
import pytest

from underwater_vision.data.spatial import assign_spatial, spatial_split
from underwater_vision.geometry.reference import (
    fundamental_from_poses,
    project,
    reference_labels,
    triangulate_observations,
)


def scene():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)
    poses = [np.eye(4) for _ in range(4)]
    for i, pose in enumerate(poses):
        pose[0, 3] = i * 0.3
    point = np.array([0.2, 0.1, 4.0])
    xy = np.array([project(point, pose, K) for pose in poses])
    return K, poses, point, xy


def test_reference_geometry_without_ransac():
    K, poses, point, xy = scene()
    F = fundamental_from_poses(poses[0], poses[1], K)
    assert abs(np.r_[xy[1], 1] @ F @ np.r_[xy[0], 1]) < 1e-9
    np.testing.assert_allclose(triangulate_observations(poses, xy, K), point, atol=1e-9)
    with pytest.raises(ValueError, match="baseline"):
        fundamental_from_poses(poses[0], poses[0], K)


def test_reference_ambiguous_band_is_excluded():
    K, poses, _, xy = scene()
    a = np.repeat(xy[:1], 3, axis=0)
    b = np.repeat(xy[1:2], 3, axis=0)
    b[:, 1] += [0, 4, 8]
    labels, _, _ = reference_labels(a, b, poses[0], poses[1], K, np.zeros(8))
    np.testing.assert_array_equal(labels, [1, -1, 0])


def test_spatial_split_retains_test_anchors_and_exclusion_gap():
    names = [str(i) for i in range(100)]
    centers = np.c_[np.arange(100), np.sin(np.arange(100)), np.zeros(100)]
    split = spatial_split(names, centers, names[10:20], gap_fraction=0.02)
    groups = split["groups"]
    assert set(names[10:20]).issubset(groups["test"])
    assert not set(groups["train"]) & set(groups["test"])
    assert min(split["minimum_center_distances"].values()) > split["specification"]["gap"]
    labels = assign_spatial(centers, split["specification"])
    assert set(labels) == {"train", "validation", "test", "excluded"}
