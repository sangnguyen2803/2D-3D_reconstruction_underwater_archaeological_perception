import numpy as np
import pytest
import torch

from underwater_vision.data.spatial import assign_spatial, spatial_split
from underwater_vision.evaluation.uncertainty import binary_metrics, uncertainty_metrics
from underwater_vision.geometry.reference import (
    fundamental_from_poses,
    project,
    reference_labels,
    triangulate_observations,
)
from underwater_vision.matching.learned_filter import MatchFilter
from underwater_vision.pointcloud.learned_confidence import holdout_observation
from underwater_vision.research_pipeline import unique_matches
from underwater_vision.retrieval.pairs import retrieve_pairs


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


def test_context_filter_is_permutation_equivariant():
    torch.manual_seed(3)
    model = MatchFilter("context").eval()
    x = torch.randn(30, 12)
    order = torch.randperm(30)
    with torch.no_grad():
        np.testing.assert_allclose(model(x)[order], model(x[order]), atol=1e-6)
    assert model(torch.empty(0, 12)).shape == (0,)


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


def test_retrieval_never_contains_self_pairs_or_duplicates():
    descriptors = np.random.default_rng(2).normal(size=(20, 8))
    pairs, neighbors = retrieve_pairs(descriptors, 3)
    assert all(a < b for a, b in pairs)
    assert len(set(pairs)) == len(pairs)
    assert all(i not in row for i, row in enumerate(neighbors))


def test_probability_filter_retains_unique_highest_scoring_pairs():
    ids = np.array([[0, 1], [1, 1], [2, 2]])
    accepted = unique_matches(ids, np.array([0.8, 0.9, 0.2]), 0.5)
    np.testing.assert_array_equal(accepted, [1])


def test_binary_metrics_exclude_ambiguous_labels():
    result = binary_metrics([0, 1, -1, 1], [0.1, 0.9, 0.5, 0.8])
    assert result["samples"] == 3
    assert result["roc_auc"] == 1
    assert result["average_precision"] == 1
