import numpy as np
import pytest
import torch

from underwater_vision.matching.geometry_precision import augment_coordinates, precision_features


def test_axis_augmentation_preserves_match_geometry_and_appearance():
    x = torch.tensor([[0.2, 0.5, 0.8, 0.7, 0.1, 0.2, 0.4, 0.6, 0.3, 0.4]])
    for turns in range(4):
        for flip in [False, True]:
            augmented = augment_coordinates(x, turns, flip)
            torch.testing.assert_close(augmented[:, :4], x[:, :4])
            torch.testing.assert_close(augmented[:, 8:10], augmented[:, 6:8] - augmented[:, 4:6])
            torch.testing.assert_close(
                torch.linalg.vector_norm(augmented[:, 8:10], dim=1),
                torch.linalg.vector_norm(x[:, 8:10], dim=1),
            )
    torch.testing.assert_close(augment_coordinates(x, 4), x)


def test_conditional_features_verify_frozen_match_alignment():
    local = np.eye(4, 128, dtype=np.float32)
    a = {
        "local": local,
        "xy": np.array([[10, 20], [20, 30], [30, 40], [40, 50]], np.float32),
        "quality": np.ones(4),
    }
    b = {"local": local, "xy": a["xy"] + np.array([5, 2]), "quality": np.ones(4) * 0.5}
    pairs = np.array([[0, 0], [2, 2]], np.uint32)
    x = precision_features(a, b, pairs)
    assert x.shape == (2, 10)
    np.testing.assert_array_equal(x[:, :2], np.zeros((2, 2)))
    np.testing.assert_allclose(x[:, 8:10], x[:, 6:8] - x[:, 4:6], atol=1e-7)
    with pytest.raises(ValueError, match="alignment"):
        precision_features(a, b, np.array([[0, 1]], np.uint32))
