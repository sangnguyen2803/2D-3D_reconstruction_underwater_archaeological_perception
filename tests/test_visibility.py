import cv2
import numpy as np

from underwater_vision.visibility.consistency import update_repeatability
from underwater_vision.visibility.deterministic import estimate_visibility, sample_map


def test_blur_lowers_reliability():
    random = np.random.default_rng(1)
    rgb = random.integers(20, 200, (96, 96, 3), dtype=np.uint8)
    sharp, _ = estimate_visibility(rgb)
    blurred, _ = estimate_visibility(cv2.GaussianBlur(rgb, (25, 25), 6))
    assert 0 <= blurred.min() <= blurred.max() <= 1
    assert sharp.mean() > blurred.mean() + 0.2


def test_flat_region_is_not_normalized_to_good_visibility():
    visibility, _ = estimate_visibility(np.full((64, 64, 3), 80, np.uint8))
    assert visibility.mean() < 0.15


def test_map_sampling_and_boundaries():
    values = np.arange(9, dtype=np.float32).reshape(3, 3)
    np.testing.assert_allclose(
        sample_map(values, np.array([[1, 1], [0.5, 0.5], [-1, -1]])), [4, 2, 0]
    )


def test_repeatability_unobserved_and_failure():
    features = {n: {"xy": np.zeros((3, 2)), "reliability": np.ones(3)} for n in ["a", "b"]}
    results = [
        {
            "a": "a",
            "b": "b",
            "pairs": np.array([[0, 0], [1, 1]]),
            "inliers": np.array([True, False]),
        }
    ]
    update_repeatability(features, results)
    assert features["a"]["repeatability"][0] > features["a"]["repeatability"][2]
    assert features["a"]["repeatability"][1] < features["a"]["repeatability"][2]
