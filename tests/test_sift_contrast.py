import cv2
import numpy as np
import pytest

from underwater_vision.features.local import extract_sift


def test_low_contrast_texture_coverage_and_descriptor_normalization():
    rng = np.random.default_rng(4)
    texture = rng.normal(127, 10, (240, 320)).clip(0, 255).astype(np.uint8)
    gray = cv2.GaussianBlur(texture, (3, 3), 0.5)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    default_xy, default_descriptors = extract_sift(rgb)
    legacy, _ = cv2.SIFT_create(nfeatures=4096).detectAndCompute(gray, None)
    np.testing.assert_array_equal(
        default_xy, np.asarray([k.pt for k in legacy], np.float32).reshape(-1, 2)
    )
    xy, descriptors = extract_sift(rgb, contrast_threshold=0.01)
    assert len(xy) > len(default_xy)
    assert descriptors.shape == (len(xy), 128)
    np.testing.assert_allclose(np.linalg.norm(descriptors, axis=1), 1, atol=1e-6)
    assert default_descriptors.shape == (len(default_xy), 128)
    with pytest.raises(ValueError):
        extract_sift(rgb, contrast_threshold=0)
