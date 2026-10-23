import numpy as np
import pytest

from underwater_vision.preprocessing.images import resize_rgb


@pytest.mark.parametrize("shape", [(1, 1000, 3), (1000, 1, 3)])
def test_resize_keeps_thin_images_nonempty(shape):
    resized, (sx, sy) = resize_rgb(np.zeros(shape, np.uint8), max_size=1)
    assert resized.shape == (1, 1, 3)
    assert sx == 1 / shape[1]
    assert sy == 1 / shape[0]


@pytest.mark.parametrize("max_size", [0, -1, 1.5, np.nan, np.inf, True])
def test_resize_rejects_invalid_max_size(max_size):
    with pytest.raises(ValueError, match="max_size must be a positive integer"):
        resize_rgb(np.zeros((10, 20, 3), np.uint8), max_size=max_size)


@pytest.mark.parametrize("shape", [(0, 10, 3), (10, 0, 3), (10, 20), (10, 20, 4)])
def test_resize_rejects_empty_or_non_rgb_images(shape):
    with pytest.raises(ValueError, match="nonempty RGB image"):
        resize_rgb(np.zeros(shape, np.uint8))
