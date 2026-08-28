import numpy as np
import pytest

from underwater_vision.pointcloud.mesh_cleanup import trim_mesh


def test_unsupported_extension_removed_without_losing_observed_patch():
    x, y = np.meshgrid(np.arange(8), np.arange(8))
    observed = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    vertices = np.vstack([observed, [[40, 40, 10], [41, 40, 10], [40, 41, 10]]])
    patch = []
    for row in range(7):
        for column in range(7):
            index = row * 8 + column
            patch.extend([[index, index + 1, index + 8], [index + 1, index + 8, index + 9]])
    faces = np.array([*patch, [64, 65, 66]])
    original = vertices.copy()
    used, cleaned, report = trim_mesh(vertices, faces, observed)
    assert len(cleaned) == 98
    assert not np.isin([64, 65, 66], used).any()
    assert report["removed_area_fraction"] > 0
    assert report["dense_point_coverage_before"] == report["dense_point_coverage_after"] == 1
    np.testing.assert_array_equal(vertices, original)
    with pytest.raises(ValueError):
        trim_mesh(vertices, faces, observed, radius_factor=-1)
