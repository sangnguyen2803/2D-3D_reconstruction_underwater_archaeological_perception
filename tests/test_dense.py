import numpy as np

from underwater_vision.pointcloud.dense_confidence import read_colmap_array


def test_colmap_depth_array_layout(tmp_path):
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "depth.bin"
    with path.open("wb") as f:
        f.write(b"4&3&1&")
        depth.T.ravel(order="F").tofile(f)
    np.testing.assert_array_equal(read_colmap_array(path), depth)
