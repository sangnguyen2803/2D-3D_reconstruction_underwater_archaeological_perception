import numpy as np
import pycolmap

from underwater_vision.data.dataset_loader import Frame
from underwater_vision.reconstruction.colmap.database import create_database, insert_matches


def test_database_roundtrip(tmp_path):
    frames = [Frame(tmp_path / n, n, None) for n in ["a.jpg", "b.jpg"]]
    features = {f.name: {"xy": np.array([[4, 5], [6, 7]], np.float32)} for f in frames}
    camera = pycolmap.Camera(model="PINHOLE", width=100, height=100, params=[80, 80, 50, 50])
    path = tmp_path / "database.db"
    ids = create_database(path, frames, features, camera)
    pairs = np.array([[0, 1], [1, 0]], np.uint32)
    insert_matches(
        path,
        ids,
        [{"a": "a.jpg", "b": "b.jpg", "pairs": pairs, "inliers": np.array([True, False])}],
    )
    with pycolmap.Database.open(str(path)) as database:
        np.testing.assert_allclose(
            database.read_keypoints(ids["a.jpg"]), features["a.jpg"]["xy"] + 0.5
        )
        np.testing.assert_equal(database.read_matches(ids["a.jpg"], ids["b.jpg"]), pairs[:1])
