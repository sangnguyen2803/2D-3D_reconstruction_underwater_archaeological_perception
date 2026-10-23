import numpy as np
import pytest

from underwater_vision.utils.io import atomic_npz


def test_atomic_npz_creates_parent_directories(tmp_path):
    path = tmp_path / "cache" / "features" / "frame.npz"
    atomic_npz(path, xy=np.array([[1, 2]]))
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive["xy"], [[1, 2]])
    assert list(path.parent.iterdir()) == [path]


def test_atomic_npz_preserves_previous_cache_on_write_failure(tmp_path, monkeypatch):
    path = tmp_path / "frame.npz"
    atomic_npz(path, value=np.array([1]))

    def fail_write(temporary, **values):
        temporary.write_bytes(b"partial archive")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", fail_write)
    with pytest.raises(OSError, match="disk full"):
        atomic_npz(path, value=np.array([2]))
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive["value"], [1])
    assert list(tmp_path.iterdir()) == [path]
