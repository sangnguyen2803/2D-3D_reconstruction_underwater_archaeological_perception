from types import SimpleNamespace

import numpy as np

from underwater_vision.features.store import feature_record
from underwater_vision.utils.io import write_rgb


def test_changed_detector_has_separate_cache_and_preserves_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rgb = np.random.default_rng(42).integers(110, 140, (192, 256, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    write_rgb(path, rgb)

    class Neural:
        provenance = {"backbone": "cache-test"}
        revision = "test"
        calls = 0

        def extract(self, image, xy):
            self.calls += 1
            return np.zeros((len(xy), 4), np.float32)

    neural = Neural()
    frame = SimpleNamespace(path=path)
    default = feature_record(frame, neural)
    changed = feature_record(frame, neural, contrast_threshold=0.01)
    repeated = feature_record(frame, neural)
    assert neural.calls == 2
    assert len(list((tmp_path / "cache/research_features_v1").glob("*.npz"))) == 2
    assert len(changed["xy"]) > len(default["xy"])
    np.testing.assert_array_equal(default["xy"], repeated["xy"])
    np.testing.assert_array_equal(default["local"], repeated["local"])
