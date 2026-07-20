"""Contracts for labeled data, inference caching and held-out training outputs."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from underwater_vision.matchability.dataset import MatchabilityDataset
from underwater_vision.matchability.infer import (
    MatchabilityPredictor,
    attach_matchability,
    select_keypoints,
)
from underwater_vision.matchability.labels import LABEL_SPEC
from underwater_vision.matchability.train import fit
from underwater_vision.utils.io import atomic_npz, write_json, write_rgb


@pytest.fixture
def labeled_dataset(tmp_path):
    root, features = tmp_path / "labels", tmp_path / "features"
    xy = np.array([[0, 0], [4, 3], [7, 5], [9, 7]], np.float32)
    labels = np.array([0, 1, -1, 1], np.float32)
    rgb = np.random.default_rng(7).integers(0, 256, (8, 10, 3), dtype=np.uint8)
    write_json(root / "manifest_v3.json", {"feature_source": str(features)})
    write_rgb(features / "images/frame.png", rgb)
    (features / "features").mkdir(parents=True)
    (root / "labels/fold0/hybrid").mkdir(parents=True)
    atomic_npz(features / "features/frame.npz", xy=xy)
    atomic_npz(root / "labels/fold0/hybrid/frame.npz", l1=labels, soft=labels)
    return root, xy, labels, rgb


def test_dataset_external_features_and_returned_arrays_are_independent(labeled_dataset):
    root, xy, labels, rgb = labeled_dataset
    dataset = MatchabilityDataset(root, ["frame"], "fold0")
    first = dataset[0]
    np.testing.assert_array_equal(first["xy"].numpy(), xy)
    np.testing.assert_array_equal(first["labels"].numpy(), labels)
    np.testing.assert_allclose(first["rgb"].numpy(), rgb.transpose(2, 0, 1) / 255)
    first["xy"][0] = 99
    first["labels"][0] = 99
    second = dataset[0]
    np.testing.assert_array_equal(second["xy"].numpy(), xy)
    np.testing.assert_array_equal(second["labels"].numpy(), labels)


def test_seeded_crop_keeps_labels_aligned_and_rejects_oversize(labeled_dataset):
    root, xy, labels, _ = labeled_dataset
    rng = np.random.default_rng(42)
    left, top = int(rng.integers(10 - 6 + 1)), int(rng.integers(8 - 5 + 1))
    shifted = xy - [left, top]
    keep = (shifted[:, 0] >= 0) & (shifted[:, 0] <= 5) & (shifted[:, 1] >= 0) & (shifted[:, 1] <= 4)
    first = MatchabilityDataset(root, ["frame"], "fold0", crop_hw=(5, 6), seed=42)[0]
    second = MatchabilityDataset(root, ["frame"], "fold0", crop_hw=(5, 6), seed=42)[0]
    np.testing.assert_array_equal(first["xy"].numpy(), shifted[keep])
    np.testing.assert_array_equal(first["labels"].numpy(), labels[keep])
    assert torch.equal(first["rgb"], second["rgb"])
    assert first["rgb"].shape == (3, 5, 6)
    assert 0 <= first["rgb"].min() <= first["rgb"].max() <= 1
    with pytest.raises(ValueError, match="crop larger"):
        MatchabilityDataset(root, ["frame"], "fold0", crop_hw=(9, 11))[0]


def test_selection_exact_count_and_provenance():
    record = {
        "xy": np.zeros((5, 2)),
        "local": np.ones((5, 128)),
        "shape": np.array([5, 5]),
        "matchability": np.array([0.2, 0.9, 0.7, 0.1, 0.4]),
        "source_indices": np.arange(5) + 10,
    }
    selected = select_keypoints(record, 3)
    assert selected["source_indices"].tolist() == [11, 12, 14]
    assert selected["local"].shape == (3, 128)
    assert selected["shape"].tolist() == [5, 5]
    with pytest.raises(ValueError):
        select_keypoints(record, 6)


def test_attach_checks_shape_and_bilinear_sample():
    record = {"xy": np.array([[0.5, 0.5]], np.float32), "shape": np.array([2, 2])}
    attached = attach_matchability(record, np.array([[0.0, 1.0], [1.0, 0.0]], np.float32), "abc")
    assert attached["matchability"][0] == 0.5
    assert str(attached["matchability_hash"]) == "abc"
    with pytest.raises(ValueError):
        attach_matchability(record, np.zeros((3, 3)), "abc")


def test_inference_cache_tracks_temperature_calibration_and_image(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "frame.png"
    write_rgb(source, np.zeros((8, 10, 3), np.uint8))
    predictor = MatchabilityPredictor.__new__(MatchabilityPredictor)
    predictor.checkpoint_hash, predictor.temperature, predictor.device = "checkpoint", 2.0, "cpu"
    predictor.calibration_hash = None
    calls = []

    def predict(rgb):
        calls.append(rgb.shape)
        return np.full(rgb.shape[:2], 0.7, np.float32)

    predictor.predict = predict
    first = predictor.cached(source)
    np.testing.assert_array_equal(first, predictor.cached(source))
    assert len(calls) == 1
    predictor.temperature = 3.0
    predictor.cached(source)
    predictor.calibration_hash = "new-calibration"
    predictor.cached(source)
    predictor.cached(source)
    assert len(calls) == 3
    write_rgb(source, np.zeros((9, 11, 3), np.uint8))
    assert predictor.cached(source).shape == (9, 11)
    assert len(calls) == 4


def test_training_keeps_test_views_out_of_fitting_and_exports_predictions(tmp_path, monkeypatch):
    import sys

    from underwater_vision.matchability import train

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class TinyModel(torch.nn.Module):
        def __init__(self, _specification, pretrained=True):
            super().__init__()
            self.kind = "resnet_unet"
            self.head = torch.nn.Conv2d(3, 1, 1)

        def forward(self, image):
            return self.head(image)

    monkeypatch.setattr(train, "MatchabilityModel", TinyModel)

    def noop(*args, **kwargs):
        return None

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(
            set_tracking_uri=noop,
            set_experiment=noop,
            start_run=lambda **kwargs: nullcontext(),
            log_params=noop,
            log_metrics=noop,
        ),
    )
    source = tmp_path / "dataset"
    groups = {"train": ["train"], "validation": ["validation"], "test": ["test"]}
    write_json(
        source / "manifest_v3.json",
        {"protocols": [{"name": "fold0", "groups": groups}], "label_spec": LABEL_SPEC},
    )
    write_json(source / "audit_v3.json", {"status": "complete"})
    (source / "features").mkdir(parents=True)
    (source / "labels/fold0/hybrid").mkdir(parents=True)
    for name in ["train", "validation", "test"]:
        rgb = np.zeros((16, 16, 3), np.uint8)
        rgb[8:] = 255
        write_rgb(source / "images" / (name + ".png"), rgb)
        atomic_npz(
            source / "features" / (name + ".npz"), xy=np.array([[2, 2], [12, 12]], np.float32)
        )
        atomic_npz(source / "labels/fold0/hybrid" / (name + ".npz"), l1=np.array([0, 1], np.int8))
    config, model, criteria = [
        tmp_path / name for name in ["train.yaml", "model.yaml", "criteria.yaml"]
    ]
    config.write_text(
        "crop_hw: null\nencoder_lr: 0.00002\nhead_lr: 0.0003\nweight_decay: 0.01\n"
        "gradient_clip: 1.0\naccumulation: 1\n"
    )
    model.write_text("kind: resnet_unet\n")
    criteria.write_text("status: frozen_after_E0\n")
    output = fit(
        SimpleNamespace(
            output=tmp_path / "tiny_v3",
            dataset=source,
            config=config,
            criteria=criteria,
            model=model,
            protocol="fold0",
            graph="hybrid",
            variant="l1",
            seed=42,
            epochs=1,
        )
    )
    saved = torch.load(output / "model_v3.pt", weights_only=True)
    assert saved["training_images"] == ["train"]
    assert saved["validation_images"] == ["validation"]
    assert saved["test_images"] == ["test"]
    assert saved["status"] == "trained" and saved["temperature"] > 0
    report = json.loads((output / "evaluation_v3.json").read_text())
    assert report["status"] == "complete" and report["test_after"]["samples"] == 2
    assert [path.name for path in (output / "predictions").glob("*.npz")] == ["test.npz"]
    with np.load(output / "predictions/test.npz") as prediction:
        np.testing.assert_array_equal(prediction["labels"], [0, 1])
        assert prediction["probability"].shape == (2,)
