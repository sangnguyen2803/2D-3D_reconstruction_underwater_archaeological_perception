"""v2 leakage, geometry, adapter contracts and deployment regressions."""

import base64
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

from underwater_vision.evaluation.segmentation import (
    confusion,
    fit_temperature,
    paired_bootstrap,
    pixel_ece,
    segmentation_metrics,
)
from underwater_vision.matching.learned_filter import MatchFilter, MatchPredictor
from underwater_vision.semantic.confidence import (
    conformal_quantile,
    multi_holdout_rows,
    robust_loss,
)
from underwater_vision.semantic.masks import (
    attach_probabilities,
    mask_keypoints,
    match_class_features,
    sample_probabilities,
)
from underwater_vision.semantic.protocol import load_manifest, new_run, require_gate, spatial_folds
from underwater_vision.semantic.schema import merge_mask, validate_mask
from underwater_vision.semantic.train import masked_loss
from underwater_vision.semantic.voting import evaluate_label_holdouts, fuse_probabilities


def test_schema_merge_ignore_is_preserved():
    mask = np.array([[0, 3, 255]], np.uint8)
    assert merge_mask(mask, {3: 1}).tolist() == [[0, 1, 255]]
    with pytest.raises(ValueError):
        merge_mask(mask, {255: 0})
    with pytest.raises(ValueError):
        validate_mask(np.array([[7]], np.uint8))


def test_ignore_pixels_never_enter_loss_or_metrics():
    logits = torch.randn(1, 6, 1, 2, requires_grad=True)
    target = torch.tensor([[[2, 255]]])
    loss = masked_loss(logits, target, 0)
    loss.backward()
    assert torch.all(logits.grad[:, :, :, 1] == 0)
    matrix = confusion(np.array([[2, 255]], np.uint8), np.array([[2, 1]], np.uint8))
    assert matrix.sum() == 1 and segmentation_metrics(matrix)["pixel_accuracy"] == 1
    assert torch.isfinite(masked_loss(logits, torch.full_like(target, 255)))


def test_abstaining_teacher_is_not_credited_for_ignore():
    matrix = confusion(np.array([[2]], np.uint8), np.array([[255]], np.uint8))
    assert segmentation_metrics(matrix)["miou"] == 0


def test_hand_manifest_rejects_wrong_spatial_role(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {"image_id": "a", "slab": "train", "role": "test", "label_source": "hand"}
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="slab"):
        load_manifest(path)


def test_pending_hand_manifest_cannot_start_final_evaluation(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"entries": [], "frozen": None}))
    with pytest.raises(ValueError, match="frozen"):
        load_manifest(path, require_frozen=True)


def test_human_gate_cannot_be_replaced_by_proposals(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"status": "awaiting_human_review"}))
    with pytest.raises(ValueError, match="human"):
        require_gate(path)


def test_dynamic_mask_and_source_indices_are_aligned():
    probability = np.zeros((6, 2, 2), np.float32)
    probability[0] = 1
    probability[0, 0, 1] = 0
    probability[5, 0, 1] = 1
    record = {
        "xy": np.array([[0, 0], [1, 0]], np.float32),
        "local": np.eye(2, dtype=np.float32),
        "shape": np.array([2, 2]),
    }
    record = attach_probabilities(record, probability, "hash")
    masked = mask_keypoints(record, 0.5)
    assert masked["xy"].tolist() == [[0, 0]]
    assert masked["source_indices"].tolist() == [0]
    assert masked["local"].shape == (1, 2) and masked["shape"].tolist() == [2, 2]


def test_bilinear_sampling_preserves_probability_mass():
    rng = np.random.default_rng(4)
    p = rng.random((6, 4, 5)).astype(np.float32)
    p /= p.sum(0)
    q, entropy = sample_probabilities(p, np.array([[1.5, 2.3], [0, 0]], np.float32))
    assert np.allclose(q.sum(1), 1) and (entropy >= 0).all()
    assert sample_probabilities(p, [])[0].shape == (0, 6)


def test_class_feature_swap_symmetry():
    a = {
        "class_probabilities": np.array([[0.2, 0.2, 0.2, 0.1, 0.1, 0.2]], np.float32),
        "class_entropy": np.array([1.2]),
    }
    b = {
        "class_probabilities": np.array([[0.1, 0.1, 0.1, 0.2, 0.2, 0.3]], np.float32),
        "class_entropy": np.array([1.5]),
    }
    pairs = np.array([[0, 0]], np.uint32)
    x, y = match_class_features(a, b, pairs), match_class_features(b, a, pairs)
    assert np.allclose(x[:, :2], y[:, :2])
    assert np.allclose(x[:, 2:], y[:, 2:][:, ::-1])
    assert match_class_features(a, b, pairs, "full").shape == (1, 16)


def test_context_equivariance_with_class_features():
    torch.manual_seed(42)
    model = MatchFilter("context", channels=28).eval()
    x = torch.randn(17, 28)
    order = torch.randperm(17)
    with torch.no_grad():
        assert torch.allclose(model(x)[order], model(x[order]), atol=1e-6)


def test_filter_rejects_feature_version_and_semantic_hash(tmp_path):
    path = tmp_path / "filter.pt"
    torch.save({"feature_spec": "match_class_v2_compact", "segmentation_hash": "a"}, path)
    with pytest.raises(ValueError, match="specification"):
        MatchPredictor(path)
    with pytest.raises(ValueError, match="hash"):
        MatchPredictor(path, "compact", "b")


def test_voting_unanimous_and_excludes_entire_image():
    p = np.eye(6)[[2, 2, 5]]
    _, label, agreement = fuse_probabilities(p, ["a", "b", "held"], "held")
    assert label == 2 and agreement == 1
    changed = p.copy()
    changed[2] = np.eye(6)[0]
    assert np.allclose(
        fuse_probabilities(changed, ["a", "b", "held"], "held")[0],
        fuse_probabilities(p, ["a", "b", "held"], "held")[0],
    )


def test_temperature_keeps_argmax_and_excludes_ignore():
    logits = np.array([[10, 1, 0, 0, 0, 0], [1, 8, 0, 0, 0, 0]], float)
    labels = np.array([0, 255], np.uint8)
    temperature = fit_temperature(logits, labels)
    assert np.array_equal(logits.argmax(-1), (logits / temperature).argmax(-1))
    assert pixel_ece(logits, labels, temperature) < 0.01


def test_paired_bootstrap_group_identity():
    result = paired_bootstrap(
        [(1, 1), (2, 2), (3, 3)], lambda rows: np.mean([a - b for a, b in rows]), repeats=100
    )
    assert result["ci95"] == [0, 0] and result["groups"] == 3


def test_new_runs_are_versioned_and_immutable(tmp_path):
    path = tmp_path / "run_v2"
    new_run(path)
    with pytest.raises(FileExistsError):
        new_run(path)
    with pytest.raises(ValueError):
        new_run(tmp_path / "bad")


def test_three_spatial_blocks_disjoint_and_retrained():
    names = [str(i) for i in range(90)]
    centers = np.c_[np.linspace(0, 9, 90), np.zeros((90, 2))]
    folds = spatial_folds(names, centers, 0.2)
    tests = [set(f["groups"]["test"]) for f in folds]
    assert not (tests[0] & tests[1] or tests[0] & tests[2] or tests[1] & tests[2])
    for f in folds:
        train, val, test = (set(f["groups"][s]) for s in ["train", "validation", "test"])
        assert not (train & val or train & test or val & test)


def test_conformal_finite_sample_quantile_and_robust_outlier_loss():
    errors = np.expm1(np.arange(10) / 10)
    radius = conformal_quantile(errors, np.zeros(10), 0.9)
    assert np.isclose(radius, 0.9)
    mu = torch.zeros(2, requires_grad=True)
    target = torch.tensor([1.0, 1000.0])
    loss = robust_loss(mu, torch.zeros(2), target)
    assert torch.isfinite(loss) and loss.item() < 50
    loss.backward()
    assert torch.isfinite(mu.grad).all()


def test_multiple_holdouts_exclude_semantic_and_quality_inputs():
    from underwater_vision.geometry.reference import project

    poses = []
    for x in [-1, 0, 1, 2]:
        pose = np.eye(4)
        pose[0, 3] = x
        poses.append(pose)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)
    point = np.array([0.1, 0.2, 5])
    xy = [project(point, pose, K) for pose in poses]
    nodes = [(str(i), 0) for i in range(4)]
    semantic = np.eye(4, dtype=np.float32)
    quality = np.ones(4)
    p = np.eye(6)[[0, 0, 0, 0]]
    row = multi_holdout_rows(nodes, poses, xy, semantic, quality, p, K, max_holdouts=1)[0]
    for i in row["heldout_indices"]:
        p[i] = np.eye(6)[5]
        quality[i] = 0
        semantic[i] *= 7
    other = multi_holdout_rows(nodes, poses, xy, semantic, quality, p, K, max_holdouts=1)[0]
    assert np.allclose(row["x"], other["x"])
    assert row["x"].shape == (17,)


def test_api_smoke_without_colmap_model():
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from underwater_vision.serving.api import create_app

    class Predictor:
        checkpoint_hash = "synthetic_test"
        metadata = {"dynamic_threshold": 0.5}

        def predict(self, rgb):
            p = np.zeros((6, *rgb.shape[:2]), np.float32)
            p[0] = 1
            return p

    client = TestClient(create_app(Predictor()))
    assert client.get("/health").json()["status"] == "ready"
    image = io.BytesIO()
    Image.fromarray(np.zeros((8, 10, 3), np.uint8)).save(image, format="PNG")
    result = client.post("/predict", files={"image": ("image.png", image.getvalue(), "image/png")})
    assert result.status_code == 200
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(result.json()["mask_png_base64"]))))
    assert decoded.shape == (8, 10) and (decoded == 0).all()
    assert client.post("/predict", files={"image": ("bad", b"not image")}).status_code == 400


def test_onnx_segformer_parity_and_dynamic_batch(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("transformers")
    import onnxruntime as ort

    from underwater_vision.semantic.model import SegmentationModel

    specification = {
        "kind": "segformer",
        "hf_config": {
            "depths": [1, 1, 1, 1],
            "hidden_sizes": [8, 16, 32, 64],
            "num_attention_heads": [1, 2, 4, 8],
            "decoder_hidden_size": 16,
        },
    }
    model = SegmentationModel(specification, pretrained=False).eval()
    torch.manual_seed(42)
    image = torch.rand(1, 3, 64, 64)
    path = tmp_path / "model.onnx"
    torch.onnx.export(
        model,
        image,
        str(path),
        opset_version=17,
        input_names=["rgb"],
        output_names=["logits"],
        dynamic_axes={"rgb": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = model(image).numpy()
    actual = session.run(None, {"rgb": image.numpy()})[0]
    assert np.allclose(actual, expected, atol=1e-4, rtol=1e-3)
    assert session.run(None, {"rgb": np.repeat(image.numpy(), 2, axis=0)})[0].shape == (
        2,
        6,
        64,
        64,
    )


def test_semantic_vote_evaluation_does_not_use_held_label():
    tracks = [
        {
            "image_ids": ["a", "b", "held"],
            "probabilities": np.eye(6)[[2, 2, 5]],
            "xy": [[0, 0], [0, 0], [0, 0]],
        }
    ]
    report = evaluate_label_holdouts(tracks, {"held": np.array([[2]], np.uint8)})
    assert report["pixel_accuracy"] == 1


def test_semantic_evaluation_rejects_segmentation_training_images(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import underwater_vision.research_pipeline as pipeline
    from underwater_vision.semantic import infer

    frames = [SimpleNamespace(name=str(i), reference_c2w=None) for i in range(3)]

    class Dataset:
        def __init__(self, *args):
            self.frames = frames

        def select(self, *args):
            return frames

    semantic = SimpleNamespace(
        checkpoint_hash="hash", metadata={"training_images": ["0"], "validation_images": []}
    )
    filter_ = SimpleNamespace(
        metadata={"backbone": "b", "revision": "r", "training_images": [], "validation_images": []}
    )
    monkeypatch.setattr(pipeline, "MermaidDataset", Dataset)
    monkeypatch.setattr(pipeline, "MatchPredictor", lambda *args: filter_)
    monkeypatch.setattr(infer, "SemanticPredictor", lambda *args: semantic)
    cfg = {
        "data": {"root": "unused", "image_dir": None, "start": 0, "count": 3, "stride": 1},
        "matching": {
            "filter_checkpoint": "unused",
            "semantic_features": "compact",
            "semantic_checkpoint": "unused",
            "evaluation_mode": True,
        },
        "model": {"backbone": "b", "revision": "r"},
    }
    with pytest.raises(ValueError, match="Segmentation training"):
        pipeline.execute_learned(cfg, tmp_path)


def test_review_api_accepts_gray_rgb_ids_and_enforces_repeat_delay(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from underwater_vision.serving.review import create_review_app

    root = tmp_path / "labels"
    audit = tmp_path / "audit"
    root.mkdir()
    audit.mkdir()
    (root / "repeat_masks").mkdir()
    image = root / "image.png"
    Image.fromarray(np.zeros((2, 3, 3), np.uint8)).save(image)
    mask = root / "mask.png"
    Image.fromarray(np.full((2, 3), 255, np.uint8)).save(mask)
    name = "image.png"
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "frozen": None,
                "repeat_tasks": [name],
                "entries": [
                    {
                        "image_id": name,
                        "role": "validation",
                        "slab": "validation",
                        "image": str(image),
                        "mask": str(mask),
                        "label_source": "hand",
                        "reviewed": False,
                    }
                ],
            }
        )
    )
    (audit / "sample_manifest.json").write_text(json.dumps({"groups": {"train": [name]}}))
    client = TestClient(create_review_app(root, audit))
    assert client.get("/").status_code == 200
    assert client.get("/tasks").json()["hand"][0]["reviewed"] is False
    stream = io.BytesIO()
    Image.fromarray(np.zeros((2, 3, 3), np.uint8)).save(stream, format="PNG")
    payload = {
        "mask_png_base64": base64.b64encode(stream.getvalue()).decode(),
        "annotator": "Test annotator",
    }
    assert client.post("/annotation/" + name, json=payload).status_code == 200
    saved = load_manifest(path)
    assert saved["entries"][0]["reviewed"] and np.asarray(Image.open(mask)).ndim == 2
    assert client.post("/annotation/" + name, json={**payload, "repeat": True}).status_code == 409


def test_point_predictor_rejects_wrong_semantic_hash(tmp_path):
    from underwater_vision.semantic.confidence import SemanticErrorPredictor

    path = tmp_path / "point.pt"
    torch.save({"feature_spec": "point_class_v2_17", "segmentation_hash": "a"}, path)
    with pytest.raises(ValueError, match="hash mismatch"):
        SemanticErrorPredictor(path, "b")


def test_detector_boxes_exclude_static_and_tiny_regions():
    from underwater_vision.semantic.detection import mask_boxes

    mask = np.zeros((20, 30), np.uint8)
    mask[2:8, 4:10] = 5
    mask[12, 25] = 5
    boxes = mask_boxes(mask, min_pixels=20)
    assert len(boxes) == 1 and boxes[0][0] == 0
    assert np.allclose(boxes[0][1:], [7 / 30, 5 / 20, 6 / 30, 6 / 20])


def test_rotated_spatial_spec_matches_membership():
    from underwater_vision.data.spatial import assign_spatial

    names = [str(i) for i in range(90)]
    centers = np.c_[np.linspace(0, 9, 90), np.zeros((90, 2))]
    for fold in spatial_folds(names, centers, 0.2):
        assignment = assign_spatial(centers, fold["specification"])
        for name, group in zip(names, assignment, strict=True):
            assert name in fold["groups"][group]


def test_semantic_checkpoint_roundtrip_cache_and_temperature_key(tmp_path):
    pytest.importorskip("transformers")
    from underwater_vision.semantic.infer import SemanticPredictor
    from underwater_vision.semantic.model import SegmentationModel

    torch.set_num_threads(4)
    specification = {
        "kind": "segformer",
        "hf_config": {
            "depths": [1, 1, 1, 1],
            "hidden_sizes": [8, 16, 32, 64],
            "num_attention_heads": [1, 2, 4, 8],
            "decoder_hidden_size": 16,
        },
    }
    model = SegmentationModel(specification, pretrained=False).eval()
    specification["hf_config"] = model.network.config.to_dict()
    specification["hf_config"].pop("num_labels", None)
    # Synthetic trained-status fixture tests loading, not segmentation quality.
    saved = {
        "feature_spec": "segmentation_six_v2",
        "status": "trained",
        "state_dict": model.state_dict(),
        "model_config": specification,
        "size": [64, 64],
        "temperature": 1.0,
    }
    path = tmp_path / "model.pt"
    torch.save(saved, path)
    image = tmp_path / "image.png"
    Image.fromarray(np.zeros((16, 20, 3), np.uint8)).save(image)
    cache = tmp_path / "cache"
    predictor = SemanticPredictor(path, device="cpu", cache_root=cache)
    first = predictor.cached(image, max_size=20)
    second = predictor.cached(image, max_size=20)
    assert np.array_equal(first, second) and np.allclose(first.sum(0), 1, atol=1e-6)
    assert len(list(cache.glob("*.npz"))) == 1
    saved["temperature"] = 2.0
    other = tmp_path / "other.pt"
    torch.save(saved, other)
    SemanticPredictor(other, device="cpu", cache_root=cache).cached(image, max_size=20)
    assert len(list(cache.glob("*.npz"))) == 2
