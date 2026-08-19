"""Geometry-label, sparse-loss and detachable v3 integration contracts."""

import numpy as np
import pytest
import torch

from underwater_vision.evaluation.matchability import metrics, paired_ap_bootstrap
from underwater_vision.matchability.calibration import fit_temperature, probabilities
from underwater_vision.matchability.infer import attach_matchability, select_keypoints
from underwater_vision.matchability.labels import (
    aggregate_labels,
    eligible_pair,
    same_slab_pairs,
    stronger_labels,
)
from underwater_vision.matchability.loss import resize_coordinates, sample_logits, sparse_bce
from underwater_vision.matchability.model import MatchabilityModel
from underwater_vision.matchability.protocol import check_evaluation, new_run
from underwater_vision.matching.learned_filter import (
    MatchFilter,
    MatchPredictor,
    matchability_features,
)
from underwater_vision.visibility.source import apply_source


def test_positive_override_n_min_and_ambiguous_ignore():
    error = [[1, np.nan], [6, np.nan], [4, 5], [2, 7], [np.nan, np.nan], [1, 5]]
    labels, soft = aggregate_labels(error)
    assert labels.tolist() == [1, -1, 0, -1, -1, 1]
    assert soft.tolist() == [1, -1, 0, -1, -1, 0.5]


def test_ineligible_pairs_never_make_negatives():
    assert not eligible_pair([8] * 20)
    y, _ = aggregate_labels(np.empty((20, 0)))
    assert np.all(y == -1)


def test_eligibility_boundary_is_inclusive():
    assert eligible_pair([1.5] * 15)
    assert not eligible_pair([1.5] * 14)


def test_l2_unknown_track_is_not_negative():
    assert stronger_labels([1, 1, 1, 0, -1], [2, 3, np.nan, np.nan, 0]).tolist() == [
        1,
        -1,
        -1,
        0,
        -1,
    ]


def test_labels_reject_bad_thresholds():
    with pytest.raises(ValueError):
        aggregate_labels([[1]], positive=4, negative=1)


def test_sampling_opencv_centers_and_fractional_points():
    values = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    xy = torch.tensor([[0.0, 0.0], [3.0, 2.0], [1.5, 1.0]])
    assert torch.allclose(sample_logits(values, xy)[0], torch.tensor([0.0, 11.0, 5.5]), atol=1e-6)


def test_resize_and_crop_alignment():
    xy = torch.tensor([[10.0, 8.0]])
    assert torch.allclose(resize_coordinates(xy, (20, 40), (40, 80)), torch.tensor([[20.5, 16.5]]))
    cropped = xy - torch.tensor([4.0, 3.0])
    ramp = torch.arange(400, dtype=torch.float32).reshape(1, 1, 20, 20)
    assert torch.allclose(sample_logits(ramp[:, :, 3:, 4:], cropped), sample_logits(ramp, xy))


def test_ignore_keypoints_have_zero_loss_gradient():
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    loss = sparse_bce(logits, torch.tensor([[0.0, 0.0], [1.0, 1.0]]), torch.tensor([-1.0, 1.0]))
    loss.backward()
    assert logits.grad[0, 0, 0, 0] == 0
    assert logits.grad[0, 0, 1, 1] < 0


def test_all_ignore_loss_can_backpropagate():
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    loss = sparse_bce(logits, torch.tensor([[0.0, 0.0]]), torch.tensor([-1.0]))
    loss.backward()
    assert loss == 0 and torch.count_nonzero(logits.grad) == 0


def test_metrics_exclude_ignore():
    assert metrics([0, 1, -1], [0.1, 0.9, 0.5])["samples"] == 2


def test_no_split_straddling_pairs():
    groups = {"train": ["a", "b"], "validation": ["c"], "test": ["d", "e"], "excluded": ["x"]}
    assert same_slab_pairs([("a", "b"), ("a", "d"), ("d", "e"), ("x", "x")], groups) == [
        ("a", "b"),
        ("d", "e"),
    ]


def test_eval_leakage_and_outside_slab():
    meta = {"training_images": ["a"], "validation_images": ["b"]}
    split = {"groups": {"test": ["c"]}}
    check_evaluation(meta, ["c"], split)
    for name in ["a", "b", "d"]:
        with pytest.raises(ValueError):
            check_evaluation(meta, [name], split)


def test_matchability_features_endpoint_symmetry():
    a, b = {"matchability": np.array([0.2, 0.8])}, {"matchability": np.array([0.5, 0.7])}
    ids = np.array([[0, 1], [1, 0]])
    forward = matchability_features(a, b, ids)
    reverse = matchability_features(b, a, ids[:, ::-1])
    assert np.allclose(forward[:, :2], reverse[:, [1, 0]])
    assert np.array_equal(forward[:, 2:], reverse[:, 2:])


def test_context_16_permutation_equivariance():
    model = MatchFilter(channels=16).eval()
    x = torch.randn(12, 16)
    order = torch.randperm(12)
    assert torch.allclose(model(x)[order], model(x[order]), atol=1e-6)


def test_filter_feature_and_hash_rejection(tmp_path):
    model = MatchFilter(channels=16)
    checkpoint = {
        "state_dict": model.state_dict(),
        "architecture": "context",
        "feature_spec": "match_matchability_v3_16",
        "matchability_hash": "abc",
        "mean": [0.0] * 16,
        "std": [1.0] * 16,
        "temperature": 1.0,
    }
    path = tmp_path / "filter.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="specification"):
        MatchPredictor(path)
    with pytest.raises(ValueError, match="hash"):
        MatchPredictor(path, matchability=True, matchability_hash="wrong")
    assert (
        len(
            MatchPredictor(path, matchability=True, matchability_hash="abc")(
                np.zeros((3, 16), np.float32)
            )
        )
        == 3
    )


def test_temperature_preserves_rank():
    z, y = np.array([-2.0, 1.0, 3.0, -1.0]), np.array([0, 1, 1, 0])
    t = fit_temperature(z, y)
    assert t > 0
    assert metrics(y, probabilities(z))["roc_auc"] == metrics(y, probabilities(z, t))["roc_auc"]


def test_selection_exact_count_and_provenance():
    record = {
        "xy": np.zeros((5, 2)),
        "local": np.ones((5, 128)),
        "shape": np.array([5, 5]),
        "matchability": np.array([0.2, 0.9, 0.7, 0.1, 0.4]),
        "source_indices": np.arange(5) + 10,
    }
    selected = select_keypoints(record, 3)
    assert len(selected["xy"]) == 3
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


def test_reliability_source_is_detachable():
    record = {"quality": np.array([0.2]), "matchability": np.array([0.8])}
    assert apply_source(record)["reliability"][0] == 0.2
    assert apply_source(record, "matchability")["reliability"][0] == 0.8
    assert "reliability" not in record


def test_v3_runs_are_immutable(tmp_path):
    path = tmp_path / "trial_v3"
    new_run(path)
    with pytest.raises(FileExistsError):
        new_run(path)
    with pytest.raises(ValueError):
        new_run(tmp_path / "bad")


def test_bootstrap_is_paired_and_clustered():
    rows = [
        {"labels": np.array([0, 1]), "good": np.array([0.1, 0.9]), "bad": np.array([0.9, 0.1])}
        for _ in range(3)
    ]
    report = paired_ap_bootstrap(rows, "good", "bad", repeats=20)
    assert report["clusters"] == 3
    assert report["ci95"][0] > 0


def test_segformer_one_channel_roundtrip_and_onnx(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    cfg = {
        "kind": "segformer_b0",
        "hf_config": {
            "depths": [1, 1, 1, 1],
            "hidden_sizes": [8, 16, 32, 64],
            "num_attention_heads": [1, 2, 4, 8],
            "decoder_hidden_size": 16,
        },
    }
    model = MatchabilityModel(cfg, pretrained=False).eval()
    image = torch.rand(1, 3, 64, 64)
    out = model(image)
    assert out.shape == (1, 1, 64, 64)
    path = tmp_path / "model.onnx"
    torch.onnx.export(
        model,
        image,
        str(path),
        dynamo=False,
        opset_version=17,
        input_names=["rgb"],
        output_names=["logits"],
        dynamic_axes={"rgb": {0: "batch"}, "logits": {0: "batch"}},
    )
    session = ort.InferenceSession(str(path))
    actual = session.run(None, {"rgb": image.numpy()})[0]
    assert np.allclose(out.detach().numpy(), actual, atol=1e-4, rtol=1e-3)
    assert session.run(None, {"rgb": np.repeat(image.numpy(), 2, 0)})[0].shape[0] == 2


def test_exact_bootstrap_with_small_score_range():
    rows = [
        {
            "labels": np.array([0, 1]),
            "good": np.array([0.00001, 0.00002]),
            "bad": np.array([0.00002, 0.00001]),
        }
        for _ in range(3)
    ]
    result = paired_ap_bootstrap(rows, "good", "bad", repeats=20)
    assert np.allclose(result["ci95"], [0.5, 0.5])


def test_retention_averages_score_ties():
    result = metrics([1, 0, 1, 0], [0.5] * 4)
    assert result["retention"]["positive_recall"][0] == 0.25


def test_fast_ause_preserves_v1_definition():
    from underwater_vision.evaluation.uncertainty import uncertainty_metrics
    from underwater_vision.matchability.confidence import fast_ause

    error, uncertainty = np.array([1.0, 4.0, 2.0, 3.0]), np.array([1.0, 1.0, 3.0, 2.0])
    assert np.isclose(
        fast_ause(error, uncertainty), uncertainty_metrics(error, uncertainty)["ause"]
    )


def test_track_matchability_excludes_heldout_views():
    from underwater_vision.geometry.reference import project
    from underwater_vision.matchability.confidence import holdout_rows

    K = np.array([[500.0, 0, 100.0], [0, 500.0, 100.0], [0, 0, 1.0]])
    poses = []
    for x in [0.0, 0.5, 1.0, 1.5]:
        pose = np.eye(4)
        pose[0, 3] = x
        poses.append(pose)
    point = np.array([0.7, 0.0, 5.0])
    xy = [project(point, p, K) for p in poses]
    nodes = [(str(i), i) for i in range(4)]
    m = np.array([0.1, 0.3, 0.7, 0.9])
    rows = holdout_rows(nodes, poses, xy, np.ones((4, 2)), np.ones(4), m, K)
    for row in rows:
        used = row["observed_indices"]
        assert row["x"].shape == (10,)
        assert np.allclose(row["x"][-2:], [m[used].mean(), m[used].min()])
        assert not set(used) & set(row["heldout_indices"])


def test_matchability_api_png_stats_and_bad_upload():
    import base64
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from underwater_vision.serving.matchability_api import create_matchability_app

    predictor = type(
        "P",
        (),
        {
            "checkpoint_hash": "x",
            "temperature": 1.0,
            "predict": lambda _self, rgb: np.full(rgb.shape[:2], 0.75, np.float32),
        },
    )()
    client = TestClient(create_matchability_app(predictor))
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(buffer, format="PNG")
    response = client.post(
        "/predict", files={"image": ("image.png", buffer.getvalue(), "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    array = np.asarray(Image.open(io.BytesIO(base64.b64decode(data["matchability_png_base64"]))))
    assert array.shape == (20, 30) and np.max(np.abs(array / 65535 - 0.75)) < 1 / 65535
    assert (
        client.post("/predict", files={"image": ("bad.png", b"bad", "image/png")}).status_code
        == 400
    )


def test_v1_compatibility_measured_artifact_if_available():
    from pathlib import Path

    from underwater_vision.matchability.protocol import read_json

    path = Path("outputs/v1_compatibility_v3/metrics.json")
    if not path.exists():
        pytest.skip("Real COLMAP experiment requires the local dataset and v1 checkpoint")
    metric = read_json(path)
    assert metric["registered_images"] == 16 and metric["sparse_points"] == 1890
    assert abs(metric["mean_reprojection_error_px"] - 0.4504205128199975) < 1e-12


def test_frozen_cache_preserves_fp32_values_strides_and_reuses_encoder(tmp_path, monkeypatch):
    from underwater_vision.matchability.frozen_cache import FrozenEvaluationCache

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "image.png"
    source.write_bytes(b"source state for fingerprint")

    class Encoder:
        specification = {"revision": "test"}
        calls = 0

        def frozen_grid(self, rgb):
            self.calls += 1
            return torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4).permute(0, 3, 1, 2) / 7

    encoder = Encoder()
    cache = FrozenEvaluationCache(maximum_bytes=100_000)
    first = cache.grid(encoder, torch.zeros(1, 3, 20, 20), source)
    second = cache.grid(encoder, torch.zeros(1, 3, 20, 20), source)
    assert encoder.calls == 1 and torch.equal(first, second)
    assert first.stride() == second.stride() and second.dtype == torch.float32
    source.write_bytes(b"changed source")
    cache.grid(encoder, torch.zeros(1, 3, 20, 20), source)
    assert encoder.calls == 2
