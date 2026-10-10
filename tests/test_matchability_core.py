"""Focused matchability core contracts, independent of pipeline integration."""

import numpy as np
import pytest
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score

from underwater_vision.evaluation.matchability import metrics, paired_ap_bootstrap
from underwater_vision.matchability.calibration import (
    calibrated_probabilities,
    fit_platt,
    fit_temperature,
    probabilities,
)
from underwater_vision.matchability.labels import (
    aggregate_labels,
    eligible_pair,
    same_slab_pairs,
    stronger_labels,
)
from underwater_vision.matchability.loss import resize_coordinates, sample_logits, sparse_bce
from underwater_vision.matchability.model import MatchabilityModel
from underwater_vision.matchability.protocol import check_evaluation, new_run


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


def test_v3_runs_are_immutable(tmp_path):
    path = tmp_path / "trial_v3"
    new_run(path)
    with pytest.raises(FileExistsError):
        new_run(path)
    with pytest.raises(ValueError):
        new_run(tmp_path / "bad")


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


def test_validation_prior_shift_and_preserved_ranking():
    rng = np.random.default_rng(21)
    logits = rng.normal(size=30000)
    labels = rng.binomial(1, expit(1.4 * logits + 1.3))
    fit_z, test_z = logits[:20000], logits[20000:]
    fit_y, test_y = labels[:20000], labels[20000:]
    calibration = fit_platt(fit_z, fit_y)
    temperature = fit_temperature(fit_z, fit_y)
    expected = expit(1.4 * test_z + 1.3)
    calibrated = calibrated_probabilities(test_z, calibration)
    assert np.mean((calibrated - expected) ** 2) < np.mean(
        (expit(test_z / temperature) - expected) ** 2
    )
    assert average_precision_score(test_y, calibrated) == average_precision_score(test_y, test_z)
    assert np.array_equal(np.argsort(calibrated), np.argsort(test_z))


def test_ignored_labels_and_invalid_calibrators():
    z = np.array([-2, -1, 0, 1, 2, np.nan])
    y = np.array([0, 1, 0, 1, 1, -1])
    assert fit_platt(z, y) == fit_platt(z[:-1], y[:-1])
    with pytest.raises(ValueError):
        fit_platt([np.nan, 0], [0, 1])
    with pytest.raises(ValueError):
        fit_platt([0, 1], [1, 1])
    with pytest.raises(ValueError):
        calibrated_probabilities(z, {"method": "positive_platt", "slope": -1, "intercept": 0})


def test_metrics_exclude_ignore():
    assert metrics([0, 1, -1], [0.1, 0.9, 0.5])["samples"] == 2


def test_temperature_preserves_rank():
    z, y = np.array([-2.0, 1.0, 3.0, -1.0]), np.array([0, 1, 1, 0])
    t = fit_temperature(z, y)
    assert t > 0
    assert metrics(y, probabilities(z))["roc_auc"] == metrics(y, probabilities(z, t))["roc_auc"]


def test_bootstrap_is_paired_and_clustered():
    rows = [
        {"labels": np.array([0, 1]), "good": np.array([0.1, 0.9]), "bad": np.array([0.9, 0.1])}
        for _ in range(3)
    ]
    report = paired_ap_bootstrap(rows, "good", "bad", repeats=20)
    assert report["clusters"] == 3
    assert report["ci95"][0] > 0


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
