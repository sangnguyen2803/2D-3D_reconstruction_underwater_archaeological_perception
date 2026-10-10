import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import average_precision_score

from underwater_vision.matchability.calibration import (
    calibrated_probabilities,
    fit_platt,
    fit_temperature,
)


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


def test_optional_predictor_calibration_and_checkpoint_guard(tmp_path):
    import json

    from underwater_vision.matchability.infer import MatchabilityPredictor

    predictor = MatchabilityPredictor.__new__(MatchabilityPredictor)
    predictor.checkpoint_hash = "checkpoint-A"
    predictor.temperature = 2.0
    logits = np.array([[-1.0, 0.0, 1.0]])
    predictor.logits = lambda rgb: logits
    predictor._set_calibration(None)
    np.testing.assert_allclose(predictor.predict(None), expit(logits / 2))
    sidecar = tmp_path / "calibration.json"
    calibration = {
        "method": "positive_platt",
        "slope": 1.2,
        "intercept": 0.7,
        "fitted_on": "validation_only",
        "checkpoint_hash": "checkpoint-A",
    }
    sidecar.write_text(json.dumps(calibration))
    predictor._set_calibration(sidecar)
    np.testing.assert_allclose(predictor.predict(None), expit(1.2 * logits + 0.7))
    assert predictor.calibration_hash is not None
    calibration["checkpoint_hash"] = "checkpoint-B"
    sidecar.write_text(json.dumps(calibration))
    with pytest.raises(ValueError, match="another checkpoint"):
        predictor._set_calibration(sidecar)
