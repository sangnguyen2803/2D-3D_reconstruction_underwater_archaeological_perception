"""Validation-only binary temperature scaling; no test-set fitting."""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit


def fit_temperature(logits, labels):
    z, y = np.asarray(logits, float), np.asarray(labels, float)
    known = y >= 0
    z, y = z[known], y[known]
    if not len(y) or len(np.unique(y)) < 2 or not np.isfinite(z).all():
        raise ValueError("Calibration needs finite validation logits and both classes")

    def nll(log_temperature):
        scaled = z / np.exp(log_temperature)
        return np.mean(np.logaddexp(0, scaled) - y * scaled)

    result = minimize_scalar(nll, bounds=(-4, 4), method="bounded")
    return float(np.exp(result.x))


def probabilities(logits, temperature=1):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return expit(np.asarray(logits, float) / temperature).astype(np.float32)


def fit_platt(logits, labels):
    """Fit a positive logit slope and intercept on validation data only.

    Positive slope preserves the ordering of the original model scores. The
    intercept can correct a prior shift that temperature scaling cannot express.
    No class balancing is applied to the calibration likelihood.
    """
    z, y = np.asarray(logits, float), np.asarray(labels, float)
    if z.shape != y.shape:
        raise ValueError("Calibration logits and labels must have matching shapes")
    known = y >= 0
    z, y = z[known], y[known]
    if (
        not len(y)
        or not np.isfinite(z).all()
        or not np.isin(y, [0, 1]).all()
        or len(np.unique(y)) < 2
    ):
        raise ValueError("Calibration needs finite logits and both binary classes")

    def objective(theta):
        slope = np.exp(theta[0])
        scaled = slope * z + theta[1]
        residual = expit(scaled) - y
        loss = np.mean(np.logaddexp(0, scaled) - y * scaled)
        gradient = np.array([np.mean(residual * slope * z), np.mean(residual)])
        return loss, gradient

    temperature = fit_temperature(z, y)
    result = minimize(
        objective,
        [-np.log(temperature), 0.0],
        jac=True,
        method="L-BFGS-B",
        bounds=[(-6, 6), (-20, 20)],
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f"Platt calibration failed: {result.message}")
    return {
        "method": "positive_platt",
        "slope": float(np.exp(result.x[0])),
        "intercept": float(result.x[1]),
        "validation_nll": float(result.fun),
        "fitted_on": "validation_only",
    }


def calibrated_probabilities(logits, calibration):
    """Apply a standalone calibrator without changing network weights."""
    if calibration.get("method") != "positive_platt":
        raise ValueError("Expected a positive_platt calibration")
    slope, intercept = calibration["slope"], calibration["intercept"]
    if not np.isfinite(slope) or slope <= 0 or not np.isfinite(intercept):
        raise ValueError("Calibration needs a finite positive slope and finite intercept")
    return expit(slope * np.asarray(logits, float) + intercept)
