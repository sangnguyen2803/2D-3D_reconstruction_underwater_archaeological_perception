"""Optional class-context point evidence, robust error loss and conformal bands."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from underwater_vision.pointcloud.learned_confidence import track_features


def point_class_features(probabilities):
    p = np.asarray(probabilities, np.float32).reshape(-1, 6)
    dominant = int(p.mean(0).argmax())
    onehot = np.eye(6, dtype=np.float32)[dominant]
    agreement = np.mean(p.argmax(1) == dominant)
    entropy = np.mean(-np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1))
    return np.r_[onehot, agreement, entropy, p[:, 5].max()].astype(np.float32)


def multi_holdout_rows(nodes, poses, xy, semantic, quality, probabilities, K, max_holdouts=3):
    """Keep one/two views out, preserving >=2 fit views; features never see them."""
    import hashlib

    from underwater_vision.geometry.reference import project, triangulate_observations

    seed = int.from_bytes(hashlib.sha256(repr(nodes).encode()).digest()[:4], "little")
    order = np.random.default_rng(seed).permutation(len(nodes))
    held_size = min(2, len(nodes) - 2)
    if held_size < 1:
        return []
    rows = []
    for turn in range(min(max_holdouts, len(nodes))):
        held = np.roll(order, turn)[:held_size].tolist()
        used = [i for i in range(len(nodes)) if i not in held]
        fit_poses = [poses[i] for i in used]
        point = triangulate_observations(fit_poses, np.asarray(xy)[used], K)
        base, heuristic = track_features(
            point,
            fit_poses,
            np.asarray(xy)[used],
            np.asarray(semantic)[used],
            np.asarray(quality)[used],
            K,
        )
        errors = [np.linalg.norm(project(point, poses[i], K) - xy[i]) for i in held]
        # Median of independently excluded projections is a robust track target.
        error = float(np.median(errors))
        row = np.r_[base, point_class_features(np.asarray(probabilities)[used])].astype(np.float32)
        if np.isfinite(error) and np.isfinite(row).all():
            rows.append(
                {
                    "x": row,
                    "error": error,
                    "heuristic": heuristic,
                    "xyz": point,
                    "heldout_indices": held,
                    "observed_indices": used,
                    "track_group": seed,
                }
            )
    return rows


class RobustErrorRegressor(nn.Module):
    def __init__(self, channels=17):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(channels, 48), nn.GELU(), nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 2)
        )

    def forward(self, x):
        result = self.network(x)
        return result[:, 0], result[:, 1].clamp(-6, 3)


def robust_loss(mu, log_scale, target, kind="student_t"):
    if kind == "huber":
        # Huber estimates the center; observed validation residuals calibrate bands.
        return F.huber_loss(mu, target, delta=0.5)
    if kind != "student_t":
        raise ValueError("Robust loss must be huber or student_t")
    distribution = torch.distributions.StudentT(df=4.0, loc=mu, scale=torch.exp(log_scale))
    return -distribution.log_prob(target).mean()


def conformal_quantile(errors, prediction, coverage=0.9):
    """Finite-sample split-conformal residual quantile, in log1p error units.

    Call with one aggregated residual per validation track, not correlated rows.
    Nominal coverage requires exchangeability and is not guaranteed under shift.
    """
    errors, prediction = np.asarray(errors), np.asarray(prediction)
    if not len(errors) or errors.shape != prediction.shape:
        raise ValueError("Aligned validation track errors/predictions required")
    residual = np.abs(np.log1p(errors) - prediction)
    rank = int(np.ceil((len(residual) + 1) * coverage))
    if rank > len(residual):
        return float("inf")
    return float(np.partition(residual, rank - 1)[rank - 1])


class SemanticErrorPredictor:
    def __init__(self, path, segmentation_hash):
        self.metadata = torch.load(path, map_location="cpu", weights_only=True)
        if (
            self.metadata.get("feature_spec") != "point_class_v2_17"
            or self.metadata.get("segmentation_hash") != segmentation_hash
        ):
            raise ValueError("Point feature specification or segmentation hash mismatch")
        self.model = RobustErrorRegressor().eval()
        self.model.load_state_dict(self.metadata["state_dict"])

    @torch.inference_mode()
    def __call__(self, x):
        x = (x - np.asarray(self.metadata["mean"])) / np.asarray(self.metadata["std"])
        mu, _ = self.model(torch.from_numpy(x.astype(np.float32)))
        mu = mu.numpy()
        radius = self.metadata["conformal_radius"]
        error = np.maximum(np.expm1(np.clip(mu, -20, 15)), 0)
        return {
            "predicted_error": error,
            "lower90": np.maximum(np.expm1(np.clip(mu - radius, -20, 15)), 0),
            "upper90": np.maximum(np.expm1(np.clip(mu + radius, -20, 15)), 0),
            "confidence": 1 / (1 + error),
        }
