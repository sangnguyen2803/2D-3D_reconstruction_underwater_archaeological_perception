"""Leave-one-view-out evidence and log-error distribution prediction."""

import hashlib

import numpy as np
import torch
from torch import nn

from underwater_vision.geometry.reference import project, triangulate_observations

CONFIDENCE_FEATURES = [
    "log_views",
    "max_angle",
    "mean_fit_error",
    "mean_quality",
    "min_quality",
    "dino_consistency",
    "dino_variation",
    "min_angle",
]


def track_features(point, poses, xy, semantic, quality, K):
    centers = np.stack([p[:3, 3] for p in poses])
    rays = centers - point
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    angles = np.rad2deg(np.arccos(np.clip(rays @ rays.T, -1, 1)))
    upper = np.triu_indices(len(rays), 1)
    maximum = float(angles[upper].max()) if len(upper[0]) else 0
    minimum = float(angles[upper].min()) if len(upper[0]) else 0
    errors = [
        np.linalg.norm(project(point, pose, K) - obs) for pose, obs in zip(poses, xy, strict=True)
    ]
    semantic = np.asarray(semantic)
    cosine = semantic @ semantic.T
    consistency = float(cosine[upper].mean()) if len(upper[0]) else 0
    fit = float(np.mean(errors))
    q = float(np.mean(quality))
    x = np.asarray(
        [
            np.log1p(len(poses)),
            maximum,
            fit,
            q,
            np.min(quality),
            consistency,
            np.mean(np.var(semantic, axis=0)),
            minimum,
        ],
        np.float32,
    )
    heuristic = (
        (1 - np.exp(-(len(poses) - 1) / 3)) * np.exp(-fit / 2) * (1 - np.exp(-maximum / 5)) * q
    )
    return x, float(heuristic)


def holdout_observation(nodes, poses, xy, semantic, quality, K):
    """Neither held-out xy, semantic nor quality enters predictor inputs."""
    if len(nodes) < 3:
        raise ValueError("Holdout error needs at least three views")
    digest = hashlib.sha256(repr(nodes).encode()).digest()
    held = int.from_bytes(digest[:4], "little") % len(nodes)
    used = [i for i in range(len(nodes)) if i != held]
    fit_poses = [poses[i] for i in used]
    fit_xy = np.asarray(xy)[used]
    point = triangulate_observations(fit_poses, fit_xy, K)
    prediction = project(point, poses[held], K)
    error = float(np.linalg.norm(prediction - xy[held]))
    x, heuristic = track_features(
        point,
        fit_poses,
        fit_xy,
        np.asarray(semantic)[used],
        np.asarray(quality)[used],
        K,
    )
    if not np.isfinite(error) or not np.isfinite(x).all():
        raise ValueError("Invalid cheirality/geometry in holdout track")
    return x, error, heuristic, point, held


class ErrorRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 48), nn.GELU(), nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 2)
        )

    def forward(self, x):
        out = self.network(x)
        return out[:, 0], out[:, 1].clamp(-6, 3)


def export_learned_confidence(reconstruction, features, output, checkpoint):
    """Production features use estimated cameras; no reference poses or errors."""
    from pathlib import Path

    from underwater_vision.pointcloud.confidence import write_ply
    from underwater_vision.utils.io import atomic_npz, write_json

    rows, xyz, rgb, point_ids = [], [], [], []
    for pid, point in reconstruction.points3D.items():
        poses, xy, semantic, quality = [], [], [], []
        for element in point.track.elements:
            image = reconstruction.images[element.image_id]
            feature = features[image.name]
            pose = np.eye(4)
            pose[:3] = image.cam_from_world().inverse().matrix()
            poses.append(pose)
            xy.append(feature["geometry_xy"][element.point2D_idx])
            semantic.append(feature["semantic"][element.point2D_idx])
            quality.append(feature["quality"][element.point2D_idx])
            K = reconstruction.cameras[image.camera_id].calibration_matrix().copy()
            K[:2, 2] -= 0.5
        row, _ = track_features(point.xyz, poses, xy, semantic, quality, K)
        if not np.isfinite(row).all():
            continue
        rows.append(row)
        xyz.append(point.xyz)
        rgb.append(point.color)
        point_ids.append(int(pid))
    prediction = ErrorPredictor(checkpoint)(np.asarray(rows, np.float32).reshape(-1, 8))
    write_ply(
        Path(output) / "sparse_learned_confidence_v1.ply",
        np.asarray(xyz),
        np.asarray(rgb),
        prediction["confidence"],
    )
    atomic_npz(
        Path(output) / "sparse_learned_confidence_v1.npz",
        xyz=np.asarray(xyz),
        rgb=np.asarray(rgb),
        point_ids=np.asarray(point_ids),
        **prediction,
    )
    write_json(
        Path(output) / "learned_confidence_v1.json",
        {
            "points": len(rows),
            "checkpoint": str(checkpoint),
            "interpretation": "predicted held-out reprojection-error proxy; ranking and intervals require held-out validation",
            "reference_poses_used_for_prediction": False,
        },
    )


class ErrorPredictor:
    def __init__(self, path):
        self.metadata = torch.load(path, map_location="cpu", weights_only=True)
        self.model = ErrorRegressor().eval()
        self.model.load_state_dict(self.metadata["state_dict"])

    @torch.inference_mode()
    def __call__(self, x):
        x = (x - np.asarray(self.metadata["mean"])) / np.asarray(self.metadata["std"])
        mu, log_var = self.model(torch.from_numpy(x.astype(np.float32)))
        scale = self.metadata.get("validation_scale", 1)
        sigma = torch.exp(log_var / 2)
        expectation = (torch.exp((mu + torch.exp(log_var) / 2).clamp(max=15)) - 1).clamp(
            min=0
        ) * scale
        low = (torch.exp((mu - 1.645 * sigma).clamp(max=15)) - 1).clamp(min=0) * scale
        high = (torch.exp((mu + 1.645 * sigma).clamp(max=15)) - 1).clamp(min=0) * scale
        return {
            "predicted_error": expectation.numpy(),
            "lower90": low.numpy(),
            "upper90": high.numpy(),
            "confidence": (1 / (1 + expectation)).numpy(),
        }
