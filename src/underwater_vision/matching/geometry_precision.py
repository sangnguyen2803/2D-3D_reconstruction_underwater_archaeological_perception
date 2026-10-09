"""Conditional localization-precision ranking on the classical match pool."""

import cv2
import numpy as np
import torch

from underwater_vision.matching.learned_filter import MatchFilter

FEATURES = ["distance", "ratio", "quality_a", "quality_b", "ax", "ay", "bx", "by", "dx", "dy"]


def precision_features(first, second, pairs, width=960, height=720, ratios=None):
    """Features for already eligible mutual ratio matches, without DINO.

    If ratios are unavailable, recompute nearest-two distances for only the
    eligible queries and verify alignment with the immutable match pool.
    """
    pairs = np.asarray(pairs)
    if not len(pairs):
        return np.empty((0, len(FEATURES)), np.float32)
    da, db = first["local"][pairs[:, 0]], second["local"][pairs[:, 1]]
    distance = np.linalg.norm(da - db, axis=1)
    if ratios is None:
        neighbors = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
            np.ascontiguousarray(da), second["local"], k=2
        )
        if not np.array_equal([m.trainIdx for m, _ in neighbors], pairs[:, 1]):
            raise ValueError("Frozen pool and descriptor nearest-neighbor alignment differ")
        ratios = np.array([m.distance / max(n.distance, 1e-9) for m, n in neighbors])
    a, b = first["xy"][pairs[:, 0]] / [width, height], second["xy"][pairs[:, 1]] / [width, height]
    return np.c_[
        distance, ratios, first["quality"][pairs[:, 0]], second["quality"][pairs[:, 1]], a, b, b - a
    ].astype(np.float32)


def augment_coordinates(x, turns=0, flip=False):
    """Joint image-axis relabeling preserves pair geometry and target errors."""
    x = x.clone()
    for _ in range(turns % 4):
        old = x.clone()
        x[:, 4], x[:, 5] = 1 - old[:, 5], old[:, 4]
        x[:, 6], x[:, 7] = 1 - old[:, 7], old[:, 6]
        x[:, 8], x[:, 9] = -old[:, 9], old[:, 8]
    if flip:
        x[:, 4], x[:, 6], x[:, 8] = 1 - x[:, 4], 1 - x[:, 6], -x[:, 8]
    return x


class PrecisionPredictor:
    def __init__(self, checkpoint, device="cpu"):
        self.metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            self.metadata.get("feature_spec") != "conditional_geometry_v1_10"
            or self.metadata["feature_names"] != FEATURES
        ):
            raise ValueError("Expected conditional geometry-precision feature specification")
        self.device = torch.device(device)
        self.model = (
            MatchFilter(self.metadata["architecture"], channels=len(FEATURES))
            .to(self.device)
            .eval()
        )
        self.model.load_state_dict(self.metadata["state_dict"])
        self.mean = np.asarray(self.metadata["mean"], np.float32)
        self.std = np.asarray(self.metadata["std"], np.float32)

    @torch.inference_mode()
    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
            raise ValueError("Expected finite conditional-precision features")
        return (
            torch.sigmoid(self.model(torch.from_numpy((x - self.mean) / self.std).to(self.device)))
            .cpu()
            .numpy()
        )
