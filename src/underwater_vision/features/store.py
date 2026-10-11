"""Reproducible local v1 features; preserve SIFT scales for the LightGlue baseline."""

from pathlib import Path

import cv2
import numpy as np

from underwater_vision.preprocessing.images import resize_rgb
from underwater_vision.utils.io import atomic_npz, fingerprint, read_rgb
from underwater_vision.visibility.deterministic import estimate_visibility, sample_map


def feature_record(
    frame,
    neural,
    max_size=960,
    max_features=4096,
    semantic_provider=None,
    matchability_provider=None,
    contrast_threshold=0.04,
):
    if not np.isfinite(contrast_threshold) or contrast_threshold <= 0:
        raise ValueError("SIFT contrast threshold must be finite and positive")
    if matchability_provider is not None:
        from underwater_vision.matchability.infer import attach_matchability

        base = feature_record(
            frame,
            neural,
            max_size,
            max_features,
            semantic_provider,
            contrast_threshold=contrast_threshold,
        )
        return attach_matchability(
            base,
            matchability_provider.cached(frame.path, max_size),
            matchability_provider.checkpoint_hash,
        )
    if semantic_provider is not None:
        from underwater_vision.semantic.masks import attach_probabilities

        base = feature_record(
            frame, neural, max_size, max_features, contrast_threshold=contrast_threshold
        )
        return attach_probabilities(
            base, semantic_provider.cached(frame.path, max_size), semantic_provider.checkpoint_hash
        )
    root = Path("cache/research_features_v1")
    root.mkdir(parents=True, exist_ok=True)
    settings = {
        "v1": 1,
        "max_size": max_size,
        "max_features": max_features,
        "backbone": neural.provenance["backbone"],
        "revision": neural.revision,
    }
    if contrast_threshold != 0.04:
        settings["contrast_threshold"] = contrast_threshold
    key = fingerprint(frame.path, settings)
    path = root / (key + ".npz")
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    image, _ = resize_rgb(read_rgb(frame.path), max_size)
    keypoints, raw = cv2.SIFT_create(
        nfeatures=max_features, contrastThreshold=contrast_threshold
    ).detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
    xy = np.asarray([k.pt for k in keypoints], np.float32).reshape(-1, 2)
    raw = np.empty((0, 128), np.float32) if raw is None else raw
    local = np.sqrt(raw / (raw.sum(1, keepdims=True) + 1e-12)).astype(np.float32)
    visibility, _ = estimate_visibility(image)
    record = {
        "xy": xy,
        "local": local,
        "semantic": neural.extract(image, xy),
        "quality": sample_map(visibility, xy),
        "reliability": sample_map(visibility, xy),
        "scales": np.asarray([k.size for k in keypoints], np.float32),
        "oris": np.asarray([np.deg2rad(k.angle) for k in keypoints], np.float32),
        "shape": np.asarray(image.shape[:2], np.int32),
    }
    atomic_npz(path, **record)
    return record
