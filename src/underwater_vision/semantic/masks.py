"""Pure array functions usable without models, COLMAP or the pipeline."""

import numpy as np
import torch
import torch.nn.functional as F

from underwater_vision.semantic.schema import DYNAMIC


def sample_probabilities(probability, xy):
    probability = np.asarray(probability, np.float32)
    if probability.ndim != 3 or probability.shape[0] != 6:
        raise ValueError("Expected 6xHxW class probabilities")
    height, width = probability.shape[1:]
    grid = np.asarray(xy, np.float32).copy().reshape(-1, 2)
    if not len(grid):
        return np.empty((0, 6), np.float32), np.empty(0, np.float32)
    grid = 2 * (grid + 0.5) / [width, height] - 1
    p = F.grid_sample(
        torch.from_numpy(probability)[None],
        torch.from_numpy(grid.astype(np.float32))[None, :, None],
        align_corners=False,
        padding_mode="border",
    )[0, :, :, 0].T.numpy()
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1)
    return p, entropy


def attach_probabilities(record, probability, checkpoint_hash):
    result = dict(record)
    result["class_probabilities"], result["class_entropy"] = sample_probabilities(
        probability, record["xy"]
    )
    result["segmentation_hash"] = np.asarray(checkpoint_hash)
    result["source_indices"] = np.arange(len(record["xy"]), dtype=np.uint32)
    return result


def mask_keypoints(record, threshold):
    if not 0 < threshold <= 1:
        raise ValueError("Dynamic threshold must be in (0,1]")
    keep = record["class_probabilities"][:, DYNAMIC] < threshold
    point_arrays = {
        "xy",
        "local",
        "semantic",
        "quality",
        "reliability",
        "scales",
        "oris",
        "geometry_xy",
        "class_probabilities",
        "class_entropy",
        "source_indices",
    }
    return {k: v[keep] if k in point_arrays else v for k, v in record.items()}


def match_class_features(a, b, pairs, mode="compact"):
    pa, pb = a["class_probabilities"][pairs[:, 0]], b["class_probabilities"][pairs[:, 1]]
    compact = np.c_[
        np.sum(pa * pb, axis=1),
        np.maximum(pa[:, DYNAMIC], pb[:, DYNAMIC]),
        a["class_entropy"][pairs[:, 0]],
        b["class_entropy"][pairs[:, 1]],
    ]
    if mode == "compact":
        return compact.astype(np.float32)
    if mode == "full":
        return np.c_[compact, pa, pb].astype(np.float32)
    raise ValueError("Semantic feature mode must be compact or full")


def feature_names(mode):
    from underwater_vision.matching.learned_filter import FEATURE_NAMES

    if mode == "off":
        return list(FEATURE_NAMES)
    names = [*FEATURE_NAMES, "same_class", "dynamic_max", "entropy_a", "entropy_b"]
    if mode == "full":
        names += [f"class_{end}_{i}" for end in ["a", "b"] for i in range(6)]
    elif mode != "compact":
        raise ValueError("Unknown feature mode")
    return names
