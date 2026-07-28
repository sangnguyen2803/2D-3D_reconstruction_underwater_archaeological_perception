"""Multi-view semantic fusion and genuinely excluded-image label prediction."""

import numpy as np

from underwater_vision.evaluation.segmentation import (
    confusion,
    paired_bootstrap,
    segmentation_metrics,
)


def fuse_probabilities(probabilities, image_ids=None, exclude_image=None, method="mean"):
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1, 6)
    if exclude_image is not None:
        if image_ids is None or len(image_ids) != len(p):
            raise ValueError("Holdout fusion requires aligned image ids")
        p = p[np.asarray(image_ids) != exclude_image]
    if not len(p):
        raise ValueError("No observed views remain after holdout")
    if method == "mean":
        fused = p.mean(0)
    elif method == "log":
        log = np.log(np.maximum(p, 1e-9)).sum(0)
        fused = np.exp(log - log.max())
    else:
        raise ValueError("Fusion must be mean or log")
    fused /= fused.sum()
    label = int(fused.argmax())
    agreement = float(np.mean(p.argmax(1) == label))
    return fused, label, agreement


def export_semantic_cloud(reconstruction, features, output, method="mean"):
    from pathlib import Path

    from plyfile import PlyData, PlyElement

    from underwater_vision.semantic.schema import COLORS
    from underwater_vision.utils.io import atomic_npz, write_json

    xyz, labels, agreement, probability, ids, errors = [], [], [], [], [], []
    for pid, point in reconstruction.points3D.items():
        p = [
            features[reconstruction.images[e.image_id].name]["class_probabilities"][e.point2D_idx]
            for e in point.track.elements
        ]
        fused, label, agree = fuse_probabilities(p, method=method)
        xyz.append(point.xyz)
        labels.append(label)
        agreement.append(agree)
        probability.append(fused)
        ids.append(pid)
        errors.append(float(point.error))
    xyz = np.asarray(xyz).reshape(-1, 3)
    labels = np.asarray(labels, np.uint8)
    agreement = np.asarray(agreement, np.float32)
    rows = np.empty(
        len(xyz),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("class_id", "u1"),
            ("agreement", "f4"),
        ],
    )
    for i, n in enumerate(["x", "y", "z"]):
        rows[n] = xyz[:, i]
    for i, n in enumerate(["red", "green", "blue"]):
        rows[n] = COLORS[labels, i]
    rows["class_id"], rows["agreement"] = labels, agreement
    PlyData([PlyElement.describe(rows, "vertex")], text=False).write(
        str(Path(output) / "semantic_sparse_v2.ply")
    )
    atomic_npz(
        Path(output) / "semantic_sparse_v2.npz",
        xyz=xyz,
        class_id=labels,
        agreement=agreement,
        probabilities=np.asarray(probability).reshape(-1, 6),
        point_ids=np.asarray(ids),
    )
    write_json(
        Path(output) / "semantic_voting_v2.json",
        {
            "points": len(xyz),
            "fusion": method,
            "evaluation": "Export only; held-out hand labels needed for Q5",
            "per_predicted_class_reprojection": {
                str(c): {
                    "points": int((labels == c).sum()),
                    "mean_reprojection_px": (
                        float(np.asarray(errors)[labels == c].mean())
                        if (labels == c).any()
                        else None
                    ),
                }
                for c in range(6)
            },
        },
    )


def evaluate_label_holdouts(tracks, hand_masks):
    """Tracks contain observed image ids/xy/probabilities; exclude entire image.

    xy must be expressed in the corresponding hand-mask pixel coordinates.
    Bootstrap images, preserving all their correlated point observations.
    """
    matrices = {n: np.zeros((6, 7), np.int64) for n in hand_masks}
    for track in tracks:
        for name in set(track["image_ids"]) & set(hand_masks):
            _, label, _ = fuse_probabilities(track["probabilities"], track["image_ids"], name)
            i = track["image_ids"].index(name)
            x, y = np.round(track["xy"][i]).astype(int)
            mask = hand_masks[name]
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                matrices[name] += confusion(
                    np.asarray([[mask[y, x]]], np.uint8), np.asarray([[label]], np.uint8)
                )
    used = [m for m in matrices.values() if m.sum()]
    if not used:
        return {"status": "no_hand_observations", "observations": 0}
    accuracy = paired_bootstrap(
        used, lambda ms: segmentation_metrics(np.sum(ms, axis=0))["pixel_accuracy"]
    )
    class_ci = {}
    for class_id in range(6):

        def recall(ms, c=class_id):
            m = np.sum(ms, axis=0)
            return m[c, c] / m[c].sum() if m[c].sum() else np.nan

        class_ci[str(class_id)] = paired_bootstrap(used, recall)
    return {
        **segmentation_metrics(np.sum(used, axis=0)),
        "accuracy_bootstrap": accuracy,
        "per_class_recall_bootstrap": class_ci,
        "images": len(used),
        "label_source": "hand_only_leave_one_image_out",
    }
