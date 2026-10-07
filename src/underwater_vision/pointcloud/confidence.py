"""Heuristic evidence confidence, not a calibrated probability of correctness."""

from pathlib import Path

import numpy as np

from underwater_vision.utils.io import write_json


def write_ply(path, xyz, rgb, confidence):
    with Path(path).open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\nelement vertex " + str(len(xyz)) + "\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(
            "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty float confidence\nend_header\n"
        )
        for p, color, score in zip(xyz, rgb, confidence, strict=True):
            handle.write(" ".join(map(str, [*p, *map(int, color), float(score)])) + "\n")


def sparse_confidence(reconstruction, features, output, settings):
    xyz, rgb, confidence, records = [], [], [], []
    for pid, point in reconstruction.points3D.items():
        views, reliability = [], []
        for element in point.track.elements:
            image = reconstruction.images[element.image_id]
            views.append(image.projection_center())
            reliability.append(features[image.name]["reliability"][element.point2D_idx])
        rays = np.asarray(views) - point.xyz
        rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
        cosine = np.clip(rays @ rays.T, -1, 1)
        angle = float(np.rad2deg(np.max(np.arccos(cosine)))) if len(rays) > 1 else 0.0
        support = 1 - np.exp(-max(0, len(views) - 1) / settings["confidence_support_scale"])
        reprojection = np.exp(-point.error / settings["confidence_error_scale"])
        parallax = 1 - np.exp(-angle / settings["confidence_angle_scale"])
        score = float(np.clip(support * reprojection * parallax * np.mean(reliability), 0, 1))
        xyz.append(point.xyz)
        rgb.append(point.color)
        confidence.append(score)
        records.append(
            {
                "point_id": int(pid),
                "views": len(views),
                "angle_degrees": angle,
                "reprojection_error_px": float(point.error),
                "confidence": score,
            }
        )
    xyz, rgb, confidence = np.asarray(xyz), np.asarray(rgb), np.asarray(confidence)
    write_ply(Path(output) / "sparse_confidence.ply", xyz, rgb, confidence)
    np.savez_compressed(
        Path(output) / "sparse_confidence.npz", xyz=xyz, rgb=rgb, confidence=confidence
    )
    write_json(Path(output) / "point_confidence.json", records)
