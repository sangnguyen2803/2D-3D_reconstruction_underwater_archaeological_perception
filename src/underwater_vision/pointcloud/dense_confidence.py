"""Depth-verified observation evidence for fused COLMAP MVS points."""

from pathlib import Path

import numpy as np
import pycolmap
from plyfile import PlyData, PlyElement

from underwater_vision.utils.io import read_rgb, write_json
from underwater_vision.visibility.deterministic import estimate_visibility


def read_colmap_array(path):
    with Path(path).open("rb") as f:
        header = bytearray()
        while header.count(b"&") < 3:
            character = f.read(1)
            if not character:
                raise ValueError("Truncated COLMAP depth header")
            header.extend(character)
        width, height, channels = map(int, header.decode("ascii").strip("&").split("&"))
        array = np.fromfile(f, dtype=np.float32)
    if len(array) != width * height * channels:
        raise ValueError("Invalid COLMAP depth array size")
    return array.reshape((width, height, channels), order="F").transpose(1, 0, 2).squeeze()


def dense_confidence(workspace, error_tolerance=0.02, batch_size=50000):
    workspace = Path(workspace)
    vertex = PlyData.read(workspace / "fused.ply")["vertex"].data
    xyz = np.c_[vertex["x"], vertex["y"], vertex["z"]]
    model = pycolmap.Reconstruction(str(workspace / "sparse"))
    n = len(xyz)
    support, quality, residual = np.zeros(n), np.zeros(n), np.zeros(n)
    for image in model.images.values():
        if not image.has_pose:
            continue
        path = workspace / "stereo" / "depth_maps" / (image.name + ".geometric.bin")
        if not path.exists():
            continue
        depth = read_colmap_array(path)
        visibility, _ = estimate_visibility(read_rgb(workspace / "images" / image.name))
        camera = model.cameras[image.camera_id]
        K = camera.calibration_matrix()
        world_to_camera = image.cam_from_world().matrix()
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            projected = xyz[start:stop] @ world_to_camera[:, :3].T + world_to_camera[:, 3]
            z = projected[:, 2]
            pixels = projected @ K.T
            pixels = pixels[:, :2] / np.maximum(z[:, None], 1e-12)
            x = np.floor(pixels[:, 0] * depth.shape[1] / camera.width).astype(int)
            y = np.floor(pixels[:, 1] * depth.shape[0] / camera.height).astype(int)
            valid = (z > 0) & (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
            ids = np.flatnonzero(valid)
            measured = depth[y[ids], x[ids]]
            error = np.abs(z[ids] - measured) / np.maximum(measured, 1e-12)
            good = (measured > 0) & (error <= error_tolerance)
            ids = ids[good]
            if not len(ids):
                continue
            px = np.clip(
                (pixels[ids, 0] * visibility.shape[1] / camera.width).astype(int),
                0,
                visibility.shape[1] - 1,
            )
            py = np.clip(
                (pixels[ids, 1] * visibility.shape[0] / camera.height).astype(int),
                0,
                visibility.shape[0] - 1,
            )
            support[start + ids] += 1
            quality[start + ids] += visibility[py, px]
            residual[start + ids] += error[good]
    score = (
        (1 - np.exp(-support / 3))
        * quality
        / np.maximum(support, 1)
        * np.exp(-residual / np.maximum(support, 1) / error_tolerance)
    )
    dtype = vertex.dtype.descr + [("confidence", "<f4"), ("supporting_views", "<u4")]
    enriched = np.empty(n, dtype=dtype)
    for name in vertex.dtype.names:
        enriched[name] = vertex[name]
    enriched["confidence"], enriched["supporting_views"] = score.astype(np.float32), support.astype(
        np.uint32
    )
    PlyData([PlyElement.describe(enriched, "vertex")], text=False).write(
        str(workspace / "fused_confidence.ply")
    )
    stats = {
        "dense_points": n,
        "points_with_depth_support": int((support > 0).sum()),
        "mean_supporting_views": float(support.mean()) if n else 0,
        "mean_confidence": float(score.mean()) if n else 0,
        "relative_depth_tolerance": error_tolerance,
        "interpretation": "heuristic_depth_consistency_and_image_visibility_not_calibrated_probability",
    }
    write_json(workspace / "confidence_metrics.json", stats)
    return stats
