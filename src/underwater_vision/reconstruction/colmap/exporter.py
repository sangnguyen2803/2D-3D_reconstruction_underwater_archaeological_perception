from pathlib import Path

import numpy as np

from underwater_vision.utils.io import write_json


def export_reconstruction(reconstruction, output):
    output = Path(output)
    reconstruction.export_PLY(str(output / "sparse_rgb.ply"))
    poses = {}
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        pose = np.eye(4)
        pose[:3] = image.cam_from_world().inverse().matrix()
        poses[image.name] = pose
    write_json(output / "camera_poses.json", {name: pose.tolist() for name, pose in poses.items()})
    points = list(reconstruction.points3D.values())
    statistics = {
        "registered_images": reconstruction.num_reg_images(),
        "sparse_points": len(points),
        "mean_reprojection_error_px": float(np.mean([p.error for p in points])) if points else None,
        "mean_track_length": float(np.mean([p.track.length() for p in points])) if points else None,
    }
    return poses, statistics
