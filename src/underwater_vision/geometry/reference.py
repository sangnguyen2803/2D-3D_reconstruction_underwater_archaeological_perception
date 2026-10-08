"""Published-camera geometry, independent of the custom RANSAC estimator.

Epipolar compatibility is a proxy label, not proof of correspondence correctness
or independently surveyed ground truth. F applies to undistorted pixels only.
"""

import cv2
import numpy as np

from underwater_vision.geometry.verification import sampson_error
from underwater_vision.reconstruction.colmap.database import make_camera


def camera_parameters(width, height, calibration, convention="brown_center"):
    camera = make_camera(width, height, calibration)
    K = camera.calibration_matrix().copy()
    K[:2, 2] -= 0.5
    distortion = np.asarray(camera.params[4:]).copy()
    if convention == "no_distortion":
        distortion[:] = 0
    elif convention == "absolute_principal":
        K[0, 2] = calibration["cx_offset"] * width / calibration["width"] - 0.5
        K[1, 2] = calibration["cy_offset"] * height / calibration["height"] - 0.5
    elif convention != "brown_center":
        raise ValueError("Unknown calibration convention")
    return K, distortion


def undistort(xy, K, distortion):
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    if not len(xy):
        return xy.copy()
    # More iterations matter at the edges of this wide-angle camera.
    operation = getattr(cv2, "undistortPointsIter", cv2.undistortPoints)
    return operation(
        xy[:, None],
        K,
        distortion,
        None,
        None,
        K,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10),
    ).reshape(-1, 2)


def fundamental_from_poses(c2w_a, c2w_b, K_a, K_b=None):
    relative = np.linalg.inv(c2w_b) @ c2w_a
    R, t = relative[:3, :3], relative[:3, 3]
    if np.linalg.norm(t) < 1e-10:
        raise ValueError("Zero camera baseline cannot define a fundamental matrix")
    skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(K_a if K_b is None else K_b).T @ skew @ R @ np.linalg.inv(K_a)
    return F / np.linalg.norm(F)


def reference_labels(xy_a, xy_b, c2w_a, c2w_b, K, distortion, positive=1.5, negative=4.0):
    if not 0 < positive < negative:
        raise ValueError("Require 0 < positive threshold < negative threshold")
    F = fundamental_from_poses(c2w_a, c2w_b, K)
    a, b = undistort(xy_a, K, distortion), undistort(xy_b, K, distortion)
    error = sampson_error(F, a, b)
    label = np.full(len(error), -1, np.int8)
    label[error <= positive] = 1
    label[error >= negative] = 0
    return label, error, F


def triangulate_observations(poses, xy, K):
    """Linear reference-camera triangulation. Input pixels are undistorted."""
    if len(poses) < 2:
        raise ValueError("Triangulation needs at least two observations")
    rows = []
    for pose, point in zip(poses, xy, strict=True):
        P = K @ np.linalg.inv(pose)[:3]
        rows.extend([point[0] * P[2] - P[0], point[1] * P[2] - P[1]])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    if abs(vt[-1, 3]) < 1e-12:
        raise ValueError("Point triangulated at infinity")
    return vt[-1, :3] / vt[-1, 3]


def project(point, c2w, K):
    camera = np.linalg.inv(c2w) @ np.r_[point, 1.0]
    pixel = K @ camera[:3]
    if camera[2] <= 1e-10:
        return np.full(2, np.nan)
    return pixel[:2] / pixel[2]
