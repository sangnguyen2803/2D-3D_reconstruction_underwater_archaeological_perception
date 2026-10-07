import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def similarity_alignment(source, target):
    """Umeyama Sim(3) alignment; monocular reconstructions have unknown scale."""
    source, target = np.asarray(source), np.asarray(target)
    if len(source) < 3 or np.linalg.matrix_rank(source - source.mean(0)) < 2:
        raise ValueError("At least 3 non-collinear centers required for alignment")
    a, b = source - source.mean(0), target - target.mean(0)
    U, d, Vt = np.linalg.svd(b.T @ a / len(a))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(U @ Vt)
    R = U @ correction @ Vt
    s = np.sum(d * np.diag(correction)) / np.mean(np.sum(a * a, axis=1))
    t = target.mean(0) - s * R @ source.mean(0)
    return float(s), R, t


def trajectory_metrics(estimated, reference):
    names = sorted(set(estimated) & set(reference))
    if len(names) < 3:
        return {"status": "insufficient_reference_overlap", "matched_poses": len(names)}
    e, g = np.stack([estimated[n] for n in names]), np.stack([reference[n] for n in names])
    try:
        scale, alignment, offset = similarity_alignment(e[:, :3, 3], g[:, :3, 3])
    except ValueError as exc:
        return {"status": str(exc), "matched_poses": len(names)}
    centers = scale * (e[:, :3, 3] @ alignment.T) + offset
    rotations = alignment @ e[:, :3, :3]
    translation = np.linalg.norm(centers - g[:, :3, 3], axis=1)
    rotation = np.rad2deg(
        Rotation.from_matrix(rotations.transpose(0, 2, 1) @ g[:, :3, :3]).magnitude()
    )
    # Relative poses are expressed in the previous camera coordinate system.
    rpe_t, rpe_r = [], []
    for k in range(len(names) - 1):
        dt_e = rotations[k].T @ (centers[k + 1] - centers[k])
        dt_g = g[k, :3, :3].T @ (g[k + 1, :3, 3] - g[k, :3, 3])
        rpe_t.append(np.linalg.norm(dt_e - dt_g))
        rel_e = rotations[k].T @ rotations[k + 1]
        rel_g = g[k, :3, :3].T @ g[k + 1, :3, :3]
        rpe_r.append(np.rad2deg(Rotation.from_matrix(rel_e.T @ rel_g).magnitude()))
    return {
        "status": "ok",
        "reference_type": "published_bundle_adjustment_not_independent_GT",
        "matched_poses": len(names),
        "alignment_scale": scale,
        "ate_rmse_reference_units": float(np.sqrt(np.mean(translation**2))),
        "translation_median_reference_units": float(np.median(translation)),
        "rotation_median_degrees": float(np.median(rotation)),
        "rpe_translation_rmse_reference_units": float(np.sqrt(np.mean(np.square(rpe_t)))),
        "rpe_rotation_median_degrees": float(np.median(rpe_r)),
    }


def dense_metrics(prediction, reference, thresholds=(0.01, 0.02, 0.05)):
    """Only call with independently measured, aligned surface reference geometry."""
    if len(prediction) == 0 or len(reference) == 0:
        raise ValueError("Empty clouds cannot be evaluated")
    accuracy = cKDTree(reference).query(prediction)[0]
    completeness = cKDTree(prediction).query(reference)[0]
    result = {
        "accuracy_mean": float(accuracy.mean()),
        "completeness_mean": float(completeness.mean()),
        "chamfer_l1": float(accuracy.mean() + completeness.mean()),
    }
    for threshold in thresholds:
        precision, recall = float(np.mean(accuracy < threshold)), float(
            np.mean(completeness < threshold)
        )
        result[f"fscore_{threshold}"] = 2 * precision * recall / max(precision + recall, 1e-12)
    return result
