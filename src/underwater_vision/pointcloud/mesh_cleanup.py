"""Optional density-adaptive Poisson mesh trimming using observed dense points."""

import numpy as np
from scipy.spatial import cKDTree


def trim_mesh(vertices, faces, observed, radius_factor=3.0, neighbors=8):
    """Keep faces whose vertices are near measured points at the local spacing.

    This measures observed-point support, not surface accuracy or ground truth.
    It never modifies input arrays or fills unobserved regions.
    """
    vertices, observed = np.asarray(vertices, float), np.asarray(observed, float)
    faces = np.asarray(faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or observed.ndim != 2
        or observed.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not np.issubdtype(faces.dtype, np.integer)
        or not len(faces)
        or faces.min() < 0
        or faces.max() >= len(vertices)
        or not np.isfinite(vertices).all()
        or not np.isfinite(observed).all()
        or neighbors < 1
        or len(observed) <= neighbors
        or not np.isfinite(radius_factor)
        or radius_factor <= 0
    ):
        raise ValueError("Need finite XYZ arrays, valid triangles and positive support parameters")
    tree = cKDTree(observed)
    spacing = tree.query(observed, k=neighbors + 1, workers=4)[0][:, -1] / np.sqrt(neighbors)
    spacing = np.maximum(spacing, np.finfo(float).eps)
    distance, nearest = tree.query(vertices, workers=4)
    supported = distance <= radius_factor * spacing[nearest]
    keep = supported[faces].all(axis=1)
    if not keep.any():
        raise ValueError("No supported mesh faces remain")
    used = np.unique(faces[keep])
    remap = np.full(len(vertices), -1, np.int64)
    remap[used] = np.arange(len(used))
    triangle = vertices[faces]
    area = (
        np.linalg.norm(
            np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1
        )
        / 2
    )
    if area.sum() <= 0:
        raise ValueError("Mesh has no nondegenerate surface area")
    before_distance = cKDTree(vertices).query(observed, workers=4)[0]
    after_distance = cKDTree(vertices[used]).query(observed, workers=4)[0]
    measured = {
        "radius_factor": radius_factor,
        "spacing_neighbors": neighbors,
        "spacing_definition": f"Distance to {neighbors}th other point / sqrt({neighbors}), approximating surface sample spacing",
        "vertices_before": len(vertices),
        "vertices_after": len(used),
        "faces_before": len(faces),
        "faces_after": int(keep.sum()),
        "area_before": float(area.sum()),
        "area_after": float(area[keep].sum()),
        "removed_area_fraction": float(area[~keep].sum() / area.sum()),
        "dense_points": len(observed),
        "dense_point_coverage_before": float(np.mean(before_distance <= 2 * spacing)),
        "dense_point_coverage_after": float(np.mean(after_distance <= 2 * spacing)),
        "coverage_definition": "Fraction of observed dense points within two local spacings of a retained mesh vertex, not point-to-triangle accuracy",
        "mesh_to_observed_distance_p95_before": float(np.quantile(distance, 0.95)),
        "mesh_to_observed_distance_p95_after": float(np.quantile(distance[used], 0.95)),
        "scope": "Geometric support cleanup; no independent surface ground truth or full-scene accuracy claim",
    }
    return used, remap[faces[keep]], measured
