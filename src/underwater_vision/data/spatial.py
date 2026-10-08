"""Deterministic spatial blocks with excluded boundary bands."""

import numpy as np
from scipy.spatial import cKDTree


def assign_spatial(points, specification):
    values = (np.asarray(points) - specification["origin"]) @ np.asarray(specification["axis"])
    low, high = specification["test_interval"]
    boundary, gap = specification["validation_boundary"], specification["gap"]
    labels = np.full(len(values), "excluded", dtype="<U10")
    inside = (values >= low) & (values <= high)
    labels[inside] = "test"
    outside = (values < low - gap) | (values > high + gap)
    labels[outside & (values < boundary - gap)] = "train"
    labels[outside & (values > boundary + gap)] = "validation"
    return labels


def spatial_split(names, centers, test_names, gap_fraction=0.035):
    centers = np.asarray(centers)
    origin = centers.mean(0)
    _, _, vt = np.linalg.svd(centers - origin, full_matrices=False)
    axis = vt[0]
    values = (centers - origin) @ axis
    gap = max(float(np.ptp(values) * gap_fraction), 1e-6)
    selected = np.isin(names, list(test_names))
    if not selected.any():
        raise ValueError("No test anchors in spatial split")
    low, high = float(values[selected].min()), float(values[selected].max())
    low, high = low - gap / 2, high + gap / 2
    remaining = values[(values < low - gap) | (values > high + gap)]
    if len(remaining) < 10:
        raise ValueError("Test anchors span most of the scene; choose a localized block")
    specification = {
        "origin": origin.tolist(),
        "axis": axis.tolist(),
        "test_interval": [low, high],
        "validation_boundary": float(np.quantile(remaining, 0.7)),
        "gap": gap,
        "method": "reference_center_principal_axis_spatial_slabs_with_boundary_exclusion",
        "limitation": "camera-center separation is not proof of disjoint visible surfaces",
    }
    labels = assign_spatial(centers, specification)
    groups = {
        s: [n for n, label in zip(names, labels, strict=True) if label == s]
        for s in ["train", "validation", "test", "excluded"]
    }
    if any(len(groups[s]) < 3 for s in ["train", "validation", "test"]):
        raise ValueError("Insufficient views in a spatial block")
    distances = {}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        distances[f"{a}_to_{b}"] = float(
            cKDTree(centers[labels == a]).query(centers[labels == b])[0].min()
        )
    return {
        "specification": specification,
        "groups": groups,
        "minimum_center_distances": distances,
        "reference_views": len(names),
    }
