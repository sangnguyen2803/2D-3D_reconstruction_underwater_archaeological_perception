from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def visibility_figure(image, visibility, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(image)
    ax[0].set_title("Underwater RGB")
    artist = ax[1].imshow(visibility, vmin=0, vmax=1, cmap="viridis")
    ax[1].set_title("Local evidence reliability (heuristic)")
    fig.colorbar(artist, ax=ax[1], fraction=0.03)
    for a in ax:
        a.axis("off")
    save(fig, path)


def matching_figure(a, b, xy_a, xy_b, result, path):
    h = max(a.shape[0], b.shape[0])
    canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], a.shape[1] :] = b
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    for ax, mode in zip(
        axes, ["Candidate matches", "Geometric inliers", "Confidence-weighted inliers"], strict=True
    ):
        ax.imshow(canvas)
        ids = np.arange(len(result["pairs"]))
        if mode != "Candidate matches":
            ids = ids[result["inliers"]]
        for idx in ids[:: max(1, len(ids) // 150)]:
            i, j = result["pairs"][idx]
            p, q = xy_a[i], xy_b[j] + [a.shape[1], 0]
            color = (
                plt.cm.viridis(result["score"][idx]) if mode.startswith("Confidence") else "lime"
            )
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=0.5, alpha=0.7)
        ax.set_title(mode)
        ax.axis("off")
    save(fig, path)


def reconstruction_figure(reconstruction, path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    points = list(reconstruction.points3D.values())
    if points:
        xyz = np.array([p.xyz for p in points])
        rgb = np.array([p.color for p in points]) / 255
        step = max(1, len(points) // 30000)
        ax.scatter(*xyz[::step].T, c=rgb[::step], s=0.5)
    cameras = np.array(
        [i.projection_center() for i in reconstruction.images.values() if i.has_pose]
    )
    if len(cameras):
        ax.scatter(*cameras.T, c="red", s=10, label="Cameras")
    ax.set_title("Sparse reconstruction and registered camera centers")
    ax.set_xlabel("X (SfM units)")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    save(fig, path)


def learned_reliability_figure(image, xy, scores, path):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)
    points = ax.scatter(xy[:, 0], xy[:, 1], c=scores, s=9, vmin=0, vmax=1, cmap="viridis")
    ax.set_title("Learned geometric reliability at localized features")
    ax.axis("off")
    fig.colorbar(points, ax=ax, fraction=0.03)
    save(fig, path)


def trajectory_figure(estimated, reference, path):
    from underwater_vision.evaluation.metrics import similarity_alignment

    names = sorted(set(estimated) & set(reference))
    if len(names) < 3:
        return
    e = np.array([estimated[n][:3, 3] for n in names])
    g = np.array([reference[n][:3, 3] for n in names])
    try:
        scale, rotation, translation = similarity_alignment(e, g)
    except ValueError:
        # Match metric handling: a degenerate trajectory cannot be Sim(3) aligned.
        return
    e = scale * (e @ rotation.T) + translation
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(*g.T, label="Published BA reference")
    ax.plot(*e.T, label="Estimated (Sim(3) aligned)")
    ax.legend()
    ax.set_title("Camera trajectory reference agreement")
    error = fig.add_subplot(122)
    error.plot(np.linalg.norm(e - g, axis=1))
    error.set_xlabel("Selected acquisition frame order")
    error.set_ylabel("Center error (reference units)")
    save(fig, path)
