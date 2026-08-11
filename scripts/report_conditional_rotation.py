"""Secondary angular audit; never substitutes for the primary ATE criterion."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from underwater_vision.evaluation.metrics import similarity_alignment
from underwater_vision.matchability.protocol import digest, read_json
from underwater_vision.utils.io import write_json


def angular_error(poses, reference, names):
    e = np.stack([poses[n] for n in names])
    g = np.stack([reference[n] for n in names])
    _, alignment, _ = similarity_alignment(e[:, :3, 3], g[:, :3, 3])
    rotations = alignment @ e[:, :3, :3]
    return float(
        np.median(
            np.rad2deg(
                Rotation.from_matrix(rotations.transpose(0, 2, 1) @ g[:, :3, :3]).magnitude()
            )
        )
    )


def main():
    root = Path("outputs/conditional_precision_geometry_v1")
    report = read_json(root / "comparison_v1.json")
    baseline = read_json("outputs/detector_coverage_v1/comparison_v1.json")
    controls = read_json("outputs/segformer_strong_baseline_v1/comparison_v1.json")
    result = {
        "status": "complete",
        "source_sha256": digest(__file__),
        "scope": "Secondary exploratory rotation metric, all three blocks and controls. 500 paired four-camera cluster draws, refit Sim3 on every draw, same draws averaged across three seeds. Conditional fixed-region intervals, unadjusted for multiple comparisons. Primary ATE criterion remains unchanged.",
        "blocks": {},
    }
    for block in ["block273", "block599", "block1116"]:
        reference = read_json(
            Path("outputs/reconstruction_improvement_v1") / block / "reference_poses_v1.json"
        )
        after = sorted([r for r in report["runs"] if r["block"] == block], key=lambda r: r["seed"])
        result["blocks"][block] = {}
        for method in ["low_contrast_ratio", "ratio90", "random90"]:
            source = baseline if method == "low_contrast_ratio" else controls
            before = sorted(
                [r for r in source["runs"] if r["block"] == block and r["method"] == method],
                key=lambda r: r["seed"],
            )
            pairs = [
                (
                    read_json(Path(a["path"]) / "camera_poses.json"),
                    read_json(Path(b["path"]) / "camera_poses.json"),
                )
                for a, b in zip(before, after, strict=True)
            ]
            names = sorted(set(reference).intersection(*(set(a) & set(b) for a, b in pairs)))
            assert len(names) == 32
            rng = np.random.default_rng(42)
            draws = []
            for _ in range(500):
                indices = np.concatenate(
                    [(int(s) + np.arange(4)) % len(names) for s in rng.integers(len(names), size=8)]
                )
                sampled = [names[i] for i in indices]
                draws.append(
                    np.mean(
                        [
                            angular_error(b, reference, sampled)
                            - angular_error(a, reference, sampled)
                            for a, b in pairs
                        ]
                    )
                )
            first = float(np.mean([angular_error(a, reference, names) for a, _ in pairs]))
            second = float(np.mean([angular_error(b, reference, names) for _, b in pairs]))
            result["blocks"][block][method] = {
                "before_median_degrees_seed_mean": first,
                "after_median_degrees_seed_mean": second,
                "relative_reduction": 1 - second / first,
                "paired_ci95_delta_degrees": np.quantile(draws, [0.025, 0.975]).tolist(),
            }
    path = root / "secondary_rotation_v1.json"
    assert not path.exists()
    write_json(path, result)
    print(result)


if __name__ == "__main__":
    main()
