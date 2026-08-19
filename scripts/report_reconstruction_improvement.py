"""Assess every SfM run; compare pose errors on identical camera populations."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from underwater_vision.evaluation.metrics import similarity_alignment
from underwater_vision.evaluation.reconstruction_comparison import compare_poses
from underwater_vision.matchability.protocol import digest, read_json
from underwater_vision.utils.io import write_json


def average(records):
    return {
        key: float(np.mean([r[key] for r in records]))
        for key in [
            "registered_images",
            "sparse_points",
            "mean_reprojection_error_px",
            "mean_track_length",
        ]
    }


def compare_runs(before_rows, after_rows, reference):
    pairs = []
    common = set(reference)
    for first, second in zip(before_rows, after_rows, strict=True):
        a = read_json(Path(first["path"]) / "camera_poses.json")
        b = read_json(Path(second["path"]) / "camera_poses.json")
        pairs.append((a, b, first, second))
        common &= set(a) & set(b)
    results = []
    for a, b, first, _second in pairs:
        result = compare_poses({n: a[n] for n in common}, {n: b[n] for n in common}, reference)
        result.update(seed=first["seed"], registered_before=len(a), registered_after=len(b))
        results.append(result)
    if any(r["status"] != "ok" for r in results):
        return {"status": "insufficient_common_cameras", "runs": results}
    draws = np.mean([r["bootstrap_deltas"] for r in results], axis=0)
    before, after = np.mean([r["ate_before"] for r in results]), np.mean(
        [r["ate_after"] for r in results]
    )
    return {
        "status": "ok",
        "common_cameras": len(common),
        "ate_before": float(before),
        "ate_after": float(after),
        "relative_ate_reduction": float(1 - after / before),
        "paired_ci95_delta": np.quantile(draws, [0.025, 0.975]).tolist(),
        "before": average(before_rows),
        "after": average(after_rows),
        "registration_not_lower": all(len(b) >= len(a) for a, b, _, _ in pairs),
        "runs": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("outputs/reconstruction_improvement_v1"))
    parser.add_argument("--detector", type=Path, default=Path("outputs/detector_coverage_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reconstruction_assessment_v1"))
    args = parser.parse_args()
    base, detector = read_json(args.base / "comparison_v1.json"), read_json(
        args.detector / "comparison_v1.json"
    )
    assert base["status"] == detector["status"] == "complete"
    assert len(base["runs"]) == 54 and len(detector["runs"]) == 9
    assert digest(args.base / "executed_benchmark_source_v1.py") == base["source_sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "complete",
        "base_sha256": digest(args.base / "comparison_v1.json"),
        "detector_sha256": digest(args.detector / "comparison_v1.json"),
        "methods": {},
        "detector": {},
        "segformer": {},
        "statistics": "500 paired four-consecutive-camera cluster bootstrap draws; re-fit Sim3 in every draw. The same common camera set and draws are used across all three seeds. Missing-camera counts are reported separately. Fixed-region, already reconstructed-network conditional intervals, not independent scene uncertainty.",
    }
    blocks = ["block273", "block599", "block1116"]
    for block in blocks:
        groups = {
            method: sorted(
                [r for r in base["runs"] if r["block"] == block and r["method"] == method],
                key=lambda r: r["seed"],
            )
            for method in base["methods"]
        }
        assert all(len(v) == 3 for v in groups.values())
        result["methods"][block] = {k: average(v) for k, v in groups.items()}
        reference = read_json(args.base / block / "reference_poses_v1.json")
        low = sorted([r for r in detector["runs"] if r["block"] == block], key=lambda r: r["seed"])
        result["detector"][block] = compare_runs(groups["ratio_full"], low, reference)
        result["segformer"][block] = {
            control: compare_runs(groups[control], groups["segformer75_ratio"], reference)
            for control in ["ratio_full", "response75_ratio", "random75_ratio"]
        }
    write_json(args.output / "assessment_v1.json", result)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, block in zip(axes, blocks, strict=True):
        comparison = result["detector"][block]
        ax.bar(
            ["Original", "Low contrast"],
            [comparison["before"]["registered_images"], comparison["after"]["registered_images"]],
            color=["#627a91", "#168c78"],
        )
        ax.set_ylim(0, 35)
        ax.axhline(32, color="gray", linestyle="--")
        ax.set_title(block)
        ax.set_ylabel("Registered views / 32 (three-seed mean)")
    fig.tight_layout()
    fig.savefig(args.output / "registration_v1.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, block in zip(axes, blocks, strict=True):
        reference = read_json(args.base / block / "reference_poses_v1.json")
        before = read_json(args.base / block / "ratio_full_s42/camera_poses.json")
        after = read_json(args.detector / block / "low_contrast_s42/camera_poses.json")
        common = sorted(set(reference) & set(before) & set(after))
        center = np.array([np.array(reference[n])[:3, 3] for n in sorted(reference)])
        offset = center.mean(0)
        _, vectors = np.linalg.eigh(np.cov(center.T))
        projection = vectors[:, -2:]
        values = (center - offset) @ projection
        ax.plot(values[:, 0], values[:, 1], "k.--", label="Reference (published BA)", alpha=0.5)
        for poses, label, color in [
            (before, "Original", "#627a91"),
            (after, "Low contrast", "#168c78"),
        ]:
            source = np.array([np.array(poses[n])[:3, 3] for n in common])
            target = np.array([np.array(reference[n])[:3, 3] for n in common])
            s, rotation, shift = similarity_alignment(source, target)
            all_centers = np.array([np.array(poses[n])[:3, 3] for n in sorted(poses)])
            view = (s * (all_centers @ rotation.T) + shift - offset) @ projection
            ax.plot(
                view[:, 0],
                view[:, 1],
                ".-",
                color=color,
                label=f"{label} ({len(poses)} views)",
                alpha=0.8,
            )
        ax.set_title(block + " / seed42")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Reference-plane axis 1")
        ax.set_ylabel("Reference-plane axis 2")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "trajectories_v1.png", dpi=180)
    plt.close(fig)
    rows = [
        "# Reconstruction improvement v1",
        "",
        "Completed a 54-run learned-feature SfM ablation and a nine-run low-contrast detector ablation. The original 16-view artifacts and all historical reports remain intact. No new dense MVS or independent surface-accuracy claim is made.",
        "",
        "## Controlled setup",
        "",
        base["protocol"],
        "",
        base["selection"],
        "",
        detector["protocol"],
        "",
        "The detector change is now optional in the source and `experiment=low_contrast` profile. The default remains SIFT contrast 0.04. Nondefault contrast enters the feature-cache key. Existing v1 match filters reject a changed contrast because their training detector distribution was 0.04. The historical orchestration also includes symmetric ratio scoring and custom verification; the controlled scripts here use forward ratio plus mutual top-1 and standard COLMAP verification. Use the supplied benchmark scripts for exact reproduction of the reported metrics.",
        "",
        "## Detector change: native reconstruction statistics",
        "",
        "Three-seed means; compare ATE on common camera sets in the next table, not different registration populations.",
        "",
        "| Region | Registered before / after | Points before / after | Reprojection before / after (px) |",
        "|---|---:|---:|---:|",
    ]
    for block, c in result["detector"].items():
        b, a = c["before"], c["after"]
        rows.append(
            f"| {block} | {b['registered_images']:.1f} / {a['registered_images']:.1f} | {b['sparse_points']:.0f} / {a['sparse_points']:.0f} | {b['mean_reprojection_error_px']:.4f} / {a['mean_reprojection_error_px']:.4f} |"
        )
    rows += [
        "",
        "## Detector change: identical-camera pose agreement",
        "",
        "| Region | Common cameras | ATE before / after | Relative reduction | Paired 95% CI, after minus before |",
        "|---|---:|---:|---:|---:|",
    ]
    for block, c in result["detector"].items():
        lo, hi = c["paired_ci95_delta"]
        rows.append(
            f"| {block} | {c['common_cameras']} | {c['ate_before']:.6f} / {c['ate_after']:.6f} | {c['relative_ate_reduction']*100:.2f}% | [{lo:.6f}, {hi:.6f}] |"
        )
    rows += [
        "",
        result["statistics"],
        "",
        "ATE uses published bundle-adjusted poses after similarity alignment; units are the dataset reference scale, not independently verified meters. This measures reference-pose agreement, not ground-truth surface accuracy. Native reprojection errors and point counts have different point populations and are not sufficient accuracy evidence on their own. Increased feature coverage costs additional extraction, descriptor matching and mapping compute; this is not a speed improvement.",
        "",
        "## SegFormer selection: matched 75% per-image budgets",
        "",
        "| Region | Control | Common-camera ATE reduction | Paired 95% CI, after minus before |",
        "|---|---|---:|---:|",
    ]
    for block, controls in result["segformer"].items():
        for control, c in controls.items():
            lo, hi = c["paired_ci95_delta"]
            rows.append(
                f"| {block} | {control} | {c['relative_ate_reduction']*100:.2f}% | [{lo:.6f}, {hi:.6f}] |"
            )
    rows += [
        "",
        "All 54 runs are retained, including failures or regressions. The equal-budget SegFormer result must be assessed against both response and random selection, not only the full-feature baseline. Registration loss is an adverse outcome even when conditional common-camera ATE is lower. No test pose or reference label was used to choose a keypoint or a match. These are development experiments on one acquisition, not an untouched external benchmark.",
        "",
        "An exploratory fixed-intrinsics Soft-L1 global BA probe on block273 lowered reprojection error but increased ATE from 0.026967 to 0.030424; it was not promoted into the pipeline. Its artifacts remain in `outputs/robust_ba_probe_v1/`.",
        "",
        "## Evidence and reproduction",
        "",
        "- [54-run metrics](outputs/reconstruction_improvement_v1/comparison_v1.json)",
        "- [Nine detector runs](outputs/detector_coverage_v1/comparison_v1.json)",
        "- [Paired assessment](outputs/reconstruction_assessment_v1/assessment_v1.json)",
        "- [Registered cameras](outputs/reconstruction_assessment_v1/registration_v1.png)",
        "- [Camera trajectories](outputs/reconstruction_assessment_v1/trajectories_v1.png)",
        "",
        "```powershell",
        ".venv\\Scripts\\python.exe scripts/benchmark_reconstruction.py --output outputs/reconstruction_reproduction_v1",
        ".venv\\Scripts\\python.exe scripts/benchmark_detector_coverage.py --contrast 0.01 --output outputs/detector_reproduction_v1",
        "```",
        "",
        "Both benchmark output directories must be new. The detector benchmark defaults to the preserved controlled baseline. New code and reports have not been committed or pushed.",
    ]
    target = Path("RECONSTRUCTION_IMPROVEMENT_v1.md")
    assert not target.exists()
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(
        {
            block: {k: v for k, v in c.items() if k not in {"runs"}}
            for block, c in result["detector"].items()
        }
    )


if __name__ == "__main__":
    main()
