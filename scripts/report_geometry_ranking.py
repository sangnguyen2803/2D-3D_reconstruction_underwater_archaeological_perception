"""Report all learned-ranking geometry outcomes, including adverse results."""

import argparse
from pathlib import Path

from report_reconstruction_improvement import average, compare_runs

from underwater_vision.matchability.protocol import digest, read_json
from underwater_vision.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/geometry_ranking_v1"))
    parser.add_argument("--controls", type=Path)
    parser.add_argument(
        "--baseline", type=Path, default=Path("outputs/reconstruction_improvement_v1")
    )
    parser.add_argument("--baseline-method", default="ratio_full")
    parser.add_argument("--report", type=Path, default=Path("LEARNED_GEOMETRY_ASSESSMENT_v1.md"))
    args = parser.parse_args()
    current = read_json(args.input / "comparison_v1.json")
    base_root = Path("outputs/reconstruction_improvement_v1")
    base = read_json(args.baseline / "comparison_v1.json")
    assert current["status"] == "complete" and len(current["runs"]) == 9 * len(current["methods"])
    control_report = read_json(args.controls / "comparison_v1.json") if args.controls else current
    result = {
        "status": "complete",
        "comparison_sha256": digest(args.input / "comparison_v1.json"),
        "primary_method": current.get("primary_method", "tight90"),
        "baseline_sha256": digest(args.baseline / "comparison_v1.json"),
        "methods": {},
        "comparisons": {},
        "scope": "Already examined spatial development blocks; no independent scene or surveyed ground truth. Multiple learned scorers evaluated; report every comparison, do not promote the best test row as a preselected winner.",
    }
    rows = [
        "# Learned geometry assessment v1",
        "",
        "This new report preserves all earlier results and assesses whether learned match ranking improves actual sparse camera geometry. Detector features are identical between controls and learned scorers."
        + (
            " All methods use contrast0.01, including the strong full baseline."
            if "detector_baseline_sha256" in current
            else " All methods use contrast0.04."
        ),
        "",
        "## Method and separation",
        "",
        current["protocol"],
        "",
        (
            "Conditional geometry inputs contain descriptor distance/ratio, two local reliability values and normalized endpoint/relative coordinates; no DINO or SegFormer score enters this model. The source dataset has455,158 eligible matches from388 train pairs across226 images, and221 validation pairs across65 images. The initial precision model selects context epoch4 on validation. An observability checkpoint, if specified, warm-starts that architecture and maximizes validation retained utility after ten epochs. Reference-derived ray-angle utility is a proxy, not independent3D ground truth. Target used in this run: "
            + current.get("target", "tight epipolar precision")
            if "precision90" in current["methods"]
            else ""
        ),
        "",
        (
            "A match network is initialized from its original v1 checkpoint and fine-tuned with a soft target exp(-0.5*(reference epipolar error/0.5px)^2). Twenty epochs are evaluated on spatial validation pairs only; the checkpoint minimizes clipped reference error of retained validation matches. This is tight epipolar compatibility supervision, not independent 3D ground truth. No SegFormer weights are retrained."
            if "tight90" in current["methods"]
            else (
                "The conditional scorer uses the immutable classical RAW match pool, before COLMAP's geometric verification. It does not consume stored RANSAC inliers or reference test geometry. Recomputed descriptor neighbors are checked against frozen pairs."
                if "precision90" in current["methods"]
                else "SegFormer weights are frozen. CPU float32 full-resolution logits are sampled at the new contrast0.01 keypoint coordinates, then sigmoid endpoint scores rank matches. Models were originally supervised at contrast0.04 keypoints; no claim is made about classification accuracy under this detector shift. No network or score parameter is fitted on these test images."
            )
        ),
        "",
        "Every scorer shares identical keypoints, pairs and exact per-pair retained match counts. Ratio90 and random90 isolate the effects of ranking from budget reduction; ratio_full measures the full classical baseline. Endpoint SegFormer evidence ranks correspondences rather than deleting keypoints.",
        "",
        "## Native reconstruction statistics",
        "",
        "Three-seed means. These point/reprojection metrics use each run's native point population.",
        "",
        "| Block | Method | Registered | Sparse points | Reprojection px |",
        "|---|---|---:|---:|---:|",
    ]
    for block in ["block273", "block599", "block1116"]:
        groups = {
            method: sorted(
                [r for r in current["runs"] if r["block"] == block and r["method"] == method],
                key=lambda r: r["seed"],
            )
            for method in current["methods"]
        }
        groups["ratio_full"] = sorted(
            [
                r
                for r in base["runs"]
                if r["block"] == block and r["method"] == args.baseline_method
            ],
            key=lambda r: r["seed"],
        )
        assert len(groups["ratio_full"]) == 3
        for method in ["ratio90", "random90"]:
            if method not in groups:
                groups[method] = sorted(
                    [
                        r
                        for r in control_report["runs"]
                        if r["block"] == block and r["method"] == method
                    ],
                    key=lambda r: r["seed"],
                )
                assert len(groups[method]) == 3
        result["methods"][block] = {}
        for method, runs in groups.items():
            if all(r["status"] == "complete" for r in runs):
                stat = average(runs)
                result["methods"][block][method] = stat
                rows.append(
                    f"| {block} | {method} | {stat['registered_images']:.1f} | {stat['sparse_points']:.0f} | {stat['mean_reprojection_error_px']:.4f} |"
                )
            else:
                result["methods"][block][method] = {"status": "failed_runs_present", "runs": runs}
                rows.append(f"| {block} | {method} | failure retained | — | — |")
        reference = read_json(base_root / block / "reference_poses_v1.json")
        result["comparisons"][block] = {}
        for method in ["mlp90", "tight90", "segformer90", "tight_segformer90", "precision90"]:
            if method not in groups:
                continue
            result["comparisons"][block][method] = {}
            for control in ["ratio_full", "ratio90", "random90"]:
                if all(r["status"] == "complete" for r in groups[method] + groups[control]):
                    c = compare_runs(groups[control], groups[method], reference)
                    if c["status"] == "ok":
                        c["supported_improvement"] = bool(
                            c["registration_not_lower"] and c["paired_ci95_delta"][1] < 0
                        )
                    result["comparisons"][block][method][control] = c
                else:
                    result["comparisons"][block][method][control] = {
                        "status": "failed_runs_present"
                    }
    rows += [
        "",
        "## Common-camera pose comparisons",
        "",
        "ATE after minus before uses 500 paired four-camera cluster bootstrap draws, refitting Sim3 every draw and averaging identical draws across seeds. Report registration separately. Intervals are conditional on fixed regions and reconstructed networks; no correction for the multiple ablation comparisons is applied.",
        "",
        "| Block | Learned scorer | Control | Common cameras | ATE control → learned | Reduction | Paired 95% delta CI | No lost cameras |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for block, methods in result["comparisons"].items():
        for method, controls in methods.items():
            for control, c in controls.items():
                if c["status"] == "ok":
                    low, high = c["paired_ci95_delta"]
                    rows.append(
                        f"| {block} | {method} | {control} | {c['common_cameras']} | {c['ate_before']:.6f} → {c['ate_after']:.6f} | {100*c['relative_ate_reduction']:.2f}% | [{low:.6f}, {high:.6f}] | {c['registration_not_lower']} |"
                    )
                else:
                    rows.append(
                        f"| {block} | {method} | {control} | — | {c['status']} | — | — | — |"
                    )
    rows += [
        "",
        "## Interpretation and reproducibility",
        "",
        "A geometry improvement requires camera coverage to be maintained and lower ATE against both the full baseline and matched-budget controls. A lower point estimate with a CI crossing zero is inconclusive. Increased point counts, improved match AP or lower calibration error alone do not establish geometry improvement. These tests do not measure dense-mesh accuracy or generalization to an unseen acquisition.",
        "",
        "The standalone benchmark is the exact numerical reproduction entry point. Conditional rankers have ten inputs and contrast0.01; older tight rankers have twelve inputs and contrast0.04. They are different checkpoints and must not be interchanged. All outputs are ranking scores, not calibrated probabilities. Existing v1 filters and SegFormer remain optional components. Historical endpoint orchestration samples calibrated probability maps; controlled SegFormer scripts sample raw logits before sigmoid. The default threshold path remains unchanged.",
        "",
        "```powershell",
        (
            ".venv\\Scripts\\python.exe scripts/benchmark_conditional_precision.py --training <conditional-or-observability-training-folder> --output <new-run-folder>"
            if "precision90" in current["methods"]
            else ".venv\\Scripts\\python.exe scripts/benchmark_geometry_ranking.py --ranker <checkpoint> --output <new-run-folder>"
        ),
        "```",
        "",
        "Both outputs must be new. `benchmark_geometry_ranking.py --ranker <checkpoint>` selects the independently stored ranker. `--expanded` on the training script uses all 226 v1 training images and 65 validation images. `benchmark_segformer_strong_baseline.py` performs the contrast0.01 comparison without either learned MLP. All reports, models and run directories from this assessment are new; no commit or push is performed.",
    ]
    path = args.input / "assessment_v1.json"
    assert not path.exists()
    write_json(path, result)
    report_path = args.report
    assert not report_path.exists()
    report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    for block, methods in result["comparisons"].items():
        print(
            block,
            {
                method: {
                    control: {
                        k: v
                        for k, v in c.items()
                        if k
                        in {
                            "ate_before",
                            "ate_after",
                            "relative_ate_reduction",
                            "paired_ci95_delta",
                            "registration_not_lower",
                            "supported_improvement",
                        }
                    }
                    for control, c in controls.items()
                }
                for method, controls in methods.items()
            },
        )


if __name__ == "__main__":
    main()
