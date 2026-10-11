"""Create a complete contribution report without discarding failed ablations."""

from pathlib import Path

import numpy as np

from underwater_vision.matchability.protocol import digest, read_json


def main():
    families = [
        ("Original-budget ranking", "geometry_ranking_v1"),
        ("Expanded precision MLP", "geometry_ranking_expanded_v1"),
        ("Expanded precision context", "geometry_context_ranking_v1"),
        ("SegFormer / strong detector", "segformer_strong_baseline_v1"),
        ("Conditional precision", "conditional_precision_geometry_v1"),
        ("Triangulation utility", "observability_geometry_v1"),
    ]
    reports = [
        (label, name, read_json(Path("outputs") / name / "comparison_v1.json"))
        for label, name in families
    ]
    assert all(
        r["status"] == "complete" and all(x["status"] == "complete" for x in r["runs"])
        for _, _, r in reports
    )
    count = sum(len(r["runs"]) for _, _, r in reports)
    baseline = read_json("outputs/detector_coverage_v1/comparison_v1.json")
    strong = reports[3][2]
    conditional = reports[4][2]
    observability = reports[5][2]
    rotation = read_json("outputs/conditional_precision_geometry_v1/secondary_rotation_v1.json")
    rows = [
        "# Learned 3D Contribution: Improvement and Limitations v1",
        "",
        "## Decision",
        "",
        "A broad learned-model improvement over the strongest classical reconstruction is **not established**. The new training and controlled evaluation provide two localized geometric gains, with adverse results retained. Keep low-contrast RootSIFT as the reconstruction reference and keep learned ranking optional. Do not replace the default with a model selected from the best test row.",
        "",
        "- Conditional context filtering reduces median camera rotation error on block599 from 1.6123 to 1.3228 degrees (17.95%), with 32/32 registered views and improvements against ratio/random controls. This is a secondary exploratory metric; ATE does not improve there.",
        "- Frozen SegFormer endpoint ranking on the original contrast 0.04 detector reduces block599 ATE by 2.36% against the full baseline, 5.59% against matched-budget ratio ranking and 4.38% against random ranking. All three paired unadjusted intervals exclude zero. This is one previously examined 32-image region, with 22 registered cameras, and the gain does not transfer to the stronger contrast 0.01 baseline.",
        "",
        "## What changed",
        "",
        "1. Conservative match-budget selection preserves every detector keypoint and ranks only mutual ratio 0.8 survivors. Every pair retains 90% or at least 128 candidates when available. Equal per-pair counts across controls isolate ranking from budget reduction.",
        "2. Reference precision supervision uses a soft 0.5px epipolar-error target, not RANSAC labels. Original-pair, expanded MLP and expanded context models were tested separately. Expanded training uses 226 train views and 65 validation views.",
        "3. A fresh conditional ten-input model learns within the strong contrast 0.01 survivor distribution: 455,158 matches / 388 train pairs; 221 validation pairs. Features are descriptor distance/ratio, local reliability and normalized pair coordinates. No DINO or SegFormer score enters it. Joint axis augmentation and validation-only architecture/checkpoint selection select context epoch 4.",
        "4. A further observability model warm-starts that context checkpoint. Its train/validation target multiplies epipolar precision by min(1,(sin(reference ray angle)/sin(4deg))^2). The 4-degree scale comes from the existing mapper initialization angle, and validation retained utility selects among ten epochs. This is a reference-derived triangulation-information proxy, not measured 3D point error.",
        "5. Optional profiles and isolated checkpoint folders preserve the existing v1 threshold path. Ten-input conditional checkpoints use standalone benchmark entry points; they are not interchangeable with the older twelve-input filters.",
        "",
        "## Completed experiment inventory",
        "",
        f"**{count} new complete sparse reconstruction runs** across six experiment families, three blocks and mapper seeds42/43/44. This count excludes the previous 54-run baseline and 9-run detector experiment. There are 96 distinct evaluated RGB images, not 126 distinct scenes or a full 1,244-view reconstruction.",
        "",
        "| Family | New runs | Artifact |",
        "|---|---:|---|",
    ]
    for label, name, report in reports:
        rows.append(
            f"| {label} | {len(report['runs'])} | `outputs/{name}` / comparison SHA256 `{digest(Path('outputs') / name / 'comparison_v1.json')}` |"
        )
    rows += [
        "",
        "## Strong baseline: primary camera-position accuracy",
        "",
        "All methods below register 32/32 cameras per block/seed. ATE is Sim3-aligned camera-center RMSE in published reference units, not verified metres. Values are three-seed means. Lower is better. Ratio90/random90 share the exact learned match budget; full RootSIFT retains the entire eligible pool.",
        "",
        "| Block | Full RootSIFT .01 | Ratio90 | Random90 | SegFormer90 | Conditional90 | Observability90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for block in ["block273", "block599", "block1116"]:
        columns = [
            (baseline, "low_contrast_ratio"),
            (strong, "ratio90"),
            (strong, "random90"),
            (strong, "segformer90"),
            (conditional, "precision90"),
            (observability, "precision90"),
        ]
        values = [
            np.mean(
                [
                    r["trajectory"]["ate_rmse_reference_units"]
                    for r in report["runs"]
                    if r["block"] == block and r["method"] == method
                ]
            )
            for report, method in columns
        ]
        rows.append("| " + block + " | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    rows += [
        "",
        "Use the paired ATE intervals in [SegFormer strong-baseline assessment](SEGFORMER_STRONG_BASELINE_v1.md), [conditional precision assessment](CONDITIONAL_PRECISION_GEOMETRY_v1.md) and [observability assessment](OBSERVABILITY_GEOMETRY_v1.md). Full registration alone does not establish geometric accuracy. Native reprojection and point counts use differing point populations and cannot replace paired pose evaluation.",
        "",
        "## Secondary angular audit: all regions and controls",
        "",
        "Camera orientations are aligned using the same center-derived Sim3 rotation. The table averages per-seed median angular errors. Intervals use 500 paired circular moving blocks of four cameras, refitting Sim3 in each draw and averaging the same draws across seeds. These are exploratory, conditional intervals without multiple-comparison adjustment, not a primary success declaration.",
        "",
        "| Block | Control | Control degrees | Conditional degrees | Reduction | 95% delta CI degrees |",
        "|---|---|---:|---:|---:|---|",
    ]
    for block, comparisons in rotation["blocks"].items():
        for control, comparison in comparisons.items():
            lo, hi = comparison["paired_ci95_delta_degrees"]
            rows.append(
                f"| {block} | {control} | {comparison['before_median_degrees_seed_mean']:.4f} | {comparison['after_median_degrees_seed_mean']:.4f} | {100*comparison['relative_reduction']:.2f}% | [{lo:.4f}, {hi:.4f}] |"
            )
    rows += [
        "",
        "## Interpretation for interviews and CV",
        "",
        "Keep classification AP, calibration and reconstruction claims separate. AP 0.9473 and SegFormer fold AP 0.84-0.89 remain classification results. The strongest broad coverage/ATE change remains the detector improvement. Learned geometry evidence is currently localized and exploratory.",
        "",
        "A defensible technical statement is: **Developed reference-supervised context filtering and controlled SfM ablations; reduced median camera rotation error by 18% on one 32-view region against a strong RootSIFT baseline, while identifying limited transfer to positional accuracy and other regions.**",
        "",
        "Do not state that learned filtering improves full-dataset 3D reconstruction or dense mesh accuracy. The pose reference is published bundle adjustment, not independently surveyed geometry. Test regions were already examined in earlier development, and many scorers were tested. A new acquisition or genuinely untouched region is required for a confirmatory claim. No new dense-MVS run or ground-truth surface evaluation was performed in this task.",
        "",
        "## Reproduction and next evidence",
        "",
        "Run commands from the repository root with output paths that do not exist. Executed source copies and checkpoint hashes accompany the new conditional/observability experiments. The original-budget benchmark launched before cosmetic formatting; its archived formatted source is not the exact launch bytes, a limitation recorded here.",
        "",
        "```powershell",
        ".venv\\Scripts\\python.exe scripts/train_conditional_precision.py --output outputs/conditional_reproduction_v1",
        ".venv\\Scripts\\python.exe scripts/train_observability_ranker.py --source outputs/conditional_reproduction_v1 --output outputs/observability_reproduction_v1",
        ".venv\\Scripts\\python.exe scripts/benchmark_conditional_precision.py --training outputs/observability_reproduction_v1 --candidate-cache outputs/conditional_precision_geometry_v1 --output outputs/observability_geometry_reproduction_v1",
        "```",
        "",
        "These benchmark scripts intentionally use the existing locked images, pools and control artifacts; the reproduction command is not a general new-dataset benchmark. For a substantive next accuracy claim, reserve an untouched region/acquisition, lock the primary metric and model before evaluation, and supervise held-out reprojection or triangulated 3D error. Do not continue selecting architectures from these same three test blocks.",
        "",
        "Validation: 119 tests passed; Ruff and Black checks passed. Prior summaries and results remain preserved. No commit/push or checkpoint deletion was performed.",
        "",
        "Documentation clarification: generic closing reproduction paragraphs in earlier individual reports refer to the historical twelve-input .04 ranker. The conditional 10-input and strong 0.01 experiments use the distinct scripts and protocols described in this report; their numeric tables remain valid.",
    ]
    path = Path("LEARNED_3D_CONTRIBUTION_v1.md")
    assert not path.exists()
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
