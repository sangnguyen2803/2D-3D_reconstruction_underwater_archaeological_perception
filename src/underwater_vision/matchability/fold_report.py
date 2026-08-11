"""Complete primary-model spatial benchmark with exact paired image-bootstrap CIs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from underwater_vision.evaluation.matchability import binned_analysis, paired_ap_bootstrap
from underwater_vision.matchability.build import load_features
from underwater_vision.matchability.protocol import digest, new_run, read_json
from underwater_vision.matchability.report import BASELINES, arrays, distribution, mean_bootstrap
from underwater_vision.utils.io import write_json


def collect(suite, output):
    suite, root = Path(suite), new_run(output)
    manifest = read_json(suite / "fold_manifest_v3.json")
    dataset = Path(manifest["dataset"])
    data_manifest = read_json(dataset / "manifest_v3.json")
    if digest(dataset / "manifest_v3.json") != manifest["dataset_manifest_sha256"]:
        raise ValueError("Dataset changed during benchmark")
    if digest("configs/matchability/criteria_v3.yaml") != manifest["criteria_sha256"]:
        raise ValueError("Frozen criteria changed during benchmark")
    runs = {}
    for entry in manifest["runs"]:
        path = Path(entry["path"])
        metric = read_json(path / "evaluation_v3.json")
        if metric["status"] != "complete":
            raise ValueError("Incomplete fold/seed run")
        if entry["reused"] and digest(path / "evaluation_v3.json") != entry["evaluation_sha256"]:
            raise ValueError("Preserved evaluation changed")
        key = (entry["kind"], entry["fold"], entry["seed"])
        if key in runs:
            raise ValueError("Duplicate fold/seed")
        runs[key] = {
            **entry,
            "evaluation_sha256": digest(path / "evaluation_v3.json"),
            "metric": metric,
        }
    expected = {
        (kind, fold, seed)
        for kind in ["segformer_b0", "baselines"]
        for fold in range(3)
        for seed in [42, 43, 44]
    }
    if set(runs) != expected:
        raise ValueError("Expected exactly nine dense and nine baseline runs")
    report = {
        "status": "completed_E1_E3_primary_model",
        "scope": manifest["scope"],
        "primary_backbone": "segformer_b0",
        "epochs": 4,
        "folds": [0, 1, 2],
        "seeds": [42, 43, 44],
        "independent_spatial_regions": 3,
        "independent_scenes": 1,
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "criteria_sha256": manifest["criteria_sha256"],
        "evaluation_source_sha256": {
            str(p): digest(p)
            for p in [
                Path(__file__),
                Path("src/underwater_vision/evaluation/matchability.py"),
                Path("src/underwater_vision/matchability/report.py"),
            ]
        },
        "run_provenance": [{k: v for k, v in r.items() if k != "metric"} for r in runs.values()],
        "per_fold": {},
        "overall": {},
        "paired": {},
        "criteria": {},
        "statistics": "500 paired image-cluster percentile bootstrap draws; exact weighted AP with ties pooled. Same image draws across the three model seeds, then average their AP differences. CIs are conditional on these fixed spatial regions, not new-scene uncertainty.",
        "baseline_selection": "Strongest baseline is chosen by highest mean test AP over the three seeds in each fold; reported CIs condition on that choice. No test result fits model weights or calibration.",
        "downstream_state": "E4-E7 are not completed by this E1/E3 benchmark; previous partial downstream results remain separate.",
    }
    binned = []
    for fold in range(3):
        models = {name: [] for name in ["segformer_b0", *BASELINES]}
        calibration = []
        split = next(p for p in data_manifest["protocols"] if p["name"] == f"fold{fold}")
        for seed in [42, 43, 44]:
            dense = runs[("segformer_b0", fold, seed)]["metric"]
            baseline = runs[("baselines", fold, seed)]["metric"]
            models["segformer_b0"].append({"seed": seed, **dense["test_after"]})
            for name in BASELINES:
                models[name].append({"seed": seed, **baseline["models"][name]})
            calibration.append(
                {
                    "seed": seed,
                    "temperature": dense["temperature"],
                    "best_validation_ap": dense["best_validation_ap"],
                    "history": dense["history"],
                    "before": dense["test_before"],
                    "after": dense["test_after"],
                }
            )
        strongest = max(
            BASELINES, key=lambda n: np.mean([r["average_precision"] for r in models[n]])
        )
        tabular = max(
            ["tabular_logistic", "tabular_boosting"],
            key=lambda n: np.mean([r["average_precision"] for r in models[n]]),
        )
        report["per_fold"][f"fold{fold}"] = {
            "split_counts": {k: len(v) for k, v in split["groups"].items()},
            "models": {
                name: {
                    key: distribution([r.get(key) for r in values])
                    for key in ["average_precision", "roc_auc", "ece", "brier"]
                }
                for name, values in models.items()
            },
            "strongest_baseline": strongest,
            "strongest_tabular": tabular,
            "dense_seeds": models["segformer_b0"],
            "calibration": calibration,
        }
        report["paired"][f"fold{fold}"] = {}
        compare = list(dict.fromkeys([strongest, tabular, "dino_mlp"]))
        per_seed = {name: [] for name in compare}
        for seed in [42, 43, 44]:
            dense_root = Path(runs[("segformer_b0", fold, seed)]["path"])
            baseline_root = Path(runs[("baselines", fold, seed)]["path"])
            dense_files = sorted((dense_root / "predictions").glob("*.npz"))
            expected_names = set(split["groups"]["test"])
            if {f.name.removesuffix(".npz") for f in dense_files} != expected_names:
                raise ValueError("Prediction image population differs from test slab")
            if {
                f.name.removesuffix(".npz") for f in (baseline_root / "predictions").glob("*.npz")
            } != expected_names:
                raise ValueError("Baseline test population differs")
            rows = []
            for file in dense_files:
                dense, baseline = arrays(file), arrays(baseline_root / "predictions" / file.name)
                name = str(dense["name"])
                if str(baseline["name"]) != name or not np.array_equal(
                    dense["labels"], baseline["labels"]
                ):
                    raise ValueError("Dense/baseline labels or image identifiers differ")
                with np.load(dataset / "labels" / f"fold{fold}" / "hybrid" / (name + ".npz")) as z:
                    if not np.array_equal(dense["labels"], z["l1"]):
                        raise ValueError("Saved test labels differ from immutable L1")
                rows.append(
                    {
                        "labels": dense["labels"],
                        "dense": dense["probability"],
                        **{n: baseline[n] for n in compare},
                    }
                )
            for name in compare:
                print(f"Paired AP bootstrap fold{fold} seed{seed} vs {name}", flush=True)
                per_seed[name].append(paired_ap_bootstrap(rows, "dense", name, seed=42 + fold))
            if seed == 42:
                features = [load_features(dataset, str(arrays(f)["name"])) for f in dense_files]
                labels, score = np.concatenate([r["labels"] for r in rows]), np.concatenate(
                    [r["dense"] for r in rows]
                )
                binned.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "bins": binned_analysis(
                            labels,
                            score,
                            np.concatenate([f["statistics"] for f in features]),
                            np.concatenate([f["xy"] for f in features]),
                            (720, 960),
                            {
                                "sharpness": np.concatenate([f["sharpness"] for f in features]),
                                "brightness": np.concatenate(
                                    [np.full(len(f["xy"]), f["brightness"]) for f in features]
                                ),
                            },
                        ),
                    }
                )
        for name in compare:
            report["paired"][f"fold{fold}"][name] = {
                "mean_three_seeds": mean_bootstrap(per_seed[name]),
                "per_seed": per_seed[name],
            }
    for name in ["segformer_b0", *BASELINES]:
        report["overall"][name] = {
            metric: distribution(
                [
                    v
                    for fold in report["per_fold"].values()
                    for v in fold["models"][name][metric]["values"]
                ]
            )
            for metric in ["average_precision", "roc_auc", "ece", "brier"]
        }
    successes = []
    for fold, stats in report["per_fold"].items():
        comparison = report["paired"][fold][stats["strongest_baseline"]]["mean_three_seeds"]
        if comparison.get("status") != "ok":
            raise ValueError("Failed paired three-seed comparison")
        if comparison["ci95"][0] > 0:
            successes.append(fold)
    report["criteria"]["Q1"] = {
        "status": "met" if len(successes) >= 2 else "not_met",
        "positive_CI_folds": successes,
        "required_positive_folds": 2,
    }
    failures = [
        {"fold": fold, "seed": row["seed"], "ece": row["ece"]}
        for fold, stats in report["per_fold"].items()
        for row in stats["dense_seeds"]
        if row["ece"] > 0.05
    ]
    report["criteria"]["Q2"] = {
        "status": "met" if not failures else "not_met",
        "maximum_ece": 0.05,
        "failures": failures,
        "rule": "Every seed of each spatial fold must satisfy the bound; calibration uses validation only",
    }
    report["criteria"].update(
        {key: {"status": "not_evaluated_by_this_run"} for key in ["Q3", "Q4", "Q5"]}
    )
    write_json(root / "comparison_v3.json", report)
    write_json(root / "binned_analysis_v3.json", binned)
    return report


def markdown_table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def save_figures(report, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    names = ["segformer_b0", *BASELINES]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for (fold, stats), ax in zip(report["per_fold"].items(), axes, strict=True):
        ap = [stats["models"][n]["average_precision"] for n in names]
        ax.barh(
            names,
            [s["mean"] for s in ap],
            xerr=[s["std"] for s in ap],
            color=["#2378a7"] + ["#89979e"] * 7,
        )
        ax.set(title=f"{fold}: 3 seeds", xlabel="Test AP, mean ± seed SD", xlim=(0, 1))
    fig.tight_layout()
    fig.savefig(root / "benchmark_v3.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for (fold, stats), ax in zip(report["per_fold"].items(), axes, strict=True):
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
        for phase, style in [("before", "--"), ("after", "-")]:
            for record in stats["calibration"]:
                bins = record[phase]["calibration"]
                ax.plot(
                    [b["predicted"] for b in bins],
                    [b["observed"] for b in bins],
                    linestyle=style,
                    label=f"seed{record['seed']} {phase}",
                )
        ax.set(
            title=fold,
            xlabel="Predicted score",
            ylabel="Observed positive fraction",
            xlim=(0, 1),
            ylim=(0, 1),
        )
    axes[-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(root / "calibration_v3.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for (fold, stats), ax in zip(report["per_fold"].items(), axes, strict=True):
        for run in stats["dense_seeds"]:
            curve = run["retention"]
            ax.plot(
                curve["keypoint_fraction"], curve["positive_recall"], label=f"seed{run['seed']}"
            )
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="uniform retention")
        ax.set(
            title=fold,
            xlabel="Labeled keypoints retained",
            ylabel="Positives retained",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "retention_v3.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (fold, stats) in enumerate(report["per_fold"].items()):
        comparison = report["paired"][fold][stats["strongest_baseline"]]["mean_three_seeds"]
        ax.plot([i, i], comparison["ci95"], color="#2378a7")
        ax.scatter([i], [comparison["estimate"]], color="#2378a7")
    ax.axhline(0, color="grey", linestyle="--")
    ax.set(
        xticks=range(3),
        xticklabels=list(report["per_fold"]),
        ylabel="SegFormer AP − strongest baseline AP",
        title="Mean of 3 seeds; paired 95% image-bootstrap CI",
    )
    fig.tight_layout()
    fig.savefig(root / "paired_ap_v3.png", dpi=170)
    plt.close(fig)
    return root


def write_report(report, path, figures, comparison):
    def fmt(value):
        return f"{value:.6f}"

    rows, calibration, differences = [], [], []
    for fold, stats in report["per_fold"].items():
        name = stats["strongest_baseline"]
        for run in stats["dense_seeds"]:
            rows.append(
                [
                    fold,
                    run["seed"],
                    fmt(run["average_precision"]),
                    fmt(run["roc_auc"]),
                    fmt(run["ece"]),
                    fmt(run["brier"]),
                ]
            )
        ci = report["paired"][fold][name]["mean_three_seeds"]
        differences.append(
            [
                fold,
                fmt(stats["models"]["segformer_b0"]["average_precision"]["mean"]),
                name,
                fmt(stats["models"][name]["average_precision"]["mean"]),
                fmt(ci["estimate"]),
                str(ci["ci95"]),
            ]
        )
        for run in stats["calibration"]:
            calibration.append(
                [
                    fold,
                    run["seed"],
                    fmt(run["temperature"]),
                    fmt(run["before"]["ece"]),
                    fmt(run["after"]["ece"]),
                    fmt(run["before"]["brier"]),
                    fmt(run["after"]["brier"]),
                ]
            )
    summary_rows = [
        [
            name,
            fmt(s["average_precision"]["mean"]),
            fmt(s["average_precision"]["std"]),
            fmt(s["roc_auc"]["mean"]),
        ]
        for name, s in report["overall"].items()
    ]
    text = f"""# SegFormer-B0 three-fold benchmark v3

Completed primary dense-matchability E1/E3: **3 spatial folds × 3 seeds**, with all seven controls evaluated on each matching fold/seed; trainable controls are refitted on that fold's training views. Four-epoch budgets, labels, thresholds and spatial partitions are unchanged. Two completed fold0 SegFormer and two baseline runs were reused after verifying configurations, dataset/model hashes and prediction populations. The interrupted seed44 was rerun in a new directory. Old results/reports remain preserved.

## Pre-registered outcomes

Q1: **{report['criteria']['Q1']['status']}**, positive paired-CI regions: {report['criteria']['Q1']['positive_CI_folds']}; required at least two of three. Q2: **{report['criteria']['Q2']['status']}**, requiring ECE ≤0.05 for every seed of each region. The exact failing seeds are recorded in the comparison JSON. Q3/Q4/Q5 are not evaluated by this run; the earlier negative/partial filter and confidence results remain separate evidence.

## Per-seed test results

{markdown_table(['Fold','Seed','AP','ROC AUC','ECE after','Brier after'], rows)}

## Comparison with the strongest controls

{markdown_table(['Fold','SegFormer mean AP','Strongest control','Control mean AP','AP difference','Paired 95% CI'], differences)}

The strongest control is selected by mean test AP across three seeds in that region, including tabular methods. CIs are conditional on this choice. The saved JSON additionally compares against the strongest tabular control and fixed DINO-MLP control. Each seed comparison uses identical original L1 test labels/keypoints and images, verified against immutable labels. No test labels fit model weights, temperature or normalization.

## Equal-weight summary across the nine runs

{markdown_table(['Method','Mean AP','SD across fold/seed runs','Mean ROC AUC'], summary_rows)}

The SD combines region and seed variation; it is not a confidence interval. There are **three spatial regions in one scene**, not nine independent regions or three scenes. AP is average precision, not accuracy. All methods share each fold's same known-label population. Fold prevalences differ, so paired within-fold differences are the primary comparison.

## Calibration before/after validation-only temperature

{markdown_table(['Fold','Seed','Temperature','ECE before','ECE after','Brier before','Brier after'], calibration)}

Temperature is fitted on spatial validation only, after validation-AP checkpoint selection. Useful ranking and calibrated probabilities are separate findings. If Q2 fails, describe predictions as matchability scores rather than universally calibrated probabilities.

## Figures and reproducibility

![Fold AP comparison](figures_v3/segformer_folds_v3/benchmark_v3.png)

![Paired AP differences](figures_v3/segformer_folds_v3/paired_ap_v3.png)

![Calibration](figures_v3/segformer_folds_v3/calibration_v3.png)

![Retention](figures_v3/segformer_folds_v3/retention_v3.png)

Comparison: `{comparison}`. Run/source/criteria/dataset hashes and reused run references: `outputs/segformer_folds_v3/fold_manifest_v3.json`. Binned contrast, scale, sharpness, brightness and frame-radius results are saved for seed42 in each fold. Paired differences use 500 exact weighted-AP image-cluster resamples, aligned across three seeds before averaging. They condition on the fixed regions; nearby images and shared surfaces can remain correlated, so the image bootstrap may understate broader spatial uncertainty. New-scene generalization is unmeasured.

## Remaining scope

This completes the requested three-fold **SegFormer matchability** comparison. It does not complete downstream filter/point-confidence folds, L2/soft training, new SfM/MVS comparisons or deployment latency. Labels describe RootSIFT/top1 epipolar compatibility using published BA cameras, not independently surveyed match truth. Spatial camera gaps do not ensure disjoint surfaces. Sparse supervision does not validate every pixel or another detector/scene. Refraction remains unmodeled. Four epochs are the fixed comparison budget and do not establish convergence.
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(text)
    write_json(
        figures / "provenance_v3.json",
        {
            "comparison_sha256": digest(comparison),
            "report_sha256": digest(path),
            "regions": 3,
            "seeds": [42, 43, 44],
        },
    )


def evaluate(suite, output, report_path, figure_root):
    report = collect(suite, output)
    figures = save_figures(report, figure_root)
    write_report(report, report_path, figures, Path(output) / "comparison_v3.json")
    print(
        {
            "status": report["status"],
            "Q1": report["criteria"]["Q1"],
            "Q2": report["criteria"]["Q2"],
        },
        flush=True,
    )
    return report
