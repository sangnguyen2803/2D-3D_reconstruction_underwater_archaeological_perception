"""Measured comparisons across seeds/folds; preserve negative and missing runs."""

from pathlib import Path

import numpy as np

from underwater_vision.evaluation.matchability import binned_analysis, paired_ap_bootstrap
from underwater_vision.matchability.build import load_features
from underwater_vision.matchability.protocol import digest, new_run, read_json
from underwater_vision.utils.io import write_json

SEEDS = [42, 43, 44]
BACKBONES = ["resnet_unet", "segformer_b0", "dino_head"]
BASELINES = [
    "random",
    "response",
    "scale_response",
    "handcrafted",
    "dino_mlp",
    "tabular_logistic",
    "tabular_boosting",
]


def distribution(values):
    values = [float(v) for v in values if v is not None]
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
        "values": values,
    }


def mean_bootstrap(reports):
    complete = [r for r in reports if r.get("status", "ok") == "ok"]
    if len(complete) != 3:
        return {"status": "incomplete_three_seed_comparison"}
    draws = np.mean([r["bootstrap_differences"] for r in complete], axis=0)
    return {
        "status": "ok",
        "estimate": float(np.mean([r["estimate"] for r in complete])),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "seed_count": 3,
        "clusters": complete[0].get("clusters", complete[0].get("tracks")),
        "method": "Mean of three seeded model AP/AUSE differences; common paired cluster resamples within each spatial fold",
        "bootstrap_differences": draws.tolist(),
    }


def arrays(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def dense_folder(suite, model, protocol, seed):
    original = suite / f"{model}_{protocol}_s{seed}_v3"
    corrected = suite / f"{model}_{protocol}_s{seed}_corrected_v3"
    return corrected if (corrected / "evaluation_v3.json").exists() else original


def suite_dataset(suite):
    plan = read_json(suite / "suite_plan_v3.json")
    for command in plan["commands"]:
        if command[0].endswith("train_matchability.py"):
            return Path(command[command.index("--dataset") + 1])
    raise ValueError("Suite lacks a dataset input manifest")


def aggregate(suite, output):
    suite, root = Path(suite), new_run(output)
    dataset = suite_dataset(suite)
    result = {
        "status": "measured",
        "suite": str(suite),
        "dataset": str(dataset),
        "dataset_manifest_sha256": digest(dataset / "manifest_v3.json"),
        "criteria_sha256": digest("configs/matchability/criteria_v3.yaml"),
        "independent_spatial_test_blocks": 3,
        "seeds": SEEDS,
        "dense_runs": [],
        "baseline_runs": [],
        "filter_runs": [],
        "confidence_runs": [],
        "missing_or_failed": [],
        "paired_dense": {},
        "paired_filters": {},
        "paired_point_ause": {},
    }
    for fold in range(3):
        protocol = f"fold{fold}"
        for seed in SEEDS:
            for model in BACKBONES:
                folder = dense_folder(suite, model, protocol, seed)
                file = folder / "evaluation_v3.json"
                if file.exists():
                    result["dense_runs"].append(
                        {
                            "model": model,
                            "fold": fold,
                            "seed": seed,
                            "path": str(folder),
                            **read_json(file),
                        }
                    )
                else:
                    result["missing_or_failed"].append(str(folder))
            for kind, key in [
                ("baselines", "baseline_runs"),
                ("filters", "filter_runs"),
                ("confidence", "confidence_runs"),
            ]:
                folder = suite / f"{kind}_{protocol}_s{seed}_v3"
                file = folder / "evaluation_v3.json"
                if file.exists():
                    result[key].append(
                        {"fold": fold, "seed": seed, "path": str(folder), **read_json(file)}
                    )
                else:
                    result["missing_or_failed"].append(str(folder))
        baseline_runs = [r for r in result["baseline_runs"] if r["fold"] == fold]
        if len(baseline_runs) == 3:
            strongest = max(
                BASELINES,
                key=lambda name: np.mean(
                    [r["models"][name]["average_precision"] for r in baseline_runs]
                ),
            )
            result["paired_dense"][protocol] = {"strongest_baseline": strongest, "models": {}}
            for model in BACKBONES:
                comparisons = []
                for seed in SEEDS:
                    model_root = dense_folder(suite, model, protocol, seed)
                    baseline_root = suite / f"baselines_{protocol}_s{seed}_v3"
                    if not (model_root / "evaluation_v3.json").exists():
                        continue
                    rows = []
                    for file in sorted((model_root / "predictions").glob("*.npz")):
                        dense, baseline = arrays(file), arrays(
                            baseline_root / "predictions" / file.name
                        )
                        if not np.array_equal(dense["labels"], baseline["labels"]):
                            raise ValueError("Dense/baseline label population mismatch")
                        rows.append(
                            {
                                "labels": dense["labels"],
                                "dense": dense["probability"],
                                "baseline": baseline[strongest],
                            }
                        )
                    comparisons.append(
                        paired_ap_bootstrap(rows, "dense", "baseline", seed=42 + fold)
                    )
                result["paired_dense"][protocol]["models"][model] = mean_bootstrap(comparisons)
        filter_runs = [r for r in result["filter_runs"] if r["fold"] == fold]
        if len(filter_runs) == 3:
            result["paired_filters"][protocol] = {
                "context": mean_bootstrap([r["paired_context_16_minus_12"] for r in filter_runs]),
                "mlp": mean_bootstrap([r["paired_mlp_16_minus_12"] for r in filter_runs]),
            }
        point_runs = [r for r in result["confidence_runs"] if r["fold"] == fold]
        if len(point_runs) == 3:
            result["paired_point_ause"][protocol] = {
                name: mean_bootstrap([r["paired_ause"][name] for r in point_runs])
                for name in ["v1_gaussian_8", "robust_8", "heuristic"]
            }
    result["summary"] = {}
    for model in BACKBONES:
        runs = [r for r in result["dense_runs"] if r["model"] == model]
        result["summary"][model] = {
            metric: distribution([r["test_after"].get(metric) for r in runs])
            for metric in ["average_precision", "roc_auc", "ece", "brier"]
        }
    for name in BASELINES:
        result["summary"][name] = {
            metric: distribution([r["models"][name].get(metric) for r in result["baseline_runs"]])
            for metric in ["average_precision", "roc_auc"]
        }
    for name in ["context_12", "context_16", "mlp_12", "mlp_16"]:
        result["summary"][name] = {
            metric: distribution(
                [r["models"][name]["test"].get(metric) for r in result["filter_runs"]]
            )
            for metric in ["average_precision", "roc_auc", "precision", "recall", "f1"]
        }
    for name in ["v1_gaussian_8", "robust_8", "matchability_10", "heuristic", "view_count"]:
        result["summary"][name] = {
            metric: (
                distribution([r["models"][name].get(metric) for r in result["confidence_runs"]])
                if name not in {"heuristic", "view_count"} or metric != "prediction_mae_px"
                else distribution([])
            )
            for metric in ["ause", "spearman_error", "prediction_mae_px", "interval_coverage90"]
        }
    dense_success = sum(
        r["models"].get("segformer_b0", {}).get("ci95", [-1])[0] > 0
        for r in result["paired_dense"].values()
    )
    filter_success = sum(
        r["context"].get("ci95", [-1])[0] > 0 for r in result["paired_filters"].values()
    )
    q2 = {
        f"fold{fold}": distribution(
            [
                r["test_after"]["ece"]
                for r in result["dense_runs"]
                if r["model"] == "segformer_b0" and r["fold"] == fold
            ]
        )
        for fold in range(3)
    }
    result["criteria"] = {
        "Q1": {
            "status": (
                "met"
                if dense_success >= 2
                else (
                    "not_met"
                    if all(
                        result["paired_dense"]
                        .get(f"fold{f}", {})
                        .get("models", {})
                        .get("segformer_b0", {})
                        .get("status")
                        == "ok"
                        for f in range(3)
                    )
                    else "pending"
                )
            ),
            "positive_CI_folds": dense_success,
            "primary_dense_model": "segformer_b0",
        },
        "Q2": {
            "status": (
                "met"
                if all(r["n"] == 3 and max(r["values"]) <= 0.05 for r in q2.values())
                else "not_met" if all(r["n"] == 3 for r in q2.values()) else "pending"
            ),
            "fold_ece": q2,
            "interpretation": "scores unless ECE <=.05 across all three seeds of every spatial fold",
        },
        "Q3": {
            "status": (
                "met"
                if filter_success >= 2
                else "not_met" if len(result["paired_filters"]) == 3 else "pending"
            ),
            "positive_CI_folds": filter_success,
        },
        "Q4": {"status": "pending", "curves": []},
        "Q5": {"status": "pending", "paired": result["paired_point_ause"]},
    }
    for run in result["filter_runs"]:
        original_budget = Path(run["path"]) / "budgets_v3.json"
        corrected_budget = (
            suite / "budgets_corrected_v3" / Path(run["path"]).name / "budgets_v3.json"
        )
        budget = read_json(corrected_budget if corrected_budget.exists() else original_budget)
        result["criteria"]["Q4"]["curves"].append(
            {"fold": run["fold"], "seed": run["seed"], **budget}
        )
    if len(result["filter_runs"]) == 9:
        wins = []
        for fold in range(3):
            folded = [r for r in result["criteria"]["Q4"]["curves"] if r["fold"] == fold]
            for n in [512, 1024, 2048, 4096]:
                values = {
                    s: [
                        c["original_compatible_candidate_recall"]
                        for r in folded
                        for c in r["curves"]
                        if c["budget"] == n and c["source"] == s and c["pairs"] > 0
                    ]
                    for s in ["matchability", "response", "random"]
                }
                if all(len(v) == 3 for v in values.values()):
                    wins.append(
                        {
                            "fold": fold,
                            "budget": n,
                            "matchability_wins": np.mean(values["matchability"])
                            > max(np.mean(values["response"]), np.mean(values["random"])),
                        }
                    )
        result["criteria"]["Q4"].update(
            status=(
                "inconclusive_missing_budgets"
                if len(wins) < 12
                else "met" if all(r["matchability_wins"] for r in wins) else "not_met"
            ),
            primary_metric="reference_compatible_match_recall (frozen E0 criterion)",
            feasible_comparisons=[
                {**r, "matchability_wins": bool(r["matchability_wins"])} for r in wins
            ],
            limitation="Exact N only where enough keypoints exist at BOTH endpoints; unavailable budgets remain unmeasured",
        )
    if len(result["paired_point_ause"]) == 3:
        pooled = {}
        rng = np.random.default_rng(177)
        for name in ["v1_gaussian_8", "robust_8", "heuristic"]:
            reports = [result["paired_point_ause"][f"fold{f}"][name] for f in range(3)]
            if all(r["status"] == "ok" for r in reports):
                draws = np.mean(
                    [rng.permutation(r["bootstrap_differences"]) for r in reports], axis=0
                )
                pooled[name] = {
                    "estimate": float(np.mean([r["estimate"] for r in reports])),
                    "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
                }
        result["criteria"]["Q5"].update(
            status=(
                "met"
                if all(
                    pooled.get(n, {}).get("ci95", [0, 1])[1] < 0
                    for n in ["v1_gaussian_8", "heuristic"]
                )
                else "not_met"
            ),
            equally_weighted_fold_mean=pooled,
            limitation="Conditional track-bootstrap CI for the three fixed regions; no new-scene generalization claim",
        )
    previous = {
        "v1_context_AP": 0.945910,
        "v1_point_AUSE": 0.491018,
        "v1_heuristic_AUSE": 0.419388,
        "v1_context_SfM": {"images": 16, "points": 1890, "reprojection_px": 0.4504205128199975},
        "comparison_caveat": "Rotated-fold match/point populations differ from historical development v1; causal comparisons use retrained baselines on exactly the v3 populations",
    }
    result["historical_v1"] = previous
    compatible = Path("outputs/v1_compatibility_v3/metrics.json")
    if compatible.exists():
        m = read_json(compatible)
        result["compatibility"] = {
            "registered_images": m["registered_images"],
            "points": m["sparse_points"],
            "reprojection_px": m["mean_reprojection_error_px"],
            "absolute_reprojection_difference": abs(
                m["mean_reprojection_error_px"] - previous["v1_context_SfM"]["reprojection_px"]
            ),
            "passed": m["registered_images"] == 16
            and m["sparse_points"] == 1890
            and abs(m["mean_reprojection_error_px"] - previous["v1_context_SfM"]["reprojection_px"])
            < 1e-12,
        }
    for name, file in [
        ("deployment", suite / "deployment_v3/deployment_v3.json"),
        ("sfm", suite / "sfm_v3/evaluation_v3.json"),
    ]:
        if file.exists():
            result[name] = read_json(file)
    result["status"] = (
        "measured_complete"
        if not result["missing_or_failed"] and "sfm" in result and "deployment" in result
        else "measured_partial"
    )
    corrected = suite / "dino_head_fold0_s43_corrected_v3/evaluation_v3.json"
    approximate = suite / "dino_head_fold0_s43_v3/evaluation_v3.json"
    if approximate.exists():
        result["superseded_or_experimental"] = [
            {
                "path": str(approximate.parent),
                "reason": "Early BF16 pre-head cache failed strict parity; retained for traceability",
                "superseded": corrected.exists(),
            }
        ]
    for name, file in [
        (
            "cache_lossless_audit",
            Path("outputs/matchability_frozen_cache_lossless_retry_v3/parity_v3.json"),
        ),
        ("cache_failed_audit", Path("outputs/matchability_frozen_cache_audit_v3/parity_v3.json")),
        ("docker_smoke", Path("outputs/matchability_docker_smoke_v3/api_smoke_v3.json")),
    ]:
        if file.exists():
            result[name] = read_json(file)
    write_json(root / "comparison_v3.json", result)
    write_binned_analysis(result, root, dataset)
    return result


def write_binned_analysis(result, root, dataset):
    reports = {"dense": [], "filters": [], "confidence": []}
    for run in result["dense_runs"]:
        if run["model"] != "segformer_b0" or run["seed"] != 42:
            continue
        rows = [arrays(f) for f in sorted((Path(run["path"]) / "predictions").glob("*.npz"))]
        features = [load_features(dataset, str(r["name"])) for r in rows]
        y, p = np.concatenate([r["labels"] for r in rows]), np.concatenate(
            [r["probability"] for r in rows]
        )
        reports["dense"].append(
            {
                "fold": run["fold"],
                "seed": 42,
                "bins": binned_analysis(
                    y,
                    p,
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
    for run in result["filter_runs"]:
        if run["seed"] != 42:
            continue
        folder = Path(run["path"])
        records = [arrays(f) for f in sorted((folder / "pairs").glob("test_*.npz"))]
        predictions = [arrays(f) for f in sorted((folder / "predictions").glob("*.npz"))]
        covariates = {
            k: np.concatenate([r[k] for r in records])
            for k in ["contrast", "scale", "sharpness", "brightness"]
        }
        xy = np.concatenate([r["x"][:, 6:8] * [960, 720] for r in records])
        stats = np.c_[np.zeros(len(xy)), covariates["scale"], covariates["contrast"]]
        y = np.concatenate([p["labels"] for p in predictions])
        reports["filters"].append(
            {
                "fold": run["fold"],
                "seed": 42,
                "models": {
                    name: binned_analysis(
                        y,
                        np.concatenate([p[name] for p in predictions]),
                        stats,
                        xy,
                        (720, 960),
                        {
                            "sharpness": covariates["sharpness"],
                            "brightness": covariates["brightness"],
                        },
                    )
                    for name in ["context_12", "context_16"]
                },
            }
        )
    for run in result["confidence_runs"]:
        if run["seed"] == 42:
            file = Path(run["path"]) / "binned_confidence_v3.json"
            reports["confidence"].append(
                {
                    "fold": run["fold"],
                    "seed": 42,
                    "bins": read_json(file) if file.exists() else {"status": "unmeasured"},
                }
            )
    write_json(root / "binned_analysis_v3.json", reports)
