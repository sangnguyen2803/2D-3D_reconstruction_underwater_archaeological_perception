"""Reproducible E0-E8 suite; fresh immutable root, --plan-only for review."""

import argparse
import subprocess
import sys
from pathlib import Path

from underwater_vision.matchability.protocol import new_run
from underwater_vision.utils.io import write_json


def commands(root, epochs=4, include_labels=True, dataset=None):
    dataset = dataset or root / "labels_v3"
    result = []
    if include_labels:
        result += [
            ["scripts/build_matchability_dataset.py", "--output", str(dataset)],
            ["scripts/audit_matchability_labels.py", "--dataset", str(dataset), "--freeze"],
        ]
    for fold in range(3):
        protocol = f"fold{fold}"
        for seed in [42, 43, 44]:
            for model in ["resnet_unet", "segformer_b0", "dino_head"]:
                result.append(
                    [
                        "scripts/train_matchability.py",
                        "--dataset",
                        str(dataset),
                        "--protocol",
                        protocol,
                        "--seed",
                        str(seed),
                        "--epochs",
                        str(epochs),
                        "--model",
                        f"configs/matchability/model/{model}.yaml",
                        "--output",
                        str(root / f"{model}_{protocol}_s{seed}_v3"),
                    ]
                )
            result.append(
                [
                    "scripts/train_tabular_baseline.py",
                    "--dataset",
                    str(dataset),
                    "--protocol",
                    protocol,
                    "--seed",
                    str(seed),
                    "--output",
                    str(root / f"baselines_{protocol}_s{seed}_v3"),
                ]
            )
            result.append(
                [
                    "scripts/train_match_filter.py",
                    "--matchability-features",
                    "on",
                    "--matchability-checkpoint",
                    str(root / f"segformer_b0_{protocol}_s{seed}_v3/model_v3.pt"),
                    "--matchability-dataset",
                    str(dataset),
                    "--protocol",
                    protocol,
                    "--seed",
                    str(seed),
                    "--output",
                    str(root / f"filters_{protocol}_s{seed}_v3"),
                ]
            )
            result.append(
                [
                    "scripts/train_point_confidence.py",
                    "--matchability-features",
                    "--matchability-checkpoint",
                    str(root / f"segformer_b0_{protocol}_s{seed}_v3/model_v3.pt"),
                    "--matches",
                    str(root / f"filters_{protocol}_s{seed}_v3"),
                    "--seed",
                    str(seed),
                    "--output",
                    str(root / f"confidence_{protocol}_s{seed}_v3"),
                ]
            )
    result += [
        [
            "scripts/evaluate_matchability.py",
            "--suite",
            str(root),
            "--output",
            str(root / "comparison_v3"),
        ],
        [
            "scripts/export_onnx.py",
            "--task",
            "matchability",
            "--checkpoint",
            str(root / "segformer_b0_fold0_s42_v3/model_v3.pt"),
            "--output",
            str(root / "deployment_v3"),
        ],
        ["scripts/run_matchability_sfm.py", "--suite", str(root), "--output", str(root / "sfm_v3")],
    ]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--dataset", type=Path, help="Reuse an immutable completed E0 dataset")
    a = p.parse_args()
    root = new_run(a.output)
    plan = commands(root, a.epochs, a.dataset is None, a.dataset)
    write_json(
        root / "suite_plan_v3.json",
        {
            "status": "plan_only" if a.plan_only else "running",
            "commands": plan,
            "seeds": [42, 43, 44],
            "folds": 3,
            "labels": "reference BA geometry, not surveyed truth",
        },
    )
    if not a.plan_only:
        for i, command in enumerate(plan):
            print(f"Suite {i + 1}/{len(plan)}: {' '.join(command)}", flush=True)
            subprocess.run([sys.executable, *command], check=True)
        write_json(root / "suite_status_v3.json", {"status": "completed"})


if __name__ == "__main__":
    main()
