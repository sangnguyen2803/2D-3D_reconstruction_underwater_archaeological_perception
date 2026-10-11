"""Complete SegFormer E1/E3 on three spatial folds; reuse only verified finished runs."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from underwater_vision.matchability.protocol import digest, new_run, read_json
from underwater_vision.utils.io import write_json


def verified_run(folder, kind, protocol, seed, dataset):
    evaluation = folder / "evaluation_v3.json"
    if not evaluation.exists():
        return False
    report = read_json(evaluation)
    manifest = read_json(dataset / "manifest_v3.json")
    split = next(p for p in manifest["protocols"] if p["name"] == protocol)
    if report.get("status") != "complete":
        raise ValueError(f"Incomplete result cannot be reused: {folder}")
    if kind == "segformer_b0":
        config = read_json(folder / "resolved_config_v3.json")
        expected = {
            "protocol": protocol,
            "seed": seed,
            "epochs": 4,
            "variant": "l1",
            "graph": "hybrid",
            "manifest_sha256": digest(dataset / "manifest_v3.json"),
            "criteria_sha256": digest("configs/matchability/criteria_v3.yaml"),
        }
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Dense run differs from frozen protocol: {folder}")
        if config["model_config"]["kind"] != "segformer_b0":
            raise ValueError("Wrong dense backbone")
        if config["train_config"] != yaml.safe_load(
            Path("configs/matchability/train.yaml").read_text()
        ):
            raise ValueError("Training configuration differs from the original runs")
        if config["spatial_split"] != split:
            raise ValueError("Spatial split differs")
        if digest(folder / "model_v3.pt") != report["checkpoint_hash"]:
            raise ValueError("Dense checkpoint hash mismatch")
        names = config["test_images"]
    else:
        if report["protocol"] != protocol or report["seed"] != seed:
            raise ValueError("Baseline fold/seed mismatch")
        if report["manifest_sha256"] != digest(dataset / "manifest_v3.json"):
            raise ValueError("Baseline dataset mismatch")
        if (
            report["training_images"] != split["groups"]["train"]
            or report["validation_images"] != split["groups"]["validation"]
        ):
            raise ValueError("Baseline fitting roles mismatch")
        names = report["test_images"]
    if names != split["groups"]["test"]:
        raise ValueError("Test images differ from the locked split")
    actual = {p.name.removesuffix(".npz") for p in (folder / "predictions").glob("*.npz")}
    if actual != set(names):
        raise ValueError(f"Missing or additional predictions: {folder}")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=Path("outputs/matchability_labels_v3"))
    p.add_argument("--reuse-suite", type=Path, default=Path("outputs/matchability_suite_v3"))
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args()
    root = new_run(a.output)
    runs, commands = [], []
    for fold in range(3):
        protocol = f"fold{fold}"
        for seed in [42, 43, 44]:
            for kind in ["segformer_b0", "baselines"]:
                name = f"{kind}_{protocol}_s{seed}_v3"
                old = a.reuse_suite / name
                reused = verified_run(old, kind, protocol, seed, a.dataset)
                folder = old if reused else root / name
                record = {
                    "kind": kind,
                    "protocol": protocol,
                    "fold": fold,
                    "seed": seed,
                    "path": str(folder),
                    "reused": reused,
                }
                if reused:
                    record["evaluation_sha256"] = digest(folder / "evaluation_v3.json")
                runs.append(record)
                if not reused:
                    script = (
                        "scripts/train_matchability.py"
                        if kind == "segformer_b0"
                        else "scripts/train_tabular_baseline.py"
                    )
                    command = [
                        script,
                        "--dataset",
                        str(a.dataset),
                        "--protocol",
                        protocol,
                        "--seed",
                        str(seed),
                        "--output",
                        str(folder),
                    ]
                    if kind == "segformer_b0":
                        command += [
                            "--model",
                            "configs/matchability/model/segformer_b0.yaml",
                            "--epochs",
                            "4",
                        ]
                    commands.append(command)
    manifest = {
        "scope": "SegFormer dense E1/E3 and every Q1 baseline; downstream E4-E7 are separate",
        "status": "plan_only" if a.plan_only else "running",
        "dataset": str(a.dataset),
        "dataset_manifest_sha256": digest(a.dataset / "manifest_v3.json"),
        "criteria_sha256": digest("configs/matchability/criteria_v3.yaml"),
        "train_config_sha256": digest("configs/matchability/train.yaml"),
        "primary_backbone": "segformer_b0",
        "epochs": 4,
        "seeds": [42, 43, 44],
        "folds": [0, 1, 2],
        "runs": runs,
        "commands": commands,
        "interrupted_prior_run": str(a.reuse_suite / "segformer_b0_fold0_s44_v3"),
        "source_sha256": {
            str(f): digest(f)
            for f in sorted(Path("src/underwater_vision/matchability").glob("*.py"))
        },
    }
    write_json(root / "fold_manifest_v3.json", manifest)
    if a.plan_only:
        return
    environment = dict(
        os.environ, MLFLOW_DISABLE_AGENT_HINT="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"
    )
    for i, command in enumerate(commands):
        write_json(
            root / "execution_status_v3.json",
            {
                "status": "running",
                "command_index": i + 1,
                "commands": len(commands),
                "command": command,
            },
        )
        print(f"SegFormer folds {i + 1}/{len(commands)}: {' '.join(command)}", flush=True)
        completed = subprocess.run([sys.executable, *command], env=environment, check=False)
        if completed.returncode:
            write_json(
                root / "execution_status_v3.json",
                {"status": "failed", "command": command, "exit_code": completed.returncode},
            )
            raise SystemExit(completed.returncode)
    for run in runs:
        verified_run(Path(run["path"]), run["kind"], run["protocol"], run["seed"], a.dataset)
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_segformer_folds.py",
            "--suite",
            str(root),
            "--output",
            str(root / "comparison_v3"),
        ],
        check=True,
        env=environment,
    )
    write_json(
        root / "execution_status_v3.json",
        {
            "status": "completed",
            "dense_runs": 9,
            "baseline_runs": 9,
            "reused": sum(r["reused"] for r in runs),
        },
    )


if __name__ == "__main__":
    main()
