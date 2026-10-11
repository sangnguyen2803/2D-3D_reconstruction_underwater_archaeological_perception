"""Identical historical 16-view inputs, three seeds and L1/L2/soft ablations."""

import argparse
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from underwater_vision.matchability.protocol import digest, new_run, read_json
from underwater_vision.pipeline import run
from underwater_vision.utils.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=4)
    a = p.parse_args()
    from underwater_vision.matchability.report import suite_dataset

    dataset = suite_dataset(a.suite)
    root = new_run(a.output)
    l2 = Path("outputs/matchability_l2_labels_v3")
    # A fresh-suite invocation builds the stronger labels below its own root.
    if not (l2 / "audit_v3.json").exists():
        l2 = root / "strong_labels_v3"
        subprocess.run(
            [
                sys.executable,
                "scripts/build_matchability_l2.py",
                "--dataset",
                str(dataset),
                "--output",
                str(l2),
            ],
            check=True,
        )
    registered = read_json(l2 / "manifest_v3.json")
    if registered["parent_manifest_sha256"] != digest(dataset / "manifest_v3.json"):
        raise ValueError("L2 labels belong to a different parent dataset")
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        base = compose(
            config_name="config",
            overrides=[
                "experiment=learned_matches",
                "data.image_list=configs/evaluation_images.txt",
            ],
        )
        classical = compose(
            config_name="config",
            overrides=["experiment=classical", "data.image_list=configs/evaluation_images.txt"],
        )
    base.output_root = str(root)
    # These images are deliberately development data, already examined in v1.
    report = {
        "status": "complete",
        "input_images": Path(base.data.image_list).read_text().splitlines(),
        "interpretation": "Historical development block; not new spatial-test evidence for Q1-Q5",
        "label_ablation": [],
        "reconstructions": [],
        "parent_manifest_sha256": digest(dataset / "manifest_v3.json"),
    }
    for seed in [42, 43, 44]:
        dense = root / f"segformer_v1partition_s{seed}_v3"
        filters = root / f"filters_v1partition_s{seed}_v3"
        subprocess.run(
            [
                sys.executable,
                "scripts/train_matchability.py",
                "--dataset",
                str(dataset),
                "--protocol",
                "v1_partition",
                "--seed",
                str(seed),
                "--epochs",
                str(a.epochs),
                "--output",
                str(dense),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/train_match_filter.py",
                "--matchability-features",
                "on",
                "--matchability-checkpoint",
                str(dense / "model_v3.pt"),
                "--matchability-dataset",
                str(dataset),
                "--protocol",
                "v1_partition",
                "--seed",
                str(seed),
                "--test-views",
                "16",
                "--output",
                str(filters),
            ],
            check=True,
        )
        for variant in ["l2", "soft"]:
            folder = root / f"segformer_{variant}_s{seed}_v3"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/train_matchability.py",
                    "--dataset",
                    str(l2 if variant == "l2" else dataset),
                    "--protocol",
                    "v1_partition",
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(a.epochs),
                    "--output",
                    str(folder),
                ],
                check=True,
            )
            metric = read_json(folder / "evaluation_v3.json")
            report["label_ablation"].append(
                {
                    "seed": seed,
                    "variant": variant,
                    "common_labels": "L1 test; training variants only",
                    "path": str(folder),
                    "test": metric["test_after"],
                }
            )
        report["label_ablation"].append(
            {
                "seed": seed,
                "variant": "l1",
                "common_labels": "L1 test",
                "path": str(dense),
                "test": read_json(dense / "evaluation_v3.json")["test_after"],
            }
        )
        for method in ["classical", "v1_context", "matchability_filter", "matchability_top512"]:
            cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
            cfg.seed = seed
            cfg.run_name = f"{method}_s{seed}_v3"
            if method == "classical":
                cfg = OmegaConf.create(OmegaConf.to_container(classical, resolve=True))
                cfg.output_root, cfg.run_name, cfg.seed = str(root), f"{method}_s{seed}_v3", seed
            elif method == "v1_context":
                cfg.matching.filter_checkpoint = str(filters / "context_12_v3.pt")
            elif method.startswith("matchability"):
                cfg.matching.filter_checkpoint = str(filters / "context_16_v3.pt")
                cfg.matching.matchability_features = "on"
                cfg.matching.matchability_checkpoint = str(dense / "model_v3.pt")
                if method.endswith("top512"):
                    cfg.matching.keypoint_budget = 512
                    cfg.matching.keypoint_selection = "matchability"
            try:
                output = run(cfg)
                metric = read_json(output / "metrics.json")
                report["reconstructions"].append(
                    {
                        "method": method,
                        "seed": seed,
                        "status": "complete",
                        "path": str(output),
                        **metric,
                    }
                )
            except Exception as exc:
                # Preserve invalid exact budgets / failed reconstruction, never substitute images.
                report["reconstructions"].append(
                    {
                        "method": method,
                        "seed": seed,
                        "status": "failed",
                        "error": str(exc),
                        "path": str(root / cfg.run_name),
                    }
                )
    write_json(root / "evaluation_v3.json", report)


if __name__ == "__main__":
    main()
