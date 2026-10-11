"""Plan or execute E1-E6 after human labeling; each fold trains independent models."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from underwater_vision.semantic.protocol import load_manifest, new_run, require_gate, stratified
from underwater_vision.utils.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/semantic_suite_v2"))
    p.add_argument(
        "--manifest", type=Path, default=Path("outputs/semantic_labels_v2/manifest.json")
    )
    p.add_argument(
        "--fold-manifests",
        nargs=3,
        type=Path,
        help="Three independently frozen manifests with fold-specific reviewed validation/test and pseudo train masks",
    )
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--segmentation-epochs", type=int, default=40)
    a = p.parse_args()
    if not a.plan_only:
        require_gate("outputs/semantic_audit_v2/gate_review.json")
    manifests = a.fold_manifests or [a.manifest]
    # Final 3-fold runs require actual labels for all rotations; never fake this
    # by splitting a single test list and reusing a student that saw those blocks.
    if not a.plan_only:
        for file in manifests:
            load_manifest(file, require_frozen=True)
    root = new_run(a.output)
    commands = []
    for fold, file in enumerate(manifests):
        manifest = load_manifest(file)
        groups = manifest["spatial_split"]["groups"]
        selected = {
            "train": stratified(groups["train"], min(64, len(groups["train"]))),
            "validation": stratified(groups["validation"], min(32, len(groups["validation"]))),
            "test": [e["image_id"] for e in manifest["entries"] if e["role"] == "test"][:16],
        }
        spatial = {"selected": selected, "split": manifest["spatial_split"]}
        spatial_path = root / f"fold{fold}_spatial_manifest_v2.json"
        write_json(spatial_path, spatial)
        test_list = root / f"fold{fold}_images_v2.txt"
        test_list.write_text("\n".join(selected["test"]), encoding="utf-8")
        for seed in a.seeds:
            prefix = f"fold{fold}_seed{seed}"

            def command(script, *args):
                commands.append([sys.executable, "scripts/" + script, *map(str, args)])

            segmentations = {}
            for kind in ["dino_probe", "dino_decoder", "segformer_b0"]:
                folder = root / (prefix + "_" + kind + "_v2")
                checkpoint = folder / "segmentation_v2.pt"
                command(
                    "train_segmentation.py",
                    "--manifest",
                    file,
                    "--model-config",
                    f"configs/semantic/model/{kind}.yaml",
                    "--seed",
                    seed,
                    "--epochs",
                    a.segmentation_epochs,
                    "--output",
                    folder,
                )
                command(
                    "evaluate_segmentation.py",
                    "--manifest",
                    file,
                    "--checkpoint",
                    checkpoint,
                    "--output",
                    root / (prefix + "_" + kind + "_evaluation_v2"),
                )
                segmentations[kind] = checkpoint
            student = segmentations["segformer_b0"]
            teacher = root / (prefix + "_teacher_v2")
            command(
                "export_semantic_cache.py",
                "--manifest",
                file,
                "--teacher",
                "--role",
                "test",
                "--output",
                teacher,
            )
            command(
                "evaluate_segmentation.py",
                "--manifest",
                file,
                "--teacher",
                teacher,
                "--output",
                root / (prefix + "_teacher_evaluation_v2"),
            )
            filters = {}
            for mode in ["off", "compact", "full"]:
                folder = root / (prefix + "_matches_" + mode + "_v2")
                command(
                    "train_match_filter.py",
                    "--version",
                    2,
                    "--semantic-features",
                    mode,
                    "--semantic-checkpoint",
                    student,
                    "--spatial-manifest",
                    spatial_path,
                    "--seed",
                    seed,
                    "--output",
                    folder,
                )
                filters[mode] = folder
            for mode in ["compact", "full"]:
                command(
                    "evaluate_semantic_matches.py",
                    "--run",
                    filters[mode],
                    "--mode",
                    mode,
                    "--baseline",
                    filters["off"] / "context_v2.pt",
                    "--segmentation-checkpoint",
                    student,
                    "--output",
                    root / (prefix + "_match_eval_" + mode + "_v2"),
                )
            command(
                "evaluate_semantic_matches.py",
                "--run",
                filters["compact"],
                "--mode",
                "compact",
                "--architecture",
                "mlp",
                "--baseline",
                filters["off"] / "context_v2.pt",
                "--segmentation-checkpoint",
                student,
                "--output",
                root / (prefix + "_match_eval_mlp_v2"),
            )
            shared = [
                # Paths are passed as argv, not interpolated into a shell.
                f"data.image_list={json.dumps(test_list.as_posix())}",
                f"output_root={json.dumps(root.as_posix())}",
                f"seed={seed}",
            ]
            baseline = root / (prefix + "_classical_v2")
            command(
                "run_experiment.py", "experiment=classical", f"run_name={baseline.name}", *shared
            )
            confidence = root / (prefix + "_confidence_v2")
            command(
                "train_point_confidence.py",
                "--semantic-features",
                "--semantic-checkpoint",
                student,
                "--conformal",
                "--seed",
                seed,
                "--matches",
                filters["compact"],
                "--baseline",
                baseline,
                "--output",
                confidence,
            )
            for variant in ["off", "compact", "student_mask", "teacher_mask"]:
                mode = "compact" if variant == "compact" else "off"
                run = root / (prefix + "_sfm_" + variant + "_v2")
                overrides = [
                    "experiment=semantic_matches",
                    f"run_name={run.name}",
                    *shared,
                    f"matching.filter_checkpoint={json.dumps((filters[mode]/'context_v2.pt').as_posix())}",
                    f"matching.semantic_features={mode}",
                    f"matching.semantic_checkpoint={json.dumps(student.as_posix())}",
                    f"matching.dynamic_mask={str(variant.endswith('_mask')).lower()}",
                ]
                if variant == "teacher_mask":
                    overrides += [
                        "matching.semantic_checkpoint=null",
                        f"matching.semantic_teacher_cache={json.dumps(teacher.as_posix())}",
                        "matching.dynamic_threshold=0.5",
                    ]
                if variant == "compact":
                    overrides += [
                        f"reconstruction.semantic_confidence_checkpoint={json.dumps((confidence/'error_model_v2.pt').as_posix())}"
                    ]
                command("run_experiment.py", *overrides)
                if variant == "compact":
                    command(
                        "evaluate_semantic_voting.py",
                        "--run",
                        run,
                        "--checkpoint",
                        student,
                        "--manifest",
                        file,
                        "--output",
                        root / (prefix + "_voting_evaluation_v2"),
                    )
            command(
                "evaluate_dynamic_masking.py",
                "--matches",
                filters["off"],
                "--baseline",
                filters["off"] / "context_v2.pt",
                "--checkpoint",
                student,
                "--teacher",
                teacher,
                "--manifest",
                file,
                "--output",
                root / (prefix + "_dynamic_evaluation_v2"),
            )
    write_json(
        root / "plan_v2.json",
        {
            "criteria": yaml.safe_load(Path("configs/semantic/criteria_v2.yaml").read_text()),
            "commands": commands,
            "seeds": a.seeds,
            "spatial_folds": len(manifests),
            "final_protocol": len(manifests) == 3 and len(a.seeds) == 3,
            "status": "plan_only" if a.plan_only else "running",
            "requires": "human Gate 0, frozen criteria, frozen hand labels and per-fold pseudo masks",
        },
    )
    if not a.plan_only:
        for command in commands:
            subprocess.run(command, check=True)
        command = [
            sys.executable,
            "scripts/aggregate_semantic_results.py",
            "--suite",
            str(root),
            "--output",
            str(root / "comparison_v2"),
        ]
        subprocess.run(command, check=True)
    print(
        f"Prepared {len(commands)} commands; folds={len(manifests)}, seeds={a.seeds}. No hand labels are synthesized."
    )


if __name__ == "__main__":
    main()
