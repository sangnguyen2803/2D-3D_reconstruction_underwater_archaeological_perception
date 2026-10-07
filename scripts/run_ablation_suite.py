"""Sequential controlled suite. Explicit paths ensure reproducibility as data grows."""

import argparse
import subprocess
import sys


def invoke(*arguments):
    subprocess.run([sys.executable, *arguments], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-list", default="configs/evaluation_images.txt")
    p.add_argument("--training-list", default="configs/training_images.txt")
    p.add_argument("--prefix", default="suite")
    p.add_argument("--tracking", action="store_true")
    p.add_argument("--include-small", action="store_true")
    args = p.parse_args()
    shared = [f"data.image_list={args.eval_list}", f"tracking={str(args.tracking).lower()}"]
    names = []
    for experiment in ["classical", "neural", "visibility_aware", "consistency"]:
        name = f"{args.prefix}_{experiment}"
        invoke("scripts/run_experiment.py", f"experiment={experiment}", f"run_name={name}", *shared)
        names.append("outputs/" + name)
    train_run = f"{args.prefix}_training"
    invoke(
        "scripts/run_experiment.py",
        "experiment=neural",
        f"data.image_list={args.training_list}",
        "stage=matches",
        f"run_name={train_run}",
    )
    checkpoint = f"outputs/models/{args.prefix}_reliability.pt"
    invoke("scripts/train_reliability.py", "--run", "outputs/" + train_run, "--output", checkpoint)
    invoke(
        "scripts/run_experiment.py",
        "experiment=learned",
        *shared,
        f"run_name={args.prefix}_learned",
        f"matching.learned_checkpoint={checkpoint}",
    )
    names.append(f"outputs/{args.prefix}_learned")
    if args.include_small:
        invoke(
            "scripts/run_experiment.py",
            "experiment=visibility_aware",
            *shared,
            "model.backbone=dinov2_vits14",
            f"run_name={args.prefix}_small",
        )
        names.append(f"outputs/{args.prefix}_small")
    invoke(
        "scripts/compare_experiments.py", *names, "--output", f"outputs/{args.prefix}_comparison"
    )


if __name__ == "__main__":
    main()
