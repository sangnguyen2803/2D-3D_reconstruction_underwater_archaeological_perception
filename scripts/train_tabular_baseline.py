"""Run every Q1 statistical control, including frozen DINO+MLP."""

import argparse
from pathlib import Path

from underwater_vision.matchability.tabular_baseline import train_baselines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("outputs/matchability_labels_v3"))
    p.add_argument("--protocol", default="fold0")
    p.add_argument("--graph", default="hybrid", choices=["sequential", "hybrid"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    train_baselines(a.dataset, a.protocol, a.output, a.seed, a.graph)


if __name__ == "__main__":
    main()
