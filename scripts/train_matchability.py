"""Train a detachable v3 dense model without semantic annotations."""

import argparse
from pathlib import Path

from underwater_vision.matchability.train import fit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("outputs/matchability_labels_v3"))
    p.add_argument("--protocol", default="fold0")
    p.add_argument("--graph", choices=["sequential", "hybrid"], default="hybrid")
    p.add_argument("--variant", choices=["l1", "l2", "soft"], default="l1")
    p.add_argument(
        "--model", type=Path, default=Path("configs/matchability/model/segformer_b0.yaml")
    )
    p.add_argument("--config", type=Path, default=Path("configs/matchability/train.yaml"))
    p.add_argument("--criteria", type=Path, default=Path("configs/matchability/criteria_v3.yaml"))
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    fit(p.parse_args())


if __name__ == "__main__":
    main()
