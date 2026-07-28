"""Train a standalone semantic model using the frozen hand-label manifest."""

import argparse
from pathlib import Path

import yaml

from underwater_vision.semantic.train import train_segmentation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/semantic/train.yaml"))
    p.add_argument("--output", type=Path, default=Path("outputs/segformer_seed42_v2"))
    p.add_argument("--model-config", type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--epochs", type=int)
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for name in ["model_config", "seed", "manifest", "epochs"]:
        value = getattr(args, name)
        if value is not None:
            config[name] = str(value) if isinstance(value, Path) else value
    train_segmentation(config, args.output)


if __name__ == "__main__":
    main()
