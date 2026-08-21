"""Reevaluate exported camera poses against published Mermaid reference estimates."""

import argparse
import json
from pathlib import Path

import numpy as np

from underwater_vision.evaluation.metrics import trajectory_metrics
from underwater_vision.utils.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    args = p.parse_args()

    def load(name):
        return {
            k: np.array(v)
            for k, v in json.loads((args.run / name).read_text(encoding="utf-8")).items()
        }

    metrics = trajectory_metrics(load("camera_poses.json"), load("reference_poses.json"))
    write_json(args.run / "trajectory_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
