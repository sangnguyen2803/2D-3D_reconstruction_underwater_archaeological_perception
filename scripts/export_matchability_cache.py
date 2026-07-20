"""Cache independent full-resolution matchability maps for an explicit image list."""

import argparse
from pathlib import Path

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.matchability.infer import MatchabilityPredictor
from underwater_vision.matchability.protocol import new_run
from underwater_vision.utils.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration", type=Path, help="Optional validation-fitted Platt sidecar")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", default="data/raw/mermaid")
    a = p.parse_args()
    root = new_run(a.output)
    predictor = MatchabilityPredictor(a.checkpoint, calibration=a.calibration)
    lookup = {f.name: f for f in MermaidDataset(a.data_root).frames}
    names = a.images.read_text().splitlines()
    for name in names:
        predictor.cached(lookup[name].path)
    write_json(
        root / "cache_manifest_v3.json",
        {
            "images": names,
            "checkpoint_hash": predictor.checkpoint_hash,
            "temperature": predictor.temperature,
            "calibration_hash": predictor.calibration_hash,
            "max_size": 960,
        },
    )


if __name__ == "__main__":
    main()
