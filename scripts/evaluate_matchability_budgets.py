"""Re-evaluate immutable E5 pools with independent per-image random selections."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.matchability.build import load_features
from underwater_vision.matchability.experiments import evaluate_budgets
from underwater_vision.matchability.protocol import new_run, read_json
from underwater_vision.utils.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = new_run(a.output)
    torch.set_num_threads(2)
    lookup = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
    calibration = yaml.safe_load(Path("configs/data/mermaid.yaml").read_text())["calibration"]
    for folder in sorted(a.suite.glob("filters_fold*_v3")):
        if not (folder / "evaluation_v3.json").exists():
            continue
        manifest = read_json(folder / "spatial_manifest_v3.json")
        features = {}
        for name in manifest["selected"]["test"]:
            feature = load_features(Path(manifest["dataset"]), name)
            with np.load(folder / "maps" / (name + ".npz")) as z:
                if not np.array_equal(feature["xy"], z["xy"]):
                    raise ValueError("Budget pool coordinates changed")
                feature["matchability"] = z["matchability"]
            features[name] = feature
        records = []
        for file in sorted((folder / "pairs").glob("test_*.npz")):
            with np.load(file) as z:
                records.append({k: z[k] for k in z.files})
        target = new_run(root / folder.name)
        seed = read_json(folder / "evaluation_v3.json")["seed"]
        evaluate_budgets(records, features, lookup, calibration, seed, target)
        write_json(
            target / "provenance_v3.json",
            {
                "source": str(folder),
                "change": "Independent per-image random seed and explicit runtime warm-up; same RootSIFT pool, maps, labels and exact budgets",
            },
        )
    write_json(root / "status_v3.json", {"status": "complete"})


if __name__ == "__main__":
    main()
