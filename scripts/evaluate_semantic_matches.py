"""Common-pool Q2/Q3 evaluation with paired bootstrap over complete test pairs."""

import argparse
from pathlib import Path

import numpy as np

from underwater_vision.evaluation.segmentation import paired_bootstrap
from underwater_vision.evaluation.uncertainty import binary_metrics
from underwater_vision.matching.learned_filter import MatchPredictor
from underwater_vision.semantic.protocol import digest_file, new_run
from underwater_vision.utils.io import atomic_npz, write_json


def evaluate(
    run, baseline, output, segmentation_checkpoint, mode="compact", architecture="context"
):
    root = new_run(output)
    learned = MatchPredictor(
        run / (architecture + "_v2.pt"),
        mode,
        digest_file(segmentation_checkpoint) if mode != "off" else None,
    )
    original = MatchPredictor(baseline)
    if set(learned.metadata["training_images"]) != set(original.metadata["training_images"]) or set(
        learned.metadata["validation_images"]
    ) != set(original.metadata["validation_images"]):
        raise ValueError("Paired model comparison requires identical spatial fitting blocks")
    records = []
    for file in sorted((run / "pairs").glob("test_*.npz")):
        with np.load(file) as z:
            r = {k: z[k] for k in z.files}
        r["new_probability"], r["baseline_probability"] = learned(r["x"]), original(r["x"][:, :12])
        records.append(r)
    if not records:
        raise ValueError("No held-out pair records")

    def metrics(rows, key, threshold):
        return binary_metrics(
            np.concatenate([r["labels"] for r in rows]),
            np.concatenate([r[key] for r in rows]),
            threshold,
        )

    baseline_metrics = metrics(records, "baseline_probability", original.metadata["threshold"])
    new_metrics = metrics(records, "new_probability", learned.metadata["threshold"])

    def difference(rows):
        return (
            metrics(rows, "new_probability", learned.metadata["threshold"])["average_precision"]
            - metrics(rows, "baseline_probability", original.metadata["threshold"])[
                "average_precision"
            ]
        )

    ci = paired_bootstrap(records, difference)
    per_class = {}
    for c in range(6):
        selected = [
            {
                "labels": r["labels"][r["class_region"] == c],
                "new_probability": r["new_probability"][r["class_region"] == c],
                "baseline_probability": r["baseline_probability"][r["class_region"] == c],
            }
            for r in records
            if "class_region" in r
        ]
        if selected:
            per_class[str(c)] = {
                "new": metrics(selected, "new_probability", learned.metadata["threshold"]),
                "baseline": metrics(
                    selected, "baseline_probability", original.metadata["threshold"]
                ),
            }
    atomic_npz(
        root / "predictions_v2.npz",
        labels=np.concatenate([r["labels"] for r in records]),
        new_probability=np.concatenate([r["new_probability"] for r in records]),
        baseline_probability=np.concatenate([r["baseline_probability"] for r in records]),
    )
    result = {
        "architecture": architecture,
        "baseline": baseline_metrics,
        "learned": new_metrics,
        "ap_difference": ci,
        "per_class": per_class,
        "class_region_source": "predicted segmentation probabilities, not ground-truth semantic classes",
        "test_pairs": len(records),
        "segmentation_hash": digest_file(segmentation_checkpoint),
        "q3_status": (
            "single_region_positive_ci"
            if ci["ci95"] and ci["ci95"][0] > 0
            else "null_or_insufficient"
        ),
        "final_q3_requires": "positive paired CI in at least two of three independently retrained spatial folds",
    }
    write_json(root / "evaluation_v2.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--baseline", type=Path, default=Path("outputs/match_filter_v1/context_v1.pt"))
    p.add_argument("--segmentation-checkpoint", type=Path, required=True)
    p.add_argument("--mode", choices=["off", "compact", "full"], default="compact")
    p.add_argument("--architecture", choices=["context", "mlp"], default="context")
    p.add_argument("--output", type=Path, default=Path("outputs/semantic_match_evaluation_v2"))
    a = p.parse_args()
    evaluate(a.run, a.baseline, a.output, a.segmentation_checkpoint, a.mode, a.architecture)


if __name__ == "__main__":
    main()
