"""Validation-only Platt ablation on the nine frozen SegFormer fold/seed runs."""

import argparse
import gc
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from underwater_vision.evaluation.matchability import metrics
from underwater_vision.matchability.calibration import calibrated_probabilities, fit_platt
from underwater_vision.matchability.dataset import MatchabilityDataset
from underwater_vision.matchability.model import load_model
from underwater_vision.matchability.protocol import digest, read_json
from underwater_vision.matchability.train import evaluate_logits
from underwater_vision.utils.io import atomic_npz, write_json


def cluster_statistics(y, p):
    known = y >= 0
    y, p = y[known], p[known]
    bins = np.zeros((10, 3))
    for i, left in enumerate(np.linspace(0, 0.9, 10)):
        keep = (p >= left) & (p < left + 0.1 if left < 0.9 else p <= 1)
        bins[i] = [keep.sum(), p[keep].sum(), y[keep].sum()]
    return len(y), float(np.square(p - y).sum()), bins


def bootstrap_differences(rows, weights):
    result = {}
    for method in ["temperature", "platt"]:
        values = [r[method] for r in rows]
        n = weights @ np.array([v[0] for v in values])
        brier = weights @ np.array([v[1] for v in values]) / n
        bins = np.einsum("bi,ijk->bjk", weights, np.array([v[2] for v in values]))
        ece = np.abs(bins[:, :, 1] - bins[:, :, 2]).sum(axis=1) / n
        result[method] = {"ece": ece, "brier": brier}
    return {key: result["platt"][key] - result["temperature"][key] for key in ["ece", "brier"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("outputs/segformer_folds_v3"))
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration_improvement_v1"))
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--reuse-validation", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = read_json(args.suite / "fold_manifest_v3.json")
    dataset = Path(manifest["dataset"])
    assert digest(dataset / "manifest_v3.json") == manifest["dataset_manifest_sha256"]
    data_manifest = read_json(dataset / "manifest_v3.json")
    runs = [r for r in manifest["runs"] if r["kind"] == "segformer_b0"]
    assert {(r["fold"], r["seed"]) for r in runs} == {
        (fold, seed) for fold in range(3) for seed in [42, 43, 44]
    }
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {
        "status": "running",
        "protocol": "Fixed positive-slope Platt scaling, unweighted NLL, validation only; compare against saved temperature scaling on identical held-out logits. No network retraining or test-based model selection.",
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "source_sha256": {
            str(p): digest(p)
            for p in [Path(__file__), Path("src/underwater_vision/matchability/calibration.py")]
        },
        "device": device,
        "validation_limit": args.validation_limit,
        "validation_selection": "Evenly spaced indices of the locked validation list; fixed compute budget, not test-selected. Zero uses all images. Saved temperature baseline used full validation.",
        "runs": [],
    }
    write_json(args.output / "protocol_v1.json", report)
    bootstrap = []
    for entry in runs:
        root = Path(entry["path"])
        run_out = args.output / f"fold{entry['fold']}_s{entry['seed']}"
        run_out.mkdir()
        checkpoint = root / "model_v3.pt"
        checkpoint_hash = digest(checkpoint)
        old_report = read_json(root / "evaluation_v3.json")
        assert checkpoint_hash == old_report["checkpoint_hash"]
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        split = next(p for p in data_manifest["protocols"] if p["name"] == entry["protocol"])
        full_validation_names = saved["validation_images"]
        assert set(full_validation_names) == set(split["groups"]["validation"])
        validation_names = full_validation_names
        if args.validation_limit:
            indices = np.linspace(
                0,
                len(validation_names) - 1,
                min(args.validation_limit, len(validation_names)),
                dtype=int,
            )
            validation_names = [validation_names[i] for i in indices]
        assert not set(validation_names) & set(saved["test_images"])
        reuse = (
            args.reuse_validation / run_out.name / "validation_logits_v1.npz"
            if args.reuse_validation is not None
            else None
        )
        print(
            f"{entry['protocol']} seed {entry['seed']}: calibrating on {len(validation_names)} validation images",
            flush=True,
        )
        if reuse is not None and reuse.exists():
            source_calibration = read_json(reuse.parent / "calibration_v1.json")
            assert source_calibration["checkpoint_hash"] == checkpoint_hash
            assert source_calibration["validation_images"] == full_validation_names
            with np.load(reuse) as data:
                cached_z, cached_y = data["logits"], data["labels"]
            rows, start = [], 0
            for name in full_validation_names:
                with np.load(
                    dataset / "labels" / entry["protocol"] / saved["graph"] / (name + ".npz")
                ) as labels:
                    original = labels[saved["variant"]]
                end = start + len(original)
                assert np.array_equal(original, cached_y[start:end])
                if name in validation_names:
                    rows.append({"logits": cached_z[start:end], "labels": cached_y[start:end]})
                start = end
            assert start == len(cached_z)
        else:
            model, _ = load_model(checkpoint, device)
            validation = MatchabilityDataset(
                dataset, validation_names, entry["protocol"], saved["graph"], cache_rgb=False
            )
            rows = evaluate_logits(model, validation, device)
            del model, validation
        z = np.concatenate([r["logits"] for r in rows])
        y = np.concatenate([r["labels"] for r in rows])
        calibration = fit_platt(z, y)
        calibration.update(
            checkpoint_hash=checkpoint_hash,
            protocol=entry["protocol"],
            seed=entry["seed"],
            validation_images=validation_names,
        )
        write_json(run_out / "calibration_v1.json", calibration)
        atomic_npz(run_out / "validation_logits_v1.npz", logits=z, labels=y)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        # The calibrator is frozen and written before loading any held-out predictions.
        files = sorted((root / "predictions").glob("*.npz"))
        assert {p.stem for p in files} == set(saved["test_images"]) == set(split["groups"]["test"])
        test_rows = []
        for path in files:
            with np.load(path) as data:
                test_rows.append(
                    {
                        "labels": data["labels"],
                        "logits": data["logits"],
                        "temperature": data["probability"],
                    }
                )
        tz = np.concatenate([r["logits"] for r in test_rows])
        ty = np.concatenate([r["labels"] for r in test_rows])
        before = np.concatenate([r["temperature"] for r in test_rows])
        after = calibrated_probabilities(tz, calibration)
        result = {
            "fold": entry["fold"],
            "seed": entry["seed"],
            "checkpoint_hash": checkpoint_hash,
            "source_evaluation_sha256": digest(root / "evaluation_v3.json"),
            "calibration": calibration,
            "before": metrics(ty, before),
            "after": metrics(ty, after),
        }
        for key in ["average_precision", "ece", "brier"]:
            assert abs(result["before"][key] - old_report["test_after"][key]) < 1e-7
        assert (
            abs(result["after"]["average_precision"] - result["before"]["average_precision"]) < 1e-5
        )
        clustered = [
            {
                "temperature": cluster_statistics(r["labels"], r["temperature"]),
                "platt": cluster_statistics(
                    r["labels"], calibrated_probabilities(r["logits"], calibration)
                ),
            }
            for r in test_rows
        ]
        # Identical image draws across all three seeds within a fold.
        rng = np.random.default_rng(42 + entry["fold"])
        weights = rng.multinomial(
            len(clustered), np.full(len(clustered), 1 / len(clustered)), size=500
        )
        draws = bootstrap_differences(clustered, weights)
        result["paired_ci95_delta"] = {
            k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in draws.items()
        }
        bootstrap.append(draws)
        assert digest(checkpoint) == checkpoint_hash
        write_json(run_out / "comparison_v1.json", result)
        report["runs"].append(result)
        write_json(args.output / "comparison_v1.json", report)
        print(
            f"{entry['protocol']} seed {entry['seed']}: ECE {result['before']['ece']:.4f} -> {result['after']['ece']:.4f}, Brier {result['before']['brier']:.4f} -> {result['after']['brier']:.4f}",
            flush=True,
        )
    report["overall"] = {
        method: {
            key: float(np.mean([r[method][key] for r in report["runs"]]))
            for key in ["average_precision", "ece", "brier"]
        }
        for method in ["before", "after"]
    }
    report["overall"]["paired_ci95_delta"] = {
        k: np.quantile(np.mean([d[k] for d in bootstrap], axis=0), [0.025, 0.975]).tolist()
        for k in ["ece", "brier"]
    }
    report["overall"]["ece_below_005"] = {
        method: sum(r[method]["ece"] <= 0.05 for r in report["runs"])
        for method in ["before", "after"]
    }
    report["statistics"] = (
        "Equal-weight macro mean across nine runs. 500 paired image-cluster bootstrap draws, identical draws across seeds of each fold; fixed-fold conditional CIs. One acquisition with spatial slabs, not independent scenes. Calibration labels measure reference epipolar compatibility, not dense pixel truth."
    )
    report["status"] = "complete"
    write_json(args.output / "comparison_v1.json", report)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(9)
    ax.bar(
        x - 0.18,
        [r["before"]["ece"] * 100 for r in report["runs"]],
        width=0.36,
        label="Temperature baseline",
    )
    ax.bar(
        x + 0.18,
        [r["after"]["ece"] * 100 for r in report["runs"]],
        width=0.36,
        label="Validation-only Platt",
    )
    ax.axhline(5, color="gray", linestyle="--", label="Frozen 5% ECE target")
    ax.set_xticks(x, [f"F{r['fold']} / S{r['seed']}" for r in report["runs"]], rotation=25)
    ax.set_ylabel("Held-out ECE (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "calibration_comparison_v1.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
