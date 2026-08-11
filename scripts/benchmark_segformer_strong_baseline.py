"""Measure SegFormer's incremental geometry benefit over low-contrast RootSIFT."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
import yaml
from scipy.special import expit

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.evaluation.metrics import trajectory_metrics
from underwater_vision.matchability.infer import MatchabilityPredictor
from underwater_vision.matchability.loss import sample_logits
from underwater_vision.matchability.protocol import digest, read_json
from underwater_vision.matching.budget import budget_indices, endpoint_ranking
from underwater_vision.matching.learned_filter import candidates
from underwater_vision.reconstruction.colmap.database import (
    create_database,
    insert_matches,
    make_camera,
)
from underwater_vision.reconstruction.colmap.exporter import export_reconstruction
from underwater_vision.reconstruction.colmap.mapper import reconstruct
from underwater_vision.utils.io import atomic_npz, read_rgb, write_json

BLOCKS = [("block273", "fold2"), ("block599", "fold1"), ("block1116", "fold2")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/segformer_strong_baseline_v1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    pycolmap.logging.minloglevel = 2
    base = Path("outputs/reconstruction_improvement_v1")
    detector = Path("outputs/detector_coverage_v1")
    frames = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
    manifest = read_json("outputs/segformer_folds_v3/fold_manifest_v3.json")
    checkpoints = {
        (r["protocol"], r["seed"]): Path(r["path"])
        for r in manifest["runs"]
        if r["kind"] == "segformer_b0"
    }
    settings = read_json(base / "comparison_v1.json")["settings"]
    calibration = yaml.safe_load(Path("configs/data/mermaid.yaml").read_text())["calibration"]
    report = {
        "status": "running",
        "protocol": "Compare SegFormer endpoint match ranking with the already improved contrast0.01 RootSIFT baseline, on identical cached keypoints/descriptors/RGB960 inputs, 171 window6 pairs and fixed intrinsics/COLMAP settings. Ratio90, random90 and SegFormer90 retain exactly the same match count in each pair: ceil(90% mutual ratio0.8 survivors), minimum128 or all available. Frozen corresponding held-out fold/seed SegFormer predicts raw logits at new detector locations. No reference pose/label selects matches. No new network training or tuning in this experiment. These are already examined development regions, not blind scenes.",
        "primary_method": "segformer90",
        "primary_success": "No registration loss and lower common-camera ATE than full low-contrast baseline AND matched-budget ratio/random controls; report every region and seed, including failures.",
        "prediction_device": "cpu_float32",
        "source_sha256": digest(__file__),
        "detector_baseline_sha256": digest(detector / "comparison_v1.json"),
        "methods": ["ratio90", "random90", "segformer90"],
        "runs": [],
    }
    (args.output / "executed_source_v1.py").write_bytes(Path(__file__).read_bytes())
    write_json(args.output / "protocol_v1.json", report)
    for block, fold in BLOCKS:
        root = args.output / block
        root.mkdir()
        names = read_json(base / block / "input_manifest_v1.json")["names"]
        features = {}
        for name in names:
            with np.load(detector / block / (name + ".npz")) as z:
                xy, local = z["xy"].copy(), z["local"].copy()
            features[name] = {
                "xy": xy,
                "local": local,
                "semantic": np.zeros((len(xy), 1)),
                "quality": np.ones(len(xy)),
                "shape": np.array([720, 960]),
            }
        prepared = []
        for i in range(32):
            for j in range(i + 1, min(32, i + 7)):
                a, b = names[i], names[j]
                ids, x = candidates(features[a], features[b], 960, 720)
                prepared.append((a, b, ids, x))
        for seed in [42, 43, 44]:
            model_root = checkpoints[(fold, seed)]
            metadata = read_json(model_root / "resolved_config_v3.json")
            assert not set(names) & set(metadata["training_images"] + metadata["validation_images"])
            predictor = MatchabilityPredictor(model_root / "model_v3.pt", device="cpu")
            endpoint = {}
            begin = time.perf_counter()
            for index, name in enumerate(names):
                values = predictor.logits(read_rgb(base / block / "images" / name))
                logits = sample_logits(
                    torch.from_numpy(values)[None, None], torch.from_numpy(features[name]["xy"])
                )[0].numpy()
                endpoint[name] = expit(logits)
                atomic_npz(
                    root / f"prediction_s{seed}_{name}.npz",
                    logits=logits,
                    xy=features[name]["xy"],
                    checkpoint_sha256=np.asarray(predictor.checkpoint_hash),
                )
                if (index + 1) % 8 == 0:
                    print(f"{block} seed{seed}: predicted {index+1}/32 images", flush=True)
            prediction_seconds = time.perf_counter() - begin
            del predictor
            expected_counts = None
            for method in report["methods"]:
                run = root / f"{method}_s{seed}"
                run.mkdir()
                results = []
                rng = np.random.default_rng(seed)
                for a, b, ids, x in prepared:
                    if method == "ratio90":
                        score = 1 - x[:, 1]
                    elif method == "random90":
                        score = rng.random(len(ids))
                    else:
                        score = endpoint_ranking(ids, endpoint[a], endpoint[b])
                    keep = budget_indices(x, score)
                    results.append(
                        {"a": a, "b": b, "pairs": ids[keep], "inliers": np.ones(len(keep), bool)}
                    )
                counts = [len(r["pairs"]) for r in results]
                if expected_counts is None:
                    expected_counts = counts
                assert counts == expected_counts
                image_ids = create_database(
                    run / "database.db",
                    [frames[n] for n in names],
                    features,
                    make_camera(960, 720, calibration),
                )
                insert_matches(run / "database.db", image_ids, results)
                pairs_file = run / "pairs.txt"
                pairs_file.write_text(
                    "\n".join(r["a"] + " " + r["b"] for r in results if len(r["pairs"]) >= 15)
                )
                result = {
                    "block": block,
                    "method": method,
                    "seed": seed,
                    "path": str(run),
                    "selected_matches": sum(counts),
                    "per_pair_counts": counts,
                    "segformer_checkpoint_sha256": (
                        digest(model_root / "model_v3.pt") if method == "segformer90" else None
                    ),
                    "prediction_seconds": prediction_seconds if method == "segformer90" else 0,
                }
                print(f"{block} seed{seed}: mapping {method}", flush=True)
                begin = time.perf_counter()
                try:
                    model, _ = reconstruct(
                        run / "database.db",
                        base / block / "images",
                        run / "sparse",
                        pairs_file,
                        settings,
                        seed,
                    )
                    poses, statistics = export_reconstruction(model, run)
                    result.update(
                        status="complete",
                        **statistics,
                        trajectory=trajectory_metrics(
                            poses, {n: frames[n].reference_c2w for n in names}
                        ),
                    )
                except RuntimeError as error:
                    result.update(
                        status="failed", error=str(error), registered_images=0, sparse_points=0
                    )
                result["mapping_seconds"] = time.perf_counter() - begin
                write_json(run / "metrics_v1.json", result)
                report["runs"].append(result)
                write_json(args.output / "comparison_v1.json", report)
                print(
                    {
                        k: result.get(k)
                        for k in ["block", "method", "seed", "registered_images", "trajectory"]
                    },
                    flush=True,
                )
    report["status"] = "complete"
    write_json(args.output / "comparison_v1.json", report)


if __name__ == "__main__":
    main()
