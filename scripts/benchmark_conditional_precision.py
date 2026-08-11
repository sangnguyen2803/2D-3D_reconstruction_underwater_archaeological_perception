"""Evaluate the validation-selected precision network on frozen classical pools."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
import yaml

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.evaluation.metrics import trajectory_metrics
from underwater_vision.matchability.protocol import digest, read_json
from underwater_vision.matching.budget import budget_indices
from underwater_vision.matching.geometry_precision import PrecisionPredictor, precision_features
from underwater_vision.reconstruction.colmap.database import (
    create_database,
    insert_matches,
    make_camera,
)
from underwater_vision.reconstruction.colmap.exporter import export_reconstruction
from underwater_vision.reconstruction.colmap.mapper import reconstruct
from underwater_vision.utils.io import atomic_npz, read_rgb, write_json
from underwater_vision.visibility.deterministic import estimate_visibility, sample_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/conditional_precision_geometry_v1")
    )
    parser.add_argument("--training", type=Path, default=Path("outputs/conditional_precision_v1"))
    parser.add_argument("--candidate-cache", type=Path)
    args = parser.parse_args()
    training = read_json(args.training / "training_v1.json")
    assert training["status"] == "complete"
    winner = training["selected_architecture"]
    checkpoint = args.training / f"{winner}_v1.pt"
    predictor = PrecisionPredictor(checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "executed_source_v1.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    pycolmap.logging.minloglevel = 2
    base = Path("outputs/reconstruction_improvement_v1")
    detector = Path("outputs/detector_coverage_v1")
    controls = read_json("outputs/segformer_strong_baseline_v1/comparison_v1.json")
    settings = read_json(base / "comparison_v1.json")["settings"]
    frames = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
    calibration = yaml.safe_load(Path("configs/data/mermaid.yaml").read_text())["calibration"]
    report = {
        "status": "running",
        "primary_method": "precision90",
        "methods": ["precision90"],
        "source_sha256": digest(__file__),
        "checkpoint_sha256": digest(checkpoint),
        "architecture": winner,
        "target": predictor.metadata["target"],
        "training_sha256": digest(args.training / "training_v1.json"),
        "detector_baseline_sha256": digest(detector / "comparison_v1.json"),
        "protocol": "One architecture/checkpoint selected solely using spatial validation before test scoring. Same contrast0.01 keypoints and immutable RAW mutual ratio0.8 match pools read from classical databases, not RANSAC inliers. Exact retained count per pair matches existing ratio90/random90 controls, minimum128 and90% retention. No test labels or poses select matches. All three locked32-image blocks and mapper seeds42/43/44; fixed intrinsics/COLMAP. Fresh conditional model learned precision on the same detector distribution, with joint coordinate relabeling. Already examined development regions, not a blind new scene.",
        "runs": [],
    }
    write_json(args.output / "protocol_v1.json", report)
    for block in ["block273", "block599", "block1116"]:
        root = args.output / block
        root.mkdir()
        names = read_json(base / block / "input_manifest_v1.json")["names"]
        assert not set(names) & set(
            predictor.metadata["training_images"] + predictor.metadata["validation_images"]
        )
        features = {}
        for name in names:
            with np.load(detector / block / (name + ".npz")) as z:
                xy, local = z["xy"].copy(), z["local"].copy()
            rgb = read_rgb(base / block / "images" / name)
            reliability, _ = estimate_visibility(rgb)
            features[name] = {
                "xy": xy,
                "local": local,
                "quality": sample_map(reliability, xy),
                "shape": np.array(rgb.shape[:2]),
            }
        results = []
        counts = []
        with pycolmap.Database.open(
            str(detector / block / "low_contrast_s42/database.db")
        ) as database:
            ids = {im.name: im.image_id for im in database.read_all_images()}
            for i in range(32):
                for j in range(i + 1, min(32, i + 7)):
                    a, b = names[i], names[j]
                    pairs = database.read_matches(ids[a], ids[b])
                    if args.candidate_cache:
                        with np.load(
                            args.candidate_cache / block / f"ranked_{i:02}_{j:02}.npz"
                        ) as cached:
                            assert np.array_equal(pairs, cached["pairs"])
                            x = cached["x"].copy()
                        assert np.allclose(
                            x[:, 4:6] * [960, 720], features[a]["xy"][pairs[:, 0]], atol=1e-4
                        )
                        assert np.allclose(
                            x[:, 6:8] * [960, 720], features[b]["xy"][pairs[:, 1]], atol=1e-4
                        )
                    else:
                        x = precision_features(features[a], features[b], pairs)
                    score = predictor(x)
                    stub = np.zeros((len(pairs), 3), np.float32)
                    stub[:, 2] = 1
                    keep = budget_indices(stub, score)
                    results.append(
                        {"a": a, "b": b, "pairs": pairs[keep], "inliers": np.ones(len(keep), bool)}
                    )
                    counts.append(len(keep))
                    atomic_npz(
                        root / f"ranked_{i:02}_{j:02}.npz",
                        pairs=pairs,
                        x=x,
                        score=score,
                        retained=keep,
                    )
                if (i + 1) % 8 == 0:
                    print(f"{block}: ranked queries {i+1}/32", flush=True)
        for seed in [42, 43, 44]:
            control = next(
                r
                for r in controls["runs"]
                if r["block"] == block and r["method"] == "ratio90" and r["seed"] == seed
            )
            assert counts == control["per_pair_counts"]
            run = root / f"precision90_s{seed}"
            run.mkdir()
            image_ids = create_database(
                run / "database.db",
                [frames[n] for n in names],
                features,
                make_camera(960, 720, calibration),
            )
            insert_matches(run / "database.db", image_ids, results)
            pair_file = run / "pairs.txt"
            pair_file.write_text(
                "\n".join(r["a"] + " " + r["b"] for r in results if len(r["pairs"]) >= 15)
            )
            result = {
                "block": block,
                "method": "precision90",
                "seed": seed,
                "path": str(run),
                "selected_matches": sum(counts),
                "per_pair_counts": counts,
                "architecture": winner,
            }
            begin = time.perf_counter()
            print(f"{block}: mapping precision seed{seed}", flush=True)
            try:
                model, _ = reconstruct(
                    run / "database.db",
                    base / block / "images",
                    run / "sparse",
                    pair_file,
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
                {k: result.get(k) for k in ["block", "seed", "registered_images", "trajectory"]},
                flush=True,
            )
    report["status"] = "complete"
    write_json(args.output / "comparison_v1.json", report)


if __name__ == "__main__":
    main()
