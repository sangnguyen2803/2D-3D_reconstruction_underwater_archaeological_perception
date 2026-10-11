"""Learned correspondence ranking with fixed feature/pair/retained-match budgets."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
import yaml
from benchmark_reconstruction import BLOCKS
from scipy.special import expit

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.evaluation.metrics import trajectory_metrics
from underwater_vision.matchability.build import load_features
from underwater_vision.matchability.protocol import digest, read_json
from underwater_vision.matchability.tabular_baseline import DinoFeatureCache
from underwater_vision.matching.budget import budget_indices
from underwater_vision.matching.learned_filter import MatchPredictor, candidates
from underwater_vision.reconstruction.colmap.database import (
    create_database,
    insert_matches,
    make_camera,
)
from underwater_vision.reconstruction.colmap.exporter import export_reconstruction
from underwater_vision.reconstruction.colmap.mapper import reconstruct
from underwater_vision.utils.io import atomic_npz, write_json

METHODS = ["ratio90", "random90", "mlp90", "tight90", "segformer90", "tight_segformer90"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/geometry_ranking_v1"))
    parser.add_argument(
        "--ranker", type=Path, default=Path("outputs/geometry_ranker_v1/ranker_v1.pt")
    )
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    pycolmap.logging.minloglevel = 2
    parent = Path("outputs/reconstruction_improvement_v1")
    base = read_json(parent / "comparison_v1.json")
    manifest = read_json("outputs/segformer_folds_v3/fold_manifest_v3.json")
    checkpoints = {
        (r["protocol"], r["seed"]): Path(r["path"])
        for r in manifest["runs"]
        if r["kind"] == "segformer_b0"
    }
    predictors = {
        "mlp": MatchPredictor("outputs/match_filter_v1/mlp_v1.pt"),
        "tight": MatchPredictor(args.ranker),
    }
    frames = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
    calibration = yaml.safe_load(Path("configs/data/mermaid.yaml").read_text())["calibration"]
    cache = DinoFeatureCache(Path("outputs/matchability_labels_v3"))
    report = {
        "status": "running",
        "protocol": "Frozen SIFT contrast0.04 features, same 171 window6 pairs and fixed intrinsics. Every scorer retains ceil(90% mutual ratio0.8 survivors), minimum128 or all available. Same retained count for every pair. SegFormer uses endpoint scores rather than dropping keypoints. No test pose or reference label selects matches. Geometry MLP checkpoint chosen using original validation pairs only. This is a fixed development comparison on already examined spatial test blocks.",
        "primary_success": "No lost cameras versus ratio_full; lower common-camera ATE than full baseline and matched-budget ratio/random controls, paired95% interval below zero. Retain all regressions. No claim of new-scene generalization.",
        "methods": args.methods,
        "retention": 0.9,
        "minimum_per_pair": 128,
        "source_sha256": digest(__file__),
        "checkpoint_sha256": {
            k: digest("outputs/match_filter_v1/mlp_v1.pt" if k == "mlp" else args.ranker)
            for k in predictors
        },
        "baseline_sha256": digest(parent / "comparison_v1.json"),
        "ranker_path": str(args.ranker),
        "ranker_architecture": predictors["tight"].metadata["architecture"],
        "candidate_cache": str(args.candidate_cache) if args.candidate_cache else None,
        "runs": [],
    }
    write_json(args.output / "protocol_v1.json", report)
    if args.candidate_cache:
        cached_report = read_json(args.candidate_cache / "comparison_v1.json")
        assert cached_report["baseline_sha256"] == report["baseline_sha256"]
    for block, _, fold in BLOCKS:
        block_root = args.output / block
        block_root.mkdir()
        source = parent / block
        names = read_json(source / "input_manifest_v1.json")["names"]
        for predictor in predictors.values():
            assert not set(names) & set(
                predictor.metadata["training_images"] + predictor.metadata["validation_images"]
            )
        features = {}
        for name in names:
            record = load_features(Path("outputs/matchability_labels_v3"), name)
            record["semantic"] = cache.descriptors(name, record["xy"])
            features[name] = record
        pairs = [(i, j) for i in range(32) for j in range(i + 1, min(32, i + 7))]
        prepared = []
        for k, (i, j) in enumerate(pairs):
            a, b = names[i], names[j]
            h, w = map(int, features[a]["shape"])
            if args.candidate_cache:
                with np.load(args.candidate_cache / block / f"candidates_{k:04}.npz") as z:
                    ids, x = z["pairs"].copy(), z["x"].copy()
                assert np.allclose(x[:, 6:8], features[a]["xy"][ids[:, 0]] / [w, h])
                assert np.allclose(x[:, 8:10], features[b]["xy"][ids[:, 1]] / [w, h])
            else:
                ids, x = candidates(features[a], features[b], w, h)
            scores = {key: predictor(x) for key, predictor in predictors.items()}
            atomic_npz(block_root / f"candidates_{k:04}.npz", pairs=ids, x=x, **scores)
            prepared.append((a, b, ids, x, scores))
        static = {}
        for seed in [42, 43, 44]:
            model_root = checkpoints[(fold, seed)]
            metadata = read_json(model_root / "resolved_config_v3.json")
            assert not set(names) & set(metadata["training_images"] + metadata["validation_images"])
            maps = {}
            for name in names:
                with np.load(model_root / "predictions" / (name + ".npz")) as z:
                    assert len(z["logits"]) == len(features[name]["xy"])
                    maps[name] = expit(z["logits"])
            for method in args.methods:
                key = (
                    method,
                    seed if method in {"random90", "segformer90", "tight_segformer90"} else 0,
                )
                if key not in static:
                    results = []
                    rng = np.random.default_rng(seed)
                    for a, b, ids, x, scores in prepared:
                        endpoint = np.sqrt(maps[a][ids[:, 0]] * maps[b][ids[:, 1]])
                        ranking = {
                            "ratio90": 1 - x[:, 1],
                            "random90": rng.random(len(ids)),
                            "mlp90": scores["mlp"],
                            "tight90": scores["tight"],
                            "segformer90": endpoint,
                            "tight_segformer90": endpoint * scores["tight"],
                        }[method]
                        keep = budget_indices(x, ranking)
                        results.append(
                            {
                                "a": a,
                                "b": b,
                                "pairs": ids[keep],
                                "inliers": np.ones(len(keep), bool),
                            }
                        )
                    static[key] = results
                results = static[key]
                counts = [len(r["pairs"]) for r in results]
                # Exact per-pair budgets, not just equal total counts.
                reference_counts = [
                    len(budget_indices(x, 1 - x[:, 1])) for _, _, _, x, _ in prepared
                ]
                assert counts == reference_counts
                root = block_root / f"{method}_s{seed}"
                root.mkdir()
                h, w = map(int, features[names[0]]["shape"])
                image_ids = create_database(
                    root / "database.db",
                    [frames[n] for n in names],
                    features,
                    make_camera(w, h, calibration),
                )
                insert_matches(root / "database.db", image_ids, results)
                pair_file = root / "pairs.txt"
                pair_file.write_text(
                    "\n".join(r["a"] + " " + r["b"] for r in results if len(r["pairs"]) >= 15)
                )
                begin = time.perf_counter()
                result = {
                    "block": block,
                    "method": method,
                    "seed": seed,
                    "path": str(root),
                    "selected_matches": sum(counts),
                    "per_pair_counts": counts,
                    "features": sum(len(v["xy"]) for v in features.values()),
                    "segformer_checkpoint_sha256": (
                        digest(model_root / "model_v3.pt") if "segformer" in method else None
                    ),
                }
                try:
                    model, _ = reconstruct(
                        root / "database.db",
                        source / "images",
                        root / "sparse",
                        pair_file,
                        base["settings"],
                        seed,
                    )
                    poses, statistics = export_reconstruction(model, root)
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
                write_json(root / "metrics_v1.json", result)
                report["runs"].append(result)
                write_json(args.output / "comparison_v1.json", report)
                print(
                    {
                        k: result.get(k)
                        for k in [
                            "block",
                            "method",
                            "seed",
                            "registered_images",
                            "sparse_points",
                            "mean_reprojection_error_px",
                            "trajectory",
                        ]
                    },
                    flush=True,
                )
    report["status"] = "complete"
    write_json(args.output / "comparison_v1.json", report)


if __name__ == "__main__":
    main()
