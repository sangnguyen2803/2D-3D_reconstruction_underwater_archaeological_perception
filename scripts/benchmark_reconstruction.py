"""Controlled 32-view SfM ablation using frozen features and held-out model scores."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.evaluation.metrics import trajectory_metrics
from underwater_vision.matchability.build import load_features
from underwater_vision.matchability.protocol import digest, read_json
from underwater_vision.matchability.tabular_baseline import DinoFeatureCache
from underwater_vision.matching.learned_filter import MatchPredictor, candidates
from underwater_vision.reconstruction.colmap.database import (
    create_database,
    insert_matches,
    make_camera,
)
from underwater_vision.reconstruction.colmap.exporter import export_reconstruction
from underwater_vision.reconstruction.colmap.mapper import reconstruct
from underwater_vision.research_pipeline import unique_matches
from underwater_vision.utils.io import atomic_npz, write_json

METHODS = [
    "ratio_full",
    "context_full",
    "response75_ratio",
    "random75_ratio",
    "segformer75_ratio",
    "segformer75_context",
]
BLOCKS = [("block273", 273, "fold2"), ("block599", 599, "fold1"), ("block1116", 1116, "fold2")]


def select_record(record, scores, fraction=0.75):
    count = len(record["xy"])
    retained = max(2, int(np.ceil(count * fraction)))
    ids = np.sort(np.argsort(-scores, kind="stable")[:retained])
    return {
        k: (
            v[ids]
            if isinstance(v, np.ndarray) and v.ndim and len(v) == count and k != "shape"
            else v
        )
        for k, v in record.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/reconstruction_improvement_v1")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    data_root = Path("outputs/matchability_labels_v3")
    dataset_manifest = read_json(data_root / "manifest_v3.json")
    fold_manifest = read_json("outputs/segformer_folds_v3/fold_manifest_v3.json")
    checkpoints = {
        (r["protocol"], r["seed"]): Path(r["path"])
        for r in fold_manifest["runs"]
        if r["kind"] == "segformer_b0"
    }
    predictor = MatchPredictor("outputs/match_filter_v1/context_v1.pt")
    trained = set(predictor.metadata["training_images"] + predictor.metadata["validation_images"])
    v1_test = set(
        next(p for p in dataset_manifest["protocols"] if p["name"] == "v1_partition")["groups"][
            "test"
        ]
    )
    dataset = MermaidDataset("data/raw/mermaid")
    frame_lookup = {f.name: f for f in dataset.frames}
    calibration = yaml.safe_load(Path("configs/data/mermaid.yaml").read_text())["calibration"]
    settings = yaml.safe_load(Path("configs/reconstruction/default.yaml").read_text())
    settings.update(image_count=32, dense=False)
    report = {
        "status": "running",
        "protocol": "Three filename-contiguous 32-image blocks in intersection of v1 and corresponding SegFormer test regions. Same 171 window-6 pairs and fixed intrinsics. 75% per-image keypoint budgets are identical for response/random/SegFormer selectors. SegFormer training seed and mapper seed paired across 42/43/44; v1 context weights fixed seed42. No reference poses or epipolar labels used to select matches or fit geometry.",
        "selection": "First 32 images in the first qualifying contiguous intersection run for fold1, and first 32 images from each of the two qualifying runs for fold2. No qualifying 32-contiguous-image intersection exists for fold0. These regions were already examined in prior learning benchmarks.",
        "primary_success": "No lower registration than control; lower reference-aligned translation error on identical registered cameras; report rotation and reprojection together. Point count alone is not accuracy.",
        "source_sha256": digest(__file__),
        "context_checkpoint_sha256": digest("outputs/match_filter_v1/context_v1.pt"),
        "dataset_manifest_sha256": digest(data_root / "manifest_v3.json"),
        "settings": settings,
        "methods": METHODS,
        "runs": [],
    }
    write_json(args.output / "protocol_v1.json", report)
    cache = DinoFeatureCache(data_root)
    for block, start, fold in BLOCKS:
        names = [f"SR202204_LDM-S_D01_{i:04}.JPG" for i in range(start, start + 32)]
        assert not set(names) & trained and set(names) <= v1_test
        fold_test = set(
            next(p for p in dataset_manifest["protocols"] if p["name"] == fold)["groups"]["test"]
        )
        assert set(names) <= fold_test
        frames = [frame_lookup[n] for n in names]
        block_root = args.output / block
        block_root.mkdir()
        image_dir = block_root / "images"
        image_dir.mkdir()
        features = {}
        for name in names:
            record = load_features(data_root, name)
            record["semantic"] = cache.descriptors(name, record["xy"])
            features[name] = record
            # Reuse already decoded, unchanged RGB960 PNGs without duplicating them.
            (image_dir / name).hardlink_to((data_root / "images" / (name + ".png")).resolve())
        reference = {f.name: f.reference_c2w for f in frames}
        write_json(
            block_root / "reference_poses_v1.json", {n: p.tolist() for n, p in reference.items()}
        )
        write_json(
            block_root / "input_manifest_v1.json",
            {
                "names": names,
                "fold": fold,
                "features": {n: len(v["xy"]) for n, v in features.items()},
            },
        )
        pairs = [(i, j) for i in range(32) for j in range(i + 1, min(32, i + 7))]
        precomputed = {}
        for seed in [42, 43, 44]:
            model_root = checkpoints[(fold, seed)]
            dense_metadata = read_json(model_root / "resolved_config_v3.json")
            assert not set(names) & set(
                dense_metadata["training_images"] + dense_metadata["validation_images"]
            )
            scores = {}
            for name in names:
                with np.load(model_root / "predictions" / (name + ".npz")) as z:
                    assert len(z["logits"]) == len(features[name]["xy"])
                    scores[name] = z["logits"]
            rng = np.random.default_rng(seed)
            for method in METHODS:
                selected = features
                if "75" in method:
                    selected = {}
                    for name, record in features.items():
                        rank = (
                            record["response"]
                            if method.startswith("response")
                            else (
                                rng.random(len(record["xy"]))
                                if method.startswith("random")
                                else scores[name]
                            )
                        )
                        selected[name] = select_record(record, rank)
                key = (method, seed if method.startswith(("segformer", "random")) else 0)
                run_root = block_root / f"{method}_s{seed}"
                run_root.mkdir()
                begin = time.perf_counter()
                print(f"{block} {method} seed{seed}: matching/reconstruction", flush=True)
                if key not in precomputed:
                    results = []
                    for i, j in pairs:
                        a, b = names[i], names[j]
                        fa, fb = selected[a], selected[b]
                        h, w = map(int, fa["shape"])
                        ids, x = candidates(fa, fb, w, h)
                        if method.endswith("context") or method == "context_full":
                            probability = predictor(x)
                            keep = unique_matches(ids, probability, predictor.metadata["threshold"])
                        else:
                            keep = np.flatnonzero((x[:, 1] < 0.8) & (x[:, 2] > 0.5))
                        results.append(
                            {
                                "a": a,
                                "b": b,
                                "pairs": ids[keep],
                                "inliers": np.ones(len(keep), bool),
                            }
                        )
                    precomputed[key] = results
                results = precomputed[key]
                h, w = map(int, features[names[0]]["shape"])
                ids = create_database(
                    run_root / "database.db", frames, selected, make_camera(w, h, calibration)
                )
                insert_matches(run_root / "database.db", ids, results)
                pair_file = run_root / "pairs.txt"
                pair_file.write_text(
                    "\n".join(r["a"] + " " + r["b"] for r in results if len(r["pairs"]) >= 15)
                )
                result = {
                    "block": block,
                    "fold": fold,
                    "method": method,
                    "seed": seed,
                    "images": len(names),
                    "pairs": len(pairs),
                    "selected_features": sum(len(v["xy"]) for v in selected.values()),
                    "selected_matches": sum(len(r["pairs"]) for r in results),
                    "path": str(run_root),
                    "segformer_checkpoint_hash": (
                        digest(model_root / "model_v3.pt")
                        if method.startswith("segformer")
                        else None
                    ),
                }
                try:
                    model, _ = reconstruct(
                        run_root / "database.db",
                        image_dir,
                        run_root / "sparse",
                        pair_file,
                        settings,
                        seed,
                    )
                    poses, statistics = export_reconstruction(model, run_root)
                    result.update(
                        status="complete",
                        **statistics,
                        trajectory=trajectory_metrics(poses, reference),
                    )
                    atomic_npz(
                        run_root / "points_v1.npz",
                        xyz=np.array([p.xyz for p in model.points3D.values()]),
                    )
                except RuntimeError as error:
                    result.update(
                        status="failed", error=str(error), registered_images=0, sparse_points=0
                    )
                result["runtime_seconds"] = time.perf_counter() - begin
                write_json(run_root / "metrics_v1.json", result)
                report["runs"].append(result)
                write_json(args.output / "comparison_v1.json", report)
                print(
                    json.dumps(
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
                        }
                    ),
                    flush=True,
                )
        del precomputed
    report["status"] = "complete"
    write_json(args.output / "comparison_v1.json", report)


if __name__ == "__main__":
    main()
