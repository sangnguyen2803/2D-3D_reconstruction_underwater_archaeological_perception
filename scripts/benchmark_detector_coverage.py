"""Paired detector-coverage ablation on the same 32-view reconstruction blocks."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import yaml

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.evaluation.metrics import trajectory_metrics
from underwater_vision.features.local import extract_sift
from underwater_vision.matchability.protocol import digest, read_json
from underwater_vision.matching.learned_filter import candidates
from underwater_vision.reconstruction.colmap.database import (
    create_database,
    insert_matches,
    make_camera,
)
from underwater_vision.reconstruction.colmap.exporter import export_reconstruction
from underwater_vision.reconstruction.colmap.mapper import reconstruct
from underwater_vision.utils.io import atomic_npz, read_rgb, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/detector_coverage_v1"))
    parser.add_argument("--contrast", type=float, default=0.01)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(4)
    pycolmap.logging.minloglevel = 2
    parent = Path("outputs/reconstruction_improvement_v1")
    original = read_json(parent / "comparison_v1.json")
    assert original["status"] == "complete"
    frames = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
    calibration = yaml.safe_load(Path("configs/data/mermaid.yaml").read_text())["calibration"]
    settings = original["settings"]
    report = {
        "status": "running",
        "contrast_threshold": args.contrast,
        "baseline_threshold": 0.04,
        "baseline_report_sha256": digest(parent / "comparison_v1.json"),
        "protocol": "Same immutable RGB960 pixels, 32-view blocks, window-6 pair graph, 4096 feature cap, mutual ratio0.8 matching, fixed calibration and COLMAP settings. Only SIFT contrast threshold changes. Actual detected feature/match counts and costs are explicitly reported. No reference pose or epipolar label fits or selects matches. This detector setting is a development hypothesis, not a blind new-scene validation.",
        "source_sha256": {
            str(p): digest(p)
            for p in [Path(__file__), Path("src/underwater_vision/features/local.py")]
        },
        "runs": [],
    }
    write_json(args.output / "protocol_v1.json", report)
    for block in ["block273", "block599", "block1116"]:
        source = parent / block
        names = read_json(source / "input_manifest_v1.json")["names"]
        selected = [frames[n] for n in names]
        block_root = args.output / block
        block_root.mkdir()
        features = {}
        extraction_begin = time.perf_counter()
        for name in names:
            rgb = read_rgb(source / "images" / name)
            xy, descriptors = extract_sift(rgb, 4096, args.contrast)
            features[name] = {
                "xy": xy,
                "local": descriptors,
                "quality": np.ones(len(xy)),
                "semantic": np.zeros((len(xy), 1)),
                "shape": np.asarray(rgb.shape[:2]),
            }
            atomic_npz(block_root / (name + ".npz"), xy=xy, local=descriptors)
        extraction_seconds = time.perf_counter() - extraction_begin
        results = []
        match_begin = time.perf_counter()
        for i in range(32):
            for j in range(i + 1, min(32, i + 7)):
                a, b = names[i], names[j]
                h, w = map(int, features[a]["shape"])
                ids, x = candidates(features[a], features[b], w, h)
                keep = (x[:, 1] < 0.8) & (x[:, 2] > 0.5)
                results.append(
                    {"a": a, "b": b, "pairs": ids[keep], "inliers": np.ones(keep.sum(), bool)}
                )
        matching_seconds = time.perf_counter() - match_begin
        counts = [len(f["xy"]) for f in features.values()]
        print(
            f"{block} contrast={args.contrast}: keypoint min/median/max={min(counts)}/{np.median(counts)}/{max(counts)}",
            flush=True,
        )
        write_json(block_root / "feature_counts_v1.json", dict(zip(names, counts, strict=True)))
        for seed in [42, 43, 44]:
            run_root = block_root / f"low_contrast_s{seed}"
            run_root.mkdir()
            h, w = map(int, features[names[0]]["shape"])
            ids = create_database(
                run_root / "database.db", selected, features, make_camera(w, h, calibration)
            )
            insert_matches(run_root / "database.db", ids, results)
            pairs = run_root / "pairs.txt"
            pairs.write_text(
                "\n".join(r["a"] + " " + r["b"] for r in results if len(r["pairs"]) >= 15)
            )
            begin = time.perf_counter()
            record = {
                "block": block,
                "method": "low_contrast_ratio",
                "seed": seed,
                "path": str(run_root),
                "features": sum(counts),
                "selected_matches": sum(len(r["pairs"]) for r in results),
                "pairs": len(results),
                "extraction_seconds": extraction_seconds,
                "matching_seconds": matching_seconds,
            }
            try:
                model, _ = reconstruct(
                    run_root / "database.db",
                    source / "images",
                    run_root / "sparse",
                    pairs,
                    settings,
                    seed,
                )
                poses, statistics = export_reconstruction(model, run_root)
                reference = {n: f.reference_c2w for n, f in zip(names, selected, strict=True)}
                record.update(
                    status="complete", **statistics, trajectory=trajectory_metrics(poses, reference)
                )
            except RuntimeError as error:
                record.update(
                    status="failed", error=str(error), registered_images=0, sparse_points=0
                )
            record["mapping_seconds"] = time.perf_counter() - begin
            write_json(run_root / "metrics_v1.json", record)
            report["runs"].append(record)
            write_json(args.output / "comparison_v1.json", report)
            print(json.dumps(record), flush=True)
    report["status"] = "complete"
    write_json(args.output / "comparison_v1.json", report)


if __name__ == "__main__":
    main()
