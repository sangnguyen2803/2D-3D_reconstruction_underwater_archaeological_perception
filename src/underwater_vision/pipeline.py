"""Experiment orchestration with immutable run directories and shared caches."""

import json
import random
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.evaluation.metrics import trajectory_metrics
from underwater_vision.features.local import extract_sift
from underwater_vision.geometry.verification import verify
from underwater_vision.matching.matcher import match_local, score_matches
from underwater_vision.preprocessing.images import resize_rgb
from underwater_vision.reconstruction.colmap.database import (
    create_database,
    insert_matches,
    make_camera,
)
from underwater_vision.reconstruction.colmap.exporter import export_reconstruction
from underwater_vision.reconstruction.colmap.mapper import reconstruct
from underwater_vision.utils.io import atomic_npz, fingerprint, read_rgb, write_json, write_rgb


def run(config):
    cfg = OmegaConf.to_container(config, resolve=True)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    name = cfg["run_name"] or cfg["experiment_name"] + "_" + datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S"
    )
    output = Path(cfg["output_root"]) / name
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, output / "config.yaml")
    start = time.perf_counter()
    try:
        metrics = execute(cfg, output)
        metrics["runtime_seconds"] = time.perf_counter() - start
        write_json(output / "metrics.json", metrics)
        write_json(output / "status.json", {"status": "completed", "stage": cfg["stage"]})
        if cfg["tracking"]:
            import os

            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            import mlflow

            mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
            mlflow.set_experiment("underwater-vision")
            with mlflow.start_run(run_name=name):
                mlflow.log_params(
                    {
                        "experiment": cfg["experiment_name"],
                        "seed": seed,
                        "backbone": cfg["model"]["backbone"],
                        "image_count": metrics["images"],
                    }
                )
                mlflow.log_metrics(
                    {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                )
                mlflow.log_artifacts(str(output))
                if cfg["matching"].get("learned_checkpoint"):
                    mlflow.log_artifact(cfg["matching"]["learned_checkpoint"], "model")
        print(json.dumps(metrics, indent=2), flush=True)
    except Exception as exc:
        write_json(
            output / "status.json",
            {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()},
        )
        raise
    return output


def execute(cfg, output):
    data, model, matching = cfg["data"], cfg["model"], cfg["matching"]
    dataset = MermaidDataset(data["root"], data["image_dir"])
    frames = dataset.select(data["start"], data["count"], data["stride"])
    if data.get("image_list"):
        requested = Path(data["image_list"]).read_text(encoding="utf-8").splitlines()
        lookup = {f.name: f for f in dataset.frames}
        missing = set(requested) - set(lookup)
        if missing:
            raise FileNotFoundError(f"Images missing from immutable selection: {sorted(missing)}")
        frames = [lookup[n] for n in requested if n]
    if len(frames) < 2:
        raise ValueError("At least two input images required")
    if matching["reliability"] == "learned":
        import hashlib

        from underwater_vision.visibility.learned import load_head

        _, head_metadata = load_head(matching["learned_checkpoint"])
        if (
            head_metadata["backbone"] != model["backbone"]
            or head_metadata["revision"] != model["revision"]
        ):
            raise ValueError("Reliability checkpoint backbone/revision mismatch")
        # Numeric acquisition IDs provide an additional exclusion around training frames.
        import re

        training_ids = [
            int(re.search(r"(\d+)\.[^.]+$", n).group(1)) for n in head_metadata["training_images"]
        ]
        evaluation_ids = [int(re.search(r"(\d+)\.[^.]+$", f.name).group(1)) for f in frames]
        if min(abs(a - b) for a in training_ids for b in evaluation_ids) <= 20:
            raise ValueError(
                "Training/evaluation leakage: require a >20 acquisition-frame exclusion gap"
            )
        with Path(matching["learned_checkpoint"]).open("rb") as handle:
            matching["checkpoint_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        write_json(
            output / "reliability_checkpoint.json",
            {
                "file": matching["learned_checkpoint"],
                "sha256": matching["checkpoint_sha256"],
                "training_images": head_metadata["training_images"],
            },
        )
    image_dir = output / "images"
    cache = Path(cfg["cache_root"])
    (cache / "features").mkdir(parents=True, exist_ok=True)
    (cache / "pairs").mkdir(parents=True, exist_ok=True)
    (cache / "visibility").mkdir(parents=True, exist_ok=True)
    features, feature_keys = {}, {}
    neural = None
    if model["extractor"] == "sift_dinov2":
        from underwater_vision.features.dinov2 import DinoExtractor

        neural = DinoExtractor(model["backbone"], model["device"], model["revision"])
        write_json(output / "backbone.json", neural.provenance)
    elif model["extractor"] != "sift":
        raise ValueError("Supported extractors: sift, sift_dinov2")
    for index, frame in enumerate(frames):
        image, _scale = resize_rgb(read_rgb(frame.path), data["max_size"], data["enhance"])
        write_rgb(image_dir / frame.name, image)
        settings = {
            "data": {"max_size": data["max_size"], "enhance": data["enhance"]},
            "model": {k: model[k] for k in ["extractor", "max_features", "backbone", "revision"]},
            "version": 1,
            "revision": neural.provenance if neural else None,
        }
        key = fingerprint(frame.path, settings)
        feature_keys[frame.name] = key
        path = cache / "features" / (key + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                feat = {k: z[k] for k in z.files}
        else:
            xy, descriptor = extract_sift(image, model["max_features"])
            feat = {"xy": xy, "local": descriptor}
            if neural:
                feat["semantic"] = neural.extract(image, xy)
            atomic_npz(path, **feat)
        feat["reliability"] = np.ones(len(feat["xy"]), np.float32)
        import cv2

        frame_camera = make_camera(image.shape[1], image.shape[0], data["calibration"])
        K = frame_camera.calibration_matrix()
        K[:2, 2] -= 0.5  # Convert COLMAP centers to OpenCV's coordinate convention.
        feat["geometry_xy"] = (
            cv2.undistortPoints(
                feat["xy"][:, None, :], K, np.asarray(frame_camera.params[4:]), P=K
            ).reshape(-1, 2)
            if len(feat["xy"])
            else feat["xy"]
        )
        if matching["reliability"] != "none":
            from underwater_vision.visibility.deterministic import estimate_visibility, sample_map

            visibility_key = fingerprint(
                frame.path,
                {
                    "max_size": data["max_size"],
                    "enhance": data["enhance"],
                    "window": matching["quality_window"],
                    "version": 1,
                },
            )
            visibility_cache = cache / "visibility" / (visibility_key + ".npz")
            if visibility_cache.exists():
                with np.load(visibility_cache, allow_pickle=False) as cached:
                    reliability = cached["reliability"]
            else:
                reliability, _ = estimate_visibility(image, matching["quality_window"])
                atomic_npz(visibility_cache, reliability=reliability)
            if matching["reliability"] == "learned":
                from underwater_vision.visibility.learned import predict

                feat["reliability"] = predict(feat["semantic"], matching["learned_checkpoint"])
            elif matching["reliability"] == "deterministic":
                feat["reliability"] = sample_map(reliability, feat["xy"])
            else:
                raise ValueError("Unknown reliability method")
            from underwater_vision.visualization.figures import visibility_figure

            if index < 3:
                visibility_figure(
                    image, reliability, output / "figures" / (frame.path.stem + "_visibility.png")
                )
                if matching["reliability"] == "learned":
                    from underwater_vision.visualization.figures import learned_reliability_figure

                    learned_reliability_figure(
                        image,
                        feat["xy"],
                        feat["reliability"],
                        output / "figures" / (frame.path.stem + "_learned.png"),
                    )
        features[frame.name] = feat
        print(f'Features {index+1}/{len(frames)}: {frame.name} ({len(feat["xy"])})', flush=True)
    write_json(
        output / "input_manifest.json",
        [
            {"name": f.name, "path": str(f.path.resolve()), "cache_key": feature_keys[f.name]}
            for f in frames
        ],
    )
    if cfg["stage"] == "features":
        return {"images": len(frames), "features": sum(len(f["xy"]) for f in features.values())}
    if cfg["stage"] not in {"all", "matches"}:
        raise ValueError("stage must be all, features or matches")
    results = []
    for i in range(len(frames)):
        for j in range(i + 1, min(i + 1 + matching["window"], len(frames))):
            a, b = frames[i].name, frames[j].name
            fa, fb = features[a], features[b]
            key = fingerprint(
                frames[i].path,
                {
                    "a": feature_keys[a],
                    "b": feature_keys[b],
                    "matching": matching,
                    "semantic_weight": model["semantic_weight"],
                    "seed": cfg["seed"],
                    "calibration": data["calibration"],
                    "version": 3,
                },
            )
            path = cache / "pairs" / (key + ".npz")
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    result = {k: z[k] for k in z.files}
            else:
                desc_a, desc_b = fa["local"], fb["local"]
                if "semantic" in fa:
                    alpha = model["semantic_weight"]
                    desc_a = np.c_[
                        np.sqrt(1 - alpha) * desc_a, np.sqrt(alpha) * fa["semantic"]
                    ].astype(np.float32)
                    desc_b = np.c_[
                        np.sqrt(1 - alpha) * desc_b, np.sqrt(alpha) * fb["semantic"]
                    ].astype(np.float32)
                pairs, local = match_local(desc_a, desc_b, matching["ratio"])
                use_reliability = matching["reliability"] != "none"
                score = score_matches(
                    pairs,
                    local,
                    fa.get("semantic"),
                    fb.get("semantic"),
                    fa["reliability"] if use_reliability else None,
                    fb["reliability"] if use_reliability else None,
                    model["semantic_weight"],
                )
                keep = score >= matching["min_score"]
                pairs, score = pairs[keep], score[keep]
                F, inliers, errors = verify(
                    fa["geometry_xy"][pairs[:, 0]],
                    fb["geometry_xy"][pairs[:, 1]],
                    score,
                    matching["ransac_threshold"],
                    matching["ransac_iterations"],
                    cfg["seed"],
                    use_reliability,
                )
                result = {
                    "pairs": pairs,
                    "score": score,
                    "inliers": inliers,
                    "errors": errors,
                    "F": np.zeros((3, 3)) if F is None else F,
                }
                atomic_npz(path, **result)
            result.update(a=a, b=b)
            results.append(result)
    if matching["cross_view"]:
        from underwater_vision.visibility.consistency import update_repeatability

        update_repeatability(features, results, matching["consistency_prior"])
        # Each pair is reverified after per-feature consistency aggregation.
        for result in results:
            fa, fb = features[result["a"]], features[result["b"]]
            pairs = result["pairs"]
            score = (
                result["score"]
                * fa["repeatability"][pairs[:, 0]]
                * fb["repeatability"][pairs[:, 1]]
            )
            F, mask, error = verify(
                fa["geometry_xy"][pairs[:, 0]],
                fb["geometry_xy"][pairs[:, 1]],
                score,
                matching["ransac_threshold"],
                matching["ransac_iterations"],
                cfg["seed"],
                True,
            )
            result.update(
                score=score, inliers=mask, errors=error, F=np.zeros((3, 3)) if F is None else F
            )
    pair_stats = []
    (output / "matches").mkdir(exist_ok=True)
    for k, r in enumerate(results):
        np.savez_compressed(output / "matches" / f"{k:06d}.npz", **r)
        n, ni = len(r["pairs"]), int(r["inliers"].sum())
        pair_stats.append(
            {
                "a": r["a"],
                "b": r["b"],
                "matches": n,
                "inliers": ni,
                "inlier_ratio": ni / max(n, 1),
                "verified": ni >= 15,
                "median_epipolar_error_px": (
                    float(np.median(r["errors"][r["inliers"]])) if ni else None
                ),
            }
        )
    write_json(output / "pair_metrics.json", pair_stats)
    total = sum(p["matches"] for p in pair_stats)
    inliers = sum(p["inliers"] for p in pair_stats)
    metrics = {
        "images": len(frames),
        "pairs": len(results),
        "matches": total,
        "inliers": inliers,
        "inlier_ratio": inliers / max(total, 1),
        "geometric_success_rate": sum(p["verified"] for p in pair_stats) / max(len(pair_stats), 1),
    }
    if cfg["stage"] == "matches":
        return metrics
    height, width = read_rgb(image_dir / frames[0].name).shape[:2]
    camera = make_camera(width, height, data["calibration"])
    database = output / "database.db"
    ids = create_database(database, frames, features, camera)
    insert_matches(database, ids, results)
    pairs_file = output / "pairs.txt"
    pairs_file.write_text(
        "\n".join(r["a"] + " " + r["b"] for r in results if r["inliers"].sum() >= 15),
        encoding="utf-8",
    )
    settings = dict(cfg["reconstruction"], image_count=len(frames))
    write_json(output / "metrics.json", metrics)
    try:
        reconstruction, sparse_path = reconstruct(
            database, image_dir, output / "sparse", pairs_file, settings, cfg["seed"]
        )
    except RuntimeError as exc:
        if "registered no model" in str(exc):
            metrics.update(
                registered_images=0,
                sparse_points=0,
                sparse_status="failed",
                failure_reason=str(exc),
            )
            write_json(output / "metrics.json", metrics)
        raise
    poses, sparse_stats = export_reconstruction(reconstruction, output)
    metrics.update(sparse_stats)
    reference = {f.name: f.reference_c2w for f in frames if f.reference_c2w is not None}
    metrics["trajectory"] = trajectory_metrics(poses, reference)
    from underwater_vision.visualization.figures import trajectory_figure

    trajectory_figure(poses, reference, output / "figures" / "trajectory.png")
    write_json(output / "reference_poses.json", {n: p.tolist() for n, p in reference.items()})
    from underwater_vision.pointcloud.confidence import sparse_confidence

    sparse_confidence(reconstruction, features, output, settings)
    from underwater_vision.visualization.figures import matching_figure, reconstruction_figure

    reconstruction_figure(reconstruction, output / "figures" / "reconstruction.png")
    if results:
        matching_figure(
            read_rgb(image_dir / results[0]["a"]),
            read_rgb(image_dir / results[0]["b"]),
            features[results[0]["a"]]["xy"],
            features[results[0]["b"]]["xy"],
            results[0],
            output / "figures" / "matches.png",
        )
    metrics["dense_status"] = "not_requested"
    if settings["dense"]:
        from underwater_vision.reconstruction.colmap.mvs import run_mvs

        cloud = run_mvs(
            sparse_path,
            image_dir,
            output / "dense",
            settings["colmap_executable"],
            settings["dense_max_size"],
            settings["mesh"],
        )
        metrics["dense_status"] = "completed"
        metrics["dense_cloud"] = str(cloud)
        from underwater_vision.pointcloud.dense_confidence import dense_confidence

        metrics.update(dense_confidence(output / "dense"))
    return metrics
