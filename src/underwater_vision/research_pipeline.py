"""v1 probability-filtered COLMAP input; reference poses never select inliers."""

import hashlib
from pathlib import Path

import numpy as np

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.data.spatial import assign_spatial
from underwater_vision.evaluation.metrics import trajectory_metrics
from underwater_vision.features.dinov2 import DinoExtractor
from underwater_vision.features.store import feature_record
from underwater_vision.geometry.reference import camera_parameters, undistort
from underwater_vision.matching.learned_filter import MatchPredictor, candidates
from underwater_vision.preprocessing.images import resize_rgb
from underwater_vision.reconstruction.colmap.database import (
    create_database,
    insert_matches,
    make_camera,
)
from underwater_vision.reconstruction.colmap.exporter import export_reconstruction
from underwater_vision.reconstruction.colmap.mapper import reconstruct
from underwater_vision.retrieval.pairs import retrieve_pairs
from underwater_vision.utils.io import atomic_npz, read_rgb, write_json, write_rgb


def unique_matches(pairs, probability, threshold):
    accepted, seen_a, seen_b = [], set(), set()
    for i in np.argsort(-probability, kind="stable"):
        a, b = pairs[i]
        if probability[i] >= threshold and a not in seen_a and b not in seen_b:
            accepted.append(int(i))
            seen_a.add(a)
            seen_b.add(b)
    return np.asarray(sorted(accepted), np.int32)


def execute_learned(cfg, output):
    data, matching = cfg["data"], cfg["matching"]
    dataset = MermaidDataset(data["root"], data["image_dir"])
    lookup = {f.name: f for f in dataset.frames}
    frames = dataset.select(data["start"], data["count"], data["stride"])
    if data.get("image_list"):
        frames = [lookup[n] for n in Path(data["image_list"]).read_text().splitlines() if n]
    if len(frames) < 3:
        raise ValueError("v1 SfM requires >=3 frames")
    semantic_mode = matching.get("semantic_features", "off")
    semantic_provider = None
    if matching.get("semantic_checkpoint"):
        from underwater_vision.semantic.infer import SemanticPredictor

        semantic_provider = SemanticPredictor(matching["semantic_checkpoint"])
    elif matching.get("semantic_teacher_cache"):
        from underwater_vision.semantic.infer import TeacherCache

        semantic_provider = TeacherCache(matching["semantic_teacher_cache"])
    if (semantic_mode != "off" or matching.get("dynamic_mask")) and semantic_provider is None:
        raise ValueError("Semantic features/masking require a standalone semantic provider")
    matchability_provider = None
    from underwater_vision.visibility.source import validate_source_images

    visibility_cfg = validate_source_images(
        cfg.get("visibility", {}), [f.name for f in frames], matching.get("evaluation_mode", True)
    )
    matchability_checkpoint = matching.get("matchability_checkpoint") or (
        visibility_cfg.get("checkpoint") if visibility_cfg.get("source") == "matchability" else None
    )
    if matchability_checkpoint:
        from underwater_vision.matchability.infer import MatchabilityPredictor

        matchability_provider = MatchabilityPredictor(matchability_checkpoint)
    matchability = matching.get("matchability_features", "off") == "on"
    if matchability and matchability_provider is None:
        raise ValueError("Matchability features require a standalone map provider")
    predictor = MatchPredictor(
        matching["filter_checkpoint"],
        semantic_mode,
        semantic_provider.checkpoint_hash if semantic_provider else None,
        matchability,
        matchability_provider.checkpoint_hash if matchability_provider else None,
    )
    if (
        predictor.metadata.get("output_interpretation")
        == "tight_geometry_ranking_score_not_calibrated_probability"
        and matching.get("selection_policy") != "conservative_budget"
    ):
        raise ValueError("Tight-geometry ranker scores require conservative_budget selection")
    if (
        predictor.metadata["backbone"] != cfg["model"]["backbone"]
        or predictor.metadata["revision"] != cfg["model"]["revision"]
    ):
        raise ValueError("Filter backbone/revision mismatch")
    trained = set(predictor.metadata["training_images"] + predictor.metadata["validation_images"])
    evaluation_mode = matching.get("evaluation_mode", True)
    if evaluation_mode and matchability_provider:
        from underwater_vision.matchability.protocol import check_evaluation

        check_evaluation(
            matchability_provider.metadata,
            [f.name for f in frames],
            matchability_provider.metadata["spatial_split"],
        )
    if evaluation_mode and any(f.name in trained for f in frames):
        raise ValueError("Filter training/validation image leakage")
    if evaluation_mode and semantic_provider and hasattr(semantic_provider, "metadata"):
        used = set(
            semantic_provider.metadata.get("training_images", [])
            + semantic_provider.metadata.get("validation_images", [])
        )
        if used & {f.name for f in frames}:
            raise ValueError("Segmentation training/validation image leakage")
    for f in frames:
        if evaluation_mode and (
            f.reference_c2w is not None
            and assign_spatial(
                [f.reference_c2w[:3, 3]], predictor.metadata["spatial_split"]["specification"]
            )[0]
            != "test"
        ):
            raise ValueError("v1 evaluation frame is outside the locked spatial test block")
    neural = DinoExtractor(
        cfg["model"]["backbone"], cfg["model"]["device"], cfg["model"]["revision"]
    )
    if (
        data["enhance"]
        or data["max_size"] != predictor.metadata["image_size"]
        or cfg["model"]["max_features"] != predictor.metadata["max_features"]
    ):
        raise ValueError(
            "v1 filter requires its training preprocessing (960px, 4096 SIFT, no CLAHE)"
        )
    endpoint_only = (
        matching.get("selection_policy") == "conservative_budget"
        and matching.get("ranking_source") == "matchability"
        and not matchability
    )
    if cfg["model"].get("contrast_threshold", 0.04) != 0.04 and not endpoint_only:
        raise ValueError("Existing v1 match filters require their training SIFT contrast threshold")
    features, manifest = {}, []
    image_dir = output / "images"
    for f in frames:
        feat = feature_record(
            f,
            neural,
            data["max_size"],
            cfg["model"]["max_features"],
            semantic_provider,
            matchability_provider,
            contrast_threshold=cfg["model"].get("contrast_threshold", 0.04),
        )
        if visibility_cfg.get("source"):
            from underwater_vision.visibility.source import apply_source

            feat = apply_source(feat, visibility_cfg["source"], visibility_cfg.get("checkpoint"))
        if matching.get("keypoint_budget"):
            from underwater_vision.matchability.infer import select_keypoints

            source = matching.get("keypoint_selection", "matchability")
            if source == "response":
                import cv2

                rgb, _ = resize_rgb(read_rgb(f.path), data["max_size"])
                keypoints, _ = cv2.SIFT_create(
                    nfeatures=cfg["model"]["max_features"]
                ).detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
                if not np.array_equal(
                    feat["xy"], np.asarray([k.pt for k in keypoints], np.float32)
                ):
                    raise ValueError("Response/keypoint alignment mismatch")
                feat["response"] = np.asarray([k.response for k in keypoints], np.float32)
            feat = select_keypoints(feat, matching["keypoint_budget"], source, cfg["seed"])
        if matching.get("dynamic_mask"):
            from underwater_vision.semantic.masks import mask_keypoints

            threshold = matching.get("dynamic_threshold")
            if threshold is None:
                threshold = getattr(semantic_provider, "metadata", {}).get("dynamic_threshold")
            if threshold is None:
                raise ValueError("Masking threshold must be chosen on validation")
            feat = mask_keypoints(feat, threshold)
        height, width = map(int, feat["shape"])
        K, distortion = camera_parameters(width, height, data["calibration"])
        feat["geometry_xy"] = undistort(feat["xy"], K, distortion)
        features[f.name] = feat
        image, _ = resize_rgb(read_rgb(f.path), data["max_size"], data["enhance"])
        write_rgb(image_dir / f.name, image)
        manifest.append({"name": f.name, "path": str(f.path.resolve())})
    write_json(output / "input_manifest.json", manifest)
    write_json(
        output
        / (
            "filter_provenance_v3.json"
            if output.name.endswith("_v3")
            else "filter_provenance_v1.json"
        ),
        {
            "architecture": predictor.metadata["architecture"],
            "sha256": hashlib.sha256(Path(matching["filter_checkpoint"]).read_bytes()).hexdigest(),
            "threshold": predictor.metadata["threshold"],
            "selection_policy": matching.get("selection_policy", "threshold"),
            "ranking_source": matching.get("ranking_source", "filter"),
            "score_interpretation": (
                "match_ranking_score_not_calibrated_probability"
                if matching.get("selection_policy") == "conservative_budget"
                else predictor.metadata.get("output_interpretation", "match_probability")
            ),
            "match_retention": matching.get("match_retention", 0.9),
            "minimum_matches": matching.get("minimum_matches", 128),
            "label_source": predictor.metadata["labels"],
            "evaluation_mode": evaluation_mode,
            "training_or_validation_overlap": len({f.name for f in frames} & trained),
            "semantic_features": semantic_mode,
            "segmentation_hash": semantic_provider.checkpoint_hash if semantic_provider else None,
            "dynamic_mask": matching.get("dynamic_mask", False),
            "matchability_features": matchability,
            "matchability_hash": (
                matchability_provider.checkpoint_hash if matchability_provider else None
            ),
            "keypoint_budget": matching.get("keypoint_budget"),
            "visibility_source": cfg.get("visibility", {}).get("source", "handcrafted"),
            "visibility_checkpoint_sha256": visibility_cfg.get("checkpoint_sha256"),
        },
    )
    if cfg["stage"] == "features":
        return {"images": len(frames), "features": sum(len(f["xy"]) for f in features.values())}
    pairs = [
        (i, j)
        for i in range(len(frames))
        for j in range(i + 1, min(len(frames), i + matching["window"] + 1))
    ]
    if matching.get("pairing", "window") != "window":
        with np.load(matching["retrieval_file"]) as z:
            index = {n: i for i, n in enumerate(z["image_names"].tolist())}
            descriptors = z["descriptors"][[index[f.name] for f in frames]]
        pairs, _ = retrieve_pairs(
            descriptors,
            matching["retrieval_top_k"],
            matching["window"] if matching["pairing"] == "hybrid" else 0,
        )
    results = []
    (output / "matches").mkdir()
    selected = 0
    for k, (i, j) in enumerate(pairs):
        a, b = frames[i].name, frames[j].name
        fa, fb = features[a], features[b]
        height, width = map(int, fa["shape"])
        ids, x = candidates(fa, fb, width, height, semantic_mode, matchability)
        probability = np.ones(len(x), np.float32) if endpoint_only else predictor(x)
        policy = matching.get("selection_policy", "threshold")
        if policy == "conservative_budget":
            from underwater_vision.matching.budget import budget_indices, endpoint_ranking

            source = matching.get("ranking_source", "filter")
            ranking = probability
            if source in {"matchability", "product"}:
                if "matchability" not in fa or "matchability" not in fb:
                    raise ValueError("Endpoint ranking requires a matchability provider")
                ranking = endpoint_ranking(
                    ids,
                    fa["matchability"],
                    fb["matchability"],
                    probability if source == "product" else None,
                )
            elif source != "filter":
                raise ValueError("Unknown learned ranking source")

            accepted = budget_indices(
                x,
                ranking,
                matching.get("match_retention", 0.9),
                matching.get("minimum_matches", 128),
            )
            probability = ranking
        elif policy == "threshold":
            accepted = unique_matches(ids, probability, predictor.metadata["threshold"])
        else:
            raise ValueError("Unknown learned match selection policy")
        record = {
            "a": a,
            "b": b,
            "pairs": ids[accepted],
            "score": probability[accepted],
            "inliers": np.ones(len(accepted), bool),
        }
        results.append(record)
        atomic_npz(output / "matches" / f"{k:06d}.npz", **record)
        selected += len(accepted)
    metrics = {
        "images": len(frames),
        "pairs": len(pairs),
        "selected_matches": selected,
        "filter": predictor.metadata["architecture"],
        "pairing": matching.get("pairing", "window"),
        "geometric_backend": "standard independent COLMAP verification; no custom weighted RANSAC",
    }
    if cfg["stage"] == "matches":
        return metrics
    camera = make_camera(width, height, data["calibration"])
    database = output / "database.db"
    ids = create_database(database, frames, features, camera)
    insert_matches(database, ids, results)
    pair_file = output / "pairs.txt"
    pair_file.write_text("\n".join(r["a"] + " " + r["b"] for r in results if len(r["pairs"]) >= 15))
    settings = dict(cfg["reconstruction"], image_count=len(frames))
    model, sparse_path = reconstruct(
        database, image_dir, output / "sparse", pair_file, settings, cfg["seed"]
    )
    poses, statistics = export_reconstruction(model, output)
    metrics.update(statistics)
    reference = {f.name: f.reference_c2w for f in frames if f.reference_c2w is not None}
    metrics["trajectory"] = trajectory_metrics(poses, reference)
    write_json(output / "reference_poses.json", {n: p.tolist() for n, p in reference.items()})
    from underwater_vision.pointcloud.confidence import sparse_confidence
    from underwater_vision.visualization.figures import reconstruction_figure, trajectory_figure

    sparse_confidence(model, features, output, settings)
    if settings.get("semantic_voting"):
        if semantic_provider is None:
            raise ValueError("Semantic voting requires per-view class probabilities")
        from underwater_vision.semantic.voting import export_semantic_cloud

        export_semantic_cloud(model, features, output)
    if settings.get("semantic_confidence_checkpoint"):
        from underwater_vision.semantic.experiments import export_semantic_confidence

        export_semantic_confidence(
            model,
            features,
            output,
            settings["semantic_confidence_checkpoint"],
            semantic_provider.checkpoint_hash,
        )
    if settings.get("learned_confidence_checkpoint"):
        from underwater_vision.pointcloud.learned_confidence import export_learned_confidence

        export_learned_confidence(
            model, features, output, settings["learned_confidence_checkpoint"]
        )
    if settings.get("matchability_confidence_checkpoint"):
        if matchability_provider is None:
            raise ValueError("Point matchability prediction requires a map provider")
        from underwater_vision.matchability.confidence import export_confidence

        export_confidence(
            model,
            features,
            output,
            settings["matchability_confidence_checkpoint"],
            matchability_provider.checkpoint_hash,
        )
    reconstruction_figure(model, output / "figures/reconstruction.png")
    trajectory_figure(poses, reference, output / "figures/trajectory.png")
    metrics["dense_status"] = "not_requested"
    if settings["dense"]:
        from underwater_vision.pointcloud.dense_confidence import dense_confidence
        from underwater_vision.reconstruction.colmap.mvs import run_mvs

        run_mvs(
            sparse_path,
            image_dir,
            output / "dense",
            settings["colmap_executable"],
            settings["dense_max_size"],
            settings["mesh"],
        )
        metrics.update(dense_confidence(output / "dense"), dense_status="completed")
    return metrics
