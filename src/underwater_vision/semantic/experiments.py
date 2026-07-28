"""Optional learned adapters. Standalone trainers share array APIs with the pipeline."""

import copy
import json
from pathlib import Path

import numpy as np
import torch

from underwater_vision.evaluation.uncertainty import (
    binary_metrics,
    choose_threshold,
    uncertainty_metrics,
)
from underwater_vision.matching.learned_filter import MatchFilter
from underwater_vision.semantic.infer import SemanticPredictor
from underwater_vision.semantic.masks import feature_names
from underwater_vision.semantic.protocol import new_run, require_gate
from underwater_vision.utils.io import atomic_npz, write_json


def pair_class_arrays(record, lookup, semantic, base_features):
    """Add probabilities on exactly the v1 proposal/keypoint population."""
    from underwater_vision.features.dinov2 import DinoExtractor
    from underwater_vision.features.store import feature_record
    from underwater_vision.semantic.masks import match_class_features

    for endpoint in ["a", "b"]:
        name = str(record[endpoint])
        if name not in base_features:
            if "_neural" not in base_features:
                base_features["_neural"] = DinoExtractor(
                    "dinov2_vitb14", "auto", "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
                )
            base_features[name] = feature_record(
                lookup[name], base_features["_neural"], semantic_provider=semantic
            )
    a, b = base_features[str(record["a"])], base_features[str(record["b"])]
    return a, b, match_class_features(a, b, record["pairs"])


def train_semantic_matches(args):
    require_gate("outputs/semantic_audit_v2/gate_review.json")
    if args.semantic_features != "off" and not args.semantic_checkpoint:
        raise ValueError("Class feature training requires a segmentation checkpoint")
    from underwater_vision.data.dataset_loader import MermaidDataset
    from underwater_vision.semantic.masks import match_class_features

    root = new_run(args.output)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    semantic = SemanticPredictor(args.semantic_checkpoint) if args.semantic_checkpoint else None
    lookup = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
    manifest = (
        json.loads((args.base_matches / "spatial_manifest_v1.json").read_text())
        if args.spatial_manifest is None
        else json.loads(args.spatial_manifest.read_text())
    )
    if args.spatial_manifest is not None:
        build_reference_pairs(manifest, root / "reference_pairs_v2")
        source = root / "reference_pairs_v2"
    else:
        source = args.base_matches / "pairs"
    write_json(root / "spatial_manifest_v2.json", manifest)
    base_features = {}
    records = {s: [] for s in ["train", "validation", "test"]}
    (root / "pairs").mkdir()
    for group in records:
        for file in sorted(source.glob(group + "_*.npz")):
            with np.load(file) as z:
                record = {k: z[k] for k in z.files if k not in {"tensor"}}
            if semantic:
                a, b, _ = pair_class_arrays(record, lookup, semantic, base_features)
                if args.semantic_features != "off":
                    record["x"] = np.c_[
                        record["x"][:, :12],
                        match_class_features(a, b, record["pairs"], args.semantic_features),
                    ].astype(np.float32)
                record["class_region"] = (
                    a["class_probabilities"][record["pairs"][:, 0]]
                    + b["class_probabilities"][record["pairs"][:, 1]]
                ).argmax(1)
                record["dynamic_probability"] = np.maximum(
                    a["class_probabilities"][record["pairs"][:, 0], 5],
                    b["class_probabilities"][record["pairs"][:, 1], 5],
                )
            atomic_npz(root / "pairs" / file.name, **record)
            records[group].append(record)
    if not all(records.values()):
        raise ValueError("Need reference pair records in every spatial block")
    if semantic:
        allowed = set(semantic.metadata["training_images"])
        if not set(manifest["selected"]["train"]).issubset(allowed):
            raise ValueError("Segmentation checkpoint uses a different spatial training block")
        prohibited = set(
            semantic.metadata["training_images"] + semantic.metadata["validation_images"]
        )
        if prohibited & set(manifest["selected"]["test"]):
            raise ValueError("Segmentation test leakage")
    base_features.clear()
    semantic = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_x = np.concatenate([r["x"][r["labels"] >= 0] for r in records["train"]])
    mean, std = train_x.mean(0), np.maximum(train_x.std(0), 1e-3)
    for rows in records.values():
        for row in rows:
            row["tensor"] = torch.from_numpy((row["x"] - mean) / std)
    vy = np.concatenate([r["labels"] for r in records["validation"]])
    result = {
        "models": {},
        "seed": args.seed,
        "semantic_features": args.semantic_features,
        "spatial_regions": 1,
        "criteria": "Final Q3 needs three fold-specific retrained models and paired CIs",
    }
    from underwater_vision.semantic.protocol import digest_file

    segmentation_hash = digest_file(args.semantic_checkpoint) if args.semantic_checkpoint else None
    for architecture in ["mlp", "context"]:
        torch.manual_seed(args.seed)
        model = MatchFilter(architecture, channels=len(mean))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        best = -1
        state = None
        history = []
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for i in np.random.default_rng(args.seed + epoch).permutation(len(records["train"])):
                r = records["train"][i]
                valid = r["labels"] >= 0
                if not valid.any():
                    continue
                logits = model(r["tensor"])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[valid], torch.from_numpy(r["labels"][valid].astype(np.float32))
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                logits = np.concatenate([model(r["tensor"]).numpy() for r in records["validation"]])
            ap = binary_metrics(vy, torch.sigmoid(torch.from_numpy(logits)).numpy())[
                "average_precision"
            ]
            history.append(
                {"epoch": epoch + 1, "train_bce": float(np.mean(losses)), "validation_ap": ap}
            )
            if ap > best:
                best, state = ap, copy.deepcopy(model.state_dict())
        model.load_state_dict(state)
        with torch.no_grad():
            logits = np.concatenate([model(r["tensor"]).numpy() for r in records["validation"]])
        known = vy >= 0
        temperatures = np.linspace(0.5, 3, 51)
        nll = [
            torch.nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(logits[known]) / t, torch.from_numpy(vy[known].astype(np.float32))
            ).item()
            for t in temperatures
        ]
        temperature = float(temperatures[np.argmin(nll)])
        probability = torch.sigmoid(torch.from_numpy(logits) / temperature).numpy()
        checkpoint = {
            "state_dict": state,
            "architecture": architecture,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "temperature": temperature,
            "threshold": choose_threshold(vy, probability),
            "feature_spec": (
                "match_v1_12"
                if args.semantic_features == "off"
                else f"match_class_v2_{args.semantic_features}"
            ),
            "feature_names": feature_names(args.semantic_features),
            "segmentation_hash": segmentation_hash,
            "semantic_features": args.semantic_features,
            "training_images": manifest["selected"]["train"],
            "validation_images": manifest["selected"]["validation"],
            "spatial_split": manifest["split"],
            "backbone": "dinov2_vitb14",
            "revision": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
            "image_size": 960,
            "max_features": 4096,
            "labels": "reference_F_1.5_positive_4_negative_ambiguous_excluded",
            "seed": args.seed,
        }
        path = root / (architecture + "_v2.pt")
        torch.save(checkpoint, path)
        result["models"][architecture] = {
            "history": history,
            "validation": binary_metrics(vy, probability, checkpoint["threshold"]),
            "checkpoint_sha256": digest_file(path),
        }
    result["selected_architecture"] = max(
        result["models"], key=lambda a: result["models"][a]["validation"]["average_precision"]
    )
    write_json(root / "training_v2.json", result)


def build_reference_pairs(manifest, output):
    """Reference label generation for alternate spatial folds, independent of RANSAC."""
    import yaml

    from underwater_vision.data.dataset_loader import MermaidDataset
    from underwater_vision.features.dinov2 import DinoExtractor
    from underwater_vision.features.store import feature_record
    from underwater_vision.geometry.reference import camera_parameters, reference_labels
    from underwater_vision.matching.learned_filter import candidates

    lookup = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
    calibration = yaml.safe_load(Path("configs/data/mermaid.yaml").read_text())["calibration"]
    neural = DinoExtractor("dinov2_vitb14", "auto", "7764ea0f912e53c92e82eb78a2a1631e92725fc8")
    output.mkdir()
    features = {}
    for group, names in manifest["selected"].items():
        if group not in {"train", "validation", "test"}:
            continue
        centers = np.stack([lookup[n].reference_c2w[:3, 3] for n in names])
        distance = np.linalg.norm(centers[:, None] - centers[None], axis=2)
        np.fill_diagonal(distance, np.inf)
        pairs = set()
        for i in range(len(names)):
            for j in np.argsort(distance[i])[:2]:
                pairs.add(tuple(sorted((names[i], names[j]))))
            pairs.add(tuple(sorted((names[i], names[(i + len(names) // 2) % len(names)]))))
        for i, (a, b) in enumerate(sorted(pairs)):
            for name in [a, b]:
                if name not in features:
                    features[name] = feature_record(lookup[name], neural)
            fa, fb = features[a], features[b]
            h, w = fa["shape"]
            ids, x = candidates(fa, fb, w, h)
            K, distortion = camera_parameters(int(w), int(h), calibration)
            labels, error, F = reference_labels(
                fa["xy"][ids[:, 0]],
                fb["xy"][ids[:, 1]],
                lookup[a].reference_c2w,
                lookup[b].reference_c2w,
                K,
                distortion,
            )
            atomic_npz(
                output / f"{group}_{i:04d}.npz",
                a=np.asarray(a),
                b=np.asarray(b),
                pairs=ids,
                x=x,
                labels=labels,
                reference_error=error,
                reference_F=F,
            )


def train_semantic_confidence(args):
    require_gate("outputs/semantic_audit_v2/gate_review.json")
    if not args.semantic_checkpoint or not args.conformal:
        raise ValueError("v2 confidence requires semantic checkpoint and --conformal")
    from underwater_vision.semantic.confidence import (
        RobustErrorRegressor,
        SemanticErrorPredictor,
        conformal_quantile,
        robust_loss,
    )

    root = new_run(args.output)
    arrays = build_confidence_dataset(args, root)
    train, validation, test = (arrays[s] for s in ["train", "validation", "test"])
    if any(len(set(r["track_group"].tolist())) < 30 for r in arrays.values()):
        raise ValueError("Need >=30 independent tracks in each block")
    mean, std = train["x"].mean(0), np.maximum(train["x"].std(0), 1e-4)
    tx = torch.from_numpy((train["x"] - mean) / std)
    ty = torch.from_numpy(np.log1p(train["error"]).astype(np.float32))
    vx = torch.from_numpy((validation["x"] - mean) / std)
    vy = torch.from_numpy(np.log1p(validation["error"]).astype(np.float32))
    validation_groups = np.unique(validation["track_group"])
    validation_groups = np.random.default_rng(args.seed).permutation(validation_groups)
    selection_groups = validation_groups[: len(validation_groups) // 2]
    calibration_groups = validation_groups[len(validation_groups) // 2 :]
    selection = torch.from_numpy(np.isin(validation["track_group"], selection_groups))
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model = RobustErrorRegressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    best = float("inf")
    state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(tx), generator=torch.Generator().manual_seed(args.seed + epoch))
        for ids in order.split(512):
            mu, scale = model(tx[ids])
            loss = robust_loss(mu, scale, ty[ids], args.robust_loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            mu, scale = model(vx)
            loss = float(
                robust_loss(mu[selection], scale[selection], vy[selection], args.robust_loss)
            )
        history.append({"epoch": epoch + 1, "validation_robust_loss": loss})
        if loss < best:
            best, state = loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    with torch.no_grad():
        mu, _ = model(vx)
    # Aggregate repeated holdouts per track before conformal calibration.
    groups = calibration_groups
    errors = [np.median(validation["error"][validation["track_group"] == g]) for g in groups]
    prediction = [np.median(mu.numpy()[validation["track_group"] == g]) for g in groups]
    radius = conformal_quantile(errors, prediction)
    from underwater_vision.semantic.protocol import digest_file

    hash_ = digest_file(args.semantic_checkpoint)
    saved = {
        "state_dict": state,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "conformal_radius": radius,
        "feature_spec": "point_class_v2_17",
        "segmentation_hash": hash_,
        "seed": args.seed,
        "robust_loss": args.robust_loss,
        "training_target": "median multiple held-out reference reprojections",
        "spatial_manifest": json.loads((root / "spatial_manifest_v2.json").read_text()),
    }
    path = root / "error_model_v2.pt"
    torch.save(saved, path)
    predicted = SemanticErrorPredictor(path, hash_)(test["x"])
    # Report independent-track metrics using median repeated holdouts.
    ids = np.unique(test["track_group"])

    def aggregate(value):
        return np.asarray([np.median(value[test["track_group"] == g]) for g in ids])

    error = aggregate(test["error"])
    models = {
        "learned_robust": uncertainty_metrics(error, aggregate(predicted["predicted_error"])),
        "heuristic": uncertainty_metrics(error, aggregate(1 - test["heuristic"])),
        "view_count": uncertainty_metrics(error, aggregate(1 / np.expm1(test["x"][:, 0]))),
    }
    from underwater_vision.pointcloud.learned_confidence import ErrorPredictor

    baseline_path = train_v1_confidence_baseline(train, validation, selection.numpy(), root, args)
    v1 = ErrorPredictor(baseline_path)(test["x"][:, :8])
    models["v1_learned"] = uncertainty_metrics(error, aggregate(v1["predicted_error"]))
    low, high = aggregate(predicted["lower90"]), aggregate(predicted["upper90"])
    models["learned_robust"]["interval90_coverage"] = float(
        np.mean((error >= low) & (error <= high))
    )
    from underwater_vision.evaluation.segmentation import paired_bootstrap

    comparisons = {}
    for baseline in ["heuristic", "v1_learned"]:
        uncertainty = (
            aggregate(1 - test["heuristic"])
            if baseline == "heuristic"
            else aggregate(v1["predicted_error"])
        )
        triples = list(
            zip(error, aggregate(predicted["predicted_error"]), uncertainty, strict=True)
        )
        comparisons[baseline] = paired_bootstrap(
            triples,
            lambda rows: uncertainty_metrics(np.asarray(rows)[:, 0], np.asarray(rows)[:, 1])["ause"]
            - uncertainty_metrics(np.asarray(rows)[:, 0], np.asarray(rows)[:, 2])["ause"],
        )
    per_class = {}
    classes = np.asarray(
        [np.bincount(test["x"][test["track_group"] == g, 8:14].argmax(1)).argmax() for g in ids]
    )
    for c in range(6):
        keep = classes == c
        per_class[str(c)] = (
            uncertainty_metrics(error[keep], aggregate(predicted["predicted_error"])[keep])
            if keep.sum() >= 3
            else {"tracks": int(keep.sum()), "status": "insufficient"}
        )
    atomic_npz(
        root / "predictions_v2.npz",
        **predicted,
        observed_error=test["error"],
        track_group=test["track_group"],
    )
    write_json(
        root / "evaluation_v2.json",
        {
            "models": models,
            "ause_differences": comparisons,
            "per_class": per_class,
            "history": history,
            "independent_test_tracks": len(ids),
            "conformal_validation_tracks": len(groups),
            "model_selection_validation_tracks": len(selection_groups),
            "conformal_radius_log_error": radius,
            "caveat": "Coverage requires exchangeability; spatial shift remains. v1 8-feature Gaussian architecture is retrained on identical tracks in this fold; calibration tracks are excluded from model selection.",
        },
    )


def train_v1_confidence_baseline(train, validation, selection, root, args):
    """Retrain the v1 Gaussian/8-input baseline on this fold, avoiding old-model leakage."""
    from underwater_vision.pointcloud.learned_confidence import ErrorRegressor

    mean = train["x"][:, :8].mean(0)
    std = np.maximum(train["x"][:, :8].std(0), 1e-4)
    tx = torch.from_numpy((train["x"][:, :8] - mean) / std)
    ty = torch.from_numpy(np.log1p(train["error"]).astype(np.float32))
    vx = torch.from_numpy((validation["x"][selection, :8] - mean) / std)
    vy = torch.from_numpy(np.log1p(validation["error"][selection]).astype(np.float32))
    torch.manual_seed(args.seed)
    model = ErrorRegressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    best = float("inf")
    state = None
    for epoch in range(args.epochs):
        order = torch.randperm(len(tx), generator=torch.Generator().manual_seed(args.seed + epoch))
        for ids in order.split(512):
            mu, lv = model(tx[ids])
            loss = (0.5 * (lv + (ty[ids] - mu) ** 2 * torch.exp(-lv))).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        with torch.no_grad():
            mu, lv = model(vx)
            value = float((0.5 * (lv + (vy - mu) ** 2 * torch.exp(-lv))).mean())
        if value < best:
            best, state = value, copy.deepcopy(model.state_dict())
    path = root / "v1_baseline_retrained_v2.pt"
    torch.save(
        {
            "state_dict": state,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "validation_scale": 1.0,
            "seed": args.seed,
            "label": "same multi-held-out reference tracks as v2; Gaussian v1 architecture",
        },
        path,
    )
    return path


def build_confidence_dataset(args, root):
    # The original trainer's track constructor is shared, while labels/features differ.
    import pycolmap
    import yaml

    from underwater_vision.data.dataset_loader import MermaidDataset
    from underwater_vision.data.spatial import assign_spatial
    from underwater_vision.features.dinov2 import DinoExtractor
    from underwater_vision.features.store import feature_record
    from underwater_vision.geometry.reference import camera_parameters, undistort
    from underwater_vision.geometry.verification import verify
    from underwater_vision.semantic.confidence import multi_holdout_rows

    # Local pure union/find implementation avoids importing a scripts package.
    parent = {}
    members = {}

    def find(node):
        if node not in parent:
            parent[node] = node
            members[node] = {node}
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def join(a, b):
        a, b = find(a), find(b)
        if a != b and not ({n[0] for n in members[a]} & {n[0] for n in members[b]}):
            parent[b] = a
            members[a] |= members.pop(b)

    manifest_path = args.matches / "spatial_manifest_v2.json"
    if not manifest_path.exists():
        manifest_path = args.matches / "spatial_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text())
    write_json(root / "spatial_manifest_v2.json", manifest)
    lookup = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
    calibration = yaml.safe_load(Path("configs/data/mermaid.yaml").read_text())["calibration"]
    K, distortion = camera_parameters(960, 720, calibration)
    semantic = SemanticPredictor(args.semantic_checkpoint)
    prohibited = set(semantic.metadata["training_images"] + semantic.metadata["validation_images"])
    if prohibited & set(manifest["selected"]["test"]):
        raise ValueError("Confidence segmentation test leakage")
    neural = DinoExtractor("dinov2_vitb14", "auto", "7764ea0f912e53c92e82eb78a2a1631e92725fc8")
    features = {}

    def feature(name):
        if name not in features:
            f = feature_record(lookup[name], neural, semantic_provider=semantic)
            f["undistorted"] = undistort(f["xy"], K, distortion)
            features[name] = f
        return features[name]

    groups = {}
    for group in ["train", "validation"]:
        parent.clear()
        members.clear()
        for file in sorted((args.matches / "pairs").glob(group + "_*.npz")):
            with np.load(file) as z:
                a, b, pairs, x = str(z["a"]), str(z["b"]), z["pairs"], z["x"]
            pairs = pairs[(x[:, 1] < 0.8) & (x[:, 2] > 0.5)]
            _, mask, _ = verify(
                feature(a)["undistorted"][pairs[:, 0]],
                feature(b)["undistorted"][pairs[:, 1]],
                np.ones(len(pairs)),
                iterations=300,
            )
            for i, j in pairs[mask]:
                join((a, int(i)), (b, int(j)))
        groups[group] = [sorted(nodes) for nodes in members.values() if len(nodes) >= 3]
    model = pycolmap.Reconstruction(str(args.baseline / "sparse/0"))
    groups["test"] = [
        sorted((model.images[e.image_id].name, int(e.point2D_idx)) for e in point.track.elements)
        for point in model.points3D.values()
        if len(point.track.elements) >= 3
    ]
    result = {}
    for group, tracks in groups.items():
        rows = []
        for nodes in tracks:
            if any(n not in manifest["split"]["groups"][group] for n, _ in nodes):
                raise ValueError("Confidence track outside its spatial split")
            try:
                labels = multi_holdout_rows(
                    nodes,
                    [lookup[n].reference_c2w for n, _ in nodes],
                    [feature(n)["undistorted"][i] for n, i in nodes],
                    [feature(n)["semantic"][i] for n, i in nodes],
                    [feature(n)["quality"][i] for n, i in nodes],
                    [feature(n)["class_probabilities"][i] for n, i in nodes],
                    K,
                )
            except ValueError:
                continue
            rows.extend(
                r
                for r in labels
                if assign_spatial([r["xyz"]], manifest["split"]["specification"])[0] == group
            )
        if not rows:
            raise ValueError(f"No eligible {group} tracks")
        result[group] = {
            key: np.asarray([r[key] for r in rows])
            for key in ["x", "error", "heuristic", "xyz", "track_group"]
        }
        atomic_npz(root / (group + "_v2.npz"), **result[group])
    return result


def export_semantic_confidence(reconstruction, features, output, checkpoint, segmentation_hash):
    from underwater_vision.pointcloud.confidence import write_ply
    from underwater_vision.pointcloud.learned_confidence import track_features
    from underwater_vision.semantic.confidence import SemanticErrorPredictor, point_class_features

    rows, xyz, rgb = [], [], []
    for point in reconstruction.points3D.values():
        poses, xy, descriptors, quality, p = [], [], [], [], []
        for e in point.track.elements:
            image = reconstruction.images[e.image_id]
            f = features[image.name]
            pose = np.eye(4)
            pose[:3] = image.cam_from_world().inverse().matrix()
            poses.append(pose)
            xy.append(f["geometry_xy"][e.point2D_idx])
            descriptors.append(f["semantic"][e.point2D_idx])
            quality.append(f["quality"][e.point2D_idx])
            p.append(f["class_probabilities"][e.point2D_idx])
            K = reconstruction.cameras[image.camera_id].calibration_matrix().copy()
            K[:2, 2] -= 0.5
        base, _ = track_features(point.xyz, poses, xy, descriptors, quality, K)
        row = np.r_[base, point_class_features(p)]
        if np.isfinite(row).all():
            rows.append(row)
            xyz.append(point.xyz)
            rgb.append(point.color)
    predicted = SemanticErrorPredictor(checkpoint, segmentation_hash)(
        np.asarray(rows, np.float32).reshape(-1, 17)
    )
    write_ply(
        Path(output) / "sparse_semantic_confidence_v2.ply",
        np.asarray(xyz),
        np.asarray(rgb),
        predicted["confidence"],
    )
    atomic_npz(Path(output) / "sparse_semantic_confidence_v2.npz", xyz=np.asarray(xyz), **predicted)
    write_json(
        Path(output) / "semantic_confidence_v2.json",
        {
            "points": len(rows),
            "reference_poses_used_for_prediction": False,
            "segmentation_hash": segmentation_hash,
        },
    )
