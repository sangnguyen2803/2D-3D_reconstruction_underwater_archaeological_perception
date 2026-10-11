"""Reference holdout point errors, 10-input robust learner and conformal bands."""

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.integrate import trapezoid

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.evaluation.uncertainty import uncertainty_metrics
from underwater_vision.geometry.reference import (
    camera_parameters,
    project,
    triangulate_observations,
    undistort,
)
from underwater_vision.matchability.build import load_features
from underwater_vision.matchability.infer import MatchabilityPredictor, sample_probability
from underwater_vision.matchability.protocol import digest, new_run, read_json
from underwater_vision.matchability.tabular_baseline import DinoFeatureCache
from underwater_vision.pointcloud.learned_confidence import ErrorRegressor, track_features
from underwater_vision.semantic.confidence import (
    RobustErrorRegressor,
    conformal_quantile,
    robust_loss,
)
from underwater_vision.utils.io import atomic_npz, read_rgb, write_json


class Tracks:
    """One keypoint per view; conflicting track unions are rejected."""

    def __init__(self):
        self.parent, self.members = {}, {}

    def root(self, node):
        if node not in self.parent:
            self.parent[node], self.members[node] = node, {node}
        if self.parent[node] != node:
            self.parent[node] = self.root(self.parent[node])
        return self.parent[node]

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        if a == b or {n[0] for n in self.members[a]} & {n[0] for n in self.members[b]}:
            return
        if len(self.members[a]) < len(self.members[b]):
            a, b = b, a
        self.parent[b] = a
        self.members[a] |= self.members.pop(b)

    def tracks(self):
        return [sorted(v) for v in self.members.values() if len(v) >= 3]


def holdout_rows(nodes, poses, xy, descriptors, quality, matchability, K):
    """Three deterministic held-out sets; their xy/descriptors/maps never enter X."""
    group = int.from_bytes(hashlib.sha256(repr(nodes).encode()).digest()[:4], "little")
    order = np.random.default_rng(group).permutation(len(nodes))
    size = min(2, len(nodes) - 2)
    result = []
    for turn in range(min(3, len(nodes))):
        held = np.roll(order, turn)[:size]
        used = np.array([i for i in range(len(nodes)) if i not in held])
        try:
            point = triangulate_observations([poses[i] for i in used], np.asarray(xy)[used], K)
            x, heuristic = track_features(
                point,
                [poses[i] for i in used],
                np.asarray(xy)[used],
                np.asarray(descriptors)[used],
                np.asarray(quality)[used],
                K,
                np.asarray(matchability)[used],
            )
            error = float(
                np.median([np.linalg.norm(project(point, poses[i], K) - xy[i]) for i in held])
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(error) and np.isfinite(x).all():
            result.append(
                {
                    "x": x,
                    "error": error,
                    "heuristic": heuristic,
                    "group": group,
                    "heldout_indices": held,
                    "observed_indices": used,
                }
            )
    return result


def group_rows(data):
    order = np.argsort(data["group"], kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(data["group"][order])) + 1]
    return data["group"][order][starts], np.split(order, starts[1:])


def aggregate_prediction(data, values):
    groups, indices = group_rows(data)
    return groups, np.asarray([np.median(np.asarray(values)[ids]) for ids in indices])


class PointMatchabilityPredictor:
    def __init__(self, checkpoint, matchability_hash):
        self.metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            self.metadata.get("feature_spec") != "point_matchability_v3_10"
            or self.metadata.get("matchability_hash") != matchability_hash
        ):
            raise ValueError("Point feature specification or matchability hash mismatch")
        self.model = RobustErrorRegressor(10).eval()
        self.model.load_state_dict(self.metadata["state_dict"])

    @torch.inference_mode()
    def __call__(self, x):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != 10:
            raise ValueError("Point matchability predictor requires ten inputs")
        z = (x - np.asarray(self.metadata["mean"])) / np.asarray(self.metadata["std"])
        mu, _ = self.model(torch.as_tensor(z, dtype=torch.float32))
        mu = mu.numpy()
        radius = self.metadata["conformal_radius"]
        error = np.maximum(np.expm1(np.clip(mu, -20, 15)), 0)
        return {
            "predicted_error": error,
            "lower90": np.maximum(np.expm1(np.clip(mu - radius, -20, 15)), 0),
            "upper90": np.maximum(np.expm1(np.clip(mu + radius, -20, 15)), 0),
            "confidence": 1 / (1 + error),
        }


def point_bootstrap(error, a, b, repeats=500, seed=42):
    rng = np.random.default_rng(seed)
    difference = []
    for _ in range(repeats):
        ids = rng.integers(len(error), size=len(error))
        difference.append(fast_ause(error[ids], a[ids]) - fast_ause(error[ids], b[ids]))
    return {
        "estimate": uncertainty_metrics(error, a)["ause"] - uncertainty_metrics(error, b)["ause"],
        "ci95": np.quantile(difference, [0.025, 0.975]).tolist(),
        "tracks": len(error),
        "bootstrap_repeats": repeats,
        "unit": "track after median aggregation of correlated holdouts",
        "bootstrap_differences": difference,
    }


def fast_ause(error, predicted):
    """Vectorized exact tie-aware v1 AUSE; useful for 500 track resamples."""
    removed = np.linspace(0, 0.99, 100)
    retained = np.maximum(1, np.ceil(len(error) * (1 - removed)).astype(int))
    curves = []
    for values in [predicted, error]:
        order = np.argsort(values, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(values[order])) + 1]
        sizes = np.diff(np.r_[starts, len(error)])
        averages = np.add.reduceat(error[order], starts) / sizes
        tied = np.repeat(averages, sizes)
        curves.append(np.cumsum(tied)[retained - 1] / retained / max(error.mean(), 1e-12))
    return max(0.0, float(trapezoid(curves[0] - curves[1], removed)))


def train_confidence(args):
    root = new_run(args.output)
    manifest = read_json(args.matches / "spatial_manifest_v3.json")
    if digest(args.matchability_checkpoint) != manifest["matchability_hash"]:
        raise ValueError("Point learner and match filter must use the same matchability checkpoint")
    source = Path(manifest["dataset"])
    cache = DinoFeatureCache(source)
    predictor = MatchabilityPredictor(args.matchability_checkpoint)
    lookup = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
    calibration = yaml.safe_load(Path("configs/data/mermaid.yaml").read_text())["calibration"]
    datasets = {}
    all_pairs = read_json(source / "pairs_v3.json")
    selected = {s: ns for s, ns in manifest["split"]["groups"].items() if s != "excluded"}
    (root / "sampled_maps_v3").mkdir()
    write_json(
        root / "input_manifest_v3.json",
        {
            "selected": selected,
            "parent_filter_manifest_sha256": digest(args.matches / "spatial_manifest_v3.json"),
            "matchability_hash": manifest["matchability_hash"],
            "scope": "All non-excluded views in each fold; full hybrid label graph",
        },
    )
    for role, names in selected.items():
        features = {}
        for name in names:
            f = load_features(source, name)
            f["semantic"] = cache.descriptors(name, f["xy"])
            saved = args.matches / "maps" / (name + ".npz")
            if saved.exists():
                with np.load(saved) as z:
                    f["matchability"] = z["matchability"]
            else:
                values = predictor.predict(read_rgb(source / "images" / (name + ".png")))
                f["matchability"] = sample_probability(values, f["xy"])
            atomic_npz(root / "sampled_maps_v3" / (name + ".npz"), matchability=f["matchability"])
            h, w = map(int, f["shape"])
            K, distortion = camera_parameters(w, h, calibration)
            f["geometry_xy"] = undistort(f["xy"], K, distortion)
            features[name] = f
        tracks = Tracks()
        allowed = set(names)
        for index, info in enumerate(all_pairs):
            a, b = info["a"], info["b"]
            if a not in allowed or b not in allowed:
                continue
            with np.load(source / "pairs" / f"{index:06d}.npz") as z:
                forward, reverse = z["forward"], z["reverse"]
                error, backward_error = z["forward_error"], z["reverse_error"]
            if np.sum(error <= 1.5) + np.sum(backward_error <= 1.5) < 15:
                continue
            valid = (error <= 1.5) & (reverse[forward[:, 1], 0] == forward[:, 0])
            for i, j in forward[valid]:
                tracks.join((a, int(i)), (b, int(j)))
        rows = []
        for nodes in tracks.tracks():
            poses = [lookup[n].reference_c2w for n, _ in nodes]
            xy = [features[n]["geometry_xy"][idx] for n, idx in nodes]
            descriptors = [features[n]["semantic"][idx] for n, idx in nodes]
            quality = [features[n]["quality"][idx] for n, idx in nodes]
            matchability = [features[n]["matchability"][idx] for n, idx in nodes]
            covariates = []
            for n, idx in nodes:
                f = features[n]
                height, width = map(int, f["shape"])
                x_pixel, y_pixel = f["xy"][idx]
                covariates.append(
                    [
                        f["statistics"][idx, 2],
                        f["scales"][idx],
                        f["sharpness"][idx],
                        float(f["brightness"]),
                        np.hypot((x_pixel - width / 2) / width, (y_pixel - height / 2) / height),
                    ]
                )
            for row in holdout_rows(nodes, poses, xy, descriptors, quality, matchability, K):
                row["statistics"] = np.asarray(covariates)[row["observed_indices"]].mean(0)
                rows.append(row)
        if len(rows) < 20:
            raise ValueError(f"Too few {role} reference tracks for a meaningful point experiment")
        datasets[role] = {
            key: np.asarray([r[key] for r in rows])
            for key in ["x", "error", "heuristic", "group", "statistics"]
        }
        atomic_npz(root / (role + "_v3.npz"), **datasets[role])
        print(
            f"Point holdouts {role}: {len(rows)} rows from {len(np.unique(datasets[role]['group']))} tracks",
            flush=True,
        )
    train, validation, test = [datasets[s] for s in ["train", "validation", "test"]]
    cache = predictor = features = None
    torch.cuda.empty_cache()
    # Entire validation tracks are assigned either selection OR conformal fitting.
    val_groups, _ = group_rows(validation)
    selection_groups = np.random.default_rng(args.seed).permutation(val_groups)[::2]
    selection = np.isin(validation["group"], selection_groups)
    calibration_mask = ~selection
    if len(np.unique(validation["group"][calibration_mask])) < 10:
        raise ValueError("Need >=10 independent validation calibration tracks")
    report = {
        "models": {},
        "seed": args.seed,
        "protocol": manifest["split"]["name"],
        "matchability_hash": manifest["matchability_hash"],
        "counts": {
            s: {"rows": len(d["x"]), "tracks": len(np.unique(d["group"]))}
            for s, d in datasets.items()
        },
        "target": "Reference-triangulated points, median held-out reprojection; matching is a RootSIFT/reference-compatibility selected proxy",
        "conformal": "Separate validation tracks from checkpoint selection; exchangeability not guaranteed across spatial regions",
    }
    torch.set_num_threads(4)
    _, test_error = aggregate_prediction(test, test["error"])
    predictions = {}
    for name, channels in [("v1_gaussian_8", 8), ("robust_8", 8), ("matchability_10", 10)]:
        torch.manual_seed(args.seed)
        model = ErrorRegressor() if name == "v1_gaussian_8" else RobustErrorRegressor(channels)
        mean, std = train["x"][:, :channels].mean(0), np.maximum(
            train["x"][:, :channels].std(0), 1e-3
        )
        tx = torch.from_numpy(((train["x"][:, :channels] - mean) / std).astype(np.float32))
        target = torch.from_numpy(np.log1p(train["error"]).astype(np.float32))
        vx = torch.from_numpy(((validation["x"][:, :channels] - mean) / std).astype(np.float32))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
        best, state = float("inf"), None
        for _epoch in range(args.epochs):
            model.train()
            for idx in torch.randperm(len(tx)).split(512):
                mu, log_scale = model(tx[idx])
                loss = (
                    (0.5 * (log_scale + (mu - target[idx]).square() / log_scale.exp())).mean()
                    if name == "v1_gaussian_8"
                    else robust_loss(mu, log_scale, target[idx], args.robust_loss)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                mu, _ = model(vx)
                score = torch.nn.functional.huber_loss(
                    mu[selection],
                    torch.as_tensor(np.log1p(validation["error"][selection]), dtype=torch.float32),
                ).item()
            if score < best:
                best, state = score, copy.deepcopy(model.state_dict())
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            vmu, _ = model(vx)
            tmu, tscale = model(
                torch.from_numpy(((test["x"][:, :channels] - mean) / std).astype(np.float32))
            )
        calibration_data = {k: v[calibration_mask] for k, v in validation.items()}
        _, cal_mu = aggregate_prediction(calibration_data, vmu.numpy()[calibration_mask])
        _, cal_error = aggregate_prediction(calibration_data, calibration_data["error"])
        radius = conformal_quantile(cal_error, cal_mu)
        _, mu = aggregate_prediction(test, tmu.numpy())
        predicted = np.maximum(np.expm1(np.clip(mu, -20, 15)), 0)
        if name == "v1_gaussian_8":
            _, predicted = aggregate_prediction(
                test,
                np.maximum(np.expm1(np.clip(tmu.numpy() + np.exp(tscale.numpy()) / 2, -20, 15)), 0),
            )
        predictions[name] = predicted
        metric = uncertainty_metrics(test_error, predicted)
        metric["interval_coverage90"] = float(
            np.mean((np.log1p(test_error) >= mu - radius) & (np.log1p(test_error) <= mu + radius))
        )
        metric["conformal_radius_log1p"] = radius
        report["models"][name] = metric
        checkpoint = {
            "state_dict": state,
            "feature_spec": (
                "point_matchability_v3_10" if channels == 10 else f"point_baseline_v3_{name}"
            ),
            "matchability_hash": manifest["matchability_hash"] if channels == 10 else None,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "conformal_radius": radius,
            "training_images": selected["train"],
            "validation_images": selected["validation"],
            "spatial_split": manifest["split"],
            "seed": args.seed,
            "loss": args.robust_loss,
            "epochs": args.epochs,
            "dataset_manifest_sha256": digest(source / "manifest_v3.json"),
            "criteria_sha256": digest("configs/matchability/criteria_v3.yaml"),
            "selection_rule": "validation Huber log-error, independent calibration track subset",
        }
        torch.save(checkpoint, root / (name + "_v3.pt"))
    _, heuristic = aggregate_prediction(test, test["heuristic"])
    _, support = aggregate_prediction(test, test["x"][:, 0])
    predictions["heuristic"], predictions["view_count"] = 1 - heuristic, -support
    for name in ["heuristic", "view_count"]:
        report["models"][name] = uncertainty_metrics(test_error, predictions[name])
    report["paired_ause"] = {
        name: point_bootstrap(
            test_error, predictions["matchability_10"], predictions[name], seed=42
        )
        for name in ["v1_gaussian_8", "robust_8", "heuristic"]
    }
    atomic_npz(root / "test_predictions_v3.npz", error=test_error, **predictions)
    binned = {}
    for column, name in enumerate(
        ["local_contrast", "scale", "sharpness", "brightness", "frame_radius"]
    ):
        _, covariate = aggregate_prediction(test, test["statistics"][:, column])
        edges = np.quantile(covariate, [0.25, 0.5, 0.75])
        bins = np.digitize(covariate, edges)
        binned[name] = {
            "edges": edges.tolist(),
            "models": {
                model_name: [
                    (
                        uncertainty_metrics(test_error[bins == b], value[bins == b])
                        if np.sum(bins == b) >= 3
                        else {"status": "too_few_tracks"}
                    )
                    for b in range(4)
                ]
                for model_name, value in predictions.items()
            },
        }
    write_json(root / "binned_confidence_v3.json", binned)
    write_json(root / "evaluation_v3.json", report)
    return root


def export_confidence(reconstruction, features, output, checkpoint, matchability_hash):
    """Deployment consumes estimated COLMAP cameras; no reference target enters X."""
    from underwater_vision.pointcloud.confidence import write_ply

    predictor = PointMatchabilityPredictor(checkpoint, matchability_hash)
    rows, xyz, rgb, ids = [], [], [], []
    for pid, point in reconstruction.points3D.items():
        poses, xy, descriptors, quality, matchability = [], [], [], [], []
        for element in point.track.elements:
            image = reconstruction.images[element.image_id]
            feature = features[image.name]
            pose = np.eye(4)
            pose[:3] = image.cam_from_world().inverse().matrix()
            poses.append(pose)
            idx = element.point2D_idx
            xy.append(feature["geometry_xy"][idx])
            descriptors.append(feature["semantic"][idx])
            quality.append(feature["quality"][idx])
            matchability.append(feature["matchability"][idx])
            K = reconstruction.cameras[image.camera_id].calibration_matrix().copy()
            K[:2, 2] -= 0.5
        x, _ = track_features(point.xyz, poses, xy, descriptors, quality, K, matchability)
        if not np.isfinite(x).all():
            continue
        rows.append(x)
        xyz.append(point.xyz)
        rgb.append(point.color)
        ids.append(int(pid))
    prediction = predictor(np.asarray(rows, np.float32).reshape(-1, 10))
    write_ply(
        Path(output) / "sparse_matchability_confidence_v3.ply",
        np.asarray(xyz),
        np.asarray(rgb),
        prediction["confidence"],
    )
    atomic_npz(
        Path(output) / "sparse_matchability_confidence_v3.npz",
        xyz=np.asarray(xyz),
        rgb=np.asarray(rgb),
        point_ids=np.asarray(ids),
        **prediction,
    )
    write_json(
        Path(output) / "point_matchability_confidence_v3.json",
        {
            "points": len(rows),
            "checkpoint_hash": digest(checkpoint),
            "matchability_hash": matchability_hash,
            "reference_poses_used_for_prediction": False,
            "target": "held-out reprojection-error proxy; spatial calibration shift may break nominal coverage",
        },
    )
