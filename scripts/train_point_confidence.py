"""Train C using reference-camera held-out views and spatial point partitions."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pycolmap
import torch
from omegaconf import OmegaConf

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.data.spatial import assign_spatial
from underwater_vision.evaluation.uncertainty import uncertainty_metrics
from underwater_vision.features.dinov2 import DinoExtractor
from underwater_vision.features.store import feature_record
from underwater_vision.geometry.reference import camera_parameters, undistort
from underwater_vision.geometry.verification import verify
from underwater_vision.pointcloud.learned_confidence import (
    CONFIDENCE_FEATURES,
    ErrorPredictor,
    ErrorRegressor,
    holdout_observation,
)
from underwater_vision.utils.io import atomic_npz, write_json


class Tracks:
    def __init__(self):
        self.parent, self.members = {}, {}

    def root(self, node):
        if node not in self.parent:
            self.parent[node] = node
            self.members[node] = {node}
        if self.parent[node] != node:
            self.parent[node] = self.root(self.parent[node])
        return self.parent[node]

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        if a == b:
            return
        if {n[0] for n in self.members[a]} & {n[0] for n in self.members[b]}:
            return
        self.parent[b] = a
        self.members[a] |= self.members.pop(b)

    def tracks(self):
        return [sorted(group) for group in self.members.values() if len(group) >= 3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/point_confidence_v1"))
    p.add_argument("--matches", type=Path, default=Path("outputs/match_filter_v1"))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--baseline", type=Path, default=Path("outputs/classical_eval"))
    args = p.parse_args()
    if (args.output / "evaluation_v1.json").exists():
        raise FileExistsError("Existing confidence results are immutable")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.matches / "spatial_manifest_v1.json").read_text())
    split = manifest["split"]
    data = OmegaConf.to_container(OmegaConf.load("configs/data/mermaid.yaml"))
    lookup = {f.name: f for f in MermaidDataset(data["root"]).frames}
    neural = DinoExtractor("dinov2_vitb14", "auto", "7764ea0f912e53c92e82eb78a2a1631e92725fc8")
    features = {}

    def feature(name):
        if name not in features:
            f = feature_record(lookup[name], neural)
            height, width = f["shape"]
            K, distortion = camera_parameters(int(width), int(height), data["calibration"])
            f["undistorted"] = undistort(f["xy"], K, distortion)
            features[name] = f
        return features[name]

    K, _ = camera_parameters(960, 720, data["calibration"])
    groups = {}
    for group in ["train", "validation"]:
        tracks = Tracks()
        for path in sorted((args.matches / "pairs").glob(group + "_*.npz")):
            with np.load(path) as r:
                a, b, pairs, x = str(r["a"]), str(r["b"]), r["pairs"], r["x"]
            fa, fb = feature(a), feature(b)
            eligible = (x[:, 1] < 0.8) & (x[:, 2] > 0.5)
            pairs = pairs[eligible]
            _, mask, _ = verify(
                fa["undistorted"][pairs[:, 0]],
                fb["undistorted"][pairs[:, 1]],
                np.ones(len(pairs)),
                iterations=300,
            )
            for i, j in pairs[mask]:
                tracks.join((a, int(i)), (b, int(j)))
        groups[group] = tracks.tracks()
        print(f"{group}: {len(groups[group])} candidate 3-view tracks", flush=True)
    # Locked test tracks are from the original classical reconstruction, not the
    # match-filter training targets. Predictor fitting never uses their errors.
    reconstruction = pycolmap.Reconstruction(str(args.baseline / "sparse/0"))
    groups["test"] = []
    for point in reconstruction.points3D.values():
        nodes = [
            (reconstruction.images[e.image_id].name, int(e.point2D_idx))
            for e in point.track.elements
        ]
        if len(nodes) >= 3:
            groups["test"].append(sorted(nodes))
    arrays, counts = {}, {}
    for group, tracks in groups.items():
        x, y, h, xyz, nviews, heldouts = [], [], [], [], [], []
        invalid, excluded = 0, 0
        for nodes in tracks:
            poses = [lookup[n].reference_c2w for n, _ in nodes]
            xy = [feature(n)["undistorted"][i] for n, i in nodes]
            semantic = [feature(n)["semantic"][i] for n, i in nodes]
            quality = [feature(n)["quality"][i] for n, i in nodes]
            try:
                row, error, heuristic, point, held = holdout_observation(
                    nodes, poses, xy, semantic, quality, K
                )
            except ValueError:
                invalid += 1
                continue
            if assign_spatial([point], split["specification"])[0] != group:
                excluded += 1
                continue
            x.append(row)
            y.append(error)
            h.append(heuristic)
            xyz.append(point)
            nviews.append(len(nodes) - 1)
            heldouts.append(nodes[held][0])
        arrays[group] = {
            "x": np.asarray(x, np.float32),
            "error": np.asarray(y),
            "heuristic": np.asarray(h),
            "xyz": np.asarray(xyz),
            "observed_views": np.asarray(nviews),
            "heldout_view": np.asarray(heldouts),
        }
        counts[group] = {
            "eligible": len(y),
            "invalid_geometry": invalid,
            "spatially_excluded": excluded,
        }
        atomic_npz(args.output / f"{group}_v1.npz", **arrays[group])
        print(f"{group}: {len(y)} eligible spatial/held-out points", flush=True)
    if any(len(arrays[s]["error"]) < 30 for s in arrays):
        raise ValueError(f"Need >=30 independent points per spatial block: {counts}")
    train, validation, test = (arrays[s] for s in ["train", "validation", "test"])
    mean = train["x"].mean(0)
    std = np.maximum(train["x"].std(0), 1e-4)
    x = torch.from_numpy((train["x"] - mean) / std)
    y = torch.from_numpy(np.log1p(train["error"]).astype(np.float32))
    vx = torch.from_numpy((validation["x"] - mean) / std)
    vy = torch.from_numpy(np.log1p(validation["error"]).astype(np.float32))
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = ErrorRegressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    best, state, history = float("inf"), None, []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(42 + epoch))
        for ids in order.split(512):
            mu, lv = model(x[ids])
            loss = (0.5 * (lv + (y[ids] - mu) ** 2 * torch.exp(-lv))).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            mu, lv = model(vx)
            val = (0.5 * (lv + (vy - mu) ** 2 * torch.exp(-lv))).mean().item()
        history.append({"epoch": epoch + 1, "validation_log_error_nll_without_constant": val})
        if val < best:
            best, state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    with torch.no_grad():
        mu, lv = model(vx)
        prediction = torch.expm1((mu + torch.exp(lv) / 2).clamp(max=15)).clamp(min=0).numpy()
    scale = float(
        np.clip(
            np.dot(prediction, validation["error"]) / max(np.dot(prediction, prediction), 1e-9),
            0.1,
            10,
        )
    )
    torch.save(
        {
            "state_dict": state,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "validation_scale": scale,
            "feature_names": CONFIDENCE_FEATURES,
            "spatial_split": split,
            "seed": 42,
            "label": "reference_camera_leave_one_view_out_reprojection_px",
            "limitation": "BA reference proxy, not independently surveyed XYZ error",
        },
        args.output / "error_model_v1.pt",
    )
    predicted = ErrorPredictor(args.output / "error_model_v1.pt")(test["x"])
    evaluation = {
        "models": {},
        "counts": counts,
        "spatial_split": split["specification"],
        "selection": "validation NLL; validation-only scale; locked test errors never fitted",
        "history": history,
        "validation_scale": scale,
        "label": "held-out reference reprojection on old classical tracks; all features exclude held-out observation",
    }
    # Calibrate each simple baseline to px on validation, never the test labels.
    for name, validation_u, test_u in [
        ("heuristic", 1 - validation["heuristic"], 1 - test["heuristic"]),
        ("view_count", 1 / validation["observed_views"], 1 / test["observed_views"]),
    ]:
        factor = float(
            np.dot(validation_u, validation["error"])
            / max(np.dot(validation_u, validation_u), 1e-9)
        )
        evaluation["models"][name] = uncertainty_metrics(test["error"], factor * test_u)
    evaluation["models"]["learned"] = uncertainty_metrics(
        test["error"], predicted["predicted_error"]
    )
    evaluation["models"]["learned"]["interval90_coverage"] = float(
        np.mean((test["error"] >= predicted["lower90"]) & (test["error"] <= predicted["upper90"]))
    )
    atomic_npz(
        args.output / "predictions_v1.npz",
        **predicted,
        observed_error=test["error"],
        xyz=test["xyz"],
    )
    write_json(args.output / "evaluation_v1.json", evaluation)
    print(
        {
            k: {n: v for n, v in m.items() if n not in {"curve", "calibration"}}
            for k, m in evaluation["models"].items()
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
