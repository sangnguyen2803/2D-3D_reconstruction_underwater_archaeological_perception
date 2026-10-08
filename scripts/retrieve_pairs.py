"""Full-dataset frozen DINO CLS retrieval with reference co-frustum evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.features.dinov2 import DinoExtractor
from underwater_vision.geometry.reference import (
    camera_parameters,
    fundamental_from_poses,
    triangulate_observations,
    undistort,
)
from underwater_vision.matching.matcher import match_local
from underwater_vision.preprocessing.images import resize_rgb
from underwater_vision.retrieval.pairs import overlap_proxy, retrieval_metrics, retrieve_pairs
from underwater_vision.utils.io import atomic_npz, fingerprint, read_rgb, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/retrieval_v1"))
    p.add_argument("--audit", type=Path, default=Path("outputs/reference_audit_v1"))
    p.add_argument("--top-k", type=int, default=12)
    args = p.parse_args()
    audit = json.loads((args.audit / "audit_v1.json").read_text())
    if audit["status"] != "passed_proxy_sanity_check":
        raise ValueError("Reference audit must pass before retrieval evaluation")
    args.output.mkdir(parents=True, exist_ok=False)
    data = OmegaConf.to_container(OmegaConf.load("configs/data/mermaid.yaml"))
    frames = MermaidDataset(data["root"]).frames
    neural = DinoExtractor("dinov2_vitb14", "auto", "7764ea0f912e53c92e82eb78a2a1631e92725fc8")
    cache = Path("cache/global_dino_v1")
    cache.mkdir(parents=True, exist_ok=True)
    descriptors = []
    for i, frame in enumerate(frames):
        key = fingerprint(
            frame.path, {"global_v1": True, "max_size": 336, "backbone": neural.provenance}
        )
        path = cache / (key + ".npz")
        if path.exists():
            with np.load(path) as z:
                descriptor = z["global_descriptor"]
        else:
            rgb, _ = resize_rgb(read_rgb(frame.path), 336)
            descriptor = neural.extract_global(rgb)
            atomic_npz(path, global_descriptor=descriptor)
        descriptors.append(descriptor)
        if i % 64 == 0:
            print(f"Global descriptors {i + 1}/{len(frames)}", flush=True)
    d = np.asarray(descriptors)
    pairs, neighbors = retrieve_pairs(d, args.top_k)
    width, height = 960, 720
    K, distortion = camera_parameters(width, height, data["calibration"])
    # Estimate representative depth from reference-camera triangulation of a
    # distributed sample of tight-ratio correspondences. No RANSAC is used.
    depths = []
    records = json.loads((args.audit / "pairs_v1.json").read_text())
    lookup = {f.name: f for f in frames}
    feature_cache = Path("cache/reference_features_v1")
    for record in records[::4]:

        def load(frame):
            key = fingerprint(frame.path, {"audit_v1": True, "max_size": 960, "features": 2048})
            with np.load(feature_cache / (key + ".npz")) as z:
                return z["xy"], z["local"]

        a, b = lookup[record["a"]], lookup[record["b"]]
        xa, da = load(a)
        xb, db = load(b)
        matches, _ = match_local(da, db, 0.7)
        ua, ub = undistort(xa, K, distortion), undistort(xb, K, distortion)
        for ia, ib in matches[:: max(1, len(matches) // 20)]:
            try:
                point = triangulate_observations(
                    [a.reference_c2w, b.reference_c2w], [ua[ia], ub[ib]], K
                )
            except ValueError:
                continue
            z = (np.linalg.inv(a.reference_c2w) @ np.r_[point, 1])[2]
            if np.isfinite(z) and z > 0:
                depths.append(z)
    if len(depths) < 20:
        raise ValueError("Insufficient positive reference triangulations for overlap proxy")
    depth = float(np.median(depths))
    poses = np.stack([f.reference_c2w for f in frames])
    truth, fraction = overlap_proxy(poses, K, width, height, depth)
    report = retrieval_metrics(neighbors, truth)
    report.update(
        scene_depth_reference_units=depth,
        evaluated_views=len(frames),
        unique_retrieved_pairs=len(pairs),
        backbone=neural.provenance,
        limitation="Reference co-frustum at median depth ignores occlusion/relief; not actual overlap GT",
    )
    sensitivity = []
    for factor in [0.5, 1.0, 2.0]:
        for threshold in [0.25, 0.5]:
            labels, _ = overlap_proxy(poses, K, width, height, depth * factor, threshold)
            sensitivity.append(
                {
                    "depth_factor": factor,
                    "overlap_threshold": threshold,
                    **retrieval_metrics(neighbors, labels),
                }
            )
    report["overlap_sensitivity"] = sensitivity
    F = np.stack([fundamental_from_poses(poses[i], poses[j], K) for i, j in pairs])
    atomic_npz(
        args.output / "retrieval_v1.npz",
        descriptors=d,
        neighbors=neighbors,
        pairs=np.asarray(pairs, np.int32),
        overlap=truth,
        overlap_fraction=fraction,
        reference_F=F,
        image_names=np.asarray([f.name for f in frames]),
    )
    write_json(args.output / "metrics_v1.json", report)
    write_json(args.output / "pairs_v1.json", [[frames[i].name, frames[j].name] for i, j in pairs])
    print(report, flush=True)


if __name__ == "__main__":
    main()
