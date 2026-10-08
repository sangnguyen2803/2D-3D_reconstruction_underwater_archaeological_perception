"""Locked spatial-test PR/AUC versus ratio, RANSAC, old fusion and LightGlue."""

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.evaluation.uncertainty import binary_metrics
from underwater_vision.features.dinov2 import DinoExtractor
from underwater_vision.features.store import feature_record
from underwater_vision.geometry.reference import camera_parameters, reference_labels, undistort
from underwater_vision.geometry.verification import verify
from underwater_vision.matching.learned_filter import MatchPredictor
from underwater_vision.matching.lightglue_adapter import LightGlueAdapter
from underwater_vision.matching.matcher import match_local, score_matches
from underwater_vision.utils.io import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("outputs/match_filter_v1"))
    p.add_argument("--output", type=Path, default=Path("outputs/match_evaluation_v1"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    predictors = {n: MatchPredictor(args.run / f"{n}_v1.pt") for n in ["mlp", "context"]}
    neural = DinoExtractor("dinov2_vitb14", "auto", "7764ea0f912e53c92e82eb78a2a1631e92725fc8")
    lightglue = LightGlueAdapter()
    data = OmegaConf.to_container(OmegaConf.load("configs/data/mermaid.yaml"))
    lookup = {f.name: f for f in MermaidDataset(data["root"]).frames}
    features = {}

    def feature(name):
        if name not in features:
            features[name] = feature_record(lookup[name], neural)
        return features[name]

    scores = {
        n: [] for n in ["ratio_test", "ransac", "old_visibility", "lightglue", "mlp", "context"]
    }
    labels, lightglue_labels, lightglue_probabilities = [], [], []
    for k, path in enumerate(sorted((args.run / "pairs").glob("test_*.npz"))):
        with np.load(path) as z:
            a, b, pairs, x, y = str(z["a"]), str(z["b"]), z["pairs"], z["x"], z["labels"]
        labels.extend(y)
        fa, fb = feature(a), feature(b)
        height, width = fa["shape"]
        K, distortion = camera_parameters(int(width), int(height), data["calibration"])
        scores["ratio_test"].extend(np.where(x[:, 2] > 0.5, 1 - x[:, 1], 0))
        eligible = (x[:, 2] > 0.5) & (x[:, 1] < 0.8)
        _, mask, _ = verify(
            undistort(fa["xy"][pairs[eligible, 0]], K, distortion),
            undistort(fb["xy"][pairs[eligible, 1]], K, distortion),
            np.ones(eligible.sum()),
        )
        ransac = np.zeros(len(pairs))
        ransac[np.flatnonzero(eligible)[mask]] = 1
        scores["ransac"].extend(ransac)
        alpha = 0.35
        fused_a = np.c_[np.sqrt(1 - alpha) * fa["local"], np.sqrt(alpha) * fa["semantic"]].astype(
            np.float32
        )
        fused_b = np.c_[np.sqrt(1 - alpha) * fb["local"], np.sqrt(alpha) * fb["semantic"]].astype(
            np.float32
        )
        old_pairs, local = match_local(fused_a, fused_b, 0.8)
        old_scores = score_matches(
            old_pairs, local, fa["semantic"], fb["semantic"], fa["quality"], fb["quality"], alpha
        )
        old_lookup = {tuple(pair): score for pair, score in zip(old_pairs, old_scores, strict=True)}
        scores["old_visibility"].extend([old_lookup.get(tuple(pair), 0) for pair in pairs])
        lg_pairs, lg_score = lightglue(fa, fb, int(width), int(height))
        lg_y, _, _ = reference_labels(
            fa["xy"][lg_pairs[:, 0]],
            fb["xy"][lg_pairs[:, 1]],
            lookup[a].reference_c2w,
            lookup[b].reference_c2w,
            K,
            distortion,
        )
        lightglue_labels.extend(lg_y)
        lightglue_probabilities.extend(lg_score)
        lg_lookup = {tuple(pair): score for pair, score in zip(lg_pairs, lg_score, strict=True)}
        scores["lightglue"].extend([lg_lookup.get(tuple(pair), 0) for pair in pairs])
        for name, predictor in predictors.items():
            scores[name].extend(predictor(x))
        if k % 15 == 0:
            print(f"Independent matching evaluation {k+1}/75", flush=True)
    thresholds = {
        "ratio_test": 0.2,
        "ransac": 0.5,
        "old_visibility": 0.08,
        "lightglue": 0.1,
        **{n: p.metadata["threshold"] for n, p in predictors.items()},
    }
    metrics = {}
    for n, score in scores.items():
        metrics[n] = binary_metrics(labels, score, thresholds[n])
        metrics[n].pop("threshold_curve", None)
    native_labels = np.asarray(lightglue_labels)
    known_native = native_labels >= 0
    native = {
        "returned_matches": len(native_labels),
        "known_reference_labels": int(known_native.sum()),
        "reference_precision": float(native_labels[known_native].mean()),
        "recall": None,
        "reason": "Native matches are already selected; recall needs a common proposal denominator",
    }
    report = {
        "models": metrics,
        "lightglue_native_matches": native,
        "common_pool": "all RootSIFT top-1 proposals; LightGlue/fused matches outside pool excluded from aligned metrics",
        "labels": "reference F <=1.5px positive, >=4px negative; ambiguous excluded",
        "lightglue_revision": "eb42fee2d71449efb0aa5c10549752b5d75384d8",
        "evaluation_views": 16,
        "pairs": 75,
        "threshold_selection": "spatial validation only for learned models",
    }
    write_json(args.output / "evaluation_v1.json", report)
    print(
        {
            n: {k: v for k, v in m.items() if k not in {"pr_curve", "calibration"}}
            for n, m in metrics.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
