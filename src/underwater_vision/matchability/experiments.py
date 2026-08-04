"""Fold-specific correspondence filtering and equal-budget comparisons."""

import copy
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from underwater_vision.data.dataset_loader import MermaidDataset
from underwater_vision.evaluation.matchability import paired_ap_bootstrap
from underwater_vision.evaluation.uncertainty import binary_metrics, choose_threshold
from underwater_vision.geometry.reference import camera_parameters, reference_labels
from underwater_vision.matchability.build import bidirectional_nearest, load_features
from underwater_vision.matchability.calibration import fit_temperature, probabilities
from underwater_vision.matchability.infer import (
    MatchabilityPredictor,
    attach_matchability,
    select_keypoints,
)
from underwater_vision.matchability.protocol import digest, new_run, read_json
from underwater_vision.matchability.tabular_baseline import DinoFeatureCache
from underwater_vision.matching.learned_filter import (
    FEATURE_NAMES,
    MATCHABILITY_FEATURE_NAMES,
    MatchFilter,
    candidates,
)
from underwater_vision.utils.io import atomic_npz, write_json


def selected_names(names, count):
    return [names[i] for i in np.linspace(0, len(names) - 1, min(count, len(names)), dtype=int)]


def near_and_distant_pairs(names, lookup):
    centers = np.stack([lookup[n].reference_c2w[:3, 3] for n in names])
    distance = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    np.fill_diagonal(distance, np.inf)
    pairs = set()
    for i, name in enumerate(names):
        for j in np.argsort(distance[i])[:2]:
            pairs.add(tuple(sorted((name, names[j]))))
        pairs.add(tuple(sorted((name, names[(i + len(names) // 2) % len(names)]))))
    return sorted(pairs)


def filter_metric(y, p, threshold):
    result = binary_metrics(y, p, threshold)
    result.pop("threshold_curve", None)
    return result


def train_filters(args):
    source = Path(args.matchability_dataset)
    root = new_run(args.output)
    manifest = read_json(source / "manifest_v3.json")
    split = next(s for s in manifest["protocols"] if s["name"] == args.protocol)
    selected = {
        s: selected_names(split["groups"][s], n)
        for s, n in [
            ("train", args.train_views),
            ("validation", args.validation_views),
            ("test", args.test_views),
        ]
    }
    if args.protocol == "v1_partition":
        selected["test"] = Path("configs/evaluation_images.txt").read_text().splitlines()
        if not set(selected["test"]).issubset(set(split["groups"]["test"])):
            raise ValueError("Historical development list is outside the v1 spatial test slab")
    predictor = MatchabilityPredictor(args.matchability_checkpoint)
    if predictor.metadata["spatial_split"] != split:
        raise ValueError("Match filter and dense model must share one spatial fold")
    fit_names = set(predictor.metadata["training_images"] + predictor.metadata["validation_images"])
    if fit_names & set(selected["test"]):
        raise ValueError("Dense model leaked into correspondence test images")
    lookup = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
    data = yaml.safe_load(Path("configs/data/mermaid.yaml").read_text())
    cache = DinoFeatureCache(source)
    features = {}
    (root / "maps").mkdir()
    for name in sum(selected.values(), []):
        feature = load_features(source, name)
        feature["semantic"] = cache.descriptors(name, feature["xy"])
        values = predictor.cached(lookup[name].path)
        feature = attach_matchability(feature, values, predictor.checkpoint_hash)
        features[name] = feature
        atomic_npz(
            root / "maps" / (name + ".npz"), matchability=feature["matchability"], xy=feature["xy"]
        )
    spatial_manifest = {
        "selected": selected,
        "split": split,
        "dataset": str(source),
        "label_spec": manifest["label_spec"],
        "matchability_checkpoint": str(args.matchability_checkpoint),
        "matchability_hash": predictor.checkpoint_hash,
        "manifest_sha256": digest(source / "manifest_v3.json"),
    }
    write_json(root / "spatial_manifest_v3.json", spatial_manifest)
    records = {s: [] for s in selected}
    (root / "pairs").mkdir()
    for group, names in selected.items():
        pair_list = (
            near_and_distant_pairs(names, lookup)
            if group != "test"
            else [
                (names[i], names[j])
                for i in range(len(names))
                for j in range(i + 1, min(len(names), i + 7))
            ]
        )
        for i, (a, b) in enumerate(pair_list):
            fa, fb = features[a], features[b]
            h, w = map(int, fa["shape"])
            ids, x = candidates(fa, fb, w, h, matchability=True)
            K, distortion = camera_parameters(w, h, data["calibration"])
            labels, error, F = reference_labels(
                fa["xy"][ids[:, 0]],
                fb["xy"][ids[:, 1]],
                lookup[a].reference_c2w,
                lookup[b].reference_c2w,
                K,
                distortion,
            )
            stats = (fa["statistics"][ids[:, 0]] + fb["statistics"][ids[:, 1]]) / 2
            record = {
                "a": np.asarray(a),
                "b": np.asarray(b),
                "pairs": ids,
                "x": x,
                "labels": labels,
                "reference_error": error.astype(np.float32),
                "reference_F": F,
                "contrast": stats[:, 2],
                "scale": stats[:, 1],
                "sharpness": (fa["sharpness"][ids[:, 0]] + fb["sharpness"][ids[:, 1]]) / 2,
                "brightness": np.full(
                    len(ids), (float(fa["brightness"]) + float(fb["brightness"])) / 2, np.float32
                ),
            }
            atomic_npz(root / "pairs" / f"{group}_{i:04d}.npz", **record)
            records[group].append(record)
    evaluate_budgets(records["test"], features, lookup, data["calibration"], args.seed, root)
    features.clear()
    predictor = cache = None
    torch.cuda.empty_cache()
    torch.set_num_threads(4)
    vy = np.concatenate([r["labels"] for r in records["validation"]])
    report = {
        "status": "complete",
        "models": {},
        "seed": args.seed,
        "protocol": args.protocol,
        "pair_counts": {s: len(rs) for s, rs in records.items()},
        "view_counts": {s: len(ns) for s, ns in selected.items()},
        "matchability_hash": spatial_manifest["matchability_hash"],
        "baseline_protocol": "Retrain 12-input v1 architectures on these exact pairs; do not reuse a checkpoint fitted to another rotated test block",
    }
    (root / "predictions").mkdir()
    for channels, architecture in [(12, "context"), (12, "mlp"), (16, "context"), (16, "mlp")]:
        torch.manual_seed(args.seed)
        model = MatchFilter(architecture, channels)
        train_x = np.concatenate([r["x"][r["labels"] >= 0, :channels] for r in records["train"]])
        mean, std = train_x.mean(0), np.maximum(train_x.std(0), 1e-3)
        tensors = {
            s: [
                torch.from_numpy(((r["x"][:, :channels] - mean) / std).astype(np.float32))
                for r in rows
            ]
            for s, rows in records.items()
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        history, best, state = [], -1.0, None
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for i in np.random.default_rng(args.seed + epoch).permutation(len(records["train"])):
                r = records["train"][i]
                known = r["labels"] >= 0
                if not known.any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                z = model(tensors["train"][i])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    z[known], torch.from_numpy(r["labels"][known].astype(np.float32))
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                z = np.concatenate([model(t).numpy() for t in tensors["validation"]])
            ap = filter_metric(vy, probabilities(z), 0.5)["average_precision"]
            history.append({"epoch": epoch + 1, "bce": float(np.mean(losses)), "validation_ap": ap})
            if ap > best:
                best, state = ap, copy.deepcopy(model.state_dict())
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            z = np.concatenate([model(t).numpy() for t in tensors["validation"]])
        temperature = fit_temperature(z, vy)
        threshold = choose_threshold(vy, probabilities(z, temperature))
        name = f"{architecture}_{channels}"
        checkpoint = {
            "state_dict": state,
            "architecture": architecture,
            "feature_spec": "match_v1_12" if channels == 12 else "match_matchability_v3_16",
            "feature_names": FEATURE_NAMES + (MATCHABILITY_FEATURE_NAMES if channels == 16 else []),
            "matchability_hash": spatial_manifest["matchability_hash"] if channels == 16 else None,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "temperature": temperature,
            "threshold": threshold,
            "training_images": selected["train"],
            "validation_images": selected["validation"],
            "spatial_split": split,
            "backbone": "dinov2_vitb14",
            "revision": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
            "image_size": 960,
            "max_features": 4096,
            "labels": "published_reference_F_compatibility_with_ambiguous_band_excluded",
            "seed": args.seed,
        }
        torch.save(checkpoint, root / (name + "_v3.pt"))
        with torch.no_grad():
            test_p = [probabilities(model(t).numpy(), temperature) for t in tensors["test"]]
        y = np.concatenate([r["labels"] for r in records["test"]])
        report["models"][name] = {
            "test": filter_metric(y, np.concatenate(test_p), threshold),
            "validation": filter_metric(vy, probabilities(z, temperature), threshold),
            "history": history,
            "checkpoint_sha256": digest(root / (name + "_v3.pt")),
        }
        for i, p in enumerate(test_p):
            records["test"][i][name] = p
        print(
            f"{args.protocol} seed {args.seed} {name}: test AP={report['models'][name]['test']['average_precision']:.5f}",
            flush=True,
        )
    for i, r in enumerate(records["test"]):
        atomic_npz(
            root / "predictions" / f"test_{i:04d}.npz",
            labels=r["labels"],
            **{k: r[k] for k in report["models"]},
        )
    report["paired_context_16_minus_12"] = paired_ap_bootstrap(
        records["test"], "context_16", "context_12"
    )
    report["paired_mlp_16_minus_12"] = paired_ap_bootstrap(records["test"], "mlp_16", "mlp_12")
    write_json(root / "evaluation_v3.json", report)
    return root


def evaluate_budgets(records, features, lookup, calibration, seed, root):
    result = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for budget in [512, 1024, 2048, 4096]:
        for source in ["matchability", "response", "random"]:
            (
                compatible,
                candidates_count,
                reference_total,
                original_retained,
                elapsed,
                pair_count,
            ) = (0, 0, 0, 0, 0.0, 0)
            for r in records:
                a, b = str(r["a"]), str(r["b"])
                fa, fb = features[a], features[b]
                if min(len(fa["xy"]), len(fb["xy"])) < budget:
                    continue
                seed_a = int.from_bytes(
                    hashlib.sha256(f"{seed}:{a}".encode()).digest()[:8], "little"
                )
                seed_b = int.from_bytes(
                    hashlib.sha256(f"{seed}:{b}".encode()).digest()[:8], "little"
                )
                sa, sb = select_keypoints(fa, budget, source, seed_a), select_keypoints(
                    fb, budget, source, seed_b
                )
                if pair_count == 0:
                    for _ in range(5):
                        bidirectional_nearest(sa["local"], sb["local"], device)
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                ids, _ = bidirectional_nearest(sa["local"], sb["local"], device)
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed += time.perf_counter() - start
                h, w = map(int, fa["shape"])
                K, distortion = camera_parameters(w, h, calibration)
                _, error, _ = reference_labels(
                    sa["xy"][ids[:, 0]],
                    sb["xy"][ids[:, 1]],
                    lookup[a].reference_c2w,
                    lookup[b].reference_c2w,
                    K,
                    distortion,
                )
                compatible += int(np.sum(error <= 1.5))
                candidates_count += len(error)
                good = r["reference_error"] <= 1.5
                reference_total += int(good.sum())
                original_retained += int(
                    np.sum(
                        good
                        & np.isin(r["pairs"][:, 0], sa["source_indices"])
                        & np.isin(r["pairs"][:, 1], sb["source_indices"])
                    )
                )
                pair_count += 1
            result.append(
                {
                    "budget": budget,
                    "source": source,
                    "pairs": pair_count,
                    "compatible_after_rematching": compatible,
                    "top1_candidates": candidates_count,
                    "precision": compatible / max(1, candidates_count),
                    "original_compatible_candidate_recall": original_retained
                    / max(1, reference_total),
                    "matching_seconds": elapsed,
                    "matching_scope": "GPU exact NN incl transfer/synchronize; excludes keypoint extraction and map inference",
                    "skipped_pairs_insufficient_keypoints": len(records) - pair_count,
                }
            )
    write_json(
        root / "budgets_v3.json",
        {
            "curves": result,
            "random_seed_rule": "SHA256(base seed:image name), independent across images and stable across pairs",
            "runtime_warmup": "5 exact-N nearest-neighbor calls per budget/source before timing",
            "recall_definition": "Original full-pool compatible top1 pairs with BOTH endpoint indices retained / all original compatible pairs. Separate compatible count recomputes top1 after selection.",
        },
    )
