"""Fine-tune match networks to tight reference geometry, train/val only."""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from underwater_vision.matchability.protocol import digest, read_json
from underwater_vision.matching.budget import budget_indices
from underwater_vision.matching.learned_filter import MatchPredictor
from underwater_vision.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/geometry_ranker_v1"))
    parser.add_argument("--architecture", choices=["mlp", "context"], default="mlp")
    parser.add_argument(
        "--pairs-from",
        type=Path,
        help="Reuse audited pairs and role manifests from an earlier expanded run",
    )
    parser.add_argument(
        "--expanded",
        action="store_true",
        help="Use all v1 train/validation images and sequential near-view pairs",
    )
    args = parser.parse_args()
    audit = read_json("outputs/reference_audit_v1/audit_v1.json")
    if audit["status"] != "passed_proxy_sanity_check":
        raise ValueError("Reference geometry audit must pass before geometry supervision")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    predictor = MatchPredictor(f"outputs/match_filter_v1/{args.architecture}_v1.pt")
    model = predictor.model
    records = {}
    files = {}
    if args.pairs_from:
        parent = read_json(args.pairs_from / "training_v1.json")
        pair_root = args.pairs_from / "pairs"
        predictor.metadata["training_images"] = parent["training_images"]
        predictor.metadata["validation_images"] = parent["validation_images"]
        assert not set(parent["training_images"]) & set(parent["validation_images"])
        for path in sorted(pair_root.glob("*.npz")):
            assert digest(path) == parent["input_files"][str(path)]
    elif args.expanded:
        from omegaconf import OmegaConf

        from underwater_vision.data.dataset_loader import MermaidDataset
        from underwater_vision.geometry.reference import camera_parameters, reference_labels
        from underwater_vision.matchability.build import load_features
        from underwater_vision.matchability.tabular_baseline import DinoFeatureCache
        from underwater_vision.matching.learned_filter import candidates
        from underwater_vision.utils.io import atomic_npz

        dataset_root = Path("outputs/matchability_labels_v3")
        manifest = read_json(dataset_root / "manifest_v3.json")
        split = next(p for p in manifest["protocols"] if p["name"] == "v1_partition")
        lookup = {f.name: f for f in MermaidDataset("data/raw/mermaid").frames}
        calibration = OmegaConf.to_container(OmegaConf.load("configs/data/mermaid.yaml"))[
            "calibration"
        ]
        cache = DinoFeatureCache(dataset_root)
        pair_root = args.output / "pairs"
        pair_root.mkdir()
        selected = {}
        for group in ["train", "validation"]:
            names = sorted(split["groups"][group])
            selected[group] = names
            features = {}
            for name in names:
                f = load_features(dataset_root, name)
                f["semantic"] = cache.descriptors(name, f["xy"])
                features[name] = f
            index = {int(n.split("_")[-1].split(".")[0]): n for n in names}
            pairs = [
                (index[i], index[j])
                for i in sorted(index)
                for j in range(i + 1, i + 7)
                if j in index
            ]
            for k, (a, b) in enumerate(pairs):
                fa, fb = features[a], features[b]
                h, w = map(int, fa["shape"])
                ids, x = candidates(fa, fb, w, h)
                K, distortion = camera_parameters(w, h, calibration)
                _, error, _ = reference_labels(
                    fa["xy"][ids[:, 0]],
                    fb["xy"][ids[:, 1]],
                    lookup[a].reference_c2w,
                    lookup[b].reference_c2w,
                    K,
                    distortion,
                )
                atomic_npz(
                    pair_root / f"{group}_{k:04}.npz",
                    a=np.asarray(a),
                    b=np.asarray(b),
                    x=x,
                    reference_error=error,
                )
                if k % 100 == 0:
                    print(f"expanded {group}: {k+1}/{len(pairs)} pairs", flush=True)
        predictor.metadata["training_images"] = selected["train"]
        predictor.metadata["validation_images"] = selected["validation"]
    else:
        pair_root = Path("outputs/match_filter_v1/pairs")
    for group in ["train", "validation"]:
        records[group] = []
        files[group] = sorted(pair_root.glob(f"{group}_*.npz"))
        for path in files[group]:
            with np.load(path) as z:
                x, error = z["x"].copy(), z["reference_error"].copy()
                names = [str(z["a"]), str(z["b"])]
            assert set(names) <= set(
                predictor.metadata[
                    f"{group}ing_images" if group == "train" else "validation_images"
                ]
            )
            valid = np.isfinite(error)
            records[group].append((x[valid], error[valid]))
    x = np.concatenate([r[0] for r in records["train"]])
    errors = np.concatenate([r[1] for r in records["train"]])
    tensor = torch.from_numpy(((x - predictor.mean) / predictor.std).astype(np.float32))
    # A soft tight-geometry target: 0.5 px scale, not the original 1.5 px class.
    target = torch.from_numpy(
        np.exp(-0.5 * np.square(np.minimum(errors, 20) / 0.5)).astype(np.float32)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0001)
    contextual_batches = []
    if args.architecture == "context":
        for px, pe in records["train"]:
            if len(px):
                contextual_batches.append(
                    (
                        torch.from_numpy(
                            ((px - predictor.mean) / predictor.std).astype(np.float32)
                        ),
                        torch.from_numpy(
                            np.exp(-0.5 * np.square(np.minimum(pe, 20) / 0.5)).astype(np.float32)
                        ),
                    )
                )
    best, state, history = float("inf"), None, []
    for epoch in range(20):
        model.train()
        losses = []
        rng = np.random.default_rng(42 + epoch)
        if args.architecture == "context":
            # Context attention must see a single pair, never a mixed-pair minibatch.
            batches = [contextual_batches[i] for i in rng.permutation(len(contextual_batches))]
        else:
            permutation = rng.permutation(len(tensor))
            batches = [
                (
                    tensor[permutation[start : start + 2048]],
                    target[permutation[start : start + 2048]],
                )
                for start in range(0, len(tensor), 2048)
            ]
        for batch_x, batch_y in batches:
            logits = model(batch_x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        selected_errors = []
        with torch.inference_mode():
            for vx, ve in records["validation"]:
                scores = model(
                    torch.from_numpy(((vx - predictor.mean) / predictor.std).astype(np.float32))
                ).numpy()
                keep = budget_indices(vx, scores)
                if len(keep):
                    selected_errors.extend(np.minimum(ve[keep], 3).tolist())
        objective = float(np.mean(selected_errors))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "validation_retained_clipped_error": objective,
            }
        )
        if objective < best:
            best, state = objective, copy.deepcopy(model.state_dict())
        print(f"epoch {epoch+1}: validation retained error {objective:.6f}", flush=True)
    metadata = copy.deepcopy(predictor.metadata)
    metadata.update(
        state_dict=state,
        temperature=1.0,
        target="exp(-0.5*(reference_epipolar_error/0.5)^2)",
        output_interpretation="tight_geometry_ranking_score_not_calibrated_probability",
    )
    torch.save(metadata, args.output / "ranker_v1.pt")
    write_json(
        args.output / "training_v1.json",
        {
            "status": "complete",
            "history": history,
            "training_candidates": len(x),
            "architecture": args.architecture,
            "source_sha256": digest(__file__),
            "pair_source": str(pair_root),
            "validation_objective": best,
            "training_images": metadata["training_images"],
            "validation_images": metadata["validation_images"],
            "input_files": {str(p): digest(p) for paths in files.values() for p in paths},
            "protocol": "Fine-tune an existing match network on the recorded spatial training pairs only; select checkpoint by clipped epipolar error of budget-retained validation matches. Context attention uses a single image pair per batch. No test pair or pose is read. Fixed retention0.9, sparse-pair guard128, target scale0.5px.",
        },
    )


if __name__ == "__main__":
    main()
